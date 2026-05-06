import torch
import torch.nn as nn
from collections import deque
from typing import Optional, Sequence, Literal, Dict

from compresso.utils.schedule import exponential_decay

ScoreFn = Literal["sum_abs", "max_abs", "snr"]

class GatedMLP(nn.Module):
    def __init__(self, d, h, k_target, score="sum_abs", num_stages=5, stability_window=5, change_threshold=0.05, activation=nn.GELU(), ema_beta=0.98):
        super().__init__()
        self.fc1 = nn.Linear(d, h)
        self.fc2 = nn.Linear(h, d)
        self.act = activation

        self.gate = GatedMaskedParam(
            fc1=self.fc1,
            fc2=self.fc2,
            k_target=k_target,
            score=score,
            num_stages=num_stages,
            stability_window=stability_window,
            change_threshold=change_threshold,
            ema_beta=ema_beta,
        )
    
    def forward(self, x):
        h = self.fc1(x)
        h = self.gate.apply_mask_in_forward(h)
        h = self.act(h)
        return self.fc2(h)

class GatedMaskedParam(nn.Module):
    """
    Structured (unit) pruning for an MLP pair:
      fc1: d -> h  (W1 shape h x d)
      fc2: h -> d  (W2 shape d x h)

    Keeps a 1D boolean mask over hidden units (length h).
    - mask[j]=0 => drop hidden unit j (row j of W1, col j of W2, bias1[j])

    Lifecycle mirrors MaskedParam / SharedMaskedParam:
      - schedule over k (number of kept hidden units)
      - step_mask(): recompute next-stage mask, measure churn vs last_mask, detect stability
      - rewind(): LTH rewind to init under chosen mask, then optionally advance stage
      - freeze_mask(): freeze at current k
      - compact(): returns a physically shrunk dense MLP (no mask needed)
    """

    def __init__(
        self,
        fc1: nn.Linear,   # d -> h
        fc2: nn.Linear,   # h -> d
        k_target: int,
        k_schedule: Optional[Sequence[int]] = None,
        num_stages: int = 10,
        stability_window: int = 5,
        change_threshold: float = 0.01,
        score: ScoreFn = "sum_abs",
        ema_beta: float = 0.98,
        store_init_for_rewind: bool = True,
    ):
        super().__init__()

        if fc1.weight.dim() != 2 or fc2.weight.dim() != 2:
            raise ValueError("fc1 and fc2 must be Linear layers with 2D weights.")

        h, d1 = fc1.weight.shape         # (h, d)
        d2, h2 = fc2.weight.shape        # (d, h)
        if h != h2 or d1 != d2:
            raise ValueError(
                f"Shape mismatch: fc1.weight {tuple(fc1.weight.shape)} "
                f"and fc2.weight {tuple(fc2.weight.shape)} must be (h,d) and (d,h)."
            )

        self.fc1 = fc1
        self.fc2 = fc2
        self.hidden_dim = int(h)
        self.d_model = int(d1)

        self.k_target = int(k_target)
        if not (1 <= self.k_target <= self.hidden_dim):
            raise ValueError(f"k_target must be in [1, {self.hidden_dim}], got {self.k_target}")

        self.score_mode: ScoreFn = score
        self.ema_beta = float(ema_beta)

        # ---- store init (buffers) for rewind ----
        self.store_init_for_rewind = bool(store_init_for_rewind)
        if self.store_init_for_rewind:
            self.register_buffer("init_fc1_w", self.fc1.weight.detach().cpu().clone(), persistent=True)
            self.register_buffer("init_fc2_w", self.fc2.weight.detach().cpu().clone(), persistent=True)
            if self.fc1.bias is not None:
                self.register_buffer("init_fc1_b", self.fc1.bias.detach().cpu().clone(), persistent=True)
            if self.fc2.bias is not None:
                self.register_buffer("init_fc2_b", self.fc2.bias.detach().cpu().clone(), persistent=True)

        # ---- schedule over k (number of kept units) ----
        if k_schedule is not None:
            ks = [int(k) for k in k_schedule]
            for i in range(1, len(ks)):
                if ks[i] > ks[i - 1]:
                    raise ValueError(f"k_schedule must be non-increasing, got {k_schedule}")
            if ks[-1] != self.k_target:
                raise ValueError(f"Last k in k_schedule ({ks[-1]}) must equal k_target ({self.k_target})")
            self.k_schedule = ks
        else:
            if num_stages < 1:
                raise ValueError("num_stages must be >= 1")
            sched = exponential_decay(self.hidden_dim, self.k_target, num_stages - 1)
            sched.append(self.k_target)
            self.k_schedule = [int(k) for k in sched]

        self.num_stages = len(self.k_schedule)
        self.stage_idx = 0
        self.k_current = int(self.hidden_dim)
        self.k_next = int(self.k_schedule[1]) if self.num_stages > 1 else int(self.k_target)

        # ---- optional EMA grad stats for unit SNR ----
        self.register_buffer("g_steps", torch.tensor(0, dtype=torch.long), persistent=True)
        self.register_buffer("g_mu", torch.zeros(self.hidden_dim), persistent=True)
        self.register_buffer("g_m2", torch.zeros(self.hidden_dim), persistent=True)

        # ---- mask state (1D!) ----
        self.register_buffer("mask", torch.ones(self.hidden_dim, dtype=torch.bool), persistent=True)
        self.register_buffer("last_mask", torch.ones(self.hidden_dim, dtype=torch.bool), persistent=True)

        # initialize last_mask as next-stage selection
        self.last_mask.copy_(self._topk_unit_mask(k=self.k_next))

        # ---- stability tracking (mask churn) ----
        self.stability_window = int(stability_window)
        self.change_threshold = float(change_threshold)
        self._recent_changes = deque(maxlen=self.stability_window)

        self.last_change = 1.0
        self.last_num_changes: Optional[int] = None
        self.stage_completed = False
        self.schedule_done = False
        self.mask_frozen = False

        
        self._just_advanced = False   # NEW: latched per rewind; consumed by compaction

    # =========================
    # scoring + mask selection
    # =========================

    @torch.no_grad()
    def update_grad_stats(self, use_mask: bool = True):
        """
        Update EMA stats for per-unit gradient magnitude.
        Uses rows of fc1.weight.grad and cols of fc2.weight.grad, summed as one per-unit signal.
        Call this once per optimizer step (after backward, before zero_grad).
        """
        if self.fc1.weight.grad is None or self.fc2.weight.grad is None:
            return

        g1 = self.fc1.weight.grad         # (h, d)
        g2 = self.fc2.weight.grad         # (d, h)

        # per-unit grad magnitude (L2) from both sides
        u1 = g1.pow(2).sum(dim=1).sqrt()  # (h,)
        u2 = g2.pow(2).sum(dim=0).sqrt()  # (h,)
        u = u1 + u2

        if use_mask:
            u = u * self.mask.to(u.dtype)

        beta = self.ema_beta
        self.g_steps += 1
        self.g_mu.mul_(beta).add_(u, alpha=1 - beta)
        self.g_m2.mul_(beta).addcmul_(u, u, value=1 - beta)

    @torch.no_grad()
    def unit_scores(self, eps: float = 1e-8) -> torch.Tensor:
        """
        Returns per-hidden-unit score (h,).
        """
        if self.score_mode in ("sum_abs", "max_abs"):
            W1 = self.fc1.weight           # (h, d)
            W2 = self.fc2.weight           # (d, h)

            s1 = W1.abs().sum(dim=1)       # (h,)
            s2 = W2.abs().sum(dim=0)       # (h,)

            if self.score_mode == "sum_abs":
                return s1 + s2
            else:
                return torch.maximum(s1, s2)

        if self.score_mode == "snr":
            var = (self.g_m2 - self.g_mu.square()).clamp_min(0.0)
            return self.g_mu / (var.sqrt() + eps)

        raise ValueError(f"Unknown score_mode={self.score_mode}")

    @torch.no_grad()
    def _topk_unit_mask(self, k: int) -> torch.Tensor:
        k = int(k)
        scores = self.unit_scores()
        idx = torch.topk(scores, k=k, largest=True).indices
        m = torch.zeros(self.hidden_dim, dtype=torch.bool, device=scores.device)
        m[idx] = True
        return m

    # =========================
    # applying the mask (dense training-time)
    # =========================

    def apply_mask_in_forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Apply current mask to the hidden activation tensor h with shape (..., hidden_dim).
        Use this in your MLP forward: h = gate.apply_mask_in_forward(h)
        """
        if self.k_current==self.k_next:
            return h
        elif self.mask_frozen:
            m = self.mask
        else:
            # during dynamic phase you might want to use current-stage selection:
            # keep it deterministic w.r.t. stage k_current (still static per step)
            m = self._topk_unit_mask(k=self.k_current)

        return h * m.to(dtype=h.dtype, device=h.device)

    # =========================
    # schedule / stability (MaskedParam-like)
    # =========================

    @torch.no_grad()
    def step_mask(self) -> float:
        """
        Compute next-stage mask (k_next), track changes vs last_mask, mark stage_completed when stable.
        """
        
        if self.mask_frozen or self.schedule_done:
            return 0.0

        new_mask = self._topk_unit_mask(k=self.k_next)
        num_changes = torch.logical_xor(self.last_mask, new_mask).sum().item()
        self.last_num_changes = int(num_changes)

        # how many units are being dropped between k_current -> k_next
        num_to_compress = int(self.k_current) - int(self.k_next)
        unstable_fraction = 0.0 if num_to_compress <= 0 else float(num_changes) / float(num_to_compress)

        self._recent_changes.append(unstable_fraction)
        if len(self._recent_changes) >= self.stability_window:
            avg_change = sum(self._recent_changes) / len(self._recent_changes)
            if avg_change <= self.change_threshold:
                self.stage_completed = True

        self.last_mask.copy_(new_mask)
        self.last_change = sum(self._recent_changes) / len(self._recent_changes)
        return float(self.last_change)

    @torch.no_grad()
    def rewind(self) -> Dict:
        """
        LTH-style rewind to init under chosen mask.
        If stage_completed: adopt k_next mask, advance stage; else keep k_current mask.
        """
        if not self.store_init_for_rewind:
            raise RuntimeError("store_init_for_rewind=False; cannot rewind without init buffers.")

        # ----- IMPORTANT: latch whether we advanced BEFORE we mutate anything -----
        advanced = bool(self.stage_completed)
        self._just_advanced = advanced  # NEW: used by controller to decide compaction

        chosen = self._topk_unit_mask(k=self.k_next) if advanced else self._topk_unit_mask(k=self.k_current)
        self.mask.copy_(chosen)

        dev = self.fc1.weight.device
        dt = self.fc1.weight.dtype
        m = chosen.to(device=dev, dtype=dt)

        # restore init and apply structural zeros
        W1 = self.init_fc1_w.to(device=dev, dtype=dt)
        W2 = self.init_fc2_w.to(device=dev, dtype=dt)

        self.fc1.weight.data.copy_(W1 * m[:, None])
        self.fc2.weight.data.copy_(W2 * m[None, :])

        if self.fc1.bias is not None and hasattr(self, "init_fc1_b"):
            b1 = self.init_fc1_b.to(device=dev, dtype=dt)
            self.fc1.bias.data.copy_(b1 * m)
        if self.fc2.bias is not None and hasattr(self, "init_fc2_b"):
            b2 = self.init_fc2_b.to(device=dev, dtype=dt)
            self.fc2.bias.data.copy_(b2)

        # advance schedule if stage completed
        if advanced:
            if self.stage_idx < (self.num_stages - 1):
                self.stage_idx += 1
            self.k_current = int(self.k_schedule[self.stage_idx])
            self.k_next = (
                int(self.k_schedule[self.stage_idx + 1])
                if (self.stage_idx + 1) < self.num_stages
                else int(self.k_target)
            )

        # reset tracking
        self.last_mask.copy_(self._topk_unit_mask(k=self.k_next))
        self._recent_changes = deque(maxlen=self.stability_window)
        self.stage_completed = False
        self.schedule_done = (self.stage_idx == (self.num_stages - 1))

        stats = self.get_stats()
        stats["advanced"] = advanced  # NEW: for logging/debug
        return stats

    @torch.no_grad()
    def freeze_mask(self):
        """
        Freeze mask at current k_current selection.
        """
        self.mask.copy_(self._topk_unit_mask(k=self.k_current))
        self.mask_frozen = True

    # =========================
    # compaction (real compression)
    # =========================

    @torch.no_grad()
    def compact(self) -> nn.Module:
        """
        Physically shrink fc1/fc2 to only active units and return a compact dense MLP module.
        (No COO export needed.)
        """
        keep = torch.nonzero(self.mask, as_tuple=False).squeeze(1)
        k = int(keep.numel())
        if k == 0:
            raise RuntimeError("Mask has no active units; cannot compact.")

        dev = self.fc1.weight.device
        dt = self.fc1.weight.dtype

        fc1_new = nn.Linear(self.d_model, k, bias=self.fc1.bias is not None, device=dev, dtype=dt)
        fc2_new = nn.Linear(k, self.d_model, bias=self.fc2.bias is not None, device=dev, dtype=dt)

        fc1_new.weight.copy_(self.fc1.weight[keep, :])
        fc2_new.weight.copy_(self.fc2.weight[:, keep])

        if self.fc1.bias is not None:
            fc1_new.bias.copy_(self.fc1.bias[keep])
        if self.fc2.bias is not None:
            fc2_new.bias.copy_(self.fc2.bias)

        # You can wrap activation outside or keep it here; keeping it inside is convenient.
        return nn.Sequential(fc1_new, nn.GELU(), fc2_new)
    


    @torch.no_grad()
    def spawn_compacted_gate(self) -> tuple[nn.Linear, nn.Linear, "GatedMaskedParam"]:
        """
        Create NEW (fc1, fc2, gate) after a rewind advanced and mask is set.
        This physically shrinks to the current mask, and rebases the schedule:
          new_schedule = [k_current_old] + remaining tail (starting at old stage_idx)
        After compaction:
          - new hidden_dim = kept_units (== old k_current)
          - new k_current = new hidden_dim (dense in reduced space)
          - new k_next = next entry in rebased schedule
          - new init buffers = (old init * mask) sliced (so future rewinds are consistent)
        """
        if not self._just_advanced:
            raise RuntimeError("spawn_compacted_gate() called but gate did not just advance.")

        keep = torch.nonzero(self.mask, as_tuple=False).squeeze(1)
        keep_cpu = keep.detach().to("cpu", non_blocking=True).long()
        k_new = int(keep.numel())
        if k_new == 0:
            raise RuntimeError("Mask has no active units; cannot compact.")

        dev = self.fc1.weight.device
        dt = self.fc1.weight.dtype
        d_model = self.d_model

        # --- build new compact linears ---
        fc1_new = nn.Linear(d_model, k_new, bias=self.fc1.bias is not None, device=dev, dtype=dt)
        fc2_new = nn.Linear(k_new, d_model, bias=self.fc2.bias is not None, device=dev, dtype=dt)

        fc1_new.weight.copy_(self.fc1.weight[keep, :])
        fc2_new.weight.copy_(self.fc2.weight[:, keep])

        if self.fc1.bias is not None:
            fc1_new.bias.copy_(self.fc1.bias[keep])
        if self.fc2.bias is not None:
            fc2_new.bias.copy_(self.fc2.bias)

        # --- rebase schedule ---
        # old stage_idx has already been advanced inside rewind().
        # At this moment, old k_current equals old schedule[old stage_idx]
        # and equals k_new (units kept).
        tail = [int(x) for x in self.k_schedule[self.stage_idx:]]  # starts with current stage value
        # Sanity: first element should match k_new
        tail[0] = k_new  # force consistency in case of tiny drift

        # new gate: schedule starts at k_new and continues down
        gate_new = GatedMaskedParam(
            fc1=fc1_new,
            fc2=fc2_new,
            k_target=int(tail[-1]),
            k_schedule=tail,
            num_stages=len(tail),
            stability_window=self.stability_window,
            change_threshold=self.change_threshold,
            score=self.score_mode,
            ema_beta=self.ema_beta,
            store_init_for_rewind=self.store_init_for_rewind,
        )
        gate_new = gate_new.to(self.fc1.weight.device)
        # Make the new gate start "dense in its reduced space"
        gate_new.stage_idx = 0
        gate_new.k_current = k_new
        gate_new.k_next = int(tail[1]) if len(tail) > 1 else int(tail[-1])

        # In reduced space, everything is active initially
        gate_new.mask.fill_(True)
        gate_new.last_mask.copy_(gate_new._topk_unit_mask(gate_new.k_next))
        gate_new._recent_changes = deque(maxlen=gate_new.stability_window)
        gate_new.stage_completed = False
        gate_new.schedule_done = (gate_new.stage_idx == (gate_new.num_stages - 1))
        gate_new.mask_frozen = False
        gate_new._just_advanced = False  # consumed

        # --- set init buffers for the new gate to the *masked init* sliced ---
        if gate_new.store_init_for_rewind:
            # old init under mask (after rewind, weights == init * mask already)
            # We want init buffers to represent the starting point of the reduced search space.
            gate_new.init_fc1_w.copy_(self.fc1.weight.detach().cpu()[keep_cpu, :])
            gate_new.init_fc2_w.copy_(self.fc2.weight.detach().cpu()[:, keep_cpu])

            if self.fc1.bias is not None and hasattr(gate_new, "init_fc1_b"):
                gate_new.init_fc1_b.copy_(self.fc1.bias.detach().cpu()[keep_cpu])

            if self.fc2.bias is not None and hasattr(gate_new, "init_fc2_b"):
                gate_new.init_fc2_b.copy_(self.fc2.bias.detach().cpu())

        # --- slice EMA stats too (optional, but makes SNR stable if you use it) ---
        gate_new.g_mu.copy_(self.g_mu.detach().cpu()[keep_cpu])
        gate_new.g_m2.copy_(self.g_m2.detach().cpu()[keep_cpu])
        gate_new.g_steps.copy_(self.g_steps.detach().cpu())

        return fc1_new, fc2_new, gate_new

    # ---- keep init on cpu (optional same trick as you used) ----
    def _apply(self, fn):
        super_ret = super()._apply(fn)
        for name, buf in self.named_buffers(recurse=False):
            if name.startswith("init_") and buf is not None:
                self._buffers[name] = buf.detach().cpu()
        return super_ret

    # =========================
    # stats (MaskedParam-like)
    # =========================

    def get_stats(self) -> Dict:
        stats = {
            "hidden_dim": self.hidden_dim,
            "d_model": self.d_model,
            "score_mode": self.score_mode,
            "stage_idx": self.stage_idx,
            "num_stages": self.num_stages,
            "k_schedule": str(dict(enumerate(self.k_schedule))),
            "k_target": int(self.k_target),
            "k_current": int(self.k_current),
            "k_next": int(self.k_next),
            "kept_units": int(self.mask.sum().item()),
            "last_change": float(self.last_change),
            "change_threshold": float(self.change_threshold),
            "schedule_done": bool(self.schedule_done),
            "last_num_changes": self.last_num_changes,
            "mask_frozen": bool(self.mask_frozen),
        }
        return stats

    def extra_repr(self) -> str:
        return "\n".join(["GatedMaskedMLPParam with config:"] + [f"  {k}: {v}" for k, v in self.get_stats().items()])




class GatedMaskedAttentionParam(nn.Module):
    """
    Structured (latent-dim) pruning for multi-head attention with separate projections:

      q = q_proj(x) -> reshape (B,T,h,d_head) -> gate_qk -> used only for attention scores
      k = k_proj(x) -> reshape (B,T,h,d_head) -> gate_qk
      v = v_proj(x) -> reshape (B,T,h,d_head) -> gate_v  -> used for value transport
      out = attn(q,k,v) -> concat -> o_proj

    Two masks:
      - m_qk: (h, d_head)  shared for Q and K
      - m_v : (h, d_head)  for V (and structurally couples to o_proj during compact/rewind)

    Schedule/stability tracked per mask (optionally same schedule).
    """

    def __init__(
        self,
        q_proj: nn.Linear,
        k_proj: nn.Linear,
        v_proj: nn.Linear,
        o_proj: nn.Linear,
        num_heads: int,

        # targets are "k per head"
        k_target_qk: int,
        k_target_v: int,

        # optional explicit schedules (per head k), otherwise exponential decay from d_head
        k_schedule_qk: Optional[Sequence[int]] = None,
        k_schedule_v: Optional[Sequence[int]] = None,
        num_stages: int = 10,

        stability_window: int = 5,
        change_threshold: float = 0.01,

        score: ScoreFn = "sum_abs",
        ema_beta: float = 0.98,

        store_init_for_rewind: bool = True,
    ):
        super().__init__()

        self.q_proj = q_proj
        self.k_proj = k_proj
        self.v_proj = v_proj
        self.o_proj = o_proj

        self.num_heads = int(num_heads)
        self.score_mode: ScoreFn = score
        self.ema_beta = float(ema_beta)
        self.store_init_for_rewind = bool(store_init_for_rewind)

        # infer dims
        d_model = q_proj.in_features
        if not (q_proj.in_features == k_proj.in_features == v_proj.in_features == o_proj.out_features == d_model):
            raise ValueError("Expected q/k/v in_features == d_model and o_proj.out_features == d_model.")

        # by default GPT-style projections have out_features == d_model
        # after compaction we'll allow smaller out_features for q/k/v and smaller in_features for o_proj
        self.d_model = int(d_model)

        if self.d_model % self.num_heads != 0:
            raise ValueError(f"d_model={self.d_model} must be divisible by num_heads={self.num_heads}.")
        self.d_head_full = self.d_model // self.num_heads

        self.k_target_qk = int(k_target_qk)
        self.k_target_v = int(k_target_v)

        if not (1 <= self.k_target_qk <= self.d_head_full):
            raise ValueError(f"k_target_qk must be in [1,{self.d_head_full}], got {self.k_target_qk}")
        if not (1 <= self.k_target_v <= self.d_head_full):
            raise ValueError(f"k_target_v must be in [1,{self.d_head_full}], got {self.k_target_v}")

        # ----- init snapshots (CPU) for rewind -----
        if self.store_init_for_rewind:
            self.register_buffer("init_q_w", self.q_proj.weight.detach().cpu().clone(), persistent=True)
            self.register_buffer("init_k_w", self.k_proj.weight.detach().cpu().clone(), persistent=True)
            self.register_buffer("init_v_w", self.v_proj.weight.detach().cpu().clone(), persistent=True)
            self.register_buffer("init_o_w", self.o_proj.weight.detach().cpu().clone(), persistent=True)
            if self.q_proj.bias is not None:
                self.register_buffer("init_q_b", self.q_proj.bias.detach().cpu().clone(), persistent=True)
                self.register_buffer("init_k_b", self.k_proj.bias.detach().cpu().clone(), persistent=True)
                self.register_buffer("init_v_b", self.v_proj.bias.detach().cpu().clone(), persistent=True)
            if self.o_proj.bias is not None:
                self.register_buffer("init_o_b", self.o_proj.bias.detach().cpu().clone(), persistent=True)

        # ----- schedules (k per head) -----
        def make_schedule(k_target: int, sched: Optional[Sequence[int]]):
            if sched is not None:
                ks = [int(k) for k in sched]
                for i in range(1, len(ks)):
                    if ks[i] > ks[i - 1]:
                        raise ValueError(f"schedule must be non-increasing, got {sched}")
                if ks[-1] != k_target:
                    raise ValueError(f"schedule last value {ks[-1]} must equal k_target={k_target}")
                return ks
            else:
                base = exponential_decay(self.d_head_full, k_target, num_stages - 1)
                base.append(k_target)
                return [int(k) for k in base]

        self.k_schedule_qk = make_schedule(self.k_target_qk, k_schedule_qk)
        self.k_schedule_v = make_schedule(self.k_target_v, k_schedule_v)

        self.num_stages_qk = len(self.k_schedule_qk)
        self.num_stages_v = len(self.k_schedule_v)

        # stage state (separate)
        self.stage_idx_qk = 0
        self.k_current_qk = self.d_head_full
        self.k_next_qk = self.k_schedule_qk[1] if self.num_stages_qk > 1 else self.k_target_qk

        self.stage_idx_v = 0
        self.k_current_v = self.d_head_full
        self.k_next_v = self.k_schedule_v[1] if self.num_stages_v > 1 else self.k_target_v

        # ----- masks (buffers) -----
        self.register_buffer("mask_qk", torch.ones(self.num_heads, self.d_head_full, dtype=torch.bool), persistent=True)
        self.register_buffer("mask_v",  torch.ones(self.num_heads, self.d_head_full, dtype=torch.bool), persistent=True)

        self.register_buffer("last_mask_qk", torch.ones_like(self.mask_qk), persistent=True)
        self.register_buffer("last_mask_v",  torch.ones_like(self.mask_v), persistent=True)

        self.last_mask_qk.copy_(self._topk_mask_qk(self.k_next_qk))
        self.last_mask_v.copy_(self._topk_mask_v(self.k_next_v))

        # ----- stability tracking -----
        self.stability_window = int(stability_window)
        self.change_threshold = float(change_threshold)

        self._recent_changes_qk = deque(maxlen=self.stability_window)
        self._recent_changes_v = deque(maxlen=self.stability_window)

        self.last_change_qk = 1.0
        self.last_change_v = 1.0
        self.last_num_changes_qk: Optional[int] = None
        self.last_num_changes_v: Optional[int] = None

        self.stage_completed_qk = False
        self.stage_completed_v = False
        self.schedule_done_qk = False
        self.schedule_done_v = False

        self.mask_frozen_qk = False
        self.mask_frozen_v = False

        # ----- optional EMA grad stats for SNR (per head-dim) -----
        self.register_buffer("g_steps", torch.tensor(0, dtype=torch.long), persistent=True)
        self.register_buffer("g_mu_qk", torch.zeros(self.num_heads, self.d_head_full), persistent=True)
        self.register_buffer("g_m2_qk", torch.zeros(self.num_heads, self.d_head_full), persistent=True)
        self.register_buffer("g_mu_v",  torch.zeros(self.num_heads, self.d_head_full), persistent=True)
        self.register_buffer("g_m2_v",  torch.zeros(self.num_heads, self.d_head_full), persistent=True)

    # ========= scoring =========

    @torch.no_grad()
    def update_grad_stats(self, use_masks: bool = True):
        """
        Call after backward, before optimizer.step/zero_grad.
        Collect per (head, dim) gradient magnitude proxies for SNR scoring.
        """
        if self.q_proj.weight.grad is None or self.k_proj.weight.grad is None or self.v_proj.weight.grad is None:
            return

        beta = self.ema_beta
        self.g_steps += 1

        # reshape grads: (out, in) where out == d_model initially
        # map out-dim to (head, d_head)
        def per_headdim_l2(g_out_in: torch.Tensor, out_features_expected: int):
            # g: (out, in)
            g = g_out_in
            out = g.shape[0]
            if out != out_features_expected:
                # if already compacted, we cannot maintain full (h,d_head_full) EMA without extra bookkeeping
                # keep simple: skip EMA updates post-compaction
                return None
            g2 = g.pow(2).sum(dim=1).sqrt()  # (out,)
            return g2.view(self.num_heads, self.d_head_full)

        gq = per_headdim_l2(self.q_proj.weight.grad, self.d_model)
        gk = per_headdim_l2(self.k_proj.weight.grad, self.d_model)
        gv = per_headdim_l2(self.v_proj.weight.grad, self.d_model)

        if gq is None or gk is None or gv is None:
            return

        # QK tied signal
        uqk = gq + gk
        uv = gv

        if use_masks:
            uqk = uqk * self.mask_qk.to(uqk.dtype)
            uv = uv * self.mask_v.to(uv.dtype)

        self.g_mu_qk.mul_(beta).add_(uqk, alpha=1 - beta)
        self.g_m2_qk.mul_(beta).addcmul_(uqk, uqk, value=1 - beta)

        self.g_mu_v.mul_(beta).add_(uv, alpha=1 - beta)
        self.g_m2_v.mul_(beta).addcmul_(uv, uv, value=1 - beta)

    @torch.no_grad()
    def _scores_qk(self, eps: float = 1e-8) -> torch.Tensor:
        """
        returns (h, d_head_full)
        """
        if self.score_mode in ("sum_abs", "max_abs"):
            # Use row magnitudes of Wq/Wk, reshaped to (h, d_head_full)
            Wq = self.q_proj.weight
            Wk = self.k_proj.weight
            if Wq.shape[0] != self.d_model or Wk.shape[0] != self.d_model:
                # post-compaction scoring not supported in this simple version
                raise RuntimeError("Scoring after compaction is not supported in this version.")

            s_q = Wq.abs().sum(dim=1).view(self.num_heads, self.d_head_full)
            s_k = Wk.abs().sum(dim=1).view(self.num_heads, self.d_head_full)

            if self.score_mode == "sum_abs":
                return s_q + s_k
            return torch.maximum(s_q, s_k)

        if self.score_mode == "snr":
            var = (self.g_m2_qk - self.g_mu_qk.square()).clamp_min(0.0)
            return self.g_mu_qk / (var.sqrt() + eps)

        raise ValueError(f"Unknown score_mode={self.score_mode}")

    @torch.no_grad()
    def _scores_v(self, eps: float = 1e-8) -> torch.Tensor:
        """
        returns (h, d_head_full)
        """
        if self.score_mode in ("sum_abs", "max_abs"):
            Wv = self.v_proj.weight
            Wo = self.o_proj.weight
            if Wv.shape[0] != self.d_model:
                raise RuntimeError("Scoring after compaction is not supported in this version.")

            # V importance by row magnitude; O importance by corresponding column magnitude
            s_v = Wv.abs().sum(dim=1).view(self.num_heads, self.d_head_full)

            # Wo columns correspond to concatenated head dims at input of o_proj
            if Wo.shape[1] != self.d_model:
                # if o_proj already compacted in_features, we can’t map cleanly back to full dims
                raise RuntimeError("Scoring after compaction is not supported in this version.")

            s_o = Wo.abs().sum(dim=0).view(self.num_heads, self.d_head_full)

            if self.score_mode == "sum_abs":
                return s_v + s_o
            return torch.maximum(s_v, s_o)

        if self.score_mode == "snr":
            var = (self.g_m2_v - self.g_mu_v.square()).clamp_min(0.0)
            return self.g_mu_v / (var.sqrt() + eps)

        raise ValueError(f"Unknown score_mode={self.score_mode}")

    @torch.no_grad()
    def _topk_mask_from_scores(self, scores: torch.Tensor, k_per_head: int) -> torch.Tensor:
        """
        scores: (h, d_head_full)
        returns mask: (h, d_head_full) with exactly k_per_head True per head
        """
        k = int(k_per_head)
        idx = torch.topk(scores, k=k, dim=1, largest=True).indices  # (h, k)
        m = torch.zeros_like(scores, dtype=torch.bool)
        m.scatter_(1, idx, True)
        return m

    @torch.no_grad()
    def _topk_mask_qk(self, k_per_head: int) -> torch.Tensor:
        return self._topk_mask_from_scores(self._scores_qk(), k_per_head)

    @torch.no_grad()
    def _topk_mask_v(self, k_per_head: int) -> torch.Tensor:
        return self._topk_mask_from_scores(self._scores_v(), k_per_head)

    # ========= apply gating in forward =========

    def apply_qk(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        q,k: (B,T,h,d_head_full) in the *pre-compaction* layout.
        returns gated q,k.
        """
        if self.mask_frozen_qk:
            m = self.mask_qk
        else:
            m = self._topk_mask_qk(self.k_current_qk)

        m = m.to(dtype=q.dtype, device=q.device)  # (h, d_head_full)
        q = q * m[None, None, :, :]
        k = k * m[None, None, :, :]
        return q, k

    def apply_v(self, v: torch.Tensor) -> torch.Tensor:
        """
        v: (B,T,h,d_head_full) pre-compaction layout.
        """
        if self.mask_frozen_v:
            m = self.mask_v
        else:
            m = self._topk_mask_v(self.k_current_v)

        m = m.to(dtype=v.dtype, device=v.device)
        v = v * m[None, None, :, :]
        return v

    # ========= schedule / stability =========

    @torch.no_grad()
    def step_mask(self) -> Dict[str, float]:
        """
        Update both masks toward their next-stage k and track stability.
        Returns last_change for qk and v.
        """
        out = {"qk": 0.0, "v": 0.0}

        # --- QK ---
        if (not self.mask_frozen_qk) and (not self.schedule_done_qk):
            new_mask = self._topk_mask_qk(self.k_next_qk)
            num_changes = torch.logical_xor(self.last_mask_qk, new_mask).sum().item()
            self.last_num_changes_qk = int(num_changes)

            num_to_compress = self.num_heads * (int(self.k_current_qk) - int(self.k_next_qk))
            unstable_fraction = 0.0 if num_to_compress <= 0 else float(num_changes) / float(num_to_compress)
            self._recent_changes_qk.append(unstable_fraction)

            if len(self._recent_changes_qk) >= self.stability_window:
                avg = sum(self._recent_changes_qk) / len(self._recent_changes_qk)
                if avg <= self.change_threshold:
                    self.stage_completed_qk = True

            self.last_mask_qk.copy_(new_mask)
            self.last_change_qk = sum(self._recent_changes_qk) / len(self._recent_changes_qk)
            out["qk"] = float(self.last_change_qk)

        # --- V ---
        if (not self.mask_frozen_v) and (not self.schedule_done_v):
            new_mask = self._topk_mask_v(self.k_next_v)
            num_changes = torch.logical_xor(self.last_mask_v, new_mask).sum().item()
            self.last_num_changes_v = int(num_changes)

            num_to_compress = self.num_heads * (int(self.k_current_v) - int(self.k_next_v))
            unstable_fraction = 0.0 if num_to_compress <= 0 else float(num_changes) / float(num_to_compress)
            self._recent_changes_v.append(unstable_fraction)

            if len(self._recent_changes_v) >= self.stability_window:
                avg = sum(self._recent_changes_v) / len(self._recent_changes_v)
                if avg <= self.change_threshold:
                    self.stage_completed_v = True

            self.last_mask_v.copy_(new_mask)
            self.last_change_v = sum(self._recent_changes_v) / len(self._recent_changes_v)
            out["v"] = float(self.last_change_v)

        return out

    @torch.no_grad()
    def rewind(self) -> Dict:
        """
        LTH rewind under chosen masks:
          - If stage_completed_*: adopt k_next_* mask and advance stage.
          - Else: keep k_current_* mask.

        This version assumes projections are still in pre-compaction layout (out_features == d_model).
        """
        if not self.store_init_for_rewind:
            raise RuntimeError("store_init_for_rewind=False; cannot rewind without init buffers.")

        if self.q_proj.weight.shape[0] != self.d_model:
            raise RuntimeError("rewind() in this version expects pre-compaction projections.")

        # choose masks
        chosen_qk = self._topk_mask_qk(self.k_next_qk) if self.stage_completed_qk else self._topk_mask_qk(self.k_current_qk)
        chosen_v  = self._topk_mask_v(self.k_next_v)   if self.stage_completed_v  else self._topk_mask_v(self.k_current_v)

        self.mask_qk.copy_(chosen_qk)
        self.mask_v.copy_(chosen_v)

        dev = self.q_proj.weight.device
        dt = self.q_proj.weight.dtype

        # build row masks for out_features=d_model (concatenated heads)
        rowmask_qk = chosen_qk.reshape(-1).to(device=dev, dtype=dt)  # (d_model,)
        rowmask_v  = chosen_v.reshape(-1).to(device=dev, dtype=dt)   # (d_model,)

        # Q/K: zero rows not kept
        Wq = self.init_q_w.to(device=dev, dtype=dt)
        Wk = self.init_k_w.to(device=dev, dtype=dt)
        self.q_proj.weight.data.copy_(Wq * rowmask_qk[:, None])
        self.k_proj.weight.data.copy_(Wk * rowmask_qk[:, None])

        if self.q_proj.bias is not None:
            bq = self.init_q_b.to(device=dev, dtype=dt)
            bk = self.init_k_b.to(device=dev, dtype=dt)
            self.q_proj.bias.data.copy_(bq * rowmask_qk)
            self.k_proj.bias.data.copy_(bk * rowmask_qk)

        # V: zero rows not kept
        Wv = self.init_v_w.to(device=dev, dtype=dt)
        self.v_proj.weight.data.copy_(Wv * rowmask_v[:, None])
        if self.v_proj.bias is not None:
            bv = self.init_v_b.to(device=dev, dtype=dt)
            self.v_proj.bias.data.copy_(bv * rowmask_v)

        # O: zero columns corresponding to dropped V channels (structural coupling)
        Wo = self.init_o_w.to(device=dev, dtype=dt)  # (d_model, d_model)
        self.o_proj.weight.data.copy_(Wo * rowmask_v[None, :])
        if self.o_proj.bias is not None:
            bo = self.init_o_b.to(device=dev, dtype=dt)
            self.o_proj.bias.data.copy_(bo)

        # advance schedules if completed
        if self.stage_completed_qk:
            if self.stage_idx_qk < (self.num_stages_qk - 1):
                self.stage_idx_qk += 1
            self.k_current_qk = int(self.k_schedule_qk[self.stage_idx_qk])
            self.k_next_qk = int(self.k_schedule_qk[self.stage_idx_qk + 1]) if (self.stage_idx_qk + 1) < self.num_stages_qk else int(self.k_target_qk)

        if self.stage_completed_v:
            if self.stage_idx_v < (self.num_stages_v - 1):
                self.stage_idx_v += 1
            self.k_current_v = int(self.k_schedule_v[self.stage_idx_v])
            self.k_next_v = int(self.k_schedule_v[self.stage_idx_v + 1]) if (self.stage_idx_v + 1) < self.num_stages_v else int(self.k_target_v)

        # reset tracking
        self.last_mask_qk.copy_(self._topk_mask_qk(self.k_next_qk))
        self.last_mask_v.copy_(self._topk_mask_v(self.k_next_v))
        self._recent_changes_qk = deque(maxlen=self.stability_window)
        self._recent_changes_v = deque(maxlen=self.stability_window)

        self.stage_completed_qk = False
        self.stage_completed_v = False
        self.schedule_done_qk = (self.stage_idx_qk == (self.num_stages_qk - 1))
        self.schedule_done_v  = (self.stage_idx_v  == (self.num_stages_v  - 1))

        return self.get_stats()

    @torch.no_grad()
    def freeze_mask(self):
        """
        Freeze both masks at current k_current_*.
        """
        self.mask_qk.copy_(self._topk_mask_qk(self.k_current_qk))
        self.mask_v.copy_(self._topk_mask_v(self.k_current_v))
        self.mask_frozen_qk = True
        self.mask_frozen_v = True

    # ========= compaction =========

    @torch.no_grad()
    def compact(self) -> Dict:
        """
        Physically shrink q_proj/k_proj/v_proj outputs and o_proj input based on frozen masks.

        Resulting shapes:
          q_proj: (h*k_qk) x d_model
          k_proj: (h*k_qk) x d_model
          v_proj: (h*k_v ) x d_model
          o_proj: d_model x (h*k_v)

        After this, your attention implementation must reshape q,k,v using k_qk/k_v head dims.
        """
        if not (self.mask_frozen_qk and self.mask_frozen_v):
            raise RuntimeError("Call freeze_masks() before compact().")

        dev = self.q_proj.weight.device
        dt = self.q_proj.weight.dtype

        # indices of kept rows for qk and v in the concatenated layout
        keep_qk = torch.nonzero(self.mask_qk.reshape(-1), as_tuple=False).squeeze(1)  # (h*k_qk,)
        keep_v  = torch.nonzero(self.mask_v.reshape(-1), as_tuple=False).squeeze(1)   # (h*k_v,)

        k_qk = int(keep_qk.numel() // self.num_heads)
        k_v  = int(keep_v.numel()  // self.num_heads)

        # rebuild q/k/v projections with smaller out_features
        q_new = nn.Linear(self.d_model, self.num_heads * k_qk, bias=self.q_proj.bias is not None, device=dev, dtype=dt)
        k_new = nn.Linear(self.d_model, self.num_heads * k_qk, bias=self.k_proj.bias is not None, device=dev, dtype=dt)
        v_new = nn.Linear(self.d_model, self.num_heads * k_v,  bias=self.v_proj.bias is not None, device=dev, dtype=dt)

        q_new.weight.copy_(self.q_proj.weight[keep_qk, :])
        k_new.weight.copy_(self.k_proj.weight[keep_qk, :])
        v_new.weight.copy_(self.v_proj.weight[keep_v, :])

        if self.q_proj.bias is not None:
            q_new.bias.copy_(self.q_proj.bias[keep_qk])
            k_new.bias.copy_(self.k_proj.bias[keep_qk])
        if self.v_proj.bias is not None:
            v_new.bias.copy_(self.v_proj.bias[keep_v])

        # rebuild o_proj with smaller in_features = h*k_v
        o_new = nn.Linear(self.num_heads * k_v, self.d_model, bias=self.o_proj.bias is not None, device=dev, dtype=dt)

        # old o_proj weight: (d_model, d_model); we keep columns corresponding to keep_v
        o_new.weight.copy_(self.o_proj.weight[:, keep_v])
        if self.o_proj.bias is not None:
            o_new.bias.copy_(self.o_proj.bias)

        # swap modules in place
        self.q_proj = q_new
        self.k_proj = k_new
        self.v_proj = v_new
        self.o_proj = o_new

        # record new head dims (post-compaction)
        self.d_head_qk = k_qk
        self.d_head_v = k_v

        return {
            **self.get_stats(),
            "compacted": True,
            "d_head_qk": k_qk,
            "d_head_v": k_v,
            "q_out": self.q_proj.out_features,
            "v_out": self.v_proj.out_features,
            "o_in": self.o_proj.in_features,
        }

    # keep init buffers on CPU even if module .to() is called
    def _apply(self, fn):
        super_ret = super()._apply(fn)
        for name, buf in self.named_buffers(recurse=False):
            if name.startswith("init_") and buf is not None:
                self._buffers[name] = buf.detach().cpu()
        return super_ret

    def get_stats(self) -> Dict:
        return {
            "num_heads": self.num_heads,
            "d_model": self.d_model,
            "d_head_full": self.d_head_full,
            "score_mode": self.score_mode,
            "k_current_qk": int(self.k_current_qk),
            "k_next_qk": int(self.k_next_qk),
            "k_target_qk": int(self.k_target_qk),
            "k_current_v": int(self.k_current_v),
            "k_next_v": int(self.k_next_v),
            "k_target_v": int(self.k_target_v),
            "stage_idx_qk": int(self.stage_idx_qk),
            "stage_idx_v": int(self.stage_idx_v),
            "schedule_done_qk": bool(self.schedule_done_qk),
            "schedule_done_v": bool(self.schedule_done_v),
            "mask_frozen_qk": bool(self.mask_frozen_qk),
            "mask_frozen_v": bool(self.mask_frozen_v),
            "last_change_qk": float(self.last_change_qk),
            "last_change_v": float(self.last_change_v),
            "last_num_changes_qk": self.last_num_changes_qk,
            "last_num_changes_v": self.last_num_changes_v,
        }