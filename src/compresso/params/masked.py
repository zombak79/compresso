import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from typing import Optional, Sequence, Literal, List

from compresso.params.coo import CooSparseParam
from compresso.params.srp import SRPParam, SRPTensor
from compresso.functional.sparsify import topk_ste
from compresso.utils.schedule import exponential_decay

AggFn = Literal["sum_abs", "max_abs"]
TopKScoreMode = Literal["abs", "raw", "relu"]


class MaskedParam(nn.Module):
    """
    Dense parameter with a binary mask and an internal pruning schedule.
    """

    def __init__(
        self,
        weight: torch.Tensor,
        k_target: int,
        k_schedule: Optional[Sequence[int]] = None,
        num_stages: int = 10,
        stability_window: int = 5,
        change_threshold: float = 0.01,
        sparsity: Literal["row", "col"] = "row",
        allow_regrowth = True,
        score_mode: TopKScoreMode = "abs",
        ste_alpha: float = 1.0,
        post_norm_l1: bool = False,
    ):
        super().__init__()
        if weight.dim() != 2:
            raise ValueError(f"MaskedParam expects 2D weight, got shape {tuple(weight.shape)}")

        # set dim
        if sparsity == "row":
            self.dim = 1
        elif sparsity == "col":
            self.dim = 0
        else:
            raise ValueError(f"Sparsity could be row or col, got {sparsity}")

        self.rows, self.cols = weight.shape
        self.k_target = int(k_target)
        self.k_full = self.cols if self.dim==1 else self.rows
        self.allow_regrowth = allow_regrowth
        self.score_mode: TopKScoreMode = score_mode
        self.ste_alpha = float(ste_alpha)
        self.post_norm_l1 = bool(post_norm_l1)

        # --- Save original init on CPU as a buffer (checkpoint-safe) ---
        # stays on CPU by design; we move it on-demand in rewind()
        self.register_buffer("initialization", weight.detach().cpu().clone(), persistent=True)

        # --- Trainable dense weights (keep device of input weight) ---
        self.weight = nn.Parameter(weight.detach().clone())

        # --- grad stats ---
        self.register_buffer("g_mu", torch.zeros_like(self.weight, device=self.weight.device))
        self.register_buffer("g_m2", torch.zeros_like(self.weight, device=self.weight.device))
        self.register_buffer("g_steps", torch.tensor(0, dtype=torch.long, device=self.weight.device))

        # --- Schedule ---
        if k_schedule is not None:
            ks = [int(k) for k in k_schedule]
            for i in range(1, len(ks)):
                if ks[i] > ks[i - 1]:
                    raise ValueError(f"k_schedule must be non-increasing, got {k_schedule}")
            if ks[-1] != self.k_target:
                raise ValueError(
                    f"Last k in k_schedule ({ks[-1]}) must equal self.k_target ({self.k_target})"
                )
            self.k_schedule = ks
        else:
            if num_stages < 1:
                raise ValueError("num_stages must be >= 1")
            self.k_schedule = exponential_decay(self.k_full, self.k_target, num_stages - 1)
            self.k_schedule.append(self.k_target)

        self.num_stages = len(self.k_schedule)

        # These are Python scalars. To get perfect resume under exotic wrappers (FSDP), need to
        # convert them to buffers too. For now: keep minimal changes.
        self.stage_idx = 0
        self.k_current = self.k_full
        self.k_next = (
            self.k_schedule[self.stage_idx + 1] if (self.stage_idx + 1) < self.num_stages else self.k_target
        )

        # --- Mask state as buffers (device-safe, checkpoint-safe) ---
        self.register_buffer("mask", torch.ones(self.rows, self.cols, dtype=torch.bool), persistent=True)
        self.register_buffer("last_mask", torch.ones(self.rows, self.cols, dtype=torch.bool), persistent=True)
        self.last_mask.copy_(self.topk_mask(k=self.k_next))

        # --- Stability tracking ---
        self.stability_window = int(stability_window)
        self.change_threshold = float(change_threshold)
        self._recent_changes = deque(maxlen=self.stability_window)

        self.last_change = 1.0
        self.last_num_changes: Optional[int] = None
        self.stage_completed = False
        self.schedule_done = False
        self.mask_frozen = False

    def topk_weights_with_mask(self, k=None, return_mask=True):
        k = int(k) if k is not None else int(self.k_current)
        e = self.weight
        if self.score_mode == "abs":
            scores = e.abs()
        elif self.score_mode == "raw":
            scores = e
        else:
            scores = e.relu()
        idx = torch.topk(scores, k, dim=self.dim).indices
        mask = torch.zeros_like(e, dtype=torch.bool).scatter(self.dim, idx, True)
        out = topk_ste(e, k=k, dim=self.dim, score_mode=self.score_mode, ste_alpha=self.ste_alpha)
        if self.post_norm_l1:
            out = F.normalize(out, p=1.0, dim=self.dim)
        return (out, mask) if return_mask else out

    def topk_weights(self, k=None):
        return self.topk_weights_with_mask(k=k, return_mask=False)

    def topk_mask(self, k=None):
        return self.topk_weights_with_mask(k=k, return_mask=True)[1]

    def forward(self):
        if self.mask_frozen:
            out = self.weight * self.mask.to(self.weight.dtype)
            if self.post_norm_l1:
                out = F.normalize(out, p=1.0, dim=self.dim)
            return out
        else:
            return self.topk_weights()
    
    def srp(self):
        assert self.dim == 1
        return SRPTensor.from_dense(self.weight, k=int(self.k_current), score_mode=self.score_mode)

    @torch.no_grad()
    def update_grad_stats(self, beta: float = 0.98, use_mask: bool = True):
        if self.weight.grad is None:
            return
        g = self.weight.grad
        if use_mask:
            # measure signal only on currently active weights (recommended)
            g = g * (self.mask.to(g.dtype) if self.mask_frozen else self.topk_mask().to(g.dtype))

        self.g_steps += 1
        self.g_mu.mul_(beta).add_(g, alpha=1 - beta)
        self.g_m2.mul_(beta).addcmul_(g, g, value=1 - beta)

    @torch.no_grad()
    def snr(self, eps: float = 1e-8):
        var = (self.g_m2 - self.g_mu.square()).clamp_min(0.0)
        return self.g_mu.abs() #/ (var.sqrt() + eps)   # same shape as weights
        
        
        var = (self.g_m2 - self.g_mu.square()).clamp_min(0.0)
        # collapse columns/rows -> one score per row/column
        mu = self.g_mu.norm(dim=self.dim)
        std = var.sum(dim=self.dim).sqrt().add_(eps)      
        return mu / std                        

    @torch.no_grad()
    def step_mask(self):
        """Compute mask and add history"""
        if self.mask_frozen or self.schedule_done:
            return 0.0

        new_mask = self.topk_mask(k=self.k_next)

        # XOR is cheapest on bool
        num_changes = torch.logical_xor(self.last_mask, new_mask).sum().item()
        self.last_num_changes = int(num_changes)

        # how many entries could change when moving from k_current -> k_next?
        if self.dim==1:
            num_to_compress = self.rows * (int(self.k_current) - int(self.k_next))
        else:
            num_to_compress = self.cols * (int(self.k_current) - int(self.k_next))
        if num_to_compress <= 0:
            unstable_fraction = 0.0
        else:
            unstable_fraction = float(num_changes) / float(num_to_compress)

        self._recent_changes.append(unstable_fraction)

        if len(self._recent_changes) >= self.stability_window:
            avg_change = sum(self._recent_changes) / len(self._recent_changes)
            if avg_change <= self.change_threshold:
                self.stage_completed = True

        # update buffer in-place
        self.last_mask.copy_(new_mask)

        self.last_change = sum(self._recent_changes) / len(self._recent_changes)
        return float(self.last_change)

    @torch.no_grad()
    def rewind(self):
        """Back to original init with new k"""
        # choose mask to apply
        chosen_mask = self.topk_mask(k=self.k_next) if self.stage_completed else self.topk_mask()

        # update mask buffer in-place
        self.mask.copy_(chosen_mask)

        # rewind weights to init * mask
        init = self.initialization.to(device=self.weight.device, dtype=self.weight.dtype)
        self.weight.data.copy_(init * chosen_mask.to(self.weight.dtype))

        # advance stage if completed
        if self.stage_completed:
            if self.stage_idx < (self.num_stages - 1):
                self.stage_idx += 1
            self.k_current = int(self.k_schedule[self.stage_idx])
            self.k_next = (
                int(self.k_schedule[self.stage_idx + 1]) if (self.stage_idx + 1) < self.num_stages else int(self.k_target)
            )

        # refresh last_mask buffer for next step
        self.last_mask.copy_(self.topk_mask(k=self.k_next))

        # reset stability window
        self._recent_changes = deque(maxlen=self.stability_window)
        self.stage_completed = False
        self.schedule_done = (self.stage_idx == (self.num_stages - 1))

        return self.get_stats()

    @torch.no_grad()
    def freeze_mask(self):
        # freeze to current k_current topk mask
        self.mask.copy_(self.topk_mask(k=self.k_current))
        self.mask_frozen = True

    @torch.no_grad()
    def maskedparam_to_coo(self):
        """
        Export packed fixed-k COO according to self.dim:
        - dim==1 -> row-packed, fixed k per row
        - dim==0 -> col-packed, fixed k per col
        """
        W = self.weight
        rows, cols = W.shape

        # choose which mask to export
        if self.mask_frozen:
            mask = self.mask
        else:
            mask = self.topk_mask(k=self.k_current)

        # We need topk indices in a packed order, not generic COO.
        k = int(self.k_current)

        if self.dim == 1:
            # ---- row-wise sparsity: per row pick k cols ----
            # topk over abs(W) along cols
            tk = torch.topk(W.abs(), k, dim=1)
            col_idx_2d = tk.indices                       # (rows, k)
            row_idx_2d = torch.arange(rows, device=W.device).view(-1, 1).expand(rows, k)

            row_idx = row_idx_2d.reshape(-1)              # (rows*k,)
            col_idx = col_idx_2d.reshape(-1)              # (rows*k,)
            values  = W[row_idx, col_idx].detach().clone()

            indices = torch.stack([row_idx, col_idx], dim=0)

            return CooSparseParam(indices=indices,
                                values=values,
                                shape=(rows, cols),
                                packed_dim="row")

        elif self.dim == 0:
            # ---- col-wise sparsity: per col pick k rows ----
            # topk over abs(W) along rows
            tk = torch.topk(W.abs(), k, dim=0)
            row_idx_2d = tk.indices                       # (k, cols)
            col_idx_2d = torch.arange(cols, device=W.device).view(1, -1).expand(k, cols)

            # We want COL-PACKED layout: col 0 block, then col 1 block, ...
            # So flatten in column-major block order: (cols, k)
            row_idx = row_idx_2d.t().reshape(-1)          # (cols*k,)
            col_idx = col_idx_2d.t().reshape(-1)          # (cols*k,)
            values  = W[row_idx, col_idx].detach().clone()

            indices = torch.stack([row_idx, col_idx], dim=0)

            return CooSparseParam(indices=indices,
                                values=values,
                                shape=(rows, cols),
                                packed_dim="col")

        else:
            raise ValueError(f"Unsupported dim={self.dim}. Use 0 (col) or 1 (row).")

    



    @torch.no_grad()
    def maskedparam_to_srp(self) -> "SRPParam":
        """Export fixed-k row-packed SRPParam.

        Only valid for row-wise sparsity (self.dim == 1), because SRPParam stores
        exactly k entries per ROW.

        Uses:

        * frozen mask if ``mask_frozen``
        * otherwise current-stage top-k mask, ``k_current``

        Returns:

        ``SRPParam`` with ``cols`` shaped ``(rows, k_current)``, ``values``
        shaped ``(rows, k_current)``, and shape ``(rows, cols)``.
        """
        if self.dim != 1:
            raise ValueError(
                "maskedparam_to_srp() only supports row-wise sparsity (dim==1). "
                "For col-wise sparsity, export COO with packed_dim='col' (or wait for implementation of a ColPackedParam)."
            )

        W = self.weight
        rows, cols = W.shape
        k = int(self.k_current)

        # choose which mask to export
        if self.mask_frozen:
            mask = self.mask
        else:
            mask = self.topk_mask(k=k)

        # Build row-packed cols2d in the correct order: for each row, the k active columns.
        # Since mask is boolean, we can get indices by topk on mask.float().
        # But safer/cleaner: compute topk on abs(W) with regrowth constraint consistent with mask.
        #
        # We want the *active* positions. If mask is from topk already, this will match.
        # If allow_regrowth=False and mask is older, still OK.
        #
        # Easiest: use topk on (abs(W) * mask) to get k cols per row.
        score = W.abs()
        score = score * mask.to(score.dtype)

        # If something went wrong and a row has <k active entries (shouldn't happen), topk will break.
        # So assert in debug style:
        active_per_row = mask.sum(dim=1)
        if int(active_per_row.min().item()) < k:
            raise RuntimeError(
                f"Cannot export SRP: some rows have < k_current active entries. "
                f"min_active={int(active_per_row.min().item())}, k_current={k}"
            )

        tk = torch.topk(score, k=k, dim=1, largest=True)
        cols2d = tk.indices.to(torch.long)  # (rows,k)

        # Values: gather from W (not score) to preserve sign.
        values2d = W.gather(1, cols2d).detach().clone()  # (rows,k)

        # IMPORTANT: SRPParam semantics are scatter_add per row; topk gives unique cols per row, so fine.
        return SRPParam(cols=cols2d.detach().clone(), values=values2d, shape=(rows, cols), validate=True)

    # ---- keep init on cpu -----
    def _apply(self, fn):
        # apply to everything first
        super_ret = super()._apply(fn)
        # move init_* back to CPU (and keep dtype)
        for name, buf in self.named_buffers(recurse=False):
            if name.startswith("init_") and buf is not None:
                # ensure still in state_dict but kept on CPU
                self._buffers[name] = buf.detach().cpu()
        return super_ret

    def get_stats(self):
        stats = {
            "stage_idx": self.stage_idx,
            "num_stages": self.num_stages,
            "k_schedule": str(dict(enumerate(self.k_schedule))),
            "k_target": self.k_target,
            "k_current": int(self.k_current),
            "current_params": (self.rows if self.dim==1 else self.cols) * int(self.k_current),
            "target_params": (self.rows if self.dim==1 else self.cols) * int(self.k_target),
            "k_next": int(self.k_next),
            "last_change": float(self.last_change),
            "change_threshold": float(self.change_threshold),
            "schedule_done": bool(self.schedule_done),
            "last_num_changes": self.last_num_changes,
            "mask_frozen": bool(self.mask_frozen),
        }
        return stats

    def extra_repr(self):
        return "\n".join(["MaskedParam with config:"] + [f"  {k}: {v}" for k, v in self.get_stats().items()])


class _SharedMaskedParam(nn.Module):
    """
    A group of dense parameters that share ONE pruning mask and ONE schedule.

    - Holds N trainable dense weights of identical shape (rows, cols).
    - Computes a shared binary mask via top-k on an aggregate "score" tensor.
    - Applies that shared mask to all weights in forward.
    - Supports ELSA-style rewinding: weight_i <- init_i * mask at stage transitions.

    Sparsity semantics:
      - sparsity="row" => keep k per ROW (topk along dim=1) => k_full = cols
      - sparsity="col" => keep k per COL (topk along dim=0) => k_full = rows
    """

    def __init__(
        self,
        weights: Sequence[torch.Tensor],
        k_target: int,
        k_schedule: Optional[Sequence[int]] = None,
        num_stages: int = 10,
        stability_window: int = 5,
        change_threshold: float = 0.01,
        sparsity: Literal["row", "col"] = "row",
        agg: AggFn = "sum_abs",
    ):
        super().__init__()
        if len(weights) == 0:
            raise ValueError("SharedMaskedParam requires at least one weight tensor.")

        if any(w.dim() != 2 for w in weights):
            raise ValueError("All weights must be 2D tensors.")

        rows, cols = weights[0].shape
        for i, w in enumerate(weights):
            if w.shape != (rows, cols):
                raise ValueError(
                    f"All weights must have the same shape. "
                    f"weights[0]={rows, cols}, weights[{i}]={tuple(w.shape)}"
                )

        self.rows, self.cols = rows, cols
        self.k_target = int(k_target)
        self.agg: AggFn = agg

        if sparsity == "row":
            self.dim = 1
            self.k_full = self.cols
            self.packed_dim: Literal["row", "col"] = "row"
        elif sparsity == "col":
            self.dim = 0
            self.k_full = self.rows
            self.packed_dim = "col"
        else:
            raise ValueError(f"sparsity must be 'row' or 'col', got {sparsity}")

        # --- Trainable weights ---
        self.weights = nn.ParameterList([nn.Parameter(w.detach().clone()) for w in weights])

        # --- Per-weight initializations stored on CPU (checkpoint-safe buffers) ---
        # register each init_i as a buffer so state_dict captures it.
        for i, w in enumerate(weights):
            self.register_buffer(f"init_{i}", w.detach().cpu().clone(), persistent=True)

        # --- Schedule ---
        if k_schedule is not None:
            ks = [int(k) for k in k_schedule]
            for i in range(1, len(ks)):
                if ks[i] > ks[i - 1]:
                    raise ValueError(f"k_schedule must be non-increasing, got {k_schedule}")
            if ks[-1] != self.k_target:
                raise ValueError(
                    f"Last k in k_schedule ({ks[-1]}) must equal k_target ({self.k_target})"
                )
            self.k_schedule = ks
        else:
            if num_stages < 1:
                raise ValueError("num_stages must be >= 1")
            self.k_schedule = exponential_decay(self.k_full, self.k_target, num_stages - 1)
            self.k_schedule.append(self.k_target)

        self.num_stages = len(self.k_schedule)

        self.stage_idx = 0
        self.k_current = int(self.k_full)
        self.k_next = (
            int(self.k_schedule[self.stage_idx + 1]) if (self.stage_idx + 1) < self.num_stages else int(self.k_target)
        )

        # --- Shared mask state (buffers) ---
        self.register_buffer("mask", torch.ones(self.rows, self.cols, dtype=torch.bool), persistent=True)
        self.register_buffer("last_mask", torch.ones(self.rows, self.cols, dtype=torch.bool), persistent=True)
        self.last_mask.copy_(self.topk_mask(k=self.k_next))

        # --- Stability tracking ---
        self.stability_window = int(stability_window)
        self.change_threshold = float(change_threshold)
        self._recent_changes = deque(maxlen=self.stability_window)

        self.last_change = 1.0
        self.last_num_changes: Optional[int] = None
        self.stage_completed = False
        self.schedule_done = False
        self.mask_frozen = False

    # ---------- scoring / masking ----------

    def _aggregate_score(self) -> torch.Tensor:
        """
        Build an aggregate score tensor (rows, cols) from all weights.

        Default: sum_abs => score = sum_i |W_i|
        Alternative: max_abs => score = max_i |W_i|
        """
        if self.agg == "sum_abs":
            score = torch.zeros_like(self.weights[0])
            for w in self.weights:
                score.add_(w.abs())
            return score

        if self.agg == "max_abs":
            score = self.weights[0].abs()
            for w in self.weights[1:]:
                score = torch.maximum(score, w.abs())
            return score

        raise ValueError(f"Unknown agg='{self.agg}'")

    def topk_mask(self, k: Optional[int] = None) -> torch.Tensor:
        """
        Compute the shared top-k mask from the aggregate score.
        """
        k = int(k) if k is not None else int(self.k_current)
        score = self._aggregate_score()
        tk = torch.topk(score, k, dim=self.dim)
        mask = torch.zeros_like(score, dtype=torch.bool).scatter(self.dim, tk.indices, True)
        return mask

    def forward(self, return_stacked: bool = False) -> torch.Tensor | List[torch.Tensor]:
        """
        Returns masked dense weights.

        If return_stacked=False:
            returns list[Tensor] with len = num_params, each shape (rows, cols)
        If return_stacked=True:
            returns Tensor shape (num_params, rows, cols)
        """
        if self.mask_frozen:
            m = self.mask.to(self.weights[0].dtype)
        else:
            m = self.topk_mask(k=self.k_current).to(self.weights[0].dtype)

        outs = [w * m for w in self.weights]
        if return_stacked:
            return torch.stack(outs, dim=0)
        return outs

    def masked_weight(self, idx: int) -> torch.Tensor:
        """
        Convenience: get masked dense weight for one member.
        """
        return self(return_stacked=False)[idx]

    # ---------- schedule / stability ----------

    @torch.no_grad()
    def step_mask(self) -> float:
        """
        Compute next-stage mask (k_next), track changes vs last_mask, and set stage_completed when stable.
        """
        if self.mask_frozen or self.schedule_done:
            return 0.0

        new_mask = self.topk_mask(k=self.k_next)

        num_changes = torch.logical_xor(self.last_mask, new_mask).sum().item()
        self.last_num_changes = int(num_changes)

        # how many entries could change when moving from k_current -> k_next?
        # For row-sparsity: rows*(k_current-k_next)
        # For col-sparsity: cols*(k_current-k_next)
        packed = self.rows if self.dim == 1 else self.cols
        num_to_compress = packed * (int(self.k_current) - int(self.k_next))

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
    def rewind(self) -> dict:
        """
        Apply chosen mask to all params and rewind each param to its own CPU initialization * mask.

        If stage_completed=True: advances stage index / k_current / k_next.
        """
        chosen_mask = self.topk_mask(k=self.k_next) if self.stage_completed else self.topk_mask(k=self.k_current)
        self.mask.copy_(chosen_mask)

        m = chosen_mask.to(dtype=self.weights[0].dtype, device=self.weights[0].device)

        for i, w in enumerate(self.weights):
            init_cpu = getattr(self, f"init_{i}")             # CPU buffer
            init = init_cpu.to(device=w.device, dtype=w.dtype)
            w.data.copy_(init * m)

        if self.stage_completed:
            if self.stage_idx < (self.num_stages - 1):
                self.stage_idx += 1
            self.k_current = int(self.k_schedule[self.stage_idx])
            self.k_next = (
                int(self.k_schedule[self.stage_idx + 1]) if (self.stage_idx + 1) < self.num_stages else int(self.k_target)
            )

        self.last_mask.copy_(self.topk_mask(k=self.k_next))
        self._recent_changes = deque(maxlen=self.stability_window)

        self.stage_completed = False
        self.schedule_done = (self.stage_idx == (self.num_stages - 1))

        return self.get_stats()

    @torch.no_grad()
    def freeze_mask(self):
        """
        Freeze mask at current k_current.
        """
        self.mask.copy_(self.topk_mask(k=self.k_current))
        self.mask_frozen = True

    # ---------- COO export (packed, fixed-k) ----------

    @torch.no_grad()
    def shared_mask_to_packed_indices(self, k: Optional[int] = None) -> torch.Tensor:
        """
        Build packed COO indices for the CURRENT topk selection.

        If sparsity="row": returns ROW-PACKED indices (2, rows*k), packed_dim="row"
        If sparsity="col": returns COL-PACKED indices (2, cols*k), packed_dim="col"

        This does NOT call to_sparse_coo(); it builds packed indices explicitly.
        """
        k = int(k) if k is not None else int(self.k_current)

        score = self._aggregate_score()
        tk = torch.topk(score, k, dim=self.dim)

        if self.dim == 1:
            # row-wise: tk.indices is (rows, k) with column ids per row
            col_idx_2d = tk.indices
            row_idx_2d = torch.arange(self.rows, device=score.device).view(-1, 1).expand(self.rows, k)
            row_idx = row_idx_2d.reshape(-1)
            col_idx = col_idx_2d.reshape(-1)
            return torch.stack([row_idx, col_idx], dim=0)

        else:
            # col-wise: tk.indices is (k, cols) with row ids per col
            row_idx_2d = tk.indices
            col_idx_2d = torch.arange(self.cols, device=score.device).view(1, -1).expand(k, self.cols)

            # Make it COL-PACKED: col 0 block, then col 1 block, ...
            row_idx = row_idx_2d.t().reshape(-1)   # (cols*k,)
            col_idx = col_idx_2d.t().reshape(-1)   # (cols*k,)
            return torch.stack([row_idx, col_idx], dim=0)

    @torch.no_grad()
    def maskedparam_to_coo(self) -> List[CooSparseParam]:
        """
        Export each weight as a CooSparseParam that shares identical packed indices,
        but has its own trainable values.

        This returns a list of CooSparseParam (one per weight).
        """
        # Use frozen mask if frozen, otherwise current stage k_current selection
        k = int(self.k_current)
        indices = self.shared_mask_to_packed_indices(k=k)

        # Gather values for each weight in the packed order
        row = indices[0]
        col = indices[1]
        rows, cols = self.rows, self.cols

        out: List[CooSparseParam] = []
        for w in self.weights:
            values = w.detach()[row, col].clone()  # (packed_nnz,)
            out.append(
                CooSparseParam(
                    indices=indices,
                    values=values,
                    shape=(rows, cols),
                    packed_dim=self.packed_dim,  # "row" if dim==1 else "col"
                )
            )
        return out

    # ---- keep init on cpu -----
    def _apply(self, fn):
        # apply to everything first
        super_ret = super()._apply(fn)
        # move init_* back to CPU (and keep dtype)
        for name, buf in self.named_buffers(recurse=False):
            if name.startswith("init_") and buf is not None:
                # ensure still in state_dict but kept on CPU
                self._buffers[name] = buf.detach().cpu()
        return super_ret
    
    # ---------- stats ----------

    def get_stats(self) -> dict:
        packed = self.rows if self.dim == 1 else self.cols
        stats = {
            "num_params": len(self.weights),
            "agg": self.agg,
            "stage_idx": self.stage_idx,
            "num_stages": self.num_stages,
            "k_schedule": str(dict(enumerate(self.k_schedule))),
            "k_target": int(self.k_target),
            "k_full": int(self.k_full),
            "k_current": int(self.k_current),
            "k_next": int(self.k_next),
            "packed_dim": self.packed_dim,
            "current_params_per_weight": int(packed * int(self.k_current)),
            "target_params_per_weight": int(packed * int(self.k_target)),
            "last_change": float(self.last_change),
            "change_threshold": float(self.change_threshold),
            "schedule_done": bool(self.schedule_done),
            "last_num_changes": self.last_num_changes,
            "mask_frozen": bool(self.mask_frozen),
        }
        return stats

    def extra_repr(self) -> str:
        return "\n".join(["SharedMaskedParam with config:"] + [f"  {k}: {v}" for k, v in self.get_stats().items()])

class SharedMaskedParam(nn.Module):
    """
    A group of dense parameters that share ONE pruning mask and ONE schedule.

    - Holds N trainable dense weights of identical shape (rows, cols).
    - Computes a shared binary mask via top-k on an aggregate "score" tensor.
    - Applies that shared mask to all weights in forward.
    - Supports ELSA-style rewinding: weight_i <- init_i * mask at stage transitions.

    Sparsity semantics (EDGE sparsity):
      - sparsity="row" => keep k entries per ROW (topk along dim=1) => k_full = cols
      - sparsity="col" => keep k entries per COL (topk along dim=0) => k_full = rows

    Optional head-awareness:
      - If num_heads is set, top-k is performed independently within each head block
        along the selected axis (dim), enforcing equal budget per head.
      - Requires: selected axis size divisible by num_heads, and all k values divisible by num_heads.
    """

    def __init__(
        self,
        weights: Sequence[torch.Tensor],
        k_target: int,
        k_schedule: Optional[Sequence[int]] = None,
        num_stages: int = 10,
        stability_window: int = 5,
        change_threshold: float = 0.01,
        sparsity: Literal["row", "col"] = "row",
        agg: AggFn = "sum_abs",
        num_heads: Optional[int] = None,  # NEW
    ):
        super().__init__()
        if len(weights) == 0:
            raise ValueError("SharedMaskedParam requires at least one weight tensor.")
        if any(w.dim() != 2 for w in weights):
            raise ValueError("All weights must be 2D tensors.")

        rows, cols = weights[0].shape
        for i, w in enumerate(weights):
            if w.shape != (rows, cols):
                raise ValueError(
                    f"All weights must have the same shape. "
                    f"weights[0]={rows, cols}, weights[{i}]={tuple(w.shape)}"
                )

        self.rows, self.cols = int(rows), int(cols)
        self.k_target = int(k_target)
        self.agg: AggFn = agg
        self.num_heads = int(num_heads) if num_heads is not None else None

        if sparsity == "row":
            self.dim = 1
            self.k_full = self.cols
            self.packed_dim: Literal["row", "col"] = "row"
            head_axis = self.cols
        elif sparsity == "col":
            self.dim = 0
            self.k_full = self.rows
            self.packed_dim = "col"
            head_axis = self.rows
        else:
            raise ValueError(f"sparsity must be 'row' or 'col', got {sparsity}")

        # --- Head-aware validation ---
        if self.num_heads is not None:
            if self.num_heads <= 0:
                raise ValueError(f"num_heads must be positive, got {self.num_heads}")
            if head_axis % self.num_heads != 0:
                raise ValueError(
                    f"Selected axis size ({head_axis}) must be divisible by num_heads={self.num_heads}."
                )
            if self.k_target % self.num_heads != 0:
                raise ValueError(
                    f"k_target={self.k_target} must be divisible by num_heads={self.num_heads} "
                    f"for equal per-head topk."
                )

        # --- Trainable weights ---
        self.weights = nn.ParameterList([nn.Parameter(w.detach().clone()) for w in weights])

        # --- Per-weight initializations stored on CPU (checkpoint-safe buffers) ---
        for i, w in enumerate(weights):
            self.register_buffer(f"init_{i}", w.detach().cpu().clone(), persistent=True)

        # --- Schedule ---
        if k_schedule is not None:
            ks = [int(k) for k in k_schedule]
            for i in range(1, len(ks)):
                if ks[i] > ks[i - 1]:
                    raise ValueError(f"k_schedule must be non-increasing, got {k_schedule}")
            if ks[-1] != self.k_target:
                raise ValueError(
                    f"Last k in k_schedule ({ks[-1]}) must equal k_target ({self.k_target})"
                )
            if self.num_heads is not None:
                for k in ks:
                    if k % self.num_heads != 0:
                        raise ValueError(
                            f"All k in k_schedule must be divisible by num_heads={self.num_heads}. Got k={k}."
                        )
            self.k_schedule = ks
        else:
            if num_stages < 1:
                raise ValueError("num_stages must be >= 1")
            sched = exponential_decay(self.k_full, self.k_target, num_stages - 1)
            sched.append(self.k_target)

            # Snap to multiples of num_heads if needed (keep non-increasing)
            if self.num_heads is not None:
                snapped = []
                last = None
                for k in sched:
                    k = int(k)
                    k = max(self.num_heads, (k // self.num_heads) * self.num_heads)
                    if last is not None:
                        k = min(k, last)
                    snapped.append(k)
                    last = k
                snapped[-1] = self.k_target
                sched = snapped

            self.k_schedule = [int(k) for k in sched]

        self.num_stages = len(self.k_schedule)

        self.stage_idx = 0
        self.k_current = int(self.k_full)
        self.k_next = (
            int(self.k_schedule[self.stage_idx + 1])
            if (self.stage_idx + 1) < self.num_stages
            else int(self.k_target)
        )

        # --- Shared mask state (buffers) ---
        self.register_buffer("mask", torch.ones(self.rows, self.cols, dtype=torch.bool), persistent=True)
        self.register_buffer("last_mask", torch.ones(self.rows, self.cols, dtype=torch.bool), persistent=True)
        self.last_mask.copy_(self.topk_mask(k=self.k_next))

        # --- Stability tracking ---
        self.stability_window = int(stability_window)
        self.change_threshold = float(change_threshold)
        self._recent_changes = deque(maxlen=self.stability_window)

        self.last_change = 1.0
        self.last_num_changes: Optional[int] = None
        self.stage_completed = False
        self.schedule_done = False
        self.mask_frozen = False

    # ---------- scoring / masking ----------

    def _aggregate_score(self) -> torch.Tensor:
        """
        Build an aggregate score tensor (rows, cols) from all weights.

        sum_abs => score = sum_i |W_i|
        max_abs => score = max_i |W_i|
        """
        if self.agg == "sum_abs":
            score = torch.zeros_like(self.weights[0])
            for w in self.weights:
                score.add_(w.abs())
            return score

        if self.agg == "max_abs":
            score = self.weights[0].abs()
            for w in self.weights[1:]:
                score = torch.maximum(score, w.abs())
            return score

        raise ValueError(f"Unknown agg='{self.agg}'")

    def _head_aware_topk_indices(self, score: torch.Tensor, k: int) -> torch.Tensor:
        """
        Returns topk indices with optional head-awareness, matching torch.topk(..., dim=self.dim).indices
        shapes:
          - dim==1 => (rows, k) absolute column ids
          - dim==0 => (k, cols) absolute row ids
        """
        if self.num_heads is None:
            return torch.topk(score, k, dim=self.dim).indices

        nh = self.num_heads
        if k % nh != 0:
            raise ValueError(f"k={k} must be divisible by num_heads={nh}.")
        kph = k // nh

        if self.dim == 1:
            # score: (rows, cols) -> (rows, nh, head_dim)
            head_dim = self.cols // nh
            s = score.view(self.rows, nh, head_dim)
            tk = torch.topk(s, kph, dim=2).indices  # (rows, nh, kph) in [0, head_dim)
            offsets = (torch.arange(nh, device=score.device) * head_dim).view(1, nh, 1)
            abs_idx = tk + offsets                  # (rows, nh, kph) absolute col ids
            return abs_idx.reshape(self.rows, nh * kph)  # (rows, k)

        else:
            # score: (rows, cols) -> (nh, head_dim, cols)
            head_dim = self.rows // nh
            s = score.view(nh, head_dim, self.cols)
            tk = torch.topk(s, kph, dim=1).indices  # (nh, kph, cols) in [0, head_dim)
            offsets = (torch.arange(nh, device=score.device) * head_dim).view(nh, 1, 1)
            abs_idx = tk + offsets                  # (nh, kph, cols) absolute row ids
            return abs_idx.reshape(nh * kph, self.cols)  # (k, cols)

    def topk_mask(self, k: Optional[int] = None) -> torch.Tensor:
        """
        Compute the shared top-k EDGE mask from the aggregate score.
        """
        k = int(k) if k is not None else int(self.k_current)
        if k < 1 or k > self.k_full:
            raise ValueError(f"k must be in [1, {self.k_full}], got {k}")

        score = self._aggregate_score()
        idx = self._head_aware_topk_indices(score, k)

        mask = torch.zeros_like(score, dtype=torch.bool)
        if self.dim == 1:
            mask.scatter_(1, idx, True)
        else:
            mask.scatter_(0, idx, True)
        return mask

    def forward(self, return_stacked: bool = False) -> torch.Tensor | List[torch.Tensor]:
        """
        Returns masked dense weights.

        If return_stacked=False:
            returns list[Tensor] with len = num_params, each shape (rows, cols)
        If return_stacked=True:
            returns Tensor shape (num_params, rows, cols)
        """
        if self.mask_frozen:
            m = self.mask.to(self.weights[0].dtype)
        else:
            m = self.topk_mask(k=self.k_current).to(self.weights[0].dtype)

        outs = [w * m for w in self.weights]
        return torch.stack(outs, dim=0) if return_stacked else outs

    def masked_weight(self, idx: int) -> torch.Tensor:
        return self(return_stacked=False)[idx]

    # ---------- schedule / stability ----------

    @torch.no_grad()
    def step_mask(self) -> float:
        if self.mask_frozen or self.schedule_done:
            return 0.0

        new_mask = self.topk_mask(k=self.k_next)
        num_changes = torch.logical_xor(self.last_mask, new_mask).sum().item()
        self.last_num_changes = int(num_changes)

        packed = self.rows if self.dim == 1 else self.cols
        num_to_compress = packed * (int(self.k_current) - int(self.k_next))
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
    def rewind(self) -> dict:
        chosen_mask = self.topk_mask(k=self.k_next) if self.stage_completed else self.topk_mask(k=self.k_current)
        self.mask.copy_(chosen_mask)

        m = chosen_mask.to(dtype=self.weights[0].dtype, device=self.weights[0].device)
        for i, w in enumerate(self.weights):
            init_cpu = getattr(self, f"init_{i}")
            init = init_cpu.to(device=w.device, dtype=w.dtype)
            w.data.copy_(init * m)

        if self.stage_completed:
            if self.stage_idx < (self.num_stages - 1):
                self.stage_idx += 1
            self.k_current = int(self.k_schedule[self.stage_idx])
            self.k_next = (
                int(self.k_schedule[self.stage_idx + 1])
                if (self.stage_idx + 1) < self.num_stages
                else int(self.k_target)
            )

        self.last_mask.copy_(self.topk_mask(k=self.k_next))
        self._recent_changes = deque(maxlen=self.stability_window)

        self.stage_completed = False
        self.schedule_done = (self.stage_idx == (self.num_stages - 1))
        return self.get_stats()

    @torch.no_grad()
    def freeze_mask(self):
        self.mask.copy_(self.topk_mask(k=self.k_current))
        self.mask_frozen = True

    # ---------- COO export (packed, fixed-k) ----------

    @torch.no_grad()
    def shared_mask_to_packed_indices(self, k: Optional[int] = None) -> torch.Tensor:
        """
        Build packed COO indices for the CURRENT topk selection.

        If sparsity="row": returns ROW-PACKED indices (2, rows*k), packed_dim="row"
        If sparsity="col": returns COL-PACKED indices (2, cols*k), packed_dim="col"
        """
        k = int(k) if k is not None else int(self.k_current)
        if k < 1 or k > self.k_full:
            raise ValueError(f"k must be in [1, {self.k_full}], got {k}")

        score = self._aggregate_score()
        idx = self._head_aware_topk_indices(score, k)

        if self.dim == 1:
            # idx: (rows, k) absolute column ids
            col_idx_2d = idx
            row_idx_2d = torch.arange(self.rows, device=score.device).view(-1, 1).expand(self.rows, k)
            row_idx = row_idx_2d.reshape(-1)
            col_idx = col_idx_2d.reshape(-1)
            return torch.stack([row_idx, col_idx], dim=0)

        else:
            # idx: (k, cols) absolute row ids
            row_idx_2d = idx
            col_idx_2d = torch.arange(self.cols, device=score.device).view(1, -1).expand(k, self.cols)

            # COL-PACKED: col 0 block, then col 1 block, ...
            row_idx = row_idx_2d.t().reshape(-1)   # (cols*k,)
            col_idx = col_idx_2d.t().reshape(-1)   # (cols*k,)
            return torch.stack([row_idx, col_idx], dim=0)

    @torch.no_grad()
    def maskedparam_to_coo(self) -> List[CooSparseParam]:
        """
        Export each weight as a CooSparseParam that shares identical packed indices,
        but has its own trainable values.
        """
        k = int(self.k_current)
        indices = self.shared_mask_to_packed_indices(k=k)

        row = indices[0]
        col = indices[1]
        rows, cols = self.rows, self.cols

        out: List[CooSparseParam] = []
        for w in self.weights:
            values = w.detach()[row, col].clone()
            out.append(
                CooSparseParam(
                    indices=indices,
                    values=values,
                    shape=(rows, cols),
                    packed_dim=self.packed_dim,
                )
            )
        return out

    # ---- keep init on cpu -----
    def _apply(self, fn):
        super_ret = super()._apply(fn)
        for name, buf in self.named_buffers(recurse=False):
            if name.startswith("init_") and buf is not None:
                self._buffers[name] = buf.detach().cpu()
        return super_ret

    # ---------- stats ----------

    def get_stats(self) -> dict:
        packed = self.rows if self.dim == 1 else self.cols
        stats = {
            "num_params": len(self.weights),
            "agg": self.agg,
            "num_heads": self.num_heads,
            "stage_idx": self.stage_idx,
            "num_stages": self.num_stages,
            "k_schedule": str(dict(enumerate(self.k_schedule))),
            "k_target": int(self.k_target),
            "k_full": int(self.k_full),
            "k_current": int(self.k_current),
            "k_next": int(self.k_next),
            "packed_dim": self.packed_dim,
            "current_params_per_weight": int(packed * int(self.k_current)),
            "target_params_per_weight": int(packed * int(self.k_target)),
            "last_change": float(self.last_change),
            "change_threshold": float(self.change_threshold),
            "schedule_done": bool(self.schedule_done),
            "last_num_changes": self.last_num_changes,
            "mask_frozen": bool(self.mask_frozen),
        }
        return stats

    def extra_repr(self) -> str:
        return "\n".join(
            ["SharedMaskedParam with config:"]
            + [f"  {k}: {v}" for k, v in self.get_stats().items()]
        )
