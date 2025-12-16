import torch
import torch.nn as nn
from collections import deque
from typing import Optional, Sequence, Literal

from compresso.params.coo import CooSparseParam
from compresso.utils.schedule import exponential_decay


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
    ):
        super().__init__()
        if weight.dim() != 2:
            raise ValueError(f"MaskedParam expects 2D weight, got shape {tuple(weight.shape)}")

        self.rows, self.cols = weight.shape
        self.k_target = int(k_target)
        self.k_full = self.cols

        # set dim
        if sparsity == "row":
            self.dim = 1
        elif sparsity == "col":
            self.dim = 0
        else:
            raise ValueError(f"Sparsity could be row or col, got {sparsity}")

        # --- Save original init on CPU as a buffer (checkpoint-safe) ---
        # stays on CPU by design; we move it on-demand in rewind()
        self.register_buffer("initialization", weight.detach().cpu().clone(), persistent=True)

        # --- Trainable dense weights (keep device of input weight) ---
        self.weight = nn.Parameter(weight.detach().clone())

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

        # These are Python scalars. If you want perfect resume under exotic wrappers (FSDP),
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
        # topk over dim (row-wise => dim=1 keeps k per row; col-wise => dim=0 keeps k per col)
        w_abs = torch.abs(self.weight)
        w_topk = torch.topk(w_abs, k, self.dim)
        out = torch.zeros_like(self.weight).scatter(self.dim, w_topk.indices, w_topk.values) * torch.sign(self.weight)
        return (out, out.bool()) if return_mask else out

    def topk_weights(self, k=None):
        return self.topk_weights_with_mask(k=k, return_mask=False)

    def topk_mask(self, k=None):
        return self.topk_weights_with_mask(k=k, return_mask=True)[1]

    def forward(self):
        if self.mask_frozen:
            return self.weight * self.mask.to(self.weight.dtype)
        else:
            return self.topk_weights()

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
        num_to_compress = self.rows * (int(self.k_current) - int(self.k_next))
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
        Convert a finalized MaskedParam (with frozen or stable mask) into CooSparseParam.
        """
        if self.mask_frozen:
            mask = self.mask
        else:
            mask = self.topk_mask(k=self.k_current)

        W_pruned = (self.weight * mask.to(self.weight.dtype)).detach()
        coo = W_pruned.to_sparse_coo().coalesce()
        indices = coo.indices()
        values = coo.values()
        rows, cols = W_pruned.shape
        return CooSparseParam(indices=indices, values=values, shape=(rows, cols))

    def get_stats(self):
        stats = {
            "stage_idx": self.stage_idx,
            "num_stages": self.num_stages,
            "k_schedule": str(dict(enumerate(self.k_schedule))),
            "k_target": self.k_target,
            "k_current": int(self.k_current),
            "current_params": self.rows * int(self.k_current),
            "target_params": self.rows * int(self.k_target),
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