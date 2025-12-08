import torch
import torch.nn as nn
from collections import deque
from typing import Optional, Sequence

from compresso.params.coo import CooSparseParam
from compresso.utils.schedule import exponential_decay

class MaskedParam(nn.Module):
    """
    Dense parameter with a binary mask and an internal pruning schedule.

    Responsibilities (per weight matrix W ∈ R^{rows×cols}):
      - Store dense trainable weights.
      - Maintain a binary mask of the same shape.
      - Keep original initialization on CPU for masked rewind.
      - Own a schedule of per-row k values: k_schedule = [k0, k1, ..., kT].
      - On each schedule tick, recompute the mask via row-wise top-k.
      - Track how many mask entries changed and whether the current stage is stable.
      - Optionally advance to the next k in the schedule when stable.
      - Optionally export the final masked weights as COO (indices, values).
    """

    def __init__(
        self,
        weight: torch.Tensor,
        k_target: int,
        k_schedule: Optional[Sequence[int]] = None,
        num_stages: int = 10,
        stability_window: int = 5,
        change_threshold: float = 0.01,
    ):
        """
        Args:
            weight:
                Initial dense weight tensor of shape (rows, cols).
            k_target:
                Final per-row k (number of non-zero entries to keep per row)
                when the pruning schedule is complete. Ignored if k_schedule
                is provided explicitly.
            k_schedule:
                Optional explicit list/sequence of k values (per-row nnz)
                for the pruning stages. If provided, it must be non-increasing.
                Example: [4096, 2048, 1024, 512].
                If None, a simple linear schedule from full dense (cols) to
                k_target over num_stages is constructed.
            num_stages:
                Number of pruning stages to construct automatically if
                k_schedule is None. Must be >= 1.
            stability_window:
                Number of recent mask-update steps to keep in history for
                stability checking. For example, with 5 and change_threshold=0,
                the stage is considered stable once 5 consecutive updates
                produce zero mask changes.
            change_threshold:
                Maximum allowed portion of changed entries per update for it
                to be counted as "stable". 
        """
        super().__init__()
        if weight.dim() != 2:
            raise ValueError(f"MaskedParam expects 2D weight, got shape {tuple(weight.shape)}")

        self.rows, self.cols = weight.shape
        self.k_target = int(k_target)
        self.k_full = self.cols
        self.num_stages = num_stages
        # Save original init
        self.initialization = weight.detach().cpu().clone()
        
        # Trainable dense weights
        self.weight = nn.Parameter(self.initialization)

        if k_schedule is not None:
            # Make sure all values in schedule are integers
            ks = [int(k) for k in k_schedule]
            # Basic sanity: non-increasing and last == k_target
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
            # Build a exponential schedule from full density to k_target
            self.k_schedule = exponential_decay(self.k_full, self.k_target, self.num_stages-1)
            self.k_schedule.append(self.k_target)

        self.num_stages = len(self.k_schedule)
        
        self.stage_idx = 0                  # current stage index in k_schedule
        self.k_current = self.k_full        # effective k in last update; starts dense
        self.k_next = self.k_schedule[self.stage_idx+1] if (self.stage_idx+1)<self.num_stages else self.k_target
        self.mask = torch.ones(self.weight.shape, dtype=self.weight.dtype).to(self.weight.device)
        self.last_mask = self.topk_mask(k=self.k_next).to(self.weight.device)
        
        
        # Stability tracking
        self.stability_window = int(stability_window)
        self.change_threshold = change_threshold
        self._recent_changes = deque(maxlen=self.stability_window)
        self.last_change = 1.
        self.last_num_changes = None
        self.stage_completed = False     # whether current stage finished
        self.schedule_done = False       # whether all stages completed
        self.mask_frozen = False         # if True, no further updates        

    def topk_weights_with_mask(self, k=None, return_mask=True):
        k = k if k else self.k_current
        w_topk = torch.topk(torch.abs(self.weight), k, -1)
        out = torch.zeros_like(self.weight).scatter(-1, w_topk.indices, w_topk.values) * torch.sign(self.weight)
        return (out, out.bool()) if return_mask else out

    def topk_weights(self, k=None):
        return self.topk_weights_with_mask(k=k, return_mask=False)

    def forward(self):
        if self.mask_frozen:
            return self.mask*self.weight
        else:
            return self.topk_weights()
    
    def topk_mask(self, k=None):
        return self.topk_weights_with_mask(k=k, return_mask=True)[1]

    @torch.no_grad()
    def step_mask(self):
        """Compute mask and add history"""
        if self.mask_frozen or self.schedule_done:
            return 0
        mask = self.topk_mask(k=self.k_next).to(self.weight.device)
        num_changes = torch.logical_xor(self.last_mask.to(self.weight.device), mask).sum()
        self.last_num_changes = num_changes
        num_to_compress = self.rows*(self.k_current)-self.rows*(self.k_next)
        unstable_fraction = num_changes/num_to_compress
        self._recent_changes.append(unstable_fraction)
        if len(self._recent_changes)>=self.stability_window:
            if sum(self._recent_changes)/len(self._recent_changes)<=self.change_threshold:
                self.stage_completed=True
        self.last_mask=mask
        self.last_change = sum(self._recent_changes)/len(self._recent_changes)
        return self.last_change

    def rewind(self):
        """Back to original init with new k"""
        mask = self.topk_mask(k=self.k_next) if self.stage_completed else self.topk_mask()
        self.mask=mask
        new_initialization = self.initialization.to(self.weight.device)*mask
        #self.weight = nn.Parameter(new_initialization.to(self.weight.device))
        self.weight.data.copy_(new_initialization.to(self.weight.device, self.weight.dtype))
        if self.stage_completed:
            self.stage_idx += 1 if self.stage_idx < (self.num_stages - 1) else 0
            self.k_current = self.k_schedule[self.stage_idx]
            self.k_next = self.k_schedule[self.stage_idx+1] if (self.stage_idx+1)<self.num_stages else self.k_target
        self.last_mask = self.topk_mask(k=self.k_next)
        self._recent_changes = deque(maxlen=self.stability_window)
        self.stage_completed = False     # reset current stage
        # whether all stages completed
        self.schedule_done = True if self.stage_idx == (self.num_stages - 1) else False               
        return self.get_stats()

    def freeze_mask(self):
        self.mask = self.topk_mask()
        self.mask_frozen = True         # if True, no further updates

    @torch.no_grad()
    def maskedparam_to_coo(self):
        """
        Convert a finalized MaskedParam (with frozen or stable mask) into CooSparseParam.
        """
        # Final pruned dense weight, shape (out_features, in_features)
        if getattr(self, "mask_frozen", False) and hasattr(self, "mask"):
            mask = self.mask
        else:
            mask = self.topk_mask(k=self.k_current)
    
        W_pruned = (self.weight * mask.to(self.weight.dtype)).detach()  # (rows, cols)
    
        # Build COO. coalesce() to get unique, sorted indices
        coo = W_pruned.to_sparse_coo().coalesce()
        indices = coo.indices()  # (2, nnz)
        values = coo.values()    # (nnz,)
    
        rows, cols = W_pruned.shape
    
        return CooSparseParam(indices=indices, values=values, shape=(rows, cols))
        
        
    def get_stats(self):
        stats = {
            "stage_idx": self.stage_idx,
            "num_stages": self.num_stages,
            "k_schedule": str(dict(enumerate(self.k_schedule))),
            "k_target": self.k_target,
            "k_current": self.k_current,
            "current_params": self.rows*self.k_current,
            "target_params": self.rows*self.k_target,
            "stage_idx": self.stage_idx,
            "last_change": self.last_change,
            "change_threshold": self.change_threshold,
            "schedule_done": self.schedule_done,
            "last_num_changes":self.last_num_changes,
        }
        return stats
    
    def extra_repr(self):
        return "\n".join(["MaskedParam with config:"]+[f"  {k}: {v}" for k,v in self.get_stats().items()])
