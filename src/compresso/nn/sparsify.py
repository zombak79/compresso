"""Drop-in activation layer for top-k sparsification."""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from compresso.functional.sparsify import topk_ste

__all__ = ["TopKSparsify"]


class TopKSparsify(nn.Module):
    """Activation layer that applies hard top-k sparsification with STE.

    Parameters
    ----------
    k : int
        Number of entries to keep along *dim*.
    dim : int
        Dimension along which to select (default ``-1``).
    score_mode : str
        Scoring mode for top-k ranking (``abs``, ``raw``, ``relu``).
    ste_alpha : float
        Backward gradient scale for non-selected positions.
    """
    # idea: decay ste_alpha to zero during training to transition from dense to sparse gradients
    def __init__(
        self,
        k: int,
        dim: int = -1,
        score_mode: str = "abs",
        ste_alpha: float = 0.0,
    ) -> None:
        super().__init__()
        self.k = k
        self.dim = dim
        self.score_mode = score_mode
        self.ste_alpha = ste_alpha

    def forward(self, x: Tensor) -> Tensor:
        """Apply top-k sparsification to an input tensor.

        The forward pass keeps the ``k`` highest-scoring entries along
        ``dim`` and sets all other entries to zero. Ranking is controlled by
        ``score_mode`` and gradients for non-selected entries are scaled by
        ``ste_alpha`` through the straight-through estimator.

        Parameters
        ----------
        x : Tensor
            Input tensor to sparsify.

        Returns
        -------
        Tensor
            Tensor with the same shape as ``x`` and at most ``k`` non-zero
            entries along ``dim``.
        """
        return topk_ste(
            x,
            k=self.k,
            dim=self.dim,
            score_mode=self.score_mode,
            ste_alpha=self.ste_alpha,
        )

    def set_k(self, k: int) -> None:
        """Change *k* at runtime (e.g. for scheduling)."""
        self.k = k

    def extra_repr(self) -> str:
        return f"k={self.k}, dim={self.dim}, score_mode={self.score_mode!r}, ste_alpha={self.ste_alpha}"
