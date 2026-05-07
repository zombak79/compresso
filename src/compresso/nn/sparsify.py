"""Drop-in activation layer for top-k sparsification."""

from __future__ import annotations

from typing import Optional

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
    mode : str
        STE mode forwarded to :func:`~compresso.functional.sparsify.topk_ste`.
    k_backward : int | None
        If set, backward STE uses top-``k_backward`` positions instead of
        the default determined by *mode*.
    """

    def __init__(
        self,
        k: int,
        dim: int = -1,
        mode: str = "values",
        score_mode: str = "abs",
        k_backward: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.k = k
        self.dim = dim
        self.mode = mode
        self.score_mode = score_mode
        self.k_backward = k_backward

    def forward(self, x: Tensor) -> Tensor:
        return topk_ste(
            x,
            k=self.k,
            dim=self.dim,
            mode=self.mode,
            score_mode=self.score_mode,
            k_backward=self.k_backward,
        )

    def set_k(self, k: int) -> None:
        """Change *k* at runtime (e.g. for scheduling)."""
        self.k = k

    def extra_repr(self) -> str:
        parts = f"k={self.k}, dim={self.dim}, mode={self.mode!r}, score_mode={self.score_mode!r}"
        if self.k_backward is not None:
            parts += f", k_backward={self.k_backward}"
        return parts
