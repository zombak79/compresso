"""Functional top-k sparsification with straight-through estimation."""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["topk_ste"]

_VALID_SCORE_MODES = {"abs", "raw", "relu"}


def _topk_scores(x: Tensor, score_mode: str) -> Tensor:
    if score_mode == "abs":
        return x.abs()
    if score_mode == "raw":
        return x
    if score_mode == "relu":
        return x.relu()
    raise ValueError(
        f"Unknown score_mode {score_mode!r}. Expected one of {sorted(_VALID_SCORE_MODES)}."
    )


def topk_ste(
    x: Tensor,
    k: int,
    dim: int = -1,
    score_mode: str = "abs",
    ste_alpha: float = 0.0,
) -> Tensor:
    """Hard top-k forward with alpha-scaled STE on non-selected entries.

    Forward keeps exactly ``k`` signed values per slice (others are zero).
    Backward uses gradient scale ``1.0`` on selected top-k entries and
    ``ste_alpha`` on all non-selected entries.
    """
    if score_mode not in _VALID_SCORE_MODES:
        raise ValueError(
            f"Unknown score_mode {score_mode!r}. Expected one of {sorted(_VALID_SCORE_MODES)}."
        )
    if not (0.0 <= ste_alpha <= 1.0):
        raise ValueError("ste_alpha must be in [0, 1]")

    scores = _topk_scores(x, score_mode=score_mode)
    idx = torch.topk(scores, k, dim=dim).indices
    mask = torch.zeros_like(x).scatter(dim, idx, 1.0)
    masked = x * mask
    back_soft = (ste_alpha * x) + ((1.0 - ste_alpha) * masked)
    return back_soft + (masked - back_soft).detach()
