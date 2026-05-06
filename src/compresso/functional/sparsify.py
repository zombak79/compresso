"""Functional top-k sparsification with straight-through estimation."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

__all__ = ["topk_ste"]

_VALID_MODES = {"values", "mask", "values_ste_identity", "values_ste_selected"}


def topk_ste(
    x: Tensor,
    k: int,
    dim: int = -1,
    mode: str = "values",
    k_backward: Optional[int] = None,
) -> Tensor:
    """Hard top-k sparsification with straight-through estimation.

    Parameters
    ----------
    x : Tensor
        Input tensor of arbitrary shape.
    k : int
        Number of entries to keep along *dim* in the **forward** pass.
    dim : int
        Dimension along which to select top-k (default ``-1``).
    mode : str
        One of:

        * ``"values"`` / ``"values_ste_identity"`` –
          Forward returns signed top-k values (zeros elsewhere).
          Backward passes gradients to **all** positions (identity STE).
        * ``"values_ste_selected"`` –
          Forward same as above.
          Backward passes gradients **only** through selected positions.
        * ``"mask"`` –
          Returns binary ``{0, 1}`` mask. No gradient.
    k_backward : int | None
        If set, overrides the backward STE width regardless of *mode*.
        Gradients flow through the top-``k_backward`` positions (by absolute
        value), while the forward output still contains only *k* entries.

        * ``k_backward == k`` behaves like ``values_ste_selected``.
        * ``k_backward >= x.size(dim)`` behaves like ``values_ste_identity``.
        * ``k < k_backward < x.size(dim)`` is the useful middle-ground.

        Ignored when ``mode="mask"``.

    Returns
    -------
    Tensor
        Same shape as *x*.
    """
    if mode not in _VALID_MODES:
        raise ValueError(
            f"Unknown mode {mode!r}. Expected one of {sorted(_VALID_MODES)}."
        )

    # --- forward top-k selection ---
    idx = torch.topk(x.abs(), k, dim=dim).indices
    mask = torch.zeros_like(x).scatter(dim, idx, 1.0)

    if mode == "mask":
        return mask.detach()

    masked = x * mask  # forward value: exactly k entries

    # --- k_backward overrides STE width ---
    if k_backward is not None:
        if k_backward >= x.size(dim):
            # full identity STE
            return x + (masked - x).detach()
        # wider (or equal) backward mask
        back_idx = torch.topk(x.abs(), k_backward, dim=dim).indices
        back_mask = torch.zeros_like(x).scatter(dim, back_idx, 1.0)
        back_masked = x * back_mask
        # forward = masked, backward flows through back_masked
        return back_masked + (masked - back_masked).detach()

    # --- original mode logic (k_backward is None) ---
    if mode in ("values", "values_ste_identity"):
        return x + (masked - x).detach()

    # values_ste_selected: grad only through selected entries
    return masked
