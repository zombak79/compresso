from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias

import torch


RowIndex: TypeAlias = slice | torch.Tensor | Sequence[int]

_INTEGER_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}


def normalize_row_indices(
    index: RowIndex,
    *,
    rows: int,
    device: torch.device,
) -> torch.Tensor:
    """Normalize a non-scalar row index to a 1D long tensor."""
    if isinstance(index, slice):
        start, stop, step = index.indices(rows)
        length = len(range(start, stop, step))
        if length == 0:
            return torch.empty(0, device=device, dtype=torch.long)
        return torch.arange(
            start,
            stop,
            step,
            device=device,
            dtype=torch.long,
        )
    if isinstance(index, int):
        raise IndexError(
            "scalar row indexing is not supported; use param[[row]] "
            "to preserve the row-packed 2D shape"
        )
    if isinstance(index, torch.Tensor):
        if index.dim() != 1:
            raise IndexError(
                f"row indices must be 1D, got shape {tuple(index.shape)}"
            )
        if index.dtype not in _INTEGER_DTYPES:
            raise TypeError(
                f"row indices must have an integer dtype, got {index.dtype}"
            )
        indices = index.to(device=device, dtype=torch.long)
    elif isinstance(index, Sequence) and not isinstance(index, (str, bytes)):
        if len(index) == 0:
            return torch.empty(0, device=device, dtype=torch.long)
        inferred = torch.as_tensor(index, device=device)
        if inferred.dim() != 1:
            raise IndexError(
                f"row indices must be 1D, got shape {tuple(inferred.shape)}"
            )
        if inferred.dtype not in _INTEGER_DTYPES:
            raise TypeError(
                f"row indices must contain integers, got {inferred.dtype}"
            )
        indices = inferred.to(dtype=torch.long)
    else:
        raise TypeError(
            "row index must be a slice, a 1D integer tensor, or an "
            "integer sequence"
        )

    if indices.numel() == 0:
        return indices
    # Leave bounds enforcement to index_select so CUDA indexing does not incur
    # a host synchronization here. Values below -rows remain negative and are
    # rejected by index_select.
    return torch.where(indices < 0, indices + rows, indices)
