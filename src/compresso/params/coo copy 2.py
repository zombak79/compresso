import torch
import torch.nn as nn
from typing import Optional


class CooSparseParam(nn.Module):
    """
    Sparse weight in COO format with fixed indices and trainable values.

    Fixed-k per row layout assumption:
      - nnz_per_row = k is constant
      - entries are stored row-packed:
          row r occupies positions [r*k : (r+1)*k)
      - indices[0, :] contains row indices consistent with that packing
      - indices[1, :] contains column indices for each entry

    If you violate this assumption, forward(row_indices=...) is undefined.
    """

    def __init__(self, indices: torch.Tensor, values: torch.Tensor, shape: tuple[int, int]):
        super().__init__()

        if indices.dtype != torch.long:
            raise ValueError("indices must be int64 (torch.long)")
        if indices.dim() != 2 or indices.size(0) != 2:
            raise ValueError(f"indices must be shape (2, nnz), got {tuple(indices.shape)}")
        if values.dim() != 1:
            raise ValueError(f"values must be 1D, got shape {tuple(values.shape)}")

        nnz = indices.size(1)
        if values.size(0) != nnz:
            raise ValueError(f"indices nnz={nnz} but values.size(0)={values.size(0)}")

        self.rows, self.cols = shape

        # Fixed index structure (buffer, not trainable)
        self.register_buffer("indices", indices.clone())

        # Trainable values
        self.values = nn.Parameter(values.clone())

        # --- validate fixed-k row-packed layout and cache k ---
        row = self.indices[0]
        if nnz % self.rows != 0:
            raise ValueError(
                f"nnz={nnz} is not divisible by rows={self.rows}. "
                "Cannot infer constant nnz_per_row (k)."
            )
        self.nnz_per_row = nnz // self.rows

        # Optional sanity: check row packing pattern once (cheap enough at init)
        # Expect: row == repeat_interleave(arange(rows), k)
        expected = torch.repeat_interleave(
            torch.arange(self.rows, device=row.device, dtype=row.dtype),
            self.nnz_per_row
        )
        if row.numel() != expected.numel() or not torch.equal(row, expected):
            raise ValueError(
                "indices are not row-packed in fixed-k layout. "
                "Expected rows in positions [r*k:(r+1)*k)."
            )

    @property
    def shape(self) -> tuple[int, int]:
        return (self.rows, self.cols)

    def build_coo(
        self,
        *,
        row_indices: Optional[torch.Tensor] = None,
        device=None,
        dtype=None,
        is_coalesced: bool = True,
    ) -> torch.Tensor:
        """
        Build a sparse_coo_tensor with current values and fixed indices.

        Args:
            row_indices:
                Optional LongTensor of shape (R,) containing row ids to select.
                Returns a *new* COO tensor of shape (R, cols), with rows reindexed
                to [0..R-1] in the output.
                If None, returns full matrix of shape (rows, cols).
            device, dtype:
                Output device/dtype. Defaults to values' device/dtype.
            is_coalesced:
                If True, we declare the COO coalesced. This is safe if you
                guarantee no duplicate indices for a given row and row selection.
        """
        if device is None:
            device = self.values.device
        if dtype is None:
            dtype = self.values.dtype

        # Ensure buffers/params are on the same device for performance
        if self.indices.device != device:
            idx_full = self.indices.to(device=device, non_blocking=True)
        else:
            idx_full = self.indices
        v_full = self.values.to(device=device, dtype=dtype)

        if row_indices is None:
            # Full matrix
            return torch.sparse_coo_tensor(
                idx_full, v_full, size=(self.rows, self.cols), device=device, is_coalesced=is_coalesced
            )

        # --- row subset path (fast, no COO slicing) ---
        if row_indices.dtype != torch.long:
            row_indices = row_indices.long()
        row_indices = row_indices.to(device=device, non_blocking=True)

        # Shape checks
        if row_indices.dim() != 1:
            raise ValueError(f"row_indices must be 1D, got shape {tuple(row_indices.shape)}")
        R = row_indices.numel()
        if R == 0:
            # Empty selection: return empty sparse tensor
            empty_i = torch.empty((2, 0), device=device, dtype=torch.long)
            empty_v = torch.empty((0,), device=device, dtype=v_full.dtype)
            return torch.sparse_coo_tensor(empty_i, empty_v, size=(0, self.cols), device=device, is_coalesced=True)

        if row_indices.min().item() < 0 or row_indices.max().item() >= self.rows:
            raise ValueError(f"row_indices out of bounds for rows={self.rows}")

        k = self.nnz_per_row
        # base positions for each selected row: r*k
        base = row_indices * k  # (R,)

        # offsets 0..k-1
        offsets = torch.arange(k, device=device, dtype=torch.long)  # (k,)

        # positions in the flattened nnz arrays: (R,k) -> (R*k,)
        pos = (base[:, None] + offsets[None, :]).reshape(-1)  # (R*k,)

        # Gather cols and values
        col = idx_full[1].index_select(0, pos)        # (R*k,)
        val = v_full.index_select(0, pos)             # (R*k,)

        # Build new row indices 0..R-1 repeated k times
        new_row = torch.repeat_interleave(
            torch.arange(R, device=device, dtype=torch.long), k
        )  # (R*k,)

        new_idx = torch.stack([new_row, col], dim=0)  # (2, R*k)

        return torch.sparse_coo_tensor(
            new_idx, val, size=(R, self.cols), device=device, is_coalesced=is_coalesced
        )

    def forward(self, row_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        If row_indices is None:
            returns sparse COO of shape (rows, cols)
        Else:
            returns sparse COO of shape (len(row_indices), cols),
            with rows reindexed to 0..R-1 in output.
        """
        return self.build_coo(row_indices=row_indices)