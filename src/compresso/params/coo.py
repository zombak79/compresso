import torch
import torch.nn as nn
from typing import Optional, Literal


PackedDim = Literal["row", "col"]


class CooSparseParam(nn.Module):
    """
    Sparse weight in COO format with fixed indices and trainable values.

    Fixed-k packed layout assumption (choose one):
      - packed_dim="row": fixed nnz_per_row = k, row-packed
          row r occupies positions [r*k : (r+1)*k)
      - packed_dim="col": fixed nnz_per_col = k, col-packed
          col c occupies positions [c*k : (c+1)*k)

    This enables fast selection WITHOUT slicing a COO tensor:
      - packed_dim="row": forward(row_indices=...) is O(R*k)
      - packed_dim="col": forward(col_indices=...) is O(C*k)

    Notes:
      - Indices must be consistent with the chosen packing.
      - For best performance, keep indices/values on the same device.
    """

    def __init__(
        self,
        indices: torch.Tensor,
        values: torch.Tensor,
        shape: tuple[int, int],
        *,
        packed_dim: PackedDim = "row",
        validate_packing: bool = True,
    ):
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
        self.packed_dim: PackedDim = packed_dim

        # Fixed index structure (buffer)
        self.register_buffer("indices", indices.clone())
        # Trainable values
        self.values = nn.Parameter(values.clone())

        # --- infer k and validate packing ---
        if packed_dim == "row":
            packed_size = self.rows
            pack_axis = 0  # indices[0] are rows
        elif packed_dim == "col":
            packed_size = self.cols
            pack_axis = 1  # indices[1] are cols
        else:
            raise ValueError(f"packed_dim must be 'row' or 'col', got {packed_dim}")

        if nnz % packed_size != 0:
            raise ValueError(
                f"nnz={nnz} is not divisible by packed_size={packed_size} "
                f"for packed_dim='{packed_dim}'. Cannot infer constant k."
            )
        self.k = nnz // packed_size  # fixed nnz per packed unit

        if validate_packing:
            packed_idx = self.indices[pack_axis]
            expected = torch.repeat_interleave(
                torch.arange(packed_size, device=packed_idx.device, dtype=packed_idx.dtype),
                self.k,
            )
            if packed_idx.numel() != expected.numel() or not torch.equal(packed_idx, expected):
                raise ValueError(
                    f"indices are not {packed_dim}-packed in fixed-k layout. "
                    f"Expected {packed_dim}s in positions [{packed_dim}*k:({packed_dim}+1)*k)."
                )

    @property
    def shape(self) -> tuple[int, int]:
        return (self.rows, self.cols)

    def build_coo(
        self,
        *,
        row_indices: Optional[torch.Tensor] = None,
        col_indices: Optional[torch.Tensor] = None,
        device=None,
        dtype=None,
        is_coalesced: bool = True,
    ) -> torch.Tensor:
        """
        Build a sparse_coo_tensor with current values and fixed indices.

        Selection rules:
          - If no indices given: returns full matrix (rows, cols).
          - If packed_dim="row": you may pass row_indices for fast selection.
          - If packed_dim="col": you may pass col_indices for fast selection.

        Output for selection:
          - Selected dimension is reindexed to [0..R-1] or [0..C-1].
          - The other dimension stays original size.

        For a truly "submatrix" (rows AND cols), you can do it in two steps or add
        extra filtering (not O(k) anymore unless you store more structure).
        """
        if device is None:
            device = self.values.device
        if dtype is None:
            dtype = self.values.dtype

        # Put buffers/params on same device
        idx_full = self.indices.to(device=device, non_blocking=True) if self.indices.device != device else self.indices
        v_full = self.values.to(device=device, dtype=dtype)

        if row_indices is None and col_indices is None:
            return torch.sparse_coo_tensor(
                idx_full, v_full, size=(self.rows, self.cols), device=device, is_coalesced=is_coalesced
            )

        # Enforce selection matches packing
        if self.packed_dim == "row":
            if row_indices is None:
                raise ValueError("packed_dim='row' requires row_indices for fast selection")
            if col_indices is not None:
                raise ValueError("Provide only row_indices for packed_dim='row' fast path")
            return self._select_rows_rowpacked(row_indices, idx_full, v_full, device, is_coalesced)

        else:  # packed_dim == "col"
            if col_indices is None:
                raise ValueError("packed_dim='col' requires col_indices for fast selection")
            if row_indices is not None:
                raise ValueError("Provide only col_indices for packed_dim='col' fast path")
            return self._select_cols_colpacked(col_indices, idx_full, v_full, device, is_coalesced)

    def _select_rows_rowpacked(self, row_indices, idx_full, v_full, device, is_coalesced):
        if row_indices.dtype != torch.long:
            row_indices = row_indices.long()
        row_indices = row_indices.to(device=device, non_blocking=True)

        if row_indices.dim() != 1:
            raise ValueError(f"row_indices must be 1D, got {tuple(row_indices.shape)}")
        R = row_indices.numel()
        if R == 0:
            empty_i = torch.empty((2, 0), device=device, dtype=torch.long)
            empty_v = torch.empty((0,), device=device, dtype=v_full.dtype)
            return torch.sparse_coo_tensor(empty_i, empty_v, size=(0, self.cols), device=device, is_coalesced=True)

        if row_indices.min().item() < 0 or row_indices.max().item() >= self.rows:
            raise ValueError(f"row_indices out of bounds for rows={self.rows}")

        k = self.k
        base = row_indices * k
        offsets = torch.arange(k, device=device, dtype=torch.long)
        pos = (base[:, None] + offsets[None, :]).reshape(-1)

        col = idx_full[1].index_select(0, pos)
        val = v_full.index_select(0, pos)
        new_row = torch.repeat_interleave(torch.arange(R, device=device, dtype=torch.long), k)
        new_idx = torch.stack([new_row, col], dim=0)

        return torch.sparse_coo_tensor(
            new_idx, val, size=(R, self.cols), device=device, is_coalesced=is_coalesced
        )

    def _select_cols_colpacked(self, col_indices, idx_full, v_full, device, is_coalesced):
        if col_indices.dtype != torch.long:
            col_indices = col_indices.long()
        col_indices = col_indices.to(device=device, non_blocking=True)

        if col_indices.dim() != 1:
            raise ValueError(f"col_indices must be 1D, got {tuple(col_indices.shape)}")
        C = col_indices.numel()
        if C == 0:
            empty_i = torch.empty((2, 0), device=device, dtype=torch.long)
            empty_v = torch.empty((0,), device=device, dtype=v_full.dtype)
            return torch.sparse_coo_tensor(empty_i, empty_v, size=(self.rows, 0), device=device, is_coalesced=True)

        if col_indices.min().item() < 0 or col_indices.max().item() >= self.cols:
            raise ValueError(f"col_indices out of bounds for cols={self.cols}")

        k = self.k
        base = col_indices * k
        offsets = torch.arange(k, device=device, dtype=torch.long)
        pos = (base[:, None] + offsets[None, :]).reshape(-1)

        row = idx_full[0].index_select(0, pos)
        val = v_full.index_select(0, pos)
        new_col = torch.repeat_interleave(torch.arange(C, device=device, dtype=torch.long), k)
        new_idx = torch.stack([row, new_col], dim=0)

        return torch.sparse_coo_tensor(
            new_idx, val, size=(self.rows, C), device=device, is_coalesced=is_coalesced
        )

    def forward(
        self,
        row_indices: Optional[torch.Tensor] = None,
        col_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.build_coo(row_indices=row_indices, col_indices=col_indices)