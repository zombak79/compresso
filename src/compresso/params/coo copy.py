import torch
import torch.nn as nn


class CooSparseParam(nn.Module):
    """
    Sparse weight in COO format with fixed indices and trainable values.

    - indices: LongTensor of shape (2, nnz), [row_idx; col_idx]
    - values:  Parameter of shape (nnz,)
    - shape:   (rows, cols) = (out_features, in_features)
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

    @property
    def shape(self) -> tuple[int, int]:
        return (self.rows, self.cols)

    @property
    def weight(self) -> tuple[int, int]:
        return self.forward()

    def build_coo(self, device=None, dtype=None) -> torch.Tensor:
        """
        Build a sparse_coo_tensor with current values and fixed indices.

        Called in every forward during training. Cheap: no copy of values.
        """
        if device is None:
            device = self.values.device
        if dtype is None:
            dtype = self.values.dtype

        i = self.indices.to(device=device)           # (2, nnz)
        v = self.values.to(device=device, dtype=dtype)  # (nnz,)

        # No need to coalesce here if indices are unique & sorted
        W = torch.sparse_coo_tensor(i, v, size=(self.rows, self.cols), device=device)
        return W
    
    def forward(self):
        return self.build_coo()