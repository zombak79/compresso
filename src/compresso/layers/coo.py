import torch
import torch.nn as nn
from typing import Optional


class CooSparseLinear(nn.Module):
    """
    Linear layer with sparse weight in COO format:

        y = x @ W^T + b

    where:
        - W is stored as sparse COO (out_features, in_features)
        - x can be dense with shape (..., in_features)
        - output y has shape (..., out_features)
    """

    def __init__(self, sparam: "CooSparseParam", bias: bool = True):
        super().__init__()
        self.sparam = sparam
        self.in_features = sparam.cols
        self.out_features = sparam.rows

        if bias is not None:
            self.bias = nn.Parameter(torch.zeros(self.out_features))
        else:
            self.bias = None

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: dense tensor of shape (*prefix, in_features)
        returns: dense tensor of shape (*prefix, out_features)
        """
        if x.is_sparse:
            raise ValueError("CooSparseLinear expects dense input x; got sparse.")

        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"Expected last dim {self.in_features}, got {x.shape[-1]} in CooSparseLinear"
            )

        device = x.device
        dtype = torch.float32  # compute in fp32

        # sparse weight: (out, in)
        W = self.sparam.build_coo(device=device, dtype=dtype)

        # flatten prefix dims
        x32 = x.to(dtype=dtype).reshape(-1, self.in_features)  # (N, in)
        y_flat = torch.sparse.mm(W, x32.t()).t()               # (N, out)

        if self.bias is not None:
            y_flat = y_flat + self.bias.to(device=device, dtype=y_flat.dtype)

        y = y_flat.reshape(*x.shape[:-1], self.out_features)
        return y.to(x.dtype)

class CooSparseEmbedding(nn.Module):
    """
    Sparse embedding backed by a CooSparseParam (num_embeddings x embedding_dim).

    Given input_ids of shape (...,), this builds a sparse COO tensor of shape
    (*input_ids.shape, embedding_dim), such that:

        out[*, i, :] == W[input_ids[*, i], :]

    where W is stored in sparse COO as CooSparseParam.

    Padding behavior:
      - If padding_idx is not None, ALL non-zeros corresponding to that
        vocabulary row (token id) are skipped. So those positions in the
        output embedding are exactly zero rows.
    """

    def __init__(
        self,
        sparam: "CooSparseParam",
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.sparam = sparam
        self.padding_idx = padding_idx

        self.num_embeddings = sparam.rows   # vocab size
        self.embedding_dim = sparam.cols    # embedding dim

    def extra_repr(self) -> str:
        s = f"num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}"
        if self.padding_idx is not None:
            s += f", padding_idx={self.padding_idx}"
        return s

    def _build_batch_sparse(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Vectorized construction of sparse batch embedding.

        Args:
            input_ids: LongTensor of shape (...,)

        Returns:
            sparse_coo_tensor of shape (*input_ids.shape, embedding_dim)
        """
        device = input_ids.device
        input_ids = input_ids.to(torch.long)

        prefix_shape = input_ids.shape           # e.g. (B, T) or (N,)
        if len(prefix_shape) == 0:
            raise ValueError("input_ids must have at least 1 dimension")

        # Flatten ids: (N,)
        flat_ids = input_ids.reshape(-1)
        N = flat_ids.numel()
        E = self.embedding_dim

        # Global COO from CooSparseParam
        global_indices = self.sparam.indices.to(device)   # (2, nnz)
        global_values = self.sparam.values.to(device)     # (nnz,)
        global_rows = global_indices[0]                   # (nnz,) -- vocab ids
        global_cols = global_indices[1]                   # (nnz,)
        nnz = global_values.numel()

        # Degenerate cases
        if nnz == 0 or N == 0:
            indices = torch.zeros((len(prefix_shape) + 1, 0), dtype=torch.long, device=device)
            values = torch.zeros((0,), dtype=global_values.dtype, device=device)
            size = (*prefix_shape, E)
            return torch.sparse_coo_tensor(indices, values, size=size, device=device)

        # 1) Sort flat_ids so we can use searchsorted to map global_rows -> positions in batch
        sorted_ids, sorted_pos = torch.sort(flat_ids)   # (N,), (N,)

        # 2) For each global row r (vocab id), find where r appears among batch token ids
        lo = torch.searchsorted(sorted_ids, global_rows, side="left")   # (nnz,)
        hi = torch.searchsorted(sorted_ids, global_rows, side="right")  # (nnz,)

        # counts[i] = number of times vocab id global_rows[i] appears in flat_ids
        counts = hi - lo                                               # (nnz,)

        # 2a) Skip padding rows: if global_rows[i] == padding_idx, zero out its count
        if self.padding_idx is not None:
            pad = int(self.padding_idx)
            counts = counts * (global_rows != pad)

        # 3) Keep only non-zeros that actually appear in the current batch (and are not padding)
        mask = counts > 0
        if not mask.any():
            # No overlap between batch ids and embedding rows (or all padding)
            indices = torch.zeros((len(prefix_shape) + 1, 0), dtype=torch.long, device=device)
            values = torch.zeros((0,), dtype=global_values.dtype, device=device)
            size = (*prefix_shape, E)
            return torch.sparse_coo_tensor(indices, values, size=size, device=device)

        counts_sel = counts[mask]                               # (M,)
        global_idx_sel = torch.arange(nnz, device=device)[mask] # (M,)

        total_nnz_batch = int(counts_sel.sum().item())
        # Example: if a row appears 3 times, its non-zero entries are replicated 3× in the batch

        # 4) For each selected global non-zero, repeat it counts_sel[i] times
        expanded_global_idx = torch.repeat_interleave(global_idx_sel, counts_sel)  # (total_nnz_batch,)

        # 5) Compute which flattened batch position each expanded non-zero belongs to
        # Build offsets such that counts_sel[i] entries form a contiguous block
        offsets = torch.zeros_like(counts_sel)
        if counts_sel.numel() > 1:
            offsets[1:] = counts_sel.cumsum(0)[:-1]            # e.g. [0, c0, c0+c1, ...]

        expanded_offsets = torch.repeat_interleave(offsets, counts_sel)          # (total_nnz_batch,)
        starts = lo[mask]                                                       # (M,)
        expanded_starts = torch.repeat_interleave(starts, counts_sel)           # (total_nnz_batch,)

        # Positions in sorted_ids where each copy goes:
        #   sorted_positions_for_entries = starts[i] + [0 .. counts_sel[i]-1]  per block
        idx_in_block = torch.arange(total_nnz_batch, device=device) - expanded_offsets
        sorted_positions_for_entries = expanded_starts + idx_in_block          # (total_nnz_batch,)

        # Map back to original flat positions 0..N-1
        flat_positions = sorted_pos[sorted_positions_for_entries]              # (total_nnz_batch,)

        # 6) Columns and values just follow expanded_global_idx
        batch_col_idx = global_cols[expanded_global_idx]                       # (total_nnz_batch,)
        batch_vals = global_values[expanded_global_idx]                        # (total_nnz_batch,)

        # 7) Convert flat_positions into multi-dimensional prefix coordinates
        # prefix_shape: e.g. (B, T)
        prefix_sizes = torch.tensor(prefix_shape, device=device, dtype=torch.long)  # (k,)
        k = prefix_sizes.numel()

        if k == 1:
            # Single prefix dimension: coord = flat_positions
            coord_prefix = flat_positions.unsqueeze(0)   # (1, total_nnz_batch)
        else:
            # Compute strides for row-major layout
            # Example: (B, T, U) -> strides = [T*U, U, 1]
            strides = prefix_sizes.clone()
            strides[-1] = 1
            for i in range(k - 2, -1, -1):
                strides[i] = strides[i + 1] * prefix_sizes[i + 1]

            # coord_j = (flat_positions // strides[j]) % size[j]
            coord_list = []
            for j in range(k):
                coord_j = (flat_positions // strides[j]) % prefix_sizes[j]
                coord_list.append(coord_j)
            coord_prefix = torch.stack(coord_list, dim=0)  # (k, total_nnz_batch)

        # 8) Stack prefix coords with embedding dim index to form final COO indices
        # Final shape: (*prefix_shape, embedding_dim)
        indices = torch.cat(
            [coord_prefix, batch_col_idx.unsqueeze(0)],
            dim=0,  # (k+1, total_nnz_batch)
        )

        size = (*prefix_shape, E)
        E_batch = torch.sparse_coo_tensor(
            indices,
            batch_vals,
            size=size,
            device=device,
        ).coalesce()

        return E_batch

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        input: LongTensor of shape (...,)

        Returns:
            Sparse COO tensor of shape (*input.shape, embedding_dim).
        """
        return self._build_batch_sparse(input)