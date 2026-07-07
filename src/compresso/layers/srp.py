import torch
import torch.nn as nn
from typing import Optional, Any
from compresso.params.srp import SRPParam, SRPTensor

class SRPEmbedding(nn.Module):
    """
    Embedding backed by a row-packed sparse param (SRPParam).

    SRPParam represents W of shape (V, E) with exactly k non-zeros per row:
      - cols   : (V, k) long (buffer)
      - values : (V, k) float (Parameter)

    APIs:
      - forward(...)       -> dense embeddings (*prefix, E)  (nn.Embedding-like)
      - forward_srp(...)   -> SRPTensor with shape (N, E) + prefix_shape
      - forward_coo(...)   -> sparse COO (*prefix, E) (optional interop/debug)
    """

    def __init__(self, sparam: Any, padding_idx: Optional[int] = None):
        super().__init__()
        print(type(sparam))
        assert isinstance(sparam, SRPParam)
        self.sparam = sparam
        
        self.padding_idx = padding_idx

        V, E = sparam.shape  # expects SRPParam.shape == (rows, cols_total)
        self.num_embeddings = int(V)
        self.embedding_dim = int(E)

    def extra_repr(self) -> str:
        s = f"num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}"
        if self.padding_idx is not None:
            s += f", padding_idx={self.padding_idx}"
        return s

    @property
    def weight(self):
        """Expose SRPParam for introspection."""
        return self.sparam()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Dense embedding lookup.
        input_ids: LongTensor of shape (...,)
        returns: dense Tensor of shape (*input_ids.shape, embedding_dim)
        """
        if input_ids.dtype != torch.long:
            input_ids = input_ids.long()

        device = input_ids.device
        ids = input_ids
        flat = ids.reshape(-1)  # (N,)
        N = flat.numel()
        E = self.embedding_dim

        if N == 0:
            return torch.empty(*ids.shape, E, device=device, dtype=self.sparam.values.dtype)

        # SRP structure
        cols2d = self.sparam.cols#.to(device=device, non_blocking=True)  # (V,k) buffer
        vals2d = self.sparam.values                                     # (V,k) parameter (already on device)

        tok_cols = cols2d.index_select(0, flat)  # (N,k)
        tok_vals = vals2d.index_select(0, flat)  # (N,k)

        # padding -> zero row
        if self.padding_idx is not None:
            pad = int(self.padding_idx)
            pad_mask = (flat == pad)
            if pad_mask.any():
                tok_vals = tok_vals.clone()
                tok_vals[pad_mask] = 0

        # scatter k values into dense output row
        out = torch.zeros((N, E), device=device, dtype=tok_vals.dtype)
        out.scatter_add_(dim=1, index=tok_cols, src=tok_vals)

        return out.view(*ids.shape, E)

    def forward_srp(self, input_ids: torch.Tensor) -> SRPTensor:
        """
        SRP embedding lookup: returns SRPTensor representing a (N, E) row-packed sparse matrix,
        plus prefix_shape stored inside SRPTensor so you can reshape later.
        """
        if input_ids.dtype != torch.long:
            input_ids = input_ids.long()

        device = input_ids.device
        ids = input_ids
        prefix_shape = tuple(ids.shape)
        flat = ids.reshape(-1)  # (N,)
        N = flat.numel()
        E = self.embedding_dim

        # SRP structure
        cols2d = self.sparam.cols#.to(device=device, non_blocking=True)  # (V,k)
        vals2d = self.sparam.values                                     # (V,k)

        # infer k from param
        _, k = cols2d.shape

        if N == 0:
            cols = torch.empty((0, k), device=device, dtype=torch.long)
            vals = torch.empty((0, k), device=device, dtype=vals2d.dtype)
            return SRPTensor(cols=cols, vals=vals, shape=(0, E), prefix_shape=prefix_shape)

        tok_cols = cols2d.index_select(0, flat)  # (N,k)
        tok_vals = vals2d.index_select(0, flat)  # (N,k)

        if self.padding_idx is not None:
            pad = int(self.padding_idx)
            pad_mask = (flat == pad)
            if pad_mask.any():
                tok_vals = tok_vals.clone()
                tok_vals[pad_mask] = 0

        return SRPTensor(cols=tok_cols, vals=tok_vals, shape=(N, E), prefix_shape=prefix_shape)

    def forward_coo(self, input_ids: torch.Tensor, *, coalesce: bool = True) -> torch.Tensor:
        """
        Optional interop/debug: build sparse COO of shape (*input_ids.shape, E).
        This is slower than SRP and usually not what you want for the fast path.
        """
        if input_ids.dtype != torch.long:
            input_ids = input_ids.long()

        device = input_ids.device
        ids = input_ids
        prefix_shape = ids.shape
        flat = ids.reshape(-1)  # (N,)
        N = flat.numel()
        E = self.embedding_dim

        cols2d = self.sparam.cols#.to(device=device, non_blocking=True)  # (V,k)
        vals2d = self.sparam.values                                     # (V,k)
        _, k = cols2d.shape

        if N == 0:
            indices = torch.empty((len(prefix_shape) + 1, 0), device=device, dtype=torch.long)
            values = torch.empty((0,), device=device, dtype=vals2d.dtype)
            return torch.sparse_coo_tensor(indices, values, size=(*prefix_shape, E), device=device)

        tok_cols = cols2d.index_select(0, flat)  # (N,k)
        tok_vals = vals2d.index_select(0, flat)  # (N,k)

        if self.padding_idx is not None:
            pad = int(self.padding_idx)
            pad_mask = (flat == pad)
            if pad_mask.any():
                tok_vals = tok_vals.clone()
                tok_vals[pad_mask] = 0

        nnz = N * k
        token_pos = torch.arange(N, device=device, dtype=torch.long).repeat_interleave(k)  # (N*k,)
        col_idx = tok_cols.reshape(-1)                                                    # (N*k,)
        values = tok_vals.reshape(-1)                                                     # (N*k,)

        # Convert token_pos -> multi-d prefix coords
        prefix_sizes = torch.tensor(prefix_shape, device=device, dtype=torch.long)
        nd = prefix_sizes.numel()

        if nd == 1:
            coord_prefix = token_pos.unsqueeze(0)  # (1, nnz)
        else:
            strides = prefix_sizes.clone()
            strides[-1] = 1
            for i in range(nd - 2, -1, -1):
                strides[i] = strides[i + 1] * prefix_sizes[i + 1]

            coords = []
            for j in range(nd):
                coords.append((token_pos // strides[j]) % prefix_sizes[j])
            coord_prefix = torch.stack(coords, dim=0)  # (nd, nnz)

        indices = torch.cat([coord_prefix, col_idx.unsqueeze(0)], dim=0)  # (nd+1, nnz)
        out = torch.sparse_coo_tensor(indices, values, size=(*prefix_shape, E), device=device)
        return out.coalesce() if coalesce else out