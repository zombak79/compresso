import torch
import torch.nn as nn
import torch.nn.functional as F

from compresso.utils.ops import srpmm_rowpacked_core
from compresso.params.srp import SRPTensor, SRPParam

class SparseAwareLinearKernel(nn.Module):
    """
    Linear layer that works with BOTH dense and sparse inputs.

    Input:
      - dense: (..., in_features)
      - sparse: (*prefix, in_features) in COO (dim >= 2)
      - SRP input: SRPTensor representing (*prefix, in_features) but stored as (rows=N, k) with prefix_shape.

    Output:
      - dense: (..., out_features) with the SAME prefix shape.

    Behavior:
      - dense x: F.linear(x, weight, bias)   # standard nn.Linear semantics
      - sparse x:
          1) flatten prefix dims: (*prefix, in) -> (N, in)
          2) sparse.mm((N, in), (in, out)) -> (N, out)
          3) reshape back to (*prefix, out)
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features, **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.bias = None

        # init like nn.Linear
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in = in_features
            bound = 1 / (fan_in**0.5) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

        self.debug = None

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"

    def _sparse_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: sparse_coo with shape (*prefix, in_features), dim >= 2
        Returns:
            dense tensor with shape (*prefix, out_features)
        """
        if not x.is_sparse:
            raise ValueError("x must be sparse in _sparse_forward")
        
        if not x.is_coalesced():
            x = x.coalesce()

        full_shape = x.shape                     # e.g. (B, T, E)
        if len(full_shape) < 2:
            raise ValueError(f"Sparse input must have at least 2 dims, got {full_shape}")

        prefix_shape = full_shape[:-1]           # e.g. (B, T)
        in_features = full_shape[-1]
        if in_features != self.in_features:
            raise ValueError(
                f"Last dim of sparse input ({in_features}) != in_features ({self.in_features})"
            )

        # Total number of rows after flatten:
        # N = prod(prefix_shape)
        device = x.device
        prefix_sizes = torch.tensor(prefix_shape, device=device, dtype=torch.long)  # (k,)
        N = int(prefix_sizes.prod().item())

        indices = x.indices()                # (ndim, nnz)
        values = x.values()                  # (nnz,)
        out_dtype = values.dtype             # what the caller expects back
        values32 = values.to(torch.float32)          
        ndim, nnz = indices.shape

        # prefix coords: shape (k, nnz)
        k = len(prefix_shape)
        coord_prefix = indices[:k, :]        # all dims except feature dim
        col_idx = indices[k, :]              # feature dim index

        # Compute strides for row-major layout:
        # Example: prefix_shape = (B, T)    -> strides = [T, 1]
        # Example: prefix_shape = (D0,D1,D2)-> strides = [D1*D2, D2, 1]
        strides = torch.empty_like(prefix_sizes)
        strides[-1] = 1
        for i in range(k - 2, -1, -1):
            strides[i] = strides[i + 1] * prefix_sizes[i + 1]

        # row_flat = sum_i coord_prefix[i] * strides[i]
        row_flat = (coord_prefix * strides.view(-1, 1)).sum(dim=0)  # (nnz,)

        # Sanity: row_flat must be in [0, N)
        # (you can assert here during debugging if you want)
        # assert row_flat.max().item() < N

        # Build 2D sparse (N, in_features)
        indices_2d = torch.stack([row_flat, col_idx], dim=0)        # (2, nnz)
        
        with torch.cuda.amp.autocast(enabled=False):
            x2d = torch.sparse_coo_tensor(
                indices_2d,
                values32,
                size=(N, in_features),
                device=device,
                dtype=torch.float32,
            ).coalesce()

            # sparse.mm: (N, in) @ (in, out) -> (N, out)
            Wt = self.weight.t().to(device=device, dtype=torch.float32) # (in, out)
            y2d = torch.sparse.mm(x2d, Wt)                              # (N, out), dense, fp32

        if self.bias is not None:
            y2d = y2d + self.bias.to(device=device, dtype=torch.float32)

        # reshape back to (*prefix, out_features)
        out = y2d.reshape(*prefix_shape, self.out_features)
        return out.to(out_dtype)
    
    def _srp_forward(self, x: "SRPTensor") -> torch.Tensor:
        """
        x: SRPTensor representing sparse activations with logical shape (*prefix, in_features)
        but stored row-packed as (rows=N, k).
        Returns:
            dense tensor with shape (*prefix, out_features)
        """
        rows, in_features = x.shape
        if in_features != self.in_features:
            raise ValueError(
                f"SRP cols_total ({in_features}) != in_features ({self.in_features})"
            )

        # If prefix_shape isn't set, we assume it's already flattened (rows, in_features)
        prefix_shape = x.prefix_shape
        out_dtype = x.dtype
        device = x.device

        # weight: (out, in) -> need B: (in, out)
        # IMPORTANT: keep autograd; do NOT wrap in no_grad.
        B = self.weight.t()  # (in_features, out_features)

        # Accumulate in fp32 (typical); return whatever dtype the model uses
        with torch.cuda.amp.autocast(enabled=False):
            Y2d = srpmm_rowpacked_core(x.cols, x.vals, B, accum_dtype=torch.float32)

        if self.bias is not None:
            # bias add in fp32 for stability, then cast
            Y2d = Y2d.to(torch.float32) + self.bias.to(torch.float32)

        Y2d = Y2d.to(out_dtype)

        if prefix_shape is not None:
            return Y2d.view(*prefix_shape, self.out_features)
        else:
            return Y2d.view(rows, self.out_features)

    def forward(self, x) -> torch.Tensor:
        # 1) SRP fast path
        if self.debug != type(x):
            print("type(x)", type(x))
            debug=True
        else:
            debug=False
        self.debug = type(x)

        if isinstance(x, SRPParam):
            if debug:
                print(x.values)
            x=x()
            
        if isinstance(x, SRPTensor):
            return self._srp_forward(x)
            x=x.to_dense()
            if debug:
                print("to_dense")

        # 2) Dense path
        if not getattr(x, "is_sparse", False):
            return F.linear(x, self.weight, self.bias)

        # 3) COO sparse path
        return self._sparse_forward(x)