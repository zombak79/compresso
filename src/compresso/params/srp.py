import torch
import torch.nn as nn
from typing import Optional, Literal, Tuple

InitMode = Literal["topk_abs", "random_k"]

from dataclasses import dataclass
import torch

class SRPTensor:
    """
    Minimal SRP tensor container:
      - row-packed sparse matrix with fixed k nnz per row
      - represented by (cols, vals) of shape (rows, k)
      - logical dense shape is (rows, cols_total)
    Optionally carries prefix_shape so we can restore (*prefix, cols_total).
    """
    __slots__ = ("cols", "vals", "shape", "prefix_shape")

    def __init__(
        self,
        *,
        cols: torch.Tensor,          # (rows, k) long
        vals: torch.Tensor,          # (rows, k) float
        shape: Tuple[int, int],      # (rows, cols_total)
        prefix_shape: Optional[Tuple[int, ...]] = None,
    ):
        if cols.dtype != torch.long:
            raise ValueError("SRPTensor.cols must be torch.long")
        if cols.dim() != 2 or vals.dim() != 2:
            raise ValueError("SRPTensor.cols and SRPTensor.vals must be 2D (rows, k)")
        if cols.shape != vals.shape:
            raise ValueError(f"cols.shape {tuple(cols.shape)} != vals.shape {tuple(vals.shape)}")
        rows, k = cols.shape
        if shape[0] != rows:
            raise ValueError(f"shape[0]={shape[0]} must equal rows={rows}")
        self.cols = cols
        self.vals = vals
        self.shape = (int(shape[0]), int(shape[1]))
        self.prefix_shape = tuple(prefix_shape) if prefix_shape is not None else None

    @property
    def device(self):
        return self.vals.device

    @property
    def dtype(self):
        return self.vals.dtype

    @property
    def rows(self) -> int:
        return self.shape[0]

    @property
    def cols_total(self) -> int:
        return self.shape[1]

    @property
    def k(self) -> int:
        return int(self.cols.size(1))

    def to_dense(self) -> torch.Tensor:
        """Densify to (rows, cols_total) or (*prefix, cols_total) if prefix_shape is set."""
        #print(self.shape)
        #print(self.cols, self.cols.device)
        #print(self.vals, self.vals.device)
        out = torch.zeros(self.shape, device=self.vals.device, dtype=self.vals.dtype).scatter(1, self.cols, self.vals)
        return out
        
        #rows, cols_total = self.shape
        #out = torch.zeros((rows, cols_total), device=self.device, dtype=self.dtype)
        #out.scatter_add_(dim=1, index=self.cols, src=self.vals)
        #if self.prefix_shape is not None:
        #    out = out.view(*self.prefix_shape, cols_total)
        #return out

class SRPParam(nn.Module):
    """
    Structured Row-Packed sparse parameter: a matrix A of shape (rows, cols_total)
    stored as fixed-k nonzeros per row.

    Storage:
      - cols:   (rows, k) int64 buffer, fixed indices per row
      - values: (rows, k) trainable parameter

    Semantics (important!):
      A[r, cols[r, j]] += values[r, j]   # duplicates within a row accumulate (scatter_add semantics)

    This format is ideal for SRPMM kernels like:
      Y = A @ B  where B is dense (cols_total, out)

    because you can do:
      B_sel = B[cols] -> (rows, k, out)
      Y = sum_j values[...,j] * B_sel[...,j,:]
    """

    def __init__(
        self,
        cols: torch.Tensor,           # (rows, k) long
        values: torch.Tensor,         # (rows, k) float/bf16/fp16/fp32
        shape: Tuple[int, int],       # (rows, cols_total)
        *,
        validate: bool = True,
    ):
        super().__init__()
        if cols.dtype != torch.long:
            raise ValueError("cols must be torch.long")
        if cols.dim() != 2:
            raise ValueError(f"cols must be 2D (rows,k), got {tuple(cols.shape)}")
        if values.dim() != 2:
            raise ValueError(f"values must be 2D (rows,k), got {tuple(values.shape)}")

        rows, cols_total = int(shape[0]), int(shape[1])
        if cols.size(0) != rows or values.size(0) != rows:
            raise ValueError(
                f"cols/values first dim must equal rows={rows}, got cols={cols.size(0)}, values={values.size(0)}"
            )
        if cols.shape != values.shape:
            raise ValueError(f"cols and values must have same shape, got {tuple(cols.shape)} vs {tuple(values.shape)}")

        k = cols.size(1)
        if k <= 0:
            raise ValueError("k must be >= 1")

        self.rows = rows
        self.cols_total = cols_total
        self.k = int(k)

        if validate:
            if cols.numel() > 0:
                cmin = int(cols.min().item())
                cmax = int(cols.max().item())
                if cmin < 0 or cmax >= cols_total:
                    raise ValueError(f"cols out of bounds: min={cmin}, max={cmax}, allowed [0, {cols_total-1}]")

        # structure is fixed
        self.register_buffer("cols", cols.clone())
        # values are trained
        self.values = nn.Parameter(values.clone())

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.rows, self.cols_total)

    def extra_repr(self) -> str:
        return f"shape=({self.rows},{self.cols_total}), k={self.k}, dtype={self.values.dtype}"

    #@torch.no_grad()
    #def to_dense(self, *, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    #    """
    #    Materialize dense matrix A (rows, cols_total) with scatter_add semantics.
    #    Mainly for debugging/tests.
    #    """
    #    device = self.values.device
    #    dt = dtype if dtype is not None else self.values.dtype
    #
    #    A = torch.zeros(self.rows, self.cols_total, device=device, dtype=dt)
    #    A.scatter_add_(1, self.cols.to(device=device), self.values.to(dtype=dt))
    #    return A
    
    def to_dense(self) -> torch.Tensor:
        rows, cols_total = self.shape
        out = torch.zeros((rows, cols_total), device=self.device, dtype=self.dtype)
        out = out.scatter_add(1, self.cols, self.vals)   # out-of-place version keeps autograd happy
        if self.prefix_shape is not None:
            out = out.view(*self.prefix_shape, cols_total)
        return out

    #@torch.no_grad()
    def build_coo(self, *, dtype: Optional[torch.dtype] = None, coalesce: bool = True) -> torch.Tensor:
        """
        Convert to a torch sparse COO tensor (rows, cols_total).
        Duplicates are kept; coalesce() will sum them.
        """
        device = self.values.device
        dt = dtype if dtype is not None else self.values.dtype

        r = torch.arange(self.rows, device=device, dtype=torch.long).unsqueeze(1).expand(self.rows, self.k)
        idx = torch.stack([r.reshape(-1), self.cols.to(device=device).reshape(-1)], dim=0)  # (2, rows*k)
        val = self.values.to(dtype=dt).reshape(-1)

        sp = torch.sparse_coo_tensor(idx, val, size=(self.rows, self.cols_total), device=device)
        return sp.coalesce() if coalesce else sp

    #@torch.no_grad()
    def select_rows(self, row_indices: torch.Tensor) -> "SRPParam":
        """
        Fast row selection, returns a NEW SRPParam with rows=len(row_indices), same cols_total and k.
        Reindexes rows to [0..R-1], structure is still row-packed.
        """
        if row_indices.dtype != torch.long:
            row_indices = row_indices.long()
        row_indices = row_indices.to(self.cols.device)

        cols_sel = self.cols.index_select(0, row_indices)
        vals_sel = self.values.detach().index_select(0, row_indices)
        return SRPParam(cols_sel, vals_sel, shape=(cols_sel.size(0), self.cols_total), validate=False)

    # --------------------------
    # constructors / converters
    # --------------------------

    @staticmethod
    @torch.no_grad()
    def from_dense(
        A: torch.Tensor,                 # (rows, cols_total)
        k: int,
        *,
        mode: InitMode = "topk_abs",
        allow_duplicates: bool = False,  # if False, enforced by construction
    ) -> "SRPParam":
        """
        Build SRPParam from a dense matrix.

        mode:
          - "topk_abs": pick top-k |A[r,:]| per row, store signed values.
          - "random_k": pick k random cols per row, store those values.

        Note: this is a *projection* of dense -> fixed-k sparse.
        """
        if A.dim() != 2:
            raise ValueError(f"A must be 2D, got {tuple(A.shape)}")
        rows, cols_total = A.shape
        rows = int(rows); cols_total = int(cols_total)
        k = int(k)
        if not (1 <= k <= cols_total):
            raise ValueError(f"k must be in [1, cols_total], got k={k}, cols_total={cols_total}")

        device = A.device
        dt = A.dtype

        if mode == "topk_abs":
            idx = torch.topk(A.abs(), k=k, dim=1, largest=True).indices  # (rows,k)
        elif mode == "random_k":
            # sample without replacement per row
            idx = torch.stack([torch.randperm(cols_total, device=device)[:k] for _ in range(rows)], dim=0)
        else:
            raise ValueError(f"Unknown mode={mode}")

        if not allow_duplicates:
            # topk and randperm are already unique; keep this for future modes
            pass

        vals = A.gather(1, idx).to(dtype=dt)
        cols = idx.to(dtype=torch.long)

        return SRPParam(cols=cols, values=vals, shape=(rows, cols_total), validate=True)

    @staticmethod
    @torch.no_grad()
    def from_sparse_coo(
        sp: torch.Tensor,          # sparse_coo (rows, cols_total)
        *,
        k: Optional[int] = None,
        require_row_packed_fixed_k: bool = True,
    ) -> "SRPParam":
        """
        Convert from a COO sparse tensor to SRPParam.

        Cases:
          - If require_row_packed_fixed_k=True:
              expects exactly k nnz per row (same k for all rows), and we will pack rows.
              If k is None, we infer it. If not possible -> error.
          - If require_row_packed_fixed_k=False:
              we will *project* each row to top-k by abs value (needs k provided).
        """
        if not sp.is_sparse:
            raise ValueError("sp must be a sparse COO tensor")
        if sp.layout != torch.sparse_coo:
            raise ValueError(f"sp must be sparse_coo, got {sp.layout}")
        sp = sp.coalesce()

        rows, cols_total = sp.shape
        rows = int(rows); cols_total = int(cols_total)

        idx = sp.indices()   # (2, nnz)
        val = sp.values()    # (nnz,)
        r = idx[0]
        c = idx[1]

        # count nnz per row
        counts = torch.bincount(r, minlength=rows)  # (rows,)

        if require_row_packed_fixed_k:
            if k is None:
                # infer k if all rows have same count
                unique = torch.unique(counts)
                if unique.numel() != 1:
                    raise ValueError(f"Cannot infer fixed k: nnz per row vary: {unique.tolist()[:10]}")
                k = int(unique.item())
            else:
                k = int(k)

            if not torch.all(counts == k):
                raise ValueError(f"Not fixed-k: expected {k} nnz per row, got min={int(counts.min())}, max={int(counts.max())}")

            # pack into (rows,k) by sorting by row then taking chunks
            # ensure stable order: sort by (row, col)
            order = torch.argsort(r * cols_total + c)
            r2 = r[order]
            c2 = c[order]
            v2 = val[order]

            # now rows are grouped; reshape should work because each row has exactly k entries
            cols = c2.view(rows, k).to(torch.long)
            values = v2.view(rows, k)

            return SRPParam(cols=cols, values=values, shape=(rows, cols_total), validate=True)

        else:
            # projection: need k
            if k is None:
                raise ValueError("k must be provided when require_row_packed_fixed_k=False")
            k = int(k)

            # build per-row dense-ish topk using scatter into padded lists (not super fast; fine for conversion)
            # We'll do: for each row, take its entries and topk by abs.
            cols_out = torch.empty((rows, k), device=sp.device, dtype=torch.long)
            vals_out = torch.empty((rows, k), device=sp.device, dtype=val.dtype)

            # build row-wise lists via sorting by row
            order = torch.argsort(r)
            r_sorted = r[order]
            c_sorted = c[order]
            v_sorted = val[order]

            # pointers per row
            ptr = torch.zeros(rows + 1, device=sp.device, dtype=torch.long)
            ptr[1:] = torch.cumsum(counts, dim=0)

            for row in range(rows):
                start = int(ptr[row].item())
                end = int(ptr[row + 1].item())
                if start == end:
                    # empty row: fill with zeros (columns arbitrary)
                    cols_out[row].zero_()
                    vals_out[row].zero_()
                    continue
                c_row = c_sorted[start:end]
                v_row = v_sorted[start:end]
                kk = min(k, c_row.numel())
                top = torch.topk(v_row.abs(), k=kk, largest=True).indices
                cols_sel = c_row[top]
                vals_sel = v_row[top]
                # pad if needed
                if kk < k:
                    pad = k - kk
                    cols_sel = torch.cat([cols_sel, cols_sel.new_zeros(pad)], dim=0)
                    vals_sel = torch.cat([vals_sel, vals_sel.new_zeros(pad)], dim=0)
                cols_out[row] = cols_sel
                vals_out[row] = vals_sel

            return SRPParam(cols=cols_out, values=vals_out, shape=(rows, cols_total), validate=True)

    # convenience forward
    def forward(self) -> SRPTensor:
        return SRPTensor(
            cols=self.cols,          # (rows, k) long
            vals=self.values,            # (rows, k) float
            shape=(self.rows, self.cols_total),
        )