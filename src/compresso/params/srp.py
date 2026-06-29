import torch
import torch.nn as nn
from typing import Any, Optional, Literal, Tuple

InitMode = Literal["topk_abs", "random_k"]

class SRPTensor:
    """Minimal SRP tensor container.

    Stores a row-packed sparse matrix with fixed ``k`` nonzeros per row.
    The representation uses ``cols`` and ``vals`` tensors of shape
    ``(rows, k)`` with logical dense shape ``(rows, cols_total)``.

    Optionally carries ``prefix_shape`` so callers can restore
    ``(*prefix, cols_total)``.
    """
    __slots__ = ("cols", "vals", "shape", "prefix_shape")

    def __init__(
        self,
        *,
        cols: torch.Tensor,          # (rows, k) long
        vals: torch.Tensor,          # (rows, k) float
        shape: Tuple[int, int],      # (rows, cols_total)
        prefix_shape: Optional[Tuple[int, ...]] = None,
        validate: bool = True,
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
        cols_total = int(shape[1])
        if cols_total <= 0:
            raise ValueError("shape[1] (cols_total) must be >= 1")
        if validate and cols.numel() > 0:
            cmin = int(cols.min().item())
            cmax = int(cols.max().item())
            if cmin < 0 or cmax >= cols_total:
                raise ValueError(f"cols out of bounds: min={cmin}, max={cmax}, allowed [0, {cols_total-1}]")
        if prefix_shape is not None:
            rows_from_prefix = 1
            for d in prefix_shape:
                rows_from_prefix *= int(d)
            if rows_from_prefix != rows:
                raise ValueError(
                    f"prefix_shape={tuple(prefix_shape)} implies rows={rows_from_prefix}, expected {rows}"
                )
        self.cols = cols
        self.vals = vals
        self.shape = (int(shape[0]), cols_total)
        self.prefix_shape = tuple(prefix_shape) if prefix_shape is not None else None

    @property
    def device(self):
        return self.vals.device

    @property
    def dtype(self):
        return self.vals.dtype

    @property
    def requires_grad(self) -> bool:
        return bool(self.vals.requires_grad)

    @property
    def is_cuda(self) -> bool:
        return bool(self.vals.is_cuda)

    @property
    def rows(self) -> int:
        return self.shape[0]

    @property
    def cols_total(self) -> int:
        return self.shape[1]

    @property
    def k(self) -> int:
        return int(self.cols.size(1))

    @property
    def nnz(self) -> int:
        return int(self.cols.numel())

    @property
    def ndim(self) -> int:
        return self.dim()

    def __repr__(self) -> str:
        prefix = f", prefix_shape={self.prefix_shape}" if self.prefix_shape is not None else ""
        return (
            "SRPTensor("
            f"shape={self.shape}, "
            f"k={self.k}, "
            f"vals=Tensor(shape={tuple(self.vals.shape)}, dtype={self.vals.dtype}, device={self.vals.device}), "
            f"cols=Tensor(shape={tuple(self.cols.shape)}, dtype={self.cols.dtype}, device={self.cols.device})"
            f"{prefix}"
            ")"
        )

    def dim(self) -> int:
        return (len(self.prefix_shape) + 1) if self.prefix_shape is not None else 2

    def size(self, dim: Optional[int] = None):
        logical_shape = (*self.prefix_shape, self.cols_total) if self.prefix_shape is not None else self.shape
        if dim is None:
            return torch.Size(logical_shape)
        return torch.Size(logical_shape)[dim]

    def numel(self) -> int:
        out = 1
        for size in self.size():
            out *= int(size)
        return int(out)

    def is_floating_point(self) -> bool:
        return bool(torch.is_floating_point(self.vals))

    def to(self, *args, **kwargs) -> "SRPTensor":
        vals = self.vals.to(*args, **kwargs)
        device = vals.device
        cols = self.cols.to(device=device)
        return SRPTensor(
            cols=cols,
            vals=vals,
            shape=self.shape,
            prefix_shape=self.prefix_shape,
            validate=False,
        )

    def cpu(self) -> "SRPTensor":
        return self.to("cpu")

    def cuda(self, device: Optional[int | str | torch.device] = None) -> "SRPTensor":
        if device is None:
            return self.to("cuda")
        return self.to(torch.device("cuda", device) if isinstance(device, int) else device)

    def detach(self) -> "SRPTensor":
        return SRPTensor(
            cols=self.cols.detach(),
            vals=self.vals.detach(),
            shape=self.shape,
            prefix_shape=self.prefix_shape,
            validate=False,
        )

    def clone(self) -> "SRPTensor":
        return SRPTensor(
            cols=self.cols.clone(),
            vals=self.vals.clone(),
            shape=self.shape,
            prefix_shape=self.prefix_shape,
            validate=False,
        )

    def contiguous(self) -> "SRPTensor":
        return SRPTensor(
            cols=self.cols.contiguous(),
            vals=self.vals.contiguous(),
            shape=self.shape,
            prefix_shape=self.prefix_shape,
            validate=False,
        )

    def requires_grad_(self, requires_grad: bool = True) -> "SRPTensor":
        self.vals.requires_grad_(requires_grad)
        return self

    def to_dense(self) -> torch.Tensor:
        """Densify to ``(rows, cols_total)`` or ``(*prefix, cols_total)``."""
        rows, cols_total = self.shape
        out = torch.zeros((rows, cols_total), device=self.device, dtype=self.dtype)
        # Duplicates in a row accumulate by design.
        out.scatter_add_(dim=1, index=self.cols, src=self.vals)
        if self.prefix_shape is not None:
            out = out.view(*self.prefix_shape, cols_total)
        return out

    def to_coo(self) -> torch.Tensor:
        """Convert to a coalesced 2D PyTorch sparse COO tensor.

        The returned tensor has logical shape ``(rows, cols_total)``. Duplicate
        SRP columns in a row are summed by COO coalescing, matching
        :meth:`to_dense` scatter-add semantics.
        """
        row_idx = torch.arange(self.rows, device=self.cols.device, dtype=torch.long).repeat_interleave(self.k)
        col_idx = self.cols.reshape(-1)
        indices = torch.stack((row_idx, col_idx), dim=0)
        return torch.sparse_coo_tensor(
            indices,
            self.vals.reshape(-1),
            size=self.shape,
            device=self.device,
            dtype=self.dtype,
            check_invariants=False,
        ).coalesce()

    def to_csr(self) -> torch.Tensor:
        """Convert to a 2D PyTorch sparse CSR tensor."""
        return self.to_coo().to_sparse_csr()

    def to_csc(self) -> torch.Tensor:
        """Convert to a 2D PyTorch sparse CSC tensor."""
        return self.to_coo().to_sparse_csc()

    def to_bsr(self, blocksize: tuple[int, int]) -> torch.Tensor:
        """Convert to a 2D PyTorch sparse BSR tensor.

        Parameters
        ----------
        blocksize:
            Sparse block size ``(row_block, col_block)``. Both dimensions must
            divide the logical dense shape.
        """
        return self.to_coo().to_sparse_bsr(blocksize)

    def to_bsc(self, blocksize: tuple[int, int]) -> torch.Tensor:
        """Convert to a 2D PyTorch sparse BSC tensor.

        Parameters
        ----------
        blocksize:
            Sparse block size ``(row_block, col_block)``. Both dimensions must
            divide the logical dense shape.
        """
        return self.to_coo().to_sparse_bsc(blocksize)

    def to_scipy_coo(self):
        """Convert to a SciPy ``coo_matrix`` on CPU.

        This conversion detaches tensors and therefore does not preserve
        autograd history.
        """
        from scipy import sparse

        row_idx = torch.arange(self.rows, device=self.cols.device, dtype=torch.long).repeat_interleave(self.k)
        return sparse.coo_matrix(
            (
                self.vals.detach().cpu().numpy().reshape(-1),
                (row_idx.detach().cpu().numpy(), self.cols.detach().cpu().numpy().reshape(-1)),
            ),
            shape=self.shape,
        )

    def to_scipy_csr(self):
        """Convert to a SciPy ``csr_matrix`` on CPU."""
        return self.to_scipy_coo().tocsr()

    def to_scipy_csc(self):
        """Convert to a SciPy ``csc_matrix`` on CPU."""
        return self.to_scipy_coo().tocsc()

    def to_numpy_dict(self) -> dict[str, Any]:
        """Return a structural NumPy representation.

        The returned dictionary contains ``cols``, ``vals``, ``shape``, and
        ``prefix_shape``. It is intentionally not dense; use
        ``srp.to_dense().cpu().numpy()`` when a dense NumPy array is desired.
        """
        return {
            "cols": self.cols.detach().cpu().numpy(),
            "vals": self.vals.detach().cpu().numpy(),
            "shape": self.shape,
            "prefix_shape": self.prefix_shape,
        }

    def numpy(self) -> dict[str, Any]:
        """Alias for :meth:`to_numpy_dict`.

        ``SRPTensor`` is structurally represented by both ``cols`` and ``vals``,
        so this method returns structural arrays rather than a dense matrix.
        """
        return self.to_numpy_dict()

    def to_dict(self) -> dict[str, Any]:
        """Serialize SRPTensor to a torch-saveable payload."""
        return {
            "version": 1,
            "layout": "srp",
            "shape": self.shape,
            "prefix_shape": self.prefix_shape,
            "cols": self.cols,
            "vals": self.vals,
        }

    @staticmethod
    def from_dict(payload: dict[str, Any], *, validate: bool = True) -> "SRPTensor":
        """Deserialize SRPTensor from payload produced by :meth:`to_dict`."""
        if not isinstance(payload, dict):
            raise ValueError("payload must be a dict")
        if payload.get("layout", "srp") != "srp":
            raise ValueError(f"Unsupported layout: {payload.get('layout')!r}")
        if int(payload.get("version", 1)) != 1:
            raise ValueError(f"Unsupported SRP payload version: {payload.get('version')!r}")
        required = {"cols", "vals", "shape"}
        missing = sorted(required - set(payload.keys()))
        if missing:
            raise ValueError(f"Missing payload keys: {missing}")
        return SRPTensor(
            cols=payload["cols"],
            vals=payload["vals"],
            shape=tuple(payload["shape"]),
            prefix_shape=tuple(payload["prefix_shape"]) if payload.get("prefix_shape") is not None else None,
            validate=validate,
        )

    @staticmethod
    def from_dense(
        x: torch.Tensor,
        k: int,
        *,
        score_mode: Literal["abs", "raw", "relu"] = "abs",
    ) -> "SRPTensor":
        """Project dense tensor to fixed-k SRP row-packed representation.

        Accepts shape ``(*prefix, cols_total)`` and flattens prefix dims into rows.
        Stores signed values gathered from original ``x`` at selected indices.
        """
        if x.dim() < 2:
            raise ValueError(f"x must have at least 2 dims, got {tuple(x.shape)}")
        cols_total = int(x.shape[-1])
        if not (1 <= int(k) <= cols_total):
            raise ValueError(f"k must be in [1, {cols_total}], got {k}")

        prefix_shape = tuple(int(d) for d in x.shape[:-1])
        rows = 1
        for d in prefix_shape:
            rows *= d
        x2d = x.reshape(rows, cols_total)

        if score_mode == "abs":
            scores = x2d.abs()
        elif score_mode == "raw":
            scores = x2d
        elif score_mode == "relu":
            scores = x2d.relu()
        else:
            raise ValueError("score_mode must be one of {'abs', 'raw', 'relu'}")

        idx = torch.topk(scores, k=int(k), dim=-1, largest=True).indices
        vals = x2d.gather(dim=-1, index=idx)
        return SRPTensor(
            cols=idx.to(torch.long),
            vals=vals,
            shape=(rows, cols_total),
            prefix_shape=prefix_shape if len(prefix_shape) > 0 else None,
        )

class SRPParam(nn.Module):
    """Structured row-packed sparse parameter.

    Represents a matrix ``A`` of shape ``(rows, cols_total)`` stored as
    fixed-``k`` nonzeros per row.

    Storage:

    * ``cols``: ``(rows, k)`` int64 buffer with fixed indices per row.
    * ``values``: ``(rows, k)`` trainable parameter.

    Semantics:

    ``A[r, cols[r, j]] += values[r, j]``

    Duplicates within a row accumulate with scatter-add semantics. This format
    is useful for row-packed sparse matrix multiplication against a dense
    matrix ``B`` with shape ``(cols_total, out)``.
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
        out = torch.zeros((rows, cols_total), device=self.values.device, dtype=self.values.dtype)
        out = out.scatter_add(1, self.cols, self.values)
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
        """Build SRPParam from a dense matrix.

        mode:

        * ``"topk_abs"``: pick top-k absolute values per row and store signed
          values.
        * ``"random_k"``: pick k random columns per row and store those values.

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
