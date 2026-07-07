import torch

def srpmm_rowpacked_core(cols2: torch.Tensor, vals2: torch.Tensor, B: torch.Tensor, *, accum_dtype=torch.float32):
    rows, k = cols2.shape
    out_features = B.size(1)

    B_sel = B.index_select(0, cols2.reshape(-1)).view(rows, k, out_features)
    Y = (vals2.to(accum_dtype).unsqueeze(-1) * B_sel.to(accum_dtype)).sum(dim=1)
    return Y



def srpmm(cols, vals, shape, B, *, accum_dtype=torch.float32, out_dtype=None):
    rows, in_features = shape
    if B.dim() != 2 or B.size(0) != in_features:
        raise ValueError(f"B must be (in_features, out_features) with in_features={in_features}, got {tuple(B.shape)}")

    # normalize cols -> (rows, k)
    if cols.dim() == 1:
        if cols.numel() % rows != 0:
            raise ValueError(f"flat cols nnz={cols.numel()} not divisible by rows={rows}")
        k = cols.numel() // rows
        cols2 = cols.view(rows, k)
    elif cols.dim() == 2:
        if cols.size(0) != rows:
            raise ValueError(f"cols first dim must be rows={rows}, got {cols.size(0)}")
        cols2 = cols
    else:
        raise ValueError(f"cols must be 1D or 2D, got {cols.dim()}D")

    # normalize vals -> (rows, k)
    if vals.shape != cols2.shape:
        if vals.dim() == 1 and vals.numel() == cols2.numel():
            vals2 = vals.view_as(cols2)
        else:
            raise ValueError(f"vals shape {tuple(vals.shape)} must match cols shape {tuple(cols2.shape)}")
    else:
        vals2 = vals

    if cols2.dtype != torch.long:
        cols2 = cols2.long()

    # bounds check
    if cols2.numel() > 0:
        cmin = int(cols2.min().item())
        cmax = int(cols2.max().item())
        if cmin < 0 or cmax >= in_features:
            raise ValueError(f"cols out of bounds: min={cmin}, max={cmax}, in_features={in_features}")

    # actual op (keeps autograd for vals2 + B)
    Y = srpmm_rowpacked_core(cols2, vals2, B, accum_dtype=accum_dtype)

    # choose output dtype
    if out_dtype is None:
        out_dtype = vals2.dtype
    return Y.to(out_dtype)