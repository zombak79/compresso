from __future__ import annotations

import torch

from compresso.io import load_srp_tensor, save_srp_tensor
from compresso.params.srp import SRPTensor


def test_srptensor_to_dense_scatter_add_duplicate_columns():
    cols = torch.tensor([[1, 1, 3]], dtype=torch.long)
    vals = torch.tensor([[2.0, 5.0, -1.0]])
    srp = SRPTensor(cols=cols, vals=vals, shape=(1, 5))
    dense = srp.to_dense()
    expected = torch.tensor([[0.0, 7.0, 0.0, -1.0, 0.0]])
    assert torch.allclose(dense, expected)


def test_srptensor_torch_sparse_conversions_match_dense():
    cols = torch.tensor([[1, 1, 3], [0, 2, 2]], dtype=torch.long)
    vals = torch.tensor([[2.0, 5.0, -1.0], [4.0, 1.5, 2.5]])
    srp = SRPTensor(cols=cols, vals=vals, shape=(2, 5))
    dense = srp.to_dense()

    coo = srp.to_coo()
    csr = srp.to_csr()
    csc = srp.to_csc()
    bsr = srp.to_bsr((1, 1))
    bsc = srp.to_bsc((1, 1))

    assert coo.layout == torch.sparse_coo
    assert csr.layout == torch.sparse_csr
    assert csc.layout == torch.sparse_csc
    assert bsr.layout == torch.sparse_bsr
    assert bsc.layout == torch.sparse_bsc
    assert coo.is_coalesced()
    assert torch.allclose(coo.to_dense(), dense)
    assert torch.allclose(csr.to_dense(), dense)
    assert torch.allclose(csc.to_dense(), dense)
    assert torch.allclose(bsr.to_dense(), dense)
    assert torch.allclose(bsc.to_dense(), dense)


def test_srptensor_scipy_sparse_conversions_match_dense():
    cols = torch.tensor([[1, 1, 3], [0, 2, 2]], dtype=torch.long)
    vals = torch.tensor([[2.0, 5.0, -1.0], [4.0, 1.5, 2.5]])
    srp = SRPTensor(cols=cols, vals=vals, shape=(2, 5))
    expected = srp.to_dense().numpy()

    coo = srp.to_scipy_coo()
    csr = srp.to_scipy_csr()
    csc = srp.to_scipy_csc()

    assert coo.shape == srp.shape
    assert csr.shape == srp.shape
    assert csc.shape == srp.shape
    assert torch.allclose(torch.from_numpy(coo.toarray()), torch.from_numpy(expected))
    assert torch.allclose(torch.from_numpy(csr.toarray()), torch.from_numpy(expected))
    assert torch.allclose(torch.from_numpy(csc.toarray()), torch.from_numpy(expected))


def test_srptensor_numpy_returns_structural_arrays():
    cols = torch.tensor([[0, 2], [1, 3]], dtype=torch.long)
    vals = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    srp = SRPTensor(cols=cols, vals=vals, shape=(2, 5), prefix_shape=(2,))

    payload = srp.numpy()

    assert payload["cols"].shape == (2, 2)
    assert payload["vals"].shape == (2, 2)
    assert payload["shape"] == (2, 5)
    assert payload["prefix_shape"] == (2,)


def test_srptensor_prefix_shape_restore():
    cols = torch.tensor([[0, 2], [1, 3], [0, 1], [2, 3]], dtype=torch.long)
    vals = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    srp = SRPTensor(cols=cols, vals=vals, shape=(4, 4), prefix_shape=(2, 2))
    dense = srp.to_dense()
    assert dense.shape == (2, 2, 4)


def test_srptensor_repr_contains_readable_layout_summary():
    cols = torch.tensor([[0, 2], [1, 3]], dtype=torch.long)
    vals = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    srp = SRPTensor(cols=cols, vals=vals, shape=(2, 5), prefix_shape=(2,))

    text = repr(srp)

    assert "SRPTensor(" in text
    assert "shape=(2, 5)" in text
    assert "k=2" in text
    assert "vals=Tensor(shape=(2, 2)" in text
    assert "cols=Tensor(shape=(2, 2)" in text
    assert "prefix_shape=(2,)" in text


def test_srptensor_tensor_like_metadata_and_shape_helpers():
    cols = torch.tensor([[0, 2], [1, 3]], dtype=torch.long)
    vals = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    srp = SRPTensor(cols=cols, vals=vals, shape=(2, 5), prefix_shape=(2,))

    assert srp.device == vals.device
    assert srp.dtype == torch.float32
    assert srp.requires_grad is False
    assert srp.is_cuda is False
    assert srp.nnz == 4
    assert srp.numel() == 10
    assert srp.size() == torch.Size([2, 5])
    assert srp.size(0) == 2
    assert srp.size(-1) == 5
    assert srp.dim() == 2
    assert srp.ndim == 2
    assert srp.is_floating_point()


def test_srptensor_to_clone_detach_and_requires_grad():
    cols = torch.tensor([[0, 2], [1, 3]], dtype=torch.long)
    vals = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    srp = SRPTensor(cols=cols, vals=vals, shape=(2, 5))

    converted = srp.to(dtype=torch.float64)
    assert converted.vals.dtype == torch.float64
    assert converted.cols.dtype == torch.long
    assert converted.device == srp.device

    cloned = srp.clone()
    assert cloned is not srp
    assert cloned.cols.data_ptr() != srp.cols.data_ptr()
    assert cloned.vals.data_ptr() != srp.vals.data_ptr()
    assert torch.equal(cloned.cols, srp.cols)
    assert torch.allclose(cloned.vals, srp.vals)

    detached = srp.detach()
    assert detached.requires_grad is False
    assert detached.vals.grad_fn is None

    srp.requires_grad_(False)
    assert srp.requires_grad is False


def test_srptensor_contiguous_returns_contiguous_storage():
    cols = torch.tensor([[0, 2], [1, 3], [0, 1]], dtype=torch.long).t()
    vals = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).t()
    srp = SRPTensor(cols=cols, vals=vals, shape=(2, 5), validate=False)

    contiguous = srp.contiguous()

    assert contiguous.cols.is_contiguous()
    assert contiguous.vals.is_contiguous()
    assert torch.equal(contiguous.cols, srp.cols)
    assert torch.allclose(contiguous.vals, srp.vals)


def test_srp_io_roundtrip(tmp_path):
    cols = torch.tensor([[0, 2], [1, 3]], dtype=torch.long)
    vals = torch.tensor([[1.5, -2.0], [0.25, 3.0]], dtype=torch.float32)
    srp = SRPTensor(cols=cols, vals=vals, shape=(2, 4), prefix_shape=(2,))

    p = tmp_path / "x.srp.pt"
    save_srp_tensor(p, srp)
    loaded = load_srp_tensor(p)

    assert torch.equal(loaded.cols, srp.cols)
    assert torch.allclose(loaded.vals, srp.vals)
    assert loaded.shape == srp.shape
    assert loaded.prefix_shape == srp.prefix_shape
    assert torch.allclose(loaded.to_dense(), srp.to_dense())


def test_srptensor_from_dense_topk_abs_projection():
    x = torch.tensor([[0.1, -5.0, 3.0, 4.0]], dtype=torch.float32)
    srp = SRPTensor.from_dense(x, k=2, score_mode="abs")
    dense = srp.to_dense()
    # abs top-2 should keep -5 and 4
    assert dense.shape == (1, 4)
    assert dense[0, 1].item() == -5.0
    assert dense[0, 3].item() == 4.0
    assert (dense != 0).sum().item() == 2
