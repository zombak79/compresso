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


def test_srptensor_prefix_shape_restore():
    cols = torch.tensor([[0, 2], [1, 3], [0, 1], [2, 3]], dtype=torch.long)
    vals = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    srp = SRPTensor(cols=cols, vals=vals, shape=(4, 4), prefix_shape=(2, 2))
    dense = srp.to_dense()
    assert dense.shape == (2, 2, 4)


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
