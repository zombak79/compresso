"""Tests for compresso.nn.sparsify.TopKSparsify."""

import io
import pytest
import torch

from compresso.functional import topk_ste
from compresso.nn import TopKSparsify


class TestTopKSparsify:
    def test_module_matches_functional(self, dtype, device):
        x = torch.randn(4, 10, dtype=dtype, device=device)
        layer = TopKSparsify(k=3, dim=-1, mode="values")
        expected = topk_ste(x, k=3, dim=-1, mode="values")
        assert torch.equal(layer(x), expected)

    def test_module_matches_functional_selected_mode(self, dtype, device):
        x = torch.randn(4, 10, dtype=dtype, device=device, requires_grad=True)
        layer = TopKSparsify(k=3, dim=-1, mode="values_ste_selected")
        expected = topk_ste(x, k=3, dim=-1, mode="values_ste_selected")
        assert torch.equal(layer(x), expected)

    def test_state_dict_roundtrip(self):
        layer = TopKSparsify(k=5, dim=-1, mode="values")
        buf = io.BytesIO()
        torch.save(layer.state_dict(), buf)
        buf.seek(0)

        layer2 = TopKSparsify(k=5, dim=-1, mode="values")
        layer2.load_state_dict(torch.load(buf, weights_only=True))

        x = torch.randn(2, 8)
        assert torch.equal(layer(x), layer2(x))

    def test_extra_repr(self):
        layer = TopKSparsify(k=7, dim=0, mode="mask")
        r = repr(layer)
        assert "k=7" in r
        assert "dim=0" in r
        assert "mask" in r

    def test_set_k_runtime(self, dtype, device):
        x = torch.randn(3, 10, dtype=dtype, device=device)
        layer = TopKSparsify(k=2, dim=-1, mode="values")
        y2 = layer(x)
        assert ((y2 != 0).sum(dim=-1) == 2).all()

        layer.set_k(5)
        y5 = layer(x)
        assert ((y5 != 0).sum(dim=-1) == 5).all()

    def test_default_dim(self):
        layer = TopKSparsify(k=3)
        assert layer.dim == -1
        assert layer.mode == "values"
