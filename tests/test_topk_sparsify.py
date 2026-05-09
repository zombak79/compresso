"""Tests for compresso.nn.sparsify.TopKSparsify."""

import io
import torch

from compresso.functional import topk_ste
from compresso.nn import TopKSparsify


class TestTopKSparsify:
    def test_module_matches_functional(self, dtype, device):
        x = torch.randn(4, 10, dtype=dtype, device=device)
        layer = TopKSparsify(k=3, dim=-1, score_mode="abs", ste_alpha=0.1)
        expected = topk_ste(x, k=3, dim=-1, score_mode="abs", ste_alpha=0.1)
        assert torch.equal(layer(x), expected)

    def test_state_dict_roundtrip(self):
        layer = TopKSparsify(k=5, dim=-1, score_mode="raw", ste_alpha=0.25)
        buf = io.BytesIO()
        torch.save(layer.state_dict(), buf)
        buf.seek(0)

        layer2 = TopKSparsify(k=5, dim=-1, score_mode="raw", ste_alpha=0.25)
        layer2.load_state_dict(torch.load(buf, weights_only=True))

        x = torch.randn(2, 8)
        assert torch.equal(layer(x), layer2(x))

    def test_extra_repr(self):
        layer = TopKSparsify(k=7, dim=0, score_mode="relu", ste_alpha=0.01)
        r = repr(layer)
        assert "k=7" in r
        assert "dim=0" in r
        assert "relu" in r
        assert "ste_alpha=0.01" in r

    def test_set_k_runtime(self, dtype, device):
        x = torch.randn(3, 10, dtype=dtype, device=device)
        layer = TopKSparsify(k=2, dim=-1)
        y2 = layer(x)
        assert ((y2 != 0).sum(dim=-1) == 2).all()

        layer.set_k(5)
        y5 = layer(x)
        assert ((y5 != 0).sum(dim=-1) == 5).all()

    def test_defaults(self):
        layer = TopKSparsify(k=3)
        assert layer.dim == -1
        assert layer.score_mode == "abs"
        assert layer.ste_alpha == 0.0
