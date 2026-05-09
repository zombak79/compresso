"""Tests for compresso.functional.sparsify.topk_ste."""

import pytest
import torch

from compresso.functional import topk_ste


HAND_TENSOR = torch.tensor(
    [
        [0.1, -5.0, 3.0, -0.2, 4.0],
        [2.0, 2.0, -9.0, 0.0, 1.0],
        [-1.0, 0.5, 0.5, 0.5, 7.0],
    ]
)


class TestForward:
    def test_keeps_exactly_k_nonzeros_per_row(self, dtype, device):
        x = HAND_TENSOR.to(dtype=dtype, device=device)
        y = topk_ste(x, k=2, dim=-1)
        assert ((y != 0).sum(dim=-1) == 2).all()

    def test_selected_values_match_signed_input(self, dtype, device):
        x = HAND_TENSOR.to(dtype=dtype, device=device)
        y = topk_ste(x, k=2, dim=-1)
        mask = y != 0
        assert torch.allclose(y[mask], x[mask])

    def test_shape_dtype_device_preserved(self, dtype, device):
        x = HAND_TENSOR.to(dtype=dtype, device=device)
        y = topk_ste(x, k=2, dim=-1)
        assert y.shape == x.shape
        assert y.dtype == x.dtype
        assert y.device == x.device


class TestScoreMode:
    def test_raw_ranks_by_signed_values(self, dtype, device):
        x = torch.tensor([[-10.0, -1.0, 0.2, 0.1]], dtype=dtype, device=device)
        y = topk_ste(x, k=2, score_mode="raw")
        assert y[0, 2].item() == pytest.approx(0.2)
        assert y[0, 3].item() == pytest.approx(0.1)

    def test_relu_ranks_positive_part(self, dtype, device):
        x = torch.tensor([[-10.0, -1.0, 0.2, 0.1]], dtype=dtype, device=device)
        y = topk_ste(x, k=2, score_mode="relu")
        assert y[0, 2].item() == pytest.approx(0.2)
        assert y[0, 3].item() == pytest.approx(0.1)

    def test_abs_ranks_by_magnitude(self, dtype, device):
        x = torch.tensor([[-10.0, -1.0, 0.2, 0.1]], dtype=dtype, device=device)
        y = topk_ste(x, k=2, score_mode="abs")
        assert y[0, 0].item() == pytest.approx(-10.0)
        assert y[0, 1].item() == pytest.approx(-1.0)


class TestGradients:
    def test_selected_get_full_grad_rest_get_alpha(self, dtype, device):
        x = torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0]], dtype=dtype, device=device, requires_grad=True)
        y = topk_ste(x, k=2, ste_alpha=0.1, score_mode="abs")
        y.sum().backward()
        expected = torch.tensor([[1.0, 1.0, 0.1, 0.1, 0.1]], dtype=dtype, device=device)
        assert torch.allclose(x.grad, expected)

    def test_alpha_zero_is_selected_only_backward(self, dtype, device):
        x = torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0]], dtype=dtype, device=device, requires_grad=True)
        y = topk_ste(x, k=2, ste_alpha=0.0, score_mode="abs")
        y.sum().backward()
        expected = torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0]], dtype=dtype, device=device)
        assert torch.allclose(x.grad, expected)

    def test_alpha_one_is_identity_backward(self, dtype, device):
        x = torch.randn(4, 8, dtype=dtype, device=device, requires_grad=True)
        y = topk_ste(x, k=3, ste_alpha=1.0)
        y.sum().backward()
        assert torch.allclose(x.grad, torch.ones_like(x))


class TestValidation:
    def test_invalid_score_mode_raises(self):
        x = torch.randn(2, 4)
        with pytest.raises(ValueError, match="Unknown score_mode"):
            topk_ste(x, k=2, score_mode="invalid")

    def test_invalid_alpha_raises(self):
        x = torch.randn(2, 4)
        with pytest.raises(ValueError, match="ste_alpha must be in"):
            topk_ste(x, k=2, ste_alpha=1.5)
