"""Tests for compresso.functional.sparsify.topk_ste."""

import pytest
import torch

from compresso.functional import topk_ste


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Hand-crafted 3×5 tensor with known top-k by abs value.
#   row 0: [ 0.1, -5.0,  3.0, -0.2,  4.0]  top-2 by abs: idx 1 (-5.0), idx 4 (4.0)
#   row 1: [ 2.0,  2.0, -9.0,  0.0,  1.0]  top-2 by abs: idx 2 (-9.0), idx 0 or 1 (2.0)
#   row 2: [-1.0,  0.5,  0.5,  0.5,  7.0]  top-2 by abs: idx 4 (7.0), idx 0 (-1.0)
HAND_TENSOR = torch.tensor(
    [
        [0.1, -5.0, 3.0, -0.2, 4.0],
        [2.0, 2.0, -9.0, 0.0, 1.0],
        [-1.0, 0.5, 0.5, 0.5, 7.0],
    ]
)
HAND_K = 2


# ---------------------------------------------------------------------------
# Forward correctness
# ---------------------------------------------------------------------------


class TestForwardCorrectness:
    def test_keeps_exactly_k_nonzeros_per_row(self, dtype, device):
        x = HAND_TENSOR.to(dtype=dtype, device=device)
        y = topk_ste(x, k=HAND_K, dim=-1, mode="values")

        nonzeros_per_row = (y != 0).sum(dim=-1)
        assert (nonzeros_per_row == HAND_K).all()

    def test_selected_values_match_original_signed(self, dtype, device):
        x = HAND_TENSOR.to(dtype=dtype, device=device)
        y = topk_ste(x, k=HAND_K, dim=-1, mode="values")

        # Where y is nonzero, it must equal the original x
        mask = y != 0
        assert torch.allclose(y[mask], x[mask])

    def test_shape_dtype_device_preserved(self, dtype, device):
        x = HAND_TENSOR.to(dtype=dtype, device=device)
        y = topk_ste(x, k=HAND_K, dim=-1, mode="values")

        assert y.shape == x.shape
        assert y.dtype == x.dtype
        assert y.device == x.device

    def test_no_duplicate_positions(self, dtype, device):
        x = torch.randn(8, 20, dtype=dtype, device=device)
        y = topk_ste(x, k=5, dim=-1, mode="values")
        # nonzero positions per row should be exactly 5 distinct
        for row in range(8):
            idxs = torch.where(y[row] != 0)[0]
            assert idxs.numel() == 5
            assert idxs.unique().numel() == 5

    def test_deterministic_on_fixed_seed(self, dtype, device):
        torch.manual_seed(123)
        x = torch.randn(4, 10, dtype=dtype, device=device)
        y1 = topk_ste(x, k=3, dim=-1)

        torch.manual_seed(123)
        x2 = torch.randn(4, 10, dtype=dtype, device=device)
        y2 = topk_ste(x2, k=3, dim=-1)

        assert torch.equal(y1, y2)


# ---------------------------------------------------------------------------
# Mode behavior
# ---------------------------------------------------------------------------


class TestModes:
    def test_mode_values(self, dtype, device):
        x = HAND_TENSOR.to(dtype=dtype, device=device)
        y = topk_ste(x, k=HAND_K, dim=-1, mode="values")

        # nonzeros have correct sign and value
        mask = y != 0
        assert torch.allclose(y[mask], x[mask])
        # zeros are truly zero
        assert (y[~mask] == 0).all()

    def test_mode_mask(self, dtype, device):
        x = HAND_TENSOR.to(dtype=dtype, device=device)
        m = topk_ste(x, k=HAND_K, dim=-1, mode="mask")

        # binary {0, 1}
        assert set(m.unique().tolist()).issubset({0.0, 1.0})
        # exactly k ones per row
        assert (m.sum(dim=-1) == HAND_K).all()

    def test_mode_values_ste_identity_is_alias(self, dtype, device):
        x = HAND_TENSOR.to(dtype=dtype, device=device)
        y1 = topk_ste(x, k=HAND_K, dim=-1, mode="values")
        y2 = topk_ste(x, k=HAND_K, dim=-1, mode="values_ste_identity")
        assert torch.equal(y1, y2)

    def test_score_mode_raw_ranks_by_signed_values(self, dtype, device):
        x = torch.tensor([[-10.0, -1.0, 0.2, 0.1]], dtype=dtype, device=device)
        y = topk_ste(x, k=2, dim=-1, mode="values", score_mode="raw")
        # raw top-k picks 0.2 and 0.1 (largest signed values)
        assert y[0, 2].item() == pytest.approx(0.2)
        assert y[0, 3].item() == pytest.approx(0.1)
        assert (y != 0).sum().item() == 2

    def test_score_mode_relu_ignores_negative_rank(self, dtype, device):
        x = torch.tensor([[-10.0, -1.0, 0.2, 0.1]], dtype=dtype, device=device)
        y = topk_ste(x, k=2, dim=-1, mode="values", score_mode="relu")
        # relu top-k also picks positive entries here
        assert y[0, 2].item() == pytest.approx(0.2)
        assert y[0, 3].item() == pytest.approx(0.1)
        assert (y != 0).sum().item() == 2

    def test_score_mode_abs_keeps_magnitude_winners(self, dtype, device):
        x = torch.tensor([[-10.0, -1.0, 0.2, 0.1]], dtype=dtype, device=device)
        y = topk_ste(x, k=2, dim=-1, mode="values", score_mode="abs")
        # abs top-k picks -10 and -1 by magnitude
        assert y[0, 0].item() == pytest.approx(-10.0)
        assert y[0, 1].item() == pytest.approx(-1.0)
        assert (y != 0).sum().item() == 2


# ---------------------------------------------------------------------------
# Gradient tests
# ---------------------------------------------------------------------------


class TestGradients:
    def test_ste_identity_grad_exists_and_finite(self, dtype, device):
        x = torch.randn(4, 8, dtype=dtype, device=device, requires_grad=True)
        y = topk_ste(x, k=3, dim=-1, mode="values")
        y.sum().backward()

        assert x.grad is not None
        assert x.grad.isfinite().all()

    def test_ste_identity_full_gradient(self, dtype, device):
        """Identity STE: gradient flows to ALL positions as if no mask."""
        x = torch.randn(4, 8, dtype=dtype, device=device, requires_grad=True)
        y = topk_ste(x, k=3, dim=-1, mode="values_ste_identity")
        y.sum().backward()

        expected = torch.ones_like(x)
        assert torch.allclose(x.grad, expected)

    def test_ste_selected_grad_exists_and_finite(self, dtype, device):
        x = torch.randn(4, 8, dtype=dtype, device=device, requires_grad=True)
        y = topk_ste(x, k=3, dim=-1, mode="values_ste_selected")
        y.sum().backward()

        assert x.grad is not None
        assert x.grad.isfinite().all()

    def test_ste_selected_masked_gradient(self, dtype, device):
        """Selected STE: gradient flows only to top-k positions."""
        x = torch.randn(4, 8, dtype=dtype, device=device, requires_grad=True)
        y = topk_ste(x, k=3, dim=-1, mode="values_ste_selected")
        y.sum().backward()

        # Grad should be 1 where selected, 0 elsewhere
        selected_mask = (y.detach() != 0).to(dtype)
        assert torch.allclose(x.grad, selected_mask)

    def test_mask_mode_no_grad(self, dtype, device):
        """Mask mode: output is detached, no gradient to input."""
        x = torch.randn(4, 8, dtype=dtype, device=device, requires_grad=True)
        m = topk_ste(x, k=3, dim=-1, mode="mask")

        # detached tensor has no grad_fn
        assert not m.requires_grad
        assert x.grad is None


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_dim_0_column_wise(self, dtype, device):
        """Top-k along dim=0 selects k entries per column."""
        x = torch.randn(6, 4, dtype=dtype, device=device)
        y = topk_ste(x, k=2, dim=0, mode="values")

        # 2 nonzeros per column
        nonzeros_per_col = (y != 0).sum(dim=0)
        assert (nonzeros_per_col == 2).all()

    def test_k_equals_full_dim(self, dtype, device):
        """k == dim size: output equals input."""
        x = torch.randn(3, 5, dtype=dtype, device=device)
        y = topk_ste(x, k=5, dim=-1, mode="values")
        assert torch.allclose(y, x)

    def test_k_equals_1(self, dtype, device):
        """k=1: single largest absolute value per row."""
        x = HAND_TENSOR.to(dtype=dtype, device=device)
        y = topk_ste(x, k=1, dim=-1, mode="values")

        nonzeros_per_row = (y != 0).sum(dim=-1)
        assert (nonzeros_per_row == 1).all()

        # row 0: max abs is -5.0 at idx 1
        assert y[0, 1].item() == pytest.approx(-5.0)
        # row 1: max abs is -9.0 at idx 2
        assert y[1, 2].item() == pytest.approx(-9.0)
        # row 2: max abs is 7.0 at idx 4
        assert y[2, 4].item() == pytest.approx(7.0)

    def test_ties_handled_no_crash(self, dtype, device):
        """Tensor with tied absolute values doesn't crash."""
        x = torch.tensor(
            [[1.0, -1.0, 1.0, -1.0, 1.0]], dtype=dtype, device=device
        )
        y = topk_ste(x, k=3, dim=-1, mode="values")
        assert (y != 0).sum().item() == 3

    def test_3d_tensor(self, dtype, device):
        """Works on 3D tensors (batch × seq × features)."""
        x = torch.randn(2, 3, 10, dtype=dtype, device=device)
        y = topk_ste(x, k=4, dim=-1, mode="values")
        assert y.shape == x.shape
        # 4 nonzeros per last-dim slice
        nonzeros = (y != 0).sum(dim=-1)
        assert (nonzeros == 4).all()

    def test_1d_tensor(self, dtype, device):
        """Works on 1D tensor."""
        x = torch.randn(10, dtype=dtype, device=device)
        y = topk_ste(x, k=3, dim=0, mode="values")
        assert y.shape == x.shape
        assert (y != 0).sum().item() == 3


# ---------------------------------------------------------------------------
# k_backward tests
# ---------------------------------------------------------------------------


class TestKBackward:
    def test_forward_still_k_sparse(self, dtype, device):
        """Forward output has exactly k nonzeros, not k_backward."""
        x = torch.randn(4, 20, dtype=dtype, device=device)
        y = topk_ste(x, k=3, dim=-1, mode="values", k_backward=10)
        nonzeros = (y != 0).sum(dim=-1)
        assert (nonzeros == 3).all()

    def test_backward_width(self, dtype, device):
        """Gradient is nonzero at exactly k_backward positions."""
        x = torch.randn(4, 20, dtype=dtype, device=device, requires_grad=True)
        y = topk_ste(x, k=3, dim=-1, mode="values", k_backward=10)
        y.sum().backward()

        grad_nonzero = (x.grad != 0).sum(dim=-1)
        assert (grad_nonzero == 10).all()

    def test_k_backward_uses_score_mode_for_selection(self, dtype, device):
        # abs and raw select different backward supports here
        x_abs = torch.tensor([[-10.0, -1.0, 0.2, 0.1]], dtype=dtype, device=device, requires_grad=True)
        y_abs = topk_ste(x_abs, k=1, dim=-1, mode="values", score_mode="abs", k_backward=2)
        y_abs.sum().backward()
        mask_abs = x_abs.grad != 0

        x_raw = torch.tensor([[-10.0, -1.0, 0.2, 0.1]], dtype=dtype, device=device, requires_grad=True)
        y_raw = topk_ste(x_raw, k=1, dim=-1, mode="values", score_mode="raw", k_backward=2)
        y_raw.sum().backward()
        mask_raw = x_raw.grad != 0

        # abs backward support: indices 0 and 1
        assert mask_abs[0, 0].item()
        assert mask_abs[0, 1].item()
        # raw backward support: indices 2 and 3
        assert mask_raw[0, 2].item()
        assert mask_raw[0, 3].item()

    def test_backward_is_one_at_selected(self, dtype, device):
        """Grad values are 1 at the k_backward positions, 0 elsewhere."""
        x = torch.randn(4, 20, dtype=dtype, device=device, requires_grad=True)
        y = topk_ste(x, k=3, dim=-1, mode="values", k_backward=10)
        y.sum().backward()

        # grad should be binary: 1 at top-10-by-abs, 0 elsewhere
        assert set(x.grad.unique().tolist()).issubset({0.0, 1.0})

    def test_k_backward_equals_k_matches_selected(self, dtype, device):
        """k_backward == k should behave like values_ste_selected."""
        x = torch.randn(4, 20, dtype=dtype, device=device, requires_grad=True)
        y = topk_ste(x, k=5, dim=-1, mode="values", k_backward=5)
        y.sum().backward()

        x2 = x.detach().clone().requires_grad_(True)
        y2 = topk_ste(x2, k=5, dim=-1, mode="values_ste_selected")
        y2.sum().backward()

        assert torch.equal(x.grad, x2.grad)

    def test_k_backward_equals_dim_matches_identity(self, dtype, device):
        """k_backward == dim size should behave like identity STE."""
        x = torch.randn(4, 20, dtype=dtype, device=device, requires_grad=True)
        y = topk_ste(x, k=5, dim=-1, mode="values", k_backward=20)
        y.sum().backward()

        expected = torch.ones_like(x)
        assert torch.allclose(x.grad, expected)

    def test_k_backward_superset_of_k(self, dtype, device):
        """All k forward positions are within the k_backward backward set."""
        x = torch.randn(4, 20, dtype=dtype, device=device, requires_grad=True)
        y = topk_ste(x, k=3, dim=-1, mode="values", k_backward=10)
        y.sum().backward()

        fwd_mask = (y.detach() != 0)
        bwd_mask = (x.grad != 0)
        # every forward-selected position must also have gradient
        assert (fwd_mask & ~bwd_mask).sum() == 0

    def test_k_backward_ignored_for_mask_mode(self, dtype, device):
        """mask mode ignores k_backward — still returns detached binary mask."""
        x = torch.randn(4, 20, dtype=dtype, device=device, requires_grad=True)
        m = topk_ste(x, k=3, dim=-1, mode="mask", k_backward=10)
        assert not m.requires_grad
        assert (m.sum(dim=-1) == 3).all()

    def test_k_backward_with_dim_0(self, dtype, device):
        """k_backward works with column-wise selection."""
        x = torch.randn(10, 4, dtype=dtype, device=device, requires_grad=True)
        y = topk_ste(x, k=2, dim=0, mode="values", k_backward=6)
        y.sum().backward()

        # 6 nonzero grads per column
        grad_nonzero_per_col = (x.grad != 0).sum(dim=0)
        assert (grad_nonzero_per_col == 6).all()
