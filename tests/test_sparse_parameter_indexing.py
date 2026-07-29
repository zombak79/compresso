from __future__ import annotations

import pytest
import torch

from compresso import MaskedParam, SRPParam, SRPTensor
from compresso.params._indexing import normalize_row_indices


def _srp_components():
    cols = torch.tensor(
        [
            [0, 3],
            [1, 4],
            [2, 5],
            [0, 5],
        ],
        dtype=torch.long,
    )
    values = torch.tensor(
        [
            [1.0, -1.0],
            [2.0, -2.0],
            [3.0, -3.0],
            [4.0, -4.0],
        ]
    )
    return cols, values


@pytest.mark.parametrize(
    ("index", "expected"),
    [
        (slice(None), [0, 1, 2, 3, 4, 5]),
        (slice(1, None, 2), [1, 3, 5]),
        (slice(-3, None), [3, 4, 5]),
        (slice(-100, 100, 2), [0, 2, 4]),
        (slice(None, None, -1), [5, 4, 3, 2, 1, 0]),
        (slice(5, 0, -2), [5, 3, 1]),
        (slice(4, 2), []),
        (slice(2, 4, -1), []),
        (slice(100, None), []),
    ],
)
def test_normalize_row_indices_preserves_python_slice_semantics(
    index,
    expected,
):
    actual = normalize_row_indices(
        index,
        rows=6,
        device=torch.device("cpu"),
    )

    assert actual.dtype == torch.long
    assert actual.device == torch.device("cpu")
    assert actual.tolist() == expected


def test_normalize_row_indices_supports_slices_of_empty_rows():
    actual = normalize_row_indices(
        slice(None, None, -1),
        rows=0,
        device=torch.device("cpu"),
    )

    assert actual.shape == (0,)
    assert actual.dtype == torch.long


def test_normalize_row_indices_rejects_zero_slice_step():
    with pytest.raises(ValueError, match="slice step cannot be zero"):
        normalize_row_indices(
            slice(None, None, 0),
            rows=6,
            device=torch.device("cpu"),
        )


def test_normalize_row_indices_does_not_allocate_all_rows(monkeypatch):
    calls = []
    original_arange = torch.arange

    def tracked_arange(*args, **kwargs):
        calls.append(args)
        return original_arange(*args, **kwargs)

    monkeypatch.setattr(torch, "arange", tracked_arange)

    actual = normalize_row_indices(
        slice(10, 14, 2),
        rows=10_000_000,
        device=torch.device("cpu"),
    )

    assert actual.tolist() == [10, 12]
    assert calls == [(10, 14, 2)]


def test_srptensor_row_indexing_preserves_order_duplicates_and_gradients():
    cols, values = _srp_components()
    values.requires_grad_()
    tensor = SRPTensor(cols=cols, vals=values, shape=(4, 6))
    requested = torch.tensor([2, 0, 2, -1])

    selected = tensor[requested]

    assert selected.shape == (4, 6)
    assert selected.prefix_shape is None
    assert torch.equal(selected.cols, cols[[2, 0, 2, 3]])
    assert torch.equal(selected.vals, values[[2, 0, 2, 3]])
    selected.vals.sum().backward()
    expected_grad = torch.tensor(
        [
            [1.0, 1.0],
            [0.0, 0.0],
            [2.0, 2.0],
            [1.0, 1.0],
        ]
    )
    torch.testing.assert_close(values.grad, expected_grad)


def test_srpparam_row_indexing_preserves_gradients_to_original_parameter():
    cols, values = _srp_components()
    parameter = SRPParam(cols=cols, values=values, shape=(4, 6))
    requested = torch.tensor([3, 1, 3])

    selected = parameter[requested]

    assert isinstance(selected, SRPTensor)
    assert not isinstance(selected.vals, torch.nn.Parameter)
    assert selected.vals.grad_fn is not None
    assert torch.equal(selected.cols, cols[[3, 1, 3]])
    selected.to_dense().sum().backward()
    expected_grad = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [0.0, 0.0],
            [2.0, 2.0],
        ]
    )
    torch.testing.assert_close(parameter.values.grad, expected_grad)


def test_srp_select_rows_alias_and_slice_preserve_prefix_shape():
    cols, values = _srp_components()
    tensor = SRPTensor(
        cols=cols,
        vals=values,
        shape=(4, 6),
        prefix_shape=(4,),
    )

    selected = tensor.select_rows(slice(1, None, 2))

    assert selected.shape == (2, 6)
    assert selected.prefix_shape == (2,)
    assert torch.equal(selected.cols, cols[[1, 3]])
    assert torch.equal(selected.vals, values[[1, 3]])


@pytest.mark.parametrize("factory", ["tensor", "parameter"])
def test_srp_row_indexing_supports_empty_selection(factory):
    cols, values = _srp_components()
    value = (
        SRPTensor(cols=cols, vals=values, shape=(4, 6))
        if factory == "tensor"
        else SRPParam(cols=cols, values=values, shape=(4, 6))
    )

    selected = value[[]]

    assert selected.shape == (0, 6)
    assert selected.cols.shape == (0, 2)
    assert selected.vals.shape == (0, 2)


@pytest.mark.parametrize(
    ("index", "error", "match"),
    [
        (1, IndexError, "scalar"),
        (torch.tensor([[0, 1]]), IndexError, "1D"),
        (torch.tensor([0.0, 1.0]), TypeError, "integer dtype"),
        (torch.tensor([True, False, True, False]), TypeError, "integer dtype"),
        ([0, 4], IndexError, None),
        ([-5], IndexError, None),
    ],
)
def test_srpparam_row_indexing_rejects_unsupported_indices(
    index,
    error,
    match,
):
    cols, values = _srp_components()
    parameter = SRPParam(cols=cols, values=values, shape=(4, 6))

    with pytest.raises(error, match=match):
        parameter[index]


def test_srptensor_rejects_row_indexing_for_multidimensional_prefix():
    cols, values = _srp_components()
    tensor = SRPTensor(
        cols=cols,
        vals=values,
        shape=(4, 6),
        prefix_shape=(2, 2),
    )

    with pytest.raises(IndexError, match="prefix_shape"):
        tensor[[0, 1]]


@pytest.mark.parametrize("score_mode", ["abs", "raw", "relu"])
@pytest.mark.parametrize("post_norm_l1", [False, True])
def test_maskedparam_row_indexing_matches_full_forward_and_gradients(
    score_mode,
    post_norm_l1,
):
    weight = torch.tensor(
        [
            [0.1, -4.0, 2.0, 0.5],
            [3.0, -0.2, 1.0, -2.0],
            [-1.0, 0.3, 4.0, 2.0],
            [2.5, -3.0, 0.4, 1.0],
        ]
    )
    parameter = MaskedParam(
        weight,
        k_target=2,
        k_schedule=(4, 2),
        score_mode=score_mode,
        ste_alpha=0.25,
        post_norm_l1=post_norm_l1,
    )
    parameter.k_current = 2
    requested = torch.tensor([2, 0, 2])

    expected = parameter()[requested]
    expected.sum().backward()
    expected_grad = parameter.weight.grad.detach().clone()
    parameter.weight.grad = None

    actual = parameter[requested]
    actual.sum().backward()

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(parameter.weight.grad, expected_grad)


def test_maskedparam_frozen_row_indexing_matches_full_forward():
    weight = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [4.0, 3.0, 2.0, 1.0],
            [1.0, -4.0, 2.0, -3.0],
        ]
    )
    parameter = MaskedParam(weight, k_target=2, k_schedule=(4, 2))
    parameter.k_current = 2
    parameter.mask.copy_(
        torch.tensor(
            [
                [True, False, False, True],
                [False, True, True, False],
                [True, True, False, False],
            ]
        )
    )
    parameter.mask_frozen = True
    requested = [2, 0, 2]

    actual = parameter.select_rows(requested)

    torch.testing.assert_close(actual, parameter()[requested])


def test_maskedparam_row_indexing_does_not_call_full_projection(monkeypatch):
    parameter = MaskedParam(
        torch.randn(5, 4),
        k_target=2,
        k_schedule=(4, 2),
    )
    parameter.k_current = 2

    def fail_full_projection(*args, **kwargs):
        raise AssertionError("full projection was called")

    monkeypatch.setattr(parameter, "topk_weights", fail_full_projection)

    selected = parameter[[3, 1]]

    assert selected.shape == (2, 4)


def test_maskedparam_row_indexing_rejects_column_sparsity_and_scalars():
    column_sparse = MaskedParam(
        torch.randn(4, 3),
        k_target=2,
        k_schedule=(4, 2),
        sparsity="col",
    )
    with pytest.raises(NotImplementedError, match="row-wise"):
        column_sparse[[0, 1]]

    row_sparse = MaskedParam(
        torch.randn(4, 3),
        k_target=2,
        k_schedule=(3, 2),
    )
    with pytest.raises(IndexError, match="scalar"):
        row_sparse[0]


def test_to_srp_param_preserves_exact_final_mask_with_zeros_and_ties():
    weight = torch.tensor(
        [
            [0.0, 0.0, 2.0, 2.0, -1.0],
            [3.0, 3.0, 0.0, 0.0, -4.0],
            [1.0, -1.0, 1.0, -1.0, 0.0],
        ]
    )
    parameter = MaskedParam(
        weight,
        k_target=2,
        k_schedule=(5, 2),
    )
    final_mask = torch.tensor(
        [
            [True, False, False, True, False],
            [False, True, True, False, False],
            [True, False, False, False, True],
        ]
    )
    parameter.k_current = 2
    parameter.schedule_done = True
    parameter.mask.copy_(final_mask)

    sparse = parameter.to_srp_param()

    expected_cols = torch.tensor([[0, 3], [1, 2], [0, 4]])
    expected_values = weight.gather(1, expected_cols)
    assert torch.equal(sparse.cols, expected_cols)
    assert torch.equal(sparse.values, expected_values)
    exported_mask = torch.zeros_like(final_mask).scatter(1, sparse.cols, True)
    assert torch.equal(exported_mask, final_mask)
    assert isinstance(sparse.values, torch.nn.Parameter)
    assert sparse.values.requires_grad


def test_to_srp_param_copies_values_and_compatibility_alias_matches():
    weight = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]
    )
    parameter = MaskedParam(weight, k_target=2, k_schedule=(3, 2))
    parameter.k_current = 2
    parameter.mask.copy_(
        torch.tensor(
            [
                [True, False, True],
                [False, True, True],
            ]
        )
    )
    parameter.mask_frozen = True

    canonical = parameter.to_srp_param()
    compatibility = parameter.maskedparam_to_srp()
    parameter.weight.data.zero_()

    assert torch.equal(canonical.cols, compatibility.cols)
    assert torch.equal(canonical.values, compatibility.values)
    assert torch.count_nonzero(canonical.values) == 4
    canonical.values.sum().backward()
    assert canonical.values.grad is not None
    assert parameter.weight.grad is None


def test_to_srp_param_requires_final_or_frozen_exact_row_mask():
    parameter = MaskedParam(
        torch.randn(3, 4),
        k_target=2,
        k_schedule=(4, 2),
    )
    with pytest.raises(RuntimeError, match="completed schedule or frozen"):
        parameter.to_srp_param()

    parameter.schedule_done = True
    parameter.k_current = 2
    parameter.mask.copy_(
        torch.tensor(
            [
                [True, True, False, False],
                [True, False, False, False],
                [False, True, True, True],
            ]
        )
    )
    with pytest.raises(RuntimeError, match="exactly k_current=2"):
        parameter.to_srp_param()


def test_to_srp_param_rejects_column_sparse_parameter():
    parameter = MaskedParam(
        torch.randn(4, 3),
        k_target=2,
        k_schedule=(4, 2),
        sparsity="col",
    )
    parameter.mask_frozen = True

    with pytest.raises(ValueError, match="row-wise"):
        parameter.to_srp_param()
