from __future__ import annotations

import numpy as np
import pytest
import torch

from compresso import TopKSAEConfig, TopKSAETrainer
from compresso.trainers import EmbeddingsDataset
from compresso.trainers.saetrainer import _streaming_mean_variance

DIM = 6


@pytest.fixture
def memmap(tmp_path):
    """A float16 matrix on disk, as an embedding export would be."""
    path = tmp_path / "embeddings.f16"
    rng = np.random.default_rng(0)
    values = (rng.normal(size=(40, DIM)) * 0.5 + 3.0).astype(np.float16)
    handle = np.memmap(path, dtype=np.float16, mode="w+", shape=values.shape)
    handle[:] = values
    handle.flush()
    del handle
    return np.memmap(path, dtype=np.float16, mode="r", shape=values.shape)


def _shares_storage(tensor: torch.Tensor, array: np.ndarray) -> bool:
    return tensor.data_ptr() == array.__array_interface__["data"][0]


# --------------------------------------------------------------------------- #
# The dataset must not materialize its source
# --------------------------------------------------------------------------- #


def test_dataset_keeps_the_source_buffer(memmap):
    """Regression: an eager cast used to pull the whole matrix into RAM."""
    data = EmbeddingsDataset(memmap, batch_size=8, shuffle=False, dtype=torch.float32)

    assert data.embeddings.dtype == torch.float16
    assert _shares_storage(data.embeddings, memmap)


def test_batches_are_converted_on_the_way_out(memmap):
    data = EmbeddingsDataset(memmap, batch_size=8, shuffle=False, dtype=torch.float32)

    batch = data[0]

    assert batch.dtype == torch.float32
    assert batch.shape == (8, DIM)
    assert torch.allclose(batch, torch.from_numpy(np.asarray(memmap[:8], dtype=np.float32)))


def test_integer_sources_are_promoted_per_batch():
    source = np.arange(24, dtype=np.int32).reshape(4, DIM)
    data = EmbeddingsDataset(source, batch_size=2, shuffle=False)

    assert data.embeddings.dtype == torch.int32
    assert torch.is_floating_point(data[0])


def test_non_contiguous_sources_are_not_copied_up_front():
    base = np.zeros((4, DIM * 2), dtype=np.float32)
    view = base[:, ::2]
    assert not view.flags["C_CONTIGUOUS"]

    data = EmbeddingsDataset(view, batch_size=2, shuffle=False)

    assert data[0].shape == (2, DIM)
    assert data[0].is_contiguous()


def test_rows_restrict_the_dataset_without_copying(memmap):
    rows = np.array([5, 1, 9], dtype=np.int64)
    data = EmbeddingsDataset(
        memmap, batch_size=2, shuffle=False, rows=rows, dtype=torch.float32
    )

    assert data.n == 3
    assert len(data) == 2
    assert _shares_storage(data.embeddings, memmap)
    assert torch.allclose(
        data[0], torch.from_numpy(np.asarray(memmap[[5, 1]], dtype=np.float32))
    )


def test_shuffling_permutes_only_the_selected_rows(memmap):
    rows = np.array([2, 4, 6, 8], dtype=np.int64)
    data = EmbeddingsDataset(memmap, batch_size=2, shuffle=True, seed=0, rows=rows)

    data.on_epoch_end()

    assert sorted(data.indices.tolist()) == rows.tolist()


# --------------------------------------------------------------------------- #
# Streaming statistics
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("chunk_bytes", [8, 64, 4096, 1 << 20])
def test_streaming_stats_match_a_single_shot_reduction(chunk_bytes):
    rng = np.random.default_rng(3)
    x = torch.from_numpy(rng.normal(size=(37, DIM)).astype(np.float32))

    mean, variance = _streaming_mean_variance(x, name="test", chunk_bytes=chunk_bytes)
    expected_variance, expected_mean = torch.var_mean(x.double(), dim=0, correction=0)

    assert torch.allclose(mean, expected_mean, atol=1e-12)
    assert torch.allclose(variance, expected_variance, atol=1e-12)


def test_streaming_stats_survive_a_mean_that_dwarfs_the_spread():
    """The E[x^2] - E[x]^2 form collapses to zero here; Chan's merge does not."""
    rng = np.random.default_rng(4)
    x = torch.from_numpy((rng.normal(size=(50_000, 2)) * 0.01 + 50.0).astype(np.float32))

    _mean, variance = _streaming_mean_variance(x, name="test", chunk_bytes=4096)

    exact = x.double().var(dim=0, correction=0)
    assert torch.allclose(variance, exact, rtol=1e-6)

    # The same data through the cancelling formula, kept in float32 as a naive
    # streaming accumulator would: E[x^2] and E[x]^2 are both about 2500 while
    # the variance is about 1e-4, which is past float32's resolution.
    naive = x.square().mean(dim=0) - x.mean(dim=0).square()
    relative_error = ((naive.double() - exact).abs() / exact).max()
    assert relative_error > 0.5


def test_streaming_stats_honour_a_row_subset():
    rng = np.random.default_rng(5)
    x = torch.from_numpy(rng.normal(size=(20, DIM)).astype(np.float32))
    rows = np.array([7, 2, 15, 4], dtype=np.int64)

    mean, variance = _streaming_mean_variance(x, name="test", rows=rows, chunk_bytes=64)
    expected_variance, expected_mean = torch.var_mean(
        x[rows].double(), dim=0, correction=0
    )

    assert torch.allclose(mean, expected_mean, atol=1e-12)
    assert torch.allclose(variance, expected_variance, atol=1e-12)


def test_streaming_stats_never_write_to_the_source():
    x = torch.arange(24, dtype=torch.float64).reshape(4, DIM)
    before = x.clone()

    _streaming_mean_variance(x, name="test", chunk_bytes=64)

    assert torch.equal(x, before)


def test_streaming_stats_reject_non_finite_rows():
    x = torch.tensor([[1.0, float("nan")]])

    with pytest.raises(ValueError, match="test requires finite embeddings"):
        _streaming_mean_variance(x, name="test")


# --------------------------------------------------------------------------- #
# End to end on a memory-mapped float16 source
# --------------------------------------------------------------------------- #


def test_fit_runs_on_a_float16_memmap(memmap):
    trainer = TopKSAETrainer(
        TopKSAEConfig(
            hidden_dim=8,
            k=2,
            batch_size=8,
            epochs=2,
            show_progress=False,
            seed=0,
            standard_scaler_mean=True,
            standard_scaler_std=True,
            validation_frac=0.25,
            patience=2,
        )
    )
    trainer.fit(memmap)

    assert len(trainer.history) == 2
    assert "val_loss" in trainer.history[-1]
    # Statistics come from the training rows, in the model dtype, not float16.
    assert trainer.input_scaler_mean.dtype == torch.float32
    assert torch.isfinite(trainer.input_scaler_mean).all()
    assert trainer.reconstruct(memmap).shape == (memmap.shape[0], DIM)


def test_split_statistics_ignore_validation_rows_on_a_memmap(memmap):
    config = TopKSAEConfig(
        hidden_dim=8,
        k=2,
        batch_size=8,
        epochs=1,
        show_progress=False,
        seed=0,
        standard_scaler_mean=True,
        validation_frac=0.25,
    )
    trainer = TopKSAETrainer(config).fit(memmap)

    train_rows, _val_rows = trainer._split_validation(memmap, 0.25)
    rows = np.sort(train_rows)
    expected = torch.from_numpy(np.asarray(memmap, dtype=np.float64)[rows]).mean(dim=0)

    assert torch.allclose(trainer.input_scaler_mean.double(), expected, atol=1e-6)
