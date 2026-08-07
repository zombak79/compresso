from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from compresso import L1Normalize, SRPTensor, TopKSAEConfig, TopKSAETrainer
from compresso.trainers import EmbeddingsDataset


def _accelerator() -> str | None:
    """Return a non-CPU device name, used to exercise device-to-host copies."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return None


requires_accelerator = pytest.mark.skipif(
    _accelerator() is None,
    reason="needs a non-CPU device to exercise device-to-host copies",
)


def test_embeddings_dataset_batches_and_shuffle():
    x = np.arange(20, dtype=np.float32).reshape(10, 2)
    data = EmbeddingsDataset(x, batch_size=4, shuffle=True, seed=0)

    first = data[0]
    assert first.shape == (4, 2)
    before = data.indices.copy()
    data.on_epoch_end()
    assert sorted(data.indices.tolist()) == sorted(before.tolist())
    assert not np.array_equal(data.indices, before)


def test_topk_sae_trainer_fit_transform_returns_srp():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(24, 8)).astype(np.float32)
    trainer = TopKSAETrainer(
        TopKSAEConfig(
            hidden_dim=16,
            k=3,
            batch_size=8,
            epochs=2,
            post_sparsify=L1Normalize(),
            sparsify_score_mode="abs",
            sparsify_ste_alpha=0.01,
            show_progress=False,
            seed=123,
        )
    )

    srp = trainer.fit_transform(x)

    assert isinstance(srp, SRPTensor)
    assert srp.shape == (24, 16)
    assert srp.k == 3
    assert len(trainer.history) == 2
    assert {"loss", "cosine_loss", "reconstruction_mse"}.issubset(trainer.history[-1])
    assert torch.allclose(srp.vals.abs().sum(dim=1), torch.ones(24), atol=1e-5)


def test_topk_sae_trainer_fit_transform_accepts_float64_numpy_embeddings():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(20, 6))
    trainer = TopKSAETrainer(
        TopKSAEConfig(
            hidden_dim=12,
            k=3,
            batch_size=5,
            epochs=1,
            show_progress=False,
            seed=321,
        )
    )

    srp = trainer.fit_transform(x)

    assert isinstance(srp, SRPTensor)
    assert srp.shape == (20, 12)
    assert srp.vals.dtype == torch.float32


def test_topk_sae_trainer_encode_and_reconstruct_shapes():
    x = torch.randn(10, 6)
    trainer = TopKSAETrainer(
        TopKSAEConfig(
            hidden_dim=12,
            k=2,
            batch_size=5,
            epochs=1,
            show_progress=False,
            seed=7,
        )
    ).fit(x)

    codes = trainer.encode(x)
    recon = trainer.reconstruct(x)

    assert codes.shape == (10, 12)
    assert recon.shape == (10, 6)
    assert (codes != 0).sum(dim=1).tolist() == [2] * 10


def test_topk_sae_trainer_transform_accepts_float64_torch_embeddings():
    x = torch.randn(10, 6)
    trainer = TopKSAETrainer(
        TopKSAEConfig(
            hidden_dim=12,
            k=2,
            batch_size=5,
            epochs=1,
            show_progress=False,
            seed=17,
        )
    ).fit(x)

    srp = trainer.transform(x.double())

    assert isinstance(srp, SRPTensor)
    assert srp.shape == (10, 12)
    assert srp.vals.dtype == torch.float32


def test_topk_sae_trainer_cosine_lr_decay_records_learning_rate():
    x = torch.randn(12, 5)
    trainer = TopKSAETrainer(
        TopKSAEConfig(
            hidden_dim=10,
            k=2,
            batch_size=4,
            epochs=4,
            lr=0.1,
            decay=True,
            show_progress=False,
            seed=11,
        )
    ).fit(x)

    lrs = [record["lr"] for record in trainer.history]

    assert lrs[0] == 0.1
    assert all(a > b for a, b in zip(lrs, lrs[1:]))


@pytest.mark.parametrize(
    ("noise_scale", "expected_scale"),
    [
        ("global_rms", torch.tensor(0.5).sqrt()),
        ("feature_std", torch.tensor([1.0, 0.0])),
    ],
)
def test_topk_sae_trainer_fits_adaptive_gaussian_scale(noise_scale, expected_scale):
    x = torch.tensor([[1.0, 2.0], [3.0, 2.0]])
    trainer = TopKSAETrainer(
        TopKSAEConfig(
            hidden_dim=4,
            k=2,
            noise_type="gaussian",
            noise_scale=noise_scale,
            show_progress=False,
        )
    ).build(input_dim=2)

    trainer._fit_gaussian_noise_scale(x)

    assert torch.equal(trainer.input_feature_mean, torch.tensor([2.0, 2.0]))
    assert torch.equal(trainer.input_feature_variance, torch.tensor([1.0, 0.0]))
    assert torch.allclose(trainer._gaussian_noise_scale.cpu(), expected_scale)


def test_topk_sae_trainer_gaussian_noise_is_independent_of_global_rng():
    config = TopKSAEConfig(
        hidden_dim=6,
        k=2,
        noise_type="gaussian",
        noise_scale="absolute",
        noise_level=0.25,
        show_progress=False,
        seed=123,
    )
    first = TopKSAETrainer(config).build(input_dim=3)
    torch.randn(100)
    first_corrupted = first._corrupt(torch.zeros(4, 3))

    second = TopKSAETrainer(config).build(input_dim=3)
    torch.randn(7)
    second_corrupted = second._corrupt(torch.zeros(4, 3))

    assert torch.equal(first_corrupted, second_corrupted)
    assert not torch.equal(first_corrupted, torch.zeros_like(first_corrupted))


def test_topk_sae_trainer_denoising_loss_targets_clean_input():
    clean = torch.tensor([[1.0, -1.0], [0.5, 2.0]])
    corrupted = clean + 3.0
    trainer = TopKSAETrainer(
        TopKSAEConfig(
            hidden_dim=4,
            k=2,
            noise_type="gaussian",
            noise_level=1.0,
            alpha_loss=0.0,
            lr=0.0,
            show_progress=False,
            seed=9,
        )
    ).build(input_dim=2)
    trainer._corrupt = lambda _batch: corrupted

    with torch.no_grad():
        reconstruction, _sparse, _stats = trainer.sae(corrupted)
    expected_clean_mse = F.mse_loss(reconstruction, clean)
    expected_corrupted_mse = F.mse_loss(reconstruction, corrupted)

    stats = trainer.train_step(clean)

    assert torch.allclose(stats["loss"], expected_clean_mse)
    assert torch.allclose(stats["reconstruction_mse"], expected_clean_mse)
    assert torch.allclose(stats["corrupted_reconstruction_mse"], expected_corrupted_mse)


def test_topk_sae_trainer_denoising_history_and_inference_do_not_add_noise():
    x = torch.randn(12, 4)
    trainer = TopKSAETrainer(
        TopKSAEConfig(
            hidden_dim=8,
            k=2,
            noise_type="gaussian",
            noise_scale="feature_std",
            noise_level=0.2,
            batch_size=4,
            epochs=1,
            show_progress=False,
            seed=5,
        )
    ).fit(x)

    assert {
        "corrupted_cosine_loss",
        "corrupted_reconstruction_mse",
    }.issubset(trainer.history[-1])
    generator_state = trainer._noise_generator.get_state()

    trainer.encode(x)
    trainer.reconstruct(x)

    assert torch.equal(trainer._noise_generator.get_state(), generator_state)


def test_topk_sae_trainer_state_roundtrip_restores_denoising():
    x = torch.randn(12, 4)
    trainer = TopKSAETrainer(
        TopKSAEConfig(
            hidden_dim=8,
            k=2,
            noise_type="gaussian",
            noise_scale="global_rms",
            noise_level=0.2,
            batch_size=4,
            epochs=1,
            show_progress=False,
            seed=15,
        )
    ).fit(x)

    restored = TopKSAETrainer.from_state_dict(trainer.state_dict())

    assert torch.equal(restored.input_feature_mean, trainer.input_feature_mean)
    assert torch.equal(restored.input_feature_variance, trainer.input_feature_variance)
    assert torch.equal(restored._gaussian_noise_scale, trainer._gaussian_noise_scale)
    assert torch.equal(restored.reconstruct(x), trainer.reconstruct(x))
    probe = torch.zeros(3, 4)
    assert torch.equal(restored._corrupt(probe), trainer._corrupt(probe))


def test_topk_sae_trainer_loads_legacy_state_without_denoising_fields():
    x = torch.randn(8, 3)
    trainer = TopKSAETrainer(
        TopKSAEConfig(
            hidden_dim=6,
            k=2,
            epochs=1,
            show_progress=False,
            seed=21,
        )
    ).fit(x)
    current_state = trainer.state_dict()
    legacy_state = {
        key: current_state[key]
        for key in ("config", "input_dim", "model", "optimizer", "history")
    }

    restored = TopKSAETrainer.from_state_dict(legacy_state)

    assert restored.input_feature_mean is None
    assert restored.input_feature_variance is None
    assert restored._gaussian_noise_scale is None
    assert torch.equal(restored.reconstruct(x), trainer.reconstruct(x))


def test_topk_sae_trainer_state_restore_does_not_share_custom_modules():
    encoder = nn.Linear(3, 6)
    trainer = TopKSAETrainer(
        TopKSAEConfig(
            hidden_dim=6,
            k=2,
            encoder=encoder,
            show_progress=False,
        )
    ).build(input_dim=3)

    restored = TopKSAETrainer.from_state_dict(trainer.state_dict())

    assert restored.sae.encoder is not trainer.sae.encoder
    assert torch.equal(restored.sae.encoder.weight, trainer.sae.encoder.weight)


@pytest.mark.parametrize(
    ("config_kwargs", "message"),
    [
        ({"noise_type": "invalid"}, "unknown noise_type"),
        ({"noise_scale": "invalid"}, "unknown noise_scale"),
        ({"noise_level": -0.1}, "noise_level"),
        ({"noise_level": float("nan")}, "noise_level"),
        ({"alpha_loss": 1.1}, "alpha_loss"),
        ({"l1_penalty": -0.1}, "l1_penalty"),
    ],
)
def test_topk_sae_trainer_validates_denoising_config(config_kwargs, message):
    config = TopKSAEConfig(hidden_dim=4, k=2, show_progress=False, **config_kwargs)

    with pytest.raises(ValueError, match=message):
        TopKSAETrainer(config).build(input_dim=2)


def test_topk_sae_trainer_rejects_empty_or_nonfinite_adaptive_inputs():
    config = TopKSAEConfig(
        hidden_dim=4,
        k=2,
        noise_type="gaussian",
        noise_scale="feature_std",
        epochs=1,
        show_progress=False,
    )

    with pytest.raises(ValueError, match="at least one row"):
        TopKSAETrainer(config).fit(torch.empty(0, 2))
    with pytest.raises(ValueError, match="finite embeddings"):
        TopKSAETrainer(config).fit(torch.tensor([[1.0, float("nan")]]))


@pytest.mark.parametrize(
    ("config_kwargs", "message"),
    [
        ({"validation_frac": 0.0}, "validation_frac"),
        ({"validation_frac": 1.0}, "validation_frac"),
        ({"validation_frac": float("nan")}, "validation_frac"),
        ({"patience": 0}, "patience"),
        ({"min_delta": -0.1}, "min_delta"),
    ],
)
def test_topk_sae_trainer_validates_early_stopping_config(config_kwargs, message):
    config = TopKSAEConfig(hidden_dim=4, k=2, show_progress=False, **config_kwargs)

    with pytest.raises(ValueError, match=message):
        TopKSAETrainer(config).build(input_dim=2)


def test_topk_sae_trainer_rejects_conflicting_validation_sources():
    config = TopKSAEConfig(hidden_dim=4, k=2, epochs=1, validation_frac=0.2, show_progress=False)

    with pytest.raises(ValueError, match="not both"):
        TopKSAETrainer(config).fit(torch.randn(10, 3), validation_embeddings=torch.randn(4, 3))


def test_topk_sae_trainer_requires_validation_for_patience():
    config = TopKSAEConfig(hidden_dim=4, k=2, epochs=1, patience=2, show_progress=False)

    with pytest.raises(ValueError, match="patience requires"):
        TopKSAETrainer(config).fit(torch.randn(10, 3))


def test_topk_sae_trainer_rejects_validation_embeddings_with_wrong_width():
    config = TopKSAEConfig(hidden_dim=4, k=2, epochs=1, show_progress=False)

    with pytest.raises(ValueError, match="match embeddings"):
        TopKSAETrainer(config).fit(torch.randn(10, 3), validation_embeddings=torch.randn(4, 5))


def test_topk_sae_trainer_validation_frac_requires_two_rows():
    config = TopKSAEConfig(hidden_dim=4, k=2, epochs=1, validation_frac=0.5, show_progress=False)

    with pytest.raises(ValueError, match="at least 2 embedding rows"):
        TopKSAETrainer(config).fit(torch.randn(1, 3))


def test_topk_sae_trainer_validation_split_is_deterministic_and_covers_all_rows():
    x = torch.arange(40, dtype=torch.float32).reshape(20, 2)
    trainer = TopKSAETrainer(TopKSAEConfig(hidden_dim=4, k=2, seed=99, show_progress=False))

    # The split returns row indices, so neither part is copied out of the source.
    train_a, val_a = trainer._split_validation(x, 0.3)
    train_b, val_b = trainer._split_validation(x, 0.3)

    assert np.array_equal(train_a, train_b)
    assert np.array_equal(val_a, val_b)
    assert train_a.size == 14
    assert val_a.size == 6
    combined = np.concatenate([train_a, val_a])
    assert sorted(combined.tolist()) == list(range(20))
    # Rows are permuted before splitting, so validation is not simply the tail.
    assert val_a.tolist() != list(range(14, 20))


def test_topk_sae_trainer_validation_frac_records_validation_metrics():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(24, 4)).astype(np.float32)
    trainer = TopKSAETrainer(
        TopKSAEConfig(
            hidden_dim=8,
            k=2,
            batch_size=8,
            epochs=2,
            validation_frac=0.25,
            show_progress=False,
            seed=17,
        )
    ).fit(x)

    assert len(trainer.history) == 2
    for record in trainer.history:
        assert "val_loss" in record
        assert "val_cosine_loss" in record
        assert "val_reconstruction_mse" in record
    assert trainer.stopped_epoch is None
    assert trainer.best_epoch in {1, 2}


def test_topk_sae_trainer_accepts_explicit_validation_embeddings():
    rng = np.random.default_rng(8)
    x = rng.normal(size=(16, 4)).astype(np.float32)
    x_val = rng.normal(size=(6, 4)).astype(np.float32)
    trainer = TopKSAETrainer(
        TopKSAEConfig(hidden_dim=8, k=2, batch_size=8, epochs=2, show_progress=False, seed=21)
    ).fit(x, validation_embeddings=x_val)

    assert len(trainer.history) == 2
    assert all("val_loss" in record for record in trainer.history)
    assert trainer.best_val_loss is not None


def test_topk_sae_trainer_without_validation_records_no_validation_metrics():
    trainer = TopKSAETrainer(
        TopKSAEConfig(hidden_dim=6, k=2, batch_size=6, epochs=2, show_progress=False, seed=1)
    ).fit(torch.randn(12, 3))

    assert len(trainer.history) == 2
    assert not any(key.startswith("val_") for record in trainer.history for key in record)
    assert trainer.best_epoch is None
    assert trainer.best_val_loss is None
    assert trainer.stopped_epoch is None


def test_topk_sae_trainer_early_stopping_halts_when_validation_plateaus():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(20, 4)).astype(np.float32)
    trainer = TopKSAETrainer(
        TopKSAEConfig(
            hidden_dim=8,
            k=2,
            batch_size=10,
            epochs=10,
            lr=0.0,  # weights never move, so validation loss is exactly flat
            validation_frac=0.2,
            patience=2,
            show_progress=False,
            seed=7,
        )
    ).fit(x)

    assert trainer.best_epoch == 1
    assert trainer.stopped_epoch == 3
    assert len(trainer.history) == 3


def test_topk_sae_trainer_min_delta_ignores_improvements_below_threshold():
    x = np.random.default_rng(10).normal(size=(12, 3)).astype(np.float32)
    trainer = TopKSAETrainer(
        TopKSAEConfig(
            hidden_dim=6,
            k=2,
            batch_size=6,
            epochs=6,
            validation_frac=0.25,
            patience=2,
            min_delta=0.01,
            restore_best_weights=False,
            show_progress=False,
            seed=41,
        )
    )
    scripted = iter([1.0, 0.999, 0.998])
    trainer._evaluate = lambda _dataset: {"val_loss": next(scripted)}

    trainer.fit(x)

    assert trainer.best_epoch == 1
    assert trainer.best_val_loss == 1.0
    assert trainer.stopped_epoch == 3


def test_topk_sae_trainer_restores_best_epoch_weights():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(16, 4)).astype(np.float32)
    trainer = TopKSAETrainer(
        TopKSAEConfig(
            hidden_dim=8,
            k=2,
            batch_size=8,
            epochs=3,
            lr=0.5,
            validation_frac=0.25,
            restore_best_weights=True,
            show_progress=False,
            seed=5,
        )
    )
    scripted = [1.0, 0.5, 2.0]
    snapshots: list[dict[str, torch.Tensor]] = []

    def fake_evaluate(_dataset):
        snapshots.append(trainer._weight_snapshot())
        return {"val_loss": scripted[len(snapshots) - 1]}

    trainer._evaluate = fake_evaluate
    trainer.fit(x)

    assert trainer.best_epoch == 2
    assert trainer.best_val_loss == 0.5
    assert trainer.stopped_epoch is None
    best = snapshots[1]
    for key, value in trainer.sae.state_dict().items():
        assert torch.equal(value.detach().cpu(), best[key])
    # The final epoch did move the weights, so the restore above is observable.
    assert any(not torch.equal(best[key], snapshots[2][key]) for key in best)


def test_topk_sae_trainer_keeps_last_epoch_weights_when_restore_disabled():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(16, 4)).astype(np.float32)
    trainer = TopKSAETrainer(
        TopKSAEConfig(
            hidden_dim=8,
            k=2,
            batch_size=8,
            epochs=3,
            lr=0.5,
            validation_frac=0.25,
            restore_best_weights=False,
            show_progress=False,
            seed=5,
        )
    )
    scripted = [1.0, 0.5, 2.0]
    snapshots: list[dict[str, torch.Tensor]] = []

    def fake_evaluate(_dataset):
        snapshots.append(trainer._weight_snapshot())
        return {"val_loss": scripted[len(snapshots) - 1]}

    trainer._evaluate = fake_evaluate
    trainer.fit(x)

    assert trainer.best_epoch == 2
    for key, value in trainer.sae.state_dict().items():
        assert torch.equal(value.detach().cpu(), snapshots[2][key])


def test_topk_sae_trainer_eval_step_is_never_corrupted():
    x = torch.randn(8, 4)
    trainer = TopKSAETrainer(
        TopKSAEConfig(
            hidden_dim=8,
            k=2,
            noise_type="gaussian",
            noise_scale="absolute",
            noise_level=50.0,
            show_progress=False,
            seed=11,
        )
    ).build(input_dim=4)

    first = trainer.eval_step(x)["loss"]
    second = trainer.eval_step(x)["loss"]
    with torch.no_grad():
        _reconstruction, _sparse, stats = trainer.sae(x)
    alpha = float(trainer.cfg.alpha_loss)
    expected = alpha * (1.0 - stats["cosine_similarity"]) + (1.0 - alpha) * stats["reconstruction_mse"]

    # Corruption would advance the noise generator and change the loss each call.
    assert torch.equal(first, second)
    assert torch.allclose(first, expected)
    # Guard against the asserts above passing merely because noise is inactive.
    assert not torch.equal(trainer._corrupt(x), x)


def test_topk_sae_trainer_fits_noise_scale_on_train_rows_only():
    rng = np.random.default_rng(6)
    x = torch.as_tensor(rng.normal(size=(20, 3)).astype(np.float32))
    trainer = TopKSAETrainer(
        TopKSAEConfig(
            hidden_dim=6,
            k=2,
            batch_size=8,
            epochs=1,
            noise_type="gaussian",
            noise_scale="feature_std",
            validation_frac=0.25,
            show_progress=False,
            seed=13,
        )
    ).fit(x)

    train_rows, val_rows = trainer._split_validation(x, 0.25)

    assert train_rows.size == 15
    assert val_rows.size == 5
    assert torch.allclose(trainer.input_feature_mean, x[train_rows].mean(dim=0))
    assert not torch.allclose(trainer.input_feature_mean, x.mean(dim=0))


def test_topk_sae_trainer_state_dict_round_trips_early_stopping_fields():
    rng = np.random.default_rng(9)
    x = rng.normal(size=(20, 3)).astype(np.float32)
    trainer = TopKSAETrainer(
        TopKSAEConfig(
            hidden_dim=6,
            k=2,
            batch_size=10,
            epochs=6,
            lr=0.0,
            validation_frac=0.25,
            patience=2,
            show_progress=False,
            seed=31,
        )
    ).fit(x)
    state = trainer.state_dict()

    assert state["format_version"] == 4
    restored = TopKSAETrainer.from_state_dict(state)

    assert restored.best_epoch == trainer.best_epoch
    assert restored.best_val_loss == trainer.best_val_loss
    assert restored.stopped_epoch == trainer.stopped_epoch


@requires_accelerator
def test_embeddings_dataset_device_to_host_batches_are_synchronized():
    device = _accelerator()
    rows, dim, batch_size = 256, 4, 32
    source = torch.arange(rows * dim, dtype=torch.float32, device=device).reshape(rows, dim)
    expected = source.cpu()
    data = EmbeddingsDataset(source, batch_size=batch_size, shuffle=False, device="cpu")

    for batch_idx, batch in enumerate(data):
        start = batch_idx * batch_size
        assert batch.device.type == "cpu"
        assert torch.equal(batch, expected[start : start + batch_size])


@requires_accelerator
def test_topk_sae_trainer_fits_from_accelerator_embeddings_with_validation():
    rng = np.random.default_rng(0)
    x = torch.as_tensor(rng.normal(size=(120, 8)).astype(np.float32), device=_accelerator())
    trainer = TopKSAETrainer(
        TopKSAEConfig(
            hidden_dim=16,
            k=3,
            batch_size=40,
            epochs=4,
            validation_frac=0.25,
            device="cpu",
            show_progress=False,
            seed=3,
        )
    ).fit(x)

    assert len(trainer.history) == 4
    for record in trainer.history:
        assert np.isfinite(record["loss"])
        assert np.isfinite(record["val_loss"])
