from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from compresso import TopKSAEConfig, TopKSAETrainer


def _config(**overrides) -> TopKSAEConfig:
    values = dict(
        hidden_dim=8,
        k=3,
        batch_size=4,
        epochs=2,
        show_progress=False,
        seed=11,
    )
    values.update(overrides)
    return TopKSAEConfig(**values)


def _embeddings(rows: int = 16, cols: int = 5, *, seed: int = 0) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    # Deliberately anisotropic: per-feature means and scales differ a lot, so
    # standardizing is not close to a no-op.
    scales = np.array([0.01, 1.0, 10.0, 100.0, 3.0])[:cols]
    offsets = np.array([-5.0, 0.0, 2.0, 50.0, 1.0])[:cols]
    values = rng.normal(size=(rows, cols)) * scales + offsets
    return torch.from_numpy(values.astype(np.float32))


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


def test_standard_scaling_is_off_by_default():
    config = TopKSAEConfig()

    assert config.standard_scaler_mean is False
    assert config.standard_scaler_std is False
    assert config.standard_scaler_loss_space == "original"


@pytest.mark.parametrize(
    ("mean", "std"),
    [(True, False), (False, True), (True, True)],
)
def test_adaptive_noise_is_rejected_with_standard_scaling(mean, std):
    with pytest.raises(ValueError, match="adaptive noise scales"):
        TopKSAETrainer(
            _config(
                standard_scaler_mean=mean,
                standard_scaler_std=std,
                noise_type="gaussian",
                noise_scale="global_rms",
            )
        ).build(input_dim=5)


def test_absolute_noise_is_allowed_with_standard_scaling():
    trainer = TopKSAETrainer(
        _config(
            standard_scaler_std=True,
            noise_type="gaussian",
            noise_scale="absolute",
            noise_level=0.1,
        )
    ).build(input_dim=5)

    assert trainer.is_built


def test_adaptive_noise_scale_is_allowed_while_scaling_is_off():
    """noise_scale defaults to global_rms, so the check must gate on noise_type."""
    trainer = TopKSAETrainer(_config(noise_scale="global_rms")).build(input_dim=5)

    assert trainer.is_built


def test_loss_space_requires_standard_scaling():
    with pytest.raises(ValueError, match="standard_scaler_loss_space requires"):
        TopKSAETrainer(_config(standard_scaler_loss_space="scaled")).build(input_dim=5)


def test_unknown_loss_space_is_rejected():
    with pytest.raises(ValueError, match="unknown standard_scaler_loss_space"):
        TopKSAETrainer(
            _config(standard_scaler_mean=True, standard_scaler_loss_space="raw")
        ).build(input_dim=5)


# --------------------------------------------------------------------------- #
# Fitted statistics
# --------------------------------------------------------------------------- #


def test_scaler_matches_sklearn_conventions():
    x = torch.tensor([[1.0, 2.0], [3.0, 2.0]])
    trainer = TopKSAETrainer(
        _config(standard_scaler_mean=True, standard_scaler_std=True)
    ).build(input_dim=2)

    trainer._fit_standard_scaler(x)

    # ddof=0, as sklearn uses; the constant second feature keeps a scale of 1.
    assert torch.equal(trainer.input_scaler_mean, torch.tensor([2.0, 2.0]))
    assert torch.equal(trainer.input_scaler_scale, torch.tensor([1.0, 1.0]))


def test_scaler_std_is_population_not_sample():
    x = torch.tensor([[0.0], [2.0], [4.0]])
    trainer = TopKSAETrainer(_config(standard_scaler_std=True)).build(input_dim=1)

    trainer._fit_standard_scaler(x)

    # Population std of {0, 2, 4} is sqrt(8/3); the sample std would be 2.
    assert torch.allclose(trainer.input_scaler_scale, torch.tensor([(8 / 3) ** 0.5]))


@pytest.mark.parametrize(
    ("mean", "std"),
    [(True, False), (False, True), (True, True)],
)
def test_only_requested_statistics_are_fitted(mean, std):
    trainer = TopKSAETrainer(
        _config(standard_scaler_mean=mean, standard_scaler_std=std)
    ).build(input_dim=5)

    trainer._fit_standard_scaler(_embeddings())

    assert (trainer.input_scaler_mean is not None) is mean
    assert (trainer.input_scaler_scale is not None) is std


def test_standardized_training_rows_have_zero_mean_and_unit_variance():
    x = _embeddings()
    trainer = TopKSAETrainer(
        _config(standard_scaler_mean=True, standard_scaler_std=True)
    ).build(input_dim=5)
    trainer._fit_standard_scaler(x)

    scaled = trainer._standardize(x)
    variance, mean = torch.var_mean(scaled, dim=0, correction=0)

    # The first feature has mean/std around 500, so float32 centering leaves a
    # residual near 1e-5 there. Unit variance is unaffected by that shift.
    assert torch.allclose(mean, torch.zeros(5), atol=1e-4)
    assert torch.allclose(variance, torch.ones(5), atol=1e-5)


def test_standardize_round_trips_through_destandardize():
    x = _embeddings()
    trainer = TopKSAETrainer(
        _config(standard_scaler_mean=True, standard_scaler_std=True)
    ).build(input_dim=5)
    trainer._fit_standard_scaler(x)

    assert torch.allclose(trainer._destandardize(trainer._standardize(x)), x, atol=1e-3)


def test_scaler_is_fitted_on_training_rows_only():
    """Validation rows must not leak into the statistics."""
    x = _embeddings(rows=20)
    trainer = TopKSAETrainer(
        _config(standard_scaler_mean=True, standard_scaler_std=True, validation_frac=0.25)
    )
    trainer.fit(x)

    train_rows, _val_rows = trainer._resolve_validation(x, None, 5)
    expected_variance, expected_mean = torch.var_mean(
        torch.as_tensor(train_rows), dim=0, correction=0
    )

    assert torch.allclose(trainer.input_scaler_mean, expected_mean, atol=1e-5)
    assert torch.allclose(trainer.input_scaler_scale, expected_variance.sqrt(), atol=1e-5)
    # The statistics over every row would differ from the training-only ones.
    all_variance, all_mean = torch.var_mean(x, dim=0, correction=0)
    assert not torch.allclose(trainer.input_scaler_mean, all_mean, atol=1e-5)


def test_scaling_without_fit_is_rejected():
    trainer = TopKSAETrainer(_config(standard_scaler_std=True)).build(input_dim=5)

    with pytest.raises(RuntimeError, match="requires fit\\(\\) before train_step\\(\\)"):
        trainer.train_step(_embeddings(rows=4))


# --------------------------------------------------------------------------- #
# Loss space
# --------------------------------------------------------------------------- #


def test_original_loss_space_measures_against_raw_inputs():
    x = _embeddings(rows=8)
    trainer = TopKSAETrainer(
        _config(standard_scaler_mean=True, standard_scaler_std=True, epochs=1)
    )
    trainer.fit(x)

    batch = x[:4]
    with torch.no_grad():
        reconstruction, _sparse, _stats = trainer.sae(trainer._standardize(batch))
        prediction = trainer._destandardize(reconstruction)
        expected_mse = F.mse_loss(prediction, batch)

    stats = trainer.eval_step(batch)

    assert torch.allclose(stats["reconstruction_mse"], expected_mse)


def test_scaled_loss_space_measures_against_standardized_inputs():
    x = _embeddings(rows=8)
    trainer = TopKSAETrainer(
        _config(
            standard_scaler_mean=True,
            standard_scaler_std=True,
            standard_scaler_loss_space="scaled",
            epochs=1,
        )
    )
    trainer.fit(x)

    batch = x[:4]
    with torch.no_grad():
        scaled = trainer._standardize(batch)
        reconstruction, _sparse, _stats = trainer.sae(scaled)
        expected_mse = F.mse_loss(reconstruction, scaled)

    stats = trainer.eval_step(batch)

    assert torch.allclose(stats["reconstruction_mse"], expected_mse)


def test_loss_spaces_disagree_on_anisotropic_data():
    """The two spaces differ by a per-feature variance weighting."""
    x = _embeddings(rows=12)
    common = dict(standard_scaler_mean=True, standard_scaler_std=True, epochs=1)

    original = TopKSAETrainer(_config(**common)).fit(x)
    scaled = TopKSAETrainer(_config(**common, standard_scaler_loss_space="scaled")).fit(x)

    assert not np.isclose(
        original.history[-1]["reconstruction_mse"],
        scaled.history[-1]["reconstruction_mse"],
    )


def test_centering_only_leaves_mse_unchanged_between_loss_spaces():
    """MSE is translation invariant, so only the cosine term can move."""
    x = _embeddings(rows=12)
    batch = x[:4]
    common = dict(standard_scaler_mean=True, standard_scaler_std=False, epochs=1)

    original = TopKSAETrainer(_config(**common)).fit(x)
    scaled = TopKSAETrainer(_config(**common, standard_scaler_loss_space="scaled")).fit(x)
    scaled.sae.load_state_dict(original.sae.state_dict())
    scaled.input_scaler_mean = original.input_scaler_mean
    scaled._cache_scaler_tensors()

    assert torch.allclose(
        original.eval_step(batch)["reconstruction_mse"],
        scaled.eval_step(batch)["reconstruction_mse"],
        atol=1e-6,
    )


# --------------------------------------------------------------------------- #
# Interaction with the rest of the trainer
# --------------------------------------------------------------------------- #


def test_scaling_off_is_bit_identical_to_before():
    """The default path must not change: same seed, same history."""
    x = _embeddings(rows=12)

    first = TopKSAETrainer(_config()).fit(x)
    second = TopKSAETrainer(_config(standard_scaler_mean=False, standard_scaler_std=False)).fit(x)

    assert first.history == second.history


def test_reconstruct_returns_the_original_embedding_space():
    x = _embeddings(rows=12)
    trainer = TopKSAETrainer(
        _config(standard_scaler_mean=True, standard_scaler_std=True)
    ).fit(x)

    reconstruction = trainer.reconstruct(x)

    assert reconstruction.shape == x.shape
    # Raw embeddings sit far from zero on some features; a standardized-space
    # reconstruction would be near zero instead.
    assert reconstruction.abs().max() > 1.0
    assert torch.isfinite(reconstruction).all()


def test_noise_is_injected_after_standardization():
    x = _embeddings(rows=8)
    trainer = TopKSAETrainer(
        _config(
            standard_scaler_mean=True,
            standard_scaler_std=True,
            noise_type="gaussian",
            noise_scale="absolute",
            noise_level=0.25,
            epochs=1,
        )
    ).fit(x)

    seen: list[torch.Tensor] = []
    original_corrupt = trainer._corrupt

    def spy(batch):
        seen.append(batch.clone())
        return original_corrupt(batch)

    trainer._corrupt = spy
    trainer.train_step(x[:4])

    assert len(seen) == 1
    variance, mean = torch.var_mean(seen[0], dim=0, correction=0)
    # _corrupt sees standardized rows, not raw ones.
    assert mean.abs().max() < 1.0
    assert variance.max() < 5.0


def test_state_dict_round_trips_the_scaler():
    x = _embeddings(rows=12)
    trainer = TopKSAETrainer(
        _config(standard_scaler_mean=True, standard_scaler_std=True)
    ).fit(x)
    state = trainer.state_dict()

    assert state["format_version"] == 4
    assert torch.equal(state["input_scaler_mean"], trainer.input_scaler_mean)
    assert torch.equal(state["input_scaler_scale"], trainer.input_scaler_scale)

    restored = TopKSAETrainer.from_state_dict(state)

    assert torch.equal(restored.input_scaler_mean, trainer.input_scaler_mean)
    assert torch.equal(restored.input_scaler_scale, trainer.input_scaler_scale)
    assert torch.allclose(restored.reconstruct(x), trainer.reconstruct(x), atol=1e-5)


def test_older_state_dicts_load_without_a_scaler():
    x = _embeddings(rows=12)
    trainer = TopKSAETrainer(_config()).fit(x)
    state = trainer.state_dict()
    state["format_version"] = 3
    del state["input_scaler_mean"]
    del state["input_scaler_scale"]

    restored = TopKSAETrainer.from_state_dict(state)

    assert restored.input_scaler_mean is None
    assert restored.input_scaler_scale is None
    assert torch.allclose(restored.reconstruct(x), trainer.reconstruct(x), atol=1e-5)
