from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from compresso import L1Normalize, SRPTensor, TopKSAEConfig, TopKSAETrainer
from compresso.trainers import EmbeddingsDataset


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
