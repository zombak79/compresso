from __future__ import annotations

import numpy as np
import torch

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
