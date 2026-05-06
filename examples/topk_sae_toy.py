#!/usr/bin/env python
"""Toy example: train a TopKSAE on synthetic embeddings.

Usage::

    python examples/topk_sae_toy.py

No external dependencies beyond PyTorch.
"""

import torch
from compresso.nn import TopKSAE

# ── Config ──────────────────────────────────────────────────────────────
SEED = 42
N_SAMPLES = 1024
INPUT_DIM = 128
HIDDEN_DIM = 512
K = 32
EPOCHS = 50
BATCH_SIZE = 128
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    torch.manual_seed(SEED)
    device = torch.device(DEVICE)

    # ── Synthetic data ──────────────────────────────────────────────────
    data = torch.randn(N_SAMPLES, INPUT_DIM, device=device)

    # ── Model ───────────────────────────────────────────────────────────
    model = TopKSAE(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        k=K,
        tied=False,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # ── Training loop ───────────────────────────────────────────────────
    n_batches = (N_SAMPLES + BATCH_SIZE - 1) // BATCH_SIZE
    for epoch in range(1, EPOCHS + 1):
        perm = torch.randperm(N_SAMPLES, device=device)
        epoch_loss = 0.0
        for i in range(n_batches):
            batch = data[perm[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]]
            recon, codes, stats = model(batch)
            loss = stats["reconstruction_mse"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= n_batches
        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d}/{EPOCHS}  "
                f"MSE={epoch_loss:.6f}  "
                f"cos_sim={stats['cosine_similarity'].item():.4f}  "
                f"active={stats['active_count'].item():.1f}  "
                f"dead={stats['dead_features'].item():.0f}/{HIDDEN_DIM}"
            )

    # ── Final evaluation ────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        recon, codes, stats = model(data)

    print("\n── Final stats ──")
    print(f"  Reconstruction MSE : {stats['reconstruction_mse'].item():.6f}")
    print(f"  Cosine similarity  : {stats['cosine_similarity'].item():.4f}")
    print(f"  Active features    : {stats['active_count'].item():.1f} / {K}")
    print(f"  Dead features      : {stats['dead_features'].item():.0f} / {HIDDEN_DIM}")
    print(f"  Feature freq range : [{stats['activation_freq'].min().item():.4f}, "
          f"{stats['activation_freq'].max().item():.4f}]")


if __name__ == "__main__":
    main()
