from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from compresso.nn import TopKSAE


@dataclass
class StageArtifacts:
    item_embeddings_path: Path


class TwoStagePipeline:
    """Thin scaffold: stage1 provides item embeddings, stage2 runs SAE."""

    def __init__(self, workdir: str | Path = "artifacts") -> None:
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)

    def save_item_embeddings(self, embeddings: np.ndarray, name: str = "item_embeddings.npy") -> StageArtifacts:
        path = self.workdir / name
        np.save(path, embeddings)
        return StageArtifacts(item_embeddings_path=path)

    def train_sae_on_embeddings(
        self,
        embeddings: np.ndarray,
        *,
        hidden_dim: int,
        k: int,
        k_backward: int | None = None,
        sparsify_mode: str = "values",
        sparsify_score_mode: str = "abs",
        epochs: int = 5,
        batch_size: int = 256,
        lr: float = 1e-3,
        device: str = "cpu",
        loss_type: str = "mse",
        log_every_epoch: bool = True,
        val_callback=None,
    ) -> TopKSAE:
        if loss_type not in {"mse", "cosine"}:
            raise ValueError("loss_type must be 'mse' or 'cosine'")

        model = TopKSAE(
            input_dim=embeddings.shape[1],
            hidden_dim=hidden_dim,
            k=k,
            sparsify_mode=sparsify_mode,
            sparsify_score_mode=sparsify_score_mode,
            k_backward=k_backward,
        ).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)

        x = torch.from_numpy(embeddings.astype(np.float32))
        loader = torch.utils.data.DataLoader(x, batch_size=batch_size, shuffle=True)

        model.train()
        for epoch in range(1, epochs + 1):
            running_loss = 0.0
            n_batches = 0
            for xb in loader:
                xb = xb.to(device)
                recon, _, stats = model(xb)
                if loss_type == "mse":
                    loss = stats["reconstruction_mse"]
                else:
                    xb_flat = xb.reshape(xb.shape[0], -1)
                    recon_flat = recon.reshape(recon.shape[0], -1)
                    loss = (1.0 - torch.nn.functional.cosine_similarity(xb_flat, recon_flat, dim=-1)).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                running_loss += float(loss.item())
                n_batches += 1
            if log_every_epoch:
                avg_loss = running_loss / max(1, n_batches)
                msg = f"[SAE] epoch={epoch}/{epochs} {loss_type}={avg_loss:.6f}"
                if val_callback is not None:
                    val_metrics = val_callback(model)
                    msg += (
                        f" val_recall@20={val_metrics.get('recall@20', 0.0):.6f}"
                        f" val_recall@50={val_metrics.get('recall@50', 0.0):.6f}"
                        f" val_ndcg@100={val_metrics.get('ndcg@100', 0.0):.6f}"
                    )
                print(msg)

        return model
