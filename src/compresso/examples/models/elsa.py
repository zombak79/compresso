from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix


class TorchELSA(nn.Module):
    """Thin ELSA-style linear autoencoder for item embedding learning.

    Uses normalized item embedding matrix A and predicts:
        y = relu((x @ A) @ A.T - x)
    where x is a user interaction vector.
    """

    def __init__(self, n_items: int, n_factors: int) -> None:
        super().__init__()
        self.n_items = n_items
        self.n_factors = n_factors
        self.A = nn.Parameter(torch.empty(n_items, n_factors))
        nn.init.xavier_uniform_(self.A)

    def normalized_A(self) -> torch.Tensor:
        return F.normalize(self.A, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.normalized_A()
        x_a = x @ a
        x_aat = x_a @ a.T
        return F.relu(x_aat - x)

    @torch.no_grad()
    def export_item_embeddings(self) -> np.ndarray:
        return self.normalized_A().detach().cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def predict_scores(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return self(x)


def _iter_csr_batches(x: csr_matrix, batch_size: int):
    n = x.shape[0]
    for i in range(0, n, batch_size):
        m = x[i : i + batch_size]
        yield torch.from_numpy(m.toarray().astype(np.float32))


def fit_elsa(
    x_train: csr_matrix,
    *,
    n_factors: int,
    epochs: int = 10,
    batch_size: int = 512,
    lr: float = 1e-2,
    weight_decay: float = 0.0,
    device: str = "cpu",
    log_every_epoch: bool = True,
    val_callback=None,
) -> TorchELSA:
    model = TorchELSA(n_items=x_train.shape[1], n_factors=n_factors).to(device)
    opt = torch.optim.NAdam(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.train()
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        n_batches = 0
        for xb in _iter_csr_batches(x_train, batch_size=batch_size):
            xb = xb.to(device)
            y = model(xb)
            # Normalized MSE objective to match ELSA-like behavior.
            loss = F.mse_loss(F.normalize(y, dim=-1), F.normalize(xb, dim=-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            running_loss += float(loss.item())
            n_batches += 1
        if log_every_epoch:
            avg_loss = running_loss / max(1, n_batches)
            msg = f"[ELSA] epoch={epoch}/{epochs} loss={avg_loss:.6f}"
            if val_callback is not None:
                val_metrics = val_callback(model)
                msg += (
                    f" val_recall@20={val_metrics.get('recall@20', 0.0):.6f}"
                    f" val_recall@50={val_metrics.get('recall@50', 0.0):.6f}"
                    f" val_ndcg@100={val_metrics.get('ndcg@100', 0.0):.6f}"
                )
            print(msg)

    return model
