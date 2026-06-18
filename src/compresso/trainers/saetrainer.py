"""Training utilities for :class:`compresso.nn.TopKSAE`.

The objects in this module provide a small, sklearn-like API around the
low-level ``TopKSAE`` module:

>>> trainer = TopKSAETrainer(TopKSAEConfig(k=32, epochs=100))
>>> srp = trainer.fit_transform(embeddings)

The trainer intentionally optimizes for dense embedding matrices that already
fit in memory. It avoids ``torch.utils.data.DataLoader`` overhead and uses a
simple batch dataset that returns full batches directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from compresso.nn.sae import TopKSAE
from compresso.params.srp import SRPTensor

__all__ = [
    "EmbeddingsDataset",
    "L1Normalize",
    "L2Normalize",
    "TopKSAEConfig",
    "TopKSAETrainer",
]


class EmbeddingsDataset:
    """Small batch-oriented dataset for in-memory embedding matrices.

    Unlike ``torch.utils.data.Dataset``, ``__getitem__`` returns a complete
    batch, not one sample. This mirrors Keras ``PyDataset`` ergonomics and keeps
    the training loop tight for matrix-shaped embedding data.

    Parameters
    ----------
    embeddings:
        A 2D ``numpy.ndarray`` or ``torch.Tensor`` with shape ``(n, dim)``.
    batch_size:
        Number of rows returned by each batch.
    shuffle:
        Whether to shuffle row order when ``on_epoch_end`` is called.
    seed:
        Seed for the NumPy row-order generator.
    device:
        Device where returned batches should live.
    dtype:
        Optional dtype conversion for returned batches. ``None`` preserves the
        dtype from the input tensor/array as much as possible.
    """

    def __init__(
        self,
        embeddings: np.ndarray | torch.Tensor,
        *,
        batch_size: int = 128,
        shuffle: bool = True,
        seed: int = 42,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        tensor = torch.as_tensor(embeddings)
        if tensor.ndim != 2:
            raise ValueError(f"embeddings must be 2D, got shape {tuple(tensor.shape)}")
        if not torch.is_floating_point(tensor):
            tensor = tensor.float()
        if dtype is not None:
            tensor = tensor.to(dtype=dtype)

        self.embeddings = tensor.contiguous()
        self.n, self.dim = int(tensor.shape[0]), int(tensor.shape[1])
        self.indices = np.arange(self.n)
        self.rng = np.random.default_rng(seed)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.device = torch.device(device)

    def __len__(self) -> int:
        """Return the number of batches."""
        return int(np.ceil(self.n / self.batch_size))

    def __iter__(self):
        for batch_idx in range(len(self)):
            yield self[batch_idx]

    def __getitem__(self, batch_idx: int) -> torch.Tensor:
        """Return batch ``batch_idx`` as a tensor on ``self.device``."""
        start = int(batch_idx) * self.batch_size
        end = min(start + self.batch_size, self.n)
        rows = self.indices[start:end]
        batch = self.embeddings[torch.as_tensor(rows, dtype=torch.long)]
        return batch.to(self.device, non_blocking=True)

    def to(self, device: str | torch.device) -> "EmbeddingsDataset":
        """Set output device for future batches and return ``self``."""
        device = torch.device(device)
        # Probe once so invalid devices fail early.
        self.embeddings[:1].to(device)
        self.device = device
        return self

    def on_epoch_begin(self) -> None:
        """Hook called by ``TopKSAETrainer.fit`` at the beginning of an epoch."""

    def on_epoch_end(self) -> None:
        """Shuffle row order after each epoch when ``shuffle=True``."""
        if self.shuffle:
            self.rng.shuffle(self.indices)


class L1Normalize(nn.Module):
    """Apply row-wise L1 normalization."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, p=1.0, dim=-1)


class L2Normalize(nn.Module):
    """Apply row-wise L2 normalization."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, p=2.0, dim=-1)


@dataclass(frozen=True)
class TopKSAEConfig:
    """Configuration for :class:`TopKSAETrainer`.

    Parameters
    ----------
    hidden_dim:
        Width of the SAE code layer.
    k:
        Number of active code features per row.
    decoder_bias:
        Whether the default decoder linear layer uses a bias.
    pre_act:
        Optional module applied to encoder output before sparsification.
    post_sparsify:
        Optional module applied to sparse codes after top-k. For example,
        ``L1Normalize()``.
    encoder, decoder:
        Optional custom modules. When omitted, ``TopKSAE`` uses linear encoder
        and decoder layers.
    sparsify_score_mode:
        Top-k scoring mode: ``"abs"``, ``"raw"``, or ``"relu"``.
    sparsify_ste_alpha:
        Straight-through estimator leakage for non-selected positions.
    alpha_loss:
        Mixture weight for cosine loss. Training loss is
        ``alpha_loss * (1 - cosine_similarity) + (1 - alpha_loss) * mse``.
    l1_penalty:
        Optional penalty on mean absolute sparse code activation.
    batch_size:
        Number of embedding rows per training batch.
    shuffle:
        Whether to shuffle training rows between epochs.
    seed:
        Random seed used for row shuffling and Torch initialization.
    epochs:
        Number of training epochs.
    lr, weight_decay:
        AdamW optimizer parameters.
    compile:
        If ``True``, call ``torch.compile`` on the SAE when available.
    device:
        Device used for training and transforms.
    show_progress:
        Whether to show a tqdm progress bar when tqdm is installed.
    srp_score_mode:
        Score mode used by ``SRPTensor.from_dense`` during ``transform``.
    """

    hidden_dim: int = 4096
    k: int = 128
    decoder_bias: bool = False
    pre_act: nn.Module | None = None
    post_sparsify: nn.Module | None = None
    encoder: nn.Module | None = None
    decoder: nn.Module | None = None
    sparsify_score_mode: Literal["abs", "raw", "relu"] = "abs"
    sparsify_ste_alpha: float = 0.01
    alpha_loss: float = 0.01
    l1_penalty: float = 0.0
    batch_size: int = 128
    shuffle: bool = True
    seed: int = 42
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 0.0
    compile: bool = False
    device: str | torch.device = "cpu"
    show_progress: bool = True
    srp_score_mode: Literal["abs", "raw", "relu"] = "abs"


class TopKSAETrainer:
    """Efficient fit/transform wrapper around :class:`compresso.TopKSAE`.

    The trainer is intended for dense embedding matrices, such as item
    embeddings from a recommender or semantic embeddings from a text encoder.
    It exposes a compact sklearn-like API:

    >>> trainer = TopKSAETrainer(TopKSAEConfig(k=32, epochs=300))
    >>> trainer.fit(embeddings)
    >>> sparse = trainer.transform(embeddings)

    ``transform`` returns an ``SRPTensor`` containing sparse codes. Use
    ``reconstruct`` if dense reconstructions are needed.
    """

    def __init__(self, config: TopKSAEConfig | None = None) -> None:
        self.cfg = config if config is not None else TopKSAEConfig()
        self.device = torch.device(self.cfg.device)
        self.sae: TopKSAE | nn.Module | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.input_dim: int | None = None
        self.history: list[dict[str, float]] = []

    @property
    def is_built(self) -> bool:
        """Whether the underlying ``TopKSAE`` model has been initialized."""
        return self.sae is not None

    def build(self, input_dim: int) -> "TopKSAETrainer":
        """Initialize model and optimizer for inputs of size ``input_dim``."""
        if self.is_built:
            if int(input_dim) != self.input_dim:
                raise ValueError(f"trainer is already built for input_dim={self.input_dim}, got {input_dim}")
            return self
        if input_dim < 1:
            raise ValueError("input_dim must be >= 1")
        if self.cfg.hidden_dim < 1:
            raise ValueError("hidden_dim must be >= 1")
        if not 1 <= self.cfg.k <= self.cfg.hidden_dim:
            raise ValueError(f"k must be in [1, hidden_dim], got k={self.cfg.k}, hidden_dim={self.cfg.hidden_dim}")

        torch.manual_seed(int(self.cfg.seed))
        self.input_dim = int(input_dim)
        model = TopKSAE(
            input_dim=self.input_dim,
            hidden_dim=int(self.cfg.hidden_dim),
            k=int(self.cfg.k),
            decoder_bias=bool(self.cfg.decoder_bias),
            pre_act=self.cfg.pre_act,
            post_sparsify=self.cfg.post_sparsify,
            encoder=self.cfg.encoder,
            decoder=self.cfg.decoder,
            sparsify_score_mode=self.cfg.sparsify_score_mode,
            sparsify_ste_alpha=float(self.cfg.sparsify_ste_alpha),
        ).to(self.device)
        if self.cfg.compile:
            model = torch.compile(model)  # type: ignore[assignment]
        self.sae = model
        self.optimizer = torch.optim.AdamW(
            self.sae.parameters(),
            lr=float(self.cfg.lr),
            weight_decay=float(self.cfg.weight_decay),
        )
        return self

    def to(self, device: str | torch.device) -> "TopKSAETrainer":
        """Move the underlying model to ``device`` and return ``self``."""
        self.device = torch.device(device)
        if self.sae is not None:
            self.sae.to(self.device)
        return self

    def _dataset(self, embeddings: np.ndarray | torch.Tensor, *, shuffle: bool) -> EmbeddingsDataset:
        return EmbeddingsDataset(
            embeddings,
            batch_size=int(self.cfg.batch_size),
            shuffle=shuffle,
            seed=int(self.cfg.seed),
            device=self.device,
        )

    def _progress(self, iterable, *, total: int | None = None):
        if not self.cfg.show_progress:
            return iterable
        try:
            from tqdm.auto import tqdm
        except Exception:  # pragma: no cover - optional dependency fallback
            return iterable
        return tqdm(iterable, total=total)

    def train_step(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run one optimization step and return detached training stats."""
        if self.sae is None or self.optimizer is None:
            raise RuntimeError("trainer must be built before train_step")
        self.sae.train()
        self.optimizer.zero_grad(set_to_none=True)
        _reconstruction, sparse, stats = self.sae(batch)
        cosine_loss = 1.0 - stats["cosine_similarity"]
        mse = stats["reconstruction_mse"]
        loss = float(self.cfg.alpha_loss) * cosine_loss + (1.0 - float(self.cfg.alpha_loss)) * mse
        if self.cfg.l1_penalty > 0.0:
            loss = loss + float(self.cfg.l1_penalty) * sparse.abs().mean()
        loss.backward()
        self.optimizer.step()
        return {
            "loss": loss.detach(),
            "cosine_loss": cosine_loss.detach(),
            "reconstruction_mse": mse.detach(),
            "active_count": stats["active_count"].detach(),
            "dead_features": stats["dead_features"].detach(),
        }

    def fit(self, embeddings: np.ndarray | torch.Tensor) -> "TopKSAETrainer":
        """Train the SAE on dense embeddings and return ``self``."""
        dataset = self._dataset(embeddings, shuffle=bool(self.cfg.shuffle))
        self.build(dataset.dim)
        epoch_iter = self._progress(range(1, int(self.cfg.epochs) + 1), total=int(self.cfg.epochs))
        for epoch in epoch_iter:
            dataset.on_epoch_begin()
            sums: dict[str, float] = {}
            n_batches = 0
            for batch in dataset:
                stats = self.train_step(batch)
                for key, value in stats.items():
                    sums[key] = sums.get(key, 0.0) + float(value.detach().cpu().item())
                n_batches += 1
            dataset.on_epoch_end()
            record = {key: value / max(1, n_batches) for key, value in sums.items()}
            record["epoch"] = float(epoch)
            self.history.append(record)
            if hasattr(epoch_iter, "set_description"):
                epoch_iter.set_description(
                    f"LOSS: cosine {record['cosine_loss']:.4f}, "
                    f"mse {record['reconstruction_mse']:.4E}, PROGRESS"
                )
        return self

    @torch.no_grad()
    def encode(self, embeddings: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Return dense sparse-code tensor produced by the trained SAE."""
        if self.sae is None:
            raise RuntimeError("trainer must be fitted or built before encode")
        dataset = self._dataset(embeddings, shuffle=False)
        self.sae.eval()
        codes: list[torch.Tensor] = []
        for batch in self._progress(dataset, total=len(dataset)):
            _reconstruction, sparse, _stats = self.sae(batch)
            codes.append(sparse.detach().cpu())
        return torch.cat(codes, dim=0)

    @torch.no_grad()
    def reconstruct(self, embeddings: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Return dense reconstructions for ``embeddings``."""
        if self.sae is None:
            raise RuntimeError("trainer must be fitted or built before reconstruct")
        dataset = self._dataset(embeddings, shuffle=False)
        self.sae.eval()
        reconstructions: list[torch.Tensor] = []
        for batch in self._progress(dataset, total=len(dataset)):
            reconstruction, _sparse, _stats = self.sae(batch)
            reconstructions.append(reconstruction.detach().cpu())
        return torch.cat(reconstructions, dim=0)

    @torch.no_grad()
    def transform(self, embeddings: np.ndarray | torch.Tensor) -> SRPTensor:
        """Encode ``embeddings`` and return sparse codes as an ``SRPTensor``."""
        codes = self.encode(embeddings)
        return SRPTensor.from_dense(codes, k=int(self.cfg.k), score_mode=self.cfg.srp_score_mode)

    def fit_transform(self, embeddings: np.ndarray | torch.Tensor) -> SRPTensor:
        """Fit the SAE and return encoded sparse codes as an ``SRPTensor``."""
        self.fit(embeddings)
        return self.transform(embeddings)

    def state_dict(self) -> dict[str, Any]:  # type: ignore[override]
        """Return a saveable trainer state dictionary."""
        if self.sae is None:
            raise RuntimeError("trainer must be built before state_dict")
        return {
            "config": self.cfg,
            "input_dim": self.input_dim,
            "model": self.sae.state_dict(),
            "optimizer": self.optimizer.state_dict() if self.optimizer is not None else None,
            "history": list(self.history),
        }
