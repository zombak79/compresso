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

import copy
import math
import warnings
from dataclasses import dataclass, replace
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
        """Normalize each row by its L1 norm.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor whose last dimension is normalized.

        Returns
        -------
        torch.Tensor
            Tensor with the same shape as ``x`` and unit L1 norm along the
            last dimension where possible.
        """
        return F.normalize(x, p=1.0, dim=-1)


class L2Normalize(nn.Module):
    """Apply row-wise L2 normalization."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize each row by its L2 norm.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor whose last dimension is normalized.

        Returns
        -------
        torch.Tensor
            Tensor with the same shape as ``x`` and unit L2 norm along the
            last dimension where possible.
        """
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
    noise_type:
        Optional corruption applied to training inputs. ``"none"`` leaves
        inputs unchanged and ``"gaussian"`` adds Gaussian noise.
    noise_scale:
        Scaling used for Gaussian noise. ``"absolute"`` uses embedding
        coordinate units, ``"global_rms"`` uses one training-set-derived
        scale, and ``"feature_std"`` scales each input feature separately.
    noise_level:
        Gaussian standard deviation for absolute scaling, or a dimensionless
        multiplier for adaptive scaling.
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
    decay:
        If ``True``, use cosine learning-rate decay from ``lr`` to zero across
        the configured training epochs.
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
    noise_type: Literal["none", "gaussian"] = "none"
    noise_scale: Literal["absolute", "global_rms", "feature_std"] = "global_rms"
    noise_level: float = 0.1
    alpha_loss: float = 0.01
    l1_penalty: float = 0.0
    batch_size: int = 128
    shuffle: bool = True
    seed: int = 42
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 0.0
    decay: bool = False
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
        self.input_feature_mean: torch.Tensor | None = None
        self.input_feature_variance: torch.Tensor | None = None
        self._gaussian_noise_scale: torch.Tensor | None = None
        self._noise_generator: torch.Generator | None = None

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
        if not math.isfinite(float(self.cfg.alpha_loss)) or not 0.0 <= self.cfg.alpha_loss <= 1.0:
            raise ValueError("alpha_loss must be finite and in [0, 1]")
        if not math.isfinite(float(self.cfg.l1_penalty)) or self.cfg.l1_penalty < 0.0:
            raise ValueError("l1_penalty must be finite and >= 0")
        if self.cfg.noise_type not in {"none", "gaussian"}:
            raise ValueError(f"unknown noise_type: {self.cfg.noise_type}")
        if self.cfg.noise_scale not in {"absolute", "global_rms", "feature_std"}:
            raise ValueError(f"unknown noise_scale: {self.cfg.noise_scale}")
        if not math.isfinite(float(self.cfg.noise_level)) or self.cfg.noise_level < 0.0:
            raise ValueError("noise_level must be finite and >= 0")

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
        if self._gaussian_noise_scale is not None:
            self._gaussian_noise_scale = self._gaussian_noise_scale.to(self.device)
        return self

    def _model_dtype(self) -> torch.dtype:
        if self.sae is not None:
            for tensor in (*self.sae.parameters(), *self.sae.buffers()):
                if torch.is_floating_point(tensor):
                    return tensor.dtype
        return torch.get_default_dtype()

    @staticmethod
    def _input_dim(embeddings: np.ndarray | torch.Tensor) -> int:
        tensor = torch.as_tensor(embeddings)
        if tensor.ndim != 2:
            raise ValueError(f"embeddings must be 2D, got shape {tuple(tensor.shape)}")
        if tensor.shape[0] < 1:
            raise ValueError("embeddings must contain at least one row")
        return int(tensor.shape[1])

    def _dataset(self, embeddings: np.ndarray | torch.Tensor, *, shuffle: bool) -> EmbeddingsDataset:
        return EmbeddingsDataset(
            embeddings,
            batch_size=int(self.cfg.batch_size),
            shuffle=shuffle,
            seed=int(self.cfg.seed),
            device=self.device,
            dtype=self._model_dtype(),
        )

    def _progress(self, iterable, *, total: int | None = None):
        if not self.cfg.show_progress:
            return iterable
        try:
            from tqdm.auto import tqdm
        except Exception:  # pragma: no cover - optional dependency fallback
            return iterable
        return tqdm(iterable, total=total)

    def _set_lr(self, lr: float) -> None:
        if self.optimizer is None:
            raise RuntimeError("trainer must be built before setting learning rate")
        for group in self.optimizer.param_groups:
            group["lr"] = float(lr)

    def _current_lr(self) -> float:
        if self.optimizer is None:
            raise RuntimeError("trainer must be built before reading learning rate")
        return float(self.optimizer.param_groups[0]["lr"])

    def _new_noise_generator(self, device: torch.device) -> torch.Generator:
        try:
            generator = torch.Generator(device=device)
        except (RuntimeError, TypeError):
            # Some backends do not expose device-local generators.
            generator = torch.Generator()
        generator.manual_seed(int(self.cfg.seed))
        return generator

    def _randn_like(self, batch: torch.Tensor) -> torch.Tensor:
        if self._noise_generator is None:
            self._noise_generator = self._new_noise_generator(batch.device)
        generator_device = torch.device(self._noise_generator.device)
        noise = torch.randn(
            batch.shape,
            dtype=batch.dtype,
            device=generator_device,
            generator=self._noise_generator,
        )
        return noise if generator_device == batch.device else noise.to(batch.device)

    def _fit_gaussian_noise_scale(self, embeddings: np.ndarray | torch.Tensor) -> None:
        """Compute fixed training-set statistics used by adaptive Gaussian noise."""
        self.input_feature_mean = None
        self.input_feature_variance = None
        self._gaussian_noise_scale = None

        if self.cfg.noise_type != "gaussian" or self.cfg.noise_scale == "absolute":
            return

        values = torch.as_tensor(embeddings)
        if not torch.is_floating_point(values):
            values = values.float()
        elif values.dtype not in {torch.float32, torch.float64}:
            values = values.float()
        if not bool(torch.isfinite(values).all()):
            raise ValueError("adaptive Gaussian noise requires finite embeddings")

        variance, mean = torch.var_mean(values, dim=0, correction=0)
        self.input_feature_mean = mean.detach().cpu()
        self.input_feature_variance = variance.detach().cpu()

        if self.cfg.noise_scale == "global_rms":
            scale = variance.mean().sqrt()
        elif self.cfg.noise_scale == "feature_std":
            scale = variance.sqrt()
        else:  # pragma: no cover - guarded by config validation
            raise ValueError(f"unknown noise_scale: {self.cfg.noise_scale}")

        if not bool(torch.isfinite(scale).all()):
            raise ValueError("adaptive Gaussian noise produced a non-finite scale")
        self._gaussian_noise_scale = scale.to(device=self.device, dtype=self._model_dtype())

    def _corrupt(self, batch: torch.Tensor) -> torch.Tensor:
        """Return an optionally corrupted training input."""
        if self.cfg.noise_type == "none" or self.cfg.noise_level == 0.0:
            return batch
        if self.cfg.noise_type != "gaussian":
            raise ValueError(f"unknown noise_type: {self.cfg.noise_type}")

        if self.cfg.noise_scale == "absolute":
            scale: torch.Tensor | float = 1.0
        else:
            if self._gaussian_noise_scale is None:
                raise RuntimeError("adaptive Gaussian noise requires fit() before train_step()")
            scale = self._gaussian_noise_scale.to(device=batch.device, dtype=batch.dtype)
        return batch + float(self.cfg.noise_level) * scale * self._randn_like(batch)

    def train_step(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run one optimization step and return detached training stats."""
        if self.sae is None or self.optimizer is None:
            raise RuntimeError("trainer must be built before train_step")
        self.sae.train()
        self.optimizer.zero_grad(set_to_none=True)

        clean = batch
        corrupted = self._corrupt(clean)
        reconstruction, sparse, stats = self.sae(corrupted)
        if corrupted is clean:
            cosine_loss = 1.0 - stats["cosine_similarity"]
            mse = stats["reconstruction_mse"]
        else:
            cosine_loss = 1.0 - F.cosine_similarity(reconstruction, clean, dim=-1).mean()
            mse = F.mse_loss(reconstruction, clean)
        loss = float(self.cfg.alpha_loss) * cosine_loss + (1.0 - float(self.cfg.alpha_loss)) * mse
        if self.cfg.l1_penalty > 0.0:
            loss = loss + float(self.cfg.l1_penalty) * sparse.abs().mean()
        loss.backward()
        self.optimizer.step()
        result = {
            "loss": loss.detach(),
            "cosine_loss": cosine_loss.detach(),
            "reconstruction_mse": mse.detach(),
            "active_count": stats["active_count"].detach(),
            "dead_features": stats["dead_features"].detach(),
        }
        if corrupted is not clean:
            result["corrupted_cosine_loss"] = (1.0 - stats["cosine_similarity"]).detach()
            result["corrupted_reconstruction_mse"] = stats["reconstruction_mse"].detach()
        return result

    def fit(self, embeddings: np.ndarray | torch.Tensor) -> "TopKSAETrainer":
        """Train the SAE on dense embeddings and return ``self``."""
        self.build(self._input_dim(embeddings))
        self._fit_gaussian_noise_scale(embeddings)
        dataset = self._dataset(embeddings, shuffle=bool(self.cfg.shuffle))
        epochs = int(self.cfg.epochs)
        if epochs < 1:
            raise ValueError("epochs must be >= 1")
        self._set_lr(float(self.cfg.lr))
        scheduler = (
            torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=0.0)
            if self.cfg.decay
            else None
        )
        epoch_iter = self._progress(range(1, epochs + 1), total=epochs)
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
            record["lr"] = self._current_lr()
            self.history.append(record)
            if hasattr(epoch_iter, "set_postfix"):
                epoch_iter.set_postfix(
                    {
                        "loss": f"{record['loss']:.4f}",
                        "cosine": f"{record['cosine_loss']:.4f}",
                        "mse": f"{record['reconstruction_mse']:.4E}",
                        "lr": f"{record['lr']:.2E}",
                    }
                )
            if scheduler is not None:
                scheduler.step()
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
            "format_version": 2,
            "config": self.cfg,
            "input_dim": self.input_dim,
            "model": self.sae.state_dict(),
            "optimizer": self.optimizer.state_dict() if self.optimizer is not None else None,
            "history": list(self.history),
            "input_feature_mean": self.input_feature_mean,
            "input_feature_variance": self.input_feature_variance,
            "gaussian_noise_scale": (
                self._gaussian_noise_scale.detach().cpu() if self._gaussian_noise_scale is not None else None
            ),
            "noise_generator_state": (
                self._noise_generator.get_state() if self._noise_generator is not None else None
            ),
            "noise_generator_device": (
                str(self._noise_generator.device) if self._noise_generator is not None else None
            ),
        }

    def load_state_dict(
        self,
        state: dict[str, Any],
        *,
        load_optimizer: bool = True,
    ) -> "TopKSAETrainer":
        """Restore trainer state, including fitted denoising statistics."""
        format_version = int(state.get("format_version", 1))
        if format_version not in {1, 2}:
            raise ValueError(f"unsupported trainer state format_version: {format_version}")
        if state.get("input_dim") is None:
            raise ValueError("trainer state is missing input_dim")

        self.build(int(state["input_dim"]))
        if self.sae is None:
            raise RuntimeError("trainer could not be built")
        self.sae.load_state_dict(state["model"])
        optimizer_state = state.get("optimizer")
        if load_optimizer and optimizer_state is not None:
            if self.optimizer is None:
                raise RuntimeError("trainer optimizer could not be built")
            self.optimizer.load_state_dict(optimizer_state)
        self.history = list(state.get("history", []))

        feature_mean = state.get("input_feature_mean")
        feature_variance = state.get("input_feature_variance")
        gaussian_scale = state.get("gaussian_noise_scale")
        self.input_feature_mean = feature_mean.detach().cpu() if feature_mean is not None else None
        self.input_feature_variance = feature_variance.detach().cpu() if feature_variance is not None else None
        if gaussian_scale is None and self.input_feature_variance is not None:
            if self.cfg.noise_scale == "global_rms":
                gaussian_scale = self.input_feature_variance.mean().sqrt()
            elif self.cfg.noise_scale == "feature_std":
                gaussian_scale = self.input_feature_variance.sqrt()
        self._gaussian_noise_scale = (
            gaussian_scale.to(device=self.device, dtype=self._model_dtype()) if gaussian_scale is not None else None
        )

        self._noise_generator = None
        generator_state = state.get("noise_generator_state")
        saved_generator_device = state.get("noise_generator_device")
        if generator_state is not None:
            generator = self._new_noise_generator(self.device)
            current_device_type = torch.device(generator.device).type
            saved_device_type = torch.device(saved_generator_device or "cpu").type
            if current_device_type == saved_device_type:
                generator.set_state(generator_state.detach().cpu())
            else:
                warnings.warn(
                    "noise generator device changed while loading; future Gaussian noise "
                    "will restart from config.seed",
                    RuntimeWarning,
                    stacklevel=2,
                )
            self._noise_generator = generator
        return self

    @classmethod
    def from_state_dict(
        cls,
        state: dict[str, Any],
        *,
        device: str | torch.device | None = None,
        load_optimizer: bool = True,
    ) -> "TopKSAETrainer":
        """Construct a trainer from :meth:`state_dict` output."""
        config = state.get("config")
        if not isinstance(config, TopKSAEConfig):
            raise ValueError("trainer state is missing a TopKSAEConfig")
        config = copy.deepcopy(config)
        if device is not None:
            config = replace(config, device=device)
        return cls(config).load_state_dict(state, load_optimizer=load_optimizer)
