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
        Optional dtype for returned batches. Conversion happens per batch, so a
        memory-mapped or half-precision source is never materialized in full.
        ``None`` preserves the source dtype, except that integer sources are
        promoted to float.
    rows:
        Optional row indices to restrict the dataset to, as positions in
        ``embeddings``. Used to view a train or validation split without
        copying it out of the source.
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
        rows: np.ndarray | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        tensor = torch.as_tensor(embeddings)
        if tensor.ndim != 2:
            raise ValueError(f"embeddings must be 2D, got shape {tuple(tensor.shape)}")

        # Kept exactly as handed over: casting or copying here would pull a
        # memory-mapped source into RAM before a single batch is read.
        self.embeddings = tensor
        self.out_dtype = dtype
        self.dim = int(tensor.shape[1])
        if rows is None:
            self.indices = np.arange(int(tensor.shape[0]))
        else:
            self.indices = np.asarray(rows, dtype=np.int64).copy()
            if self.indices.ndim != 1:
                raise ValueError("rows must be one-dimensional")
        self.n = int(self.indices.size)
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
        # A blocking copy is required: this is a device-to-host transfer whenever
        # ``embeddings`` lives on an accelerator and ``device`` is CPU, and an
        # unsynchronized one returns memory before the copy lands.
        batch = batch.to(self.device)
        # Convert after the transfer, so a half-precision source crosses the bus
        # at half the bytes and is widened on the destination device.
        if not torch.is_floating_point(batch):
            batch = batch.float()
        if self.out_dtype is not None and batch.dtype != self.out_dtype:
            batch = batch.to(self.out_dtype)
        return batch

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


def _streaming_mean_variance(
    embeddings: np.ndarray | torch.Tensor,
    *,
    name: str,
    rows: np.ndarray | None = None,
    chunk_bytes: int = 32 * 1024 * 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-feature mean and population variance over one chunked pass.

    Reads the source in slices sized by the float64 accumulation rather than by
    the input dtype, so a memory-mapped or half-precision matrix is never
    materialized. Chunks are merged with Chan's parallel formula instead of
    ``E[x**2] - E[x]**2``: the latter cancels to zero in float32 for a feature
    whose mean dwarfs its spread, which is exactly the anisotropic case that
    standard scaling is meant to fix.

    Accumulation happens on the host, so every backend gets float64 arithmetic.
    A device-resident source therefore pays one host transfer per chunk, once
    per fit.
    """
    tensor = torch.as_tensor(embeddings)
    if tensor.ndim != 2:
        raise ValueError(f"{name} requires 2D embeddings, got shape {tuple(tensor.shape)}")
    dim = int(tensor.shape[1])
    if rows is None:
        order = None
        total = int(tensor.shape[0])
    else:
        # Sorted so a memory-mapped source is still read front to back.
        order = np.sort(np.asarray(rows, dtype=np.int64))
        total = int(order.size)
    if total < 1:
        raise ValueError(f"{name} requires at least one embedding row")

    rows_per_chunk = max(1, chunk_bytes // (dim * 8))
    count = 0
    mean = torch.zeros(dim, dtype=torch.float64)
    m2 = torch.zeros(dim, dtype=torch.float64)
    for start in range(0, total, rows_per_chunk):
        if order is None:
            block = tensor[start : start + rows_per_chunk]
        else:
            block = tensor[torch.from_numpy(order[start : start + rows_per_chunk])]
        # copy=True: a float64 host source would hand back a view, and the
        # centering below is in place.
        block = block.to(device="cpu", dtype=torch.float64, copy=True)
        if not bool(torch.isfinite(block).all()):
            raise ValueError(f"{name} requires finite embeddings")
        block_rows = int(block.shape[0])
        block_mean = block.mean(dim=0)
        block -= block_mean
        block_m2 = block.square().sum(dim=0)
        delta = block_mean - mean
        merged = count + block_rows
        mean = mean + delta * (block_rows / merged)
        m2 = m2 + block_m2 + delta.square() * (count * block_rows / merged)
        count = merged
    return mean, m2 / count


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
    standard_scaler_mean:
        Subtract the per-feature training mean from inputs before the SAE, and
        add it back on its reconstruction. Off by default.
    standard_scaler_scale:
        Division applied after centering, and undone on the reconstruction.
        ``"none"`` leaves magnitudes alone. ``"feature_std"`` divides each
        feature by its own standard deviation, matching
        ``sklearn.preprocessing.StandardScaler``; it gives every feature unit
        variance, which flattens the relative importance of coordinates and
        bends the embedding geometry, so adaptive ``noise_scale`` is rejected
        alongside it. ``"global_rms"`` divides everything by one scalar, the
        root mean per-feature variance: coordinates land at unit scale for the
        encoder while every angle and every distance ratio is preserved
        exactly, since a uniform scale is not a distortion.

        Statistics are fitted on the training rows with ``correction=0``, and
        constant features keep a scale of ``1``.

        Neither scaling mode sits well with a normalizing ``post_sparsify``:
        unit-norm codes carry no magnitude, so the rescale has to be undone by
        the decoder alone and converges several times worse. That combination
        warns rather than raising, since it is merely a bad trade rather than a
        contradiction. ``standard_scaler_mean`` is unaffected, because centering
        barely moves the magnitude.
    standard_scaler_loss_space:
        Space the reconstruction loss is measured in when standard scaling is
        active. ``"original"`` un-scales the reconstruction and compares it to
        the raw input, keeping the objective and reported metrics identical to
        an unscaled run. ``"scaled"`` compares in standardized space, which
        weights every feature equally instead of by its variance.
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
        Maximum number of training epochs. Early stopping can end training
        before this many epochs have run.
    validation_frac:
        Optional fraction of input rows held out for validation. Mutually
        exclusive with the ``validation_embeddings`` argument of
        :meth:`TopKSAETrainer.fit`. When both are unset, no validation pass
        runs and early stopping is unavailable.
    patience:
        Number of consecutive epochs without a validation improvement
        tolerated before training stops. ``None`` disables early stopping.
        Requires a validation set.
    min_delta:
        Smallest decrease in validation loss that counts as an improvement.
    restore_best_weights:
        If ``True``, reload the weights of the best-scoring epoch once
        training ends. Only applies when validation is active.
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
    standard_scaler_mean: bool = False
    standard_scaler_scale: Literal["none", "feature_std", "global_rms"] = "none"
    standard_scaler_loss_space: Literal["original", "scaled"] = "original"
    alpha_loss: float = 0.01
    l1_penalty: float = 0.0
    batch_size: int = 128
    shuffle: bool = True
    seed: int = 42
    epochs: int = 10
    validation_frac: float | None = None
    patience: int | None = None
    min_delta: float = 0.0
    restore_best_weights: bool = True
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
        self.input_scaler_mean: torch.Tensor | None = None
        self.input_scaler_scale: torch.Tensor | None = None
        self._gaussian_noise_scale: torch.Tensor | None = None
        self._scaler_mean: torch.Tensor | None = None
        self._scaler_scale: torch.Tensor | None = None
        self._noise_generator: torch.Generator | None = None
        self.best_epoch: int | None = None
        self.best_val_loss: float | None = None
        self.stopped_epoch: int | None = None

    @property
    def is_built(self) -> bool:
        """Whether the underlying ``TopKSAE`` model has been initialized."""
        return self.sae is not None

    @property
    def _standard_scaling_enabled(self) -> bool:
        return bool(self.cfg.standard_scaler_mean) or self.cfg.standard_scaler_scale != "none"

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
        if self.cfg.standard_scaler_loss_space not in {"original", "scaled"}:
            raise ValueError(f"unknown standard_scaler_loss_space: {self.cfg.standard_scaler_loss_space}")
        if self.cfg.standard_scaler_scale not in {"none", "feature_std", "global_rms"}:
            raise ValueError(f"unknown standard_scaler_scale: {self.cfg.standard_scaler_scale}")
        # Coherent but slow, unlike the rejections around it, so this one only
        # warns: normalized codes are scale-invariant, which leaves the decoder
        # alone to absorb a rescaled input from an initialization that is too
        # small by exactly that factor.
        if self.cfg.standard_scaler_scale != "none" and isinstance(
            self.cfg.post_sparsify, (L1Normalize, L2Normalize)
        ):
            warnings.warn(
                "standard_scaler_scale rescales inputs, but a normalizing post_sparsify "
                "makes the codes scale-invariant, so only the decoder can absorb it. "
                "Expect several times the reconstruction error, which more epochs do not "
                "recover. standard_scaler_scale='none' avoids it, and "
                "standard_scaler_mean is unaffected.",
                RuntimeWarning,
                stacklevel=2,
            )
        # Adaptive noise scales are derived from the scaled variances, so every
        # mode stays correct. feature_std is the one that makes them degenerate:
        # unit variance everywhere turns both adaptive modes into absolute.
        if self.cfg.standard_scaler_scale == "feature_std":
            if self.cfg.noise_type == "gaussian" and self.cfg.noise_scale != "absolute":
                raise ValueError(
                    "standard_scaler_scale='feature_std' gives every feature unit "
                    "variance, so adaptive noise scales collapse to 1: use "
                    "noise_scale='absolute' with it"
                )
        if not self._standard_scaling_enabled and self.cfg.standard_scaler_loss_space != "original":
            raise ValueError(
                "standard_scaler_loss_space requires standard_scaler_mean or standard_scaler_scale"
            )
        if self.cfg.validation_frac is not None and not (
            math.isfinite(float(self.cfg.validation_frac)) and 0.0 < float(self.cfg.validation_frac) < 1.0
        ):
            raise ValueError("validation_frac must be finite and in (0, 1)")
        if self.cfg.patience is not None and int(self.cfg.patience) < 1:
            raise ValueError("patience must be >= 1")
        if not math.isfinite(float(self.cfg.min_delta)) or self.cfg.min_delta < 0.0:
            raise ValueError("min_delta must be finite and >= 0")

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
        if self._scaler_mean is not None:
            self._scaler_mean = self._scaler_mean.to(self.device)
        if self._scaler_scale is not None:
            self._scaler_scale = self._scaler_scale.to(self.device)
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

    def _dataset(
        self,
        embeddings: np.ndarray | torch.Tensor,
        *,
        shuffle: bool,
        rows: np.ndarray | None = None,
    ) -> EmbeddingsDataset:
        return EmbeddingsDataset(
            embeddings,
            batch_size=int(self.cfg.batch_size),
            shuffle=shuffle,
            seed=int(self.cfg.seed),
            device=self.device,
            dtype=self._model_dtype(),
            rows=rows,
        )

    def _split_validation(
        self,
        embeddings: np.ndarray | torch.Tensor,
        frac: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Split rows into train and validation parts, seeded by ``config.seed``.

        Returns row indices rather than the rows themselves: materializing the
        two parts would copy the whole matrix, which a memory-mapped source
        cannot afford. Rows are permuted before splitting so that ordered
        inputs, such as item embeddings sorted by popularity, do not put a
        biased slice in the validation part. Both parts always receive at least
        one row.
        """
        n_rows = int(torch.as_tensor(embeddings).shape[0])
        if n_rows < 2:
            raise ValueError(f"validation_frac requires at least 2 embedding rows, got {n_rows}")
        n_val = max(1, min(int(round(n_rows * frac)), n_rows - 1))
        order = np.random.default_rng(int(self.cfg.seed)).permutation(n_rows)
        return order[n_val:].astype(np.int64), order[:n_val].astype(np.int64)

    def _weight_snapshot(self) -> dict[str, torch.Tensor]:
        """Return a detached CPU copy of the current model weights."""
        if self.sae is None:
            raise RuntimeError("trainer must be built before snapshotting weights")
        return {key: value.detach().cpu().clone() for key, value in self.sae.state_dict().items()}

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

    def _fit_gaussian_noise_scale(
        self,
        embeddings: np.ndarray | torch.Tensor,
        rows: np.ndarray | None = None,
        stats: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> None:
        """Compute fixed training-set statistics used by adaptive Gaussian noise."""
        self.input_feature_mean = None
        self.input_feature_variance = None
        self._gaussian_noise_scale = None

        if self.cfg.noise_type != "gaussian" or self.cfg.noise_scale == "absolute":
            return

        mean, variance = stats or _streaming_mean_variance(
            embeddings,
            name="adaptive Gaussian noise",
            rows=rows,
        )
        dtype = self._model_dtype()
        # Kept as the statistics of the embeddings handed in, not of what the
        # SAE sees after scaling, which is what their names say.
        self.input_feature_mean = mean.to(dtype)
        self.input_feature_variance = variance.to(dtype)
        self._gaussian_noise_scale = self._noise_scale_from(variance)

    def _noise_scale_from(self, variance: torch.Tensor) -> torch.Tensor:
        """Adaptive noise scale for the space the corruption is applied in.

        Noise is injected after standard scaling, so the scale has to describe
        the scaled spread rather than the raw one.
        """
        scaled = self._scaled_variance(variance)
        if self.cfg.noise_scale == "global_rms":
            scale = scaled.mean().sqrt()
        elif self.cfg.noise_scale == "feature_std":
            scale = scaled.sqrt()
        else:  # pragma: no cover - guarded by config validation
            raise ValueError(f"unknown noise_scale: {self.cfg.noise_scale}")

        if not bool(torch.isfinite(scale).all()):
            raise ValueError("adaptive Gaussian noise produced a non-finite scale")
        return scale.to(device=self.device, dtype=self._model_dtype())

    def _fit_standard_scaler(
        self,
        embeddings: np.ndarray | torch.Tensor,
        rows: np.ndarray | None = None,
        stats: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> None:
        """Compute per-feature centering and scaling from training rows only."""
        self.input_scaler_mean = None
        self.input_scaler_scale = None
        self._scaler_mean = None
        self._scaler_scale = None

        if not self._standard_scaling_enabled:
            return

        mean, variance = stats or _streaming_mean_variance(
            embeddings,
            name="standard scaling",
            rows=rows,
        )
        dtype = self._model_dtype()
        if self.cfg.standard_scaler_mean:
            self.input_scaler_mean = mean.to(dtype)
        if self.cfg.standard_scaler_scale == "feature_std":
            scale = variance.sqrt()
        elif self.cfg.standard_scaler_scale == "global_rms":
            # One scalar for the whole matrix, so angles and distance ratios
            # come through untouched; only the magnitude moves.
            scale = variance.mean().sqrt()
        else:
            scale = None
        if scale is not None:
            # A constant feature would divide by zero; sklearn leaves it at 1.
            scale = torch.where(scale > 0, scale, torch.ones_like(scale)).to(dtype)
            if not bool(torch.isfinite(scale).all()):
                raise ValueError("standard scaling produced a non-finite scale")
            self.input_scaler_scale = scale.detach().cpu()
        self._cache_scaler_tensors()

    def _scaled_variance(self, variance: torch.Tensor) -> torch.Tensor:
        """Per-feature variance after standard scaling, without a second pass.

        Scaling divides by a fixed factor, so the variance the SAE actually sees
        is the raw one over that factor squared. Centering does not enter, since
        translation leaves variance alone.
        """
        if self.input_scaler_scale is None:
            return variance
        return variance / self.input_scaler_scale.to(variance.dtype).square()

    def _cache_scaler_tensors(self) -> None:
        """Keep device copies so per-batch scaling costs no host transfer."""
        dtype = self._model_dtype()
        self._scaler_mean = (
            self.input_scaler_mean.to(device=self.device, dtype=dtype)
            if self.input_scaler_mean is not None
            else None
        )
        self._scaler_scale = (
            self.input_scaler_scale.to(device=self.device, dtype=dtype)
            if self.input_scaler_scale is not None
            else None
        )

    def _require_fitted_scaler(self) -> None:
        if self.cfg.standard_scaler_mean and self._scaler_mean is None:
            raise RuntimeError("standard scaling requires fit() before train_step()")
        if self.cfg.standard_scaler_scale != "none" and self._scaler_scale is None:
            raise RuntimeError("standard scaling requires fit() before train_step()")

    def _standardize(self, batch: torch.Tensor) -> torch.Tensor:
        """Map raw inputs into the standardized space the SAE sees."""
        if not self._standard_scaling_enabled:
            return batch
        self._require_fitted_scaler()
        out = batch
        if self._scaler_mean is not None:
            out = out - self._scaler_mean.to(device=batch.device, dtype=batch.dtype)
        if self._scaler_scale is not None:
            out = out / self._scaler_scale.to(device=batch.device, dtype=batch.dtype)
        return out

    def _destandardize(self, batch: torch.Tensor) -> torch.Tensor:
        """Map SAE outputs back to the original embedding space."""
        if not self._standard_scaling_enabled:
            return batch
        self._require_fitted_scaler()
        out = batch
        if self._scaler_scale is not None:
            out = out * self._scaler_scale.to(device=batch.device, dtype=batch.dtype)
        if self._scaler_mean is not None:
            out = out + self._scaler_mean.to(device=batch.device, dtype=batch.dtype)
        return out

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
        # Corruption lives in the space the SAE trains in, so noise is applied
        # after standardization and is measured in standardized units.
        scaled = self._standardize(clean)
        corrupted = self._corrupt(scaled)
        reconstruction, sparse, stats = self.sae(corrupted)
        if self.cfg.standard_scaler_loss_space == "scaled":
            target, prediction = scaled, reconstruction
        else:
            target, prediction = clean, self._destandardize(reconstruction)
        # The model's own stats compare its output to its input, which is only
        # the loss target when nothing was corrupted or un-scaled in between.
        if corrupted is scaled and target is scaled:
            cosine_loss = 1.0 - stats["cosine_similarity"]
            mse = stats["reconstruction_mse"]
        else:
            cosine_loss = 1.0 - F.cosine_similarity(prediction, target, dim=-1).mean()
            mse = F.mse_loss(prediction, target)
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
        if corrupted is not scaled:
            if target is scaled:
                result["corrupted_cosine_loss"] = (1.0 - stats["cosine_similarity"]).detach()
                result["corrupted_reconstruction_mse"] = stats["reconstruction_mse"].detach()
            else:
                # Report against the corrupted input in the loss space, so the
                # whole history stays in one space.
                corrupted_target = self._destandardize(corrupted)
                result["corrupted_cosine_loss"] = (
                    1.0 - F.cosine_similarity(prediction, corrupted_target, dim=-1).mean()
                ).detach()
                result["corrupted_reconstruction_mse"] = F.mse_loss(prediction, corrupted_target).detach()
        return result

    @torch.no_grad()
    def eval_step(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        """Score one batch without corruption and return detached stats.

        Validation inputs are never passed through :meth:`_corrupt`, so the
        reported loss is free of the per-epoch noise draw that makes training
        loss a poor early-stopping signal for denoising configurations.
        """
        if self.sae is None:
            raise RuntimeError("trainer must be built before eval_step")
        self.sae.eval()
        scaled = self._standardize(batch)
        reconstruction, sparse, stats = self.sae(scaled)
        if self.cfg.standard_scaler_loss_space == "scaled" or scaled is batch:
            cosine_loss = 1.0 - stats["cosine_similarity"]
            mse = stats["reconstruction_mse"]
        else:
            prediction = self._destandardize(reconstruction)
            cosine_loss = 1.0 - F.cosine_similarity(prediction, batch, dim=-1).mean()
            mse = F.mse_loss(prediction, batch)
        loss = float(self.cfg.alpha_loss) * cosine_loss + (1.0 - float(self.cfg.alpha_loss)) * mse
        if self.cfg.l1_penalty > 0.0:
            loss = loss + float(self.cfg.l1_penalty) * sparse.abs().mean()
        return {
            "loss": loss.detach(),
            "cosine_loss": cosine_loss.detach(),
            "reconstruction_mse": mse.detach(),
            "active_count": stats["active_count"].detach(),
            "dead_features": stats["dead_features"].detach(),
        }

    def _evaluate(self, dataset: EmbeddingsDataset) -> dict[str, float]:
        """Return mean ``val_``-prefixed statistics over ``dataset``."""
        sums: dict[str, float] = {}
        n_batches = 0
        for batch in dataset:
            for key, value in self.eval_step(batch).items():
                sums[key] = sums.get(key, 0.0) + float(value.cpu().item())
            n_batches += 1
        return {f"val_{key}": value / max(1, n_batches) for key, value in sums.items()}

    def _resolve_validation(
        self,
        embeddings: np.ndarray | torch.Tensor,
        validation_embeddings: np.ndarray | torch.Tensor | None,
        input_dim: int,
    ) -> tuple[
        np.ndarray | torch.Tensor,
        np.ndarray | None,
        np.ndarray | torch.Tensor | None,
        np.ndarray | None,
    ]:
        """Return ``(train source, train rows, validation source, validation rows)``.

        A ``None`` row set means "every row of that source". Splits are returned
        as row indices so that neither part is copied out of the source.
        """
        if validation_embeddings is not None and self.cfg.validation_frac is not None:
            raise ValueError("pass either validation_embeddings or config.validation_frac, not both")

        if validation_embeddings is not None:
            validation_dim = self._input_dim(validation_embeddings)
            if validation_dim != input_dim:
                raise ValueError(
                    f"validation_embeddings must have {input_dim} columns to match embeddings, got {validation_dim}"
                )
            return embeddings, None, validation_embeddings, None
        if self.cfg.validation_frac is not None:
            train_rows, val_rows = self._split_validation(embeddings, float(self.cfg.validation_frac))
            return embeddings, train_rows, embeddings, val_rows
        if self.cfg.patience is not None:
            raise ValueError("patience requires validation_embeddings or config.validation_frac")
        return embeddings, None, None, None

    def fit(
        self,
        embeddings: np.ndarray | torch.Tensor,
        *,
        validation_embeddings: np.ndarray | torch.Tensor | None = None,
    ) -> "TopKSAETrainer":
        """Train the SAE on dense embeddings and return ``self``.

        Parameters
        ----------
        embeddings:
            Training rows. When ``config.validation_frac`` is set, the held-out
            part is split off from these rows before any training statistics,
            including adaptive noise scales, are fitted.
        validation_embeddings:
            Explicit validation rows, for when the split is made by the caller.
            Mutually exclusive with ``config.validation_frac``.
        """
        input_dim = self._input_dim(embeddings)
        self.build(input_dim)
        train_source, train_rows, val_source, val_rows = self._resolve_validation(
            embeddings, validation_embeddings, input_dim
        )

        # Mean-only scaling and adaptive noise both want the same per-feature
        # statistics, so read the source once rather than once each.
        adaptive_noise = self.cfg.noise_type == "gaussian" and self.cfg.noise_scale != "absolute"
        shared_stats = (
            _streaming_mean_variance(train_source, name="standard scaling", rows=train_rows)
            if self._standard_scaling_enabled and adaptive_noise
            else None
        )
        self._fit_standard_scaler(train_source, rows=train_rows, stats=shared_stats)
        self._fit_gaussian_noise_scale(train_source, rows=train_rows, stats=shared_stats)
        dataset = self._dataset(train_source, shuffle=bool(self.cfg.shuffle), rows=train_rows)
        val_dataset = (
            self._dataset(val_source, shuffle=False, rows=val_rows) if val_source is not None else None
        )
        epochs = int(self.cfg.epochs)
        if epochs < 1:
            raise ValueError("epochs must be >= 1")
        self._set_lr(float(self.cfg.lr))
        scheduler = (
            torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=0.0)
            if self.cfg.decay
            else None
        )
        self.best_epoch = None
        self.best_val_loss = None
        self.stopped_epoch = None
        patience = int(self.cfg.patience) if self.cfg.patience is not None else None
        min_delta = float(self.cfg.min_delta)
        best_score = math.inf
        best_state: dict[str, torch.Tensor] | None = None
        stale_epochs = 0

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
            if val_dataset is not None:
                record.update(self._evaluate(val_dataset))
                # NaN never satisfies this, so a diverged epoch counts as stale.
                if record["val_loss"] < best_score - min_delta:
                    best_score = record["val_loss"]
                    stale_epochs = 0
                    self.best_epoch = epoch
                    self.best_val_loss = record["val_loss"]
                    if self.cfg.restore_best_weights:
                        best_state = self._weight_snapshot()
                else:
                    stale_epochs += 1
            self.history.append(record)
            if hasattr(epoch_iter, "set_postfix"):
                postfix = {
                    "loss": f"{record['loss']:.4f}",
                    "cosine": f"{record['cosine_loss']:.4f}",
                    "mse": f"{record['reconstruction_mse']:.4E}",
                    "lr": f"{record['lr']:.2E}",
                }
                if "val_loss" in record:
                    postfix["val_loss"] = f"{record['val_loss']:.4f}"
                epoch_iter.set_postfix(postfix)
            if scheduler is not None:
                scheduler.step()
            if patience is not None and stale_epochs >= patience:
                self.stopped_epoch = epoch
                break
        if best_state is not None and self.sae is not None:
            self.sae.load_state_dict(best_state)
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
            _reconstruction, sparse, _stats = self.sae(self._standardize(batch))
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
            reconstruction, _sparse, _stats = self.sae(self._standardize(batch))
            # Reconstructions are returned in the caller's own embedding space.
            reconstructions.append(self._destandardize(reconstruction).detach().cpu())
        return torch.cat(reconstructions, dim=0)

    @torch.no_grad()
    def transform(self, embeddings: np.ndarray | torch.Tensor) -> SRPTensor:
        """Encode ``embeddings`` and return sparse codes as an ``SRPTensor``.

        Each batch is packed as it is produced, so the dense ``(n, hidden_dim)``
        code matrix is never held whole: peak memory is the packed ``(n, k)``
        result plus one batch. Top-k runs per row, so the result is identical to
        packing the full dense matrix in one go.
        """
        if self.sae is None:
            raise RuntimeError("trainer must be fitted or built before transform")
        dataset = self._dataset(embeddings, shuffle=False)
        if len(dataset) == 0:
            raise ValueError("embeddings must contain at least one row")
        self.sae.eval()
        cols: list[torch.Tensor] = []
        vals: list[torch.Tensor] = []
        code_dim = 0
        for batch in self._progress(dataset, total=len(dataset)):
            _reconstruction, sparse, _stats = self.sae(self._standardize(batch))
            # Packed on the host, matching what encode() used to hand over, so
            # tie-breaking cannot depend on the training device.
            codes = sparse.detach().cpu()
            code_dim = int(codes.shape[1])
            packed = SRPTensor.from_dense(
                codes,
                k=int(self.cfg.k),
                score_mode=self.cfg.srp_score_mode,
            )
            cols.append(packed.cols)
            vals.append(packed.vals)
        rows = int(dataset.n)
        return SRPTensor(
            cols=torch.cat(cols, dim=0),
            vals=torch.cat(vals, dim=0),
            shape=(rows, code_dim),
            prefix_shape=(rows,),
            validate=False,
        )

    def fit_transform(
        self,
        embeddings: np.ndarray | torch.Tensor,
        *,
        validation_embeddings: np.ndarray | torch.Tensor | None = None,
    ) -> SRPTensor:
        """Fit the SAE and return encoded sparse codes as an ``SRPTensor``.

        Codes are returned for every row of ``embeddings``, including any rows
        held out for validation by ``config.validation_frac``.
        """
        self.fit(embeddings, validation_embeddings=validation_embeddings)
        return self.transform(embeddings)

    def state_dict(self) -> dict[str, Any]:  # type: ignore[override]
        """Return a saveable trainer state dictionary."""
        if self.sae is None:
            raise RuntimeError("trainer must be built before state_dict")
        return {
            "format_version": 4,
            "config": self.cfg,
            "input_dim": self.input_dim,
            "model": self.sae.state_dict(),
            "optimizer": self.optimizer.state_dict() if self.optimizer is not None else None,
            "history": list(self.history),
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
            "stopped_epoch": self.stopped_epoch,
            "input_feature_mean": self.input_feature_mean,
            "input_feature_variance": self.input_feature_variance,
            "input_scaler_mean": self.input_scaler_mean,
            "input_scaler_scale": self.input_scaler_scale,
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
        if format_version not in {1, 2, 3, 4}:
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

        best_epoch = state.get("best_epoch")
        best_val_loss = state.get("best_val_loss")
        stopped_epoch = state.get("stopped_epoch")
        self.best_epoch = int(best_epoch) if best_epoch is not None else None
        self.best_val_loss = float(best_val_loss) if best_val_loss is not None else None
        self.stopped_epoch = int(stopped_epoch) if stopped_epoch is not None else None

        # Absent before format_version 4, where standard scaling did not exist.
        # Restored first: a noise scale rebuilt below is derived from it.
        scaler_mean = state.get("input_scaler_mean")
        scaler_scale = state.get("input_scaler_scale")
        self.input_scaler_mean = scaler_mean.detach().cpu() if scaler_mean is not None else None
        self.input_scaler_scale = scaler_scale.detach().cpu() if scaler_scale is not None else None
        self._cache_scaler_tensors()

        feature_mean = state.get("input_feature_mean")
        feature_variance = state.get("input_feature_variance")
        gaussian_scale = state.get("gaussian_noise_scale")
        self.input_feature_mean = feature_mean.detach().cpu() if feature_mean is not None else None
        self.input_feature_variance = feature_variance.detach().cpu() if feature_variance is not None else None
        if gaussian_scale is None and self.input_feature_variance is not None:
            self._gaussian_noise_scale = self._noise_scale_from(self.input_feature_variance)
        else:
            self._gaussian_noise_scale = (
                gaussian_scale.to(device=self.device, dtype=self._model_dtype())
                if gaussian_scale is not None
                else None
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
