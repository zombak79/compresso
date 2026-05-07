"""TopKSAE — reference sparse autoencoder with top-k bottleneck."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from compresso.nn.sparsify import TopKSparsify

__all__ = ["TopKSAE"]


class TopKSAE(nn.Module):
    """Sparse autoencoder with hard top-k bottleneck.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input (and reconstruction).
    hidden_dim : int
        Width of the sparse code layer.
    k : int
        Number of active features per sample.
    tied : bool
        If ``True``, decoder weight is the transpose of the encoder weight.
    pre_act : nn.Module | None
        Optional activation applied to encoder output *before* sparsification.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        k: int,
        tied: bool = False,
        pre_act: Optional[nn.Module] = None,
        post_sparsify: Optional[nn.Module] = None,
        encoder: Optional[nn.Module] = None,
        decoder: Optional[nn.Module] = None,
        sparsify_mode: str = "values",
        sparsify_score_mode: str = "abs",
        k_backward: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.k = k
        self.tied = tied

        if encoder is None:
            self.encoder = nn.Linear(input_dim, hidden_dim)
        else:
            self.encoder = encoder

        self.sparsify = TopKSparsify(
            k=k,
            dim=-1,
            mode=sparsify_mode,
            score_mode=sparsify_score_mode,
            k_backward=k_backward,
        )

        if pre_act is not None:
            self.pre_act = pre_act
        else:
            self.pre_act = None

        if post_sparsify is not None:
            self.post_sparsify = post_sparsify
        else:
            self.post_sparsify = None

        if tied:
            if decoder is not None:
                raise ValueError("Cannot provide a custom decoder when tied=True.")
            # Only a bias for the decoder path; weight is encoder.weight.T
            self.decoder_bias = nn.Parameter(torch.zeros(input_dim))
        else:
            if decoder is None:
                self.decoder = nn.Linear(hidden_dim, input_dim)
            else:
                self.decoder = decoder

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def get_decoder_weight(self) -> Tensor:
        """Return the effective decoder weight matrix ``(input_dim, hidden_dim)``."""
        if self.tied:
            return self.encoder.weight.t()
        return self.decoder.weight

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: Tensor):
        """
        Returns
        -------
        reconstruction : Tensor  ``(B, input_dim)``
        codes : Tensor  ``(B, hidden_dim)``   — sparse, exactly *k* nonzeros per row
        stats : dict[str, Tensor]
        """
        h = self.encoder(x)
        if self.pre_act is not None:
            h = self.pre_act(h)
        codes = self.sparsify(h)
        if self.post_sparsify is not None:
            codes = self.post_sparsify(codes)

        if self.tied:
            # encoder.weight is (hidden_dim, input_dim)
            # We want codes @ encoder.weight = (B, H) @ (H, D) = (B, D)
            # F.linear(input, weight, bias) computes input @ weight.T + bias
            # So pass encoder.weight.t() → (D, H), then F.linear does codes @ (D,H).T = codes @ (H,D)
            reconstruction = F.linear(codes, self.encoder.weight.t(), self.decoder_bias)
        else:
            reconstruction = self.decoder(codes)

        stats = self._compute_stats(x, reconstruction, codes)
        return reconstruction, codes, stats

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def _compute_stats(self, x: Tensor, recon: Tensor, codes: Tensor) -> dict:
        active_mask = codes != 0  # (B, H)

        active_count = active_mask.float().sum(dim=-1).mean()
        activation_freq = active_mask.float().mean(dim=0)  # (H,)
        dead_features = (activation_freq == 0).sum()

        x_flat = x.reshape(x.shape[0], -1)
        recon_flat = recon.reshape(recon.shape[0], -1)
        if x_flat.shape != recon_flat.shape:
            raise ValueError(
                f"Input and reconstruction must match per-sample flattened shape, got "
                f"{tuple(x_flat.shape)} vs {tuple(recon_flat.shape)}"
            )

        diff = x_flat - recon_flat
        reconstruction_mse = (diff * diff).mean()

        # Cosine similarity (per sample, then average)
        cos = F.cosine_similarity(x_flat, recon_flat, dim=-1).mean()

        return {
            "active_count": active_count,
            "activation_freq": activation_freq,
            "reconstruction_mse": reconstruction_mse,
            "cosine_similarity": cos,
            "dead_features": dead_features,
        }
