"""Tests for compresso.nn.sae.TopKSAE."""

import io
import pytest
import torch
import torch.nn as nn

from compresso.nn import TopKSAE


# ---------------------------------------------------------------------------
# Dimensions used throughout
# ---------------------------------------------------------------------------
B, D, H, K = 8, 32, 64, 8


# ---------------------------------------------------------------------------
# Forward shape & structure
# ---------------------------------------------------------------------------


class TestForwardShape:
    def test_output_shapes(self, dtype, device):
        model = TopKSAE(input_dim=D, hidden_dim=H, k=K).to(dtype=dtype, device=device)
        x = torch.randn(B, D, dtype=dtype, device=device)
        recon, codes, stats = model(x)

        assert recon.shape == (B, D)
        assert codes.shape == (B, H)

    def test_exactly_k_active_codes(self, dtype, device):
        model = TopKSAE(input_dim=D, hidden_dim=H, k=K).to(dtype=dtype, device=device)
        x = torch.randn(B, D, dtype=dtype, device=device)
        _, codes, _ = model(x)

        nonzeros_per_sample = (codes != 0).sum(dim=-1)
        assert (nonzeros_per_sample == K).all()

    def test_stats_dict_keys(self, dtype, device):
        model = TopKSAE(input_dim=D, hidden_dim=H, k=K).to(dtype=dtype, device=device)
        x = torch.randn(B, D, dtype=dtype, device=device)
        _, _, stats = model(x)

        expected_keys = {
            "active_count",
            "activation_freq",
            "reconstruction_mse",
            "cosine_similarity",
            "dead_features",
        }
        assert expected_keys.issubset(stats.keys())


# ---------------------------------------------------------------------------
# Training smoke test
# ---------------------------------------------------------------------------


class TestTraining:
    def test_loss_decreases(self, device):
        model = TopKSAE(input_dim=D, hidden_dim=H, k=K).to(device=device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        x = torch.randn(64, D, device=device)

        first_loss = None
        last_loss = None
        for _ in range(30):
            recon, _, stats = model(x)
            loss = stats["reconstruction_mse"]
            opt.zero_grad()
            loss.backward()
            opt.step()
            if first_loss is None:
                first_loss = loss.item()
            last_loss = loss.item()

        assert last_loss < first_loss

    def test_gradients_exist(self, device):
        model = TopKSAE(input_dim=D, hidden_dim=H, k=K).to(device=device)
        x = torch.randn(B, D, device=device)
        recon, _, stats = model(x)
        stats["reconstruction_mse"].backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No grad for {name}"
                assert param.grad.isfinite().all(), f"Non-finite grad for {name}"


# ---------------------------------------------------------------------------
# Tied / untied decoder
# ---------------------------------------------------------------------------


class TestTiedDecoder:
    def test_tied_weight_sharing(self):
        model = TopKSAE(input_dim=D, hidden_dim=H, k=K, tied=True)
        x = torch.randn(B, D)
        recon, _, _ = model(x)

        # With tied decoder, decoder weight should equal encoder weight transposed
        enc_w = model.encoder.weight  # (H, D)
        dec_w = model.get_decoder_weight()  # should be (D, H)
        assert torch.equal(dec_w, enc_w.t())

    def test_untied_independent(self):
        model = TopKSAE(input_dim=D, hidden_dim=H, k=K, tied=False)
        enc_w = model.encoder.weight  # (H, D)
        dec_w = model.get_decoder_weight()  # (D, H)
        # Untied: decoder weight is independent — extremely unlikely to equal encoder.T
        assert not torch.equal(dec_w, enc_w.t())


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_save_load_roundtrip(self, device):
        model = TopKSAE(input_dim=D, hidden_dim=H, k=K, tied=False).to(device)
        x = torch.randn(B, D, device=device)
        recon1, codes1, _ = model(x)

        buf = io.BytesIO()
        torch.save(model.state_dict(), buf)
        buf.seek(0)

        model2 = TopKSAE(input_dim=D, hidden_dim=H, k=K, tied=False).to(device)
        model2.load_state_dict(torch.load(buf, map_location=device, weights_only=True))
        recon2, codes2, _ = model2(x)

        assert torch.allclose(recon1, recon2)
        assert torch.equal(codes1, codes2)

    def test_save_load_tied(self, device):
        model = TopKSAE(input_dim=D, hidden_dim=H, k=K, tied=True).to(device)
        x = torch.randn(B, D, device=device)
        recon1, codes1, _ = model(x)

        buf = io.BytesIO()
        torch.save(model.state_dict(), buf)
        buf.seek(0)

        model2 = TopKSAE(input_dim=D, hidden_dim=H, k=K, tied=True).to(device)
        model2.load_state_dict(torch.load(buf, map_location=device, weights_only=True))
        recon2, codes2, _ = model2(x)

        assert torch.allclose(recon1, recon2)
        assert torch.equal(codes1, codes2)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_zero_input(self, device):
        model = TopKSAE(input_dim=D, hidden_dim=H, k=K).to(device)
        x = torch.zeros(B, D, device=device)
        recon, codes, stats = model(x)
        # Should not crash; reconstruction should be near-zero since input is zero
        assert recon.isfinite().all()
        assert codes.isfinite().all()

    def test_single_sample(self, device):
        model = TopKSAE(input_dim=D, hidden_dim=H, k=K).to(device)
        x = torch.randn(1, D, device=device)
        recon, codes, stats = model(x)
        assert recon.shape == (1, D)
        assert codes.shape == (1, H)

    def test_pre_act(self, device):
        model = TopKSAE(input_dim=D, hidden_dim=H, k=K, pre_act=nn.ReLU()).to(device)
        x = torch.randn(B, D, device=device)
        recon, codes, stats = model(x)
        # With ReLU pre-activation, codes should be non-negative
        assert (codes >= 0).all()


class TestCustomModules:
    def test_custom_encoder_decoder(self, device):
        encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, H),
        )
        decoder = nn.Sequential(
            nn.Linear(H, 28 * 28),
            nn.Unflatten(1, (28, 28)),
        )
        model = TopKSAE(
            input_dim=28 * 28,
            hidden_dim=H,
            k=K,
            encoder=encoder,
            decoder=decoder,
        ).to(device)

        x = torch.randn(B, 28, 28, device=device)
        recon, codes, stats = model(x)
        assert recon.shape == (B, 28, 28)
        assert codes.shape == (B, H)
        assert (codes != 0).sum(dim=-1).eq(K).all()
        assert "reconstruction_mse" in stats

    def test_post_sparsify_hook_runs(self, device):
        model = TopKSAE(
            input_dim=D,
            hidden_dim=H,
            k=K,
            post_sparsify=nn.ReLU(),
            sparsify_mode="values",
            sparsify_score_mode="abs",
            k_backward=K,
        ).to(device)
        x = torch.randn(B, D, device=device)
        _, codes, _ = model(x)
        # ReLU post-sparsify should clip negative selected values
        assert (codes >= 0).all()
