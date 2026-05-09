from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from compresso.examples.checkpoint import load_recsys_checkpoint
from compresso.examples.retrieval import evaluate_item_embeddings_with_holdout
from compresso.io import load_srp_tensor
from compresso.nn import TopKSAE


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_path", type=str, required=True)
    p.add_argument("--sae_model_path", type=str, required=True)
    p.add_argument("--sae_sparse_path", type=str, required=True)
    p.add_argument("--compressed_elsa_srp_path", type=str, default=None)
    p.add_argument("--device", type=str, default="mps")
    p.add_argument("--eval_batch_size", type=int, default=1024)
    return p.parse_args()


def resolve_device(requested: str) -> str:
    req = requested.lower()
    if req == "cpu":
        return "cpu"
    if req == "mps":
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    if req == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return req


def select_metrics(metrics: dict) -> dict:
    return {
        "recall@20": metrics.get("recall@20", 0.0),
        "recall@50": metrics.get("recall@50", 0.0),
        "ndcg@100": metrics.get("ndcg@100", 0.0),
    }


def percent_drop(base: float, new: float) -> float:
    if base == 0:
        return 0.0
    return ((base - new) / base) * 100.0


def bytes_to_mb(n_bytes: int) -> float:
    return float(n_bytes) / (1024.0 * 1024.0)


def _calibrated_recall(target_sets: list[set[int]], pred_ranked: list[np.ndarray], k: int) -> float:
    vals = []
    for tset, pred in zip(target_sets, pred_ranked):
        if not tset:
            continue
        top = pred[:k]
        hits = sum(1 for i in top if int(i) in tset)
        denom = min(k, len(tset))
        vals.append(hits / denom if denom > 0 else 0.0)
    return float(np.mean(vals)) if vals else 0.0


def _ndcg(target_sets: list[set[int]], pred_ranked: list[np.ndarray], k: int) -> float:
    vals = []
    for tset, pred in zip(target_sets, pred_ranked):
        if not tset:
            continue
        dcg = 0.0
        for rank, item_idx in enumerate(pred[:k], start=1):
            if int(item_idx) in tset:
                dcg += 1.0 / np.log2(rank + 1)
        ideal_len = min(k, len(tset))
        idcg = sum(1.0 / np.log2(i + 1) for i in range(1, ideal_len + 1))
        vals.append(dcg / idcg if idcg > 0 else 0.0)
    return float(np.mean(vals)) if vals else 0.0


def _compute_topk_kernel_trick(
    z: torch.Tensor,  # (N,H), sparse code embeddings (dense tensor, mostly zeros)
    decoder_map: torch.Tensor,  # (H,D) mapping from codes -> reconstructed dense
    source_indices: list[np.ndarray],
    k: int,
    *,
    batch_size: int = 512,
) -> list[np.ndarray]:
    """Top-k retrieval in code space using decoder kernel trick.

    We emulate normalized reconstructed-space scoring:
      r_i = normalize(z_i @ W)
      profile_u = sum_{s in S_u} r_s
      score(u, j) = relu(profile_u · r_j)

    without materializing full reconstructed matrix r in D dimensions.
    """
    device = z.device
    n_items, h = z.shape
    k_eff = min(k, n_items)

    # K = W W^T, shape (H,H)
    k_mat = decoder_map @ decoder_map.t()

    # Per-item reconstructed norms: ||z_i W|| = sqrt(z_i K z_i^T)
    norms_sq = (z @ k_mat * z).sum(dim=1).clamp_min(1e-12)
    norms = torch.sqrt(norms_sq)
    z_scaled = z / norms.unsqueeze(1)  # corresponds to normalized reconstructed items

    preds: list[np.ndarray] = []
    for start in range(0, len(source_indices), batch_size):
        batch = source_indices[start : start + batch_size]
        b = len(batch)
        lengths = [len(x) for x in batch]
        flat_src = np.concatenate(batch, axis=0)
        flat_src_t = torch.from_numpy(flat_src).long().to(device)
        owners = torch.repeat_interleave(
            torch.arange(b, device=device, dtype=torch.long),
            torch.tensor(lengths, device=device, dtype=torch.long),
        )

        x = torch.zeros((b, n_items), device=device, dtype=z.dtype)
        x[owners, flat_src_t] = 1.0

        # p' = sum normalized source items in code-space preimage
        p_prime = x @ z_scaled  # (b,H)
        # score = p' K z_scaled^T
        scores = torch.relu((p_prime @ k_mat) @ z_scaled.t())
        scores[owners, flat_src_t] = -torch.inf

        topk_idx = torch.topk(scores, k_eff, dim=1, largest=True, sorted=True).indices
        preds.extend([row.detach().cpu().numpy() for row in topk_idx])

    return preds


def eval_kernel_trick(
    *,
    z_codes: np.ndarray,
    decoder_map: np.ndarray,
    source_indices: list[np.ndarray],
    target_indices: list[np.ndarray],
    k: int,
    batch_size: int,
    device: str,
) -> dict[str, float]:
    z = torch.from_numpy(z_codes.astype(np.float32)).to(device)
    w = torch.from_numpy(decoder_map.astype(np.float32)).to(device)
    pred_ranked = _compute_topk_kernel_trick(z, w, source_indices, k=k, batch_size=batch_size)
    target_sets = [set(x.tolist()) for x in target_indices]
    return {
        f"recall@{k}": _calibrated_recall(target_sets, pred_ranked, k),
        f"ndcg@{k}": _ndcg(target_sets, pred_ranked, k),
        "n_eval_users": float(len(target_sets)),
    }


def main():
    args = parse_args()
    device = resolve_device(args.device)

    ckpt = load_recsys_checkpoint(args.checkpoint_path)
    elsa_embs = ckpt["item_embeddings"].astype(np.float32)
    test_source_indices = ckpt["test_source_indices"]
    test_target_indices = ckpt["test_target_indices"]

    # 1) ELSA baseline
    m1_20 = evaluate_item_embeddings_with_holdout(
        item_embeddings=elsa_embs,
        source_indices=test_source_indices,
        target_indices=test_target_indices,
        k=20,
        score_batch_size=args.eval_batch_size,
    )
    m1_50 = evaluate_item_embeddings_with_holdout(
        item_embeddings=elsa_embs,
        source_indices=test_source_indices,
        target_indices=test_target_indices,
        k=50,
        score_batch_size=args.eval_batch_size,
    )
    m1_100 = evaluate_item_embeddings_with_holdout(
        item_embeddings=elsa_embs,
        source_indices=test_source_indices,
        target_indices=test_target_indices,
        k=100,
        score_batch_size=args.eval_batch_size,
    )
    elsa_metrics = {
        "recall@20": m1_20["recall@20"],
        "recall@50": m1_50["recall@50"],
        "ndcg@100": m1_100["ndcg@100"],
    }

    # 2) Sparse SRP embeddings (codes)
    srp_codes = load_srp_tensor(args.sae_sparse_path)
    z_codes = srp_codes.to_dense().detach().cpu().numpy().astype(np.float32)
    m2_20 = evaluate_item_embeddings_with_holdout(
        item_embeddings=z_codes,
        source_indices=test_source_indices,
        target_indices=test_target_indices,
        k=20,
        score_batch_size=args.eval_batch_size,
    )
    m2_50 = evaluate_item_embeddings_with_holdout(
        item_embeddings=z_codes,
        source_indices=test_source_indices,
        target_indices=test_target_indices,
        k=50,
        score_batch_size=args.eval_batch_size,
    )
    m2_100 = evaluate_item_embeddings_with_holdout(
        item_embeddings=z_codes,
        source_indices=test_source_indices,
        target_indices=test_target_indices,
        k=100,
        score_batch_size=args.eval_batch_size,
    )
    srp_metrics = {
        "recall@20": m2_20["recall@20"],
        "recall@50": m2_50["recall@50"],
        "ndcg@100": m2_100["ndcg@100"],
    }

    # 3) Sparse SRP + decoder kernel trick
    model_blob = torch.load(Path(args.sae_model_path), map_location="cpu", weights_only=False)
    cfg = model_blob.get("config", {})
    hidden_dim = int(model_blob.get("hidden_dim", cfg.get("sae_hidden_dim", z_codes.shape[1])))
    k = int(model_blob.get("k", cfg.get("sae_k", 128)))
    score_mode = str(model_blob.get("score_mode", cfg.get("sae_score_mode", "abs")))
    ste_alpha = float(model_blob.get("ste_alpha", cfg.get("sae_ste_alpha", 0.0)))
    decoder_bias = "decoder.bias" in model_blob["model_state_dict"]

    sae = TopKSAE(
        input_dim=elsa_embs.shape[1],
        hidden_dim=hidden_dim,
        k=k,
        decoder_bias=decoder_bias,
        sparsify_score_mode=score_mode,
        sparsify_ste_alpha=ste_alpha,
    )
    sae.load_state_dict(model_blob["model_state_dict"], strict=True)
    sae.eval()

    if sae.tied:
        decoder_map = sae.encoder.weight.detach().cpu().numpy().astype(np.float32)  # (H,D)
    else:
        decoder_map = sae.decoder.weight.detach().cpu().numpy().astype(np.float32).T  # (H,D)
    kernel = decoder_map @ decoder_map.T

    m3_20 = eval_kernel_trick(
        z_codes=z_codes,
        decoder_map=decoder_map,
        source_indices=test_source_indices,
        target_indices=test_target_indices,
        k=20,
        batch_size=args.eval_batch_size,
        device=device,
    )
    m3_50 = eval_kernel_trick(
        z_codes=z_codes,
        decoder_map=decoder_map,
        source_indices=test_source_indices,
        target_indices=test_target_indices,
        k=50,
        batch_size=args.eval_batch_size,
        device=device,
    )
    m3_100 = eval_kernel_trick(
        z_codes=z_codes,
        decoder_map=decoder_map,
        source_indices=test_source_indices,
        target_indices=test_target_indices,
        k=100,
        batch_size=args.eval_batch_size,
        device=device,
    )
    kernel_metrics = {
        "recall@20": m3_20["recall@20"],
        "recall@50": m3_50["recall@50"],
        "ndcg@100": m3_100["ndcg@100"],
    }

    compressed_elsa_srp_metrics = None
    compressed_elsa_srp_mb = None
    if args.compressed_elsa_srp_path:
        compressed_srp = load_srp_tensor(args.compressed_elsa_srp_path)
        compressed_srp_dense = compressed_srp.to_dense().detach().cpu().numpy().astype(np.float32)
        c20 = evaluate_item_embeddings_with_holdout(
            item_embeddings=compressed_srp_dense,
            source_indices=test_source_indices,
            target_indices=test_target_indices,
            k=20,
            score_batch_size=args.eval_batch_size,
        )
        c50 = evaluate_item_embeddings_with_holdout(
            item_embeddings=compressed_srp_dense,
            source_indices=test_source_indices,
            target_indices=test_target_indices,
            k=50,
            score_batch_size=args.eval_batch_size,
        )
        c100 = evaluate_item_embeddings_with_holdout(
            item_embeddings=compressed_srp_dense,
            source_indices=test_source_indices,
            target_indices=test_target_indices,
            k=100,
            score_batch_size=args.eval_batch_size,
        )
        compressed_elsa_srp_metrics = {
            "recall@20": c20["recall@20"],
            "recall@50": c50["recall@50"],
            "ndcg@100": c100["ndcg@100"],
        }
        compressed_elsa_srp_mb = bytes_to_mb(
            int(compressed_srp.cols.numel() * compressed_srp.cols.element_size())
            + int(compressed_srp.vals.numel() * compressed_srp.vals.element_size())
        )

    print("ELSA metrics:", select_metrics(elsa_metrics))
    print("SRP sparse code metrics:", select_metrics(srp_metrics))
    print("SRP + decoder kernel-trick metrics:", select_metrics(kernel_metrics))
    if compressed_elsa_srp_metrics is not None:
        print("CompressedELSA SRP metrics:", select_metrics(compressed_elsa_srp_metrics))
    print(
        "Perf drop vs ELSA (SRP sparse): "
        f"recall@20={percent_drop(elsa_metrics['recall@20'], srp_metrics['recall@20']):.2f}% "
        f"recall@50={percent_drop(elsa_metrics['recall@50'], srp_metrics['recall@50']):.2f}% "
        f"ndcg@100={percent_drop(elsa_metrics['ndcg@100'], srp_metrics['ndcg@100']):.2f}%"
    )
    print(
        "Perf drop vs ELSA (kernel trick): "
        f"recall@20={percent_drop(elsa_metrics['recall@20'], kernel_metrics['recall@20']):.2f}% "
        f"recall@50={percent_drop(elsa_metrics['recall@50'], kernel_metrics['recall@50']):.2f}% "
        f"ndcg@100={percent_drop(elsa_metrics['ndcg@100'], kernel_metrics['ndcg@100']):.2f}%"
    )
    if compressed_elsa_srp_metrics is not None:
        print(
            "Perf drop vs ELSA (CompressedELSA SRP): "
            f"recall@20={percent_drop(elsa_metrics['recall@20'], compressed_elsa_srp_metrics['recall@20']):.2f}% "
            f"recall@50={percent_drop(elsa_metrics['recall@50'], compressed_elsa_srp_metrics['recall@50']):.2f}% "
            f"ndcg@100={percent_drop(elsa_metrics['ndcg@100'], compressed_elsa_srp_metrics['ndcg@100']):.2f}%"
        )
    elsa_mb = bytes_to_mb(int(elsa_embs.nbytes))
    srp_mb = bytes_to_mb(
        int(srp_codes.cols.numel() * srp_codes.cols.element_size())
        + int(srp_codes.vals.numel() * srp_codes.vals.element_size())
    )
    kernel_mb = bytes_to_mb(int(kernel.nbytes))
    print(
        "Inference size (MB): "
        f"ELSA_dense={elsa_mb:.2f} "
        f"SRP_sparse={srp_mb:.2f} "
        f"kernel_K={kernel_mb:.2f}"
    )
    if compressed_elsa_srp_mb is not None:
        print(f"Inference size (MB) CompressedELSA SRP: {compressed_elsa_srp_mb:.2f}")


if __name__ == "__main__":
    main()
