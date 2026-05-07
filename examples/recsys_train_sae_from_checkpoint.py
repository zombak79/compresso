from __future__ import annotations

import argparse
import random

import numpy as np
import torch

from compresso.examples.checkpoint import load_recsys_checkpoint
from compresso.examples.retrieval import evaluate_item_embeddings_with_holdout
from compresso.examples.runners import TwoStagePipeline


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_path", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--sae_hidden_dim", type=int, default=4096)
    p.add_argument("--sae_k", type=int, default=128)
    p.add_argument("--sae_k_backward", type=int, default=None)
    p.add_argument("--sae_mode", type=str, default="values")
    p.add_argument("--sae_score_mode", type=str, default="abs", choices=["abs", "raw", "relu"])
    p.add_argument("--sae_loss", type=str, default="mse", choices=["mse", "cosine"])
    p.add_argument("--sae_epochs", type=int, default=10)
    p.add_argument("--sae_batch_size", type=int, default=1024)
    p.add_argument("--sae_lr", type=float, default=1e-3)

    p.add_argument("--device", type=str, default="mps")
    p.add_argument("--eval_batch_size", type=int, default=1024)
    p.add_argument("--eval_debug", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--eval_debug_users", type=int, default=10)
    return p.parse_args()


def select_metrics(metrics: dict) -> dict:
    return {
        "recall@20": metrics.get("recall@20", 0.0),
        "recall@50": metrics.get("recall@50", 0.0),
        "ndcg@100": metrics.get("ndcg@100", 0.0),
    }


def eval_multi_k(embs, source_indices, target_indices, *, eval_batch_size, eval_debug, eval_debug_users):
    out = {}
    debug_rows = None
    for k in (10, 20, 50, 100):
        m = evaluate_item_embeddings_with_holdout(
            item_embeddings=embs,
            source_indices=source_indices,
            target_indices=target_indices,
            k=k,
            score_batch_size=eval_batch_size,
            debug=eval_debug and k == 100,
            debug_users=eval_debug_users,
        )
        out.update({kk: vv for kk, vv in m.items() if kk != "n_eval_users"})
        if "debug" in m:
            debug_rows = m["debug"]
    if debug_rows is not None:
        out["debug_ndcg@100"] = debug_rows
    return out


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ckpt = load_recsys_checkpoint(args.checkpoint_path)
    item_embs = ckpt["item_embeddings"]
    source_indices = ckpt["source_indices"]
    target_indices = ckpt["target_indices"]

    base_all = eval_multi_k(
        item_embs,
        source_indices,
        target_indices,
        eval_batch_size=args.eval_batch_size,
        eval_debug=args.eval_debug,
        eval_debug_users=args.eval_debug_users,
    )
    print("Original embedding metrics:", select_metrics(base_all))
    if args.eval_debug and "debug_ndcg@100" in base_all:
        print("Original ndcg@100 debug rows:", base_all["debug_ndcg@100"])

    pipeline = TwoStagePipeline(workdir="artifacts/sae_from_checkpoint")
    def _sae_val_callback(model):
        with torch.no_grad():
            x_val = torch.from_numpy(item_embs.astype(np.float32)).to(args.device)
            _, codes_val, _ = model(x_val)
            embs_val = codes_val.detach().cpu().numpy().astype(np.float32)
        m20 = evaluate_item_embeddings_with_holdout(
            item_embeddings=embs_val,
            source_indices=source_indices,
            target_indices=target_indices,
            k=20,
            score_batch_size=args.eval_batch_size,
        )
        m50 = evaluate_item_embeddings_with_holdout(
            item_embeddings=embs_val,
            source_indices=source_indices,
            target_indices=target_indices,
            k=50,
            score_batch_size=args.eval_batch_size,
        )
        m100 = evaluate_item_embeddings_with_holdout(
            item_embeddings=embs_val,
            source_indices=source_indices,
            target_indices=target_indices,
            k=100,
            score_batch_size=args.eval_batch_size,
        )
        return {
            "recall@20": m20.get("recall@20", 0.0),
            "recall@50": m50.get("recall@50", 0.0),
            "ndcg@100": m100.get("ndcg@100", 0.0),
        }

    sae = pipeline.train_sae_on_embeddings(
        item_embs,
        hidden_dim=args.sae_hidden_dim,
        k=args.sae_k,
        k_backward=args.sae_k_backward,
        sparsify_mode=args.sae_mode,
        sparsify_score_mode=args.sae_score_mode,
        epochs=args.sae_epochs,
        batch_size=args.sae_batch_size,
        lr=args.sae_lr,
        device=args.device,
        loss_type=args.sae_loss,
        val_callback=_sae_val_callback,
    )

    with torch.no_grad():
        x = torch.from_numpy(item_embs.astype(np.float32)).to(args.device)
        _, codes, stats = sae(x)
        sae_embs = codes.detach().cpu().numpy().astype(np.float32)
        dead = int(stats["dead_features"].item()) if hasattr(stats["dead_features"], "item") else int(stats["dead_features"])
        dead_frac = dead / float(args.sae_hidden_dim)
        print(f"SAE dead neurons: {dead}/{args.sae_hidden_dim} ({dead_frac:.2%})")

    sae_all = eval_multi_k(
        sae_embs,
        source_indices,
        target_indices,
        eval_batch_size=args.eval_batch_size,
        eval_debug=args.eval_debug,
        eval_debug_users=args.eval_debug_users,
    )
    print("SAE embedding metrics:", select_metrics(sae_all))
    if args.eval_debug and "debug_ndcg@100" in sae_all:
        print("SAE ndcg@100 debug rows:", sae_all["debug_ndcg@100"])


if __name__ == "__main__":
    main()
