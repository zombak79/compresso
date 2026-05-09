from __future__ import annotations

import argparse
import random

import numpy as np
import torch

from compresso.examples.checkpoint import save_recsys_checkpoint
from compresso.examples.datasets import MovieLens20M
from compresso.examples.models.elsa import fit_elsa
from compresso.examples.retrieval import build_eval_holdout, evaluate_item_embeddings_with_holdout


def eval_three_metrics(item_embs, source_indices, target_indices, eval_batch_size):
    out = {}
    for k in (20, 50, 100):
        m = evaluate_item_embeddings_with_holdout(
            item_embeddings=item_embs,
            source_indices=source_indices,
            target_indices=target_indices,
            k=k,
            score_batch_size=eval_batch_size,
        )
        out.update({kk: vv for kk, vv in m.items() if kk != "n_eval_users"})
    return {
        "recall@20": out.get("recall@20", 0.0),
        "recall@50": out.get("recall@50", 0.0),
        "ndcg@100": out.get("ndcg@100", 0.0),
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--checkpoint_path", type=str, default="artifacts/ml20m/elsa_checkpoint.npz")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_users", type=int, default=2500)
    p.add_argument("--test_users", type=int, default=5000)
    p.add_argument("--min_user_support", type=int, default=5)
    p.add_argument("--item_min_support", type=int, default=1)
    p.add_argument("--min_value_to_keep", type=float, default=4.0)
    p.add_argument("--set_all_values_to", type=float, default=1.0)

    p.add_argument("--elsa_dim", type=int, default=1024)
    p.add_argument("--elsa_epochs", type=int, default=10)
    p.add_argument("--elsa_batch_size", type=int, default=1024)
    p.add_argument("--elsa_lr", type=float, default=0.1)
    p.add_argument("--elsa_weight_decay", type=float, default=0.0)
    p.add_argument("--elsa_beta1", type=float, default=0.9)
    p.add_argument("--elsa_beta2", type=float, default=0.999)
    p.add_argument("--elsa_eps", type=float, default=1e-8)
    p.add_argument("--elsa_momentum_decay", type=float, default=0.004)
    p.add_argument("--elsa_decoupled_weight_decay", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--elsa_grad_clip_norm", type=float, default=None)
    p.add_argument("--elsa_grad_accum_steps", type=int, default=1)
    p.add_argument("--elsa_use_ema", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--elsa_ema_momentum", type=float, default=0.99)
    p.add_argument("--elsa_ema_overwrite_frequency", type=int, default=150)

    p.add_argument("--eval_batch_size", type=int, default=1024)
    p.add_argument("--eval_fold", type=int, default=0, choices=[0, 1])
    p.add_argument("--device", type=str, default="mps")
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


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)

    ds = MovieLens20M(data_dir=args.data_dir)
    raw_df = ds.get_interactions()
    proc_df = ds.preprocess_interactions_for_recsys(
        raw_df,
        min_value_to_keep=args.min_value_to_keep,
        user_min_support=args.min_user_support,
        item_min_support=args.item_min_support,
        set_all_values_to=args.set_all_values_to,
    )
    split = ds.split_users_strong_generalization(
        val_users=args.val_users,
        test_users=args.test_users,
        min_user_support=1,
        random_state=args.seed,
        interactions=proc_df,
    )

    x_train, _, item_ids = ds.to_sparse_matrix(split.train)
    val_holdout = build_eval_holdout(
        train_item_ids=item_ids,
        eval_interactions=split.val,
        min_user_support=args.min_user_support,
        random_state=args.seed,
        eval_fold=args.eval_fold,
    )

    def _elsa_val_callback(model):
        embs = model.export_item_embeddings()
        return eval_three_metrics(
            embs,
            val_holdout["source_indices"],
            val_holdout["target_indices"],
            args.eval_batch_size,
        )

    elsa = fit_elsa(
        x_train,
        n_factors=args.elsa_dim,
        epochs=args.elsa_epochs,
        batch_size=args.elsa_batch_size,
        lr=args.elsa_lr,
        weight_decay=args.elsa_weight_decay,
        beta1=args.elsa_beta1,
        beta2=args.elsa_beta2,
        eps=args.elsa_eps,
        momentum_decay=args.elsa_momentum_decay,
        decoupled_weight_decay=args.elsa_decoupled_weight_decay,
        grad_clip_norm=args.elsa_grad_clip_norm,
        grad_accum_steps=args.elsa_grad_accum_steps,
        use_ema=args.elsa_use_ema,
        ema_momentum=args.elsa_ema_momentum,
        ema_overwrite_frequency=args.elsa_ema_overwrite_frequency,
        device=device,
        val_callback=_elsa_val_callback,
    )
    item_embs = elsa.export_item_embeddings()

    test_holdout = build_eval_holdout(
        train_item_ids=item_ids,
        eval_interactions=split.test,
        min_user_support=args.min_user_support,
        random_state=args.seed,
        eval_fold=args.eval_fold,
    )

    metrics = eval_three_metrics(
        item_embs,
        test_holdout["source_indices"],
        test_holdout["target_indices"],
        args.eval_batch_size,
    )
    print("ELSA checkpoint metrics:", metrics)

    ckpt = save_recsys_checkpoint(
        args.checkpoint_path,
        item_ids=test_holdout["item_ids"],
        item_embeddings=item_embs,
        val_source_indices=val_holdout["source_indices"],
        val_target_indices=val_holdout["target_indices"],
        test_source_indices=test_holdout["source_indices"],
        test_target_indices=test_holdout["target_indices"],
    )
    print(f"Saved checkpoint to: {ckpt}")


if __name__ == "__main__":
    main()
