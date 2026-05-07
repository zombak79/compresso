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
    p.add_argument("--val_users", type=int, default=10000)
    p.add_argument("--test_users", type=int, default=10000)
    p.add_argument("--min_user_support", type=int, default=5)
    p.add_argument("--item_min_support", type=int, default=1)
    p.add_argument("--min_value_to_keep", type=float, default=4.0)
    p.add_argument("--set_all_values_to", type=float, default=1.0)

    p.add_argument("--elsa_dim", type=int, default=512)
    p.add_argument("--elsa_epochs", type=int, default=10)
    p.add_argument("--elsa_batch_size", type=int, default=1024)
    p.add_argument("--elsa_lr", type=float, default=0.1)
    p.add_argument("--elsa_weight_decay", type=float, default=0.0)

    p.add_argument("--eval_batch_size", type=int, default=1024)
    p.add_argument("--eval_fold", type=int, default=0, choices=[0, 1])
    p.add_argument("--device", type=str, default="mps")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

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
        device=args.device,
        val_callback=_elsa_val_callback,
    )
    item_embs = elsa.export_item_embeddings()

    holdout = build_eval_holdout(
        train_item_ids=item_ids,
        eval_interactions=split.test,
        min_user_support=args.min_user_support,
        random_state=args.seed,
        eval_fold=args.eval_fold,
    )

    metrics = eval_three_metrics(
        item_embs,
        holdout["source_indices"],
        holdout["target_indices"],
        args.eval_batch_size,
    )
    print("ELSA checkpoint metrics:", metrics)

    ckpt = save_recsys_checkpoint(
        args.checkpoint_path,
        item_ids=holdout["item_ids"],
        item_embeddings=item_embs,
        source_indices=holdout["source_indices"],
        target_indices=holdout["target_indices"],
    )
    print(f"Saved checkpoint to: {ckpt}")


if __name__ == "__main__":
    main()
