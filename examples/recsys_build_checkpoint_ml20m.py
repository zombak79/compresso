from __future__ import annotations

import argparse
import random

import numpy as np

from compresso.examples.checkpoint import save_recsys_split, update_checkpoint
from compresso.examples.datasets import MovieLens20M
from compresso.examples.retrieval import build_eval_holdout


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--checkpoint_path", type=str, default="artifacts/ml20m/recsys_checkpoint.zip")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_users", type=int, default=2500)
    p.add_argument("--test_users", type=int, default=5000)
    p.add_argument("--min_user_support", type=int, default=5)
    p.add_argument("--item_min_support", type=int, default=1)
    p.add_argument("--min_value_to_keep", type=float, default=4.0)
    p.add_argument("--set_all_values_to", type=float, default=1.0)
    p.add_argument("--eval_fold", type=int, default=0, choices=[0, 1])
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

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
    test_holdout = build_eval_holdout(
        train_item_ids=item_ids,
        eval_interactions=split.test,
        min_user_support=args.min_user_support,
        random_state=args.seed,
        eval_fold=args.eval_fold,
    )

    with update_checkpoint(args.checkpoint_path) as root:
        save_recsys_split(
            root,
            item_ids=test_holdout["item_ids"],
            x_train=x_train,
            val_source_indices=val_holdout["source_indices"],
            val_target_indices=val_holdout["target_indices"],
            test_source_indices=test_holdout["source_indices"],
            test_target_indices=test_holdout["target_indices"],
            metadata={
                "dataset": "ml20m",
                "seed": args.seed,
                "val_users": args.val_users,
                "test_users": args.test_users,
                "min_user_support": args.min_user_support,
                "item_min_support": args.item_min_support,
                "min_value_to_keep": args.min_value_to_keep,
                "set_all_values_to": args.set_all_values_to,
                "eval_fold": args.eval_fold,
            },
        )
    print(f"Saved data split checkpoint to: {args.checkpoint_path}")


if __name__ == "__main__":
    main()
