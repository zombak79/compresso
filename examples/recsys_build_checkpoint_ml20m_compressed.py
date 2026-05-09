from __future__ import annotations

import argparse
import random

import numpy as np
import torch

from compresso.examples.checkpoint import save_recsys_checkpoint
from compresso.examples.datasets import MovieLens20M
from compresso.examples.models.elsa import fit_compressed_elsa
from compresso.examples.retrieval import build_eval_holdout, evaluate_item_embeddings_with_holdout
from compresso.io import save_srp_tensor


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
    p.add_argument("--checkpoint_path", type=str, default="artifacts/ml20m/elsa_checkpoint_compressed.npz")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val_users", type=int, default=2500)
    p.add_argument("--test_users", type=int, default=5000)
    p.add_argument("--min_user_support", type=int, default=5)
    p.add_argument("--item_min_support", type=int, default=1)
    p.add_argument("--min_value_to_keep", type=float, default=4.0)
    p.add_argument("--set_all_values_to", type=float, default=1.0)

    p.add_argument("--elsa_dim", type=int, default=1024)
    p.add_argument("--sparse_k_target", type=int, default=128)
    p.add_argument("--sparse_num_stages", type=int, default=10)
    p.add_argument("--sparse_stability_window", type=int, default=5)
    p.add_argument("--sparse_change_threshold", type=float, default=0.01)
    p.add_argument("--sparse_mask_update_interval", type=int, default=10)
    p.add_argument("--sparse_score_mode", type=str, default="abs", choices=["abs", "raw", "relu"])
    p.add_argument("--sparse_ste_alpha", type=float, default=1.0)
    p.add_argument("--sparse_post_norm_l1", action=argparse.BooleanOptionalAction, default=False)

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

    def _val_callback(model):
        embs = model.export_item_embeddings()
        return eval_three_metrics(
            embs,
            val_holdout["source_indices"],
            val_holdout["target_indices"],
            args.eval_batch_size,
        )

    model = fit_compressed_elsa(
        x_train,
        n_factors=args.elsa_dim,
        k_target=args.sparse_k_target,
        num_stages=args.sparse_num_stages,
        stability_window=args.sparse_stability_window,
        change_threshold=args.sparse_change_threshold,
        score_mode=args.sparse_score_mode,
        ste_alpha=args.sparse_ste_alpha,
        post_norm_l1=args.sparse_post_norm_l1,
        mask_update_interval=args.sparse_mask_update_interval,
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
        device=device,
        val_callback=_val_callback,
    )
    item_embs = model.export_item_embeddings()

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
    print("Compressed ELSA checkpoint metrics:", metrics)

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

    srp_path = args.checkpoint_path.replace(".npz", "_compressed_elsa.srp.pt")
    save_srp_tensor(srp_path, model.A.srp())
    print(f"Saved CompressedELSA SRP weights to: {srp_path}")


if __name__ == "__main__":
    main()
