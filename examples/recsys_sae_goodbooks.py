"""Two-stage example: ELSA embedding training -> TopKSAE compression."""

from __future__ import annotations

import argparse
import random
import numpy as np
import torch

from compresso.examples.datasets import Goodbooks
from compresso.examples.models.elsa import fit_elsa
from compresso.examples.retrieval import build_eval_holdout, evaluate_item_embeddings, evaluate_item_embeddings_with_holdout
from compresso.examples.runners import TwoStagePipeline


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--artifacts_dir", type=str, default="artifacts/goodbooks")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_users", type=int, default=1000)
    p.add_argument("--test_users", type=int, default=2500)
    p.add_argument("--min_user_support", type=int, default=5)
    p.add_argument("--item_min_support", type=int, default=1)
    p.add_argument("--min_value_to_keep", type=float, default=4.0)
    p.add_argument("--set_all_values_to", type=float, default=1.0)

    p.add_argument("--elsa_dim", type=int, default=512)
    p.add_argument("--elsa_epochs", type=int, default=10)
    p.add_argument("--elsa_batch_size", type=int, default=1024)
    p.add_argument("--elsa_lr", type=float, default=0.1)
    p.add_argument("--elsa_weight_decay", type=float, default=0.0)

    p.add_argument("--sae_hidden_dim", type=int, default=4096)
    p.add_argument("--sae_k", type=int, default=128)
    p.add_argument("--sae_epochs", type=int, default=10)
    p.add_argument("--sae_batch_size", type=int, default=1024)
    p.add_argument("--sae_lr", type=float, default=1e-3)

    p.add_argument("--device", type=str, default="mps")
    p.add_argument("--eval_batch_size", type=int, default=1024)
    p.add_argument("--eval_fold", type=int, default=0, choices=[0, 1])
    p.add_argument("--eval_debug", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--eval_debug_users", type=int, default=10)
    return p.parse_args()


def select_metrics(metrics: dict) -> dict:
    return {
        "recall@20": metrics.get("recall@20", 0.0),
        "recall@50": metrics.get("recall@50", 0.0),
        "ndcg@100": metrics.get("ndcg@100", 0.0),
    }


def eval_multi_k(
    item_ids,
    embs,
    test_df,
    *,
    eval_batch_size: int = 512,
    eval_fold: int = 0,
    random_state: int = 42,
    eval_debug: bool = False,
    eval_debug_users: int = 5,
):
    out = {}
    debug_rows = None
    for k in (10, 20, 50, 100):
        m = evaluate_item_embeddings(
            train_item_ids=item_ids,
            item_embeddings=embs,
            eval_interactions=test_df,
            k=k,
            eval_fold=eval_fold,
            score_batch_size=eval_batch_size,
            random_state=random_state,
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

    ds = Goodbooks(data_dir=args.data_dir)
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
        m20 = evaluate_item_embeddings_with_holdout(
            item_embeddings=embs,
            source_indices=val_holdout["source_indices"],
            target_indices=val_holdout["target_indices"],
            k=20,
            score_batch_size=args.eval_batch_size,
        )
        m50 = evaluate_item_embeddings_with_holdout(
            item_embeddings=embs,
            source_indices=val_holdout["source_indices"],
            target_indices=val_holdout["target_indices"],
            k=50,
            score_batch_size=args.eval_batch_size,
        )
        m100 = evaluate_item_embeddings_with_holdout(
            item_embeddings=embs,
            source_indices=val_holdout["source_indices"],
            target_indices=val_holdout["target_indices"],
            k=100,
            score_batch_size=args.eval_batch_size,
        )
        return {"recall@20": m20["recall@20"], "recall@50": m50["recall@50"], "ndcg@100": m100["ndcg@100"]}

    def _sae_val_callback(model):
        with torch.no_grad():
            x_val = torch.from_numpy(item_embs.astype(np.float32)).to(args.device)
            _, codes_val, _ = model(x_val)
            embs = codes_val.detach().cpu().numpy().astype(np.float32)
        m20 = evaluate_item_embeddings_with_holdout(
            item_embeddings=embs,
            source_indices=val_holdout["source_indices"],
            target_indices=val_holdout["target_indices"],
            k=20,
            score_batch_size=args.eval_batch_size,
        )
        m50 = evaluate_item_embeddings_with_holdout(
            item_embeddings=embs,
            source_indices=val_holdout["source_indices"],
            target_indices=val_holdout["target_indices"],
            k=50,
            score_batch_size=args.eval_batch_size,
        )
        m100 = evaluate_item_embeddings_with_holdout(
            item_embeddings=embs,
            source_indices=val_holdout["source_indices"],
            target_indices=val_holdout["target_indices"],
            k=100,
            score_batch_size=args.eval_batch_size,
        )
        return {"recall@20": m20["recall@20"], "recall@50": m50["recall@50"], "ndcg@100": m100["ndcg@100"]}

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

    elsa_all = eval_multi_k(
        item_ids,
        item_embs,
        split.test,
        eval_batch_size=args.eval_batch_size,
        eval_fold=args.eval_fold,
        random_state=args.seed,
        eval_debug=args.eval_debug,
        eval_debug_users=args.eval_debug_users,
    )
    elsa_metrics = select_metrics(elsa_all)
    print("ELSA retrieval metrics:", elsa_metrics)
    if args.eval_debug and "debug_ndcg@100" in elsa_all:
        print("ELSA ndcg@100 debug rows:", elsa_all["debug_ndcg@100"])

    pipeline = TwoStagePipeline(workdir=args.artifacts_dir)
    artifacts = pipeline.save_item_embeddings(item_embs)
    print(f"Saved stage-1 item embeddings to: {artifacts.item_embeddings_path}")

    sae = pipeline.train_sae_on_embeddings(
        item_embs,
        hidden_dim=args.sae_hidden_dim,
        k=args.sae_k,
        epochs=args.sae_epochs,
        batch_size=args.sae_batch_size,
        lr=args.sae_lr,
        device=args.device,
        val_callback=_sae_val_callback,
    )

    with torch.no_grad():
        x = torch.from_numpy(item_embs.astype(np.float32)).to(args.device)
        _, codes, _ = sae(x)
        sae_embs = codes.detach().cpu().numpy().astype(np.float32)

    sae_all = eval_multi_k(
        item_ids,
        sae_embs,
        split.test,
        eval_batch_size=args.eval_batch_size,
        eval_fold=args.eval_fold,
        random_state=args.seed,
        eval_debug=args.eval_debug,
        eval_debug_users=args.eval_debug_users,
    )
    sae_metrics = select_metrics(sae_all)
    print("SAE retrieval metrics:", sae_metrics)
    if args.eval_debug and "debug_ndcg@100" in sae_all:
        print("SAE ndcg@100 debug rows:", sae_all["debug_ndcg@100"])


if __name__ == "__main__":
    main()
