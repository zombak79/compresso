#!/usr/bin/env python
"""Regenerate every figure used in the Compresso documentation.

This script is *not* run during the Sphinx/CI build. The rendered PNGs live in
``docs/source/_static/`` and are committed to the repository, so building the
docs needs no datasets, models, or GPUs. Run this only when you want to
regenerate the figures::

    pip install -r docs/requirements.txt
    python docs/gen_figures.py            # both figure sets
    python docs/gen_figures.py --basic    # only the digit/SAE figures
    python docs/gen_figures.py --recsys   # only the Goodbooks clustering figures

Everything is deterministic (fixed seeds). Downloads and intermediate tensors
are cached under ``data/`` (git-ignored), so re-runs are cheap.

Extra, docs-only dependencies (see ``docs/requirements.txt``):
``matplotlib``, ``datasets``, ``sentence-transformers``, ``scikit-learn``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "source" / "_static"
DATA = ROOT / "data"
# Keep HuggingFace downloads inside the (git-ignored) project data dir.
os.environ.setdefault("HF_HOME", str(DATA / "hf_cache"))

OUT.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# ----------------------------------------------------------------------------
# Basic example: a top-k SAE as a learned, interpretable dictionary over images
# ----------------------------------------------------------------------------
def make_basic_figures() -> None:
    from datasets import load_dataset

    from compresso import TopKSAEConfig, TopKSAETrainer

    n = 20_000
    ds = load_dataset("ylecun/mnist", split="train")
    imgs = (
        np.stack([np.asarray(im, dtype=np.float32) for im in ds["image"][:n]]) / 255.0
    )
    X = imgs.reshape(n, -1)  # (n, 784) dense embeddings in [0, 1]

    cfg = TopKSAEConfig(
        hidden_dim=196,
        k=20,
        batch_size=512,
        epochs=60,
        lr=1e-3,
        decay=True,
        sparsify_score_mode="abs",
        sparsify_ste_alpha=0.01,
        device="cpu",
        show_progress=False,
        seed=0,
    )
    trainer = TopKSAETrainer(cfg).fit(X)
    h = trainer.history
    print(
        "basic: final mse=%.4f cos=%.4f dead=%.0f/196"
        % (
            h[-1]["reconstruction_mse"],
            1 - h[-1]["cosine_loss"],
            h[-1]["dead_features"],
        )
    )

    # 1) training curve
    ep = [r["epoch"] for r in h]
    mse = [r["reconstruction_mse"] for r in h]
    cos = [1 - r["cosine_loss"] for r in h]
    fig, ax1 = plt.subplots(figsize=(6, 3.5))
    ax1.plot(ep, mse, color="#c0392b")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("reconstruction MSE", color="#c0392b")
    ax2 = ax1.twinx()
    ax2.plot(ep, cos, color="#2471a3")
    ax2.set_ylabel("cosine similarity", color="#2471a3")
    ax1.set_title("Top-k SAE training on digit images (MNIST)")
    fig.tight_layout()
    fig.savefig(OUT / "basic_training_curve.png", dpi=110)
    plt.close(fig)

    # 2) reconstructions
    recon = trainer.reconstruct(X[:10]).numpy().reshape(-1, 28, 28)
    fig, axes = plt.subplots(2, 10, figsize=(10, 2.2))
    for i in range(10):
        axes[0, i].imshow(X[i].reshape(28, 28), cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(recon[i].clip(0, 1), cmap="gray")
        axes[1, i].axis("off")
    fig.suptitle("Original (top) vs sparse reconstruction (bottom)")
    fig.tight_layout()
    fig.savefig(OUT / "basic_reconstructions.png", dpi=110)
    plt.close(fig)

    # 3) learned dictionary atoms (decoder columns), most-used first
    W = trainer.sae.get_decoder_weight().detach().cpu().numpy()  # (784, 196)
    codes = trainer.encode(X[:5000]).numpy()
    order = np.argsort(-(codes != 0).mean(0))
    fig, axes = plt.subplots(8, 12, figsize=(10, 6.8))
    for j, ax in enumerate(axes.flat):
        a = W[:, order[j]].reshape(28, 28)
        m = np.abs(a).max() + 1e-9
        ax.imshow(a, cmap="RdBu_r", vmin=-m, vmax=m)
        ax.axis("off")
    fig.suptitle("Learned dictionary atoms (decoder columns), most-used first")
    fig.tight_layout()
    fig.savefig(OUT / "basic_atoms.png", dpi=110)
    plt.close(fig)

    # 4) one digit as an additive sum of its active atoms
    i = 0
    code = trainer.encode(X[i : i + 1]).numpy()[0]
    active = np.where(code != 0)[0]
    active = active[np.argsort(-np.abs(code[active]))][:6]
    fig, axes = plt.subplots(1, len(active) + 2, figsize=(2 * (len(active) + 2), 2.2))
    axes[0].imshow(X[i].reshape(28, 28), cmap="gray")
    axes[0].set_title("input")
    axes[0].axis("off")
    acc = np.zeros(784)
    for t, j in enumerate(active):
        contrib = W[:, j] * code[j]
        acc = acc + contrib
        axes[t + 1].imshow(contrib.reshape(28, 28), cmap="RdBu_r")
        axes[t + 1].set_title(f"atom {j}\nw={code[j]:.2f}", fontsize=8)
        axes[t + 1].axis("off")
    axes[-1].imshow(acc.reshape(28, 28).clip(0, 1), cmap="gray")
    axes[-1].set_title("sum")
    axes[-1].axis("off")
    fig.suptitle("A digit reconstructed as a sum of a few active atoms")
    fig.tight_layout()
    fig.savefig(OUT / "basic_additive.png", dpi=110)
    plt.close(fig)
    print("basic: wrote 4 figures")


# ----------------------------------------------------------------------------
# Recsys example: sparse codes of book embeddings -> interpretable clusters
# ----------------------------------------------------------------------------
def _goodbooks_inputs():
    """Return (emb, tag_matrix, tag_names, titles). Embeddings are cached."""
    import pandas as pd
    from scipy.sparse import csr_matrix

    sys.path.insert(0, str(ROOT / "examples"))
    from recsys_lib.datasets.goodbooks import Goodbooks

    cache = DATA / "goodbooks_cache"
    cache.mkdir(parents=True, exist_ok=True)
    gb = Goodbooks(data_dir=str(DATA))
    gb.download()
    root = gb.root
    books = pd.read_csv(root / "books.csv")
    meta = gb.get_item_metadata()
    n = len(meta)
    titles = meta["title"].astype(str).to_numpy()

    emb_path = cache / "emb.npy"
    if emb_path.exists():
        emb = np.load(emb_path)
    else:
        from sentence_transformers import SentenceTransformer

        text = (
            meta["title"].fillna("")
            + ". by "
            + meta["authors"].fillna("")
            + ". "
            + meta["description"].fillna("")
        ).tolist()
        model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2", device="cpu"
        )
        emb = model.encode(
            text, batch_size=128, show_progress_bar=False, normalize_embeddings=True
        ).astype("float32")
        np.save(emb_path, emb)
    print("recsys: embeddings", emb.shape)

    # Build an entity-tag matrix from Goodreads shelf tags (filtered to themes).
    bt = pd.read_csv(root / "book_tags.csv")
    tg = pd.read_csv(root / "tags.csv")
    gid2row = {int(g): i for i, g in enumerate(books["goodreads_book_id"].to_numpy())}
    tagid2name = dict(zip(tg.tag_id, tg.tag_name))
    generic = {
        "to-read",
        "currently-reading",
        "favorites",
        "books-i-own",
        "owned",
        "to-buy",
        "library",
        "default",
        "books",
        "ebook",
        "kindle",
        "my-books",
        "owned-books",
        "wish-list",
        "re-read",
        "favourites",
        "i-own",
        "audiobook",
        "audiobooks",
        "ebooks",
        "series",
        "novels",
        "book-club",
        "favorite-books",
        "all-time-favorites",
        "my-library",
        "general",
        "finished",
        "default-shelf",
    }
    totals = bt.groupby("tag_id")["count"].sum().sort_values(ascending=False)
    sel = []
    for tid in totals.index:
        name = str(tagid2name.get(tid, "")).strip()
        if not name or name in generic or len(name) < 3:
            continue
        if any(ch.isdigit() for ch in name) and name.replace("-", "").isdigit():
            continue
        sel.append(tid)
        if len(sel) >= 150:
            break
    tagcol = {tid: j for j, tid in enumerate(sel)}
    tag_names = [tagid2name[t] for t in sel]
    rows, cols, vals = [], [], []
    btf = bt[bt.tag_id.isin(set(sel))]
    for gid, tid, cnt in zip(
        btf.goodreads_book_id.to_numpy(), btf.tag_id.to_numpy(), btf["count"].to_numpy()
    ):
        r = gid2row.get(int(gid))
        if r is None:
            continue
        rows.append(r)
        cols.append(tagcol[tid])
        vals.append(float(cnt))
    tags = csr_matrix((vals, (rows, cols)), shape=(n, len(sel)), dtype=np.float32)
    return emb, tags, tag_names, titles


def make_recsys_figures() -> None:
    import compresso.clustering as cc
    from compresso import TopKSAEConfig, TopKSAETrainer

    emb, tags, tag_names, titles = _goodbooks_inputs()

    cfg = TopKSAEConfig(
        hidden_dim=1024,
        k=32,
        batch_size=512,
        epochs=80,
        lr=1e-3,
        decay=True,
        sparsify_score_mode="abs",
        sparsify_ste_alpha=0.01,
        device="cpu",
        show_progress=False,
        seed=0,
    )
    srp = TopKSAETrainer(cfg).fit_transform(emb)
    print("recsys: srp", srp.shape, "k", srp.k)

    clusters = cc.ClusteringPipeline(
        [
            cc.TopMSignedClustering(top_m=2, min_cluster_size=10),
            cc.EntityContainmentLink(threshold=0.9),
            cc.MaterializeLinkMerges(parent_scope="active"),
            cc.PruneRedundantRoots(),
            cc.AssignTags(
                entity_tag_matrix=tags, tag_names=tag_names, method="tfidf", top_k=8
            ),
            cc.SizeFilter(min_cluster_size=15),
        ]
    ).fit(srp)
    active = sorted(clusters.active_clusters, key=lambda c: -c.entity_count)
    print("recsys: active clusters", len(active))

    # A) cluster size distribution
    sizes = [c.entity_count for c in active]
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.bar(range(min(40, len(sizes))), sizes[:40], color="#2471a3")
    ax.set_xlabel("active cluster (sorted by size)")
    ax.set_ylabel("books in cluster")
    ax.set_title(
        f"{len(active)} interpretable clusters from sparse codes (top 40 by size)"
    )
    fig.tight_layout()
    fig.savefig(OUT / "recsys_cluster_sizes.png", dpi=110)
    plt.close(fig)

    # B) tag profiles of the largest themes
    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    for ax, c in zip(axes.flat, active[:6]):
        names = [t.name for t in c.tags[:6]][::-1]
        scores = [t.score for t in c.tags[:6]][::-1]
        ax.barh(names, scores, color="#16a085")
        ax.set_title(f"{c.cluster_id}\n({c.entity_count} books)", fontsize=9)
        ax.tick_params(labelsize=8)
    fig.suptitle("Each sparse feature becomes a themed cluster (top tf-idf shelf tags)")
    fig.tight_layout()
    fig.savefig(OUT / "recsys_cluster_tags.png", dpi=110)
    plt.close(fig)

    # C) 2D map of the dense embeddings colored by sparse-code cluster
    from sklearn.manifold import TSNE

    xy = TSNE(n_components=2, init="pca", perplexity=30, random_state=0).fit_transform(
        emb
    )
    assign = -np.ones(len(emb), dtype=int)
    for ci, c in enumerate(active[:8]):
        for e in c.entity_indices:
            if assign[e] == -1:
                assign[e] = ci

    def distinct_label(c):
        seen = []
        for t in c.tags:
            if t.name not in ("fiction", "fantasy") and t.name not in seen:
                seen.append(t.name)
            if len(seen) == 2:
                break
        return ", ".join(seen) if seen else (c.tags[0].name if c.tags else c.cluster_id)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(
        xy[assign == -1, 0], xy[assign == -1, 1], s=3, c="#dddddd", label="other"
    )
    cmap = plt.get_cmap("tab10")
    for ci, c in enumerate(active[:8]):
        m = assign == ci
        ax.scatter(xy[m, 0], xy[m, 1], s=7, color=cmap(ci), label=distinct_label(c))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Book embeddings (t-SNE) colored by sparse-code cluster")
    ax.legend(markerscale=2, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(OUT / "recsys_embedding_map.png", dpi=110)
    plt.close(fig)
    print("recsys: wrote 3 figures")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--basic", action="store_true", help="only the digit/SAE figures")
    ap.add_argument("--recsys", action="store_true", help="only the Goodbooks figures")
    args = ap.parse_args()
    run_basic = args.basic or not args.recsys
    run_recsys = args.recsys or not args.basic
    if run_basic:
        make_basic_figures()
    if run_recsys:
        make_recsys_figures()


if __name__ == "__main__":
    main()
