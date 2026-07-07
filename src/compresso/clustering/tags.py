from __future__ import annotations

from typing import Literal, Sequence

import numpy as np
from scipy import sparse

from .types import ScoredTag, SparseClusterSet


def _to_csr(x) -> sparse.csr_matrix:
    if sparse.issparse(x):
        return x.tocsr().astype(np.float32)
    return sparse.csr_matrix(np.asarray(x, dtype=np.float32))


def cluster_entity_matrix(clusters: SparseClusterSet) -> sparse.csr_matrix:
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for row, cluster in enumerate(clusters.clusters):
        rows.extend([row] * cluster.entity_count)
        cols.extend(cluster.entity_indices.tolist())
        data.extend([1.0] * cluster.entity_count)
    return sparse.csr_matrix((data, (rows, cols)), shape=(len(clusters.clusters), clusters.n_entities), dtype=np.float32)


def compute_cluster_tag_counts(clusters: SparseClusterSet, entity_tag_matrix) -> sparse.csr_matrix:
    tags = _to_csr(entity_tag_matrix)
    if tags.shape[0] != clusters.n_entities:
        raise ValueError("entity_tag_matrix must have shape (n_entities, n_tags)")
    return (cluster_entity_matrix(clusters) @ tags).tocsr()


def tfidf_score(counts: sparse.csr_matrix) -> np.ndarray:
    counts = counts.tocsr().astype(np.float32)
    dense = counts.toarray()
    row_sum = dense.sum(axis=1, keepdims=True)
    tf = np.divide(dense, row_sum, out=np.zeros_like(dense), where=row_sum > 0)
    df = np.count_nonzero(dense > 0, axis=0)
    n_docs = dense.shape[0]
    idf = np.log((n_docs + 1.0) / (df + 1.0)) + 1.0
    return tf * idf.reshape(1, -1)


def assign_cluster_tags(
    clusters: SparseClusterSet,
    entity_tag_matrix,
    tag_names: Sequence[str],
    *,
    method: Literal["tfidf", "counts"] = "tfidf",
    top_k: int = 5,
    min_score: float = 0.0,
) -> SparseClusterSet:
    """Assign top tags to clusters from an entity-tag matrix."""
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    counts = compute_cluster_tag_counts(clusters, entity_tag_matrix)
    if counts.shape[1] != len(tag_names):
        raise ValueError("tag_names length must equal entity_tag_matrix.shape[1]")
    scores = tfidf_score(counts) if method == "tfidf" else counts.toarray().astype(np.float32)
    if method not in {"tfidf", "counts"}:
        raise ValueError("method must be 'tfidf' or 'counts'")
    count_dense = counts.toarray()

    updated = []
    for row, cluster in enumerate(clusters.clusters):
        order = np.argsort(scores[row])[::-1]
        tags: list[ScoredTag] = []
        for tag_idx in order[:top_k]:
            score = float(scores[row, tag_idx])
            if score <= min_score:
                continue
            tags.append(
                ScoredTag(
                    tag_id=int(tag_idx),
                    name=str(tag_names[tag_idx]),
                    score=score,
                    count=float(count_dense[row, tag_idx]),
                    metadata={"method": method},
                )
            )
        updated.append(cluster.with_updates(tags=tuple(tags)))

    return clusters.with_clusters(
        updated,
        history_entry={
            "phase": "assign_cluster_tags",
            "method": method,
            "top_k": top_k,
            "min_score": min_score,
        },
    )
