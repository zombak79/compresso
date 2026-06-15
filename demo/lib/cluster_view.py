from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy import sparse

from compresso.clustering import SparseCluster, SparseClusterSet
from compresso.params.srp import SRPTensor


def cluster_title(cluster: SparseCluster) -> str:
    return cluster.label or cluster.description or cluster.cluster_id


def cluster_summary(cluster: SparseCluster) -> dict[str, Any]:
    return {
        "cluster_id": cluster.cluster_id,
        "label": cluster_title(cluster),
        "n_items": cluster.entity_count,
        "n_children": len(cluster.child_cluster_ids),
        "n_parents": len(cluster.parent_cluster_ids),
    }


def root_cluster_table(graph: SparseClusterSet, *, min_items: int = 1) -> pd.DataFrame:
    rows = [cluster_summary(c) for c in graph.root_clusters if c.entity_count >= min_items]
    return pd.DataFrame(rows).sort_values(["label", "n_items"], ascending=[True, False], ignore_index=True)


def child_cluster_table(graph: SparseClusterSet, cluster_id: str) -> pd.DataFrame:
    rows = [cluster_summary(c) for c in graph.children(cluster_id)]
    if not rows:
        return pd.DataFrame(columns=["cluster_id", "label", "n_items", "n_children", "n_parents"])
    return pd.DataFrame(rows).sort_values(["label", "n_items"], ascending=[True, False], ignore_index=True)


def cluster_items(metadata: pd.DataFrame, cluster: SparseCluster, *, limit: int | None = None) -> pd.DataFrame:
    indices = cluster.entity_indices if limit is None else cluster.entity_indices[:limit]
    rows = metadata.iloc[indices].copy()
    preferred = ["item_id", "title", "authors", "genres", "description"]
    cols = [c for c in preferred if c in rows.columns]
    return rows[cols] if cols else rows


def cluster_features(cluster: SparseCluster, *, limit: int | None = None) -> pd.DataFrame:
    indices = cluster.centroid.indices.tolist()
    values = cluster.centroid.values.tolist()
    rows = list(zip(indices, values))
    rows.sort(key=lambda x: abs(float(x[1])), reverse=True)
    if limit is not None:
        rows = rows[:limit]
    return pd.DataFrame([{"feature": int(i), "value": float(v)} for i, v in rows])


def user_vector_from_indices(indices: np.ndarray, n_items: int) -> sparse.csr_matrix:
    indices = np.asarray(indices, dtype=np.int64)
    data = np.ones(indices.shape[0], dtype=np.float32)
    rows = np.zeros(indices.shape[0], dtype=np.int64)
    return sparse.csr_matrix((data, (rows, indices)), shape=(1, int(n_items)), dtype=np.float32)


def srp_to_csr(srp: SRPTensor) -> sparse.csr_matrix:
    cols = srp.cols.detach().cpu().numpy().astype(np.int64, copy=False)
    vals = srp.vals.detach().cpu().numpy().astype(np.float32, copy=False)
    rows = np.repeat(np.arange(srp.rows, dtype=np.int64), srp.k)
    return sparse.csr_matrix((vals.reshape(-1), (rows, cols.reshape(-1))), shape=srp.shape, dtype=np.float32)


def cluster_centroid_matrix(graph: SparseClusterSet, *, active_only: bool = True) -> tuple[sparse.csr_matrix, list[str]]:
    clusters = graph.active_clusters if active_only else graph.clusters
    data: list[float] = []
    rows: list[int] = []
    cols: list[int] = []
    ids: list[str] = []
    for row, cluster in enumerate(clusters):
        ids.append(cluster.cluster_id)
        data.extend(float(v) for v in cluster.centroid.values.tolist())
        rows.extend([row] * len(cluster.centroid.indices))
        cols.extend(int(i) for i in cluster.centroid.indices.tolist())
    mat = sparse.csr_matrix((np.asarray(data, dtype=np.float32), (rows, cols)), shape=(len(ids), graph.n_features), dtype=np.float32)
    return mat, ids


def recommend_clusters_for_user(
    source_indices: np.ndarray,
    srp: SRPTensor,
    graph: SparseClusterSet,
    *,
    top_k: int = 10,
) -> pd.DataFrame:
    item_matrix = srp_to_csr(srp)
    x = user_vector_from_indices(source_indices, srp.rows)
    user_repr = x @ item_matrix
    centroids, cluster_ids = cluster_centroid_matrix(graph, active_only=True)
    scores = np.asarray((user_repr @ centroids.T).toarray()).ravel()
    if scores.size == 0:
        return pd.DataFrame(columns=["cluster_id", "label", "score", "n_items"])
    k = min(int(top_k), scores.size)
    order = np.argpartition(-scores, k - 1)[:k]
    order = order[np.argsort(-scores[order])]
    by_id = graph.cluster_by_id
    return pd.DataFrame(
        [
            {
                "cluster_id": cluster_ids[i],
                "label": cluster_title(by_id[cluster_ids[i]]),
                "score": float(scores[i]),
                "n_items": by_id[cluster_ids[i]].entity_count,
            }
            for i in order
            if scores[i] > 0
        ]
    )


def item_card_rows(items: pd.DataFrame, *, limit: int | None = None) -> list[dict[str, str]]:
    rows = items if limit is None else items.head(limit)
    cards: list[dict[str, str]] = []
    for _, row in rows.iterrows():
        title = str(row.get("title", row.get("original_title", "")) or "")
        authors = str(row.get("authors", "") or "")
        description = str(row.get("description", "") or "")
        cards.append(
            {
                "id": str(row.get("item_id", "")),
                "title": title,
                "authors": authors,
                "description": description[:240] + ("..." if len(description) > 240 else ""),
            }
        )
    return cards
