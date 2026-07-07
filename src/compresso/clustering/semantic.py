from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Literal

import numpy as np

from .labels import _parse_label_result
from .merge import merge_cluster_group
from .types import SparseCluster, SparseClusterSet

ClusterScope = Literal["active", "all", "leaves", "roots"]


def _progress_iter(iterable, *, enabled: bool, total: int | None = None, desc: str = ""):
    if not enabled:
        return iterable
    try:
        from tqdm.auto import tqdm
    except Exception:  # pragma: no cover - optional progress dependency
        return iterable
    return tqdm(iterable, total=total, desc=desc)


def _scoped_clusters(clusters: SparseClusterSet, scope: ClusterScope) -> tuple[SparseCluster, ...]:
    if scope == "active":
        return clusters.active_clusters
    if scope == "all":
        return clusters.clusters
    if scope == "leaves":
        return clusters.leaf_clusters
    if scope == "roots":
        return clusters.root_clusters
    raise ValueError("scope must be one of 'active', 'all', 'leaves', or 'roots'")


def default_cluster_text(cluster: SparseCluster) -> str:
    parts: list[str] = []
    if cluster.label:
        parts.append(cluster.label)
    if cluster.description:
        parts.append(cluster.description)
    return "\n".join(parts)


def default_semantic_label_text(parent: SparseCluster, children: list[SparseCluster]) -> str:
    del parent
    parts = []
    for child in children:
        text = default_cluster_text(child)
        parts.append(text if text else child.cluster_id)
    return "\n-\n".join(parts)


def _normalize_embeddings(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)


def _descendant_id_sets(clusters: SparseClusterSet) -> dict[str, set[str]]:
    child_ids_by_id = {cluster.cluster_id: tuple(cluster.child_cluster_ids) for cluster in clusters.clusters}
    cache: dict[str, set[str]] = {}

    def visit(cluster_id: str) -> set[str]:
        if cluster_id in cache:
            return cache[cluster_id]
        descendants: set[str] = set()
        for child_id in child_ids_by_id.get(cluster_id, ()):
            descendants.add(child_id)
            descendants.update(visit(child_id))
        cache[cluster_id] = descendants
        return descendants

    return {cluster.cluster_id: visit(cluster.cluster_id) for cluster in clusters.clusters}


def _maximal_cliques(neighbors: dict[int, set[int]], *, min_size: int) -> list[tuple[int, ...]]:
    cliques: list[tuple[int, ...]] = []

    def bron_kerbosch(r: set[int], p: set[int], x: set[int]) -> None:
        if not p and not x:
            if len(r) >= min_size:
                cliques.append(tuple(sorted(r)))
            return
        pivot_candidates = p | x
        pivot = max(pivot_candidates, key=lambda v: len(p & neighbors[v])) if pivot_candidates else None
        excluded = neighbors[pivot] if pivot is not None else set()
        for v in list(p - excluded):
            bron_kerbosch(r | {v}, p & neighbors[v], x & neighbors[v])
            p.remove(v)
            x.add(v)

    bron_kerbosch(set(), set(neighbors), set())
    return cliques


def _unique_semantic_id(clusters: SparseClusterSet, *, added_count: int) -> str:
    existing = set(clusters.cluster_by_id)
    base = f"merge:semantic_similarity:{len(existing) + added_count}"
    if base not in existing:
        return base
    suffix = 1
    while f"{base}:{suffix}" in existing:
        suffix += 1
    return f"{base}:{suffix}"


def _semantic_parent_metadata(
    *,
    child_ids: tuple[str, ...],
    similarities: list[float],
    threshold: float,
    round_idx: int,
    group_strategy: str,
) -> dict[str, Any]:
    return {
        "merged_from": child_ids,
        "merge_strategy": "semantic_similarity",
        "centroid_mode": "child_centroid_sum",
        "semantic_similarity": {
            "threshold": threshold,
            "round": round_idx,
            "group_strategy": group_strategy,
            "min_similarity": float(min(similarities)) if similarities else None,
            "mean_similarity": float(np.mean(similarities)) if similarities else None,
            "max_similarity": float(max(similarities)) if similarities else None,
        },
    }


def _apply_label(
    parent: SparseCluster,
    *,
    children: list[SparseCluster],
    label_fn: Callable[[object], object] | None,
    label_text_fn: Callable[[SparseCluster, list[SparseCluster]], object],
) -> SparseCluster:
    if label_fn is None:
        return parent
    text = label_text_fn(parent, children)
    result = label_fn(text)
    label, description, result_metadata = _parse_label_result(result)
    metadata = dict(parent.metadata)
    metadata["labeling"] = {
        "method": "user_fn",
        "cluster_scope": "semantic_parents",
        "text_type": type(text).__name__,
    }
    if result_metadata:
        metadata["labeling"]["result_metadata"] = result_metadata
    return parent.with_updates(
        label=label if label is not None else parent.label,
        description=description if description is not None else parent.description,
        metadata=metadata,
    )


def merge_clusters_by_semantic_similarity(
    clusters: SparseClusterSet,
    *,
    embed_fn: Callable[[list[str]], np.ndarray],
    threshold: float = 0.9,
    text_fn: Callable[[SparseCluster], str] | None = None,
    label_fn: Callable[[object], object] | None = None,
    label_text_fn: Callable[[SparseCluster, list[SparseCluster]], object] | None = None,
    cluster_scope: ClusterScope = "active",
    max_rounds: int = 10,
    min_group_size: int = 2,
    normalize_embeddings: bool = True,
    normalize_centroids: bool = True,
    verbose: bool = False,
    show_progress: bool = False,
) -> SparseClusterSet:
    """Iteratively create semantic parent clusters from similar labeled clusters.

    Similarity groups are maximal cliques over pairs with cosine similarity >=
    ``threshold``. This preserves overlap: if A-B and B-C are similar but A-C is
    not, two parent nodes are created instead of forcing a single partition.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    if max_rounds < 1:
        raise ValueError("max_rounds must be >= 1")
    if min_group_size < 2:
        raise ValueError("min_group_size must be >= 2")

    out = clusters
    text_fn = text_fn or default_cluster_text
    label_text_fn = label_text_fn or default_semantic_label_text
    group_strategy = "maximal_cliques"
    total_added = 0

    for round_idx in range(1, max_rounds + 1):
        candidates_all = _scoped_clusters(out, cluster_scope)
        candidate_texts: list[str] = []
        candidates: list[SparseCluster] = []
        for cluster in candidates_all:
            text = str(text_fn(cluster) or "").strip()
            if text:
                candidates.append(cluster)
                candidate_texts.append(text)
        if len(candidates) < min_group_size:
            out = out.append_history(
                {
                    "phase": "merge_clusters_by_semantic_similarity",
                    "round": round_idx,
                    "n_nodes_added": 0,
                    "stop_reason": "not_enough_text_clusters",
                }
            )
            break

        embeddings = np.asarray(embed_fn(candidate_texts), dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[0] != len(candidates):
            raise ValueError("embed_fn must return a 2D array with one row per input text")
        if normalize_embeddings:
            embeddings = _normalize_embeddings(embeddings)
        sim = embeddings @ embeddings.T
        descendant_ids = _descendant_id_sets(out)

        neighbors = {i: set() for i in range(len(candidates))}
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                a = candidates[i].cluster_id
                b = candidates[j].cluster_id
                if b in descendant_ids.get(a, set()) or a in descendant_ids.get(b, set()):
                    continue
                if float(sim[i, j]) >= threshold:
                    neighbors[i].add(j)
                    neighbors[j].add(i)

        cliques = _maximal_cliques(neighbors, min_size=min_group_size)
        if not cliques:
            out = out.append_history(
                {
                    "phase": "merge_clusters_by_semantic_similarity",
                    "round": round_idx,
                    "n_nodes_added": 0,
                    "threshold": threshold,
                    "group_strategy": group_strategy,
                    "stop_reason": "no_similar_groups",
                }
            )
            break

        by_id = out.cluster_by_id
        updated_by_id = dict(by_id)
        added: list[SparseCluster] = []
        participated: set[str] = set()
        for clique in _progress_iter(
            cliques,
            enabled=show_progress,
            total=len(cliques),
            desc=f"semantic_similarity round {round_idx}",
        ):
            children = [candidates[i] for i in clique]
            child_ids = tuple(child.cluster_id for child in children)
            pair_sims = [float(sim[i, j]) for pos, i in enumerate(clique) for j in clique[pos + 1 :]]
            parent = merge_cluster_group(
                children,
                cluster_id=_unique_semantic_id(out.with_clusters(tuple(out.clusters) + tuple(added)), added_count=len(added)),
                phase="semantic_similarity",
                normalize_centroid=normalize_centroids,
            ).with_updates(
                metadata=_semantic_parent_metadata(
                    child_ids=child_ids,
                    similarities=pair_sims,
                    threshold=threshold,
                    round_idx=round_idx,
                    group_strategy=group_strategy,
                )
            )
            parent = _apply_label(parent, children=children, label_fn=label_fn, label_text_fn=label_text_fn)
            added.append(parent)
            participated.update(child_ids)
            for child in children:
                current = updated_by_id[child.cluster_id]
                parents = tuple(dict.fromkeys(current.parent_cluster_ids + (parent.cluster_id,)))
                updated_by_id[child.cluster_id] = current.with_updates(parent_cluster_ids=parents)

        all_clusters = [updated_by_id[cluster.cluster_id] for cluster in out.clusters] + added
        previous_active = tuple(out.active_cluster_ids or ())
        next_active = [cluster.cluster_id for cluster in added]
        next_active.extend(cluster_id for cluster_id in previous_active if cluster_id not in participated)
        out = out.with_clusters(
            all_clusters,
            history_entry={
                "phase": "merge_clusters_by_semantic_similarity",
                "round": round_idx,
                "threshold": threshold,
                "group_strategy": group_strategy,
                "n_nodes_added": len(added),
                "n_candidate_clusters": len(candidates),
                "n_similarity_groups": len(cliques),
            },
        ).with_active_cluster_ids(
            tuple(dict.fromkeys(next_active)),
            history_entry={
                "phase": "activate_semantic_similarity_merges",
                "round": round_idx,
                "n_active_before": len(previous_active),
                "n_active_after": len(tuple(dict.fromkeys(next_active))),
            },
        )
        total_added += len(added)
        if verbose:
            print(
                f"[merge_clusters_by_semantic_similarity] round={round_idx}/{max_rounds} "
                f"added={len(added)} active={len(out.active_clusters)}"
            )
        if not added:
            break

    return out.append_history(
        {
            "phase": "merge_clusters_by_semantic_similarity_done",
            "max_rounds": max_rounds,
            "n_nodes_added": total_added,
        }
    )
