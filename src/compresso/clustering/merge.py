from __future__ import annotations

from collections import defaultdict
from typing import Callable, Literal

import numpy as np

from .types import ScoredTag, SparseCluster, SparseClusterSet, SparseVector


def _progress_iter(iterable, *, enabled: bool, total: int | None = None, desc: str = ""):
    if not enabled:
        return iterable
    try:
        from tqdm.auto import tqdm
    except Exception:  # pragma: no cover - optional progress dependency
        return iterable
    return tqdm(iterable, total=total, desc=desc)


def _connected_components(
    n: int,
    should_link: Callable[[int, int], bool],
    *,
    show_progress: bool = False,
    desc: str = "connected_components",
) -> list[list[int]]:
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in _progress_iter(range(n), enabled=show_progress, total=n, desc=desc):
        for j in range(i + 1, n):
            if should_link(i, j):
                union(i, j)

    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    return list(groups.values())


def _entity_iou(a: SparseCluster, b: SparseCluster) -> float:
    ea = set(a.entity_indices.tolist())
    eb = set(b.entity_indices.tolist())
    if not ea and not eb:
        return 0.0
    return len(ea & eb) / float(len(ea | eb))


def _entity_containment(a: SparseCluster, b: SparseCluster) -> float:
    ea = set(a.entity_indices.tolist())
    eb = set(b.entity_indices.tolist())
    denom = min(len(ea), len(eb))
    if denom == 0:
        return 0.0
    return len(ea & eb) / float(denom)


def _feature_set(cluster: SparseCluster, *, signed: bool = True) -> set[int] | set[tuple[int, int]]:
    indices = cluster.centroid.indices.tolist()
    if not signed:
        return {int(idx) for idx in indices}
    values = cluster.centroid.values.tolist()
    return {(int(idx), 1 if float(value) >= 0.0 else -1) for idx, value in zip(indices, values)}


def _feature_containment(a: SparseCluster, b: SparseCluster, *, signed: bool = True) -> float:
    fa = _feature_set(a, signed=signed)
    fb = _feature_set(b, signed=signed)
    denom = min(len(fa), len(fb))
    if denom == 0:
        return 0.0
    return len(fa & fb) / float(denom)


def _tag_vector(cluster: SparseCluster) -> dict[int, float]:
    return {tag.tag_id: tag.score for tag in cluster.tags}


def _weighted_jaccard(a: dict[int, float], b: dict[int, float]) -> float:
    keys = set(a) | set(b)
    if not keys:
        return 0.0
    num = sum(min(float(a.get(k, 0.0)), float(b.get(k, 0.0))) for k in keys)
    den = sum(max(float(a.get(k, 0.0)), float(b.get(k, 0.0))) for k in keys)
    return num / den if den > 0 else 0.0


def _tag_cosine(a: dict[int, float], b: dict[int, float]) -> float:
    keys = set(a) | set(b)
    if not keys:
        return 0.0
    dot = sum(float(a.get(k, 0.0)) * float(b.get(k, 0.0)) for k in keys)
    na = sum(float(v) ** 2 for v in a.values()) ** 0.5
    nb = sum(float(v) ** 2 for v in b.values()) ** 0.5
    return dot / (na * nb) if na > 0 and nb > 0 else 0.0


def merge_cluster_group(clusters: list[SparseCluster], *, cluster_id: str, normalize_centroid: bool = True) -> SparseCluster:
    if not clusters:
        raise ValueError("clusters must not be empty")
    source_ids: list[str] = []
    for cluster in clusters:
        source_ids.extend(cluster.source_cluster_ids or (cluster.cluster_id,))
    entities = np.unique(np.concatenate([cluster.entity_indices for cluster in clusters])).astype(np.int64)
    centroid = SparseVector.sum([cluster.centroid for cluster in clusters], normalize=normalize_centroid)
    return SparseCluster(
        cluster_id=cluster_id,
        centroid=centroid,
        entity_indices=entities,
        source_cluster_ids=tuple(dict.fromkeys(source_ids)),
        metadata={"merged_from": tuple(cluster.cluster_id for cluster in clusters)},
    )


def merge_cluster_set(
    clusters: SparseClusterSet,
    groups: list[list[int]],
    *,
    phase: str,
    normalize_centroids: bool = True,
) -> SparseClusterSet:
    merged: list[SparseCluster] = []
    for group_idx, group in enumerate(groups):
        members = [clusters.clusters[i] for i in group]
        if len(members) == 1:
            merged.append(members[0])
        else:
            merged.append(merge_cluster_group(members, cluster_id=f"merge:{phase}:{group_idx}", normalize_centroid=normalize_centroids))
    return clusters.with_clusters(
        merged,
        history_entry={
            "phase": phase,
            "n_clusters_before": len(clusters.clusters),
            "n_clusters_after": len(merged),
        },
    )


def merge_clusters_by_entity_iou(
    clusters: SparseClusterSet,
    *,
    threshold: float,
    max_rounds: int = 10,
    normalize_centroids: bool = True,
    verbose: bool = False,
    show_progress: bool = False,
) -> SparseClusterSet:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    out = clusters
    for round_idx in range(1, max_rounds + 1):
        if verbose:
            print(
                f"[merge_clusters_by_entity_iou] round={round_idx}/{max_rounds} "
                f"clusters={len(out.clusters)} threshold={threshold:.6f}"
            )
        groups = _connected_components(
            len(out.clusters),
            lambda i, j: _entity_iou(out.clusters[i], out.clusters[j]) >= threshold,
            show_progress=show_progress,
            desc=f"entity_iou round {round_idx}",
        )
        if all(len(g) == 1 for g in groups):
            if verbose:
                print(f"[merge_clusters_by_entity_iou] unchanged: clusters={len(out.clusters)}")
            return out.append_history({"phase": "merge_clusters_by_entity_iou", "threshold": threshold, "changed": False})
        before = len(out.clusters)
        out = merge_cluster_set(out, groups, phase="merge_clusters_by_entity_iou", normalize_centroids=normalize_centroids)
        if verbose:
            print(f"[merge_clusters_by_entity_iou] merged: {before} -> {len(out.clusters)}")
    return out


def merge_clusters_by_entity_containment(
    clusters: SparseClusterSet,
    *,
    threshold: float = 1.0,
    max_rounds: int = 10,
    normalize_centroids: bool = True,
    verbose: bool = False,
    show_progress: bool = False,
) -> SparseClusterSet:
    """Merge clusters when one cluster's entity set is mostly contained in another.

    Containment is symmetric and defined as ``|A intersect B| / min(|A|, |B|)``.
    A strict subset therefore has score 1.0 even when IoU is small.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    out = clusters
    for round_idx in range(1, max_rounds + 1):
        if verbose:
            print(
                f"[merge_clusters_by_entity_containment] round={round_idx}/{max_rounds} "
                f"clusters={len(out.clusters)} threshold={threshold:.6f}"
            )
        groups = _connected_components(
            len(out.clusters),
            lambda i, j: _entity_containment(out.clusters[i], out.clusters[j]) >= threshold,
            show_progress=show_progress,
            desc=f"entity_containment round {round_idx}",
        )
        if all(len(g) == 1 for g in groups):
            if verbose:
                print(f"[merge_clusters_by_entity_containment] unchanged: clusters={len(out.clusters)}")
            return out.append_history(
                {"phase": "merge_clusters_by_entity_containment", "threshold": threshold, "changed": False}
            )
        before = len(out.clusters)
        out = merge_cluster_set(
            out,
            groups,
            phase="merge_clusters_by_entity_containment",
            normalize_centroids=normalize_centroids,
        )
        if verbose:
            print(f"[merge_clusters_by_entity_containment] merged: {before} -> {len(out.clusters)}")
    return out


def merge_clusters_by_feature_containment(
    clusters: SparseClusterSet,
    *,
    threshold: float = 1.0,
    signed: bool = True,
    max_rounds: int = 10,
    normalize_centroids: bool = True,
    verbose: bool = False,
    show_progress: bool = False,
) -> SparseClusterSet:
    """Merge clusters when one centroid's feature support is mostly contained in another.

    Containment is ``|F_A intersect F_B| / min(|F_A|, |F_B|)``. With
    ``signed=True`` the support elements are ``(feature_id, sign)`` pairs.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    out = clusters
    for round_idx in range(1, max_rounds + 1):
        if verbose:
            print(
                f"[merge_clusters_by_feature_containment] round={round_idx}/{max_rounds} "
                f"clusters={len(out.clusters)} threshold={threshold:.6f} signed={signed}"
            )
        groups = _connected_components(
            len(out.clusters),
            lambda i, j: _feature_containment(out.clusters[i], out.clusters[j], signed=signed) >= threshold,
            show_progress=show_progress,
            desc=f"feature_containment round {round_idx}",
        )
        if all(len(g) == 1 for g in groups):
            if verbose:
                print(f"[merge_clusters_by_feature_containment] unchanged: clusters={len(out.clusters)}")
            return out.append_history(
                {
                    "phase": "merge_clusters_by_feature_containment",
                    "threshold": threshold,
                    "signed": signed,
                    "changed": False,
                }
            )
        before = len(out.clusters)
        out = merge_cluster_set(
            out,
            groups,
            phase="merge_clusters_by_feature_containment",
            normalize_centroids=normalize_centroids,
        )
        if verbose:
            print(f"[merge_clusters_by_feature_containment] merged: {before} -> {len(out.clusters)}")
    return out


def merge_clusters_by_tag_similarity(
    clusters: SparseClusterSet,
    *,
    threshold: float,
    metric: Literal["weighted_jaccard", "cosine"] = "weighted_jaccard",
    max_rounds: int = 10,
    normalize_centroids: bool = True,
    verbose: bool = False,
    show_progress: bool = False,
) -> SparseClusterSet:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    if metric not in {"weighted_jaccard", "cosine"}:
        raise ValueError("metric must be 'weighted_jaccard' or 'cosine'")
    sim_fn = _weighted_jaccard if metric == "weighted_jaccard" else _tag_cosine
    out = clusters
    for round_idx in range(1, max_rounds + 1):
        if verbose:
            print(
                f"[merge_clusters_by_tag_similarity] round={round_idx}/{max_rounds} "
                f"clusters={len(out.clusters)} threshold={threshold:.6f} metric={metric}"
            )
        tag_vectors = [_tag_vector(cluster) for cluster in out.clusters]
        groups = _connected_components(
            len(out.clusters),
            lambda i, j: sim_fn(tag_vectors[i], tag_vectors[j]) >= threshold,
            show_progress=show_progress,
            desc=f"tag_similarity round {round_idx}",
        )
        if all(len(g) == 1 for g in groups):
            if verbose:
                print(f"[merge_clusters_by_tag_similarity] unchanged: clusters={len(out.clusters)}")
            return out.append_history({"phase": "merge_clusters_by_tag_similarity", "threshold": threshold, "metric": metric, "changed": False})
        before = len(out.clusters)
        out = merge_cluster_set(out, groups, phase="merge_clusters_by_tag_similarity", normalize_centroids=normalize_centroids)
        if verbose:
            print(f"[merge_clusters_by_tag_similarity] merged: {before} -> {len(out.clusters)}")
    return out


def filter_clusters_by_size(
    clusters: SparseClusterSet,
    *,
    min_cluster_size: int,
) -> SparseClusterSet:
    if min_cluster_size < 1:
        raise ValueError("min_cluster_size must be >= 1")
    kept = [cluster for cluster in clusters.clusters if cluster.entity_count >= min_cluster_size]
    return clusters.with_clusters(
        kept,
        history_entry={
            "phase": "filter_clusters_by_size",
            "min_cluster_size": min_cluster_size,
            "n_clusters_before": len(clusters.clusters),
            "n_clusters_after": len(kept),
        },
    )
