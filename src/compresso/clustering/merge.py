from __future__ import annotations

from collections import defaultdict
from typing import Callable, Literal

import numpy as np

from .types import ScoredTag, SparseCluster, SparseClusterSet, SparseVector

ClusterScope = Literal["active", "all", "leaves", "roots"]


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


def _entity_iou_sets(ea: set[int], eb: set[int]) -> float:
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


def _entity_containment_sets(ea: set[int], eb: set[int]) -> float:
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


def _feature_containment_sets(fa, fb) -> float:
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


def merge_cluster_group(
    clusters: list[SparseCluster],
    *,
    cluster_id: str,
    phase: str | None = None,
    normalize_centroid: bool = True,
) -> SparseCluster:
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
        child_cluster_ids=tuple(cluster.cluster_id for cluster in clusters),
        metadata={
            "merged_from": tuple(cluster.cluster_id for cluster in clusters),
            "merge_strategy": phase,
            "centroid_mode": "child_centroid_sum",
        },
    )


def _unique_merge_id(clusters: SparseClusterSet, *, phase: str, group_idx: int) -> str:
    existing = set(clusters.cluster_by_id)
    base = f"merge:{phase}:{len(existing) + group_idx}"
    if base not in existing:
        return base
    suffix = 1
    while f"{base}:{suffix}" in existing:
        suffix += 1
    return f"{base}:{suffix}"


def merge_cluster_set(
    clusters: SparseClusterSet,
    groups: list[list[int]],
    *,
    phase: str,
    normalize_centroids: bool = True,
) -> SparseClusterSet:
    """Create hierarchy-preserving parent nodes for grouped active clusters."""
    active = clusters.active_clusters
    by_id = clusters.cluster_by_id
    updated_by_id = dict(by_id)
    next_active_ids: list[str] = []
    added: list[SparseCluster] = []
    for group_idx, group in enumerate(groups):
        members = [active[i] for i in group]
        if len(members) == 1:
            next_active_ids.append(members[0].cluster_id)
        else:
            parent_id = _unique_merge_id(clusters, phase=phase, group_idx=group_idx + len(added))
            parent = merge_cluster_group(
                members,
                cluster_id=parent_id,
                phase=phase,
                normalize_centroid=normalize_centroids,
            )
            added.append(parent)
            next_active_ids.append(parent.cluster_id)
            for member in members:
                parents = tuple(dict.fromkeys(member.parent_cluster_ids + (parent.cluster_id,)))
                updated_by_id[member.cluster_id] = member.with_updates(parent_cluster_ids=parents)

    all_clusters = [updated_by_id[cluster.cluster_id] for cluster in clusters.clusters] + added
    return clusters.with_clusters(all_clusters).with_active_cluster_ids(
        next_active_ids,
        history_entry={
            "phase": phase,
            "n_active_before": len(active),
            "n_active_after": len(next_active_ids),
            "n_nodes_before": len(clusters.clusters),
            "n_nodes_after": len(all_clusters),
            "n_nodes_added": len(added),
        },
    )


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


def _has_path(clusters: SparseClusterSet, *, ancestor_id: str, descendant_id: str) -> bool:
    return any(c.cluster_id == descendant_id for c in clusters.descendants(ancestor_id))


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


def _link_cluster_pairs(
    clusters: SparseClusterSet,
    links: list[tuple[str, str]],
    *,
    phase: str,
    metadata: dict,
) -> SparseClusterSet:
    if not links:
        return clusters.append_history({"phase": phase, "n_links_added": 0, **metadata})

    by_id = dict(clusters.cluster_by_id)
    n_added = 0
    for parent_id, child_id in links:
        parent = by_id[parent_id]
        child = by_id[child_id]
        if child_id not in parent.child_cluster_ids:
            parent_children = tuple(dict.fromkeys(parent.child_cluster_ids + (child_id,)))
            by_id[parent_id] = parent.with_updates(child_cluster_ids=parent_children)
            n_added += 1
        if parent_id not in child.parent_cluster_ids:
            child_parents = tuple(dict.fromkeys(child.parent_cluster_ids + (parent_id,)))
            by_id[child_id] = child.with_updates(parent_cluster_ids=child_parents)

    all_clusters = [by_id[cluster.cluster_id] for cluster in clusters.clusters]
    return clusters.with_clusters(
        all_clusters,
        history_entry={"phase": phase, "n_links_added": n_added, **metadata},
    )


def materialize_link_merges(
    clusters: SparseClusterSet,
    *,
    parent_scope: ClusterScope = "active",
    include_descendants: bool = False,
    min_children: int = 1,
    normalize_centroids: bool = True,
    activate: bool = True,
    verbose: bool = False,
) -> SparseClusterSet:
    """Turn existing graph links into explicit non-destructive merge nodes.

    Link passes only add parent/child edges between existing clusters. This
    transform materializes those edges as new parent nodes so linked structures
    are easier to inspect while preserving all original clusters.
    """
    if min_children < 1:
        raise ValueError("min_children must be >= 1")

    candidates = _scoped_clusters(clusters, parent_scope)
    by_id = clusters.cluster_by_id
    updated_by_id = dict(by_id)
    added: list[SparseCluster] = []
    materialized_by_parent: dict[str, str] = {}
    existing_materialized_parents = {
        str(cluster.metadata.get("materialized_parent_id"))
        for cluster in clusters.clusters
        if cluster.metadata.get("materialized_parent_id") is not None
    }

    for parent in candidates:
        if parent.cluster_id in existing_materialized_parents:
            continue
        if parent.metadata.get("materialized_from_links"):
            continue
        child_ids = tuple(parent.child_cluster_ids)
        if include_descendants:
            child_ids = tuple(dict.fromkeys(child_ids + tuple(c.cluster_id for c in clusters.descendants(parent.cluster_id))))
        if len(child_ids) < min_children:
            continue

        member_ids = tuple(dict.fromkeys((parent.cluster_id,) + child_ids))
        members = [by_id[cluster_id] for cluster_id in member_ids]
        materialized_id = _unique_merge_id(
            clusters.with_clusters(tuple(clusters.clusters) + tuple(added)),
            phase="materialize_link_merges",
            group_idx=len(added),
        )
        materialized = merge_cluster_group(
            members,
            cluster_id=materialized_id,
            phase="materialize_link_merges",
            normalize_centroid=normalize_centroids,
        ).with_updates(
            metadata={
                "merged_from": member_ids,
                "merge_strategy": "materialize_link_merges",
                "centroid_mode": "child_centroid_sum",
                "materialized_from_links": True,
                "materialized_parent_id": parent.cluster_id,
                "include_descendants": include_descendants,
            }
        )
        added.append(materialized)
        materialized_by_parent[parent.cluster_id] = materialized.cluster_id
        for member_id in member_ids:
            member = updated_by_id[member_id]
            parents = tuple(dict.fromkeys(member.parent_cluster_ids + (materialized.cluster_id,)))
            updated_by_id[member_id] = member.with_updates(parent_cluster_ids=parents)

    if not added:
        return clusters.append_history(
            {
                "phase": "materialize_link_merges",
                "n_nodes_added": 0,
                "parent_scope": parent_scope,
                "include_descendants": include_descendants,
            }
        )

    all_clusters = [updated_by_id[cluster.cluster_id] for cluster in clusters.clusters] + added
    out = clusters.with_clusters(
        all_clusters,
        history_entry={
            "phase": "materialize_link_merges",
            "n_nodes_before": len(clusters.clusters),
            "n_nodes_after": len(all_clusters),
            "n_nodes_added": len(added),
            "parent_scope": parent_scope,
            "include_descendants": include_descendants,
        },
    )
    if activate:
        next_active_ids = [materialized_by_parent.get(cluster_id, cluster_id) for cluster_id in (clusters.active_cluster_ids or ())]
        out = out.with_active_cluster_ids(
            tuple(dict.fromkeys(next_active_ids)),
            history_entry={
                "phase": "activate_materialized_link_merges",
                "n_active_before": len(clusters.active_cluster_ids or ()),
                "n_active_after": len(tuple(dict.fromkeys(next_active_ids))),
            },
        )
    if verbose:
        print(f"[materialize_link_merges] added={len(added)} activate={activate}")
    return out


def link_clusters_by_entity_containment(
    clusters: SparseClusterSet,
    *,
    threshold: float = 1.0,
    child_scope: ClusterScope = "leaves",
    parent_scope: ClusterScope = "all",
    require_parent_larger: bool = True,
    skip_existing_ancestors: bool = True,
    verbose: bool = False,
    show_progress: bool = False,
) -> SparseClusterSet:
    """Add DAG parent links when a child's entities are contained in a parent.

    This is non-destructive: it does not create new nodes and does not change
    the active frontier. A child may receive multiple parents.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    children = _scoped_clusters(clusters, child_scope)
    parents = _scoped_clusters(clusters, parent_scope)
    parent_entity_sets = {p.cluster_id: set(p.entity_indices.tolist()) for p in parents}
    child_entity_sets = {c.cluster_id: set(c.entity_indices.tolist()) for c in children}
    parent_by_id = {p.cluster_id: p for p in parents}
    parent_ids_by_entity: dict[int, list[str]] = defaultdict(list)
    for parent in parents:
        for entity_idx in parent_entity_sets[parent.cluster_id]:
            parent_ids_by_entity[int(entity_idx)].append(parent.cluster_id)
    descendant_ids = _descendant_id_sets(clusters)

    links: list[tuple[str, str]] = []
    if verbose:
        print(
            f"[link_clusters_by_entity_containment] children={len(children)} parents={len(parents)} "
            f"threshold={threshold:.6f} child_scope={child_scope} parent_scope={parent_scope}"
        )
    for child in _progress_iter(
        children,
        enabled=show_progress,
        total=len(children),
        desc="entity_containment_link children",
    ):
        child_set = child_entity_sets[child.cluster_id]
        overlap_counts: dict[str, int] = defaultdict(int)
        for entity_idx in child_set:
            for parent_id in parent_ids_by_entity.get(int(entity_idx), ()):
                overlap_counts[parent_id] += 1
        for parent_id, overlap in overlap_counts.items():
            if parent_id == child.cluster_id:
                continue
            parent = parent_by_id[parent_id]
            parent_set = parent_entity_sets[parent_id]
            if require_parent_larger and len(parent_set) <= len(child_set):
                continue
            if child.cluster_id in parent.child_cluster_ids:
                continue
            if parent_id in descendant_ids.get(child.cluster_id, set()):
                continue
            if skip_existing_ancestors and child.cluster_id in descendant_ids.get(parent_id, set()):
                continue
            denom = min(len(child_set), len(parent_set))
            if denom > 0 and (overlap / float(denom)) >= threshold:
                links.append((parent_id, child.cluster_id))
    if verbose:
        print(f"[link_clusters_by_entity_containment] links_added={len(links)}")
    return _link_cluster_pairs(
        clusters,
        links,
        phase="link_clusters_by_entity_containment",
        metadata={
            "threshold": threshold,
            "child_scope": child_scope,
            "parent_scope": parent_scope,
            "require_parent_larger": require_parent_larger,
        },
    )


def link_clusters_by_feature_containment(
    clusters: SparseClusterSet,
    *,
    threshold: float = 1.0,
    signed: bool = True,
    child_scope: ClusterScope = "leaves",
    parent_scope: ClusterScope = "all",
    require_parent_larger: bool = True,
    skip_existing_ancestors: bool = True,
    verbose: bool = False,
    show_progress: bool = False,
) -> SparseClusterSet:
    """Add DAG parent links when a child's feature support is contained in a parent."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    children = _scoped_clusters(clusters, child_scope)
    parents = _scoped_clusters(clusters, parent_scope)
    parent_feature_sets = {p.cluster_id: _feature_set(p, signed=signed) for p in parents}
    child_feature_sets = {c.cluster_id: _feature_set(c, signed=signed) for c in children}
    parent_by_id = {p.cluster_id: p for p in parents}
    parent_ids_by_feature: dict[object, list[str]] = defaultdict(list)
    for parent in parents:
        for feature in parent_feature_sets[parent.cluster_id]:
            parent_ids_by_feature[feature].append(parent.cluster_id)
    descendant_ids = _descendant_id_sets(clusters)

    links: list[tuple[str, str]] = []
    if verbose:
        print(
            f"[link_clusters_by_feature_containment] children={len(children)} parents={len(parents)} "
            f"threshold={threshold:.6f} signed={signed} child_scope={child_scope} parent_scope={parent_scope}"
        )
    for child in _progress_iter(
        children,
        enabled=show_progress,
        total=len(children),
        desc="feature_containment_link children",
    ):
        child_set = child_feature_sets[child.cluster_id]
        overlap_counts: dict[str, int] = defaultdict(int)
        for feature in child_set:
            for parent_id in parent_ids_by_feature.get(feature, ()):
                overlap_counts[parent_id] += 1
        for parent_id, overlap in overlap_counts.items():
            if parent_id == child.cluster_id:
                continue
            parent = parent_by_id[parent_id]
            parent_set = parent_feature_sets[parent_id]
            if require_parent_larger and len(parent_set) <= len(child_set):
                continue
            if child.cluster_id in parent.child_cluster_ids:
                continue
            if parent_id in descendant_ids.get(child.cluster_id, set()):
                continue
            if skip_existing_ancestors and child.cluster_id in descendant_ids.get(parent_id, set()):
                continue
            denom = min(len(child_set), len(parent_set))
            if denom > 0 and (overlap / float(denom)) >= threshold:
                links.append((parent_id, child.cluster_id))
    if verbose:
        print(f"[link_clusters_by_feature_containment] links_added={len(links)}")
    return _link_cluster_pairs(
        clusters,
        links,
        phase="link_clusters_by_feature_containment",
        metadata={
            "threshold": threshold,
            "signed": signed,
            "child_scope": child_scope,
            "parent_scope": parent_scope,
            "require_parent_larger": require_parent_larger,
        },
    )


def prune_redundant_active_clusters(
    clusters: SparseClusterSet,
    *,
    verbose: bool = False,
) -> SparseClusterSet:
    """Remove active clusters that already sit under another active cluster."""
    active_ids = tuple(clusters.active_cluster_ids or ())
    active_set = set(active_ids)
    descendant_ids = _descendant_id_sets(clusters)
    covered: set[str] = set()
    for candidate_id in active_ids:
        for other_id in active_ids:
            if candidate_id == other_id:
                continue
            if candidate_id in descendant_ids.get(other_id, set()):
                covered.add(candidate_id)
                break
    kept = [cluster_id for cluster_id in active_ids if cluster_id not in covered]
    if verbose:
        print(f"[prune_redundant_active_clusters] active: {len(active_set)} -> {len(kept)}")
    return clusters.with_active_cluster_ids(
        kept,
        history_entry={
            "phase": "prune_redundant_active_clusters",
            "n_active_before": len(active_ids),
            "n_active_after": len(kept),
            "n_pruned": len(covered),
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
        active = out.active_clusters
        entity_sets = [set(cluster.entity_indices.tolist()) for cluster in active]
        if verbose:
            print(
                f"[merge_clusters_by_entity_iou] round={round_idx}/{max_rounds} "
                f"active_clusters={len(active)} threshold={threshold:.6f}"
            )
        groups = _connected_components(
            len(active),
            lambda i, j: _entity_iou_sets(entity_sets[i], entity_sets[j]) >= threshold,
            show_progress=show_progress,
            desc=f"entity_iou round {round_idx}",
        )
        if all(len(g) == 1 for g in groups):
            if verbose:
                print(f"[merge_clusters_by_entity_iou] unchanged: active_clusters={len(active)}")
            return out.append_history({"phase": "merge_clusters_by_entity_iou", "threshold": threshold, "changed": False})
        before = len(active)
        out = merge_cluster_set(out, groups, phase="merge_clusters_by_entity_iou", normalize_centroids=normalize_centroids)
        if verbose:
            print(f"[merge_clusters_by_entity_iou] merged: {before} -> {len(out.active_clusters)}")
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
        active = out.active_clusters
        entity_sets = [set(cluster.entity_indices.tolist()) for cluster in active]
        if verbose:
            print(
                f"[merge_clusters_by_entity_containment] round={round_idx}/{max_rounds} "
                f"active_clusters={len(active)} threshold={threshold:.6f}"
            )
        groups = _connected_components(
            len(active),
            lambda i, j: _entity_containment_sets(entity_sets[i], entity_sets[j]) >= threshold,
            show_progress=show_progress,
            desc=f"entity_containment round {round_idx}",
        )
        if all(len(g) == 1 for g in groups):
            if verbose:
                print(f"[merge_clusters_by_entity_containment] unchanged: active_clusters={len(active)}")
            return out.append_history(
                {"phase": "merge_clusters_by_entity_containment", "threshold": threshold, "changed": False}
            )
        before = len(active)
        out = merge_cluster_set(
            out,
            groups,
            phase="merge_clusters_by_entity_containment",
            normalize_centroids=normalize_centroids,
        )
        if verbose:
            print(f"[merge_clusters_by_entity_containment] merged: {before} -> {len(out.active_clusters)}")
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
        active = out.active_clusters
        feature_sets = [_feature_set(cluster, signed=signed) for cluster in active]
        if verbose:
            print(
                f"[merge_clusters_by_feature_containment] round={round_idx}/{max_rounds} "
                f"active_clusters={len(active)} threshold={threshold:.6f} signed={signed}"
            )
        groups = _connected_components(
            len(active),
            lambda i, j: _feature_containment_sets(feature_sets[i], feature_sets[j]) >= threshold,
            show_progress=show_progress,
            desc=f"feature_containment round {round_idx}",
        )
        if all(len(g) == 1 for g in groups):
            if verbose:
                print(f"[merge_clusters_by_feature_containment] unchanged: active_clusters={len(active)}")
            return out.append_history(
                {
                    "phase": "merge_clusters_by_feature_containment",
                    "threshold": threshold,
                    "signed": signed,
                    "changed": False,
                }
            )
        before = len(active)
        out = merge_cluster_set(
            out,
            groups,
            phase="merge_clusters_by_feature_containment",
            normalize_centroids=normalize_centroids,
        )
        if verbose:
            print(f"[merge_clusters_by_feature_containment] merged: {before} -> {len(out.active_clusters)}")
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
                f"active_clusters={len(out.active_clusters)} threshold={threshold:.6f} metric={metric}"
            )
        active = out.active_clusters
        tag_vectors = [_tag_vector(cluster) for cluster in active]
        groups = _connected_components(
            len(active),
            lambda i, j: sim_fn(tag_vectors[i], tag_vectors[j]) >= threshold,
            show_progress=show_progress,
            desc=f"tag_similarity round {round_idx}",
        )
        if all(len(g) == 1 for g in groups):
            if verbose:
                print(f"[merge_clusters_by_tag_similarity] unchanged: active_clusters={len(out.active_clusters)}")
            return out.append_history({"phase": "merge_clusters_by_tag_similarity", "threshold": threshold, "metric": metric, "changed": False})
        before = len(out.active_clusters)
        out = merge_cluster_set(out, groups, phase="merge_clusters_by_tag_similarity", normalize_centroids=normalize_centroids)
        if verbose:
            print(f"[merge_clusters_by_tag_similarity] merged: {before} -> {len(out.active_clusters)}")
    return out


def filter_clusters_by_size(
    clusters: SparseClusterSet,
    *,
    min_cluster_size: int,
) -> SparseClusterSet:
    if min_cluster_size < 1:
        raise ValueError("min_cluster_size must be >= 1")
    kept = [cluster.cluster_id for cluster in clusters.active_clusters if cluster.entity_count >= min_cluster_size]
    return clusters.with_active_cluster_ids(
        kept,
        history_entry={
            "phase": "filter_clusters_by_size",
            "min_cluster_size": min_cluster_size,
            "n_active_before": len(clusters.active_clusters),
            "n_active_after": len(kept),
        },
    )
