from __future__ import annotations

from collections import defaultdict
from typing import Callable, Literal

import numpy as np
import torch

from compresso.params.srp import SRPTensor
from .types import ScoredTag, SparseCluster, SparseClusterSet, SparseVector

ClusterScope = Literal["active", "all", "leaves", "roots"]
CoverageScope = Literal["active", "all"]


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


def _centroid_similarity_matrix(
    clusters: tuple[SparseCluster, ...],
    *,
    metric: Literal["cosine", "dot"] = "cosine",
) -> np.ndarray:
    n = len(clusters)
    sims = np.zeros((n, n), dtype=np.float32)
    maps: list[dict[int, float]] = []
    norms: list[float] = []
    for cluster in clusters:
        cmap = {int(i): float(v) for i, v in zip(cluster.centroid.indices.tolist(), cluster.centroid.values.tolist())}
        maps.append(cmap)
        norms.append(sum(v * v for v in cmap.values()) ** 0.5)

    for i in range(n):
        sims[i, i] = -np.inf
        mi = maps[i]
        ni = norms[i]
        for j in range(i + 1, n):
            mj = maps[j]
            nj = norms[j]
            if len(mi) <= len(mj):
                dot = sum(v * mj.get(k, 0.0) for k, v in mi.items())
            else:
                dot = sum(v * mi.get(k, 0.0) for k, v in mj.items())
            if metric == "cosine":
                sim = dot / (ni * nj) if ni > 0.0 and nj > 0.0 else 0.0
            elif metric == "dot":
                sim = dot
            else:
                raise ValueError("metric must be 'cosine' or 'dot'")
            sims[i, j] = sim
            sims[j, i] = sim
    return sims


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


def _covered_entity_indices(clusters: SparseClusterSet, scope: CoverageScope) -> set[int]:
    if scope == "active":
        candidates = clusters.active_clusters
    elif scope == "all":
        candidates = clusters.clusters
    else:
        raise ValueError("coverage_scope must be one of 'active' or 'all'")
    covered: set[int] = set()
    for cluster in candidates:
        covered.update(int(idx) for idx in cluster.entity_indices.tolist())
    return covered


def _sparse_vector_map(vector: SparseVector) -> dict[int, float]:
    return {int(i): float(v) for i, v in zip(vector.indices.tolist(), vector.values.tolist())}


def _sparse_dot(a: dict[int, float], b: dict[int, float]) -> float:
    if len(a) <= len(b):
        return sum(v * b.get(k, 0.0) for k, v in a.items())
    return sum(v * a.get(k, 0.0) for k, v in b.items())


def _sparse_norm(a: dict[int, float]) -> float:
    return sum(v * v for v in a.values()) ** 0.5


def _srp_row_map(srp: SRPTensor, row_idx: int) -> dict[int, float]:
    cols = srp.cols[int(row_idx)].detach().cpu().tolist()
    vals = srp.vals[int(row_idx)].detach().cpu().tolist()
    out: dict[int, float] = {}
    for col, val in zip(cols, vals):
        out[int(col)] = out.get(int(col), 0.0) + float(val)
    return out


def _centroid_from_srp_rows(
    srp: SRPTensor,
    core_rows: np.ndarray,
    assigned_rows: np.ndarray,
    *,
    assigned_weight: float = 1.0,
    centroid_top_m: int | None = None,
    normalize: bool = True,
) -> SparseVector:
    if assigned_weight < 0.0:
        raise ValueError("assigned_weight must be >= 0")
    if centroid_top_m is not None and centroid_top_m < 1:
        raise ValueError("centroid_top_m must be >= 1 when provided")
    rows = torch.cat(
        [
            torch.as_tensor(core_rows, dtype=torch.long, device=srp.vals.device),
            torch.as_tensor(assigned_rows, dtype=torch.long, device=srp.vals.device),
        ],
        dim=0,
    )
    if rows.numel() == 0:
        return SparseVector(np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float32), srp.cols_total)

    weights = torch.ones(rows.numel(), dtype=srp.vals.dtype, device=srp.vals.device)
    if len(assigned_rows) > 0:
        weights[-len(assigned_rows) :] = float(assigned_weight)
    selected_cols = srp.cols.index_select(0, rows)
    selected_vals = srp.vals.index_select(0, rows) * weights[:, None]
    dense = torch.zeros(srp.cols_total, dtype=srp.vals.dtype, device=srp.vals.device)
    dense.scatter_add_(0, selected_cols.reshape(-1), selected_vals.reshape(-1))
    denom = weights.sum()
    if float(denom.detach().cpu().item()) > 0.0:
        dense = dense / denom

    if centroid_top_m is not None:
        k = min(int(centroid_top_m), int(srp.cols_total))
        idx = torch.topk(dense.abs(), k=k, largest=True, sorted=True).indices
        vals = dense.gather(0, idx)
        keep = vals != 0
        idx = idx[keep]
        vals = vals[keep]
    else:
        idx = torch.nonzero(dense != 0, as_tuple=False).flatten()
        vals = dense[idx]
    vector = SparseVector(
        idx.detach().cpu().numpy().astype(np.int64),
        vals.detach().cpu().numpy().astype(np.float32),
        srp.cols_total,
    )
    return vector.normalized() if normalize else vector


def _normalize_label(label: str, *, case_sensitive: bool = False) -> str:
    out = " ".join(str(label).strip().split())
    return out if case_sensitive else out.lower()


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


def merge_clusters_by_duplicate_label(
    clusters: SparseClusterSet,
    *,
    cluster_scope: ClusterScope = "active",
    case_sensitive: bool = False,
    mark_children_hidden: bool = True,
    min_group_size: int = 2,
    normalize_centroids: bool = True,
    verbose: bool = False,
) -> SparseClusterSet:
    """Merge clusters with identical normalized labels into canonical parents."""
    if min_group_size < 2:
        raise ValueError("min_group_size must be >= 2")
    candidates = _scoped_clusters(clusters, cluster_scope)
    label_groups: dict[str, list[SparseCluster]] = defaultdict(list)
    label_by_key: dict[str, str] = {}
    for cluster in candidates:
        if not cluster.label:
            continue
        key = _normalize_label(cluster.label, case_sensitive=case_sensitive)
        if not key:
            continue
        label_groups[key].append(cluster)
        label_by_key.setdefault(key, " ".join(str(cluster.label).strip().split()))

    groups = [members for key, members in sorted(label_groups.items()) if len(members) >= min_group_size]
    if not groups:
        return clusters.append_history(
            {
                "phase": "merge_clusters_by_duplicate_label",
                "cluster_scope": cluster_scope,
                "case_sensitive": case_sensitive,
                "min_group_size": min_group_size,
                "n_groups": 0,
                "changed": False,
            }
        )

    updated_by_id = dict(clusters.cluster_by_id)
    added: list[SparseCluster] = []
    member_to_parent: dict[str, str] = {}
    for group_idx, members in enumerate(groups):
        key = _normalize_label(members[0].label or "", case_sensitive=case_sensitive)
        parent_id = _unique_merge_id(
            clusters.with_clusters(tuple(clusters.clusters) + tuple(added)),
            phase="duplicate_label",
            group_idx=len(added) + group_idx,
        )
        parent = merge_cluster_group(
            members,
            cluster_id=parent_id,
            phase="duplicate_label",
            normalize_centroid=normalize_centroids,
        )
        descriptions = [member.description for member in members if member.description]
        parent = parent.with_updates(
            label=label_by_key[key],
            description=descriptions[0] if descriptions else None,
            metadata={
                **dict(parent.metadata),
                "duplicate_label": label_by_key[key],
                "duplicate_label_key": key,
                "mark_children_hidden": mark_children_hidden,
            },
        )
        added.append(parent)
        for member in members:
            member_to_parent[member.cluster_id] = parent.cluster_id
            metadata = dict(member.metadata)
            if mark_children_hidden:
                metadata["render_hidden"] = True
                metadata["render_hidden_reason"] = "duplicate_label"
                metadata["render_hidden_parent_id"] = parent.cluster_id
            parents = tuple(dict.fromkeys(member.parent_cluster_ids + (parent.cluster_id,)))
            updated_by_id[member.cluster_id] = member.with_updates(parent_cluster_ids=parents, metadata=metadata)

    next_active_ids: list[str] = []
    for cluster_id in clusters.active_cluster_ids or ():
        next_active_ids.append(member_to_parent.get(cluster_id, cluster_id))
    next_active_ids = list(dict.fromkeys(next_active_ids))
    all_clusters = [updated_by_id[cluster.cluster_id] for cluster in clusters.clusters] + added
    out = clusters.with_clusters(
        all_clusters,
        history_entry={
            "phase": "merge_clusters_by_duplicate_label",
            "cluster_scope": cluster_scope,
            "case_sensitive": case_sensitive,
            "min_group_size": min_group_size,
            "n_groups": len(groups),
            "n_nodes_added": len(added),
            "mark_children_hidden": mark_children_hidden,
        },
    ).with_active_cluster_ids(
        next_active_ids,
        history_entry={
            "phase": "activate_duplicate_label_merges",
            "n_active_before": len(clusters.active_cluster_ids or ()),
            "n_active_after": len(next_active_ids),
        },
    )
    if verbose:
        print(f"[merge_clusters_by_duplicate_label] groups={len(groups)} active={len(clusters.active_clusters)} -> {len(out.active_clusters)}")
    return out


def compact_hidden_clusters(
    clusters: SparseClusterSet,
    *,
    hidden_key: str = "render_hidden",
    verbose: bool = False,
) -> SparseClusterSet:
    """Remove explicitly hidden nodes and rewire visible parents/children."""
    by_id = clusters.cluster_by_id
    hidden_ids = {cluster.cluster_id for cluster in clusters.clusters if bool(cluster.metadata.get(hidden_key))}
    if not hidden_ids:
        return clusters.append_history({"phase": "compact_hidden_clusters", "hidden_key": hidden_key, "n_removed": 0})

    visible_ids = set(by_id) - hidden_ids

    def visible_ancestors(cluster_id: str, seen: set[str] | None = None) -> list[str]:
        seen = set() if seen is None else seen
        if cluster_id in seen:
            return []
        seen.add(cluster_id)
        out: list[str] = []
        for parent_id in by_id[cluster_id].parent_cluster_ids:
            if parent_id not in by_id:
                continue
            if parent_id in visible_ids:
                out.append(parent_id)
            else:
                out.extend(visible_ancestors(parent_id, seen))
        return out

    def visible_descendants(cluster_id: str, seen: set[str] | None = None) -> list[str]:
        seen = set() if seen is None else seen
        if cluster_id in seen:
            return []
        seen.add(cluster_id)
        out: list[str] = []
        for child_id in by_id[cluster_id].child_cluster_ids:
            if child_id not in by_id:
                continue
            if child_id in visible_ids:
                out.append(child_id)
            else:
                out.extend(visible_descendants(child_id, seen))
        return out

    updated: list[SparseCluster] = []
    for cluster in clusters.clusters:
        if cluster.cluster_id in hidden_ids:
            continue
        parent_ids: list[str] = []
        for parent_id in cluster.parent_cluster_ids:
            if parent_id not in by_id:
                continue
            if parent_id in visible_ids:
                parent_ids.append(parent_id)
            else:
                parent_ids.extend(visible_ancestors(parent_id))
        child_ids: list[str] = []
        for child_id in cluster.child_cluster_ids:
            if child_id not in by_id:
                continue
            if child_id in visible_ids:
                child_ids.append(child_id)
            else:
                child_ids.extend(visible_descendants(child_id))
        parent_ids = [cluster_id for cluster_id in dict.fromkeys(parent_ids) if cluster_id != cluster.cluster_id]
        child_ids = [cluster_id for cluster_id in dict.fromkeys(child_ids) if cluster_id != cluster.cluster_id]
        updated.append(cluster.with_updates(parent_cluster_ids=tuple(parent_ids), child_cluster_ids=tuple(child_ids)))

    next_active_ids: list[str] = []
    for cluster_id in clusters.active_cluster_ids or ():
        if cluster_id in visible_ids:
            next_active_ids.append(cluster_id)
        elif cluster_id in by_id:
            replacements = visible_descendants(cluster_id) or visible_ancestors(cluster_id)
            next_active_ids.extend(replacements)
    next_active_ids = [cluster_id for cluster_id in dict.fromkeys(next_active_ids) if cluster_id in visible_ids]
    out = clusters.with_clusters(
        updated,
        history_entry={
            "phase": "compact_hidden_clusters",
            "hidden_key": hidden_key,
            "n_nodes_before": len(clusters.clusters),
            "n_nodes_after": len(updated),
            "n_removed": len(hidden_ids),
        },
    ).with_active_cluster_ids(
        next_active_ids,
        history_entry={
            "phase": "activate_compacted_hidden_clusters",
            "n_active_before": len(clusters.active_cluster_ids or ()),
            "n_active_after": len(next_active_ids),
        },
    )
    if verbose:
        print(f"[compact_hidden_clusters] nodes={len(clusters.clusters)} -> {len(updated)} removed={len(hidden_ids)}")
    return out


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


def merge_clusters_by_centroid_similarity(
    clusters: SparseClusterSet,
    *,
    threshold: float,
    metric: Literal["cosine", "dot"] = "cosine",
    top_k: int | None = None,
    max_rounds: int = 10,
    min_group_size: int = 2,
    normalize_centroids: bool = True,
    verbose: bool = False,
    show_progress: bool = False,
) -> SparseClusterSet:
    """Merge active clusters whose sparse centroids are similar.

    This operates on the current active frontier, creates non-destructive
    parent merge nodes, and preserves merged clusters as children.

    With ``metric="cosine"``, ``threshold`` must be in ``[-1, 1]``. With
    ``metric="dot"``, raw centroid dot products are compared to ``threshold``.
    If ``top_k`` is provided, a pair can link only when either cluster is among
    the other's top-k most similar centroid neighbors.
    """
    if metric not in {"cosine", "dot"}:
        raise ValueError("metric must be 'cosine' or 'dot'")
    if metric == "cosine" and not -1.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [-1, 1] for cosine similarity")
    if top_k is not None and int(top_k) < 1:
        raise ValueError("top_k must be >= 1 when provided")
    if min_group_size < 2:
        raise ValueError("min_group_size must be >= 2")

    out = clusters
    for round_idx in range(1, max_rounds + 1):
        active = out.active_clusters
        if verbose:
            print(
                f"[merge_clusters_by_centroid_similarity] round={round_idx}/{max_rounds} "
                f"active_clusters={len(active)} threshold={threshold:.6f} metric={metric} top_k={top_k}"
            )
        sims = _centroid_similarity_matrix(active, metric=metric)
        allowed: np.ndarray | None = None
        if top_k is not None and len(active) > 1:
            k = min(int(top_k), len(active) - 1)
            allowed = np.zeros_like(sims, dtype=bool)
            for i in range(len(active)):
                idx = np.argpartition(-sims[i], k - 1)[:k]
                allowed[i, idx] = True

        def should_link(i: int, j: int) -> bool:
            if sims[i, j] < threshold:
                return False
            if allowed is None:
                return True
            return bool(allowed[i, j] or allowed[j, i])

        groups = _connected_components(
            len(active),
            should_link,
            show_progress=show_progress,
            desc=f"centroid_similarity round {round_idx}",
        )
        expanded_groups: list[list[int]] = []
        for group in groups:
            if len(group) >= min_group_size:
                expanded_groups.append(group)
            else:
                expanded_groups.extend([[idx] for idx in group])

        if all(len(g) == 1 for g in expanded_groups):
            if verbose:
                print(f"[merge_clusters_by_centroid_similarity] unchanged: active_clusters={len(active)}")
            return out.append_history(
                {
                    "phase": "merge_clusters_by_centroid_similarity",
                    "threshold": threshold,
                    "metric": metric,
                    "top_k": top_k,
                    "min_group_size": min_group_size,
                    "changed": False,
                }
            )
        before = len(active)
        out = merge_cluster_set(
            out,
            expanded_groups,
            phase="merge_clusters_by_centroid_similarity",
            normalize_centroids=normalize_centroids,
        )
        if verbose:
            print(f"[merge_clusters_by_centroid_similarity] merged: {before} -> {len(out.active_clusters)}")
    return out


def assign_unclustered_to_nearest_cluster(
    clusters: SparseClusterSet,
    srp: SRPTensor,
    *,
    metric: Literal["cosine", "dot"] = "cosine",
    min_similarity: float | None = None,
    top_k_clusters: int = 1,
    cluster_scope: ClusterScope = "active",
    coverage_scope: CoverageScope = "active",
    assigned_weight: float = 1.0,
    centroid_top_m: int | None = None,
    centroid_top_k: int | None = None,
    normalize_centroids: bool = True,
    verbose: bool = False,
) -> SparseClusterSet:
    """Assign uncovered entities to nearest cluster centroids.

    This is a coverage-expansion step, not a discovery step. It creates one
    non-destructive expanded parent for each cluster that receives uncovered
    entities. Original clusters are preserved as children and newly assigned
    entity ids are stored in parent metadata.
    """
    if metric not in {"cosine", "dot"}:
        raise ValueError("metric must be 'cosine' or 'dot'")
    if top_k_clusters < 1:
        raise ValueError("top_k_clusters must be >= 1")
    if assigned_weight < 0.0:
        raise ValueError("assigned_weight must be >= 0")
    if centroid_top_m is not None and centroid_top_k is not None and int(centroid_top_m) != int(centroid_top_k):
        raise ValueError("centroid_top_m and deprecated centroid_top_k must match when both are provided")
    if centroid_top_m is None:
        centroid_top_m = centroid_top_k
    if centroid_top_m is not None and centroid_top_m < 1:
        raise ValueError("centroid_top_m must be >= 1 when provided")
    if srp.rows != clusters.n_entities:
        raise ValueError(f"srp.rows={srp.rows} must match clusters.n_entities={clusters.n_entities}")
    if srp.cols_total != clusters.n_features:
        raise ValueError(f"srp.cols_total={srp.cols_total} must match clusters.n_features={clusters.n_features}")

    candidates = _scoped_clusters(clusters, cluster_scope)
    if not candidates:
        return clusters.append_history(
            {
                "phase": "assign_unclustered_to_nearest_cluster",
                "changed": False,
                "reason": "no_candidate_clusters",
                "cluster_scope": cluster_scope,
                "coverage_scope": coverage_scope,
            }
        )

    covered = _covered_entity_indices(clusters, coverage_scope)
    unclustered = [idx for idx in range(clusters.n_entities) if idx not in covered]
    if not unclustered:
        return clusters.append_history(
            {
                "phase": "assign_unclustered_to_nearest_cluster",
                "changed": False,
                "cluster_scope": cluster_scope,
                "coverage_scope": coverage_scope,
                "metric": metric,
                "min_similarity": min_similarity,
                "top_k_clusters": top_k_clusters,
                "n_unclustered_before": 0,
                "n_assigned": 0,
                "n_unassigned_after": 0,
                "n_expanded_clusters": 0,
            }
        )

    candidate_maps = [_sparse_vector_map(cluster.centroid) for cluster in candidates]
    candidate_norms = [_sparse_norm(cmap) for cmap in candidate_maps]
    assignments: dict[str, list[tuple[int, float]]] = defaultdict(list)

    k = min(int(top_k_clusters), len(candidates))
    for entity_idx in unclustered:
        row_map = _srp_row_map(srp, entity_idx)
        row_norm = _sparse_norm(row_map)
        scores: list[tuple[int, float]] = []
        for candidate_idx, (candidate_map, candidate_norm) in enumerate(zip(candidate_maps, candidate_norms)):
            dot = _sparse_dot(row_map, candidate_map)
            if metric == "cosine":
                score = dot / (row_norm * candidate_norm) if row_norm > 0.0 and candidate_norm > 0.0 else 0.0
            else:
                score = dot
            scores.append((candidate_idx, float(score)))
        scores.sort(key=lambda item: item[1], reverse=True)
        for candidate_idx, score in scores[:k]:
            if min_similarity is not None and score < float(min_similarity):
                continue
            assignments[candidates[candidate_idx].cluster_id].append((entity_idx, score))

    assigned_entities = {entity_idx for values in assignments.values() for entity_idx, _ in values}
    if not assignments:
        return clusters.append_history(
            {
                "phase": "assign_unclustered_to_nearest_cluster",
                "changed": False,
                "cluster_scope": cluster_scope,
                "coverage_scope": coverage_scope,
                "metric": metric,
                "min_similarity": min_similarity,
                "top_k_clusters": top_k_clusters,
                "n_unclustered_before": len(unclustered),
                "n_assigned": 0,
                "n_unassigned_after": len(unclustered),
                "n_expanded_clusters": 0,
            }
        )

    by_id = clusters.cluster_by_id
    updated_by_id = dict(by_id)
    added: list[SparseCluster] = []
    expanded_by_child_id: dict[str, str] = {}

    working = clusters
    for child_idx, (child_id, assigned) in enumerate(assignments.items()):
        child = updated_by_id[child_id]
        assigned_rows = np.asarray([entity_idx for entity_idx, _ in assigned], dtype=np.int64)
        assigned_scores = np.asarray([score for _, score in assigned], dtype=np.float32)
        expanded_id = _unique_merge_id(
            working.with_clusters(tuple(working.clusters) + tuple(added)),
            phase="assign_unclustered_to_nearest_cluster",
            group_idx=child_idx,
        )
        entity_indices = np.unique(np.concatenate([child.entity_indices, assigned_rows])).astype(np.int64)
        centroid = _centroid_from_srp_rows(
            srp,
            child.entity_indices,
            assigned_rows,
            assigned_weight=assigned_weight,
            centroid_top_m=centroid_top_m if centroid_top_m is not None else max(1, int(child.centroid.indices.size)),
            normalize=normalize_centroids,
        )
        expanded = SparseCluster(
            cluster_id=expanded_id,
            centroid=centroid,
            entity_indices=entity_indices,
            source_cluster_ids=tuple(dict.fromkeys(child.source_cluster_ids or (child.cluster_id,))),
            child_cluster_ids=(child.cluster_id,),
            label=None,
            description=None,
            stats={
                "core_entity_count": int(child.entity_count),
                "assigned_entity_count": int(assigned_rows.size),
                "mean_assignment_similarity": float(assigned_scores.mean()) if assigned_scores.size else 0.0,
                "min_assignment_similarity": float(assigned_scores.min()) if assigned_scores.size else 0.0,
                "max_assignment_similarity": float(assigned_scores.max()) if assigned_scores.size else 0.0,
            },
            metadata={
                "assignment_method": "nearest_cluster",
                "base_cluster_id": child.cluster_id,
                "base_label": child.label,
                "base_description": child.description,
                "assigned_entity_indices": tuple(int(idx) for idx in assigned_rows.tolist()),
                "assigned_similarities": tuple(float(score) for score in assigned_scores.tolist()),
                "metric": metric,
                "min_similarity": min_similarity,
                "top_k_clusters": int(top_k_clusters),
                "assigned_weight": float(assigned_weight),
                "centroid_mode": "mean_srp_rows",
                "centroid_top_m": centroid_top_m,
                "centroid_top_k": centroid_top_m,
                "centroid_top_m_effective": int(centroid_top_m if centroid_top_m is not None else max(1, int(child.centroid.indices.size))),
            },
        )
        added.append(expanded)
        expanded_by_child_id[child.cluster_id] = expanded.cluster_id
        child_parents = tuple(dict.fromkeys(child.parent_cluster_ids + (expanded.cluster_id,)))
        updated_by_id[child.cluster_id] = child.with_updates(parent_cluster_ids=child_parents)

    next_active_ids: list[str] = []
    for cluster_id in clusters.active_cluster_ids or ():
        next_active_ids.append(expanded_by_child_id.get(cluster_id, cluster_id))
    for child_id, expanded_id in expanded_by_child_id.items():
        if child_id not in set(clusters.active_cluster_ids or ()) and expanded_id not in next_active_ids:
            next_active_ids.append(expanded_id)
    next_active_ids = list(dict.fromkeys(next_active_ids))

    all_clusters = [updated_by_id[cluster.cluster_id] for cluster in clusters.clusters] + added
    n_unassigned_after = len([idx for idx in unclustered if idx not in assigned_entities])
    if verbose:
        print(
            "[assign_unclustered_to_nearest_cluster] "
            f"unclustered={len(unclustered)} assigned={len(assigned_entities)} "
            f"remaining={n_unassigned_after} expanded_clusters={len(added)}"
        )
    return clusters.with_clusters(all_clusters).with_active_cluster_ids(
        next_active_ids,
        history_entry={
            "phase": "assign_unclustered_to_nearest_cluster",
            "changed": True,
            "cluster_scope": cluster_scope,
            "coverage_scope": coverage_scope,
            "metric": metric,
            "min_similarity": min_similarity,
            "top_k_clusters": top_k_clusters,
            "assigned_weight": float(assigned_weight),
            "centroid_top_m": centroid_top_m,
            "centroid_top_k": centroid_top_m,
            "normalize_centroids": bool(normalize_centroids),
            "n_unclustered_before": len(unclustered),
            "n_assigned": len(assigned_entities),
            "n_unassigned_after": n_unassigned_after,
            "n_expanded_clusters": len(added),
        },
    )


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
