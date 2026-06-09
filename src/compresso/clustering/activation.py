from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Literal

import numpy as np
import torch

from compresso.params.srp import SRPTensor
from .types import SparseCluster, SparseClusterSet, SparseVector


def _progress_iter(iterable, *, enabled: bool, total: int | None = None, desc: str = ""):
    if not enabled:
        return iterable
    try:
        from tqdm.auto import tqdm
    except Exception:  # pragma: no cover - optional progress dependency
        return iterable
    return tqdm(iterable, total=total, desc=desc)


def _as_srp_2d(srp: SRPTensor) -> SRPTensor:
    if srp.prefix_shape is not None:
        return SRPTensor(cols=srp.cols.reshape(srp.rows, srp.k), vals=srp.vals.reshape(srp.rows, srp.k), shape=srp.shape)
    return srp


def build_activation_clusters(
    srp: SRPTensor,
    *,
    mode: Literal["dominant_signed", "top_m_signed", "combo_signed"] = "dominant_signed",
    top_m: int = 1,
    combo_size: int = 1,
    min_cluster_size: int = 1,
    entity_ids: np.ndarray | None = None,
    normalize_centroids: bool = True,
    ignore_zero: bool = True,
    show_progress: bool = False,
) -> SparseClusterSet:
    """Build base clusters from signed sparse features.

    Rows are entities and columns are sparse features.

    Modes:
      - dominant_signed: assign each entity to its strongest signed feature.
      - top_m_signed: assign each entity to each of its top-m signed features.
      - combo_signed: assign each entity to exact AND-combinations of size
        combo_size drawn from its top-m signed features.
    """
    if mode not in {"dominant_signed", "top_m_signed", "combo_signed"}:
        raise ValueError("mode must be 'dominant_signed', 'top_m_signed', or 'combo_signed'")
    if mode == "dominant_signed":
        top_m = 1
        combo_size = 1
    if mode == "top_m_signed":
        combo_size = 1
    if top_m < 1 or top_m > srp.k:
        raise ValueError(f"top_m must be in [1, {srp.k}], got {top_m}")
    if combo_size < 1 or combo_size > top_m:
        raise ValueError(f"combo_size must be in [1, {top_m}], got {combo_size}")
    if min_cluster_size < 1:
        raise ValueError("min_cluster_size must be >= 1")

    srp = _as_srp_2d(srp)
    cols = srp.cols.detach().cpu()
    vals = srp.vals.detach().cpu()
    groups: dict[tuple[tuple[int, int], ...], list[int]] = defaultdict(list)
    activation_values: dict[tuple[tuple[int, int], ...], list[float]] = defaultdict(list)

    top_idx = torch.topk(vals.abs(), k=top_m, dim=1, largest=True, sorted=True).indices
    for row in _progress_iter(range(srp.rows), enabled=show_progress, total=srp.rows, desc="build_activation_clusters: rows"):
        signed_features: list[tuple[int, int, float]] = []
        seen_features: set[tuple[int, int]] = set()
        for pos in top_idx[row].tolist():
            value = float(vals[row, pos].item())
            if ignore_zero and value == 0.0:
                continue
            feature = int(cols[row, pos].item())
            sign = 1 if value >= 0.0 else -1
            signed = (feature, sign)
            if signed in seen_features:
                continue
            seen_features.add(signed)
            signed_features.append((feature, sign, value))

        for combo in combinations(signed_features, combo_size):
            key = tuple(sorted((feature, sign) for feature, sign, _ in combo))
            score = float(combo[0][2]) if combo_size == 1 else float(np.mean([abs(value) for _, _, value in combo]))
            groups[key].append(row)
            activation_values[key].append(score)

    clusters: list[SparseCluster] = []
    group_items = sorted(groups.items())
    for key, rows in _progress_iter(
        group_items,
        enabled=show_progress,
        total=len(group_items),
        desc="build_activation_clusters: clusters",
    ):
        if len(rows) < min_cluster_size:
            continue
        features = np.asarray([feature for feature, _ in key], dtype=np.int64)
        signs = np.asarray([float(sign) for _, sign in key], dtype=np.float32)
        centroid = SparseVector(features, signs, srp.cols_total)
        if normalize_centroids:
            centroid = centroid.normalized()
        act = np.asarray(activation_values[key], dtype=np.float32)
        act_abs = np.abs(act)
        parts = [f"{feature}:{'pos' if sign > 0 else 'neg'}" for feature, sign in key]
        if len(parts) == 1:
            cluster_id = f"feature:{parts[0]}"
        else:
            cluster_id = "combo:" + "&".join(parts)
        clusters.append(
            SparseCluster(
                cluster_id=cluster_id,
                centroid=centroid,
                entity_indices=np.asarray(rows, dtype=np.int64),
                source_cluster_ids=(cluster_id,),
                stats={
                    "mean_activation": float(act.mean()) if act.size else 0.0,
                    "mean_abs_activation": float(act_abs.mean()) if act_abs.size else 0.0,
                    "max_abs_activation": float(act_abs.max()) if act_abs.size else 0.0,
                },
                metadata={
                    "features": tuple(int(feature) for feature, _ in key),
                    "signs": tuple(int(sign) for _, sign in key),
                    "combo_size": len(key),
                    **({"feature_id": int(key[0][0]), "sign": int(key[0][1])} if len(key) == 1 else {}),
                },
            )
        )

    return SparseClusterSet(
        clusters=tuple(clusters),
        n_entities=srp.rows,
        n_features=srp.cols_total,
        entity_ids=entity_ids,
        assignment_mode=mode,
        history=(
            {
                "phase": "build_activation_clusters",
                "mode": mode,
                "top_m": top_m,
                "combo_size": combo_size,
                "min_cluster_size": min_cluster_size,
                "n_clusters": len(clusters),
            },
        ),
    )


def assign_to_clusters(
    srp: SRPTensor,
    clusters: SparseClusterSet,
    *,
    top_k: int = 1,
) -> list[list[tuple[str, float]]]:
    """Assign new sparse rows to clusters by centroid overlap score."""
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    srp = _as_srp_2d(srp)
    if srp.cols_total != clusters.n_features:
        raise ValueError("srp feature dimension must match cluster set")

    candidate_clusters = clusters.active_clusters
    centroid_maps = [dict(zip(c.centroid.indices.tolist(), c.centroid.values.tolist())) for c in candidate_clusters]
    out: list[list[tuple[str, float]]] = []
    cols = srp.cols.detach().cpu().numpy()
    vals = srp.vals.detach().cpu().numpy()
    for row in range(srp.rows):
        row_pairs = dict(zip(cols[row].tolist(), vals[row].tolist()))
        scored = []
        for cluster, cmap in zip(candidate_clusters, centroid_maps):
            score = 0.0
            for idx, cval in cmap.items():
                score += float(row_pairs.get(idx, 0.0)) * float(cval)
            scored.append((cluster.cluster_id, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        out.append(scored[:top_k])
    return out
