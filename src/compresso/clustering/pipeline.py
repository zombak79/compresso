from __future__ import annotations

from collections.abc import Callable
from typing import Literal, Sequence

import numpy as np

from compresso.params.srp import SRPTensor
from .activation import build_activation_clusters
from .merge import filter_clusters_by_size, merge_clusters_by_entity_iou, merge_clusters_by_tag_similarity
from .tags import assign_cluster_tags
from .types import SparseClusterSet


ClusterBuildStep = Callable[[SRPTensor], SparseClusterSet]
ClusterTransformStep = Callable[[SparseClusterSet], SparseClusterSet]


def _step_name(step) -> str:
    if hasattr(step, "__name__"):
        return str(step.__name__)
    if hasattr(step, "func"):
        return _step_name(step.func)
    return step.__class__.__name__


def run_clustering_pipeline(
    srp: SRPTensor,
    steps: Sequence[ClusterBuildStep | ClusterTransformStep],
    *,
    verbose: bool = False,
) -> SparseClusterSet:
    """Run an explicit sparse clustering pipeline.

    The first step consumes an ``SRPTensor`` and must return a
    ``SparseClusterSet``. Every following step consumes and returns a
    ``SparseClusterSet``. This keeps base cluster construction separate from
    optional merge/tag/filter phases while staying easy to compose with
    ``functools.partial`` or small lambdas.
    """
    if not steps:
        raise ValueError("steps must contain at least one SRP -> SparseClusterSet function")

    if verbose:
        print(f"[cluster_pipeline] step 1/{len(steps)}: {_step_name(steps[0])}")
    first = steps[0](srp)  # type: ignore[arg-type]
    if not isinstance(first, SparseClusterSet):
        raise TypeError("first pipeline step must return SparseClusterSet")
    if verbose:
        print(f"[cluster_pipeline] step 1/{len(steps)} done: clusters={len(first.clusters)}")

    clusters = first
    for idx, step in enumerate(steps[1:], start=2):
        before = len(clusters.clusters)
        if verbose:
            print(f"[cluster_pipeline] step {idx}/{len(steps)}: {_step_name(step)} clusters_in={before}")
        out = step(clusters)  # type: ignore[arg-type]
        if not isinstance(out, SparseClusterSet):
            raise TypeError("cluster pipeline transform steps must return SparseClusterSet")
        clusters = out
        if verbose:
            print(f"[cluster_pipeline] step {idx}/{len(steps)} done: clusters={len(clusters.clusters)}")
    return clusters


def cluster_srp(
    srp: SRPTensor,
    *,
    mode: Literal["dominant_signed", "top_m_signed", "combo_signed"] = "dominant_signed",
    top_m: int = 1,
    combo_size: int = 1,
    min_cluster_size: int = 1,
    post_merge_min_cluster_size: int | None = None,
    entity_ids: np.ndarray | None = None,
    activation_iou_threshold: float | None = None,
    entity_tag_matrix=None,
    tag_names: Sequence[str] | None = None,
    tag_method: Literal["tfidf", "counts"] = "tfidf",
    top_k_tags: int = 5,
    tag_similarity_threshold: float | None = None,
    tag_similarity_metric: Literal["weighted_jaccard", "cosine"] = "weighted_jaccard",
    max_merge_rounds: int = 10,
    verbose: bool = False,
    show_progress: bool = False,
) -> SparseClusterSet:
    """Convenience pipeline for sparse clustering.

    Semantic labeling/merging is intentionally not part of v1; callers can add
    it later as callbacks operating on the returned SparseClusterSet.
    """
    clusters = build_activation_clusters(
        srp,
        mode=mode,
        top_m=top_m,
        combo_size=combo_size,
        min_cluster_size=min_cluster_size,
        entity_ids=entity_ids,
        show_progress=show_progress,
    )
    if verbose:
        print(f"[cluster_srp] build_activation_clusters: clusters={len(clusters.clusters)}")
    if activation_iou_threshold is not None:
        before = len(clusters.clusters)
        clusters = merge_clusters_by_entity_iou(
            clusters,
            threshold=activation_iou_threshold,
            max_rounds=max_merge_rounds,
            verbose=verbose,
            show_progress=show_progress,
        )
        if verbose:
            print(f"[cluster_srp] merge_clusters_by_entity_iou: {before} -> {len(clusters.clusters)}")
    if entity_tag_matrix is not None:
        if tag_names is None:
            raise ValueError("tag_names must be provided when entity_tag_matrix is provided")
        if verbose:
            print(f"[cluster_srp] assign_cluster_tags: clusters={len(clusters.clusters)}")
        clusters = assign_cluster_tags(
            clusters,
            entity_tag_matrix=entity_tag_matrix,
            tag_names=tag_names,
            method=tag_method,
            top_k=top_k_tags,
        )
        if tag_similarity_threshold is not None:
            for _ in range(max_merge_rounds):
                before = len(clusters.clusters)
                clusters = merge_clusters_by_tag_similarity(
                    clusters,
                    threshold=tag_similarity_threshold,
                    metric=tag_similarity_metric,
                    max_rounds=1,
                    verbose=verbose,
                    show_progress=show_progress,
                )
                if len(clusters.clusters) == before:
                    if verbose:
                        print(f"[cluster_srp] merge_clusters_by_tag_similarity: unchanged at {len(clusters.clusters)}")
                    break
                if verbose:
                    print(f"[cluster_srp] merge_clusters_by_tag_similarity: {before} -> {len(clusters.clusters)}")
                clusters = assign_cluster_tags(
                    clusters,
                    entity_tag_matrix=entity_tag_matrix,
                    tag_names=tag_names,
                    method=tag_method,
                    top_k=top_k_tags,
                )
    if post_merge_min_cluster_size is not None:
        before = len(clusters.clusters)
        clusters = filter_clusters_by_size(clusters, min_cluster_size=post_merge_min_cluster_size)
        if verbose:
            print(f"[cluster_srp] filter_clusters_by_size: {before} -> {len(clusters.clusters)}")
    return clusters
