from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from compresso.params.srp import SRPTensor
from .activation import build_activation_clusters
from .merge import (
    filter_clusters_by_size,
    link_clusters_by_entity_containment,
    link_clusters_by_feature_containment,
    materialize_link_merges,
    merge_clusters_by_entity_containment,
    merge_clusters_by_entity_iou,
    merge_clusters_by_feature_containment,
    merge_clusters_by_tag_similarity,
    prune_redundant_active_clusters,
)
from .tags import assign_cluster_tags
from .types import SparseClusterSet


ClusterBuildStep = Callable[[SRPTensor], SparseClusterSet]
ClusterTransformStep = Callable[[SparseClusterSet], SparseClusterSet]


class AbstractClustering(ABC):
    @abstractmethod
    def __call__(self, srp: SRPTensor) -> SparseClusterSet:
        raise NotImplementedError


class AbstractClusterTransform(ABC):
    @abstractmethod
    def __call__(self, clusters: SparseClusterSet) -> SparseClusterSet:
        raise NotImplementedError


class AbstractMerging(AbstractClusterTransform):
    pass


@dataclass(frozen=True)
class DominantSignedClustering(AbstractClustering):
    min_cluster_size: int = 1
    show_progress: bool = False

    def __call__(self, srp: SRPTensor) -> SparseClusterSet:
        return build_activation_clusters(
            srp,
            mode="dominant_signed",
            min_cluster_size=self.min_cluster_size,
            show_progress=self.show_progress,
        )


@dataclass(frozen=True)
class TopMSignedClustering(AbstractClustering):
    top_m: int = 1
    min_cluster_size: int = 1
    show_progress: bool = False

    def __call__(self, srp: SRPTensor) -> SparseClusterSet:
        return build_activation_clusters(
            srp,
            mode="top_m_signed",
            top_m=self.top_m,
            min_cluster_size=self.min_cluster_size,
            show_progress=self.show_progress,
        )


@dataclass(frozen=True)
class ComboSignedClustering(AbstractClustering):
    top_m: int = 1
    combo_size: int = 1
    min_cluster_size: int = 1
    show_progress: bool = False

    def __call__(self, srp: SRPTensor) -> SparseClusterSet:
        return build_activation_clusters(
            srp,
            mode="combo_signed",
            top_m=self.top_m,
            combo_size=self.combo_size,
            min_cluster_size=self.min_cluster_size,
            show_progress=self.show_progress,
        )


@dataclass(frozen=True)
class EntityIoUMerge(AbstractMerging):
    threshold: float
    max_rounds: int = 10
    normalize_centroids: bool = True
    verbose: bool = False
    show_progress: bool = False

    def __call__(self, clusters: SparseClusterSet) -> SparseClusterSet:
        return merge_clusters_by_entity_iou(
            clusters,
            threshold=self.threshold,
            max_rounds=self.max_rounds,
            normalize_centroids=self.normalize_centroids,
            verbose=self.verbose,
            show_progress=self.show_progress,
        )


@dataclass(frozen=True)
class EntityContainmentMerge(AbstractMerging):
    threshold: float = 1.0
    max_rounds: int = 10
    normalize_centroids: bool = True
    verbose: bool = False
    show_progress: bool = False

    def __call__(self, clusters: SparseClusterSet) -> SparseClusterSet:
        return merge_clusters_by_entity_containment(
            clusters,
            threshold=self.threshold,
            max_rounds=self.max_rounds,
            normalize_centroids=self.normalize_centroids,
            verbose=self.verbose,
            show_progress=self.show_progress,
        )


@dataclass(frozen=True)
class FeatureContainmentMerge(AbstractMerging):
    threshold: float = 1.0
    signed: bool = True
    max_rounds: int = 10
    normalize_centroids: bool = True
    verbose: bool = False
    show_progress: bool = False

    def __call__(self, clusters: SparseClusterSet) -> SparseClusterSet:
        return merge_clusters_by_feature_containment(
            clusters,
            threshold=self.threshold,
            signed=self.signed,
            max_rounds=self.max_rounds,
            normalize_centroids=self.normalize_centroids,
            verbose=self.verbose,
            show_progress=self.show_progress,
        )


@dataclass(frozen=True)
class EntityContainmentLink(AbstractClusterTransform):
    threshold: float = 1.0
    child_scope: Literal["active", "all", "leaves", "roots"] = "leaves"
    parent_scope: Literal["active", "all", "leaves", "roots"] = "all"
    require_parent_larger: bool = True
    skip_existing_ancestors: bool = True
    verbose: bool = False
    show_progress: bool = False

    def __call__(self, clusters: SparseClusterSet) -> SparseClusterSet:
        return link_clusters_by_entity_containment(
            clusters,
            threshold=self.threshold,
            child_scope=self.child_scope,
            parent_scope=self.parent_scope,
            require_parent_larger=self.require_parent_larger,
            skip_existing_ancestors=self.skip_existing_ancestors,
            verbose=self.verbose,
            show_progress=self.show_progress,
        )


@dataclass(frozen=True)
class FeatureContainmentLink(AbstractClusterTransform):
    threshold: float = 1.0
    signed: bool = True
    child_scope: Literal["active", "all", "leaves", "roots"] = "leaves"
    parent_scope: Literal["active", "all", "leaves", "roots"] = "all"
    require_parent_larger: bool = True
    skip_existing_ancestors: bool = True
    verbose: bool = False
    show_progress: bool = False

    def __call__(self, clusters: SparseClusterSet) -> SparseClusterSet:
        return link_clusters_by_feature_containment(
            clusters,
            threshold=self.threshold,
            signed=self.signed,
            child_scope=self.child_scope,
            parent_scope=self.parent_scope,
            require_parent_larger=self.require_parent_larger,
            skip_existing_ancestors=self.skip_existing_ancestors,
            verbose=self.verbose,
            show_progress=self.show_progress,
        )


@dataclass(frozen=True)
class MaterializeLinkMerges(AbstractClusterTransform):
    parent_scope: Literal["active", "all", "leaves", "roots"] = "active"
    include_descendants: bool = False
    min_children: int = 1
    normalize_centroids: bool = True
    activate: bool = True
    verbose: bool = False

    def __call__(self, clusters: SparseClusterSet) -> SparseClusterSet:
        return materialize_link_merges(
            clusters,
            parent_scope=self.parent_scope,
            include_descendants=self.include_descendants,
            min_children=self.min_children,
            normalize_centroids=self.normalize_centroids,
            activate=self.activate,
            verbose=self.verbose,
        )


@dataclass(frozen=True)
class TagSimilarityMerge(AbstractMerging):
    threshold: float
    metric: Literal["weighted_jaccard", "cosine"] = "weighted_jaccard"
    max_rounds: int = 10
    normalize_centroids: bool = True
    verbose: bool = False
    show_progress: bool = False

    def __call__(self, clusters: SparseClusterSet) -> SparseClusterSet:
        return merge_clusters_by_tag_similarity(
            clusters,
            threshold=self.threshold,
            metric=self.metric,
            max_rounds=self.max_rounds,
            normalize_centroids=self.normalize_centroids,
            verbose=self.verbose,
            show_progress=self.show_progress,
        )


@dataclass(frozen=True)
class AssignTags(AbstractClusterTransform):
    entity_tag_matrix: object
    tag_names: Sequence[str]
    method: Literal["tfidf", "counts"] = "tfidf"
    top_k: int = 5
    min_score: float = 0.0

    def __call__(self, clusters: SparseClusterSet) -> SparseClusterSet:
        return assign_cluster_tags(
            clusters,
            entity_tag_matrix=self.entity_tag_matrix,
            tag_names=self.tag_names,
            method=self.method,
            top_k=self.top_k,
            min_score=self.min_score,
        )


@dataclass(frozen=True)
class SizeFilter(AbstractClusterTransform):
    min_cluster_size: int

    def __call__(self, clusters: SparseClusterSet) -> SparseClusterSet:
        return filter_clusters_by_size(clusters, min_cluster_size=self.min_cluster_size)


@dataclass(frozen=True)
class PruneRedundantRoots(AbstractClusterTransform):
    verbose: bool = False

    def __call__(self, clusters: SparseClusterSet) -> SparseClusterSet:
        return prune_redundant_active_clusters(clusters, verbose=self.verbose)


@dataclass(frozen=True)
class ClusteringPipeline:
    steps: Sequence[ClusterBuildStep | ClusterTransformStep]
    verbose: bool = False

    def fit(self, srp: SRPTensor) -> SparseClusterSet:
        return run_clustering_pipeline(srp, self.steps, verbose=self.verbose)

    def __call__(self, srp: SRPTensor) -> SparseClusterSet:
        return self.fit(srp)


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
        print(f"[cluster_pipeline] step 1/{len(steps)} done: active_clusters={len(first.active_clusters)} nodes={len(first.clusters)}")

    clusters = first
    for idx, step in enumerate(steps[1:], start=2):
        before = len(clusters.active_clusters)
        if verbose:
            print(f"[cluster_pipeline] step {idx}/{len(steps)}: {_step_name(step)} active_clusters_in={before}")
        out = step(clusters)  # type: ignore[arg-type]
        if not isinstance(out, SparseClusterSet):
            raise TypeError("cluster pipeline transform steps must return SparseClusterSet")
        clusters = out
        if verbose:
            print(f"[cluster_pipeline] step {idx}/{len(steps)} done: active_clusters={len(clusters.active_clusters)} nodes={len(clusters.clusters)}")
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
        print(f"[cluster_srp] build_activation_clusters: active_clusters={len(clusters.active_clusters)} nodes={len(clusters.clusters)}")
    if activation_iou_threshold is not None:
        before = len(clusters.active_clusters)
        clusters = merge_clusters_by_entity_iou(
            clusters,
            threshold=activation_iou_threshold,
            max_rounds=max_merge_rounds,
            verbose=verbose,
            show_progress=show_progress,
        )
        if verbose:
            print(f"[cluster_srp] merge_clusters_by_entity_iou: {before} -> {len(clusters.active_clusters)}")
    if entity_tag_matrix is not None:
        if tag_names is None:
            raise ValueError("tag_names must be provided when entity_tag_matrix is provided")
        if verbose:
            print(f"[cluster_srp] assign_cluster_tags: active_clusters={len(clusters.active_clusters)} nodes={len(clusters.clusters)}")
        clusters = assign_cluster_tags(
            clusters,
            entity_tag_matrix=entity_tag_matrix,
            tag_names=tag_names,
            method=tag_method,
            top_k=top_k_tags,
        )
        if tag_similarity_threshold is not None:
            for _ in range(max_merge_rounds):
                before = len(clusters.active_clusters)
                clusters = merge_clusters_by_tag_similarity(
                    clusters,
                    threshold=tag_similarity_threshold,
                    metric=tag_similarity_metric,
                    max_rounds=1,
                    verbose=verbose,
                    show_progress=show_progress,
                )
                if len(clusters.active_clusters) == before:
                    if verbose:
                        print(f"[cluster_srp] merge_clusters_by_tag_similarity: unchanged at {len(clusters.active_clusters)}")
                    break
                if verbose:
                    print(f"[cluster_srp] merge_clusters_by_tag_similarity: {before} -> {len(clusters.active_clusters)}")
                clusters = assign_cluster_tags(
                    clusters,
                    entity_tag_matrix=entity_tag_matrix,
                    tag_names=tag_names,
                    method=tag_method,
                    top_k=top_k_tags,
                )
    if post_merge_min_cluster_size is not None:
        before = len(clusters.active_clusters)
        clusters = filter_clusters_by_size(clusters, min_cluster_size=post_merge_min_cluster_size)
        if verbose:
            print(f"[cluster_srp] filter_clusters_by_size: {before} -> {len(clusters.active_clusters)}")
    return clusters
