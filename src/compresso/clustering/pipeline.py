from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from compresso.params.srp import SRPTensor
from .activation import build_activation_clusters, build_feature_path_clusters, build_srp_similarity_clusters
from .labels import label_clusters
from .merge import (
    assign_unclustered_to_nearest_cluster,
    compact_hidden_clusters,
    filter_clusters_by_size,
    link_clusters_by_entity_containment,
    link_clusters_by_feature_containment,
    materialize_link_merges,
    merge_clusters_by_centroid_similarity,
    merge_clusters_by_entity_containment,
    merge_clusters_by_entity_iou,
    merge_clusters_by_feature_containment,
    merge_clusters_by_duplicate_label,
    merge_clusters_by_tag_similarity,
    prune_redundant_active_clusters,
)
from .semantic import merge_clusters_by_semantic_similarity
from .tags import assign_cluster_tags
from .types import SparseCluster, SparseClusterSet


ClusterBuildStep = Callable[[SRPTensor], SparseClusterSet]
ClusterTransformStep = Callable[[SparseClusterSet], SparseClusterSet]


class AbstractClustering(ABC):
    """Base class for pipeline steps that build clusters from an ``SRPTensor``.

    Clustering steps are the first step in a :class:`ClusteringPipeline`.
    They consume sparse entity representations and return a
    :class:`~compresso.clustering.SparseClusterSet`.
    """

    @abstractmethod
    def __call__(self, srp: SRPTensor) -> SparseClusterSet:
        raise NotImplementedError


class AbstractClusterTransform(ABC):
    """Base class for pipeline steps that transform an existing cluster graph.

    Transform steps consume and return a :class:`SparseClusterSet`. They are
    used after a clustering step for linking, merging, labeling, tagging, or
    filtering clusters.
    """

    @abstractmethod
    def __call__(self, clusters: SparseClusterSet) -> SparseClusterSet:
        raise NotImplementedError


class AbstractMerging(AbstractClusterTransform):
    """Base class for transforms that create merge-parent clusters.

    Merge steps are non-destructive by convention: original clusters remain in
    the graph as children and new parent nodes become the active frontier.
    """

    pass


@dataclass(frozen=True)
class DominantSignedClustering(AbstractClustering):
    """Cluster each entity by its single strongest signed feature.

    This is the simplest activation clustering mode. For every SRP row, the
    feature with largest absolute activation is selected, including its sign.
    Entities with the same ``(feature, sign)`` key form one cluster.

    Parameters
    ----------
    min_cluster_size:
        Drop clusters with fewer entities than this value during construction.
    show_progress:
        Show a tqdm progress bar while assigning rows.
    """

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
    """Cluster each entity by each of its top-m signed features.

    Every entity may belong to multiple clusters: one for each of its
    strongest ``top_m`` signed feature activations. With ``top_m=1`` this is
    equivalent to :class:`DominantSignedClustering`.

    Parameters
    ----------
    top_m:
        Number of strongest features per entity to use.
    min_cluster_size:
        Drop clusters with fewer entities than this value.
    show_progress:
        Show tqdm progress while building clusters.
    """

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
    """Cluster entities by exact combinations of signed features.

    For each entity, take its top ``top_m`` signed features and create exact
    AND-combination keys of length ``combo_size``. Two entities share a cluster
    only when they contain the same signed feature combination.

    Parameters
    ----------
    top_m:
        Candidate feature pool size for each entity.
    combo_size:
        Size of exact signed-feature combinations.
    min_cluster_size:
        Drop clusters with fewer entities than this value.
    show_progress:
        Show tqdm progress while building clusters.
    """

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
class FeaturePathClustering(AbstractClustering):
    """Build a hierarchy by recursively splitting on next strongest features.

    At the root level each entity starts from one of its top ``top_m`` signed
    features. Inside each cluster, entities are split again by the next
    strongest feature not already used in the path. This creates an explicit
    feature-path hierarchy such as ``feature A -> feature B -> feature C``.

    Parameters
    ----------
    top_m:
        Number of starting signed features per entity.
    max_depth:
        Maximum feature-path depth. ``None`` allows paths up to available
        sparse features.
    min_cluster_size:
        Stop or drop nodes smaller than this size. ``None`` disables the size
        stop.
    min_activation:
        Ignore candidate features whose absolute activation is below this
        value. ``None`` disables the threshold.
    show_progress:
        Show tqdm progress while building paths.
    """

    top_m: int = 1
    max_depth: int | None = None
    min_cluster_size: int | None = None
    min_activation: float | None = None
    show_progress: bool = False

    def __call__(self, srp: SRPTensor) -> SparseClusterSet:
        return build_feature_path_clusters(
            srp,
            top_m=self.top_m,
            max_depth=self.max_depth,
            min_cluster_size=self.min_cluster_size,
            min_activation=self.min_activation,
            show_progress=self.show_progress,
        )


@dataclass(frozen=True)
class SRPSimilarityClustering(AbstractClustering):
    """Build clusters as connected components in SRP similarity space.

    The SRP rows are compared by dense similarity, edges with similarity above
    ``threshold`` are kept, and connected components become clusters. This is
    useful when the cluster should be defined by representation similarity
    rather than by a shared feature id.

    Parameters
    ----------
    threshold:
        Minimum pairwise similarity for an edge.
    top_k:
        Limit each row to its top-k nearest neighbors before thresholding.
        ``None`` compares against all rows.
    min_cluster_size:
        Drop connected components smaller than this size.
    normalize_rows:
        If ``True``, L2-normalize rows before similarity computation.
    min_local_density:
        Optional fraction of within-component neighbors an entity must keep to
        remain in a component.
    centroid_top_k:
        If provided, keep only this many largest centroid features.
    batch_size:
        Number of rows scored per matrix multiplication batch.
    show_progress:
        Show tqdm progress while scoring batches.
    """

    threshold: float
    top_k: int | None = 100
    min_cluster_size: int = 2
    normalize_rows: bool = True
    min_local_density: float | None = None
    centroid_top_k: int | None = None
    batch_size: int = 1024
    show_progress: bool = False

    def __call__(self, srp: SRPTensor) -> SparseClusterSet:
        return build_srp_similarity_clusters(
            srp,
            threshold=self.threshold,
            top_k=self.top_k,
            min_cluster_size=self.min_cluster_size,
            normalize_rows=self.normalize_rows,
            min_local_density=self.min_local_density,
            centroid_top_k=self.centroid_top_k,
            batch_size=self.batch_size,
            show_progress=self.show_progress,
        )


@dataclass(frozen=True)
class EntityIoUMerge(AbstractMerging):
    """Merge active clusters whose entity sets have high intersection-over-union.

    Active clusters are treated as nodes in a similarity graph. Pairs with
    entity IoU greater than or equal to ``threshold`` are connected, connected
    components are materialized as new parent clusters, and the process repeats
    until convergence or ``max_rounds``.

    Parameters
    ----------
    threshold:
        Minimum entity IoU in ``[0, 1]``.
    max_rounds:
        Maximum iterative merge rounds.
    normalize_centroids:
        L2-normalize newly created parent centroids.
    verbose:
        Print merge progress.
    show_progress:
        Show tqdm progress while comparing clusters.
    """

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
    """Merge active clusters when one entity set is mostly contained in another.

    Containment is ``|A intersection B| / min(|A|, |B|)``. A strict subset has
    score ``1.0`` even when IoU is small. This is useful for collapsing small
    highly-specific clusters into broader parents.

    Parameters
    ----------
    threshold:
        Minimum containment score in ``[0, 1]``.
    max_rounds:
        Maximum iterative merge rounds.
    normalize_centroids:
        L2-normalize newly created parent centroids.
    verbose:
        Print merge progress.
    show_progress:
        Show tqdm progress while comparing clusters.
    """

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
    """Merge active clusters when feature supports are mostly contained.

    Feature containment is ``|F_A intersection F_B| / min(|F_A|, |F_B|)``.
    With ``signed=True``, positive and negative use of the same feature are
    treated as different support elements.

    Parameters
    ----------
    threshold:
        Minimum feature-containment score in ``[0, 1]``.
    signed:
        Compare signed ``(feature, sign)`` pairs instead of feature ids only.
    max_rounds:
        Maximum iterative merge rounds.
    normalize_centroids:
        L2-normalize newly created parent centroids.
    verbose:
        Print merge progress.
    show_progress:
        Show tqdm progress while comparing clusters.
    """

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
    """Add parent-child links based on entity containment without merging nodes.

    This is a graph-linking step, not a merge step. It preserves the active
    frontier and only adds DAG edges from contained child clusters to broader
    parent clusters. A child may receive multiple parents.

    Parameters
    ----------
    threshold:
        Minimum entity containment score in ``[0, 1]``.
    child_scope:
        Which clusters may become children: ``"active"``, ``"all"``,
        ``"leaves"``, or ``"roots"``.
    parent_scope:
        Which clusters may become parents.
    require_parent_larger:
        Require the parent candidate to contain more entities than the child.
    skip_existing_ancestors:
        Avoid adding links that are already implied by existing ancestry.
    verbose:
        Print linking progress.
    show_progress:
        Show tqdm progress while scanning child clusters.
    """

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
    """Add parent-child links based on feature-support containment.

    This is the feature analogue of :class:`EntityContainmentLink`. It creates
    graph edges without creating merge nodes or changing the active frontier.

    Parameters
    ----------
    threshold:
        Minimum feature-containment score in ``[0, 1]``.
    signed:
        Compare signed ``(feature, sign)`` pairs instead of feature ids only.
    child_scope:
        Which clusters may become children.
    parent_scope:
        Which clusters may become parents.
    require_parent_larger:
        Require the parent candidate to have a larger feature support.
    skip_existing_ancestors:
        Avoid adding links already implied by existing ancestry.
    verbose:
        Print linking progress.
    show_progress:
        Show tqdm progress while scanning child clusters.
    """

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
    """Convert existing links into explicit merge-parent nodes.

    Link steps create edges only. This transform adds a new non-destructive
    parent node for linked structures so that downstream renderers and
    recommendation code can treat the linked group as a concrete cluster.

    Parameters
    ----------
    parent_scope:
        Which linked parent clusters should be materialized.
    include_descendants:
        If ``True``, include all descendants of a linked parent, not just its
        direct children.
    min_children:
        Minimum number of linked children required to create a materialized
        node.
    normalize_centroids:
        L2-normalize materialized parent centroids.
    activate:
        If ``True``, add materialized nodes to the active frontier.
    verbose:
        Print materialization details.
    """

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
class CentroidSimilarityMerge(AbstractMerging):
    """Merge active clusters whose sparse centroids are similar.

    This creates non-destructive parent nodes from connected components in a
    centroid-similarity graph. It is useful after initial clustering when
    different feature handles point to nearby regions in sparse space.

    Parameters
    ----------
    threshold:
        Minimum centroid similarity. For cosine this must be in ``[-1, 1]``.
    metric:
        Similarity function: ``"cosine"`` or raw ``"dot"`` product.
    top_k:
        Optional nearest-neighbor restriction. A pair is considered only when
        one cluster is in the other's top-k neighbors.
    max_rounds:
        Maximum iterative merge rounds.
    min_group_size:
        Minimum connected component size to materialize.
    normalize_centroids:
        L2-normalize newly created parent centroids.
    verbose:
        Print merge progress.
    show_progress:
        Show tqdm progress while comparing centroids.
    """

    threshold: float
    metric: Literal["cosine", "dot"] = "cosine"
    top_k: int | None = None
    max_rounds: int = 10
    min_group_size: int = 2
    normalize_centroids: bool = True
    verbose: bool = False
    show_progress: bool = False

    def __call__(self, clusters: SparseClusterSet) -> SparseClusterSet:
        return merge_clusters_by_centroid_similarity(
            clusters,
            threshold=self.threshold,
            metric=self.metric,
            top_k=self.top_k,
            max_rounds=self.max_rounds,
            min_group_size=self.min_group_size,
            normalize_centroids=self.normalize_centroids,
            verbose=self.verbose,
            show_progress=self.show_progress,
        )


@dataclass(frozen=True)
class AssignUnclusteredToNearestCluster(AbstractClusterTransform):
    """Expand coverage by assigning uncovered entities to nearest clusters.

    This step does not create singleton clusters. Instead, for each cluster
    that receives newly assigned entities, it creates an expanded parent node
    with the original cluster as a child. This preserves the discovered core
    cluster while adding coverage for entities outside the current frontier.

    Parameters
    ----------
    srp:
        Original entity representations used for nearest-cluster assignment.
    metric:
        Similarity between entity rows and cluster centroids: ``"cosine"`` or
        ``"dot"``.
    min_similarity:
        Optional minimum similarity required for assignment.
    top_k_clusters:
        Number of nearest clusters to assign each uncovered entity to.
    cluster_scope:
        Which clusters can receive assignments.
    coverage_scope:
        Which clusters count as already covering an entity: ``"active"`` or
        ``"all"``.
    assigned_weight:
        Weight of newly assigned entity vectors when recomputing expanded
        centroids.
    centroid_top_m:
        Keep only this many largest centroid features in expanded parents.
    centroid_top_k:
        Deprecated alias for ``centroid_top_m``.
    normalize_centroids:
        L2-normalize expanded parent centroids.
    verbose:
        Print assignment summary.
    """

    srp: SRPTensor
    metric: Literal["cosine", "dot"] = "cosine"
    min_similarity: float | None = None
    top_k_clusters: int = 1
    cluster_scope: Literal["active", "all", "leaves", "roots"] = "active"
    coverage_scope: Literal["active", "all"] = "active"
    assigned_weight: float = 1.0
    centroid_top_m: int | None = None
    centroid_top_k: int | None = None
    normalize_centroids: bool = True
    verbose: bool = False

    def __call__(self, clusters: SparseClusterSet) -> SparseClusterSet:
        return assign_unclustered_to_nearest_cluster(
            clusters,
            self.srp,
            metric=self.metric,
            min_similarity=self.min_similarity,
            top_k_clusters=self.top_k_clusters,
            cluster_scope=self.cluster_scope,
            coverage_scope=self.coverage_scope,
            assigned_weight=self.assigned_weight,
            centroid_top_m=self.centroid_top_m,
            centroid_top_k=self.centroid_top_k,
            normalize_centroids=self.normalize_centroids,
            verbose=self.verbose,
        )


@dataclass(frozen=True)
class TagSimilarityMerge(AbstractMerging):
    """Merge active clusters with similar assigned tag profiles.

    Run :class:`AssignTags` before this step. Tags are compared either by
    weighted Jaccard over tag scores or by cosine similarity of tag-score
    vectors. Matching groups are materialized as non-destructive parents.

    Parameters
    ----------
    threshold:
        Minimum tag-profile similarity in ``[0, 1]``.
    metric:
        Tag similarity metric: ``"weighted_jaccard"`` or ``"cosine"``.
    max_rounds:
        Maximum iterative merge rounds.
    normalize_centroids:
        L2-normalize newly created parent centroids.
    verbose:
        Print merge progress.
    show_progress:
        Show tqdm progress while comparing clusters.
    """

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
class LabelDuplicateMerge(AbstractMerging):
    """Merge clusters that have exactly the same label string.

    This is useful after LLM or rule-based labeling, where duplicate labels can
    indicate duplicate semantic segments. When ``mark_children_hidden=True``,
    merged children are preserved but marked as hidden for later compaction or
    UI rendering.

    Parameters
    ----------
    cluster_scope:
        Which clusters are considered for duplicate-label grouping.
    case_sensitive:
        If ``False``, normalize labels by case before grouping.
    mark_children_hidden:
        Mark duplicate children with metadata key ``render_hidden``.
    min_group_size:
        Minimum number of clusters with the same label required to merge.
    normalize_centroids:
        L2-normalize newly created parent centroids.
    verbose:
        Print merge summary.
    """

    cluster_scope: Literal["active", "all", "leaves", "roots"] = "active"
    case_sensitive: bool = False
    mark_children_hidden: bool = True
    min_group_size: int = 2
    normalize_centroids: bool = True
    verbose: bool = False

    def __call__(self, clusters: SparseClusterSet) -> SparseClusterSet:
        return merge_clusters_by_duplicate_label(
            clusters,
            cluster_scope=self.cluster_scope,
            case_sensitive=self.case_sensitive,
            mark_children_hidden=self.mark_children_hidden,
            min_group_size=self.min_group_size,
            normalize_centroids=self.normalize_centroids,
            verbose=self.verbose,
        )


@dataclass(frozen=True)
class CompactHiddenClusters(AbstractClusterTransform):
    """Remove hidden clusters and rewire visible graph edges.

    Hidden nodes are usually produced by :class:`LabelDuplicateMerge` with
    ``mark_children_hidden=True``. This transform physically removes them from
    the graph while preserving visible ancestor/descendant connectivity.

    Parameters
    ----------
    hidden_key:
        Metadata key used to decide whether a cluster should be removed.
    verbose:
        Print compaction summary.
    """

    hidden_key: str = "render_hidden"
    verbose: bool = False

    def __call__(self, clusters: SparseClusterSet) -> SparseClusterSet:
        return compact_hidden_clusters(clusters, hidden_key=self.hidden_key, verbose=self.verbose)


@dataclass(frozen=True)
class SemanticSimilarityMerge(AbstractMerging):
    """Merge clusters whose labels/descriptions are semantically similar.

    The user supplies ``embed_fn`` so Compresso does not depend on a specific
    embedding model or API key. Candidate cluster texts are embedded, similar
    clusters are grouped with a maximal-clique strategy, and new
    non-destructive semantic parent nodes are created. Optional callbacks can
    label those new parent nodes.

    Parameters
    ----------
    embed_fn:
        Function mapping a list of cluster texts to a 2D NumPy embedding array.
    threshold:
        Minimum cosine similarity between cluster text embeddings.
    text_fn:
        Function mapping a cluster to text. Defaults to cluster description,
        then label, then empty text.
    label_fn:
        Optional user function that names a newly created semantic parent.
    label_text_fn:
        Optional function that builds the input object passed to ``label_fn``
        from the parent cluster and its children.
    cluster_scope:
        Which clusters should be considered for semantic merging.
    max_rounds:
        Maximum iterative semantic merge rounds.
    min_group_size:
        Minimum similar group size to materialize.
    normalize_embeddings:
        L2-normalize text embeddings before similarity.
    normalize_centroids:
        L2-normalize newly created parent centroids.
    verbose:
        Print merge progress.
    show_progress:
        Show tqdm progress while processing rounds.
    """

    embed_fn: Callable[[list[str]], np.ndarray]
    threshold: float = 0.9
    text_fn: Callable[[SparseCluster], str] | None = None
    label_fn: Callable[[object], object] | None = None
    label_text_fn: Callable[[SparseCluster, list[SparseCluster]], object] | None = None
    cluster_scope: Literal["active", "all", "leaves", "roots"] = "active"
    max_rounds: int = 10
    min_group_size: int = 2
    normalize_embeddings: bool = True
    normalize_centroids: bool = True
    verbose: bool = False
    show_progress: bool = False

    def __call__(self, clusters: SparseClusterSet) -> SparseClusterSet:
        return merge_clusters_by_semantic_similarity(
            clusters,
            embed_fn=self.embed_fn,
            threshold=self.threshold,
            text_fn=self.text_fn,
            label_fn=self.label_fn,
            label_text_fn=self.label_text_fn,
            cluster_scope=self.cluster_scope,
            max_rounds=self.max_rounds,
            min_group_size=self.min_group_size,
            normalize_embeddings=self.normalize_embeddings,
            normalize_centroids=self.normalize_centroids,
            verbose=self.verbose,
            show_progress=self.show_progress,
        )


@dataclass(frozen=True)
class AssignTags(AbstractClusterTransform):
    """Assign top tags to clusters from an entity-tag matrix.

    The tag matrix is expected to be shaped ``(n_entities, n_tags)`` and can be
    a dense or sparse matrix-like object accepted by the implementation. Tags
    are aggregated over each cluster's entity indices and stored as
    :class:`~compresso.clustering.ScoredTag` objects.

    Parameters
    ----------
    entity_tag_matrix:
        Matrix containing entity-tag counts or weights.
    tag_names:
        Names corresponding to columns of ``entity_tag_matrix``.
    method:
        Scoring method: ``"tfidf"`` downweights globally common tags, while
        ``"counts"`` uses raw aggregate counts.
    top_k:
        Maximum number of tags stored per cluster.
    min_score:
        Drop tags whose score is not greater than this threshold.
    """

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
class LabelClusters(AbstractClusterTransform):
    """Populate cluster labels/descriptions with user-provided callbacks.

    Compresso coordinates the loop but does not own prompting, API keys, or
    model initialization. ``text_extractor`` converts a cluster plus metadata
    into a domain-specific input object, and ``label_fn`` converts that object
    into a label result. The result may be a string, ``(label, description)``,
    or a mapping understood by the implementation.

    Parameters
    ----------
    entity_metadata:
        Metadata table or object used by ``text_extractor``.
    text_extractor:
        Callable ``(cluster, entity_metadata) -> object``.
    label_fn:
        Callable that returns a label/description for one extracted text
        object.
    cluster_scope:
        Which clusters should be labeled.
    overwrite:
        If ``False``, skip clusters that already have a label.
    verbose:
        Print labeling progress.
    show_progress:
        Show tqdm progress while labeling clusters.
    """

    entity_metadata: object
    text_extractor: Callable[[SparseCluster, object], object]
    label_fn: Callable[[object], object]
    cluster_scope: Literal["active", "all", "leaves", "roots"] = "active"
    overwrite: bool = False
    verbose: bool = False
    show_progress: bool = False

    def __call__(self, clusters: SparseClusterSet) -> SparseClusterSet:
        return label_clusters(
            clusters,
            entity_metadata=self.entity_metadata,
            text_extractor=self.text_extractor,
            label_fn=self.label_fn,
            cluster_scope=self.cluster_scope,
            overwrite=self.overwrite,
            verbose=self.verbose,
            show_progress=self.show_progress,
        )


@dataclass(frozen=True)
class SizeFilter(AbstractClusterTransform):
    """Keep only active clusters with at least ``min_cluster_size`` entities.

    This changes the active frontier only; filtered-out clusters remain in the
    graph and can still be inspected through ``clusters.clusters``.

    Parameters
    ----------
    min_cluster_size:
        Minimum entity count required for an active cluster to stay active.
    """

    min_cluster_size: int

    def __call__(self, clusters: SparseClusterSet) -> SparseClusterSet:
        return filter_clusters_by_size(clusters, min_cluster_size=self.min_cluster_size)


@dataclass(frozen=True)
class PruneRedundantRoots(AbstractClusterTransform):
    """Remove active clusters that are descendants of another active cluster.

    This is usually used after link/materialization steps to make the active
    frontier easier to render. It does not delete nodes; it only updates
    ``active_cluster_ids``.

    Parameters
    ----------
    verbose:
        Print pruning summary.
    """

    verbose: bool = False

    def __call__(self, clusters: SparseClusterSet) -> SparseClusterSet:
        return prune_redundant_active_clusters(clusters, verbose=self.verbose)


@dataclass(frozen=True)
class ClusteringPipeline:
    """Composable class-based sparse clustering pipeline.

    A pipeline starts with one clustering step that consumes an
    :class:`~compresso.params.srp.SRPTensor`. Every later step consumes and
    returns a :class:`SparseClusterSet`. This separates discovery from optional
    graph linking, merging, labeling, tagging, and filtering.

    Parameters
    ----------
    steps:
        Sequence of class-based steps. The first must be an
        :class:`AbstractClustering`; later steps are usually
        :class:`AbstractClusterTransform` instances.
    verbose:
        Print one-line progress for each pipeline step.

    Examples
    --------
    >>> graph = ClusteringPipeline([
    ...     TopMSignedClustering(top_m=3, min_cluster_size=5),
    ...     EntityContainmentLink(threshold=1.0),
    ...     MaterializeLinkMerges(),
    ...     SizeFilter(min_cluster_size=10),
    ... ]).fit(srp)
    """

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
