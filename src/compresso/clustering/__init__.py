"""Public clustering API.

The first-release clustering surface is intentionally centered on graph types
and class-based pipeline steps. Lower-level builder/merge functions remain in
their implementation modules for internal tests and advanced experimentation,
but the recommended user API is:

    graph = ClusteringPipeline([...]).fit(srp)
"""

from .io import graph_from_dict, graph_to_dict, load_cluster_graph, save_cluster_graph
from .pipeline import (
    AbstractClusterTransform,
    AbstractClustering,
    AbstractMerging,
    AssignTags,
    AssignUnclusteredToNearestCluster,
    CentroidSimilarityMerge,
    ClusteringPipeline,
    ComboSignedClustering,
    CompactHiddenClusters,
    DominantSignedClustering,
    EntityContainmentLink,
    EntityContainmentMerge,
    EntityIoUMerge,
    FeatureContainmentLink,
    FeatureContainmentMerge,
    FeaturePathClustering,
    LabelClusters,
    LabelDuplicateMerge,
    MaterializeLinkMerges,
    PruneRedundantRoots,
    SRPSimilarityClustering,
    SemanticSimilarityMerge,
    SizeFilter,
    TagSimilarityMerge,
    TopMSignedClustering,
    cluster_srp,
    run_clustering_pipeline,
)
from .types import ScoredTag, SparseCluster, SparseClusterGraph, SparseClusterSet, SparseVector

__all__ = [
    "SparseVector",
    "ScoredTag",
    "SparseCluster",
    "SparseClusterGraph",
    "SparseClusterSet",
    "ClusteringPipeline",
    "AbstractClusterTransform",
    "AbstractClustering",
    "AbstractMerging",
    "DominantSignedClustering",
    "TopMSignedClustering",
    "ComboSignedClustering",
    "FeaturePathClustering",
    "SRPSimilarityClustering",
    "EntityContainmentLink",
    "FeatureContainmentLink",
    "MaterializeLinkMerges",
    "EntityIoUMerge",
    "EntityContainmentMerge",
    "FeatureContainmentMerge",
    "CentroidSimilarityMerge",
    "LabelDuplicateMerge",
    "SemanticSimilarityMerge",
    "TagSimilarityMerge",
    "CompactHiddenClusters",
    "PruneRedundantRoots",
    "AssignTags",
    "LabelClusters",
    "AssignUnclusteredToNearestCluster",
    "SizeFilter",
    "save_cluster_graph",
    "load_cluster_graph",
    "graph_to_dict",
    "graph_from_dict",
    "cluster_srp",
    "run_clustering_pipeline",
]
