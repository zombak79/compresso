from __future__ import annotations

import compresso
import compresso.clustering as cc


def test_first_release_top_level_api_is_intentional():
    expected = {
        "MaskedParam",
        "SRPTensor",
        "SRPParam",
        "topk_ste",
        "TopKSparsify",
        "TopKSAE",
        "L1Normalize",
        "L2Normalize",
        "TopKSAEConfig",
        "TopKSAETrainer",
        "SparsityController",
        "exponential_decay",
        "save_srp_tensor",
        "load_srp_tensor",
    }

    assert set(compresso.__all__) == expected
    for name in expected:
        assert hasattr(compresso, name)


def test_experimental_objects_are_not_top_level_exports():
    hidden = [
        "SharedMaskedParam",
        "CooSparseParam",
        "MaskedLinear",
        "MaskedEmbedding",
        "CooSparseLinear",
        "CooSparseEmbedding",
        "SRPEmbedding",
        "SparseAwareLinearKernel",
        "GatedMaskedParam",
        "GatedMaskedAttentionParam",
        "GatedMLP",
        "srpmm",
        "convert_masked_to_coo_inplace",
        "compact_gated_modules",
        "compact_all_gated_mlps_after_rewind",
        "masked_parameters",
    ]

    for name in hidden:
        assert not hasattr(compresso, name)


def test_first_release_clustering_api_is_intentional():
    expected = {
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
    }

    assert set(cc.__all__) == expected
    for name in expected:
        assert hasattr(cc, name)


def test_low_level_clustering_functions_are_not_public_namespace_exports():
    hidden = [
        "build_activation_clusters",
        "build_feature_path_clusters",
        "build_srp_similarity_clusters",
        "assign_to_clusters",
        "assign_cluster_tags",
        "compute_cluster_tag_counts",
        "tfidf_score",
        "label_clusters",
        "merge_clusters_by_semantic_similarity",
        "merge_clusters_by_entity_iou",
        "merge_clusters_by_entity_containment",
        "merge_clusters_by_feature_containment",
        "merge_clusters_by_centroid_similarity",
        "merge_clusters_by_tag_similarity",
        "merge_clusters_by_duplicate_label",
        "merge_cluster_group",
        "link_clusters_by_entity_containment",
        "link_clusters_by_feature_containment",
        "materialize_link_merges",
        "compact_hidden_clusters",
        "prune_redundant_active_clusters",
        "filter_clusters_by_size",
        "assign_unclustered_to_nearest_cluster",
    ]

    for name in hidden:
        assert not hasattr(cc, name)
