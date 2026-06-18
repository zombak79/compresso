from __future__ import annotations

from functools import partial

import numpy as np
import torch
from scipy.sparse import csr_matrix

from compresso.clustering import (
    assign_cluster_tags,
    build_activation_clusters,
    compact_hidden_clusters,
    cluster_srp,
    assign_unclustered_to_nearest_cluster,
    AssignUnclusteredToNearestCluster,
    CentroidSimilarityMerge,
    AssignTags,
    ClusteringPipeline,
    ComboSignedClustering,
    CompactHiddenClusters,
    EntityContainmentMerge,
    EntityContainmentLink,
    FeaturePathClustering,
    filter_clusters_by_size,
    label_clusters,
    LabelClusters,
    LabelDuplicateMerge,
    link_clusters_by_entity_containment,
    materialize_link_merges,
    MaterializeLinkMerges,
    merge_clusters_by_centroid_similarity,
    merge_clusters_by_duplicate_label,
    merge_clusters_by_entity_containment,
    merge_clusters_by_entity_iou,
    merge_clusters_by_feature_containment,
    merge_clusters_by_semantic_similarity,
    merge_clusters_by_tag_similarity,
    prune_redundant_active_clusters,
    PruneRedundantRoots,
    run_clustering_pipeline,
    SemanticSimilarityMerge,
    SizeFilter,
    SRPSimilarityClustering,
    TopMSignedClustering,
    SparseCluster,
    SparseClusterSet,
    SparseVector,
)
from compresso.params.srp import SRPTensor


def _srp(cols, vals, n_features=4):
    return SRPTensor(
        cols=torch.tensor(cols, dtype=torch.long),
        vals=torch.tensor(vals, dtype=torch.float32),
        shape=(len(cols), n_features),
    )


def _cluster(cluster_id, entities, features, signs, n_features=10):
    return SparseCluster(
        cluster_id=cluster_id,
        centroid=SparseVector(
            np.asarray(features, dtype=np.int64),
            np.asarray(signs, dtype=np.float32),
            n_features,
        ).normalized(),
        entity_indices=np.asarray(entities, dtype=np.int64),
        source_cluster_ids=(cluster_id,),
    )


def test_assign_cluster_tags_tfidf():
    srp = _srp([[0], [0], [1], [1]], [[1.0], [2.0], [1.0], [2.0]], n_features=2)
    clusters = build_activation_clusters(srp)
    entity_tag = csr_matrix(
        np.array(
            [
                [1, 0],
                [1, 0],
                [0, 1],
                [0, 1],
            ],
            dtype=np.float32,
        )
    )

    tagged = assign_cluster_tags(clusters, entity_tag, ["alpha", "beta"], top_k=1)
    by_id = tagged.cluster_by_id
    assert by_id["feature:0:pos"].tags[0].name == "alpha"
    assert by_id["feature:1:pos"].tags[0].name == "beta"
    assert by_id["feature:0:pos"].tags[0].count == 2.0


def test_merge_clusters_by_entity_iou_for_top_m_clusters():
    srp = _srp([[0, 1], [0, 1], [2, 3]], [[3.0, 2.0], [4.0, 2.5], [1.0, 0.5]], n_features=4)
    clusters = build_activation_clusters(srp, mode="top_m_signed", top_m=2)

    merged = merge_clusters_by_entity_iou(clusters, threshold=1.0)
    assert len(merged.active_clusters) == 2
    assert len(merged.clusters) == 6
    merged_cluster = next(c for c in merged.active_clusters if c.entity_indices.tolist() == [0, 1])
    assert set(merged_cluster.centroid.indices.tolist()) == {0, 1}
    singleton_cluster = next(c for c in merged.active_clusters if c.entity_indices.tolist() == [2])
    assert set(singleton_cluster.centroid.indices.tolist()) == {2, 3}


def test_merge_clusters_by_entity_iou_verbose_reports_rounds(capsys):
    srp = _srp([[0, 1], [0, 1]], [[3.0, 2.0], [4.0, 2.5]], n_features=2)
    clusters = build_activation_clusters(srp, mode="top_m_signed", top_m=2)

    merged = merge_clusters_by_entity_iou(clusters, threshold=1.0, verbose=True)

    out = capsys.readouterr().out
    assert "[merge_clusters_by_entity_iou] round=1/10 active_clusters=2 threshold=1.000000" in out
    assert "[merge_clusters_by_entity_iou] merged: 2 -> 1" in out
    assert len(merged.active_clusters) == 1
    assert len(merged.clusters) == 3


def test_merge_clusters_by_entity_containment_merges_subset_even_with_low_iou():
    clusters = SparseClusterSet(
        clusters=(
            _cluster("broad", range(49), [370, 960, 1580, 3982], [1, 1, 1, -1], n_features=5000),
            _cluster("specific", [0, 1, 2, 3, 4], [370, 717, 1580, 3982], [1, 1, 1, -1], n_features=5000),
            _cluster("other", [100, 101, 102], [12], [1], n_features=5000),
        ),
        n_entities=200,
        n_features=5000,
    )

    merged = merge_clusters_by_entity_containment(clusters, threshold=1.0)

    assert len(merged.active_clusters) == 2
    assert len(merged.clusters) == 4
    merged_cluster = next(c for c in merged.active_clusters if c.entity_count == 49)
    assert set(merged_cluster.source_cluster_ids) == {"broad", "specific"}
    assert set(merged_cluster.centroid.indices.tolist()) == {370, 717, 960, 1580, 3982}
    assert tuple(c.cluster_id for c in merged.children(merged_cluster.cluster_id)) == ("broad", "specific")
    assert merged.parents("specific")[0].cluster_id == merged_cluster.cluster_id
    assert {c.cluster_id for c in merged.leaf_clusters} == {"broad", "specific", "other"}
    assert {c.cluster_id for c in merged.root_clusters} == {merged_cluster.cluster_id, "other"}


def test_merge_clusters_by_feature_containment_uses_signed_feature_support():
    clusters = SparseClusterSet(
        clusters=(
            _cluster("broad", [0, 1, 2], [10, 22, 91], [1, -1, 1], n_features=100),
            _cluster("specific", [3, 4], [10, 22], [1, -1], n_features=100),
            _cluster("opposite", [5, 6], [10, 22], [1, 1], n_features=100),
        ),
        n_entities=10,
        n_features=100,
    )

    merged = merge_clusters_by_feature_containment(clusters, threshold=1.0, signed=True)

    assert len(merged.active_clusters) == 2
    assert len(merged.clusters) == 4
    merged_cluster = next(c for c in merged.active_clusters if set(c.source_cluster_ids) == {"broad", "specific"})
    assert merged_cluster.entity_indices.tolist() == [0, 1, 2, 3, 4]
    assert any(c.cluster_id == "opposite" for c in merged.active_clusters)


def test_merge_clusters_by_centroid_similarity_merges_similar_active_centroids():
    clusters = SparseClusterSet(
        clusters=(
            _cluster("a", [0, 1], [0, 1], [1.0, 0.0], n_features=4),
            _cluster("b", [2, 3], [0, 1], [0.95, 0.05], n_features=4),
            _cluster("c", [4, 5], [2], [1.0], n_features=4),
        ),
        n_entities=6,
        n_features=4,
    )

    merged = merge_clusters_by_centroid_similarity(clusters, threshold=0.99)

    assert len(merged.active_clusters) == 2
    assert len(merged.clusters) == 4
    merged_cluster = next(c for c in merged.active_clusters if set(c.source_cluster_ids) == {"a", "b"})
    assert merged_cluster.entity_indices.tolist() == [0, 1, 2, 3]
    assert merged_cluster.child_cluster_ids == ("a", "b")
    assert merged.parents("a")[0].cluster_id == merged_cluster.cluster_id
    assert any(c.cluster_id == "c" for c in merged.active_clusters)


def test_merge_clusters_by_centroid_similarity_top_k_limits_edges():
    clusters = SparseClusterSet(
        clusters=(
            _cluster("a", [0], [0, 1], [1.0, 0.0], n_features=4),
            _cluster("b", [1], [0, 1], [0.99, 0.01], n_features=4),
            _cluster("c", [2], [0, 1], [0.8, 0.6], n_features=4),
            _cluster("d", [3], [0, 1], [0.79, 0.61], n_features=4),
        ),
        n_entities=4,
        n_features=4,
    )

    merged = merge_clusters_by_centroid_similarity(clusters, threshold=0.75, top_k=1, max_rounds=1)

    assert len(merged.active_clusters) == 2
    assert any(set(c.source_cluster_ids) == {"a", "b"} for c in merged.active_clusters)
    assert any(set(c.source_cluster_ids) == {"c", "d"} for c in merged.active_clusters)


def test_centroid_similarity_merge_pipeline_wrapper():
    clusters = SparseClusterSet(
        clusters=(
            _cluster("a", [0], [0], [1.0], n_features=4),
            _cluster("b", [1], [0], [1.0], n_features=4),
            _cluster("c", [2], [3], [1.0], n_features=4),
        ),
        n_entities=3,
        n_features=4,
    )

    merged = CentroidSimilarityMerge(threshold=1.0)(clusters)

    assert len(merged.active_clusters) == 2
    assert any(set(c.source_cluster_ids) == {"a", "b"} for c in merged.active_clusters)


def test_assign_unclustered_to_nearest_cluster_expands_active_cluster():
    srp = _srp(
        [[0], [0], [1], [0]],
        [[1.0], [1.0], [1.0], [0.5]],
        n_features=2,
    )
    clusters = SparseClusterSet(
        clusters=(
            _cluster("a", [0, 1], [0], [1.0], n_features=2),
            _cluster("b", [2], [1], [1.0], n_features=2),
        ),
        n_entities=4,
        n_features=2,
    )

    expanded = assign_unclustered_to_nearest_cluster(clusters, srp)

    assert len(expanded.active_clusters) == 2
    parent = next(c for c in expanded.active_clusters if c.child_cluster_ids == ("a",))
    assert parent.entity_indices.tolist() == [0, 1, 3]
    assert parent.label is None
    assert parent.metadata["base_cluster_id"] == "a"
    assert parent.metadata["assigned_entity_indices"] == (3,)
    assert parent.stats["core_entity_count"] == 2
    assert parent.stats["assigned_entity_count"] == 1
    assert expanded.parents("a")[0].cluster_id == parent.cluster_id
    assert any(c.cluster_id == "b" for c in expanded.active_clusters)
    assert expanded.history[-1]["n_unclustered_before"] == 1
    assert expanded.history[-1]["n_assigned"] == 1
    assert expanded.history[-1]["n_unassigned_after"] == 0


def test_assign_unclustered_to_nearest_cluster_uses_centroid_top_m():
    srp = _srp(
        [[0, 1], [0, 2], [1, 2]],
        [[1.0, 0.5], [1.0, 0.4], [1.0, 0.9]],
        n_features=3,
    )
    clusters = SparseClusterSet(
        clusters=(_cluster("a", [0, 1], [0], [1.0], n_features=3),),
        n_entities=3,
        n_features=3,
    )

    expanded = assign_unclustered_to_nearest_cluster(clusters, srp, centroid_top_m=2)

    parent = expanded.active_clusters[0]
    assert parent.child_cluster_ids == ("a",)
    assert len(parent.centroid.indices) == 2
    assert parent.metadata["centroid_top_m"] == 2
    assert parent.metadata["centroid_top_m_effective"] == 2


def test_assign_unclustered_to_nearest_cluster_defaults_to_base_centroid_width():
    srp = _srp(
        [[0, 1], [0, 2], [1, 2]],
        [[1.0, 0.5], [1.0, 0.4], [1.0, 0.9]],
        n_features=3,
    )
    clusters = SparseClusterSet(
        clusters=(_cluster("a", [0, 1], [0], [1.0], n_features=3),),
        n_entities=3,
        n_features=3,
    )

    expanded = assign_unclustered_to_nearest_cluster(clusters, srp)

    parent = expanded.active_clusters[0]
    assert len(parent.centroid.indices) == 1
    assert parent.metadata["centroid_top_m"] is None
    assert parent.metadata["centroid_top_m_effective"] == 1


def test_assign_unclustered_to_nearest_cluster_all_coverage_ignores_inactive_covered_entities():
    srp = _srp(
        [[0], [0], [1], [0]],
        [[1.0], [1.0], [1.0], [1.0]],
        n_features=2,
    )
    clusters = SparseClusterSet(
        clusters=(
            _cluster("a", [0, 1], [0], [1.0], n_features=2),
            _cluster("inactive_cover", [2], [1], [1.0], n_features=2),
        ),
        n_entities=4,
        n_features=2,
        active_cluster_ids=("a",),
    )

    expanded = assign_unclustered_to_nearest_cluster(clusters, srp, coverage_scope="all")

    parent = expanded.active_clusters[0]
    assert parent.child_cluster_ids == ("a",)
    assert parent.metadata["assigned_entity_indices"] == (3,)
    assert expanded.history[-1]["n_unclustered_before"] == 1


def test_assign_unclustered_to_nearest_cluster_min_similarity_can_leave_unassigned():
    srp = _srp(
        [[0], [1]],
        [[1.0], [1.0]],
        n_features=2,
    )
    clusters = SparseClusterSet(
        clusters=(_cluster("a", [0], [0], [1.0], n_features=2),),
        n_entities=2,
        n_features=2,
    )

    unchanged = assign_unclustered_to_nearest_cluster(clusters, srp, min_similarity=0.5)

    assert len(unchanged.clusters) == 1
    assert unchanged.history[-1]["changed"] is False
    assert unchanged.history[-1]["n_unclustered_before"] == 1
    assert unchanged.history[-1]["n_assigned"] == 0
    assert unchanged.history[-1]["n_unassigned_after"] == 1


def test_assign_unclustered_to_nearest_cluster_pipeline_wrapper():
    srp = _srp(
        [[0], [0], [0]],
        [[1.0], [1.0], [1.0]],
        n_features=2,
    )
    clusters = SparseClusterSet(
        clusters=(_cluster("a", [0, 1], [0], [1.0], n_features=2),),
        n_entities=3,
        n_features=2,
    )

    expanded = AssignUnclusteredToNearestCluster(srp)(clusters)

    assert len(expanded.active_clusters) == 1
    assert expanded.active_clusters[0].entity_indices.tolist() == [0, 1, 2]
    assert expanded.active_clusters[0].child_cluster_ids == ("a",)


def test_entity_containment_link_allows_multiple_parents_without_consuming_child():
    clusters = SparseClusterSet(
        clusters=(
            _cluster("small", [0, 1], [10], [1], n_features=100),
            _cluster("parent-a", [0, 1, 2], [10, 20], [1, 1], n_features=100),
            _cluster("parent-b", [0, 1, 3], [10, 30], [1, 1], n_features=100),
        ),
        n_entities=10,
        n_features=100,
    )

    linked = link_clusters_by_entity_containment(
        clusters,
        threshold=1.0,
        child_scope="leaves",
        parent_scope="all",
    )

    assert len(linked.clusters) == 3
    assert {c.cluster_id for c in linked.active_clusters} == {"small", "parent-a", "parent-b"}
    assert set(linked.cluster_by_id["small"].parent_cluster_ids) == {"parent-a", "parent-b"}
    assert "small" in linked.cluster_by_id["parent-a"].child_cluster_ids
    assert "small" in linked.cluster_by_id["parent-b"].child_cluster_ids


def test_entity_containment_link_skips_existing_ancestor_and_cycles():
    clusters = SparseClusterSet(
        clusters=(
            _cluster("small", [0, 1], [10], [1], n_features=100),
            _cluster("mid", [0, 1, 2], [10, 20], [1, 1], n_features=100),
            _cluster("big", [0, 1, 2, 3], [10, 20, 30], [1, 1, 1], n_features=100),
        ),
        n_entities=10,
        n_features=100,
    )
    linked = link_clusters_by_entity_containment(clusters, threshold=1.0)
    relinked = link_clusters_by_entity_containment(linked, threshold=1.0)

    assert set(relinked.cluster_by_id["small"].parent_cluster_ids) == {"mid", "big"}
    assert set(relinked.cluster_by_id["mid"].parent_cluster_ids) == {"big"}
    assert "big" not in relinked.cluster_by_id["small"].child_cluster_ids


def test_prune_redundant_active_clusters_removes_descendant_frontier_nodes():
    clusters = SparseClusterSet(
        clusters=(
            _cluster("small", [0, 1], [10], [1], n_features=100),
            _cluster("big", [0, 1, 2], [10, 20], [1, 1], n_features=100),
        ),
        n_entities=10,
        n_features=100,
    )
    linked = link_clusters_by_entity_containment(clusters, threshold=1.0)
    pruned = prune_redundant_active_clusters(linked)

    assert {c.cluster_id for c in linked.active_clusters} == {"small", "big"}
    assert {c.cluster_id for c in pruned.active_clusters} == {"big"}
    assert "small" in pruned.cluster_by_id


def test_materialize_link_merges_creates_visible_parent_nodes_non_destructively():
    clusters = SparseClusterSet(
        clusters=(
            _cluster("small", [0, 1], [10], [1], n_features=100),
            _cluster("big", [0, 1, 2], [10, 20], [1, 1], n_features=100),
        ),
        n_entities=10,
        n_features=100,
    )
    linked = link_clusters_by_entity_containment(clusters, threshold=1.0)

    materialized = materialize_link_merges(linked, parent_scope="active")

    assert len(materialized.clusters) == 3
    merge_nodes = [c for c in materialized.clusters if c.metadata.get("materialized_from_links")]
    assert len(merge_nodes) == 1
    parent = merge_nodes[0]
    assert parent.cluster_id.startswith("merge:materialize_link_merges:")
    assert set(parent.child_cluster_ids) == {"big", "small"}
    assert parent.entity_indices.tolist() == [0, 1, 2]
    assert set(materialized.cluster_by_id["small"].parent_cluster_ids) == {"big", parent.cluster_id}
    assert set(materialized.cluster_by_id["big"].parent_cluster_ids) == {parent.cluster_id}
    assert {c.cluster_id for c in materialized.active_clusters} == {"small", parent.cluster_id}


def test_merge_clusters_by_duplicate_label_marks_children_hidden():
    clusters = SparseClusterSet(
        clusters=(
            _cluster("a", [0, 1], [10], [1], n_features=100).with_updates(label="Vampire Romance"),
            _cluster("b", [2, 3], [20], [1], n_features=100).with_updates(label=" vampire   romance "),
            _cluster("c", [4], [30], [1], n_features=100).with_updates(label="Space Opera"),
        ),
        n_entities=5,
        n_features=100,
    )

    merged = merge_clusters_by_duplicate_label(clusters)

    duplicate_parents = [c for c in merged.clusters if c.metadata.get("merge_strategy") == "duplicate_label"]
    assert len(duplicate_parents) == 1
    parent = duplicate_parents[0]
    assert parent.label == "Vampire Romance"
    assert parent.entity_indices.tolist() == [0, 1, 2, 3]
    assert set(parent.child_cluster_ids) == {"a", "b"}
    assert merged.cluster_by_id["a"].metadata["render_hidden"] is True
    assert merged.cluster_by_id["b"].metadata["render_hidden_parent_id"] == parent.cluster_id
    assert {c.cluster_id for c in merged.active_clusters} == {parent.cluster_id, "c"}


def test_compact_hidden_clusters_removes_hidden_nodes_and_rewires_children():
    hidden_a = _cluster("a", [0, 1], [10], [1], n_features=100).with_updates(
        label="Vampire Romance",
        metadata={"render_hidden": True},
        parent_cluster_ids=("m",),
        child_cluster_ids=("a1",),
    )
    hidden_b = _cluster("b", [2, 3], [20], [1], n_features=100).with_updates(
        label="Vampire Romance",
        metadata={"render_hidden": True},
        parent_cluster_ids=("m",),
        child_cluster_ids=("b1",),
    )
    parent = _cluster("m", [0, 1, 2, 3], [10, 20], [1, 1], n_features=100).with_updates(
        label="Vampire Romance",
        child_cluster_ids=("a", "b"),
    )
    a1 = _cluster("a1", [0], [11], [1], n_features=100).with_updates(
        label="Teen Vampire Romance",
        parent_cluster_ids=("a",),
    )
    b1 = _cluster("b1", [2], [21], [1], n_features=100).with_updates(
        label="Dark Vampire Romance",
        parent_cluster_ids=("b",),
    )
    graph = SparseClusterSet(
        clusters=(parent, hidden_a, hidden_b, a1, b1),
        n_entities=4,
        n_features=100,
        active_cluster_ids=("m",),
    )

    compacted = compact_hidden_clusters(graph)

    assert set(compacted.cluster_by_id) == {"m", "a1", "b1"}
    assert set(compacted.cluster_by_id["m"].child_cluster_ids) == {"a1", "b1"}
    assert compacted.cluster_by_id["a1"].parent_cluster_ids == ("m",)
    assert compacted.cluster_by_id["b1"].parent_cluster_ids == ("m",)
    assert compacted.active_cluster_ids == ("m",)


def test_label_duplicate_merge_and_compact_hidden_clusters_pipeline_classes():
    clusters = SparseClusterSet(
        clusters=(
            _cluster("a", [0, 1], [10], [1], n_features=100).with_updates(label="Vampire Romance"),
            _cluster("b", [2, 3], [20], [1], n_features=100).with_updates(label="Vampire Romance"),
        ),
        n_entities=4,
        n_features=100,
    )

    graph = LabelDuplicateMerge()(clusters)
    graph = CompactHiddenClusters()(graph)

    assert len(graph.clusters) == 1
    assert graph.active_clusters[0].label == "Vampire Romance"
    assert graph.active_clusters[0].entity_indices.tolist() == [0, 1, 2, 3]


def test_label_clusters_stores_user_function_string_result():
    clusters = SparseClusterSet(
        clusters=(
            _cluster("books", [0, 1], [10], [1], n_features=100),
            _cluster("other", [2], [20], [1], n_features=100),
        ),
        n_entities=3,
        n_features=100,
        active_cluster_ids=("books",),
    )
    metadata = np.asarray(["Harry Potter 1", "Harry Potter 2", "Other book"], dtype=object)

    labeled = label_clusters(
        clusters,
        entity_metadata=metadata,
        text_extractor=lambda cluster, meta: "\n-\n".join(str(meta[i]) for i in cluster.entity_indices),
        label_fn=lambda text: "Harry Potter Series" if "Harry Potter" in text else "Other",
    )

    assert labeled.cluster_by_id["books"].label == "Harry Potter Series"
    assert labeled.cluster_by_id["books"].description is None
    assert labeled.cluster_by_id["books"].metadata["labeling"]["text_type"] == "str"
    assert labeled.cluster_by_id["other"].label is None
    assert labeled.history[-1]["n_labeled"] == 1


def test_label_clusters_accepts_dict_result_and_overwrite_flag():
    clusters = SparseClusterSet(
        clusters=(_cluster("books", [0, 1], [10], [1], n_features=100),),
        n_entities=2,
        n_features=100,
    )
    metadata = np.asarray(["A", "B"], dtype=object)

    labeled = label_clusters(
        clusters,
        entity_metadata=metadata,
        text_extractor=lambda cluster, meta: [str(meta[i]) for i in cluster.entity_indices],
        label_fn=lambda text: {"label": "First", "description": "First description", "model": "test"},
    )
    skipped = label_clusters(
        labeled,
        entity_metadata=metadata,
        text_extractor=lambda cluster, meta: [str(meta[i]) for i in cluster.entity_indices],
        label_fn=lambda text: {"label": "Second", "description": "Second description"},
    )
    overwritten = label_clusters(
        skipped,
        entity_metadata=metadata,
        text_extractor=lambda cluster, meta: [str(meta[i]) for i in cluster.entity_indices],
        label_fn=lambda text: {"label": "Second", "description": "Second description"},
        overwrite=True,
    )

    assert labeled.cluster_by_id["books"].label == "First"
    assert labeled.cluster_by_id["books"].description == "First description"
    assert labeled.cluster_by_id["books"].metadata["labeling"]["result_metadata"] == {"model": "test"}
    assert skipped.cluster_by_id["books"].label == "First"
    assert skipped.history[-1]["n_skipped"] == 1
    assert overwritten.cluster_by_id["books"].label == "Second"
    assert overwritten.cluster_by_id["books"].description == "Second description"


def test_semantic_similarity_merge_creates_overlapping_clique_parents():
    clusters = SparseClusterSet(
        clusters=(
            _cluster("a", [0, 1], [10], [1], n_features=100).with_updates(label="A"),
            _cluster("b", [2, 3], [20], [1], n_features=100).with_updates(label="B"),
            _cluster("c", [4, 5], [30], [1], n_features=100).with_updates(label="C"),
        ),
        n_entities=6,
        n_features=100,
    )
    gram = np.array(
        [
            [1.0, 0.91, 0.85],
            [0.91, 1.0, 0.92],
            [0.85, 0.92, 1.0],
        ],
        dtype=np.float32,
    )
    vectors = np.linalg.cholesky(gram).astype(np.float32)
    vector_by_text = {"A": vectors[0], "B": vectors[1], "C": vectors[2]}

    graph = merge_clusters_by_semantic_similarity(
        clusters,
        embed_fn=lambda texts: np.stack([vector_by_text[text] for text in texts]),
        threshold=0.9,
        max_rounds=1,
        label_fn=lambda text: text,
        label_text_fn=lambda parent, children: "+".join(child.label for child in children if child.label),
    )

    semantic_parents = [c for c in graph.clusters if c.metadata.get("merge_strategy") == "semantic_similarity"]
    assert len(semantic_parents) == 2
    assert {tuple(c.child_cluster_ids) for c in semantic_parents} == {("a", "b"), ("b", "c")}
    assert len(graph.cluster_by_id["b"].parent_cluster_ids) == 2
    assert {c.label for c in semantic_parents} == {"A+B", "B+C"}
    assert len(graph.active_clusters) == 2


def test_semantic_similarity_merge_can_iteratively_merge_new_labeled_parents():
    clusters = SparseClusterSet(
        clusters=(
            _cluster("a", [0, 1], [10], [1], n_features=100).with_updates(label="A"),
            _cluster("b", [2, 3], [20], [1], n_features=100).with_updates(label="B"),
            _cluster("c", [4, 5], [30], [1], n_features=100).with_updates(label="C"),
        ),
        n_entities=6,
        n_features=100,
    )
    child_gram = np.array(
        [
            [1.0, 0.91, 0.85],
            [0.91, 1.0, 0.92],
            [0.85, 0.92, 1.0],
        ],
        dtype=np.float32,
    )
    child_vectors = np.linalg.cholesky(child_gram).astype(np.float32)
    vector_by_text = {
        "A": child_vectors[0],
        "B": child_vectors[1],
        "C": child_vectors[2],
        "A+B": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "B+C": np.array([0.95, (1.0 - 0.95**2) ** 0.5, 0.0], dtype=np.float32),
    }

    graph = merge_clusters_by_semantic_similarity(
        clusters,
        embed_fn=lambda texts: np.stack([vector_by_text[text] for text in texts]),
        threshold=0.9,
        max_rounds=2,
        label_fn=lambda text: text,
        label_text_fn=lambda parent, children: "+".join(child.label for child in children if child.label),
    )

    semantic_parents = [c for c in graph.clusters if c.metadata.get("merge_strategy") == "semantic_similarity"]
    assert len(semantic_parents) == 3
    assert len(graph.active_clusters) == 1
    top = graph.active_clusters[0]
    assert top.label == "A+B+B+C"
    assert {graph.cluster_by_id[child_id].label for child_id in top.child_cluster_ids} == {"A+B", "B+C"}


def test_clustering_pipeline_semantic_similarity_merge_class():
    clusters = SparseClusterSet(
        clusters=(
            _cluster("a", [0], [10], [1], n_features=100).with_updates(label="A"),
            _cluster("b", [1], [20], [1], n_features=100).with_updates(label="B"),
        ),
        n_entities=2,
        n_features=100,
    )
    vector_by_text = {
        "A": np.array([1.0, 0.0], dtype=np.float32),
        "B": np.array([0.95, (1.0 - 0.95**2) ** 0.5], dtype=np.float32),
    }

    graph = SemanticSimilarityMerge(
        embed_fn=lambda texts: np.stack([vector_by_text[text] for text in texts]),
        threshold=0.9,
        max_rounds=1,
    )(clusters)

    assert len(graph.active_clusters) == 1
    assert graph.active_clusters[0].child_cluster_ids == ("a", "b")


def test_merge_clusters_by_tag_similarity():
    srp = _srp([[0], [1], [2]], [[1.0], [1.0], [1.0]], n_features=3)
    clusters = build_activation_clusters(srp)
    entity_tag = csr_matrix(np.array([[1, 0], [1, 0], [0, 1]], dtype=np.float32))
    tagged = assign_cluster_tags(clusters, entity_tag, ["same", "other"], top_k=1)

    merged = merge_clusters_by_tag_similarity(tagged, threshold=1.0)
    assert len(merged.active_clusters) == 2
    assert len(merged.clusters) == 4
    assert any(c.entity_indices.tolist() == [0, 1] for c in merged.active_clusters)


def test_cluster_srp_pipeline_with_tag_merge_reassigns_tags():
    srp = _srp([[0], [1], [2]], [[1.0], [1.0], [1.0]], n_features=3)
    entity_tag = csr_matrix(np.array([[1, 0], [1, 0], [0, 1]], dtype=np.float32))

    clusters = cluster_srp(
        srp,
        entity_tag_matrix=entity_tag,
        tag_names=["same", "other"],
        tag_similarity_threshold=1.0,
        top_k_tags=1,
    )

    assert len(clusters.active_clusters) == 2
    assert len(clusters.clusters) == 4
    merged = next(c for c in clusters.active_clusters if c.entity_indices.tolist() == [0, 1])
    assert merged.tags[0].name == "same"


def test_post_merge_min_cluster_size_filters_after_merging():
    srp = _srp([[0], [1], [2]], [[1.0], [1.0], [1.0]], n_features=3)
    entity_tag = csr_matrix(np.array([[1, 0], [1, 0], [0, 1]], dtype=np.float32))

    clusters = cluster_srp(
        srp,
        min_cluster_size=1,
        entity_tag_matrix=entity_tag,
        tag_names=["same", "other"],
        tag_similarity_threshold=1.0,
        post_merge_min_cluster_size=2,
    )

    assert len(clusters.active_clusters) == 1
    assert len(clusters.clusters) == 4
    assert clusters.active_clusters[0].entity_indices.tolist() == [0, 1]


def test_filter_clusters_by_size_direct():
    srp = _srp([[0], [1]], [[1.0], [1.0]], n_features=2)
    clusters = build_activation_clusters(srp, min_cluster_size=1)
    filtered = filter_clusters_by_size(clusters, min_cluster_size=2)
    assert len(filtered.active_clusters) == 0
    assert len(filtered.clusters) == 2


def test_run_clustering_pipeline_composes_build_and_transform_steps(capsys):
    srp = _srp(
        [[0, 1], [0, 1], [0, 2]],
        [[2.0, -1.0], [3.0, -2.0], [4.0, 1.0]],
        n_features=4,
    )
    entity_tag = csr_matrix(np.array([[1, 0], [1, 0], [0, 1]], dtype=np.float32))

    clusters = run_clustering_pipeline(
        srp,
        [
            partial(
                build_activation_clusters,
                mode="combo_signed",
                top_m=2,
                combo_size=2,
                min_cluster_size=1,
            ),
            lambda c: assign_cluster_tags(c, entity_tag, ["same", "other"], top_k=1),
            lambda c: filter_clusters_by_size(c, min_cluster_size=2),
        ],
        verbose=True,
    )

    out = capsys.readouterr().out
    assert "[cluster_pipeline] step 1/3: build_activation_clusters" in out
    assert "active_clusters_in=" in out
    assert len(clusters.active_clusters) == 1
    assert len(clusters.clusters) == 2
    cluster = clusters.active_clusters[0]
    assert cluster.cluster_id == "combo:0:pos&1:neg"
    assert cluster.entity_indices.tolist() == [0, 1]
    assert cluster.tags[0].name == "same"


def test_clustering_pipeline_classes_produce_cluster_graph(capsys):
    srp = _srp(
        [[0, 1], [0, 1], [2, 3]],
        [[2.0, -1.0], [3.0, -2.0], [4.0, 1.0]],
        n_features=4,
    )
    entity_tag = csr_matrix(np.array([[1, 0], [1, 0], [0, 1]], dtype=np.float32))

    graph = ClusteringPipeline(
        [
            TopMSignedClustering(top_m=2, min_cluster_size=1),
            EntityContainmentMerge(threshold=1.0),
            AssignTags(entity_tag, ["same", "other"], top_k=1),
            SizeFilter(min_cluster_size=2),
        ],
        verbose=True,
    ).fit(srp)

    out = capsys.readouterr().out
    assert "[cluster_pipeline] step 1/4: TopMSignedClustering" in out
    assert "[cluster_pipeline] step 2/4: EntityContainmentMerge" in out
    assert len(graph.active_clusters) == 1
    parent = graph.active_clusters[0]
    assert parent.tags[0].name == "same"
    assert graph.children(parent.cluster_id)


def test_clustering_pipeline_supports_feature_path_clustering(capsys):
    srp = _srp(
        [[10, 22], [10, 22], [33, 22]],
        [[3.0, -2.0], [4.0, -1.0], [5.0, -4.0]],
        n_features=100,
    )

    graph = ClusteringPipeline(
        [
            FeaturePathClustering(max_depth=2, min_cluster_size=1),
            SizeFilter(min_cluster_size=2),
        ],
        verbose=True,
    ).fit(srp)

    out = capsys.readouterr().out
    assert "[cluster_pipeline] step 1/2: FeaturePathClustering" in out
    assert {c.cluster_id for c in graph.active_clusters} == {"path:10:pos/22:neg"}
    assert graph.parents("path:10:pos/22:neg")[0].cluster_id == "path:10:pos"


def test_clustering_pipeline_supports_srp_similarity_clustering(capsys):
    srp = _srp(
        [[0], [0], [1]],
        [[1.0], [0.9], [1.0]],
        n_features=2,
    )

    graph = ClusteringPipeline(
        [
            SRPSimilarityClustering(threshold=0.9, top_k=None, min_cluster_size=2),
        ],
        verbose=True,
    ).fit(srp)

    out = capsys.readouterr().out
    assert "[cluster_pipeline] step 1/1: SRPSimilarityClustering" in out
    assert [c.entity_indices.tolist() for c in graph.active_clusters] == [[0, 1]]


def test_clustering_pipeline_link_classes_and_prune_roots():
    clusters = SparseClusterSet(
        clusters=(
            _cluster("small", [0, 1], [10], [1], n_features=100),
            _cluster("big", [0, 1, 2], [10, 20], [1, 1], n_features=100),
        ),
        n_entities=10,
        n_features=100,
    )

    graph = EntityContainmentLink(threshold=1.0)(clusters)
    graph = PruneRedundantRoots()(graph)

    assert {c.cluster_id for c in graph.active_clusters} == {"big"}
    assert graph.parents("small")[0].cluster_id == "big"


def test_clustering_pipeline_materializes_link_merges_then_prunes_roots():
    clusters = SparseClusterSet(
        clusters=(
            _cluster("small", [0, 1], [10], [1], n_features=100),
            _cluster("big", [0, 1, 2], [10, 20], [1, 1], n_features=100),
        ),
        n_entities=10,
        n_features=100,
    )

    graph = EntityContainmentLink(threshold=1.0)(clusters)
    graph = MaterializeLinkMerges(parent_scope="active")(graph)
    graph = PruneRedundantRoots()(graph)

    assert len(graph.clusters) == 3
    active_ids = {c.cluster_id for c in graph.active_clusters}
    assert len(active_ids) == 1
    active_id = next(iter(active_ids))
    assert active_id.startswith("merge:materialize_link_merges:")
    assert graph.cluster_by_id[active_id].entity_indices.tolist() == [0, 1, 2]


def test_clustering_pipeline_label_clusters_class():
    clusters = SparseClusterSet(
        clusters=(_cluster("books", [0, 1], [10], [1], n_features=100),),
        n_entities=2,
        n_features=100,
    )
    metadata = np.asarray(["Equal Rites", "Wyrd Sisters"], dtype=object)

    graph = LabelClusters(
        entity_metadata=metadata,
        text_extractor=lambda cluster, meta: "\n-\n".join(str(meta[i]) for i in cluster.entity_indices),
        label_fn=lambda text: "Discworld Witches",
    )(clusters)

    assert graph.cluster_by_id["books"].label == "Discworld Witches"


def test_graph_fill_missing_cluster_labels_counts_active_root_descendants():
    root = _cluster("root", [0, 1, 2], [1], [1], n_features=10).with_updates(
        label="Root",
        child_cluster_ids=("child-a", "child-b"),
    )
    child_a = _cluster("child-a", [0, 1], [2], [1], n_features=10).with_updates(
        parent_cluster_ids=("root",),
        child_cluster_ids=("leaf",),
    )
    child_b = _cluster("child-b", [2], [3], [1], n_features=10).with_updates(
        label="Child B",
        parent_cluster_ids=("root",),
    )
    leaf = _cluster("leaf", [0], [4], [1], n_features=10).with_updates(parent_cluster_ids=("child-a",))
    inactive_root = _cluster("inactive-root", [3], [5], [1], n_features=10)
    graph = SparseClusterSet(
        clusters=(root, child_a, child_b, leaf, inactive_root),
        n_entities=4,
        n_features=10,
        active_cluster_ids=("root",),
    )

    same_graph, n_changes, n_missing = graph.fill_missing_cluster_labels()
    _, _, n_missing_all_roots = graph.fill_missing_cluster_labels(active_roots_only=False)

    assert same_graph is graph
    assert n_changes == 0
    assert n_missing == 2
    assert n_missing_all_roots == 3


def test_graph_fill_missing_cluster_labels_returns_updated_graph():
    root = _cluster("root", [0, 1], [1], [1], n_features=10).with_updates(
        label="Root",
        child_cluster_ids=("child",),
    )
    child = _cluster("child", [0], [2], [1], n_features=10).with_updates(parent_cluster_ids=("root",))
    graph = SparseClusterSet(
        clusters=(root, child),
        n_entities=2,
        n_features=10,
        active_cluster_ids=("root",),
    )

    updated, n_changes, n_missing = graph.fill_missing_cluster_labels(
        text_fn=lambda cluster: cluster.cluster_id,
        label_fn=lambda text: {"label": f"Label {text}", "description": f"Description {text}", "model": "test"},
    )

    assert n_missing == 1
    assert n_changes == 1
    assert graph.cluster_by_id["child"].label is None
    assert updated.cluster_by_id["child"].label == "Label child"
    assert updated.cluster_by_id["child"].description == "Description child"
    assert updated.cluster_by_id["child"].metadata["label_fill"]["result_metadata"] == {"model": "test"}
    assert updated.history[-1]["phase"] == "fill_missing_cluster_labels"


def test_cluster_srp_verbose_reports_phases(capsys):
    srp = _srp([[0], [0], [1]], [[1.0], [2.0], [1.0]], n_features=2)

    clusters = cluster_srp(srp, min_cluster_size=1, post_merge_min_cluster_size=2, verbose=True)

    out = capsys.readouterr().out
    assert "[cluster_srp] build_activation_clusters: active_clusters=2 nodes=2" in out
    assert "[cluster_srp] filter_clusters_by_size: 2 -> 1" in out
    assert len(clusters.active_clusters) == 1
