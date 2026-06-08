from __future__ import annotations

from functools import partial

import numpy as np
import torch
from scipy.sparse import csr_matrix

from compresso.clustering import (
    assign_cluster_tags,
    build_activation_clusters,
    cluster_srp,
    filter_clusters_by_size,
    merge_clusters_by_entity_containment,
    merge_clusters_by_entity_iou,
    merge_clusters_by_feature_containment,
    merge_clusters_by_tag_similarity,
    run_clustering_pipeline,
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
    assert len(merged.clusters) == 2
    merged_cluster = next(c for c in merged.clusters if c.entity_indices.tolist() == [0, 1])
    assert set(merged_cluster.centroid.indices.tolist()) == {0, 1}
    singleton_cluster = next(c for c in merged.clusters if c.entity_indices.tolist() == [2])
    assert set(singleton_cluster.centroid.indices.tolist()) == {2, 3}


def test_merge_clusters_by_entity_iou_verbose_reports_rounds(capsys):
    srp = _srp([[0, 1], [0, 1]], [[3.0, 2.0], [4.0, 2.5]], n_features=2)
    clusters = build_activation_clusters(srp, mode="top_m_signed", top_m=2)

    merged = merge_clusters_by_entity_iou(clusters, threshold=1.0, verbose=True)

    out = capsys.readouterr().out
    assert "[merge_clusters_by_entity_iou] round=1/10 clusters=2 threshold=1.000000" in out
    assert "[merge_clusters_by_entity_iou] merged: 2 -> 1" in out
    assert len(merged.clusters) == 1


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

    assert len(merged.clusters) == 2
    merged_cluster = next(c for c in merged.clusters if c.entity_count == 49)
    assert set(merged_cluster.source_cluster_ids) == {"broad", "specific"}
    assert set(merged_cluster.centroid.indices.tolist()) == {370, 717, 960, 1580, 3982}


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

    assert len(merged.clusters) == 2
    merged_cluster = next(c for c in merged.clusters if set(c.source_cluster_ids) == {"broad", "specific"})
    assert merged_cluster.entity_indices.tolist() == [0, 1, 2, 3, 4]
    assert any(c.cluster_id == "opposite" for c in merged.clusters)


def test_merge_clusters_by_tag_similarity():
    srp = _srp([[0], [1], [2]], [[1.0], [1.0], [1.0]], n_features=3)
    clusters = build_activation_clusters(srp)
    entity_tag = csr_matrix(np.array([[1, 0], [1, 0], [0, 1]], dtype=np.float32))
    tagged = assign_cluster_tags(clusters, entity_tag, ["same", "other"], top_k=1)

    merged = merge_clusters_by_tag_similarity(tagged, threshold=1.0)
    assert len(merged.clusters) == 2
    assert any(c.entity_indices.tolist() == [0, 1] for c in merged.clusters)


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

    assert len(clusters.clusters) == 2
    merged = next(c for c in clusters.clusters if c.entity_indices.tolist() == [0, 1])
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

    assert len(clusters.clusters) == 1
    assert clusters.clusters[0].entity_indices.tolist() == [0, 1]


def test_filter_clusters_by_size_direct():
    srp = _srp([[0], [1]], [[1.0], [1.0]], n_features=2)
    clusters = build_activation_clusters(srp, min_cluster_size=1)
    filtered = filter_clusters_by_size(clusters, min_cluster_size=2)
    assert len(filtered.clusters) == 0


def test_run_clustering_pipeline_composes_build_and_transform_steps(capsys):
    srp = _srp(
        [[0, 1], [0, 1], [0, 2]],
        [[2.0, -1.0], [3.0, -2.0], [4.0, 1.0]],
        n_features=3,
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
    assert "clusters_in=" in out
    assert len(clusters.clusters) == 1
    cluster = clusters.clusters[0]
    assert cluster.cluster_id == "combo:0:pos&1:neg"
    assert cluster.entity_indices.tolist() == [0, 1]
    assert cluster.tags[0].name == "same"


def test_cluster_srp_verbose_reports_phases(capsys):
    srp = _srp([[0], [0], [1]], [[1.0], [2.0], [1.0]], n_features=2)

    clusters = cluster_srp(srp, min_cluster_size=1, post_merge_min_cluster_size=2, verbose=True)

    out = capsys.readouterr().out
    assert "[cluster_srp] build_activation_clusters: clusters=2" in out
    assert "[cluster_srp] filter_clusters_by_size: 2 -> 1" in out
    assert len(clusters.clusters) == 1
