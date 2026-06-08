from __future__ import annotations

import torch

from compresso.clustering import assign_to_clusters, build_activation_clusters
from compresso.params.srp import SRPTensor


def _srp(cols, vals, n_features=4):
    return SRPTensor(
        cols=torch.tensor(cols, dtype=torch.long),
        vals=torch.tensor(vals, dtype=torch.float32),
        shape=(len(cols), n_features),
    )


def test_build_dominant_signed_clusters():
    srp = _srp(
        [[0, 1], [1, 2], [0, 3], [2, 3]],
        [[3.0, 1.0], [-4.0, 1.0], [-5.0, 0.5], [2.0, -6.0]],
    )
    clusters = build_activation_clusters(srp)

    by_id = clusters.cluster_by_id
    assert set(by_id) == {"feature:0:pos", "feature:0:neg", "feature:1:neg", "feature:3:neg"}
    assert by_id["feature:0:pos"].entity_indices.tolist() == [0]
    assert by_id["feature:0:neg"].entity_indices.tolist() == [2]
    assert by_id["feature:1:neg"].centroid.indices.tolist() == [1]
    assert by_id["feature:1:neg"].centroid.values.tolist() == [-1.0]


def test_top_m_signed_allows_multiple_memberships():
    srp = _srp(
        [[0, 1], [0, 1]],
        [[3.0, -2.0], [4.0, -5.0]],
    )
    clusters = build_activation_clusters(srp, mode="top_m_signed", top_m=2)

    assert clusters.cluster_by_id["feature:0:pos"].entity_indices.tolist() == [0, 1]
    assert clusters.cluster_by_id["feature:1:neg"].entity_indices.tolist() == [0, 1]
    assert clusters.entity_to_cluster_ids[0] == ["feature:0:pos", "feature:1:neg"]


def test_combo_signed_builds_exact_and_clusters():
    srp = _srp(
        [[10, 22, 91], [10, 22, 77], [10, 55, 91]],
        [[3.0, -2.0, 1.0], [4.0, -5.0, 0.5], [2.0, -0.25, 6.0]],
        n_features=100,
    )

    clusters = build_activation_clusters(
        srp,
        mode="combo_signed",
        top_m=3,
        combo_size=2,
        min_cluster_size=2,
    )

    by_id = clusters.cluster_by_id
    assert by_id["combo:10:pos&22:neg"].entity_indices.tolist() == [0, 1]
    assert by_id["combo:10:pos&91:pos"].entity_indices.tolist() == [0, 2]
    assert by_id["combo:10:pos&22:neg"].centroid.indices.tolist() == [10, 22]
    assert by_id["combo:10:pos&22:neg"].metadata["combo_size"] == 2
    assert clusters.entity_to_cluster_ids[0] == ["combo:10:pos&22:neg", "combo:10:pos&91:pos"]


def test_combo_signed_validates_combo_size_against_top_m():
    srp = _srp([[0, 1]], [[1.0, 2.0]], n_features=2)

    try:
        build_activation_clusters(srp, mode="combo_signed", top_m=2, combo_size=3)
    except ValueError as e:
        assert "combo_size" in str(e)
    else:
        raise AssertionError("Expected combo_size validation error")


def test_assign_to_clusters_by_centroid_overlap():
    train = _srp([[0, 1], [2, 1]], [[3.0, 1.0], [-2.0, 0.1]], n_features=3)
    clusters = build_activation_clusters(train)
    new = _srp([[0, 2]], [[2.0, -0.5]], n_features=3)

    assigned = assign_to_clusters(new, clusters, top_k=1)
    assert assigned[0][0][0] == "feature:0:pos"
    assert assigned[0][0][1] == 2.0
