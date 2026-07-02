from __future__ import annotations

from pathlib import Path

import numpy as np

from compresso.clustering import (
    ScoredTag,
    SparseCluster,
    SparseClusterSet,
    SparseVector,
    load_cluster_graph,
    save_cluster_graph,
)


def _graph() -> SparseClusterSet:
    child = SparseCluster(
        cluster_id="feature:1:pos",
        centroid=SparseVector(np.asarray([1], dtype=np.int64), np.asarray([1.0], dtype=np.float32), 8),
        entity_indices=np.asarray([0, 1], dtype=np.int64),
        source_cluster_ids=("feature:1:pos",),
        parent_cluster_ids=("merge:test:2",),
        tags=(ScoredTag(0, "fantasy", 0.9, count=2.0, metadata={"kind": "genre"}),),
        label="Fantasy Books",
        stats={"mean_activation": 0.5},
        metadata={"features": (1,), "signs": (1,)},
    )
    parent = SparseCluster(
        cluster_id="merge:test:2",
        centroid=SparseVector(np.asarray([1, 2], dtype=np.int64), np.asarray([0.7, -0.7], dtype=np.float32), 8),
        entity_indices=np.asarray([0, 1, 2], dtype=np.int64),
        source_cluster_ids=("feature:1:pos", "feature:2:neg"),
        child_cluster_ids=("feature:1:pos",),
        label="Speculative Fiction",
        description="A broader fantasy/speculative cluster.",
        metadata={"merge_strategy": "test"},
    )
    return SparseClusterSet(
        clusters=(child, parent),
        n_entities=3,
        n_features=8,
        active_cluster_ids=("merge:test:2",),
        entity_ids=np.asarray(["a", "b", "c"]),
        feature_ids=np.arange(8),
        assignment_mode="top_m_signed",
        history=({"phase": "test"},),
        metadata={"dataset": "synthetic"},
    )


def test_cluster_graph_json_roundtrip(tmp_path: Path):
    graph = _graph()
    path = tmp_path / "graph.json"

    save_cluster_graph(graph, path)
    loaded = load_cluster_graph(path)

    assert loaded.n_entities == graph.n_entities
    assert loaded.n_features == graph.n_features
    assert loaded.active_cluster_ids == graph.active_cluster_ids
    assert loaded.assignment_mode == graph.assignment_mode
    assert loaded.metadata == graph.metadata
    assert loaded.history == graph.history
    assert loaded.entity_ids.tolist() == ["a", "b", "c"]
    assert loaded.feature_ids.tolist() == list(range(8))
    assert loaded.cluster_by_id["feature:1:pos"].label == "Fantasy Books"
    assert loaded.cluster_by_id["feature:1:pos"].parent_cluster_ids == ("merge:test:2",)
    assert loaded.cluster_by_id["feature:1:pos"].tags[0].name == "fantasy"
    assert loaded.cluster_by_id["merge:test:2"].child_cluster_ids == ("feature:1:pos",)
    assert np.allclose(loaded.cluster_by_id["merge:test:2"].centroid.values, [0.7, -0.7])
