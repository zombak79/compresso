from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any
import json

import numpy as np

from .types import ScoredTag, SparseCluster, SparseClusterSet, SparseVector


FORMAT = "compresso.sparse_cluster_graph"
VERSION = 1


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)


def _tag_to_dict(tag: ScoredTag) -> dict[str, Any]:
    return {
        "tag_id": int(tag.tag_id),
        "name": str(tag.name),
        "score": float(tag.score),
        "count": float(tag.count),
        "metadata": _jsonable(tag.metadata),
    }


def _tag_from_dict(data: Mapping[str, Any]) -> ScoredTag:
    return ScoredTag(
        tag_id=int(data["tag_id"]),
        name=str(data["name"]),
        score=float(data["score"]),
        count=float(data.get("count", 0.0)),
        metadata=dict(data.get("metadata", {})),
    )


def _cluster_to_dict(cluster: SparseCluster) -> dict[str, Any]:
    return {
        "cluster_id": cluster.cluster_id,
        "centroid": {
            "indices": cluster.centroid.indices.tolist(),
            "values": cluster.centroid.values.tolist(),
            "size": int(cluster.centroid.size),
        },
        "entity_indices": cluster.entity_indices.tolist(),
        "source_cluster_ids": list(cluster.source_cluster_ids),
        "parent_cluster_ids": list(cluster.parent_cluster_ids),
        "child_cluster_ids": list(cluster.child_cluster_ids),
        "tags": [_tag_to_dict(tag) for tag in cluster.tags],
        "label": cluster.label,
        "description": cluster.description,
        "stats": _jsonable(cluster.stats),
        "metadata": _jsonable(cluster.metadata),
    }


def _cluster_from_dict(data: Mapping[str, Any]) -> SparseCluster:
    centroid = data["centroid"]
    return SparseCluster(
        cluster_id=str(data["cluster_id"]),
        centroid=SparseVector(
            np.asarray(centroid["indices"], dtype=np.int64),
            np.asarray(centroid["values"], dtype=np.float32),
            int(centroid["size"]),
        ),
        entity_indices=np.asarray(data["entity_indices"], dtype=np.int64),
        source_cluster_ids=tuple(str(v) for v in data.get("source_cluster_ids", ())),
        parent_cluster_ids=tuple(str(v) for v in data.get("parent_cluster_ids", ())),
        child_cluster_ids=tuple(str(v) for v in data.get("child_cluster_ids", ())),
        tags=tuple(_tag_from_dict(tag) for tag in data.get("tags", ())),
        label=data.get("label"),
        description=data.get("description"),
        stats=dict(data.get("stats", {})),
        metadata=dict(data.get("metadata", {})),
    )


def graph_to_dict(graph: SparseClusterSet) -> dict[str, Any]:
    return {
        "format": FORMAT,
        "version": VERSION,
        "n_entities": int(graph.n_entities),
        "n_features": int(graph.n_features),
        "active_cluster_ids": list(graph.active_cluster_ids or ()),
        "entity_ids": graph.entity_ids.tolist() if graph.entity_ids is not None else None,
        "feature_ids": graph.feature_ids.tolist() if graph.feature_ids is not None else None,
        "assignment_mode": graph.assignment_mode,
        "history": _jsonable(graph.history),
        "metadata": _jsonable(graph.metadata),
        "clusters": [_cluster_to_dict(cluster) for cluster in graph.clusters],
    }


def graph_from_dict(data: Mapping[str, Any]) -> SparseClusterSet:
    if data.get("format") != FORMAT:
        raise ValueError(f"Unsupported cluster graph format: {data.get('format')!r}")
    if int(data.get("version", -1)) != VERSION:
        raise ValueError(f"Unsupported cluster graph version: {data.get('version')!r}")
    return SparseClusterSet(
        clusters=tuple(_cluster_from_dict(cluster) for cluster in data["clusters"]),
        n_entities=int(data["n_entities"]),
        n_features=int(data["n_features"]),
        active_cluster_ids=tuple(str(v) for v in data.get("active_cluster_ids", ())),
        entity_ids=np.asarray(data["entity_ids"]) if data.get("entity_ids") is not None else None,
        feature_ids=np.asarray(data["feature_ids"]) if data.get("feature_ids") is not None else None,
        assignment_mode=str(data.get("assignment_mode", "dominant_signed")),
        history=tuple(dict(entry) for entry in data.get("history", ())),
        metadata=dict(data.get("metadata", {})),
    )


def save_cluster_graph(graph: SparseClusterSet, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(graph_to_dict(graph), indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_cluster_graph(path: str | Path) -> SparseClusterSet:
    return graph_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
