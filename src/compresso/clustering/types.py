from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class SparseVector:
    """Small sparse vector used as a reusable cluster handle."""

    indices: np.ndarray
    values: np.ndarray
    size: int

    def __post_init__(self) -> None:
        indices = np.asarray(self.indices, dtype=np.int64)
        values = np.asarray(self.values, dtype=np.float32)
        if indices.ndim != 1 or values.ndim != 1:
            raise ValueError("SparseVector indices and values must be 1D")
        if indices.shape[0] != values.shape[0]:
            raise ValueError("SparseVector indices and values must have the same length")
        if int(self.size) <= 0:
            raise ValueError("SparseVector size must be positive")
        if indices.size > 0 and (indices.min() < 0 or indices.max() >= int(self.size)):
            raise ValueError("SparseVector indices are out of bounds")
        object.__setattr__(self, "indices", indices)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "size", int(self.size))

    def normalized(self, p: float = 2.0, eps: float = 1e-12) -> "SparseVector":
        norm = float(np.linalg.norm(self.values, ord=p))
        if norm <= eps:
            return self
        return SparseVector(self.indices.copy(), (self.values / norm).astype(np.float32), self.size)

    def to_dense(self) -> np.ndarray:
        out = np.zeros(self.size, dtype=np.float32)
        np.add.at(out, self.indices, self.values)
        return out

    @staticmethod
    def sum(vectors: list["SparseVector"], *, normalize: bool = True) -> "SparseVector":
        if not vectors:
            raise ValueError("Cannot sum an empty list of SparseVector objects")
        size = vectors[0].size
        if any(v.size != size for v in vectors):
            raise ValueError("All SparseVector objects must have the same size")
        acc: dict[int, float] = {}
        for vector in vectors:
            for idx, val in zip(vector.indices.tolist(), vector.values.tolist()):
                acc[int(idx)] = acc.get(int(idx), 0.0) + float(val)
        indices = np.array(sorted(acc), dtype=np.int64)
        values = np.array([acc[int(i)] for i in indices], dtype=np.float32)
        out = SparseVector(indices, values, size)
        return out.normalized() if normalize else out


@dataclass(frozen=True)
class ScoredTag:
    tag_id: int
    name: str
    score: float
    count: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SparseCluster:
    cluster_id: str
    centroid: SparseVector
    entity_indices: np.ndarray
    source_cluster_ids: tuple[str, ...] = ()
    tags: tuple[ScoredTag, ...] = ()
    label: str | None = None
    description: str | None = None
    stats: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        entities = np.asarray(self.entity_indices, dtype=np.int64)
        if entities.ndim != 1:
            raise ValueError("SparseCluster.entity_indices must be 1D")
        object.__setattr__(self, "entity_indices", np.unique(entities))

    @property
    def entity_count(self) -> int:
        return int(self.entity_indices.size)

    def with_updates(self, **kwargs: Any) -> "SparseCluster":
        return replace(self, **kwargs)


@dataclass(frozen=True)
class SparseClusterSet:
    clusters: tuple[SparseCluster, ...]
    n_entities: int
    n_features: int
    entity_ids: np.ndarray | None = None
    feature_ids: np.ndarray | None = None
    assignment_mode: str = "dominant_signed"
    history: tuple[Mapping[str, Any], ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if int(self.n_entities) < 0:
            raise ValueError("n_entities must be non-negative")
        if int(self.n_features) <= 0:
            raise ValueError("n_features must be positive")
        object.__setattr__(self, "n_entities", int(self.n_entities))
        object.__setattr__(self, "n_features", int(self.n_features))
        if self.entity_ids is not None:
            entity_ids = np.asarray(self.entity_ids)
            if entity_ids.shape[0] != self.n_entities:
                raise ValueError("entity_ids length must equal n_entities")
            object.__setattr__(self, "entity_ids", entity_ids)
        if self.feature_ids is not None:
            feature_ids = np.asarray(self.feature_ids)
            if feature_ids.shape[0] != self.n_features:
                raise ValueError("feature_ids length must equal n_features")
            object.__setattr__(self, "feature_ids", feature_ids)

    @property
    def cluster_by_id(self) -> dict[str, SparseCluster]:
        return {cluster.cluster_id: cluster for cluster in self.clusters}

    @property
    def entity_to_cluster_ids(self) -> dict[int, list[str]]:
        out: dict[int, list[str]] = {}
        for cluster in self.clusters:
            for entity_idx in cluster.entity_indices.tolist():
                out.setdefault(int(entity_idx), []).append(cluster.cluster_id)
        return out

    def with_clusters(self, clusters: list[SparseCluster] | tuple[SparseCluster, ...], *, history_entry: Mapping[str, Any] | None = None) -> "SparseClusterSet":
        history = self.history + ((dict(history_entry),) if history_entry is not None else ())
        return replace(self, clusters=tuple(clusters), history=history)

    def append_history(self, entry: Mapping[str, Any]) -> "SparseClusterSet":
        return replace(self, history=self.history + (dict(entry),))
