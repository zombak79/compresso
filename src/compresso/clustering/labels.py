from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Literal

from .types import SparseCluster, SparseClusterSet

ClusterScope = Literal["active", "all", "leaves", "roots"]


def _progress_iter(iterable, *, enabled: bool, total: int | None = None, desc: str = ""):
    if not enabled:
        return iterable
    try:
        from tqdm.auto import tqdm
    except Exception:  # pragma: no cover - optional progress dependency
        return iterable
    return tqdm(iterable, total=total, desc=desc)


def _scoped_clusters(clusters: SparseClusterSet, scope: ClusterScope) -> tuple[SparseCluster, ...]:
    if scope == "active":
        return clusters.active_clusters
    if scope == "all":
        return clusters.clusters
    if scope == "leaves":
        return clusters.leaf_clusters
    if scope == "roots":
        return clusters.root_clusters
    raise ValueError("scope must be one of 'active', 'all', 'leaves', or 'roots'")


def _parse_label_result(result: object) -> tuple[str | None, str | None, dict[str, Any]]:
    if isinstance(result, str):
        return result, None, {}
    if isinstance(result, Mapping):
        data = dict(result)
        label = data.pop("label", None)
        description = data.pop("description", None)
        return (
            str(label) if label is not None else None,
            str(description) if description is not None else None,
            data,
        )
    if result is None:
        return None, None, {}
    return str(result), None, {}


def label_clusters(
    clusters: SparseClusterSet,
    *,
    entity_metadata: object,
    text_extractor: Callable[[SparseCluster, object], object],
    label_fn: Callable[[object], object],
    cluster_scope: ClusterScope = "active",
    overwrite: bool = False,
    verbose: bool = False,
    show_progress: bool = False,
) -> SparseClusterSet:
    """Populate cluster labels/descriptions using user-provided callbacks.

    ``text_extractor`` owns domain-specific metadata-to-text conversion.
    ``label_fn`` owns LLM/API/local-model prompting. Compresso only coordinates
    the calls and stores returned labels on the graph.
    """
    scoped = _scoped_clusters(clusters, cluster_scope)
    updated_by_id = clusters.cluster_by_id
    n_labeled = 0
    n_skipped = 0

    for cluster in _progress_iter(
        scoped,
        enabled=show_progress,
        total=len(scoped),
        desc="label_clusters",
    ):
        if not overwrite and cluster.label is not None:
            n_skipped += 1
            continue

        text = text_extractor(cluster, entity_metadata)
        result = label_fn(text)
        label, description, result_metadata = _parse_label_result(result)
        labeling_metadata = {
            "method": "user_fn",
            "cluster_scope": cluster_scope,
            "text_type": type(text).__name__,
        }
        if result_metadata:
            labeling_metadata["result_metadata"] = result_metadata
        metadata = dict(cluster.metadata)
        metadata["labeling"] = labeling_metadata
        updated_by_id[cluster.cluster_id] = (
            updated_by_id[cluster.cluster_id].with_updates(
                label=label if label is not None else cluster.label,
                description=description if description is not None else cluster.description,
                metadata=metadata,
            )
        )
        n_labeled += 1

    if verbose:
        print(f"[label_clusters] labeled={n_labeled} skipped={n_skipped} scope={cluster_scope}")
    return clusters.with_clusters(
        [updated_by_id[cluster.cluster_id] for cluster in clusters.clusters],
        history_entry={
            "phase": "label_clusters",
            "cluster_scope": cluster_scope,
            "n_labeled": n_labeled,
            "n_skipped": n_skipped,
            "overwrite": overwrite,
        },
    )
