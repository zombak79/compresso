from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
import json
import tempfile
import zipfile

import numpy as np
import pandas as pd

from compresso import load_srp_tensor
from compresso.clustering import SparseClusterSet, load_cluster_graph
from compresso.params.srp import SRPTensor

from .paths import DEFAULT_BUNDLE

DEFAULT_GRAPH_STAGE = "semantic_clustering_merged"
GRAPH_NAME = "graph.json"


def _read_obj_array(x: np.ndarray) -> list[np.ndarray]:
    return [np.asarray(v, dtype=np.int64) for v in x.tolist()]


@contextmanager
def extracted_bundle(path: str | Path = DEFAULT_BUNDLE) -> Iterator[Path]:
    """Extract the demo zip to a temporary directory."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        with zipfile.ZipFile(path, "r") as zf:
            zf.extractall(root)
        yield root


def load_manifest(root: str | Path) -> dict[str, Any]:
    return json.loads((Path(root) / "manifest.json").read_text(encoding="utf-8"))


def load_metadata(root: str | Path) -> pd.DataFrame:
    return pd.read_csv(Path(root) / "data" / "entity_metadata.csv", dtype={"item_id": str})


def load_split(root: str | Path) -> dict[str, Any]:
    split = np.load(Path(root) / "data" / "split.npz", allow_pickle=True)
    return {
        "item_ids": split["item_ids"].astype(str),
        "val_source_indices": _read_obj_array(split["val_source_indices"]),
        "val_target_indices": _read_obj_array(split["val_target_indices"]),
        "test_source_indices": _read_obj_array(split["test_source_indices"]),
        "test_target_indices": _read_obj_array(split["test_target_indices"]),
    }


def list_graph_stages(root: str | Path) -> list[str]:
    manifest = load_manifest(root)
    return list(manifest.get("graph_stages") or [DEFAULT_GRAPH_STAGE])


def load_graph(root: str | Path, stage: str = DEFAULT_GRAPH_STAGE) -> SparseClusterSet:
    return load_cluster_graph(Path(root) / stage / GRAPH_NAME)


def load_item_srp(root: str | Path, stage: str = "sbert_sae") -> SRPTensor:
    return load_srp_tensor(Path(root) / stage / "sparse_embeddings.srp.pt", map_location="cpu")


def load_goodbooks_bundle(path: str | Path = DEFAULT_BUNDLE, graph_stage: str = DEFAULT_GRAPH_STAGE) -> dict[str, Any]:
    """Load all core demo objects from the zip bundle.

    This helper is convenient for validation scripts. Streamlit pages may prefer
    using extracted_bundle with caching to avoid repeated zip extraction.
    """
    with extracted_bundle(path) as root:
        return {
            "manifest": load_manifest(root),
            "metadata": load_metadata(root),
            "split": load_split(root),
            "graph": load_graph(root, graph_stage),
            "srp": load_item_srp(root),
            "graph_stages": list_graph_stages(root),
        }
