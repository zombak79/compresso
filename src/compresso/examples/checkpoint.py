from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def save_recsys_checkpoint(
    path: str | Path,
    *,
    item_ids: np.ndarray,
    item_embeddings: np.ndarray,
    val_source_indices: list[np.ndarray],
    val_target_indices: list[np.ndarray],
    test_source_indices: list[np.ndarray],
    test_target_indices: list[np.ndarray],
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    val_src_obj = np.array([np.asarray(x, dtype=np.int64) for x in val_source_indices], dtype=object)
    val_tgt_obj = np.array([np.asarray(x, dtype=np.int64) for x in val_target_indices], dtype=object)
    test_src_obj = np.array([np.asarray(x, dtype=np.int64) for x in test_source_indices], dtype=object)
    test_tgt_obj = np.array([np.asarray(x, dtype=np.int64) for x in test_target_indices], dtype=object)

    np.savez_compressed(
        path,
        item_ids=np.asarray(item_ids).astype(str),
        item_embeddings=np.asarray(item_embeddings, dtype=np.float32),
        val_source_indices=val_src_obj,
        val_target_indices=val_tgt_obj,
        test_source_indices=test_src_obj,
        test_target_indices=test_tgt_obj,
    )
    return path


def load_recsys_checkpoint(path: str | Path) -> dict[str, Any]:
    data = np.load(Path(path), allow_pickle=True)
    return {
        "item_ids": data["item_ids"],
        "item_embeddings": data["item_embeddings"],
        "val_source_indices": [np.asarray(x, dtype=np.int64) for x in data["val_source_indices"].tolist()],
        "val_target_indices": [np.asarray(x, dtype=np.int64) for x in data["val_target_indices"].tolist()],
        "test_source_indices": [np.asarray(x, dtype=np.int64) for x in data["test_source_indices"].tolist()],
        "test_target_indices": [np.asarray(x, dtype=np.int64) for x in data["test_target_indices"].tolist()],
    }
