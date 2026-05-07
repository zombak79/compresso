from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def save_recsys_checkpoint(
    path: str | Path,
    *,
    item_ids: np.ndarray,
    item_embeddings: np.ndarray,
    source_indices: list[np.ndarray],
    target_indices: list[np.ndarray],
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    src_obj = np.array([np.asarray(x, dtype=np.int64) for x in source_indices], dtype=object)
    tgt_obj = np.array([np.asarray(x, dtype=np.int64) for x in target_indices], dtype=object)

    np.savez_compressed(
        path,
        item_ids=np.asarray(item_ids).astype(str),
        item_embeddings=np.asarray(item_embeddings, dtype=np.float32),
        source_indices=src_obj,
        target_indices=tgt_obj,
    )
    return path


def load_recsys_checkpoint(path: str | Path) -> dict[str, Any]:
    data = np.load(Path(path), allow_pickle=True)
    return {
        "item_ids": data["item_ids"],
        "item_embeddings": data["item_embeddings"],
        "source_indices": [np.asarray(x, dtype=np.int64) for x in data["source_indices"].tolist()],
        "target_indices": [np.asarray(x, dtype=np.int64) for x in data["target_indices"].tolist()],
    }

