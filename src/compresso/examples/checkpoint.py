from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
import json
import shutil
import tempfile
import zipfile

import numpy as np
from scipy.sparse import csr_matrix, load_npz, save_npz


MANIFEST_NAME = "manifest.json"
SPLIT_DIR = "data"
ELSA_DIR = "elsa"
SAE_DIR = "sae"
COMPRESSED_ELSA_DIR = "compressed_elsa"


def _as_obj_array(xs: list[np.ndarray]) -> np.ndarray:
    return np.array([np.asarray(x, dtype=np.int64) for x in xs], dtype=object)


def _read_obj_array(x: np.ndarray) -> list[np.ndarray]:
    return [np.asarray(v, dtype=np.int64) for v in x.tolist()]


def _zip_dir(root: Path, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file in sorted(p for p in root.rglob("*") if p.is_file()):
            zf.write(file, file.relative_to(root).as_posix())
    tmp.replace(path)


@contextmanager
def update_checkpoint(path: str | Path) -> Iterator[Path]:
    """Extract a zip checkpoint to a temp dir, let caller edit it, then rewrite it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        if path.exists():
            with zipfile.ZipFile(path, "r") as zf:
                zf.extractall(root)
        yield root
        _zip_dir(root, path)


@contextmanager
def read_checkpoint(path: str | Path) -> Iterator[Path]:
    """Extract a zip checkpoint to a read-only temp workspace."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        with zipfile.ZipFile(path, "r") as zf:
            zf.extractall(root)
        yield root


def load_manifest(root: str | Path) -> dict[str, Any]:
    path = Path(root) / MANIFEST_NAME
    if not path.exists():
        return {"format": "compresso.recsys.zip", "version": 1, "stages": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def save_manifest(root: str | Path, manifest: dict[str, Any]) -> None:
    path = Path(root) / MANIFEST_NAME
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def update_stage_manifest(root: str | Path, stage: str, metadata: dict[str, Any]) -> None:
    manifest = load_manifest(root)
    manifest.setdefault("format", "compresso.recsys.zip")
    manifest.setdefault("version", 1)
    manifest.setdefault("stages", {})[stage] = metadata
    save_manifest(root, manifest)


def save_json(root: str | Path, relpath: str, data: dict[str, Any]) -> Path:
    path = Path(root) / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_json(root: str | Path, relpath: str) -> dict[str, Any]:
    return json.loads((Path(root) / relpath).read_text(encoding="utf-8"))


def save_recsys_split(
    root: str | Path,
    *,
    item_ids: np.ndarray,
    x_train: csr_matrix,
    val_source_indices: list[np.ndarray],
    val_target_indices: list[np.ndarray],
    test_source_indices: list[np.ndarray],
    test_target_indices: list[np.ndarray],
    metadata: dict[str, Any] | None = None,
) -> None:
    root = Path(root)
    data_dir = root / SPLIT_DIR
    if data_dir.exists():
        shutil.rmtree(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    save_npz(data_dir / "train_matrix.npz", x_train.tocsr())
    np.savez_compressed(
        data_dir / "split.npz",
        item_ids=np.asarray(item_ids).astype(str),
        val_source_indices=_as_obj_array(val_source_indices),
        val_target_indices=_as_obj_array(val_target_indices),
        test_source_indices=_as_obj_array(test_source_indices),
        test_target_indices=_as_obj_array(test_target_indices),
    )
    update_stage_manifest(root, "data", metadata or {})


def load_recsys_split(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    split = np.load(root / SPLIT_DIR / "split.npz", allow_pickle=True)
    return {
        "item_ids": split["item_ids"],
        "x_train": load_npz(root / SPLIT_DIR / "train_matrix.npz").tocsr(),
        "val_source_indices": _read_obj_array(split["val_source_indices"]),
        "val_target_indices": _read_obj_array(split["val_target_indices"]),
        "test_source_indices": _read_obj_array(split["test_source_indices"]),
        "test_target_indices": _read_obj_array(split["test_target_indices"]),
    }
