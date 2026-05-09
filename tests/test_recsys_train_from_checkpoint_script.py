"""Regression smoke test for recsys_train_sae_from_checkpoint.py."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

from compresso.examples.checkpoint import save_recsys_checkpoint


def test_recsys_train_from_checkpoint_script_smoke(tmp_path: Path):
    # Tiny synthetic checkpoint
    n_items = 12
    emb_dim = 8
    rng = np.random.default_rng(0)
    item_ids = np.array([f"i{i}" for i in range(n_items)])
    item_embeddings = rng.standard_normal((n_items, emb_dim), dtype=np.float32)

    val_source_indices = [
        np.array([0, 1, 2], dtype=np.int64),
        np.array([3, 4], dtype=np.int64),
        np.array([5, 6, 7], dtype=np.int64),
        np.array([8], dtype=np.int64),
    ]
    val_target_indices = [
        np.array([3, 4], dtype=np.int64),
        np.array([0, 2], dtype=np.int64),
        np.array([1, 9], dtype=np.int64),
        np.array([10, 11], dtype=np.int64),
    ]
    test_source_indices = [
        np.array([1, 2], dtype=np.int64),
        np.array([4, 5], dtype=np.int64),
        np.array([6, 7], dtype=np.int64),
        np.array([9], dtype=np.int64),
    ]
    test_target_indices = [
        np.array([0, 3], dtype=np.int64),
        np.array([1, 8], dtype=np.int64),
        np.array([2, 10], dtype=np.int64),
        np.array([11], dtype=np.int64),
    ]

    ckpt_path = tmp_path / "elsa_checkpoint.npz"
    save_recsys_checkpoint(
        ckpt_path,
        item_ids=item_ids,
        item_embeddings=item_embeddings,
        val_source_indices=val_source_indices,
        val_target_indices=val_target_indices,
        test_source_indices=test_source_indices,
        test_target_indices=test_target_indices,
    )

    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "examples" / "recsys_train_sae_from_checkpoint.py"

    cmd = [
        sys.executable,
        str(script),
        "--checkpoint_path",
        str(ckpt_path),
        "--output_dir",
        str(tmp_path / "out"),
        "--device",
        "cpu",
        "--seed",
        "0",
        "--sae_hidden_dim",
        "16",
        "--sae_k",
        "4",
        "--sae_epochs",
        "1",
        "--sae_batch_size",
        "4",
        "--eval_batch_size",
        "4",
    ]
    proc = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)

    assert proc.returncode == 0, proc.stderr
    assert "Original embedding metrics:" in proc.stdout
    assert "SAE embedding metrics:" in proc.stdout
    assert "Perf drop vs original:" in proc.stdout
    assert (tmp_path / "out" / "sae_model.pt").exists()
    assert (tmp_path / "out" / "sae_sparse_embeddings.srp.pt").exists()
    assert (tmp_path / "out" / "sae_result.json").exists()
