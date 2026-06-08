from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from recsys_train_sbert_from_checkpoint import build_texts


def test_build_texts_joins_configured_columns_with_separator():
    metadata = pd.DataFrame(
        {
            "title": ["Book A", "Book B"],
            "authors": ["Author A", ""],
            "description": ["Description A", np.nan],
        }
    )

    texts = build_texts(
        metadata,
        text_columns=["title", "authors", "description"],
        text_separator="\n",
    )

    assert texts == ["Book A\nAuthor A\nDescription A", "Book B"]


def test_build_texts_keeps_legacy_description_with_fallback():
    metadata = pd.DataFrame(
        {
            "title": ["Book A", "Book B"],
            "authors": ["Author A", "Author B"],
            "description": ["Description A", np.nan],
        }
    )

    texts = build_texts(
        metadata,
        text_column="description",
        fallback_columns=["title", "authors"],
        text_separator=" | ",
    )

    assert texts == ["Description A", "Book B | Author B"]
