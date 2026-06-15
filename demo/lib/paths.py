from __future__ import annotations

from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = DEMO_DIR / "data"
DEFAULT_BUNDLE = DATA_DIR / "goodbooks" / "goodbooks_demo.zip"
