from __future__ import annotations

import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

from .base import RecSysDataset


class Goodbooks(RecSysDataset):
    name = "goodbooks"
    url = "https://github.com/zygmuntz/goodbooks-10k/releases/download/v1.0/goodbooks-10k.zip"

    def download(self) -> None:
        zip_path = self.root / "goodbooks-10k.zip"
        if not zip_path.exists():
            urlretrieve(self.url, zip_path)

        marker = self.root / "ratings.csv"
        if not marker.exists():
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(self.root)

    def prepare(self) -> None:
        self.download()
        ratings_path = self.root / "ratings.csv"
        books_path = self.root / "books.csv"

        if not ratings_path.exists():
            raise FileNotFoundError(f"Missing ratings file: {ratings_path}")

        ratings = pd.read_csv(ratings_path)
        ratings = ratings.rename(columns={"user_id": "user_id", "book_id": "item_id", "rating": "value"})
        ratings["user_id"] = ratings["user_id"].astype(str)
        ratings["item_id"] = ratings["item_id"].astype(str)
        ratings["timestamp"] = None

        self._interactions = ratings[["user_id", "item_id", "value", "timestamp"]].copy()

        if books_path.exists():
            books = pd.read_csv(books_path)
            books = books.rename(columns={"book_id": "item_id", "title": "title", "authors": "authors"})
            books["item_id"] = books["item_id"].astype(str)
            keep = [c for c in ["item_id", "title", "authors", "average_rating"] if c in books.columns]
            self._item_metadata = books[keep].copy()
        else:
            self._item_metadata = pd.DataFrame(columns=["item_id", "title", "authors"])

