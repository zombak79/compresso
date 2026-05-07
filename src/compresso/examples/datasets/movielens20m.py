from __future__ import annotations

import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

from .base import RecSysDataset


class MovieLens20M(RecSysDataset):
    name = "movielens20m"
    url = "https://files.grouplens.org/datasets/movielens/ml-20m.zip"

    def download(self) -> None:
        zip_path = self.root / "ml-20m.zip"
        if not zip_path.exists():
            urlretrieve(self.url, zip_path)

        extract_dir = self.root / "ml-20m"
        if not extract_dir.exists():
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(self.root)

    def prepare(self) -> None:
        self.download()
        ratings_path = self.root / "ml-20m" / "ratings.csv"
        movies_path = self.root / "ml-20m" / "movies.csv"

        if not ratings_path.exists():
            raise FileNotFoundError(f"Missing ratings file: {ratings_path}")

        ratings = pd.read_csv(ratings_path)
        ratings = ratings.rename(
            columns={
                "userId": "user_id",
                "movieId": "item_id",
                "rating": "value",
                "timestamp": "timestamp",
            }
        )
        ratings["user_id"] = ratings["user_id"].astype(str)
        ratings["item_id"] = ratings["item_id"].astype(str)

        self._interactions = ratings[["user_id", "item_id", "value", "timestamp"]].copy()

        if movies_path.exists():
            movies = pd.read_csv(movies_path)
            movies = movies.rename(columns={"movieId": "item_id", "title": "title", "genres": "genres"})
            movies["item_id"] = movies["item_id"].astype(str)
            self._item_metadata = movies[["item_id", "title", "genres"]].copy()
        else:
            self._item_metadata = pd.DataFrame(columns=["item_id", "title", "genres"])

