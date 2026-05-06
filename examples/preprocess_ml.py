from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, Features, Sequence, Value


try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None


TAG_RE = re.compile(r"\s+")


@dataclass
class Config:
    input_dir: Path
    output_dir: Path
    min_tag_freq: int = 20
    max_tag_vocab: int = 5000
    max_history_len: int = 200
    min_rating_as_positive: float = 4.0
    tag_embedding_model: Optional[str] = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 256
    use_genome_scores: bool = False
    genome_top_k: int = 10


EVENT_TYPE_RATING = 1
EVENT_TYPE_TAG = 2


SEQUENCE_FEATURES = {
    "event_type": Sequence(Value("int8")),
    "movie_id": Sequence(Value("int32")),
    "timestamp": Sequence(Value("int64")),
    "rating_value": Sequence(Value("float32")),
    "rating_bucket": Sequence(Value("int8")),
    "tag_id": Sequence(Value("int32")),
    "tag_present": Sequence(Value("int8")),
    "tag_emb": Sequence(Sequence(Value("float32"))),
}


TARGET_FEATURES = {
    "target_event_type": Value("int8"),
    "target_movie_id": Value("int32"),
    "target_timestamp": Value("int64"),
    "target_rating_value": Value("float32"),
    "target_rating_bucket": Value("int8"),
    "target_tag_id": Value("int32"),
    "target_tag_present": Value("int8"),
    "target_tag_emb": Sequence(Value("float32")),
    "target_is_positive": Value("int8"),
}


CONTEXT_FEATURES = {
    "user_id": Value("int32"),
    "history_len": Value("int32"),
    "user_num_ratings": Value("int32"),
    "user_num_tags": Value("int32"),
    "user_mean_rating": Value("float32"),
    "movie_title_history": Sequence(Value("string")),
    "movie_genres_history": Sequence(Sequence(Value("string"))),
    "movie_title_target": Value("string"),
    "movie_genres_target": Sequence(Value("string")),
}


GLOBAL_FEATURES = {
    "num_movies": Value("int32"),
    "tag_vocab_size": Value("int32"),
}


ALL_FEATURES = Features({
    **CONTEXT_FEATURES,
    **SEQUENCE_FEATURES,
    **TARGET_FEATURES,
    **GLOBAL_FEATURES,
})


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Convert MovieLens 20M data to a Tencent-like Hugging Face dataset.")
    parser.add_argument("--input_dir", type=Path, required=True, help="Directory containing movies.csv, ratings.csv, tags.csv")
    parser.add_argument("--output_dir", type=Path, required=True, help="Where to save the DatasetDict")
    parser.add_argument("--min_tag_freq", type=int, default=20)
    parser.add_argument("--max_tag_vocab", type=int, default=5000)
    parser.add_argument("--max_history_len", type=int, default=200)
    parser.add_argument("--min_rating_as_positive", type=float, default=4.0)
    parser.add_argument("--tag_embedding_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--use_genome_scores", action="store_true")
    parser.add_argument("--genome_top_k", type=int, default=10)
    args = parser.parse_args()
    return Config(**vars(args))


def normalize_tag(text: str) -> str:
    text = str(text).strip().lower()
    text = TAG_RE.sub(" ", text)
    return text


def rating_to_bucket(r: float) -> int:
    if pd.isna(r):
        return 0
    if r < 2.5:
        return 1
    if r < 4.0:
        return 2
    return 3


def maybe_load_genome(input_dir: Path, movies_df: pd.DataFrame, top_k: int) -> tuple[dict[int, list[str]], dict[int, list[float]]]:
    scores_path = input_dir / "genome-scores.csv"
    tags_path = input_dir / "genome-tags.csv"
    if not (scores_path.exists() and tags_path.exists()):
        return {}, {}

    genome_tags = pd.read_csv(tags_path)
    genome_scores = pd.read_csv(scores_path)
    genome_scores = genome_scores.merge(genome_tags, on="tagId", how="left")
    genome_scores = genome_scores.sort_values(["movieId", "relevance"], ascending=[True, False])
    genome_scores = genome_scores.groupby("movieId").head(top_k)

    names = genome_scores.groupby("movieId")["tag"].apply(list).to_dict()
    scores = genome_scores.groupby("movieId")["relevance"].apply(lambda x: [float(v) for v in x]).to_dict()
    return names, scores


def load_tables(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[int, list[str]], dict[int, list[float]]]:
    movies = pd.read_csv(cfg.input_dir / "movies.csv")
    ratings = pd.read_csv(cfg.input_dir / "ratings.csv")
    tags = pd.read_csv(cfg.input_dir / "tags.csv")

    ratings["timestamp"] = pd.to_datetime(ratings["timestamp"], unit="s", utc=True)
    tags["timestamp"] = pd.to_datetime(tags["timestamp"], unit="s", utc=True)
    tags["tag"] = tags["tag"].map(normalize_tag)
    tags = tags[tags["tag"].str.len() > 0].copy()

    genome_names, genome_scores = ({}, {})
    if cfg.use_genome_scores:
        genome_names, genome_scores = maybe_load_genome(cfg.input_dir, movies, cfg.genome_top_k)

    return movies, ratings, tags, genome_names, genome_scores


def build_tag_vocab(tags: pd.DataFrame, min_freq: int, max_vocab: int) -> dict[str, int]:
    counts = tags["tag"].value_counts()
    counts = counts[counts >= min_freq].head(max_vocab)
    vocab = {tag: idx + 1 for idx, tag in enumerate(counts.index.tolist())}
    return vocab


def build_tag_embeddings(unique_tags: Iterable[str], model_name: Optional[str], batch_size: int) -> dict[str, list[float]]:
    unique_tags = list(unique_tags)
    if not unique_tags:
        return {}

    if model_name is None:
        return {tag: [] for tag in unique_tags}

    if SentenceTransformer is None:
        raise ImportError(
            "sentence-transformers is not installed. Install it or pass --tag_embedding_model '' and edit the script to disable embeddings."
        )

    model = SentenceTransformer(model_name)
    emb = model.encode(unique_tags, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)
    emb = np.asarray(emb, dtype=np.float32)
    return {tag: emb[i].tolist() for i, tag in enumerate(unique_tags)}


def parse_movie_side_info(movies: pd.DataFrame) -> tuple[dict[int, str], dict[int, list[str]]]:
    title_map = dict(zip(movies["movieId"], movies["title"]))
    genres_map = {
        int(mid): ([] if pd.isna(genres) else str(genres).split("|"))
        for mid, genres in zip(movies["movieId"], movies["genres"])
    }
    return title_map, genres_map


def build_event_table(
    ratings: pd.DataFrame,
    tags: pd.DataFrame,
    tag_vocab: dict[str, int],
    tag_embs: dict[str, list[float]],
) -> pd.DataFrame:
    rating_events = ratings.copy()
    rating_events["event_type"] = EVENT_TYPE_RATING
    rating_events["rating_value"] = rating_events["rating"].astype(np.float32)
    rating_events["rating_bucket"] = rating_events["rating_value"].map(rating_to_bucket).astype(np.int8)
    rating_events["tag_id"] = 0
    rating_events["tag_present"] = 0
    rating_events["tag_text"] = ""
    rating_events["tag_emb"] = [[] for _ in range(len(rating_events))]
    rating_events = rating_events[[
        "userId", "movieId", "timestamp", "event_type",
        "rating_value", "rating_bucket", "tag_id", "tag_present", "tag_text", "tag_emb"
    ]]

    tag_events = tags.copy()
    tag_events["event_type"] = EVENT_TYPE_TAG
    tag_events["rating_value"] = np.float32(np.nan)
    tag_events["rating_bucket"] = 0
    tag_events["tag_id"] = tag_events["tag"].map(lambda t: int(tag_vocab.get(t, 0))).astype(np.int32)
    tag_events["tag_present"] = 1
    tag_events["tag_text"] = tag_events["tag"]
    tag_events["tag_emb"] = tag_events["tag"].map(lambda t: tag_embs.get(t, []))
    tag_events = tag_events[[
        "userId", "movieId", "timestamp", "event_type",
        "rating_value", "rating_bucket", "tag_id", "tag_present", "tag_text", "tag_emb"
    ]]

    events = pd.concat([rating_events, tag_events], ignore_index=True)
    events = events.sort_values(["userId", "timestamp", "event_type", "movieId"]).reset_index(drop=True)
    return events


def float_or_zero(value: float) -> float:
    if pd.isna(value):
        return 0.0
    return float(value)


def vector_or_empty(value) -> list[float]:
    if isinstance(value, list):
        return [float(v) for v in value]
    return []


def make_example(
    user_id: int,
    history: pd.DataFrame,
    target: pd.Series,
    title_map: dict[int, str],
    genres_map: dict[int, list[str]],
    num_movies: int,
    tag_vocab_size: int,
    max_history_len: int,
) -> dict:
    history = history.tail(max_history_len)

    history_titles = [title_map.get(int(mid), "") for mid in history["movieId"].tolist()]
    history_genres = [genres_map.get(int(mid), []) for mid in history["movieId"].tolist()]

    rating_mask = history["event_type"] == EVENT_TYPE_RATING
    user_num_ratings = int(rating_mask.sum())
    user_num_tags = int((history["event_type"] == EVENT_TYPE_TAG).sum())
    user_mean_rating = float(history.loc[rating_mask, "rating_value"].mean()) if user_num_ratings > 0 else 0.0

    ex = {
        "user_id": int(user_id),
        "history_len": int(len(history)),
        "user_num_ratings": user_num_ratings,
        "user_num_tags": user_num_tags,
        "user_mean_rating": np.float32(user_mean_rating),
        "movie_title_history": history_titles,
        "movie_genres_history": history_genres,
        "movie_title_target": title_map.get(int(target["movieId"]), ""),
        "movie_genres_target": genres_map.get(int(target["movieId"]), []),
        "event_type": history["event_type"].astype(np.int8).tolist(),
        "movie_id": history["movieId"].astype(np.int32).tolist(),
        "timestamp": (history["timestamp"].view("int64") // 10**9).astype(np.int64).tolist(),
        "rating_value": [float_or_zero(v) for v in history["rating_value"].tolist()],
        "rating_bucket": history["rating_bucket"].astype(np.int8).tolist(),
        "tag_id": history["tag_id"].astype(np.int32).tolist(),
        "tag_present": history["tag_present"].astype(np.int8).tolist(),
        "tag_emb": [vector_or_empty(v) for v in history["tag_emb"].tolist()],
        "target_event_type": int(target["event_type"]),
        "target_movie_id": int(target["movieId"]),
        "target_timestamp": int(target["timestamp"].value // 10**9),
        "target_rating_value": np.float32(float_or_zero(target["rating_value"])),
        "target_rating_bucket": int(target["rating_bucket"]),
        "target_tag_id": int(target["tag_id"]),
        "target_tag_present": int(target["tag_present"]),
        "target_tag_emb": vector_or_empty(target["tag_emb"]),
        "target_is_positive": int(float_or_zero(target["rating_value"]) >= 4.0) if int(target["event_type"]) == EVENT_TYPE_RATING else 1,
        "num_movies": int(num_movies),
        "tag_vocab_size": int(tag_vocab_size),
    }
    return ex


def split_users(examples: list[dict]) -> DatasetDict:
    train, validation, test = [], [], []
    for idx, ex in enumerate(examples):
        bucket = idx % 10
        if bucket == 8:
            validation.append(ex)
        elif bucket == 9:
            test.append(ex)
        else:
            train.append(ex)

    return DatasetDict({
        "train": Dataset.from_list(train, features=ALL_FEATURES),
        "validation": Dataset.from_list(validation, features=ALL_FEATURES),
        "test": Dataset.from_list(test, features=ALL_FEATURES),
    })


def build_dataset(cfg: Config) -> tuple[DatasetDict, dict]:
    movies, ratings, tags, genome_names, genome_scores = load_tables(cfg)
    title_map, genres_map = parse_movie_side_info(movies)

    tag_vocab = build_tag_vocab(tags, cfg.min_tag_freq, cfg.max_tag_vocab)
    tag_embs = build_tag_embeddings(tag_vocab.keys(), cfg.tag_embedding_model or None, cfg.batch_size)
    events = build_event_table(ratings, tags, tag_vocab, tag_embs)

    examples = []
    dropped_users = 0
    num_movies = int(movies["movieId"].nunique())

    for user_id, user_events in events.groupby("userId", sort=False):
        if len(user_events) < 2:
            dropped_users += 1
            continue
        history = user_events.iloc[:-1]
        target = user_events.iloc[-1]
        ex = make_example(
            user_id=int(user_id),
            history=history,
            target=target,
            title_map=title_map,
            genres_map=genres_map,
            num_movies=num_movies,
            tag_vocab_size=len(tag_vocab),
            max_history_len=cfg.max_history_len,
        )
        if cfg.use_genome_scores:
            ex["target_genome_names"] = genome_names.get(int(target["movieId"]), [])
            ex["target_genome_scores"] = genome_scores.get(int(target["movieId"]), [])
        examples.append(ex)

    dataset = split_users(examples)
    meta = {
        "input_dir": str(cfg.input_dir),
        "num_users_total": int(events["userId"].nunique()),
        "num_users_kept": len(examples),
        "num_users_dropped_lt2_events": dropped_users,
        "num_movies": int(movies["movieId"].nunique()),
        "num_rating_events": int(len(ratings)),
        "num_tag_events": int(len(tags)),
        "tag_vocab_size": int(len(tag_vocab)),
        "tag_embedding_model": cfg.tag_embedding_model,
        "max_history_len": cfg.max_history_len,
        "min_tag_freq": cfg.min_tag_freq,
        "max_tag_vocab": cfg.max_tag_vocab,
        "use_genome_scores": cfg.use_genome_scores,
        "event_type_mapping": {"rating": EVENT_TYPE_RATING, "tag": EVENT_TYPE_TAG},
        "rating_bucket_mapping": {"0": "missing_or_not_applicable", "1": "low", "2": "mid", "3": "high"},
    }
    return dataset, meta


def main() -> None:
    cfg = parse_args()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    dataset, meta = build_dataset(cfg)
    dataset.save_to_disk(str(cfg.output_dir / "hf_dataset"))

    with open(cfg.output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("Saved dataset to", cfg.output_dir / "hf_dataset")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
