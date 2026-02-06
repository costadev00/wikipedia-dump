"""Output helpers for extracted Wikipedia pages."""

from __future__ import annotations

import json
import os
from typing import Iterable, List

from datasets import Dataset, Features, Sequence, Value, concatenate_datasets


FEATURES = Features(
    {
        "text": Value("string"),
        "title": Value("string"),
        "page_id": Value("string"),
        "ns": Value("int32"),
        "section_texts": Sequence(Value("string")),
    }
)


def write_jsonl(rows_iter: Iterable[dict], out_path: str) -> None:
    """Write rows as JSONL with one object per line."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        for row in rows_iter:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _chunk_rows(rows_iter: Iterable[dict], batch_size: int) -> Iterable[List[dict]]:
    batch: List[dict] = []
    for row in rows_iter:
        batch.append(row)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def write_parquet(rows_iter: Iterable[dict], out_path: str, batch_size: int = 10_000) -> None:
    """Write rows to parquet shards and merge them into a final dataset."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    shard_paths = []
    for index, batch in enumerate(_chunk_rows(rows_iter, batch_size), start=1):
        dataset = Dataset.from_list(batch, features=FEATURES)
        shard_path = out_path.replace(
            ".parquet", f"_part_{index:05d}.parquet"
        )
        dataset.to_parquet(shard_path)
        shard_paths.append(shard_path)

    if not shard_paths:
        Dataset.from_list([], features=FEATURES).to_parquet(out_path)
        return

    datasets = [Dataset.from_parquet(path) for path in shard_paths]
    combined = concatenate_datasets(datasets)
    combined.to_parquet(out_path)
