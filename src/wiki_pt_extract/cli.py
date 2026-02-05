"""Command-line interface for Wikipedia dump extraction."""

from __future__ import annotations

import argparse
import json
import os
import re
from itertools import tee

from datasets import Dataset

from wiki_pt_extract.download import download_dump
from wiki_pt_extract.extract import iter_pages_from_bz2
from wiki_pt_extract.io import write_jsonl, write_parquet

DUMP_URL = "https://dumps.wikimedia.org/ptwiki/latest/ptwiki-latest-pages-articles.xml.bz2"
RAW_PATH = os.path.join("data", "raw", "ptwiki_articles1.bz2")
JSONL_PATH = os.path.join("data", "ptwiki_articles1.jsonl")
PARQUET_PATH = os.path.join("data", "ptwiki_articles1.parquet")

REDIRECT_RE = re.compile(r"^#(?:redirect|redirecionamento)", re.IGNORECASE)


def _is_redirect(text: str) -> bool:
    if not text:
        return False
    return bool(REDIRECT_RE.match(text.lstrip()))


def _ensure_dirs() -> None:
    os.makedirs(os.path.dirname(RAW_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(JSONL_PATH), exist_ok=True)


def build_filtered_rows(max_pages: int | None, skip_redirects: bool):
    seen = 0
    written = 0
    redirects_skipped = 0
    empty_skipped = 0

    def generator():
        nonlocal seen, written, redirects_skipped, empty_skipped
        for page in iter_pages_from_bz2(RAW_PATH):
            seen += 1
            if max_pages is not None and seen > max_pages:
                break
            text = page.get("text") or ""
            if not text.strip():
                empty_skipped += 1
                continue
            if skip_redirects and _is_redirect(text):
                redirects_skipped += 1
                continue
            written += 1
            yield page

    return generator(), lambda: (seen, written, redirects_skipped, empty_skipped)


def quality_checks(parquet_path: str) -> None:
    dataset = Dataset.from_parquet(parquet_path)
    print("Dataset features:")
    print(dataset.features)

    print("Sample rows:")
    for idx in range(min(3, len(dataset))):
        print(json.dumps(dataset[idx], ensure_ascii=False))

    if "text" not in dataset.column_names:
        raise AssertionError("Expected 'text' column in dataset")
    if not any((text or "").strip() for text in dataset["text"]):
        raise AssertionError("No non-empty text found in dataset")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Portuguese Wikipedia dumps.")
    parser.add_argument("--max-pages", type=int, default=None, help="Max pages to process")
    parser.add_argument(
        "--skip-redirects",
        action="store_true",
        help="Skip redirect pages",
    )
    args = parser.parse_args()

    _ensure_dirs()
    if not os.path.exists(RAW_PATH):
        print(f"Downloading dump to {RAW_PATH}...")
        download_dump(DUMP_URL, RAW_PATH)

    rows_iter, counts = build_filtered_rows(args.max_pages, args.skip_redirects)

    rows_for_jsonl, rows_for_parquet = tee(rows_iter)

    print(f"Writing JSONL to {JSONL_PATH}...")
    write_jsonl(rows_for_jsonl, JSONL_PATH)

    print(f"Writing Parquet to {PARQUET_PATH}...")
    write_parquet(rows_for_parquet, PARQUET_PATH)

    seen, written, redirects_skipped, empty_skipped = counts()
    print("Summary:")
    print(f"  Pages seen: {seen}")
    print(f"  Pages written: {written}")
    print(f"  Redirects skipped: {redirects_skipped}")
    print(f"  Empty texts skipped: {empty_skipped}")

    quality_checks(PARQUET_PATH)


if __name__ == "__main__":
    main()
