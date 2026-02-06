"""Publish extracted dataset to Hugging Face Hub."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from huggingface_hub import HfApi

DEFAULT_REPO = "wikipedia-pt-br-extract"
DEFAULT_ORG = None
DEFAULT_DATA_DIR = Path("data")
DEFAULT_PARQUET = DEFAULT_DATA_DIR / "ptwiki_articles1.parquet"
DEFAULT_JSONL = DEFAULT_DATA_DIR / "ptwiki_articles1.jsonl"
DEFAULT_STAGING = Path("hf_dataset")
DUMP_URL = "https://dumps.wikimedia.org/ptwiki/latest/ptwiki-latest-pages-articles.xml.bz2"

DATASET_CARD = """---
license: cc-by-sa-3.0
language:
- pt
pretty_name: Portuguese Wikipedia Extract (raw wikitext)
---

# Portuguese Wikipedia Extract (cleaned text)

This dataset contains cleaned text extracted from the latest Portuguese Wikipedia dump.

## Source

- Dump URL: {dump_url}

## Schema

Each row has:

- `text`: cleaned plain text (no templates/tags)
- `title`: page title
- `page_id`: page id from the XML dump
- `ns`: page namespace id
- `section_texts`: list of cleaned section texts (lead included)

## Notes

- Empty `text` pages are skipped.
- Redirect pages can be skipped depending on extraction flags.
- Non-main namespaces are skipped by default.
- Disambiguation pages are skipped by default.
- List items are removed by default (use `--keep-lists` to keep them).
- The dataset is provided in JSONL and Parquet formats.
"""


def build_dataset_card(out_path: Path) -> None:
    out_path.write_text(DATASET_CARD.format(dump_url=DUMP_URL), encoding="utf-8")


def stage_files(
    parquet_path: Path,
    jsonl_path: Path | None,
    include_shards: bool,
    staging_dir: Path,
) -> None:
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    build_dataset_card(staging_dir / "README.md")

    shutil.copy2(parquet_path, staging_dir / parquet_path.name)
    if jsonl_path:
        shutil.copy2(jsonl_path, staging_dir / jsonl_path.name)

    if include_shards:
        for shard in parquet_path.parent.glob("ptwiki_articles1_part_*.parquet"):
            shutil.copy2(shard, staging_dir / shard.name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish dataset to Hugging Face Hub.")
    parser.add_argument("--repo", default=DEFAULT_REPO, help="Dataset repo name")
    parser.add_argument("--org", default=DEFAULT_ORG, help="Organization/user name")
    parser.add_argument(
        "--parquet", default=str(DEFAULT_PARQUET), help="Path to merged parquet"
    )
    parser.add_argument(
        "--jsonl", default=str(DEFAULT_JSONL), help="Path to JSONL (optional)"
    )
    parser.add_argument(
        "--no-jsonl", action="store_true", help="Do not upload JSONL"
    )
    parser.add_argument(
        "--include-shards",
        action="store_true",
        help="Upload parquet shard files",
    )
    parser.add_argument(
        "--staging",
        default=str(DEFAULT_STAGING),
        help="Staging directory for upload",
    )
    args = parser.parse_args()

    parquet_path = Path(args.parquet)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Missing parquet file: {parquet_path}")

    jsonl_path = None if args.no_jsonl else Path(args.jsonl)
    if jsonl_path and not jsonl_path.exists():
        raise FileNotFoundError(f"Missing JSONL file: {jsonl_path}")

    staging_dir = Path(args.staging)
    stage_files(parquet_path, jsonl_path, args.include_shards, staging_dir)

    repo_id = f"{args.org}/{args.repo}" if args.org else args.repo

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    api.upload_folder(
        folder_path=str(staging_dir),
        repo_id=repo_id,
        repo_type="dataset",
    )

    print(f"Uploaded dataset to https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    main()
