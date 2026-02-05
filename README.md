# Portuguese Wikipedia Dump Extraction

This project downloads the latest Portuguese Wikipedia XML dump and extracts page text into JSONL and Parquet files for downstream processing. The dump used is:

- `https://dumps.wikimedia.org/ptwiki/latest/ptwiki-latest-pages-articles.xml.bz2`

The extracted `text` field is **raw wikitext**, not cleaned or rendered.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick test run (subset)

```bash
python -m wiki_pt_extract.cli --max-pages 2000 --skip-redirects
```

## Full extraction

```bash
python -m wiki_pt_extract.cli --skip-redirects
```

## Outputs

- `data/ptwiki_articles1.jsonl`
- `data/ptwiki_articles1.parquet`
- Sharded parquet files (kept alongside the merged file):
  - `data/ptwiki_articles1_part_00001.parquet`, `data/ptwiki_articles1_part_00002.parquet`, etc.

The pipeline writes shard files for each batch and then merges them into the final `data/ptwiki_articles1.parquet` dataset.

## Schema

Each row has:

- `text`: raw page wikitext
- `title`: page title
- `page_id`: page id from the XML dump

## Notes

The program creates `data/` and `data/raw/` automatically if they are missing.
