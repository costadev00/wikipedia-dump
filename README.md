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

- `text`: cleaned plain text (no templates/tags)
- `title`: page title
- `page_id`: page id from the XML dump
- `ns`: page namespace id
- `section_texts`: list of cleaned section texts (lead included)

## Notes

The program creates `data/` and `data/raw/` automatically if they are missing.

Default filters applied:

- Non-main namespaces are skipped (use `--include-non-main` to include them).
- Disambiguation pages are skipped (use `--include-disambiguation` to include them).
- Empty pages are skipped.
- List items are removed from `text` by default (use `--keep-lists` to keep them).

## Publish to Hugging Face

Dataset name: `wikipedia-pt-br-extract`

1) Generate the outputs (JSONL + Parquet):

```bash
python -m wiki_pt_extract.cli --skip-redirects
```

2) Install publish dependency and upload:

```bash
pip install -r requirements.txt
python scripts/publish_hf_dataset.py --repo wikipedia-pt-br-extract
```

This uploads the merged Parquet and JSONL to:
`https://huggingface.co/datasets/<your-username>/wikipedia-pt-br-extract`

## Token counting (Qwen3 tokenizer)

Count total tokens with the Qwen3-1.7B-Base tokenizer:

```bash
python scripts/count_tokens.py --input data/ptwiki_articles1.jsonl --format jsonl
```

Parquet example:

```bash
python scripts/count_tokens.py --input data/ptwiki_articles1.parquet --format parquet
```
