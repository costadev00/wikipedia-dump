"""Count tokens for extracted Wikipedia datasets using a Qwen tokenizer."""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Iterable

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

DEFAULT_MODEL = "Qwen/Qwen3-1.7B-Base"
DEFAULT_JSONL = "data/ptwiki_articles1.jsonl"


def _iter_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}") from exc


def _iter_parquet(path: str) -> Iterable[dict]:
    dataset = load_dataset("parquet", data_files=path, streaming=True)
    return dataset["train"]


def _count_tokens(value, tokenizer, add_special_tokens: bool) -> int:
    if value is None:
        return 0
    if isinstance(value, list):
        total = 0
        for item in value:
            text = "" if item is None else str(item)
            total += len(tokenizer.encode(text, add_special_tokens=add_special_tokens))
        return total
    text = str(value)
    return len(tokenizer.encode(text, add_special_tokens=add_special_tokens))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Count total tokens in extracted Wikipedia datasets using a Qwen tokenizer."
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_JSONL,
        help="Path to JSONL or Parquet dataset",
    )
    parser.add_argument(
        "--format",
        choices=("jsonl", "parquet"),
        default="jsonl",
        help="Input dataset format",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Hugging Face model ID for AutoTokenizer",
    )
    parser.add_argument(
        "--field",
        default="text",
        help="Dataset field to tokenize",
    )
    parser.add_argument(
        "--add-special-tokens",
        action="store_true",
        help="Include special tokens (BOS/EOS) in the count",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Stop after processing this many documents",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow custom tokenizer code from the model repo",
    )
    parser.add_argument(
        "--report",
        default="reports/token_count_report.md",
        help="Path to write a markdown report",
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code
    )

    if args.format == "jsonl":
        iterator = _iter_jsonl(args.input)
    else:
        iterator = _iter_parquet(args.input)

    total_tokens = 0
    total_docs = 0
    missing_fields = 0

    start = time.perf_counter()
    for row in tqdm(iterator, unit="docs"):
        if args.max_docs is not None and total_docs >= args.max_docs:
            break
        total_docs += 1
        if args.field not in row:
            missing_fields += 1
            continue
        total_tokens += _count_tokens(
            row.get(args.field),
            tokenizer,
            add_special_tokens=args.add_special_tokens,
        )
    elapsed = time.perf_counter() - start

    avg_tokens = (total_tokens / total_docs) if total_docs else 0
    docs_per_sec = (total_docs / elapsed) if elapsed else 0
    tokens_per_sec = (total_tokens / elapsed) if elapsed else 0

    print("Token count summary")
    print(f"  Model: {args.model}")
    print(f"  Field: {args.field}")
    print(f"  Documents: {total_docs}")
    if missing_fields:
        print(f"  Missing field: {missing_fields}")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Avg tokens/doc: {avg_tokens:.2f}")
    print(f"  Elapsed: {elapsed:.2f}s")

    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    with open(args.report, "w", encoding="utf-8") as handle:
        handle.write("# Token Count Report\n\n")
        handle.write("## Run details\n")
        handle.write(f"- Timestamp: {timestamp}\n")
        handle.write(f"- Input: {args.input}\n")
        handle.write(f"- Format: {args.format}\n")
        handle.write(f"- Model: {args.model}\n")
        handle.write(f"- Field: {args.field}\n")
        handle.write(f"- Add special tokens: {args.add_special_tokens}\n")
        handle.write(f"- Max docs: {args.max_docs}\n\n")
        handle.write("## Results\n")
        handle.write(f"- Documents: {total_docs}\n")
        if missing_fields:
            handle.write(f"- Missing field: {missing_fields}\n")
        handle.write(f"- Total tokens: {total_tokens}\n")
        handle.write(f"- Avg tokens/doc: {avg_tokens:.2f}\n")
        handle.write(f"- Elapsed: {elapsed:.2f}s\n")
        handle.write(f"- Docs/sec: {docs_per_sec:.2f}\n")
        handle.write(f"- Tokens/sec: {tokens_per_sec:.2f}\n")


if __name__ == "__main__":
    main()
