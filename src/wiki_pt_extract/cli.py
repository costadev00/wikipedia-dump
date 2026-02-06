"""Command-line interface for Wikipedia dump extraction."""

from __future__ import annotations

import argparse
import json
import os
import re
from itertools import tee

from datasets import Dataset
import mwparserfromhell

from wiki_pt_extract.download import download_dump
from wiki_pt_extract.extract import iter_pages_from_bz2
from wiki_pt_extract.io import write_jsonl, write_parquet

DUMP_URL = "https://dumps.wikimedia.org/ptwiki/latest/ptwiki-latest-pages-articles.xml.bz2"
RAW_PATH = os.path.join("data", "raw", "ptwiki_articles1.bz2")
JSONL_PATH = os.path.join("data", "ptwiki_articles1.jsonl")
PARQUET_PATH = os.path.join("data", "ptwiki_articles1.parquet")

REDIRECT_RE = re.compile(r"^#(?:redirect|redirecionamento)", re.IGNORECASE)
DISAMBIG_RE = re.compile(r"\(desambigua(?:c|ç)ao\)", re.IGNORECASE)
DISAMBIG_TEMPLATE_RE = re.compile(r"\{\{\s*desambigua", re.IGNORECASE)
LIST_LINE_RE = re.compile(r"^\s*[*#;:]+\s*.*$", re.MULTILINE)
WHITESPACE_RE = re.compile(r"\s+")
MEDIA_OPTION_RE = re.compile(
    r"\b(?:thumb|thumbnail|frameless|frame|upright|miniaturadaimagem)\b(?:\s*\|\s*\d+px)?",
    re.IGNORECASE,
)
MEDIA_SIZE_RE = re.compile(r"\b\d{2,4}px\b", re.IGNORECASE)
MEDIA_LINK_RE = re.compile(r"\blink\s*=", re.IGNORECASE)
MEDIA_PIPE_RE = re.compile(r"\|\s*(?:\d{1,4}px)?\s*\|", re.IGNORECASE)
MEDIA_ALIGN_RE = re.compile(r"\b(?:direita|esquerda|centro|center|left|right)\s*\|", re.IGNORECASE)
INFOBOX_PARAM_RE = re.compile(r"\|\s*[\wÀ-ÿ_-]+\s*=\s*[^|]+")
TEMPLATE_BRACES_RE = re.compile(r"\{\{|\}\}")
INFOBOX_NAME_RE = re.compile(r"\bInfo\s*/\s*Pa[ií]s\b", re.IGNORECASE)
EMPTY_BRACKETS_RE = re.compile(r"\(\s*\)|\[\s*\]|\{\s*\}|\s*\u0000")


def _is_redirect(text: str) -> bool:
    if not text:
        return False
    return bool(REDIRECT_RE.match(text.lstrip()))


def _is_disambiguation(title: str, text: str) -> bool:
    if title and DISAMBIG_RE.search(title):
        return True
    if text and DISAMBIG_TEMPLATE_RE.search(text):
        return True
    return False


def _normalize_text(text: str) -> str:
    if not text:
        return ""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return normalized.strip()


def _remove_noisy_tags(code) -> None:
    for tag in list(code.filter_tags(recursive=True)):
        try:
            tag_name = str(tag.tag).strip().lower() if tag.tag is not None else ""
        except Exception:
            continue
        if tag_name in {
            "ref",
            "references",
            "table",
            "gallery",
            "math",
            "code",
            "syntaxhighlight",
            "timeline",
            "pre",
            "source",
        }:
            try:
                tag.replace("")
            except Exception:
                try:
                    code.remove(tag)
                except ValueError:
                    pass


def _clean_wikitext(text: str, drop_lists: bool, min_section_chars: int) -> tuple[str, list[str]]:
    if not text:
        return "", []
    working = text
    if drop_lists:
        working = LIST_LINE_RE.sub("", working)

    code = mwparserfromhell.parse(working)
    for wikilink in list(code.filter_wikilinks(recursive=True)):
        target = str(wikilink.title).strip().lower()
        if target.startswith(
            (
                "file:",
                "ficheiro:",
                "imagem:",
                "image:",
            )
        ):
            try:
                wikilink.replace("")
            except Exception:
                try:
                    code.remove(wikilink)
                except ValueError:
                    pass
    for template in list(code.filter_templates(recursive=True)):
        try:
            template.replace("")
        except Exception:
            try:
                code.remove(template)
            except ValueError:
                pass
    _remove_noisy_tags(code)

    section_texts: list[str] = []
    for section in code.get_sections(include_lead=True, include_headings=False):
        raw = section.strip_code(normalize=True, collapse=True)
        cleaned = MEDIA_OPTION_RE.sub("", raw)
        cleaned = MEDIA_SIZE_RE.sub("", cleaned)
        cleaned = MEDIA_LINK_RE.sub("", cleaned)
        cleaned = MEDIA_PIPE_RE.sub(" ", cleaned)
        cleaned = MEDIA_ALIGN_RE.sub(" ", cleaned)
        cleaned = INFOBOX_PARAM_RE.sub(" ", cleaned)
        cleaned = INFOBOX_NAME_RE.sub(" ", cleaned)
        cleaned = TEMPLATE_BRACES_RE.sub(" ", cleaned)
        cleaned = EMPTY_BRACKETS_RE.sub(" ", cleaned)
        cleaned = WHITESPACE_RE.sub(" ", cleaned).strip()
        if cleaned and len(cleaned) >= min_section_chars:
            section_texts.append(cleaned)

    combined = "\n\n".join(section_texts)
    return combined, section_texts


def _ensure_dirs() -> None:
    os.makedirs(os.path.dirname(RAW_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(JSONL_PATH), exist_ok=True)


def build_filtered_rows(
    max_pages: int | None,
    skip_redirects: bool,
    include_non_main: bool,
    include_disambiguation: bool,
    drop_lists: bool,
    min_section_chars: int,
):
    seen = 0
    written = 0
    redirects_skipped = 0
    empty_skipped = 0
    namespace_skipped = 0
    disambiguation_skipped = 0

    def generator():
        nonlocal seen, written, redirects_skipped, empty_skipped, namespace_skipped, disambiguation_skipped
        for page in iter_pages_from_bz2(RAW_PATH):
            seen += 1
            if max_pages is not None and seen > max_pages:
                break
            if not include_non_main and page.get("ns") != 0:
                namespace_skipped += 1
                continue
            text_raw = _normalize_text(page.get("text") or "")
            text, section_texts = _clean_wikitext(text_raw, drop_lists, min_section_chars)
            page["text"] = text
            page["section_texts"] = section_texts
            if not text:
                empty_skipped += 1
                continue
            if skip_redirects and _is_redirect(text):
                redirects_skipped += 1
                continue
            if not include_disambiguation and _is_disambiguation(page.get("title") or "", text):
                disambiguation_skipped += 1
                continue
            written += 1
            yield page

    return generator(), lambda: (
        seen,
        written,
        redirects_skipped,
        empty_skipped,
        namespace_skipped,
        disambiguation_skipped,
    )


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
    parser.add_argument(
        "--include-non-main",
        action="store_true",
        help="Include pages outside the main namespace",
    )
    parser.add_argument(
        "--include-disambiguation",
        action="store_true",
        help="Include disambiguation pages",
    )
    parser.add_argument(
        "--keep-lists",
        action="store_true",
        help="Keep list items in cleaned text",
    )
    parser.add_argument(
        "--min-section-chars",
        type=int,
        default=1,
        help="Minimum characters per section text",
    )
    args = parser.parse_args()

    _ensure_dirs()
    if not os.path.exists(RAW_PATH):
        print(f"Downloading dump to {RAW_PATH}...")
        download_dump(DUMP_URL, RAW_PATH)

    rows_iter, counts = build_filtered_rows(
        args.max_pages,
        args.skip_redirects,
        args.include_non_main,
        args.include_disambiguation,
        not args.keep_lists,
        args.min_section_chars,
    )

    rows_for_jsonl, rows_for_parquet = tee(rows_iter)

    print(f"Writing JSONL to {JSONL_PATH}...")
    write_jsonl(rows_for_jsonl, JSONL_PATH)

    print(f"Writing Parquet to {PARQUET_PATH}...")
    write_parquet(rows_for_parquet, PARQUET_PATH)

    (
        seen,
        written,
        redirects_skipped,
        empty_skipped,
        namespace_skipped,
        disambiguation_skipped,
    ) = counts()
    print("Summary:")
    print(f"  Pages seen: {seen}")
    print(f"  Pages written: {written}")
    print(f"  Redirects skipped: {redirects_skipped}")
    print(f"  Empty texts skipped: {empty_skipped}")
    print(f"  Non-main namespace skipped: {namespace_skipped}")
    print(f"  Disambiguation skipped: {disambiguation_skipped}")

    quality_checks(PARQUET_PATH)


if __name__ == "__main__":
    main()
