"""Extraction helpers for Wikipedia XML dumps."""

from __future__ import annotations

import bz2
import xml.etree.ElementTree as ET
from typing import Dict, Iterator


def iter_pages_from_bz2(bz2_path: str) -> Iterator[Dict[str, str]]:
    """Iterate over pages in a compressed Wikipedia XML dump."""
    with bz2.open(bz2_path, "rb") as handle:
        context = ET.iterparse(handle, events=("end",))
        for _event, elem in context:
            if not elem.tag.endswith("page"):
                continue

            title = ""
            page_id = ""
            latest_text = ""
            namespace = 0

            for child in list(elem):
                if child.tag.endswith("title"):
                    title = child.text or ""
                elif child.tag.endswith("id"):
                    page_id = child.text or ""
                elif child.tag.endswith("ns"):
                    try:
                        namespace = int(child.text or "0")
                    except ValueError:
                        namespace = 0
                elif child.tag.endswith("revision"):
                    for revision_child in child.iter():
                        if revision_child.tag.endswith("text"):
                            latest_text = revision_child.text or ""

            yield {
                "title": title,
                "page_id": page_id,
                "text": latest_text,
                "ns": namespace,
            }

            elem.clear()
