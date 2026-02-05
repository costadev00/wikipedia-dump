"""Download utilities for Wikipedia dumps."""

from __future__ import annotations

import os
from typing import Optional

import requests
from tqdm import tqdm


def _get_remote_size(url: str) -> Optional[int]:
    try:
        response = requests.head(url, allow_redirects=True, timeout=30)
        response.raise_for_status()
    except requests.RequestException:
        return None
    content_length = response.headers.get("Content-Length")
    if not content_length:
        return None
    try:
        return int(content_length)
    except ValueError:
        return None


def download_dump(url: str, out_path: str) -> str:
    """Download a dump file with resume support.

    Args:
        url: URL to the dump.
        out_path: Destination file path.

    Returns:
        Path to the downloaded dump.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    remote_size = _get_remote_size(url)
    existing_size = os.path.getsize(out_path) if os.path.exists(out_path) else 0

    if remote_size and existing_size >= remote_size:
        return out_path

    headers = {}
    mode = "wb"
    resume_from = 0

    if remote_size and existing_size and existing_size < remote_size:
        headers["Range"] = f"bytes={existing_size}-"
        mode = "ab"
        resume_from = existing_size

    with requests.get(url, headers=headers, stream=True, timeout=60) as response:
        if resume_from and response.status_code != 206:
            mode = "wb"
            resume_from = 0
        response.raise_for_status()

        total = remote_size
        if resume_from and total is not None:
            progress_total = total
        else:
            progress_total = int(response.headers.get("Content-Length") or 0) or None

        with open(out_path, mode) as handle, tqdm(
            total=progress_total,
            initial=resume_from,
            unit="B",
            unit_scale=True,
            desc=os.path.basename(out_path),
        ) as progress:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                handle.write(chunk)
                progress.update(len(chunk))

    if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        raise RuntimeError(f"Download failed or produced empty file: {out_path}")

    return out_path
