from __future__ import annotations

import hashlib
import os
from typing import Any

from .chunker import chunk_text
from .normalize import normalize_text


def file_signature(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def build_file_state(source_path: str, raw_text: str, chunk_size: int, overlap: int) -> dict[str, Any]:
    chunks = chunk_text(raw_text, chunk_size=chunk_size, overlap=overlap)
    return {
        "source_path": source_path,
        "file_signature": file_signature(raw_text),
        "chunk_count": len(chunks),
        "chunk_signatures": [chunk.signature for chunk in chunks],
    }


def content_hash_for_path(path: str, size: int | None = None) -> str | None:
    p = os.path.abspath(path or "")
    if not p:
        return None
    ext = os.path.splitext(p)[1].lower()
    if ext == ".zim":
        return None

    raw_max = (os.environ.get("LOKUMAI_RAG_CONTENT_HASH_MAX_BYTES") or "").strip()
    max_bytes = 50 * 1024 * 1024
    if raw_max:
        try:
            max_bytes = int(raw_max)
        except Exception:
            max_bytes = 50 * 1024 * 1024
    if max_bytes <= 0:
        return None

    if size is None:
        try:
            size = int(os.stat(p).st_size)
        except Exception:
            return None
    if int(size) > int(max_bytes):
        return None

    h = hashlib.sha256()
    try:
        with open(p, "rb") as f:
            while True:
                b = f.read(1024 * 1024)
                if not b:
                    break
                h.update(b)
    except Exception:
        return None
    return h.hexdigest()
