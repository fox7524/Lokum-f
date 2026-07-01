from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .normalize import normalize_text


@dataclass(frozen=True)
class TextChunk:
    text: str
    signature: str


def chunk_signature(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 120) -> list[TextChunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    normalized = normalize_text(text)
    if not normalized:
        return []

    out: list[TextChunk] = []
    start = 0
    n = len(normalized)
    while start < n:
        end = min(n, start + chunk_size)
        chunk = normalized[start:end].strip()
        if chunk:
            out.append(TextChunk(text=chunk, signature=chunk_signature(chunk)))
        if end >= n:
            break
        start = end - overlap
    return out
