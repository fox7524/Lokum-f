from __future__ import annotations

from typing import Iterable


def build_context_block(chunks: Iterable[str]) -> str:
    clean = [chunk.strip() for chunk in chunks if chunk and chunk.strip()]
    return "\n\n---\n\n".join(clean)
