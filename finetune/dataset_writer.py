from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Any


def write_text_jsonl_stream(path: Path, texts: Iterable[str]) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for text in texts:
            handle.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_chat_jsonl_stream(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps({"messages": list(row["messages"])}, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_completion_jsonl_stream(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            payload = {
                "prompt": row["prompt"],
                "completion": row["completion"],
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            count += 1
    return count


def detect_jsonl_format(path: Path) -> str:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            obj = json.loads(stripped)
            if "messages" in obj:
                return "chat"
            if "prompt" in obj and "completion" in obj:
                return "completion"
            if "text" in obj:
                return "text"
            return "unknown"
    return "empty"


def write_jsonl_stream(path: Path, texts: Iterable[str]) -> int:
    return write_text_jsonl_stream(path, texts)
