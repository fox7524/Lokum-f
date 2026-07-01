from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def write_jsonl_stream(path: Path, texts: Iterable[str]) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for text in texts:
            handle.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            count += 1
    return count
