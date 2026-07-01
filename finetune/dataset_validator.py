from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class ValidationResult:
    total: int
    invalid: int


def validate_jsonl_rows(rows: Iterable[str]) -> ValidationResult:
    total = 0
    invalid = 0
    for row in rows:
        total += 1
        try:
            obj = json.loads(row)
        except json.JSONDecodeError:
            invalid += 1
            continue
        if not isinstance(obj, dict) or not isinstance(obj.get("text"), str) or not obj["text"].strip():
            invalid += 1
    return ValidationResult(total=total, invalid=invalid)
