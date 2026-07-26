from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class ValidationResult:
    total: int
    invalid: int


SUPPORTED_CHAT_ROLES = {"system", "user", "assistant", "tool"}


def _is_non_empty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_valid_message_list(messages: object) -> bool:
    if not isinstance(messages, list) or not messages:
        return False
    for message in messages:
        if not isinstance(message, dict):
            return False
        role = message.get("role")
        content = message.get("content")
        if role not in SUPPORTED_CHAT_ROLES:
            return False
        if not _is_non_empty_string(content):
            return False
    return True


def detect_row_format(obj: object) -> str | None:
    if not isinstance(obj, dict):
        return None

    has_text = "text" in obj
    has_messages = "messages" in obj
    has_prompt = "prompt" in obj
    has_completion = "completion" in obj

    active_formats = int(has_text) + int(has_messages) + int(has_prompt or has_completion)
    if active_formats != 1:
        return None

    if has_text:
        return "text" if _is_non_empty_string(obj.get("text")) else None

    if has_messages:
        return "chat" if _is_valid_message_list(obj.get("messages")) else None

    if has_prompt or has_completion:
        if has_prompt and has_completion and _is_non_empty_string(obj.get("prompt")) and _is_non_empty_string(obj.get("completion")):
            return "completion"
        return None

    return None


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
        if detect_row_format(obj) is None:
            invalid += 1
    return ValidationResult(total=total, invalid=invalid)
