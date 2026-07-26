import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from finetune.dataset_writer import (
    detect_jsonl_format,
    write_chat_jsonl_stream,
    write_completion_jsonl_stream,
)


def test_write_chat_jsonl_stream_preserves_utf8_and_schema(tmp_path: Path) -> None:
    out_path = tmp_path / "dataset" / "chat_train.jsonl"
    rows = [
        {
            "messages": [
                {"role": "system", "content": "Victor Hugo gibi ama dürüst konuş."},
                {"role": "user", "content": "Sefiller ne anlatır?"},
                {"role": "assistant", "content": "Adalet, merhamet ve toplumsal sefalet üzerine büyük bir romandır."},
            ]
        }
    ]

    count = write_chat_jsonl_stream(out_path, rows)

    assert count == 1
    line = out_path.read_text(encoding="utf-8").strip()
    obj = json.loads(line)
    assert detect_jsonl_format(out_path) == "chat"
    assert "messages" in obj
    assert obj["messages"][0]["content"].startswith("Victor Hugo gibi")
    assert "Sefiller" in obj["messages"][1]["content"]


def test_write_completion_jsonl_stream_preserves_schema(tmp_path: Path) -> None:
    out_path = tmp_path / "dataset" / "completion_valid.jsonl"
    rows = [
        {
            "prompt": "Victor Hugo neden sürgüne gitti?",
            "completion": "Louis-Napoléon darbesine karşı çıktığı için sürgüne gitti.",
        }
    ]

    count = write_completion_jsonl_stream(out_path, rows)

    assert count == 1
    line = out_path.read_text(encoding="utf-8").strip()
    obj = json.loads(line)
    assert detect_jsonl_format(out_path) == "completion"
    assert obj["prompt"].startswith("Victor Hugo")
    assert "sürgüne" in obj["completion"]
