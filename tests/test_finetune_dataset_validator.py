import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from finetune.dataset_validator import validate_jsonl_rows
from finetune.dataset_writer import write_jsonl_stream


def test_validate_jsonl_rows_accepts_text_rows() -> None:
    rows = ['{"text": "hello"}', '{"text": "world"}']

    result = validate_jsonl_rows(rows)

    assert result.total == 2
    assert result.invalid == 0


def test_validate_jsonl_rows_flags_bad_rows() -> None:
    rows = ['{"text": "ok"}', '{"bad": 1}', "not-json", '{"text": "   "}']

    result = validate_jsonl_rows(rows)

    assert result.total == 4
    assert result.invalid == 3


def test_write_jsonl_stream_writes_text_rows(tmp_path: Path) -> None:
    out_path = tmp_path / "dataset" / "train.jsonl"

    count = write_jsonl_stream(out_path, ["hello", "world"])

    assert count == 2
    assert out_path.exists()
    assert out_path.read_text(encoding="utf-8").splitlines() == [
        '{"text": "hello"}',
        '{"text": "world"}',
    ]
