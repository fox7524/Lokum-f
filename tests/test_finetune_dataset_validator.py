import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from finetune.dataset_validator import detect_row_format, validate_jsonl_rows
from finetune.dataset_writer import write_text_jsonl_stream


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


def test_validate_jsonl_rows_accepts_chat_rows() -> None:
    rows = [
        '{"messages":[{"role":"system","content":"You are Victor Hugo."},{"role":"user","content":"Sefiller ne anlatır?"},{"role":"assistant","content":"Toplumsal sefalet ve merhameti anlatır."}]}'
    ]

    result = validate_jsonl_rows(rows)

    assert result.total == 1
    assert result.invalid == 0


def test_validate_jsonl_rows_accepts_completion_rows() -> None:
    rows = [
        '{"prompt":"Victor Hugo neden sürgüne gitti?","completion":"Louis-Napoléon darbesine karşı çıktığı için sürgüne gitti."}'
    ]

    result = validate_jsonl_rows(rows)

    assert result.total == 1
    assert result.invalid == 0


def test_validate_jsonl_rows_rejects_mixed_schema_rows() -> None:
    rows = [
        '{"text":"x","messages":[{"role":"user","content":"x"}]}',
        '{"messages":[{"role":"user","content":"x"}],"completion":"y"}',
    ]

    result = validate_jsonl_rows(rows)

    assert result.total == 2
    assert result.invalid == 2


def test_validate_jsonl_rows_rejects_bad_chat_roles_and_empty_content() -> None:
    rows = [
        '{"messages":[{"role":"narrator","content":"x"}]}',
        '{"messages":[{"role":"user","content":"   "}]}',
    ]

    result = validate_jsonl_rows(rows)

    assert result.total == 2
    assert result.invalid == 2


def test_detect_row_format_identifies_supported_formats() -> None:
    assert detect_row_format({"text": "hello"}) == "text"
    assert detect_row_format({"messages": [{"role": "user", "content": "Merhaba"}]}) == "chat"
    assert detect_row_format({"prompt": "Q", "completion": "A"}) == "completion"
    assert detect_row_format({"text": "x", "completion": "y"}) is None


def test_write_text_jsonl_stream_writes_text_rows(tmp_path: Path) -> None:
    out_path = tmp_path / "dataset" / "train.jsonl"

    count = write_text_jsonl_stream(out_path, ["hello", "world"])

    assert count == 2
    assert out_path.exists()
    assert out_path.read_text(encoding="utf-8").splitlines() == [
        '{"text": "hello"}',
        '{"text": "world"}',
    ]
