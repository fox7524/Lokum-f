import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from finetune.job_preflight import preflight_training
from finetune_engine import FinetuneEngine


def test_preflight_rejects_missing_model(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "train.jsonl").write_text('{"text":"x"}\n', encoding="utf-8")
    (dataset_dir / "valid.jsonl").write_text('{"text":"y"}\n', encoding="utf-8")

    ok, message = preflight_training("/missing/model", dataset_dir)

    assert ok is False
    assert "model" in message.lower()


def test_preflight_accepts_valid_setup(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    dataset_dir = tmp_path / "dataset"
    model_dir.mkdir()
    dataset_dir.mkdir()
    (dataset_dir / "train.jsonl").write_text('{"text":"x"}\n', encoding="utf-8")
    (dataset_dir / "valid.jsonl").write_text('{"text":"y"}\n', encoding="utf-8")

    ok, message = preflight_training(str(model_dir), dataset_dir)

    assert ok is True
    assert message == "ok"


def test_start_training_surfaces_preflight_failure_and_skips_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "train.jsonl").write_text('{"text":"x"}\n', encoding="utf-8")
    (dataset_dir / "valid.jsonl").write_text('{"text":"y"}\n', encoding="utf-8")

    engine = FinetuneEngine("/missing/model")
    popen_calls: list[list[str]] = []

    def fake_popen(*args, **kwargs) -> subprocess.Popen:
        popen_calls.append(list(args[0]))
        raise AssertionError("subprocess should not be launched when preflight fails")

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    with pytest.raises(RuntimeError, match="Model path does not exist"):
        engine.start_training(dataset_path=str(dataset_dir))

    assert popen_calls == []
