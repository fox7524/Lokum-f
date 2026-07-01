from __future__ import annotations

from pathlib import Path


def preflight_training(model_path: str, dataset_dir: Path) -> tuple[bool, str]:
    model_dir = Path(model_path)
    if not model_dir.exists():
        return False, "Model path does not exist"

    train_path = dataset_dir / "train.jsonl"
    valid_path = dataset_dir / "valid.jsonl"
    if not train_path.exists() or not valid_path.exists():
        return False, "Dataset directory must contain train.jsonl and valid.jsonl"
    if train_path.stat().st_size == 0 or valid_path.stat().st_size == 0:
        return False, "Dataset files must be non-empty"

    return True, "ok"
