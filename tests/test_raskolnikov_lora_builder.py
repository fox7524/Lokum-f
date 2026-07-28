import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from finetune import detect_jsonl_format, validate_jsonl_rows
from tools.build_raskolnikov_lora_dataset import build_raskolnikov_lora_dataset


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_build_raskolnikov_lora_dataset_generates_outputs(tmp_path: Path) -> None:
    source_root = tmp_path / "source_root"
    rag_root = tmp_path / "Big_DATA" / "Raskolnikov" / "RAG"
    lora_root = tmp_path / "Big_DATA" / "Raskolnikov" / "LoRa"

    _write(
        source_root / "RATA" / "suc_ve_ceza_notes.txt",
        "Raskolnikov is feverish, proud, guilty, and split against himself.",
    )
    _write(
        rag_root / "metadata" / "raskolnikov_tr_en_glossary.md",
        "- `suç` ↔ `crime`\n- `ceza` ↔ `punishment`\n- `suçluluk` ↔ `guilt`\n",
    )
    _write(
        rag_root / "metadata" / "raskolnikov_persona_guide.md",
        "# Persona\nRaskolnikov is proud, unstable, analytical, and guilt-ridden.",
    )
    _write(
        rag_root / "metadata" / "raskolnikov_key_themes.md",
        "# Themes\nguilt, alienation, extraordinary man theory, confession",
    )
    _write(
        rag_root / "metadata" / "raskolnikov_relationships.md",
        "# Relationships\nSonia, Razumikhin, Dunya, Porfiry",
    )
    _write(
        rag_root / "metadata" / "raskolnikov_tr_en_title_aliases.md",
        "# Aliases\n- `Suç ve Ceza` ↔ `Crime and Punishment`\n",
    )
    _write(
        rag_root / "metadata" / "raskolnikov_tr_en_retrieval_bridges.md",
        "# Bridges\n- `Raskolnikov neden itiraf eder?` ↔ `Why does Raskolnikov confess?`\n",
    )
    _write(
        rag_root / "metadata" / "raskolnikov_character_profile.md",
        "# Profile\nHe is divided between theory, pride, fear, and guilt.",
    )
    _write(
        rag_root / "metadata" / "raskolnikov_timeline.json",
        json.dumps(
            [
                {"order": 1, "event": "Raskolnikov commits the murder."},
                {"order": 2, "event": "He moves through fever, fear, and suspicion."},
                {"order": 3, "event": "He confesses."},
            ],
            ensure_ascii=False,
        ),
    )
    _write(
        rag_root / "sources" / "source_manifest.json",
        json.dumps(
            [{"path": "RATA/suc_ve_ceza_notes.txt", "kind": "primary_or_notes", "language": "tr"}],
            ensure_ascii=False,
        ),
    )

    result = build_raskolnikov_lora_dataset(
        source_root=source_root,
        rag_root=rag_root,
        output_root=lora_root,
        seed=11,
    )

    assert result["train_examples"] >= 12
    assert result["valid_examples"] > 0
    assert detect_jsonl_format(lora_root / "train.jsonl") == "chat"
    assert detect_jsonl_format(lora_root / "valid.jsonl") == "chat"
    assert not (lora_root / "chat_train.jsonl").exists()
    assert not (lora_root / "chat_valid.jsonl").exists()
    assert not (lora_root / "completion_train.jsonl").exists()
    assert not (lora_root / "completion_valid.jsonl").exists()

    with (lora_root / "train.jsonl").open("r", encoding="utf-8") as f:
        train_result = validate_jsonl_rows(f)
    with (lora_root / "valid.jsonl").open("r", encoding="utf-8") as f:
        valid_result = validate_jsonl_rows(f)

    assert train_result.invalid == 0
    assert valid_result.invalid == 0

    manifest = json.loads((lora_root / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert manifest["formats"]["train"] == "chat"
    assert manifest["formats"]["valid"] == "chat"
    assert manifest["train_examples"] == result["train_examples"]
    assert manifest["valid_examples"] == result["valid_examples"]
