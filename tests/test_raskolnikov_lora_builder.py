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

    assert result["train_examples"] > 0
    assert result["valid_examples"] > 0
    assert detect_jsonl_format(lora_root / "chat_train.jsonl") == "chat"
    assert detect_jsonl_format(lora_root / "completion_train.jsonl") == "completion"

    with (lora_root / "chat_train.jsonl").open("r", encoding="utf-8") as f:
        chat_result = validate_jsonl_rows(f)
    with (lora_root / "completion_train.jsonl").open("r", encoding="utf-8") as f:
        completion_result = validate_jsonl_rows(f)

    assert chat_result.invalid == 0
    assert completion_result.invalid == 0
