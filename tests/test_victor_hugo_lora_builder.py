import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from finetune import detect_jsonl_format, validate_jsonl_rows
from tools.build_victor_hugo_lora_dataset import build_victor_hugo_lora_dataset


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_build_victor_hugo_lora_dataset_generates_outputs(tmp_path: Path) -> None:
    rag_root = tmp_path / "RAG"
    output_root = tmp_path / "LoRa"

    _write(
        rag_root / "metadata" / "victor_hugo_timeline.json",
        json.dumps(
            [
                {"year": 1802, "date": "26 February 1802", "event": "Born in Besançon."},
                {"year": 1851, "date": "1851", "event": "Entered exile after opposing the coup."},
            ],
            ensure_ascii=False,
        ),
    )
    _write(rag_root / "metadata" / "victor_hugo_biography.md", "# Bio\nVictor Hugo was a French writer.")
    _write(rag_root / "metadata" / "victor_hugo_tr_en_glossary.md", "- `sürgün` ↔ `exile`\n- `adalet` ↔ `justice`\n")
    _write(rag_root / "metadata" / "victor_hugo_tr_en_title_aliases.md", "# aliases\n")
    _write(rag_root / "metadata" / "victor_hugo_tr_en_retrieval_bridges.md", "# bridges\n")
    _write(rag_root / "metadata" / "key_texts_and_retrieval_notes.md", "# Notes\nLes Misérables and Notre-Dame de Paris.")
    _write(rag_root / "metadata" / "victor_hugo_persona_guide.md", "# Persona\nMoral gravity.")
    _write(rag_root / "metadata" / "in_defense_of_his_son_notes.md", "# Notes\nAgainst the death penalty.")
    _write(rag_root / "metadata" / "letter_john_brown_notes.md", "# Notes\nAgainst slavery.")
    _write(rag_root / "metadata" / "address_marseille_notes.md", "# Notes\nLabor and the republic.")

    _write(
        rag_root / "works" / "complete_bibliography.json",
        json.dumps(
            {
                "novels": [
                    {"title": "Les Misérables", "year": 1862},
                    {"title": "Notre-Dame de Paris", "year": 1831, "alt": "The Hunchback of Notre-Dame"},
                ]
            },
            ensure_ascii=False,
        ),
    )
    _write(rag_root / "works" / "complete_bibliography.md", "# Works\n- Les Misérables\n")

    result = build_victor_hugo_lora_dataset(rag_root=rag_root, output_root=output_root, seed=7)

    assert result["train_examples"] > 0
    assert result["valid_examples"] > 0

    chat_train = output_root / "chat_train.jsonl"
    completion_train = output_root / "completion_train.jsonl"
    manifest_path = output_root / "dataset_manifest.json"
    source_map_path = output_root / "source_map.json"

    assert chat_train.exists()
    assert completion_train.exists()
    assert manifest_path.exists()
    assert source_map_path.exists()

    assert detect_jsonl_format(chat_train) == "chat"
    assert detect_jsonl_format(completion_train) == "completion"

    with chat_train.open("r", encoding="utf-8") as handle:
        chat_validation = validate_jsonl_rows(handle)
    with completion_train.open("r", encoding="utf-8") as handle:
        completion_validation = validate_jsonl_rows(handle)

    assert chat_validation.invalid == 0
    assert completion_validation.invalid == 0

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["formats"]["chat_train"] == "chat"
    assert manifest["formats"]["completion_train"] == "completion"
    assert "tr" in manifest["language_mix"]
    assert "timeline" in manifest["topic_mix"]
