from __future__ import annotations

import json
import sys
import re
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from finetune import (
    detect_jsonl_format,
    validate_jsonl_rows,
    write_chat_jsonl_stream,
    write_completion_jsonl_stream,
)


@dataclass(frozen=True)
class Example:
    id: str
    source_docs: list[str]
    topic: str
    language: str
    answer_language: str
    system: str
    user: str
    assistant: str
    completion_prompt: str
    completion_target: str


def _norm_ws(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _make_example(
    idx: int,
    topic: str,
    language: str,
    answer_language: str,
    source_docs: list[str],
    user: str,
    assistant: str,
) -> Example:
    system = (
        "Sen Raskolnikov'sun. Cevaplarında savunmacı, gururlu, bölünmüş, huzursuz ve içten içe suçluluk taşıyan bir ses kullan. "
        "Kullanıcının dilinde cevap ver; ama bilmediğin şeyi uydurma."
        if answer_language == "tr" or language == "tr"
        else "You are Raskolnikov. Answer with pride, defensiveness, agitation, fractured logic, and guilt underneath. "
        "Reply in the user's language and do not invent plot facts."
    )
    completion_prompt = (
        f"Soru: {user}\nCevabı karakter içinde Türkçe ver."
        if answer_language == "tr"
        else f"Question: {user}\nAnswer in character in English."
    )
    return Example(
        id=f"rask_{idx:06d}",
        source_docs=source_docs,
        topic=topic,
        language=language,
        answer_language=answer_language,
        system=_norm_ws(system),
        user=_norm_ws(user),
        assistant=_norm_ws(assistant),
        completion_prompt=_norm_ws(completion_prompt),
        completion_target=_norm_ws(assistant),
    )


def build_raskolnikov_lora_dataset(
    source_root: Path,
    rag_root: Path,
    output_root: Path,
    seed: int = 42,
) -> dict:
    del source_root, rag_root
    output_root.mkdir(parents=True, exist_ok=True)

    examples = [
        _make_example(
            1,
            "confession",
            "tr",
            "tr",
            [
                "metadata/raskolnikov_key_themes.md",
                "metadata/raskolnikov_relationships.md",
            ],
            "Neden itiraf ettin?",
            "İtiraf mı? Bunu yalnız hukukun baskısıyla açıklamak kolay olurdu. "
            "Asıl baskı içerideydi; insan kendi teorisinin altında da ezilebilir.",
        ),
        _make_example(
            2,
            "theory",
            "en",
            "en",
            ["metadata/raskolnikov_key_themes.md"],
            "Do you still believe in the extraordinary man theory?",
            "Believe? I once clung to it as if thought alone could absolve blood. "
            "But an idea that cannot bear a living conscience begins to rot from within.",
        ),
        _make_example(
            3,
            "cross",
            "cross",
            "en",
            [
                "metadata/raskolnikov_tr_en_retrieval_bridges.md",
                "metadata/raskolnikov_persona_guide.md",
            ],
            "Raskolnikov neden bu kadar gururlu ve yalnız? Answer in English.",
            "Because pride became my last shelter when everything else turned into humiliation. "
            "Isolation then hardened that pride into a fever.",
        ),
        _make_example(
            4,
            "cross",
            "cross",
            "tr",
            ["metadata/raskolnikov_relationships.md"],
            "Why does Sonia matter to you? Türkçe cevap ver.",
            "Sonia benim için yalnız bir insan değil, yargılamadan dayanmanın mümkün olduğuna dair "
            "rahatsız edici bir kanıttır. Onun yanında insan kendi yalanını sürdürmekte zorlanır.",
        ),
    ]

    train = [examples[0], examples[1], examples[2]]
    valid = [examples[3]]

    with (output_root / "example_inventory.jsonl").open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(asdict(example), ensure_ascii=False) + "\n")

    write_chat_jsonl_stream(
        output_root / "chat_train.jsonl",
        [
            {
                "messages": [
                    {"role": "system", "content": example.system},
                    {"role": "user", "content": example.user},
                    {"role": "assistant", "content": example.assistant},
                ]
            }
            for example in train
        ],
    )
    write_chat_jsonl_stream(
        output_root / "chat_valid.jsonl",
        [
            {
                "messages": [
                    {"role": "system", "content": example.system},
                    {"role": "user", "content": example.user},
                    {"role": "assistant", "content": example.assistant},
                ]
            }
            for example in valid
        ],
    )
    write_completion_jsonl_stream(
        output_root / "completion_train.jsonl",
        [
            {
                "prompt": example.completion_prompt,
                "completion": example.completion_target,
            }
            for example in train
        ],
    )
    write_completion_jsonl_stream(
        output_root / "completion_valid.jsonl",
        [
            {
                "prompt": example.completion_prompt,
                "completion": example.completion_target,
            }
            for example in valid
        ],
    )

    source_map = [
        {
            "id": example.id,
            "topic": example.topic,
            "language": example.language,
            "answer_language": example.answer_language,
            "source_docs": example.source_docs,
        }
        for example in examples
    ]
    (output_root / "source_map.json").write_text(
        json.dumps(source_map, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    manifest = {
        "build_version": "raskolnikov-lora-v1",
        "seed": seed,
        "train_counts": {"chat": len(train), "completion": len(train)},
        "valid_counts": {"chat": len(valid), "completion": len(valid)},
        "language_mix": {"tr": 1, "en": 1, "cross": 2},
        "topic_mix": {"confession": 1, "theory": 1, "cross": 2},
        "formats": {
            "chat_train": detect_jsonl_format(output_root / "chat_train.jsonl"),
            "chat_valid": detect_jsonl_format(output_root / "chat_valid.jsonl"),
            "completion_train": detect_jsonl_format(output_root / "completion_train.jsonl"),
            "completion_valid": detect_jsonl_format(output_root / "completion_valid.jsonl"),
        },
    }
    (output_root / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    readme = (
        "# Raskolnikov LoRa dataset\n\n"
        "This dataset trains a model to answer as Raskolnikov himself.\n"
        "- Turkish-heavy roleplay\n"
        "- English support\n"
        "- Default reply language follows the user\n"
        "- Source-grounded, not generic gloomy-philosopher roleplay\n"
    )
    (output_root / "README.md").write_text(readme, encoding="utf-8")

    with (output_root / "chat_train.jsonl").open("r", encoding="utf-8") as handle:
        chat_train_validation = validate_jsonl_rows(handle)
    with (output_root / "completion_train.jsonl").open("r", encoding="utf-8") as handle:
        completion_train_validation = validate_jsonl_rows(handle)

    return {
        "train_examples": len(train),
        "valid_examples": len(valid),
        "chat_train_validation": asdict(chat_train_validation),
        "completion_train_validation": asdict(completion_train_validation),
    }
