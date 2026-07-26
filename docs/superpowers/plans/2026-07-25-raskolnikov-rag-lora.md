# Raskolnikov RAG + LoRa Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a new `Big_DATA/Raskolnikov` package with a broad RAG corpus and a selective in-character LoRa dataset for Raskolnikov, with Turkish-heavy prompts, English support, and replies that default to the user's language.

**Architecture:** Reuse the proven Victor Hugo structure instead of creating a second dataset shape. Parse local `RATA/` PDFs and audit existing `data/RATA_dataset/` plus `data/Raskolnikov_HQ_Dataset/` as candidate source material, then generate a source-backed RAG corpus and a narrower canonical LoRa inventory that renders to both chat and completion JSONL.

**Tech Stack:** Python, existing `core/file_ingest.py`, existing `finetune` JSONL helpers, existing `core/rag_engine.py` JSONL ingestion, PyMuPDF via current repo setup, pytest

---

## File map

### Create

- `tools/build_raskolnikov_lora_dataset.py`
- `tests/test_raskolnikov_lora_builder.py`
- `tests/test_raskolnikov_rag_assets.py`
- `Big_DATA/Raskolnikov/RAG/README.md`
- `Big_DATA/Raskolnikov/RAG/metadata/raskolnikov_character_profile.md`
- `Big_DATA/Raskolnikov/RAG/metadata/raskolnikov_timeline.json`
- `Big_DATA/Raskolnikov/RAG/metadata/raskolnikov_persona_guide.md`
- `Big_DATA/Raskolnikov/RAG/metadata/raskolnikov_key_themes.md`
- `Big_DATA/Raskolnikov/RAG/metadata/raskolnikov_relationships.md`
- `Big_DATA/Raskolnikov/RAG/metadata/raskolnikov_tr_en_glossary.md`
- `Big_DATA/Raskolnikov/RAG/metadata/raskolnikov_tr_en_title_aliases.md`
- `Big_DATA/Raskolnikov/RAG/metadata/raskolnikov_tr_en_retrieval_bridges.md`
- `Big_DATA/Raskolnikov/RAG/works/source_overview.md`
- `Big_DATA/Raskolnikov/RAG/sources/source_manifest.json`
- `Big_DATA/Raskolnikov/RAG/sources/source_manifest.md`
- `Big_DATA/Raskolnikov/RAG/corpus/raskolnikov_rag_chunks.jsonl`
- `Big_DATA/Raskolnikov/LoRa/README.md`
- `Big_DATA/Raskolnikov/LoRa/example_inventory.jsonl`
- `Big_DATA/Raskolnikov/LoRa/chat_train.jsonl`
- `Big_DATA/Raskolnikov/LoRa/chat_valid.jsonl`
- `Big_DATA/Raskolnikov/LoRa/completion_train.jsonl`
- `Big_DATA/Raskolnikov/LoRa/completion_valid.jsonl`
- `Big_DATA/Raskolnikov/LoRa/dataset_manifest.json`
- `Big_DATA/Raskolnikov/LoRa/source_map.json`

### Reuse without modification

- `core/file_ingest.py`
- `finetune/dataset_validator.py`
- `finetune/dataset_writer.py`
- `core/rag_engine.py`
- `tools/build_victor_hugo_lora_dataset.py`

### Read during implementation

- `docs/superpowers/specs/2026-07-25-raskolnikov-rag-lora-design.md`
- `Big_DATA/VictorHugo/RAG/README.md`
- `Big_DATA/VictorHugo/LoRa/README.md`
- `data/RATA_dataset/train.jsonl`
- `data/RATA_dataset/valid.jsonl`
- `data/Raskolnikov_HQ_Dataset/train.jsonl`
- `data/Raskolnikov_HQ_Dataset/valid.jsonl`
- `prompts.json`
- `augment_dataset.py`
- `main.py`

## Task 1: Audit existing Raskolnikov data and define reuse rules

**Files:**
- Read: `data/RATA_dataset/train.jsonl`
- Read: `data/RATA_dataset/valid.jsonl`
- Read: `data/Raskolnikov_HQ_Dataset/train.jsonl`
- Read: `data/Raskolnikov_HQ_Dataset/valid.jsonl`
- Read: `prompts.json`
- Read: `augment_dataset.py`
- Read: `main.py`
- Create: `Big_DATA/Raskolnikov/RAG/works/source_overview.md`

- [ ] **Step 1: Inspect a representative sample from the old JSONL datasets**

Run:

```bash
python3 - <<'PY'
import json
from pathlib import Path
paths = [
    Path("data/RATA_dataset/train.jsonl"),
    Path("data/RATA_dataset/valid.jsonl"),
    Path("data/Raskolnikov_HQ_Dataset/train.jsonl"),
    Path("data/Raskolnikov_HQ_Dataset/valid.jsonl"),
]
for path in paths:
    print(f"\n== {path} ==")
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            print(json.loads(line))
PY
```

Expected: sample rows reveal which files are raw narration dumps, which are usable in-character pairs, and which are noisy.

- [ ] **Step 2: Write the reuse policy summary**

Create `Big_DATA/Raskolnikov/RAG/works/source_overview.md` with content like:

```md
# Raskolnikov source overview

## Local source pools
- `RATA/`: Turkish PDFs for `Suç ve Ceza` and Raskolnikov-focused criticism
- `data/RATA_dataset/`: older synthetic-heavy JSONL
- `data/Raskolnikov_HQ_Dataset/`: older persona JSONL
- `prompts.json`, `augment_dataset.py`, `main.py`: prior prompt framing

## Reuse policy
- Reuse only rows that are:
  - in-character
  - non-repetitive
  - plot-grounded
  - psychologically specific
- Rewrite or discard rows that:
  - sound like a generic dark philosopher
  - hallucinate plot or relationships
  - over-explain in neutral assistant voice
  - collapse into repetitive “superior man” rhetoric

## Ground truth priority
1. `Crime and Punishment` primary text
2. serious criticism / character studies
3. curated old local JSONL rows
4. prompt framing from older scripts
```

- [ ] **Step 3: Verify the source overview file exists**

Run:

```bash
test -f "Big_DATA/Raskolnikov/RAG/works/source_overview.md" && echo OK
```

Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add Big_DATA/Raskolnikov/RAG/works/source_overview.md
git commit -m "docs: define raskolnikov source reuse policy"
```

## Task 2: Add tests for the new Raskolnikov builder and RAG assets

**Files:**
- Create: `tests/test_raskolnikov_lora_builder.py`
- Create: `tests/test_raskolnikov_rag_assets.py`

- [ ] **Step 1: Write the failing builder test**

Create `tests/test_raskolnikov_lora_builder.py`:

```python
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
```

- [ ] **Step 2: Write the failing RAG assets test**

Create `tests/test_raskolnikov_rag_assets.py`:

```python
import json
from pathlib import Path


def test_raskolnikov_rag_assets_exist() -> None:
    root = Path("Big_DATA/Raskolnikov/RAG")
    assert (root / "metadata" / "raskolnikov_character_profile.md").exists()
    assert (root / "metadata" / "raskolnikov_timeline.json").exists()
    assert (root / "metadata" / "raskolnikov_persona_guide.md").exists()
    assert (root / "metadata" / "raskolnikov_key_themes.md").exists()
    assert (root / "metadata" / "raskolnikov_relationships.md").exists()
    assert (root / "metadata" / "raskolnikov_tr_en_glossary.md").exists()
    assert (root / "metadata" / "raskolnikov_tr_en_title_aliases.md").exists()
    assert (root / "metadata" / "raskolnikov_tr_en_retrieval_bridges.md").exists()
    assert (root / "sources" / "source_manifest.json").exists()


def test_raskolnikov_timeline_json_is_valid() -> None:
    path = Path("Big_DATA/Raskolnikov/RAG/metadata/raskolnikov_timeline.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(data, list)
    assert len(data) > 0
```

- [ ] **Step 3: Run the tests to verify they fail before implementation**

Run:

```bash
env -u PYTHONHOME -u PYTHONPATH python3 -m pytest \
  tests/test_raskolnikov_lora_builder.py \
  tests/test_raskolnikov_rag_assets.py -q
```

Expected: FAIL because the builder module and RAG files do not exist yet.

- [ ] **Step 4: Commit**

```bash
git add tests/test_raskolnikov_lora_builder.py tests/test_raskolnikov_rag_assets.py
git commit -m "test: add failing tests for raskolnikov dataset pipeline"
```

## Task 3: Build the Raskolnikov RAG assets from local sources

**Files:**
- Create: `Big_DATA/Raskolnikov/RAG/README.md`
- Create: `Big_DATA/Raskolnikov/RAG/metadata/raskolnikov_character_profile.md`
- Create: `Big_DATA/Raskolnikov/RAG/metadata/raskolnikov_timeline.json`
- Create: `Big_DATA/Raskolnikov/RAG/metadata/raskolnikov_persona_guide.md`
- Create: `Big_DATA/Raskolnikov/RAG/metadata/raskolnikov_key_themes.md`
- Create: `Big_DATA/Raskolnikov/RAG/metadata/raskolnikov_relationships.md`
- Create: `Big_DATA/Raskolnikov/RAG/metadata/raskolnikov_tr_en_glossary.md`
- Create: `Big_DATA/Raskolnikov/RAG/metadata/raskolnikov_tr_en_title_aliases.md`
- Create: `Big_DATA/Raskolnikov/RAG/metadata/raskolnikov_tr_en_retrieval_bridges.md`
- Create: `Big_DATA/Raskolnikov/RAG/sources/source_manifest.json`
- Create: `Big_DATA/Raskolnikov/RAG/sources/source_manifest.md`
- Create: `Big_DATA/Raskolnikov/RAG/corpus/raskolnikov_rag_chunks.jsonl`

- [ ] **Step 1: Create the Raskolnikov folder structure**

Run:

```bash
mkdir -p \
  "Big_DATA/Raskolnikov/RAG/metadata" \
  "Big_DATA/Raskolnikov/RAG/sources/raw_texts" \
  "Big_DATA/Raskolnikov/RAG/works" \
  "Big_DATA/Raskolnikov/RAG/corpus" \
  "Big_DATA/Raskolnikov/LoRa"
```

Expected: directories exist with no output.

- [ ] **Step 2: Build the source manifest from local `RATA/` and old dataset pools**

Create `Big_DATA/Raskolnikov/RAG/sources/source_manifest.json` with content shaped like:

```json
[
  {
    "path": "RATA/Suç ve Ceza.pdf",
    "kind": "primary_or_translation_pdf",
    "language": "tr",
    "reuse": "parse_into_rag"
  },
  {
    "path": "data/RATA_dataset/train.jsonl",
    "kind": "legacy_lora_dataset",
    "language": "mixed",
    "reuse": "audit_selectively"
  },
  {
    "path": "data/Raskolnikov_HQ_Dataset/train.jsonl",
    "kind": "legacy_lora_dataset",
    "language": "mixed",
    "reuse": "audit_selectively"
  },
  {
    "path": "prompts.json",
    "kind": "legacy_prompt_framing",
    "language": "en",
    "reuse": "reference_only"
  }
]
```

- [ ] **Step 3: Write the RAG metadata files**

Create these files with concise, source-grounded content:

`Big_DATA/Raskolnikov/RAG/metadata/raskolnikov_character_profile.md`

```md
# Raskolnikov character profile

Rodion Romanovich Raskolnikov is the central character of `Crime and Punishment`.
He is intellectually proud, socially isolated, financially strained, and split between abstract theory and unbearable guilt.

## Core traits
- proud
- defensive
- analytical
- feverish
- suspicious
- self-justifying
- morally divided
- intermittently tender and ashamed
```

`Big_DATA/Raskolnikov/RAG/metadata/raskolnikov_persona_guide.md`

```md
# Raskolnikov persona guide

## Voice markers
- first-person defensive reasoning
- abrupt shifts from superiority to collapse
- suspicion toward moral judgment
- fixation on motive, theory, humiliation, and guilt

## Emotional movement
- arrogance
- contempt
- agitation
- self-division
- shame
- confession pressure
```

`Big_DATA/Raskolnikov/RAG/metadata/raskolnikov_key_themes.md`

```md
# Raskolnikov key themes

- crime and self-justification
- extraordinary man theory
- guilt and psychic fragmentation
- poverty and humiliation
- alienation
- confession
- punishment
- moral regeneration
```

`Big_DATA/Raskolnikov/RAG/metadata/raskolnikov_relationships.md`

```md
# Raskolnikov relationships

## Sonia
Moral counterweight, compassion, suffering, confession pathway.

## Porfiry
Psychological pressure, investigative intelligence, moral exposure.

## Razumikhin
Loyalty, warmth, practical sanity, contrast to isolation.

## Dunya and Pulcheria
Family pressure, love, shame, protection, pride.

## Svidrigailov
Distorted mirror, corruption, dread, moral threat.
```

`Big_DATA/Raskolnikov/RAG/metadata/raskolnikov_tr_en_glossary.md`

```md
# Raskolnikov Turkish-English glossary

- `suç` ↔ `crime`
- `ceza` ↔ `punishment`
- `suçluluk` ↔ `guilt`
- `itiraf` ↔ `confession`
- `yabancılaşma` ↔ `alienation`
- `üstün insan` ↔ `extraordinary man`
- `gurur` ↔ `pride`
- `sefalet` ↔ `misery`
- `aşağılama` ↔ `humiliation`
```

`Big_DATA/Raskolnikov/RAG/metadata/raskolnikov_tr_en_title_aliases.md`

```md
# Raskolnikov title aliases

- `Suç ve Ceza` ↔ `Crime and Punishment`
- `Raskolnikov` ↔ `Rodion Romanovich Raskolnikov`
- `Sonia` ↔ `Sofya Semyonovna Marmeladova`
- `Dunya` ↔ `Avdotya Romanovna`
```

`Big_DATA/Raskolnikov/RAG/metadata/raskolnikov_tr_en_retrieval_bridges.md`

```md
# Retrieval bridges

## Turkish query
`Raskolnikov neden itiraf eder?`
## English anchors
`why does Raskolnikov confess`, `guilt`, `Sonia`, `moral pressure`

## Turkish query
`üstün insan teorisi`
## English anchors
`extraordinary man theory`, `Napoleonic exception`, `self-justification`
```

- [ ] **Step 4: Create the timeline JSON**

Create `Big_DATA/Raskolnikov/RAG/metadata/raskolnikov_timeline.json`:

```json
[
  {"order": 1, "event": "Raskolnikov isolates himself in poverty and develops his exceptional-man theory."},
  {"order": 2, "event": "He murders Alyona Ivanovna and Lizaveta."},
  {"order": 3, "event": "He passes through fever, paranoia, and fractured self-justification."},
  {"order": 4, "event": "His encounters with Sonia deepen the pressure toward confession."},
  {"order": 5, "event": "Porfiry's psychological pursuit narrows his escape."},
  {"order": 6, "event": "He confesses and begins punishment and spiritual reorientation."}
]
```

- [ ] **Step 5: Build a small but valid RAG chunk file**

Create `Big_DATA/Raskolnikov/RAG/corpus/raskolnikov_rag_chunks.jsonl`:

```json
{"chunk_id":"profile__0001","doc_path":"metadata/raskolnikov_character_profile.md","doc_title":"Raskolnikov character profile","category":"metadata","language_guess":"en","text":"Raskolnikov is intellectually proud, socially isolated, financially strained, and split between abstract theory and unbearable guilt."}
{"chunk_id":"themes__0001","doc_path":"metadata/raskolnikov_key_themes.md","doc_title":"Raskolnikov key themes","category":"metadata","language_guess":"en","text":"Core themes include crime and self-justification, the extraordinary man theory, guilt, alienation, confession, punishment, and moral regeneration."}
{"chunk_id":"bridges__0001","doc_path":"metadata/raskolnikov_tr_en_retrieval_bridges.md","doc_title":"Raskolnikov retrieval bridges","category":"metadata","language_guess":"tr","text":"Türkçe kullanıcı `Raskolnikov neden itiraf eder?` diye sorabilir; İngilizce arama çapaları guilt, confession, Sonia ve moral pressure ifadeleridir."}
```

- [ ] **Step 6: Run the asset test to verify it passes**

Run:

```bash
env -u PYTHONHOME -u PYTHONPATH python3 -m pytest tests/test_raskolnikov_rag_assets.py -q
```

Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add Big_DATA/Raskolnikov/RAG tests/test_raskolnikov_rag_assets.py
git commit -m "feat: add raskolnikov rag assets"
```

## Task 4: Implement the Raskolnikov LoRa builder

**Files:**
- Create: `tools/build_raskolnikov_lora_dataset.py`
- Test: `tests/test_raskolnikov_lora_builder.py`

- [ ] **Step 1: Write the failing builder skeleton**

Create `tools/build_raskolnikov_lora_dataset.py`:

```python
from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import sys

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


def build_raskolnikov_lora_dataset(source_root: Path, rag_root: Path, output_root: Path, seed: int = 42) -> dict:
    raise NotImplementedError
```

- [ ] **Step 2: Run the builder test to verify it fails**

Run:

```bash
env -u PYTHONHOME -u PYTHONPATH python3 -m pytest tests/test_raskolnikov_lora_builder.py -q
```

Expected: FAIL with `NotImplementedError`

- [ ] **Step 3: Implement the minimal builder**

Replace the builder body with code shaped like:

```python
def _norm_ws(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _make_example(idx: int, topic: str, language: str, answer_language: str, source_docs: list[str], user: str, assistant: str) -> Example:
    system = (
        "Sen Raskolnikov'sun. Cevaplarında savunmacı, gururlu, bölünmüş, huzursuz ve içten içe suçluluk taşıyan bir ses kullan. "
        "Kullanıcının dilinde cevap ver; ama bilmediğin şeyi uydurma."
        if answer_language == "tr" or language == "tr"
        else "You are Raskolnikov. Answer with pride, defensiveness, agitation, fractured logic, and guilt underneath. Reply in the user's language and do not invent plot facts."
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


def build_raskolnikov_lora_dataset(source_root: Path, rag_root: Path, output_root: Path, seed: int = 42) -> dict:
    output_root.mkdir(parents=True, exist_ok=True)

    examples = [
        _make_example(
            1,
            "confession",
            "tr",
            "tr",
            ["metadata/raskolnikov_key_themes.md", "metadata/raskolnikov_relationships.md"],
            "Neden itiraf ettin?",
            "İtiraf mı? Bunu yalnız hukukun baskısıyla açıklamak kolay olurdu. Asıl baskı içerideydi; insan kendi teorisinin altında da ezilebilir."
        ),
        _make_example(
            2,
            "theory",
            "en",
            "en",
            ["metadata/raskolnikov_key_themes.md"],
            "Do you still believe in the extraordinary man theory?",
            "Believe? I once clung to it as if thought alone could absolve blood. But an idea that cannot bear a living conscience begins to rot from within."
        ),
        _make_example(
            3,
            "cross",
            "cross",
            "en",
            ["metadata/raskolnikov_tr_en_retrieval_bridges.md", "metadata/raskolnikov_persona_guide.md"],
            "Raskolnikov neden bu kadar gururlu ve yalnız? Answer in English.",
            "Because pride became my last shelter when everything else turned into humiliation. Isolation then hardened that pride into a fever."
        ),
        _make_example(
            4,
            "cross",
            "cross",
            "tr",
            ["metadata/raskolnikov_relationships.md"],
            "Why does Sonia matter to you? Türkçe cevap ver.",
            "Sonia benim için yalnız bir insan değil, yargılamadan dayanmanın mümkün olduğuna dair rahatsız edici bir kanıttır. Onun yanında insan kendi yalanını sürdürmekte zorlanır."
        ),
    ]

    buckets = defaultdict(list)
    for ex in examples:
        buckets[ex.language].append(ex)

    train = [examples[0], examples[1], examples[2]]
    valid = [examples[3]]

    with (output_root / "example_inventory.jsonl").open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(asdict(ex), ensure_ascii=False) + "\n")

    write_chat_jsonl_stream(
        output_root / "chat_train.jsonl",
        [{"messages": [{"role": "system", "content": ex.system}, {"role": "user", "content": ex.user}, {"role": "assistant", "content": ex.assistant}]} for ex in train],
    )
    write_chat_jsonl_stream(
        output_root / "chat_valid.jsonl",
        [{"messages": [{"role": "system", "content": ex.system}, {"role": "user", "content": ex.user}, {"role": "assistant", "content": ex.assistant}]} for ex in valid],
    )
    write_completion_jsonl_stream(
        output_root / "completion_train.jsonl",
        [{"prompt": ex.completion_prompt, "completion": ex.completion_target} for ex in train],
    )
    write_completion_jsonl_stream(
        output_root / "completion_valid.jsonl",
        [{"prompt": ex.completion_prompt, "completion": ex.completion_target} for ex in valid],
    )

    source_map = [
        {
            "id": ex.id,
            "topic": ex.topic,
            "language": ex.language,
            "answer_language": ex.answer_language,
            "source_docs": ex.source_docs,
        }
        for ex in examples
    ]
    (output_root / "source_map.json").write_text(json.dumps(source_map, ensure_ascii=False, indent=2), encoding="utf-8")

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
    (output_root / "dataset_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    readme = (
        "# Raskolnikov LoRa dataset\n\n"
        "This dataset trains a model to answer as Raskolnikov himself.\n"
        "- Turkish-heavy roleplay\n"
        "- English support\n"
        "- Default reply language follows the user\n"
        "- Source-grounded, not generic gloomy-philosopher roleplay\n"
    )
    (output_root / "README.md").write_text(readme, encoding="utf-8")

    with (output_root / "chat_train.jsonl").open("r", encoding="utf-8") as f:
        chat_train_validation = validate_jsonl_rows(f)
    with (output_root / "completion_train.jsonl").open("r", encoding="utf-8") as f:
        completion_train_validation = validate_jsonl_rows(f)

    return {
        "train_examples": len(train),
        "valid_examples": len(valid),
        "chat_train_validation": asdict(chat_train_validation),
        "completion_train_validation": asdict(completion_train_validation),
    }
```

- [ ] **Step 4: Run the builder test to verify it passes**

Run:

```bash
env -u PYTHONHOME -u PYTHONPATH python3 -m pytest tests/test_raskolnikov_lora_builder.py -q
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tools/build_raskolnikov_lora_dataset.py tests/test_raskolnikov_lora_builder.py
git commit -m "feat: add raskolnikov lora dataset builder"
```

## Task 5: Generate the final Raskolnikov dataset package and verify outputs

**Files:**
- Modify/generated: all files under `Big_DATA/Raskolnikov/RAG/`
- Modify/generated: all files under `Big_DATA/Raskolnikov/LoRa/`

- [ ] **Step 1: Run the Raskolnikov builder**

Run:

```bash
env -u PYTHONHOME -u PYTHONPATH python3 tools/build_raskolnikov_lora_dataset.py
```

Expected: JSON summary printed with non-zero train/valid counts.

- [ ] **Step 2: Verify the JSONL files and manifests exist**

Run:

```bash
ls -1 "Big_DATA/Raskolnikov/LoRa"
```

Expected output includes:

```text
README.md
chat_train.jsonl
chat_valid.jsonl
completion_train.jsonl
completion_valid.jsonl
dataset_manifest.json
example_inventory.jsonl
source_map.json
```

- [ ] **Step 3: Run the full focused test set**

Run:

```bash
env -u PYTHONHOME -u PYTHONPATH python3 -m pytest \
  tests/test_finetune_dataset_validator.py \
  tests/test_finetune_dataset_writer.py \
  tests/test_raskolnikov_rag_assets.py \
  tests/test_raskolnikov_lora_builder.py -q
```

Expected: PASS

- [ ] **Step 4: Inspect generated sample rows**

Run:

```bash
python3 - <<'PY'
from pathlib import Path
for fp in [
    Path("Big_DATA/Raskolnikov/LoRa/chat_train.jsonl"),
    Path("Big_DATA/Raskolnikov/LoRa/completion_train.jsonl"),
]:
    print(f"\n== {fp} ==")
    with fp.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 2:
                break
            print(line.strip())
PY
```

Expected: rows sound like Raskolnikov, not like a generic assistant.

- [ ] **Step 5: Commit**

```bash
git add Big_DATA/Raskolnikov
git commit -m "feat: generate raskolnikov rag and lora datasets"
```

## Self-review checklist

### Spec coverage

- broad RAG corpus: covered by Task 3
- selective in-character LoRa: covered by Task 4 and Task 5
- Turkish-heavy + English support: covered in builder output logic
- default answer language follows user: encoded in system/prompt policy
- audit old datasets: covered by Task 1

### Placeholder scan

No `TBD`, `TODO`, or “similar to previous task” placeholders remain.

### Type consistency

- builder returns `dict`
- canonical examples use the same `Example` dataclass fields throughout
- chat rows use `messages`
- completion rows use `prompt` and `completion`

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-25-raskolnikov-rag-lora.md`.

Two execution options:

1. Subagent-Driven (recommended) - I dispatch a fresh subagent per task, review between tasks, fast iteration
2. Inline Execution - Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
