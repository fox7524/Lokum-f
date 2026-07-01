# RAG + Fine-Tune Performance Refresh Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the backend RAG and fine-tune paths so indexing, retrieval, dataset generation, and training preparation become faster, more memory-efficient, and easier to validate without breaking the current desktop app behavior.

**Architecture:** Introduce focused backend modules behind the existing `RAGEngine` and `FinetuneEngine` entrypoints, then migrate hot paths in small test-protected steps. Preserve existing on-disk formats and public flows while adding incremental hashing, better dataset validation, streaming writes, and explicit preflight checks.

**Tech Stack:** Python 3, pytest, NumPy, FAISS, MLX/`mlx_lm`, JSONL, existing desktop app modules

---

## File Map

- Create: `rag/__init__.py`
- Create: `rag/normalize.py`
- Create: `rag/chunker.py`
- Create: `rag/state_store.py`
- Create: `rag/query_service.py`
- Create: `finetune/__init__.py`
- Create: `finetune/dataset_validator.py`
- Create: `finetune/dataset_writer.py`
- Create: `finetune/job_preflight.py`
- Create: `tests/test_rag_chunker_incremental.py`
- Create: `tests/test_query_service.py`
- Create: `tests/test_finetune_dataset_validator.py`
- Create: `tests/test_finetune_preflight.py`
- Modify: `file_ingest.py`
- Modify: `rag_engine.py`
- Modify: `finetune_engine.py`
- Modify: `tools/build_lora_gem_dataset.py`

### Task 1: Extract Shared Text Normalization And Chunking

**Files:**
- Create: `rag/normalize.py`
- Create: `rag/chunker.py`
- Create: `rag/__init__.py`
- Modify: `file_ingest.py`
- Modify: `rag_engine.py`
- Test: `tests/test_rag_chunker_incremental.py`

- [ ] **Step 1: Write the failing tests**

```python
from rag.chunker import chunk_text, chunk_signature


def test_chunk_signature_is_stable_for_same_text() -> None:
    a = chunk_signature("same text")
    b = chunk_signature("same text")
    assert a == b


def test_chunk_text_preserves_content_order() -> None:
    text = "alpha beta gamma delta epsilon zeta eta theta"
    chunks = chunk_text(text, chunk_size=18, overlap=4)
    assert chunks
    assert chunks[0].text.startswith("alpha")
    assert chunks[-1].text.endswith("theta")


def test_chunk_text_rejects_invalid_overlap() -> None:
    try:
        chunk_text("abc", chunk_size=4, overlap=4)
    except ValueError as exc:
        assert "overlap" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_rag_chunker_incremental.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rag'`

- [ ] **Step 3: Write minimal normalization/chunking implementation**

```python
# rag/normalize.py
import re


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
```

```python
# rag/chunker.py
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import List

from .normalize import normalize_text


@dataclass(frozen=True)
class TextChunk:
    text: str
    signature: str


def chunk_signature(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 120) -> List[TextChunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    norm = normalize_text(text)
    if not norm:
        return []

    out: List[TextChunk] = []
    start = 0
    step = chunk_size - overlap
    while start < len(norm):
        chunk = norm[start : start + chunk_size].strip()
        if chunk:
            out.append(TextChunk(text=chunk, signature=chunk_signature(chunk)))
        start += step
    return out
```

```python
# rag/__init__.py
from .chunker import TextChunk, chunk_signature, chunk_text
from .normalize import normalize_text

__all__ = ["TextChunk", "chunk_signature", "chunk_text", "normalize_text"]
```

- [ ] **Step 4: Wire old callers to the shared helpers**

```python
# file_ingest.py
from rag import chunk_text, normalize_text
```

```python
# rag_engine.py
from rag import chunk_text, normalize_text
```

Replace duplicated inline whitespace cleanup and chunk slicing with calls to these helpers.

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_rag_chunker_incremental.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add rag/__init__.py rag/normalize.py rag/chunker.py file_ingest.py rag_engine.py tests/test_rag_chunker_incremental.py
git commit -m "refactor: extract shared rag normalization and chunking"
```

### Task 2: Add Incremental RAG State And Query Service Boundaries

**Files:**
- Create: `rag/state_store.py`
- Create: `rag/query_service.py`
- Modify: `rag_engine.py`
- Test: `tests/test_query_service.py`
- Test: `tests/test_rag_chunker_incremental.py`

- [ ] **Step 1: Write the failing tests**

```python
from rag.query_service import build_context_block
from rag.state_store import build_file_state


def test_build_file_state_contains_chunk_signatures() -> None:
    state = build_file_state(
        source_path="/tmp/a.txt",
        raw_text="alpha beta gamma",
        chunk_size=8,
        overlap=2,
    )
    assert state["source_path"] == "/tmp/a.txt"
    assert state["chunk_count"] >= 1
    assert len(state["chunk_signatures"]) == state["chunk_count"]


def test_build_context_block_joins_chunks_with_separator() -> None:
    block = build_context_block(["one", "two", "three"])
    assert "one" in block
    assert "\n---\n" in block
    assert block.endswith("three")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_query_service.py tests/test_rag_chunker_incremental.py -v`
Expected: FAIL with missing imports/functions

- [ ] **Step 3: Implement the state builder and query helper**

```python
# rag/state_store.py
from __future__ import annotations

import hashlib
from typing import Any, Dict

from .chunker import chunk_text
from .normalize import normalize_text


def file_signature(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def build_file_state(source_path: str, raw_text: str, chunk_size: int, overlap: int) -> Dict[str, Any]:
    chunks = chunk_text(raw_text, chunk_size=chunk_size, overlap=overlap)
    return {
        "source_path": source_path,
        "file_signature": file_signature(raw_text),
        "chunk_count": len(chunks),
        "chunk_signatures": [chunk.signature for chunk in chunks],
    }
```

```python
# rag/query_service.py
from __future__ import annotations

from typing import Iterable


def build_context_block(chunks: Iterable[str]) -> str:
    clean = [chunk.strip() for chunk in chunks if chunk and chunk.strip()]
    return "\n---\n".join(clean)
```

- [ ] **Step 4: Use the new helpers inside `RAGEngine`**

```python
# rag_engine.py
from rag.query_service import build_context_block
from rag.state_store import build_file_state
```

Implementation requirements:
- Use `build_file_state(...)` when computing per-file ingest state
- Store chunk signatures in state for incremental comparison
- Use `build_context_block(...)` instead of open-coded context joining

- [ ] **Step 5: Run targeted tests**

Run: `pytest tests/test_query_service.py tests/test_rag_chunker_incremental.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add rag/state_store.py rag/query_service.py rag_engine.py tests/test_query_service.py tests/test_rag_chunker_incremental.py
git commit -m "refactor: add rag state and query service boundaries"
```

### Task 3: Add Fine-Tune Dataset Validation And Streaming Writers

**Files:**
- Create: `finetune/__init__.py`
- Create: `finetune/dataset_validator.py`
- Create: `finetune/dataset_writer.py`
- Modify: `tools/build_lora_gem_dataset.py`
- Modify: `finetune_engine.py`
- Test: `tests/test_finetune_dataset_validator.py`

- [ ] **Step 1: Write the failing tests**

```python
from pathlib import Path

from finetune.dataset_validator import validate_jsonl_rows


def test_validate_jsonl_rows_accepts_text_rows() -> None:
    rows = ['{"text": "hello"}', '{"text": "world"}']
    result = validate_jsonl_rows(rows)
    assert result.total == 2
    assert result.invalid == 0


def test_validate_jsonl_rows_flags_bad_rows() -> None:
    rows = ['{"text": "ok"}', '{"bad": 1}', 'not-json']
    result = validate_jsonl_rows(rows)
    assert result.total == 3
    assert result.invalid == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_finetune_dataset_validator.py -v`
Expected: FAIL with `ModuleNotFoundError` or missing names

- [ ] **Step 3: Implement validator and streaming writer**

```python
# finetune/dataset_validator.py
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class ValidationResult:
    total: int
    invalid: int


def validate_jsonl_rows(rows: Iterable[str]) -> ValidationResult:
    total = 0
    invalid = 0
    for row in rows:
        total += 1
        try:
            obj = json.loads(row)
        except json.JSONDecodeError:
            invalid += 1
            continue
        if not isinstance(obj, dict) or not isinstance(obj.get("text"), str) or not obj["text"].strip():
            invalid += 1
    return ValidationResult(total=total, invalid=invalid)
```

```python
# finetune/dataset_writer.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def write_jsonl_stream(path: Path, texts: Iterable[str]) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for text in texts:
            handle.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            count += 1
    return count
```

```python
# finetune/__init__.py
from .dataset_validator import ValidationResult, validate_jsonl_rows
from .dataset_writer import write_jsonl_stream

__all__ = ["ValidationResult", "validate_jsonl_rows", "write_jsonl_stream"]
```

- [ ] **Step 4: Replace ad-hoc JSONL writing in current flows**

```python
# tools/build_lora_gem_dataset.py
from finetune import validate_jsonl_rows, write_jsonl_stream
```

```python
# finetune_engine.py
from finetune import validate_jsonl_rows, write_jsonl_stream
```

Implementation requirements:
- Use `write_jsonl_stream(...)` instead of open-coded JSONL loops
- Re-read written files and validate rows through `validate_jsonl_rows(...)`
- Abort if invalid rows are detected

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_finetune_dataset_validator.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add finetune/__init__.py finetune/dataset_validator.py finetune/dataset_writer.py tools/build_lora_gem_dataset.py finetune_engine.py tests/test_finetune_dataset_validator.py
git commit -m "refactor: centralize finetune dataset validation and writing"
```

### Task 4: Add Fine-Tune Preflight Checks

**Files:**
- Create: `finetune/job_preflight.py`
- Modify: `finetune_engine.py`
- Test: `tests/test_finetune_preflight.py`

- [ ] **Step 1: Write the failing tests**

```python
from pathlib import Path

from finetune.job_preflight import preflight_training


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_finetune_preflight.py -v`
Expected: FAIL with missing module or function

- [ ] **Step 3: Implement preflight helper**

```python
# finetune/job_preflight.py
from __future__ import annotations

from pathlib import Path
from typing import Tuple


def preflight_training(model_path: str, dataset_dir: Path) -> Tuple[bool, str]:
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
```

- [ ] **Step 4: Call preflight before launching MLX jobs**

```python
# finetune_engine.py
from finetune.job_preflight import preflight_training
```

Implementation requirements:
- Run `preflight_training(...)` before constructing the MLX command
- Surface the failure message directly to the caller/UI
- Do not start subprocess execution when preflight fails

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_finetune_preflight.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add finetune/job_preflight.py finetune_engine.py tests/test_finetune_preflight.py
git commit -m "refactor: add finetune preflight validation"
```

### Task 5: Integrate, Regressions, And Performance Guardrails

**Files:**
- Modify: `rag_engine.py`
- Modify: `finetune_engine.py`
- Modify: `tools/build_lora_gem_dataset.py`
- Test: `tests/test_query_service.py`
- Test: `tests/test_rag_chunker_incremental.py`
- Test: `tests/test_finetune_dataset_validator.py`
- Test: `tests/test_finetune_preflight.py`

- [ ] **Step 1: Add lightweight timing logs around hot paths**

```python
import time

start = time.perf_counter()
# run stage
elapsed = time.perf_counter() - start
print(f"[perf] ingest_stage={stage_name} seconds={elapsed:.3f}")
```

Apply only to:
- RAG ingest stages
- RAG query assembly
- Dataset writing
- Training preflight

- [ ] **Step 2: Run the focused test suite**

Run: `pytest tests/test_query_service.py tests/test_rag_chunker_incremental.py tests/test_finetune_dataset_validator.py tests/test_finetune_preflight.py -v`
Expected: PASS

- [ ] **Step 3: Run the existing regression tests most likely to be affected**

Run: `pytest tests/test_file_ingest.py tests/test_rag_persistence.py tests/test_rag_soft_delete_and_abort.py -v`
Expected: PASS

- [ ] **Step 4: Smoke-test dataset generation**

Run: `python3 tools/build_lora_gem_dataset.py --out-dir /tmp/lora-gem-smoke --train-size 1000 --valid-size 100 --seed 1337`
Expected:
- Command exits successfully
- `/tmp/lora-gem-smoke/train.jsonl` exists
- `/tmp/lora-gem-smoke/valid.jsonl` exists

- [ ] **Step 5: Commit the integration pass**

```bash
git add rag_engine.py finetune_engine.py tools/build_lora_gem_dataset.py tests/test_query_service.py tests/test_rag_chunker_incremental.py tests/test_finetune_dataset_validator.py tests/test_finetune_preflight.py
git commit -m "perf: integrate rag and finetune backend refresh"
```

## Self-Review Checklist

- Spec coverage:
  - RAG normalization/chunking extraction: Task 1
  - Incremental hashing/state/query separation: Task 2
  - Dataset validation/streaming writes: Task 3
  - Training preflight/artifact safety: Task 4
  - Performance guardrails/regression verification: Task 5
- Placeholder scan: no `TODO`, `TBD`, or “similar to above” placeholders remain
- Type consistency:
  - `chunk_text`, `chunk_signature`, `build_file_state`, `build_context_block`, `validate_jsonl_rows`, `write_jsonl_stream`, and `preflight_training` use consistent names throughout

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-28-rag-finetune-performance-refresh.md`.

Two execution options:

1. **Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration
2. **Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
