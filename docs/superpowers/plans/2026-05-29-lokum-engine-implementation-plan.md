# lokum-engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract the RAG + Fine-tune engines into a standalone pip package (`lokum-engine` / `lokum_engine`) while keeping the desktop app working during migration.

**Architecture:** Create a self-contained Python package under `lokum-engine/` (in this repo for now), move reusable logic into `src/lokum_engine/`, keep the desktop app importing from the package (with a fallback to local modules until fully migrated).

**Tech Stack:** Python, pyproject.toml (PEP 621), setuptools build, unittest, optional imports for heavy deps (faiss, sentence-transformers, mlx-lm, etc).

---

## File map (what we will create / modify)

### New (library project)
- Create: `lokum-engine/pyproject.toml`
- Create: `lokum-engine/README.md`
- Create: `lokum-engine/src/lokum_engine/__init__.py`
- Create: `lokum-engine/src/lokum_engine/paths.py`
- Create: `lokum-engine/src/lokum_engine/rag/__init__.py`
- Create: `lokum-engine/src/lokum_engine/rag/engine.py`
- Create: `lokum-engine/src/lokum_engine/finetune/__init__.py`
- Create: `lokum-engine/src/lokum_engine/finetune/engine.py`
- Create: `lokum-engine/src/lokum_engine/models/__init__.py`
- Create: `lokum-engine/src/lokum_engine/models/downloader.py`
- Create: `lokum-engine/tests/test_rag_persistence.py`
- Create: `lokum-engine/tests/test_finetune_presplit_chatml.py`
- Create: `lokum-engine/tests/test_paths.py`

### Modify (desktop app)
- Modify: `main.py` (prefer importing from `lokum_engine`, fallback to local)
- Modify: `rag_engine.py` (optional: thin wrapper or keep as-is until later)
- Modify: `finetune_engine.py` (optional: thin wrapper or keep as-is until later)

---

# Task 1: Scaffold the `lokum-engine/` project

**Files:**
- Create: `lokum-engine/pyproject.toml`
- Create: `lokum-engine/README.md`
- Create: `lokum-engine/src/lokum_engine/__init__.py`

- [ ] **Step 1: Create `lokum-engine/pyproject.toml`**

Use setuptools (simple + widely supported):

```toml
[build-system]
requires = ["setuptools>=69", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "lokum-engine"
version = "0.1.0"
description = "Lokum-F engines: RAG + MLX LoRA fine-tuning utilities"
readme = "README.md"
requires-python = ">=3.10"
license = { text = "MIT" }
authors = [{ name = "fox" }]
dependencies = [
  "numpy",
  "sentence-transformers",
  "faiss-cpu",
  "PyMuPDF",
  "python-docx",
  "pillow",
  "pytesseract",
  "libzim",
  "mlx-lm",
  "huggingface_hub",
]

[project.urls]
Homepage = "https://github.com/<YOU>/lokum-engine"

[tool.setuptools]
package-dir = {"" = "src"}

[tool.setuptools.packages.find]
where = ["src"]
```

Notes:
- If `faiss-cpu` causes install friction on some systems, keep it anyway for now (user prefers “heavy OK”).
- We intentionally do **not** include PyQt5: this is engine-only.

- [ ] **Step 2: Create `lokum-engine/README.md`**

Minimal library README:

```md
# lokum-engine

Lokum-F engines packaged as a Python library:
- RAG (FAISS + sentence-transformers)
- Fine-tuning runner (MLX LoRA) + ChatML-safe presplitting

## Install
```bash
pip install lokum-engine
```

## Usage
```python
from lokum_engine.rag import RAGEngine
from lokum_engine.finetune import FinetuneEngine
```
```

- [ ] **Step 3: Create `lokum-engine/src/lokum_engine/__init__.py`**

```python
from __future__ import annotations

from lokum_engine.rag.engine import RAGEngine
from lokum_engine.finetune.engine import FinetuneEngine

__all__ = ["RAGEngine", "FinetuneEngine"]
```

- [ ] **Step 4: Quick smoke import**

Run:
```bash
python3 -c "import sys; sys.path.insert(0,'lokum-engine/src'); import lokum_engine; print(lokum_engine.RAGEngine)"
```
Expected: prints class reference without crashing.

- [ ] **Step 5: Commit**

```bash
git add lokum-engine/pyproject.toml lokum-engine/README.md lokum-engine/src/lokum_engine/__init__.py
git commit -m "feat(lokum-engine): scaffold package skeleton"
```

---

# Task 2: Port path utilities into `lokum_engine.paths`

**Files:**
- Create: `lokum-engine/src/lokum_engine/paths.py`
- Test: `lokum-engine/tests/test_paths.py`

- [ ] **Step 1: Create `lokum-engine/src/lokum_engine/paths.py`**

Copy logic from the current repo’s `lokum_paths.py`, but:
- rename module to `lokum_engine.paths`
- add `models_dir()` for HF downloads

Required API (minimum):
```python
def lokumf_home() -> Path: ...
def rag_dir() -> Path: ...
def lora_dir() -> Path: ...
def models_dir() -> Path: ...
def chat_db_path() -> Path: ...
def dev_password_file() -> Path: ...
def ensure_dir(p: Path) -> Path: ...
def get_or_create_dev_password() -> tuple[str, bool, Path]: ...
```

Env vars (supported):
- `LOKUMF_HOME`
- `LOKUMF_RAG_DIR`
- `LOKUMF_LORA_DIR`
- `LOKUMF_MODELS_DIR`
- `LOKUMF_CHAT_DB`
- `LOKUMF_DEV_PASSWORD`
- `LOKUMF_DEV_PASSWORD_FILE`

- [ ] **Step 2: Add unit test `lokum-engine/tests/test_paths.py`**

```python
import os
import unittest
from pathlib import Path

from lokum_engine.paths import lokumf_home, rag_dir, lora_dir, models_dir


class TestPaths(unittest.TestCase):
    def test_overrides(self):
        os.environ["LOKUMF_HOME"] = "/tmp/lokumf_home_test"
        self.assertEqual(str(lokumf_home()), "/tmp/lokumf_home_test")
        self.assertEqual(str(rag_dir()), "/tmp/lokumf_home_test/rag")
        self.assertEqual(str(lora_dir()), "/tmp/lokumf_home_test/lora_data")
        self.assertEqual(str(models_dir()), "/tmp/lokumf_home_test/models")
```

- [ ] **Step 3: Run tests**

```bash
PYTHONPATH=lokum-engine/src python3 -m unittest lokum-engine/tests/test_paths.py -v
```

- [ ] **Step 4: Commit**

```bash
git add lokum-engine/src/lokum_engine/paths.py lokum-engine/tests/test_paths.py
git commit -m "feat(lokum-engine): add shared paths utilities"
```

---

# Task 3: Port `RAGEngine` into `lokum_engine.rag`

**Files:**
- Create: `lokum-engine/src/lokum_engine/rag/__init__.py`
- Create: `lokum-engine/src/lokum_engine/rag/engine.py`
- Test: `lokum-engine/tests/test_rag_persistence.py`

- [ ] **Step 1: Create package init**

`lokum-engine/src/lokum_engine/rag/__init__.py`
```python
from lokum_engine.rag.engine import RAGEngine

__all__ = ["RAGEngine"]
```

- [ ] **Step 2: Create `engine.py`**

Copy the current repo’s `rag_engine.py` into `lokum-engine/src/lokum_engine/rag/engine.py`, then:
- Replace imports:
  - `from lokum_paths import ...` → `from lokum_engine.paths import rag_dir, ensure_dir`
- Keep optional imports (`faiss`, `sentence_transformers`, etc.) as-is.
- Keep persistence behavior: quarantine on load failure, state tracking, etc.

Public API to preserve:
- `RAGEngine(storage_dir: str | None = None)`
- `ingest_folder(folder_path: str, recursive: bool = True) -> bool`
- `query(query_text: str, k: int = 3) -> str`
- `query_with_sources(...) -> dict`
- `reset_database()`

- [ ] **Step 3: Add a minimal persistence test**

Port/adapt existing test logic:
`lokum-engine/tests/test_rag_persistence.py` should import from `lokum_engine.rag`.

- [ ] **Step 4: Run unit tests**

```bash
PYTHONPATH=lokum-engine/src python3 -m unittest lokum-engine/tests/test_rag_persistence.py -v
```

- [ ] **Step 5: Commit**

```bash
git add lokum-engine/src/lokum_engine/rag lokum-engine/tests/test_rag_persistence.py
git commit -m "feat(lokum-engine): port RAGEngine into library"
```

---

# Task 4: Port `FinetuneEngine` into `lokum_engine.finetune`

**Files:**
- Create: `lokum-engine/src/lokum_engine/finetune/__init__.py`
- Create: `lokum-engine/src/lokum_engine/finetune/engine.py`
- Test: `lokum-engine/tests/test_finetune_presplit_chatml.py`

- [ ] **Step 1: Create package init**

`lokum-engine/src/lokum_engine/finetune/__init__.py`
```python
from lokum_engine.finetune.engine import FinetuneEngine

__all__ = ["FinetuneEngine"]
```

- [ ] **Step 2: Create `engine.py`**

Copy current repo’s `finetune_engine.py` into:
`lokum-engine/src/lokum_engine/finetune/engine.py`

Then update:
- imports to use `lokum_engine.paths.lora_dir()` for dataset_dir
- keep ChatML-safe presplit implementation

Preserve API:
- `start_training(...)`
- `start_validation(...)`
- `presplit_dataset(...)`

- [ ] **Step 3: Add/port presplit test**

Create `lokum-engine/tests/test_finetune_presplit_chatml.py` similar to existing one but importing:
`from lokum_engine.finetune.engine import _presplit_jsonl_file`

- [ ] **Step 4: Run unit tests**

```bash
PYTHONPATH=lokum-engine/src python3 -m unittest lokum-engine/tests/test_finetune_presplit_chatml.py -v
```

- [ ] **Step 5: Commit**

```bash
git add lokum-engine/src/lokum_engine/finetune lokum-engine/tests/test_finetune_presplit_chatml.py
git commit -m "feat(lokum-engine): port FinetuneEngine into library"
```

---

# Task 5: Add HF model downloader utility

**Files:**
- Create: `lokum-engine/src/lokum_engine/models/__init__.py`
- Create: `lokum-engine/src/lokum_engine/models/downloader.py`

- [ ] **Step 1: Create models package**

`lokum-engine/src/lokum_engine/models/__init__.py`
```python
from lokum_engine.models.downloader import download_snapshot

__all__ = ["download_snapshot"]
```

- [ ] **Step 2: Create downloader**

`lokum-engine/src/lokum_engine/models/downloader.py`
```python
from __future__ import annotations

from pathlib import Path
from typing import Optional

from lokum_engine.paths import ensure_dir, models_dir


def download_snapshot(repo_id: str, *, revision: str = "main", token: Optional[str] = None) -> str:
    from huggingface_hub import snapshot_download

    target = ensure_dir(models_dir() / repo_id.replace("/", "__") / revision)
    path = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=str(target),
        local_dir_use_symlinks=False,
        token=token,
    )
    return str(Path(path).resolve())
```

- [ ] **Step 3: Commit**

```bash
git add lokum-engine/src/lokum_engine/models
git commit -m "feat(lokum-engine): add HF model downloader helper"
```

---

# Task 6: Desktop app integration (non-breaking)

**Files:**
- Modify: `main.py`
- Optional Modify: `rag_engine.py`, `finetune_engine.py`

- [ ] **Step 1: Prefer library imports with fallback**

In `main.py` where it imports `RAGEngine` / `FinetuneEngine`, switch to:

```python
try:
    from lokum_engine.rag import RAGEngine as _RAGEngine  # type: ignore
except Exception:
    from rag_engine import RAGEngine as _RAGEngine  # type: ignore
```

and same for `FinetuneEngine`.

- [ ] **Step 2: Run app compile checks**

```bash
python3 -m py_compile main.py rag_engine.py finetune_engine.py
```

- [ ] **Step 3: Commit**

```bash
git add main.py
git commit -m "chore(app): prefer lokum_engine imports with fallback"
```

---

# Task 7: Build & local install test for the library

**Files:**
- (build output only)

- [ ] **Step 1: Build**

```bash
cd lokum-engine
python3 -m pip install -U build
python3 -m build
```

Expected:
- `lokum-engine/dist/lokum_engine-0.1.0-py3-none-any.whl`
- `lokum-engine/dist/lokum-engine-0.1.0.tar.gz`

- [ ] **Step 2: Install locally (editable)**

```bash
python3 -m pip install -e lokum-engine
python3 -c "from lokum_engine import RAGEngine, FinetuneEngine; print(RAGEngine, FinetuneEngine)"
```

- [ ] **Step 3: Commit**

No commit (build artifacts should not be committed).

---

# Task 8: PyPI publishing checklist (for the user)

- [ ] **Step 1: Create PyPI account**
- [ ] **Step 2: Create an API token**
  - PyPI → Account settings → API tokens → “Add API token”
- [ ] **Step 3: Install twine**

```bash
python3 -m pip install -U twine
```

- [ ] **Step 4: Upload**

```bash
cd lokum-engine
python3 -m build
python3 -m twine upload dist/*
```

When prompted:
- username: `__token__`
- password: your `pypi-...` token

---

# Self-review checklist (this plan)

- Spec coverage: ✅ package skeleton + engines + app migration + publish steps covered.
- Placeholder scan: ✅ no “TODO later” steps; each step includes concrete file paths, code blocks, and commands.
- Type consistency: ✅ uses `lokum-engine` (PyPI) and `lokum_engine` (import) consistently.

