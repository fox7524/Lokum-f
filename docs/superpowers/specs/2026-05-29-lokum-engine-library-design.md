---
title: "lokum-engine (pip) — RAG + Fine-tune Engines Library"
date: "2026-05-29"
status: draft
---

# Goal

Extract the RAG and fine-tune engines from the desktop app into a standalone Python package:

- **PyPI name:** `lokum-engine`
- **Python import name:** `lokum_engine`
- Ships as a **single package** (heavy dependencies are acceptable).
- Keep the existing desktop app working while we migrate (non-breaking, incremental).

# Non-goals (for first iteration)

- Extras-based modular installs (`[rag]`, `[finetune]`) — user chose “single heavy package”.
- Cross-platform packaging guarantees (we’ll optimize for macOS first).
- Perfect API stability; we’ll stabilize after first publish.

# Repository strategy (within current repo first)

Create a new folder in the current repository:

`lokum-engine/`
- `pyproject.toml`
- `README.md` (library-focused)
- `src/lokum_engine/`
  - `__init__.py` (exports)
  - `paths.py` (extracted from current `lokum_paths.py`)
  - `rag/` (extracted from current `rag_engine.py`)
  - `finetune/` (extracted from current `finetune_engine.py`)
- `tests/` (minimal unit tests to protect persistence + presplit behaviors)

User will later create a new GitHub repo and push this folder’s contents.

# Public API (first cut)

Intended usage:

```python
from lokum_engine.rag import RAGEngine
from lokum_engine.finetune import FinetuneEngine
```

Additionally, convenience exports:

```python
from lokum_engine import RAGEngine, FinetuneEngine
```

# Storage conventions (privacy-first)

Keep existing conventions:

- Base: `~/.lokumai` (override: `LOKUMAI_HOME`)
- RAG store: `~/.lokumai/rag` (override: `LOKUMAI_RAG_DIR`)
- LoRA artifacts: `~/.lokumai/lora_data` (override: `LOKUMAI_LORA_DIR`)
- Chat DB: `~/.lokumai/app.db` (override: `LOKUMAI_CHAT_DB`)

These path utilities live in `lokum_engine.paths`.

# Migration plan (non-breaking)

Phase 1: introduce package without breaking app
- Copy/move engine logic into `lokum_engine`.
- Keep app-level files as thin wrappers (or update imports).

Phase 2: app consumes the pip package cleanly
- Desktop app imports from `lokum_engine`.
- Remove duplicated logic once stable.

# Packaging plan (PyPI)

Tools:
- `python -m build` → build sdist + wheel
- `python -m twine upload dist/*` → upload

Versioning:
- Start at `0.1.0`
- Bump patch/minor as changes land

Auth:
- Use PyPI API token (`pypi-...`) stored locally (never committed).

# Testing scope

- Unit tests for:
  - RAG persistence load/save behavior (existing tests can be reused/adapted)
  - presplit behavior (ChatML-safe splitting)
  - path resolution overrides

# Success criteria

- Can `pip install lokum-engine` and import engines successfully.
- Desktop app runs unchanged (or with minimal import changes).
- First PyPI release produced and installable.

