# Lokum-F — Local AI Chat Studio

<p align="center">
  <strong>A commercial-grade, local-first AI desktop app for macOS.</strong><br/>
  PyQt UI • MLX inference • Persistent RAG • Optional LoRA fine-tuning
</p>

<p align="center">
  <a href="#quickstart"><img alt="Quickstart" src="https://img.shields.io/badge/Quickstart-Ready-2ea44f"></a>
  <a href="./LICENSE"><img alt="License" src="https://img.shields.io/badge/License-MIT-blue"></a>
  <img alt="Platform" src="https://img.shields.io/badge/Platform-macOS-black">
  <img alt="Local First" src="https://img.shields.io/badge/Privacy-Local%20First-6f42c1">
</p>

---

## Overview

Lokum-F is a **desktop-first AI chat studio** that runs fully on your machine:
- A polished **chat UI** with streaming responses
- **RAG** (Retrieval-Augmented Generation) over your local files with persistent storage
- Optional **MLX LoRA** fine-tuning with a safe dataset pipeline (including ChatML-aware presplitting)

> Designed for developers who want a fast local workflow without sacrificing reliability, persistence, or UX.

---

## Features

### 🧠 Chat UX
- Streaming tokens + Stop button
- Chat history persisted in SQLite
- Dev Mode tools embedded as a right sidebar (no separate window)

### 📚 RAG (Persistent Knowledge)
- Index local folders (code + docs) and retrieve relevant context at chat time
- Vector search with **FAISS** + **sentence-transformers**
- Store survives restarts and is designed to be commit-safe

Supported inputs:
- Text/code: `.py .js .ts .md .txt .json .yaml ...`
- Documents: `.pdf` (PyMuPDF), `.docx` (python-docx)
- Archives: `.zim` (libzim / python-zim)
- Images (optional): OCR via `pillow + pytesseract + tesseract`

### 🧩 Fine-tuning (MLX LoRA)
- One-click training runner via `python -m mlx_lm lora ...`
- Live logs + graceful stop
- Dataset helpers: JSONL (ChatML) + SQLite

---

## Quickstart

### 1) Create + activate a virtualenv
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

Optional OCR dependency (images):
```bash
brew install tesseract
```

### 3) Run
```bash
python3 -u main.py
```

Wait until the UI shows: `Service: ready`.

---

## Configuration

### `prompts.json`
`prompts.json` controls:
- system prompt / user prompt
- theme
- `model_path` (MLX model directory)
- `use_rag` (enable/disable RAG)

### Environment variables (advanced)

#### Storage (local-first)
- `LOKUMF_HOME` — base app data folder (default: `~/.lokumf`)
- `LOKUMF_RAG_DIR` — RAG store directory (default: `~/.lokumf/rag`)
- `LOKUMF_LORA_DIR` — LoRA artifacts directory (default: `~/.lokumf/lora_data`)
- `LOKUMF_CHAT_DB` — chat history DB path (default: `~/.lokumf/app.db`)

#### Dev Mode password (no leaks)
- `LOKUMF_DEV_PASSWORD` — set your own password
- otherwise the app stores/uses `~/.lokumf/dev_password.txt`

#### Fine-tune memory shaping
- `LOKUMF_FT_PRESPLIT=1` — enable presplitting (recommended)
- `LOKUMF_FT_PRESPLIT_CHARS_PER_TOKEN` (default `4.0`) — lower = more aggressive split
- `LOKUMF_FT_CLEAR_CACHE_THRESHOLD` — lower = more frequent cache clears

---

## RAG: persistent knowledge

RAG stores a cumulative index under your configured RAG directory.

Typical files:
- `faiss_index.bin` — vector index
- `docs_metadata.npy` — aligned chunk texts
- `chunks_meta.npy` — per-chunk metadata
- `rag_state.json` — per-file indexing state
- `rag_meta.json` — convenience metadata

Reliability note:
- If loading fails, Lokum-F **quarantines** the store files (renames with `.corrupt.<timestamp>`) instead of silently appearing empty.

---

## LoRA fine-tuning

Recommended baseline for large models (e.g. 27B 6-bit on Apple Silicon):
- `batch_size = 1`
- `max_seq_len = 384` (then try 512)
- run validation **after** training (training-time eval can spike memory)

Why training can OOM even with “free RAM”:
Apple Metal memory can fail due to peak allocations + fragmentation, even if system monitors show headroom.

---

## Dataset generation

### Build a multi-turn dataset from `prompts.json`
This repo includes a generator that produces a **multi-turn** ChatML dataset that:
- asks questions only for **blocking unclear spots**
- continues immediately after the user answers
- avoids invalid samples (ChatML tags are never sliced during presplitting)

```bash
python3 tools/build_prompt_dataset.py
```

Outputs (default):
- `~/.lokumf/lora_data/train.jsonl`
- `~/.lokumf/lora_data/valid.jsonl`

Change dataset size:
```bash
LOKUMF_PROMPT_DATASET_SIZE=20000 python3 tools/build_prompt_dataset.py
```

---

## Project layout

Core modules:
- `main.py` — UI, streaming, persistence, Dev tools
- `rag_engine.py` — ingestion + indexing + retrieval
- `file_ingest.py` — extraction + chunking (PDF/DOCX/ZIM/OCR)
- `finetune_engine.py` — MLX LoRA runner + ChatML-aware presplit
- `lokum_paths.py` — centralized path & secrets management

---

## Security & privacy

- The app is **local-first**: chats, RAG, and LoRA artifacts are stored under `~/.lokumf/` by default.
- Repo `.gitignore` is configured to ignore sensitive/large artifacts (DBs, datasets, adapters, binaries).
- See internal notes: [`INTERNAL_SECURITY.md`](./INTERNAL_SECURITY.md)

---

## Troubleshooting

### Model path not found
- Set `model_path` in `prompts.json` to a valid MLX model folder.

### RAG engine not available
- Install requirements and ensure:
  - `sentence-transformers`
  - `faiss-cpu`

### OCR returns empty text
- `brew install tesseract`
