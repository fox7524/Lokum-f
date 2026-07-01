---
title: "LokumAI macOS DMG + HF Model Downloader + LM Studio Compatibility"
date: "2026-05-29"
status: draft
owners:
  - fox
---

# Goals

1) Ship a **macOS (Apple Silicon) .app + .dmg** so other people can install via drag-to-Applications.
2) Keep the project **open-source** while ensuring **private + large artifacts never leak to git**.
3) Add an **optional Hugging Face model downloader** in **User Settings**.
4) Ensure models remain **visible to LM Studio**, even if we standardize on `~/.lokumai/models`.

Non-goals (for this iteration):
- App Store distribution
- Mandatory code signing / notarization (can be added later)
- Bundling large base models inside the DMG

# Current repo context (what exists today)

- GUI: PyQt app (`main.py`)
- RAG store: persisted under `~/.lokumai/rag` (override via env)
- LoRA datasets/adapters: moved to `~/.lokumai/lora_data` by default
- Dev Mode password: local-only via `~/.lokumai/dev_password.txt` (or env)
- `.gitignore` updated to ignore DBs, large binaries, lora artifacts, and internal docs

# Proposed architecture

## 1) Packaging

**Approach:** PyInstaller → `.app`, then a DMG wrapper for distribution.

### Outputs
- `dist/LokumAI.app` (arm64)
- `dist/LokumAI-macos-arm64.dmg`

### DMG contents
- `LokumAI.app`
- `/Applications` symlink
- small `README` / `First Run` note (Gatekeeper + LM Studio compatibility)

### Signing / notarization strategy
Phase 1 (default): **unsigned DMG**
- User can still run (usually via right-click → Open).

Phase 2 (optional): **signed + notarized**
- Requires Apple Developer ID.
- Add scripts + docs without changing runtime behavior.

## 2) Storage conventions (privacy + git hygiene)

Default base folder: `~/.lokumai` (override: `LOKUMAI_HOME`)

### Paths
- Chats DB: `~/.lokumai/app.db` (override: `LOKUMAI_CHAT_DB`)
- RAG store: `~/.lokumai/rag` (override: `LOKUMAI_RAG_DIR`)
- LoRA artifacts: `~/.lokumai/lora_data` (override: `LOKUMAI_LORA_DIR`)
- Models: `~/.lokumai/models` (new; override: `LOKUMAI_MODELS_DIR`)

### Migration rules (one-time)
On app start:
- If repo-local `./app.db` exists and `~/.lokumai/app.db` does not → migrate.
- If repo-local `./lora_data` exists → migrate selected subfolders + top-level train/valid.

No background deletion: original files are preserved whenever migration cannot safely move.

## 3) Model Manager (User Settings)

Add a Settings section:

### 3.1 Hugging Face downloader (optional)
UI fields:
- `Repo ID` (e.g. `org/model`)
- `Revision` (optional)
- `Destination` (read-only default): `~/.lokumai/models/<repo-id>/<revision-or-main>/`
- Optional `HF Token` input (NOT persisted to disk)
Actions:
- Download (with progress)
- Cancel
- “Use this model” → set `prompts.json.model_path` to the downloaded folder

Backend:
- `huggingface_hub.snapshot_download(...)`
- If token is provided, pass it at runtime only.

Security:
- Never write HF token to `prompts.json` or any file.

### 3.2 LM Studio compatibility (visibility)

Requirement: models downloaded to `~/.lokumai/models` should still be visible in LM Studio.

**Recommended approach:** symlink `~/.lmstudio/models -> ~/.lokumai/models`

UI section in Settings:
- Toggle (default ON): “Keep models visible in LM Studio”
- Button: “Set up LM Studio symlink…”
- Button: “Restore backup…” (optional but recommended)

Symlink setup flow (with dialogs):
1) Detect:
   - if `~/.lmstudio/models` is already a symlink to `~/.lokumai/models` → show “Already configured”.
   - if `~/.lmstudio/models` exists as a directory:
     - confirm with user
     - backup to `~/.lmstudio/models.bak-YYYYMMDD-HHMMSS`
     - create symlink
2) If `~/.lmstudio` does not exist → create it.
3) If symlink creation fails → show actionable error.

Guardrails:
- Never do this automatically; only on explicit button press.
- Recommend the user closes LM Studio before running setup.

# UX considerations (non-breaking)

- All new functionality is additive and lives under Settings.
- Default behavior should not break existing users:
  - Existing `prompts.json.model_path` still works.
  - If users keep using LM Studio-managed models, they can ignore downloader.

# Error handling & telemetry (local-only)

- Show clear, user-actionable error messages:
  - HF download errors (401, 404, disk full)
  - symlink permission errors
  - target folder not writable
- No network telemetry in this iteration.

# Testing plan

Unit tests (non-GUI):
- Path resolution for `~/.lokumai/models` + overrides
- Symlink flow: detection + intended operations (dry-run unit tests)
- Downloader: validate inputs; mock download call

Manual smoke test:
- Build app → open DMG → drag install → run
- Download a small HF repo (or public tiny model) → set model_path → load
- LM Studio symlink: confirm LM Studio still sees the same models folder

# Rollout checklist

- [ ] Add `LOKUMAI_MODELS_DIR` support in `lokum_paths.py`
- [ ] Add Settings UI: HF downloader + LM Studio symlink buttons
- [ ] Add packaging scripts: PyInstaller spec + DMG creation script
- [ ] Document first-run + Gatekeeper instructions
- [ ] Optional: add “restore backup” for LM Studio models folder

