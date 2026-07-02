## Internal Security Notes (Local-Only)

This project is **local-first**. Some artifacts contain private user data and should **never** be committed to git.

### Dev Mode password

Dev Mode is password-gated.

Password sources (priority order):
1) `LOKUMF_DEV_PASSWORD` (environment variable)
2) `LOKUMF_DEV_PASSWORD_FILE` (path to a local file containing the password)
3) Auto-generated password stored at:
   - `~/.lokumf/dev_password.txt`

If the password is auto-generated, the app shows it once at startup and stores it on disk.

### Local artifact locations (defaults)

These live under `~/.lokumf/` by default:
- `app.db` — chat history (private)
- `rag/` — RAG store (private)
- `lora_data/` — datasets, adapters, configs (large/private)

Override (advanced):
- `LOKUMF_HOME`
- `LOKUMF_RAG_DIR`
- `LOKUMF_LORA_DIR`
- `LOKUMF_CHAT_DB`

### Git safety

Repo `.gitignore` is configured to ignore:
- training outputs (`lora_data/`, adapters, large binaries)
- databases (`app.db`, `dataset.db`, etc.)
- common build artifacts (`.venv`, `__pycache__`, etc.)

