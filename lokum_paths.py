"""
Centralized path management for Lokum-F.

Why this exists:
- Avoid hardcoding ~/.lokumf in multiple places
- Allow overriding paths in packaged/sandboxed environments
- Keep persistence paths stable across "current working directory" changes
"""

from __future__ import annotations

import os
import secrets
from pathlib import Path


def lokumf_home() -> Path:
    """
    Base persistence directory for Lokum-F.

    Override with:
      - LOKUMF_HOME=/custom/path
    """
    raw = (os.environ.get("LOKUMF_HOME") or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return (Path.home() / ".lokumf").expanduser().resolve()


def rag_dir() -> Path:
    """
    RAG persistent store directory.

    Override with:
      - LOKUMF_RAG_DIR=/custom/path
    """
    raw = (os.environ.get("LOKUMF_RAG_DIR") or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return lokumf_home() / "rag"


def ensure_dir(p: Path) -> Path:
    """
    Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
    """
    p.mkdir(parents=True, exist_ok=True)
    return p


def lora_dir() -> Path:
    """
    LoRA artifacts root (datasets, adapters, configs).

    Override with:
      - LOKUMF_LORA_DIR=/custom/path

    Default:
      ~/.lokumf/lora_data
    """
    raw = (os.environ.get("LOKUMF_LORA_DIR") or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return lokumf_home() / "lora_data"


def chat_db_path() -> Path:
    """
    Default chat history DB path.

    Override with:
      - LOKUMF_CHAT_DB=/custom/path/app.db
    """
    raw = (os.environ.get("LOKUMF_CHAT_DB") or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    db_dir = lokumf_home() / "database"
    ensure_dir(db_dir)
    return db_dir / "app.db"


def dev_password_file() -> Path:
    """
    Dev password storage file (local-only).

    Override with:
      - LOKUMF_DEV_PASSWORD_FILE=/custom/path/dev_password.txt
    """
    raw = (os.environ.get("LOKUMF_DEV_PASSWORD_FILE") or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return lokumf_home() / "dev_password.txt"


def config_path() -> Path:
    """
    Lokum-F Studio JSON configuration file.
    """
    return lokumf_home() / "config.json"


def get_or_create_dev_password() -> tuple[str, bool, Path]:
    """
    Returns: (password, generated_now, location)

    Priority:
    1) env LOKUMF_DEV_PASSWORD
    2) dev_password_file() contents
    3) generate random password, write to dev_password_file()
    """
    env_pw = (os.environ.get("LOKUMF_DEV_PASSWORD") or "").strip()
    if env_pw:
        return env_pw, False, dev_password_file()

    fp = dev_password_file()
    try:
        if fp.is_file():
            pw = fp.read_text(encoding="utf-8", errors="ignore").strip()
            if pw:
                return pw, False, fp
    except Exception:
        pass

    pw = secrets.token_urlsafe(12)
    try:
        ensure_dir(fp.parent)
        fp.write_text(pw + "\n", encoding="utf-8")
        try:
            os.chmod(str(fp), 0o600)
        except Exception:
            pass
    except Exception:
        # If we can't persist, still return a password (session-only).
        return pw, True, fp
    return pw, True, fp
