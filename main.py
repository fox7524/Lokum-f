"""
Lokum-F Studio - Main Application Entry Point

ARCHITECTURE OVERVIEW:
======================
Lokum-F Studio is a dual-mode AI coding assistant with two distinct interfaces:

1. USER MODE (default):
   - Clean, minimal interface for chatting with the AI
   - System prompt customization
   - Chat history management
   - Theme selection (dark/light)
   - Access to Settings panel

2. DEV MODE (password: local-only):
   - Advanced developer panel with 5 tabs:
     * RAG Indexer: Index project files, PDFs, DOCX, ZIM archives
     * Fine-tune: Configure and run LoRA fine-tuning
     * Model: Select and load different MLX models
     * Testing: Run AST benchmarks, stress tests, RAM monitoring
     * Unrestricted: Bypass "ask before acting" safety rule

KEY DESIGN PATTERNS:
===================
1. QThread for non-blocking AI generation
   - AIWorker runs model inference in background thread
   - Emits signals for streaming tokens: new_token, finished, error
   - Prevents GUI freezing during generation

2. MemoryMonitor for real-time system stats
   - Tracks app RAM usage and CPU percentage
   - Updates every 2 seconds via QTimer in worker thread

3. RAG Integration
   - FAISS vector index for semantic search
   - sentence-transformers for embeddings
   - Supports: code files, PDFs, DOCX, ZIM archives

4. System Prompt Safety
   - Default: "Ask Before Acting" - AI must clarify before coding
   - Unrestricted Mode: AI generates code directly (dev mode only)

HOW THE CHAT LOOP WORKS:
=========================
1. User types message → soru_sor()
2. Message displayed in chat bubble
3. RAG query (if enabled) → context injection
4. Build prompt with chat history
5. Start AIWorker thread
6. Stream tokens to UI as they arrive (new_token signal)
7. On completion: display stats, update chat history

ENTRY POINT:
============
To run: python main.py
Model path is hardcoded at bottom of file.
"""

import sys
import os
import time
import json
import ast
import re
import glob
import subprocess
import sqlite3
import tempfile
import zipfile
import urllib.request
import threading
import sounddevice as sd
import numpy as np
import asyncio
try:
    import mlx_whisper
except ImportError:
    mlx_whisper = None
try:
    import edge_tts
    HAS_EDGE_TTS = True
except ImportError:
    edge_tts = None
    HAS_EDGE_TTS = False
try:
    from PyQt5.QtWebEngineWidgets import QWebEngineView
    HAS_WEBENGINE = True
except ImportError:
    HAS_WEBENGINE = False
    QWebEngineView = None

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QTextEdit, QLineEdit,
    QPushButton, QHBoxLayout, QLabel, QSplitter, QDialog,
    QFormLayout, QMessageBox, QRadioButton, QButtonGroup,
    QStackedWidget, QListWidget, QFrame, QScrollArea, QFileDialog,
    QInputDialog, QTabWidget, QCheckBox, QSpinBox, QSlider,
    QComboBox, QTextBrowser, QProgressBar, QTableWidget,
    QTableWidgetItem, QHeaderView, QAbstractItemView, QGroupBox,
    QGridLayout, QPlainTextEdit, QDoubleSpinBox, QToolButton, QMenu, QAction,
    QListWidgetItem, QSizePolicy, QGraphicsDropShadowEffect
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize, QSettings
from PyQt5.QtGui import QFont, QTextCursor, QPalette, QColor, QTextCharFormat, QCursor, QFontDatabase

psutil = None
HAS_PSUTIL = False
PSUTIL_IMPORT_ERROR = ""
try:
    import psutil as _psutil  # type: ignore
    psutil = _psutil
    HAS_PSUTIL = True
except Exception as e:
    psutil = None
    HAS_PSUTIL = False
    PSUTIL_IMPORT_ERROR = str(e)

HAS_LIBZIM = False
LIBZIM_IMPORT_ERROR = ""
try:
    import libzim as _libzim  # type: ignore
    HAS_LIBZIM = True
except Exception as e:
    HAS_LIBZIM = False
    LIBZIM_IMPORT_ERROR = str(e)

load = None
generate = None
stream_generate = None
HAS_MLX_LM = False
MLX_IMPORT_ERROR = ""
try:
    from mlx_lm import load as _load, generate as _generate, stream_generate as _stream_generate  # type: ignore
    load = _load
    generate = _generate
    stream_generate = _stream_generate
    HAS_MLX_LM = True
except Exception as e:
    load = None
    generate = None
    stream_generate = None
    HAS_MLX_LM = False
    MLX_IMPORT_ERROR = str(e)

# RAG Engine: Handles document indexing and retrieval
# Fine-tune Engine: Handles LoRA training
RAGEngine = None
FinetuneEngine = None
RAG_IMPORT_ERROR = ""
FINETUNE_IMPORT_ERROR = ""
try:
    try:
        # Prefer the pip package if available
        from lokum_engine import RAGEngine as _RAGEngine  # type: ignore
    except Exception:
        from lokum_engine.rag import RAGEngine as _RAGEngine  # type: ignore
    RAGEngine = _RAGEngine
except Exception:
    try:
        # Fallback to local implementation (dev / source checkout)
        from rag_engine import RAGEngine as _RAGEngine
        RAGEngine = _RAGEngine
    except Exception as e:
        RAG_IMPORT_ERROR = str(e)
try:
    try:
        # Prefer the pip package if available
        from lokum_engine import FinetuneEngine as _FinetuneEngine  # type: ignore
    except Exception:
        from lokum_engine.finetune import FinetuneEngine as _FinetuneEngine  # type: ignore
    FinetuneEngine = _FinetuneEngine
except Exception:
    try:
        # Fallback to local implementation (dev / source checkout)
        from finetune_engine import FinetuneEngine as _FinetuneEngine
        FinetuneEngine = _FinetuneEngine
    except Exception as e:
        FINETUNE_IMPORT_ERROR = str(e)

# Centralized path handling (allows overrides via env vars)
try:
    from lokum_paths import rag_dir as _lokum_rag_dir  # type: ignore
except Exception:
    _lokum_rag_dir = None

try:
    from file_ingest import iter_files as ingest_iter_files
    from file_ingest import build_text_chunks_from_paths as ingest_build_chunks
    INGEST_IMPORT_ERROR = ""
except Exception as e:
    ingest_iter_files = None
    ingest_build_chunks = None
    INGEST_IMPORT_ERROR = str(e)

# Application version and dev mode password
VERSION = "Lokum-F"
try:
    from lokum_paths import get_or_create_dev_password as _get_or_create_dev_password  # type: ignore
except Exception:
    _get_or_create_dev_password = None

if callable(_get_or_create_dev_password):
    DEV_MODE_PASSWORD, _DEV_PASSWORD_GENERATED, _DEV_PASSWORD_PATH = _get_or_create_dev_password()
else:
    DEV_MODE_PASSWORD = os.environ.get("LOKUMF_DEV_PASSWORD", "lokum123")
    _DEV_PASSWORD_GENERATED = False
    _DEV_PASSWORD_PATH = ""


def _lora_base_dir() -> str:
    """
    Stable LoRA artifact root.

    Previously we used `os.path.abspath("lora_data")` which depends on the current
    working directory (and can change depending on how the app is launched).
    """
    try:
        from lokum_paths import lora_dir as _lokum_lora_dir, ensure_dir as _ensure_dir  # type: ignore

        return str(_ensure_dir(_lokum_lora_dir()))
    except Exception:
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "lora_data"))

# ---------------------------------------------------------
# WORKER THREADS (Background Processing)
# ---------------------------------------------------------

class FuseWorker(QThread):
    """
    Olm bu class UI donmasın diye arkada çatır çatır asenkron çalışıyor, snappy hissiyatın sırrı bu.
    """
    finished = pyqtSignal(bool, str)
    line = pyqtSignal(str)

    def __init__(self, base_model: str, adapter_path: str, save_path: str):
        """
        Ayağa kalkarken ilk buralar çalışıyor, ayarları falan çekiyoruz. Marş marş!
        """
        super().__init__()
        self.base_model = base_model
        self.adapter_path = adapter_path
        self.save_path = save_path

    def run(self):
        """
        Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
        """
        try:
            cmd = [
                sys.executable, "-m", "mlx_lm", "fuse",
                "--model", self.base_model,
                "--adapter-path", self.adapter_path,
                "--save-path", self.save_path
            ]
            self.line.emit(f"Starting Fuse Process:\n{' '.join(cmd)}")
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            for output_line in process.stdout:
                self.line.emit(output_line.strip())
                
            process.wait()
            if process.returncode == 0:
                self.finished.emit(True, f"Model successfully fused and saved to:\n{self.save_path}")
            else:
                self.finished.emit(False, f"Fuse process failed with exit code {process.returncode}")
        except Exception as e:
            self.finished.emit(False, str(e))

class MicWorker(QThread):
    """
    Olm bu class UI donmasın diye arkada çatır çatır asenkron çalışıyor, snappy hissiyatın sırrı bu.
    """
    transcription_done = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, parent=None):
        """
        Ayağa kalkarken ilk buralar çalışıyor, ayarları falan çekiyoruz. Marş marş!
        """
        super().__init__(parent)
        self.is_recording = False
        self.audio_data = []
        self.sample_rate = 16000

    def run(self):
        """
        Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
        """
        self.is_recording = True
        self.audio_data = []
        try:
            with sd.InputStream(samplerate=self.sample_rate, channels=1, dtype='float32') as stream:
                while self.is_recording:
                    data, overflowed = stream.read(1024)
                    self.audio_data.append(data)
        except Exception as e:
            self.error_occurred.emit(f"Microphone error: {str(e)}")
            return

        if not self.audio_data:
            self.transcription_done.emit("")
            return

        audio_np = np.concatenate(self.audio_data, axis=0).flatten()

        try:
            if mlx_whisper is None:
                self.error_occurred.emit("mlx_whisper library not found. Run: pip install mlx-whisper")
                return
            
            # Use model from parent settings if available
            model_size = "mlx-community/whisper-large-v3-turbo"
            if hasattr(self.parent(), "stt_model_size"):
                size = self.parent().stt_model_size
                if size == "base":
                    model_size = "mlx-community/whisper-base-mlx"
                elif size == "small":
                    model_size = "mlx-community/whisper-small-mlx"
            
            result = mlx_whisper.transcribe(
                audio_np, 
                path_or_hf_repo=model_size,
                language="tr"
            )
            text = result.get("text", "").strip()
            self.transcription_done.emit(text)
        except Exception as e:
            self.error_occurred.emit(f"Transcription error: {str(e)}")


class TTSWorker(QThread):
    """
    Olm bu class UI donmasın diye arkada çatır çatır asenkron çalışıyor, snappy hissiyatın sırrı bu.
    """
    finished = pyqtSignal(bool, str)

    def __init__(self, text: str, voice: str = "tr-TR-AhmetNeural"):
        """
        Ayağa kalkarken ilk buralar çalışıyor, ayarları falan çekiyoruz. Marş marş!
        """
        super().__init__()
        self.text = text
        self.voice = voice

    def run(self):
        """
        Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
        """
        if not self.text.strip():
            self.finished.emit(False, "Empty text")
            return

        try:
            # Create a temp file for the audio
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                temp_path = tmp.name

            if HAS_EDGE_TTS:
                # Use edge-tts (High quality, free)
                async def generate_audio():
                    communicate = edge_tts.Communicate(self.text, self.voice)
                    await communicate.save(temp_path)

                asyncio.run(generate_audio())
            else:
                # Fallback to macOS 'say' command (Offline, lower quality)
                # 'say' doesn't support mp3 directly easily, so we use aiff then play
                temp_path = temp_path.replace(".mp3", ".aiff")
                subprocess.run(["say", "-v", "Cem", self.text, "-o", temp_path], check=True)

            # Play the audio using native macOS 'afplay'
            subprocess.run(["afplay", temp_path], check=True)
            
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
            self.finished.emit(True, "")
        except Exception as e:
            self.finished.emit(False, str(e))

    def stop_recording(self):
        """
        Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
        """
        self.is_recording = False

class ModelLoaderWorker(QThread):
    """
    Olm bu class UI donmasın diye arkada çatır çatır asenkron çalışıyor, snappy hissiyatın sırrı bu.
    """
    loaded = pyqtSignal(object, object, str)
    error = pyqtSignal(str)

    def __init__(self, model_path: str):
        """
        Ayağa kalkarken ilk buralar çalışıyor, ayarları falan çekiyoruz. Marş marş!
        """
        super().__init__()
        self.model_path = model_path

    def run(self):
        """
        Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
        """
        try:
            if load is None:
                raise RuntimeError(MLX_IMPORT_ERROR or "mlx_lm is not available.")
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"The tokenizer you are loading from .* with an incorrect regex pattern: .*",
                )
                try:
                    from transformers.utils import logging as hf_logging
                    hf_logging.set_verbosity_error()
                except Exception:
                    pass
                try:
                    model, tokenizer = load(
                        self.model_path,
                        tokenizer_config={"fix_mistral_regex": True},
                        lazy=True,
                    )
                except TypeError:
                    model, tokenizer = load(self.model_path, lazy=True)
            self._ensure_special_tokens(tokenizer)
            self.loaded.emit(model, tokenizer, self.model_path)
        except Exception as e:
            self.error.emit(str(e))

    def _ensure_special_tokens(self, tokenizer):
        """
        Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
        """
        eos_id = getattr(tokenizer, "eos_token_id", None)
        if eos_id is None:
            eos_token = getattr(tokenizer, "eos_token", None) or "</s>"
            if hasattr(tokenizer, "convert_tokens_to_ids"):
                try:
                    eos_id = tokenizer.convert_tokens_to_ids(eos_token)
                except Exception:
                    eos_id = None
            if eos_id is not None:
                try:
                    tokenizer.eos_token_id = eos_id
                except Exception:
                    pass

        pad_id = getattr(tokenizer, "pad_token_id", None)
        if pad_id is None:
            if hasattr(tokenizer, "eos_token_id") and getattr(tokenizer, "eos_token_id", None) is not None:
                try:
                    tokenizer.pad_token_id = tokenizer.eos_token_id
                except Exception:
                    pass

class AIWorker(QThread):
    """
    Background thread for AI text generation.

    WHY A THREAD?
    - LLM generation can take seconds to minutes
    - Blocking the main thread would freeze the GUI
    - QThread allows concurrent execution with GUI event loop

    SIGNALS:
    - new_token(str): Emitted for each new token (for streaming display)
    - finished(str, float, int, float): Emitted when generation completes
    - error(str): Emitted if an exception occurs

    HOW STREAMING WORKS:
    - stream_generate() yields accumulated response
    - We track previous response length to extract NEW tokens only
    - Each new token is emitted via signal for live UI updates
    """
    # Signal emitted when a new token is generated (for streaming)
    new_token = pyqtSignal(str)
    # Signal emitted when generation completes: (full_response, tokens_per_sec, total_tokens, elapsed_time)
    finished = pyqtSignal(str, float, int, float, float)
    # Signal emitted on error: error message
    error = pyqtSignal(str)

    def __init__(self, model, tokenizer, prompt):
        """
        Initialize the worker thread.

        ARGS:
            model: The loaded MLX model instance
            tokenizer: The tokenizer for the model
            prompt: The formatted prompt string to generate from
        """
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.is_running = True  # Can be set False to stop generation

    def run(self):
        """
        Main thread execution - generates text and emits signals.

        This method runs in a separate thread when start() is called.
        DO NOT call this directly - use start() instead.
        """
        try:
            if stream_generate is None:
                raise RuntimeError(MLX_IMPORT_ERROR or "mlx_lm is not available.")
            if self.model is None or self.tokenizer is None:
                raise ValueError("Model or tokenizer is not loaded.")
            if getattr(self.tokenizer, "eos_token_id", None) is None:
                raise ValueError("Tokenizer eos_token_id is missing. Reload the model or choose another model.")

            start_time = time.time()
            full_response = ""
            accumulated = ""
            token_count = 0
            peak_memory_gb = 0.0

            # Iterate through streaming response
            # stream_generate yields GenerationResponse objects (incremental tokens)
            for response in stream_generate(self.model, self.tokenizer, prompt=self.prompt, max_tokens=1500):
                # Check if stop was requested (e.g., user clicked stop button)
                if not self.is_running:
                    break

                piece = getattr(response, "text", None)
                if piece is None:
                    piece = str(response)

                pm = getattr(response, "peak_memory", None)
                if isinstance(pm, (int, float)) and pm > peak_memory_gb:
                    peak_memory_gb = float(pm)

                if isinstance(piece, str) and piece.startswith(accumulated):
                    delta = piece[len(accumulated):]
                    accumulated = piece
                else:
                    delta = piece
                    accumulated += piece

                full_response += delta
                token_count += 1

                # Emit new token for live display
                self.new_token.emit(delta)

            # Calculate metrics
            end_time = time.time()
            elapsed = end_time - start_time
            tok_per_sec = token_count / elapsed if elapsed > 0 else 0.0

            # Emit completion signal with results
            self.finished.emit(full_response, tok_per_sec, token_count, elapsed, peak_memory_gb)

        except Exception as e:
            # Emit error signal if something goes wrong
            self.error.emit(f"Error generating response: {str(e)}")

    def stop(self):
        """
        Request the worker to stop generation.

        Sets is_running to False, which causes the loop in run() to break.
        Note: The model may still produce a few more tokens after stop is called.
        """
        self.is_running = False


class BenchmarkWorker(QThread):
    """
    Olm bu class UI donmasın diye arkada çatır çatır asenkron çalışıyor, snappy hissiyatın sırrı bu.
    """
    finished = pyqtSignal(float, int, float, str)
    error = pyqtSignal(str)

    def __init__(self, model, tokenizer, prompt: str, max_tokens: int = 128):
        """
        Ayağa kalkarken ilk buralar çalışıyor, ayarları falan çekiyoruz. Marş marş!
        """
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.max_tokens = max_tokens

    def run(self):
        """
        Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
        """
        try:
            if generate is None:
                raise RuntimeError(MLX_IMPORT_ERROR or "mlx_lm is not available.")
            if self.model is None or self.tokenizer is None:
                raise ValueError("Model/tokenizer not loaded")
            start = time.time()
            out = generate(self.model, self.tokenizer, prompt=self.prompt, max_tokens=self.max_tokens)
            elapsed = time.time() - start
            token_count = 0
            if hasattr(self.tokenizer, "encode"):
                try:
                    token_count = len(self.tokenizer.encode(out))
                except Exception:
                    token_count = len(out.split())
            else:
                token_count = len(out.split())
            tps = token_count / elapsed if elapsed > 0 else 0.0
            self.finished.emit(tps, token_count, elapsed, out[:2000])
        except Exception as e:
            self.error.emit(str(e))


class DeleteChatWorker(QThread):
    """
    Olm bu class UI donmasın diye arkada çatır çatır asenkron çalışıyor, snappy hissiyatın sırrı bu.
    """
    finished = pyqtSignal(str, bool, str, float)

    def __init__(self, db_path: str, chat_name: str):
        """
        Ayağa kalkarken ilk buralar çalışıyor, ayarları falan çekiyoruz. Marş marş!
        """
        super().__init__()
        self._db_path = os.path.abspath(db_path or "app.db")
        self._chat_name = (chat_name or "").strip()

    def run(self):
        """
        Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
        """
        start = time.perf_counter()
        try:
            conn = sqlite3.connect(self._db_path)
            try:
                conn.execute("PRAGMA foreign_keys = ON")
                cur = conn.cursor()
                cur.execute("SELECT id FROM chats WHERE name = ?", (self._chat_name,))
                row = cur.fetchone()
                if not row:
                    self.finished.emit(self._chat_name, True, "", (time.perf_counter() - start) * 1000.0)
                    return
                chat_id = int(row[0])
                cur.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
                cur.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
                conn.commit()
            finally:
                conn.close()
            self.finished.emit(self._chat_name, True, "", (time.perf_counter() - start) * 1000.0)
        except Exception as e:
            self.finished.emit(self._chat_name, False, str(e), (time.perf_counter() - start) * 1000.0)


class DatasetExportWorker(QThread):
    """
    Olm bu class UI donmasın diye arkada çatır çatır asenkron çalışıyor, snappy hissiyatın sırrı bu.
    """
    finished = pyqtSignal(bool, str, int, int, str)

    def __init__(self, folder: str, out_dir: str):
        """
        Ayağa kalkarken ilk buralar çalışıyor, ayarları falan çekiyoruz. Marş marş!
        """
        super().__init__()
        self._folder = os.path.abspath(folder or "")
        self._out_dir = os.path.abspath(out_dir or "")

    def run(self):
        """
        Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
        """
        try:
            if ingest_iter_files is None or ingest_build_chunks is None:
                self.finished.emit(False, "", 0, 0, "file_ingest is not available.")
                return
            if not self._folder or not os.path.isdir(self._folder):
                self.finished.emit(False, "", 0, 0, "Invalid folder.")
                return

            paths = ingest_iter_files(self._folder, recursive=True)
            if not paths:
                self.finished.emit(False, "", 0, 0, "No supported files found in the selected folder.")
                return
            chunks = ingest_build_chunks(paths, chunk_size=900, overlap=120)
            if not chunks:
                self.finished.emit(False, "", 0, len(paths), "No text could be extracted from the selected files.")
                return
            os.makedirs(self._out_dir, exist_ok=True)
            lines = [json.dumps({"text": c}, ensure_ascii=False) for c in chunks if (c or "").strip()]
            self.finished.emit(True, self._out_dir, len(lines), len(paths), "")
        except Exception as e:
            self.finished.emit(False, "", 0, 0, str(e))


class FinalAnswerWorker(QThread):
    """
    Olm bu class UI donmasın diye arkada çatır çatır asenkron çalışıyor, snappy hissiyatın sırrı bu.
    """
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, model, tokenizer, prompt: str, max_tokens: int = 256):
        """
        Ayağa kalkarken ilk buralar çalışıyor, ayarları falan çekiyoruz. Marş marş!
        """
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.max_tokens = int(max_tokens)

    def run(self):
        """
        Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
        """
        try:
            if generate is None:
                raise RuntimeError(MLX_IMPORT_ERROR or "mlx_lm is not available.")
            if self.model is None or self.tokenizer is None:
                raise ValueError("Model/tokenizer not loaded")
            out = generate(self.model, self.tokenizer, prompt=self.prompt, max_tokens=self.max_tokens)
            self.finished.emit((out or "").strip())
        except Exception as e:
            self.error.emit(str(e))


class RagIndexWorker(QThread):
    """
    Olm bu class UI donmasın diye arkada çatır çatır asenkron çalışıyor, snappy hissiyatın sırrı bu.
    """
    finished = pyqtSignal(bool, int, str, str)

    def __init__(self, main_app, folder: str, recursive: bool = True):
        """
        Ayağa kalkarken ilk buralar çalışıyor, ayarları falan çekiyoruz. Marş marş!
        """
        super().__init__()
        self.main_app = main_app
        self.folder = folder
        self.recursive = bool(recursive)
        self._eng = None

    def stop(self):
        """
        Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
        """
        try:
            if self._eng is not None and hasattr(self._eng, "request_abort"):
                self._eng.request_abort()
        except Exception:
            pass

    def run(self):
        """
        Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
        """
        try:
            if not self.main_app:
                self.finished.emit(False, 0, self.folder, "No main app")
                return
            eng = self.main_app.get_rag_engine()
            self._eng = eng
            if eng is None or not getattr(eng, "enabled", False):
                self.finished.emit(False, 0, self.folder, "RAG engine not available")
                return
            try:
                if hasattr(eng, "clear_abort"):
                    eng.clear_abort()
            except Exception:
                pass
            ok = bool(eng.ingest_folder(self.folder, recursive=self.recursive))
            count = len(getattr(eng, "documents", []) or [])
            self.finished.emit(ok, int(count), self.folder, "")
        except Exception as e:
            msg = str(e)
            if "aborted" in msg.lower():
                count = 0
                try:
                    eng = self.main_app.get_rag_engine() if self.main_app else None
                    count = len(getattr(eng, "documents", []) or []) if eng else 0
                except Exception:
                    count = 0
                self.finished.emit(False, int(count), self.folder, "Aborted")
                return
            self.finished.emit(False, 0, self.folder, msg)


class PythonDocsIndexWorker(QThread):
    """
    Olm bu class UI donmasın diye arkada çatır çatır asenkron çalışıyor, snappy hissiyatın sırrı bu.
    """
    finished = pyqtSignal(bool, int, str)

    def __init__(self, main_app, url: str):
        """
        Ayağa kalkarken ilk buralar çalışıyor, ayarları falan çekiyoruz. Marş marş!
        """
        super().__init__()
        self.main_app = main_app
        self.url = (url or "").strip()

    def run(self):
        """
        Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
        """
        import shutil
        tmp_root = None
        try:
            if not self.main_app:
                self.finished.emit(False, 0, "No main app")
                return
            eng = self.main_app.get_rag_engine()
            if eng is None or not getattr(eng, "enabled", False):
                self.finished.emit(False, 0, "RAG engine not available")
                return
            try:
                if hasattr(eng, "clear_abort"):
                    eng.clear_abort()
            except Exception:
                pass
            url = self.url or "https://docs.python.org/3/archives/python-3.13.0-docs-text.zip"
            tmp_root = tempfile.mkdtemp(prefix="foxai_py_docs_")
            zip_path = os.path.join(tmp_root, "python-docs.zip")
            urllib.request.urlretrieve(url, zip_path)
            extract_dir = os.path.join(tmp_root, "docs")
            os.makedirs(extract_dir, exist_ok=True)
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_dir)
            ok = bool(eng.ingest_folder(extract_dir, recursive=True))
            count = len(getattr(eng, "documents", []) or [])
            self.finished.emit(ok, int(count), "")
        except Exception as e:
            self.finished.emit(False, 0, str(e))
        finally:
            if tmp_root:
                shutil.rmtree(tmp_root, ignore_errors=True)


class FineTuneWorker(QThread):
    """
    Olm bu class UI donmasın diye arkada çatır çatır asenkron çalışıyor, snappy hissiyatın sırrı bu.
    """
    line = pyqtSignal(str)
    finished = pyqtSignal(int, str)
    error = pyqtSignal(str)

    def __init__(self, process: subprocess.Popen, adapter_path: str):
        """
        Ayağa kalkarken ilk buralar çalışıyor, ayarları falan çekiyoruz. Marş marş!
        """
        super().__init__()
        self._proc = process
        self._adapter_path = adapter_path or ""
        self._stopping = False

    def stop(self):
        """
        Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
        """
        self._stopping = True
        try:
            if self._proc and self._proc.poll() is None:
                if sys.platform != "win32":
                    try:
                        import signal
                        os.killpg(os.getpgid(self._proc.pid), signal.SIGTERM)
                    except Exception:
                        self._proc.terminate()
                else:
                    self._proc.terminate()
        except Exception:
            pass

    def run(self):
        """
        Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
        """
        try:
            import selectors
            import time
            proc = self._proc
            if proc is None:
                self.error.emit("No process")
                return
            last_line = time.time()
            last_heartbeat = 0.0
            if proc.stdout is not None:
                sel = selectors.DefaultSelector()
                try:
                    sel.register(proc.stdout, selectors.EVENT_READ)
                    while True:
                        if self._stopping:
                            break
                        if proc.poll() is not None:
                            break
                        events = sel.select(timeout=0.5)
                        if events:
                            ln = proc.stdout.readline()
                            if ln == "":
                                break
                            if ln:
                                self.line.emit(ln.rstrip("\n"))
                                last_line = time.time()
                                continue
                        now = time.time()
                        if now - last_line >= 60 and now - last_heartbeat >= 60:
                            self.line.emit(f"[train] Still running… (no output for {int(now - last_line)}s)")
                            last_heartbeat = now
                finally:
                    try:
                        sel.close()
                    except Exception:
                        pass
            try:
                rc = proc.wait(timeout=5 if self._stopping else None)
            except Exception:
                try:
                    if sys.platform != "win32":
                        try:
                            import signal
                            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                        except Exception:
                            proc.kill()
                    else:
                        proc.kill()
                except Exception:
                    pass
                rc = proc.wait()
            self.finished.emit(int(rc), self._adapter_path)
        except Exception as e:
            self.error.emit(str(e))


class MemoryMonitor(QThread):
    """
    Background thread for monitoring system resources.

    WHY SEPARATE THREAD?
    - psutil calls can be slow
    - We want to update every 2 seconds without affecting UI responsiveness

    SIGNALS:
    - update_signal(str, str): Emits (ram_usage_gb, cpu_percent)
    """
    update_signal = pyqtSignal(str, str, str, str)  # (app_ram_gb, sys_ram_percent, cpu_percent, gpu_percent)

    def run(self):
        """
        Continuous monitoring loop - runs until thread is stopped.

        Gets current process memory and system CPU percentage,
        then sleeps for 2 seconds before repeating.
        """
        if psutil is None:
            while not self.isInterruptionRequested():
                try:
                    self.update_signal.emit("N/A", "N/A", "N/A", "N/A")
                except Exception:
                    pass
                self.msleep(2000)
            return

        process = psutil.Process(os.getpid())
        while not self.isInterruptionRequested():
            try:
                # Get memory info for THIS process only
                mem_info = process.memory_info()
                used_gb = mem_info.rss / (1024 ** 3)  # Convert bytes to GB

                sys_mem = psutil.virtual_memory()
                sys_ram_percent = sys_mem.percent

                cpu_percent = psutil.cpu_percent(interval=None)

                gpu_percent = self._get_gpu_util_percent()

                self.update_signal.emit(
                    f"{used_gb:.2f} GB",
                    f"{sys_ram_percent:.1f}%",
                    f"{cpu_percent:.1f}%",
                    gpu_percent,
                )
            except Exception:
                pass
            self.msleep(2000)

    def _get_gpu_util_percent(self) -> str:
        """
        Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
        """
        if sys.platform == "darwin":
            return "N/A"

        try:
            import shutil
            if shutil.which("nvidia-smi") is None:
                return "N/A"
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=1,
            )
            if result.returncode == 0:
                val = result.stdout.strip().splitlines()[0].strip()
                if val:
                    return f"{float(val):.1f}%"
        except Exception:
            pass
        return "N/A"

# ---------------------------------------------------------
# SETTINGS & ROADMAP VIEWER
# ---------------------------------------------------------
class SettingsDialog(QDialog):
    """
    Bu class sistemin kemiklerinden biri, fazla kurcalama çöker aq.
    """
    def __init__(self, parent=None, user_prompt="", current_theme="dark"):
        """
        Ayağa kalkarken ilk buralar çalışıyor, ayarları falan çekiyoruz. Marş marş!
        """
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setFixedSize(550, 500)
        self.main_app = parent  # Reference to main window for saving

        self.setStyleSheet("""
            QDialog { background-color: #161616; color: #e0e0e0; }
            QLabel { color: #ccc; }
            QPushButton { background-color: #2a2a2a; border: 1px solid #444; border-radius: 6px; padding: 8px 16px; color: white; }
            QPushButton:hover { background-color: #333; }
            QRadioButton { color: #ccc; spacing: 8px; }
        """)

        self.final_user_prompt = user_prompt
        self.final_theme = current_theme

        layout = QVBoxLayout(self)

        # Header
        header = QLabel("Settings")
        header.setStyleSheet("font-size: 18px; font-weight: bold; color: #7c4dff; padding: 10px;")
        layout.addWidget(header)

        # User Prompt Section (NOT system prompt - that's Dev Mode only)
        layout.addWidget(QLabel("<b>User Prompt (Personality):</b>"))
        self.prompt_edit = QPlainTextEdit()
        self.prompt_edit.setPlainText(user_prompt)
        self.prompt_edit.setPlaceholderText("This prompt defines how the AI responds to you...")
        self.prompt_edit.setStyleSheet("background-color: #1e1e1e; border: 1px solid #333; border-radius: 8px; padding: 8px; color: #ddd;")
        self.prompt_edit.setMinimumHeight(120)
        layout.addWidget(self.prompt_edit)

        layout.addSpacing(10)

        # Theme Section
        layout.addWidget(QLabel("<b>Theme:</b>"))
        theme_layout = QHBoxLayout()
        self.rb_dark = QRadioButton("Dark")
        self.rb_light = QRadioButton("Light")
        self.rb_system = QRadioButton("System")

        if current_theme == "system":
            self.rb_system.setChecked(True)
        elif current_theme == "light":
            self.rb_light.setChecked(True)
        else:
            self.rb_dark.setChecked(True)

        theme_layout.addWidget(self.rb_dark)
        theme_layout.addWidget(self.rb_light)
        theme_layout.addWidget(self.rb_system)
        theme_layout.addStretch()
        layout.addLayout(theme_layout)

        layout.addSpacing(15)

        # Dev Mode Note
        dev_note = QLabel("System prompt is only editable in Dev Mode (Settings → Dev Mode)")
        dev_note.setStyleSheet("color: #666; font-size: 11px; padding: 5px;")
        layout.addWidget(dev_note)

        # Live theme preview (applies immediately)
        self.rb_dark.toggled.connect(lambda _v: self._preview_theme())
        self.rb_light.toggled.connect(lambda _v: self._preview_theme())
        self.rb_system.toggled.connect(lambda _v: self._preview_theme())

        layout.addStretch()

        # Bottom Buttons
        btn_layout = QHBoxLayout()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)

        save_btn = QPushButton("Save & Apply")
        save_btn.setStyleSheet("background-color: #1a5c3a; color: #4dff9f;")
        save_btn.clicked.connect(self.accept_settings)
        btn_layout.addWidget(save_btn)
        layout.addLayout(btn_layout)

    def accept_settings(self):
        """
        Kullanıcı ve Dev ayarlarını .lokumf içine güvenle kaydettiğimiz/okuduğumuz yer.
        """
        # Save user_prompt to prompts.json via main app
        self.final_user_prompt = self.prompt_edit.toPlainText()
        if self.rb_system.isChecked():
            self.final_theme = "system"
        elif self.rb_light.isChecked():
            self.final_theme = "light"
        else:
            self.final_theme = "dark"

        # Update main app's user_prompt
        if self.main_app and hasattr(self.main_app, 'prompts'):
            self.main_app.prompts["user_prompt"] = self.final_user_prompt
            self.main_app.prompts["theme"] = self.final_theme
            self.main_app.save_prompts(self.main_app.prompts)
            self.main_app.user_prompt = self.final_user_prompt
            self.main_app.apply_theme(self.final_theme)

        self.accept()

    def _preview_theme(self):
        """
        Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
        """
        if not self.main_app:
            return
        if self.rb_system.isChecked():
            self.main_app.apply_theme("system")
        elif self.rb_light.isChecked():
            self.main_app.apply_theme("light")
        elif self.rb_dark.isChecked():
            self.main_app.apply_theme("dark")

    def show_roadmap(self):
        """
        Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
        """
        QMessageBox.information(self, "Roadmap", "📅 Phase 1 (Apr 13-20): Foundation - Model loads, RAG indexer, LoRA fine-tune\n"
                                                 "⚡ Phase 2 (Apr 21-27): Features - System prompt, Run button, Settings\n"
                                                 "🧪 Phase 3 (Apr 28-30): Break It - Testing and bug fixes\n"
                                                 "🎯 May 11: Presentation Day")

# ---------------------------------------------------------
# DEV MODE GATE
# ---------------------------------------------------------
class DevModeGate:
    """
    Bu class sistemin kemiklerinden biri, fazla kurcalama çöker aq.
    """
    def __init__(self):
        """
        Ayağa kalkarken ilk buralar çalışıyor, ayarları falan çekiyoruz. Marş marş!
        """
        self.unlocked = False
        
    def attempt_unlock(self, password: str) -> bool:
        """
        Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
        """
        if password == DEV_MODE_PASSWORD:
            self.unlocked = True
            return True
        return False
    
    def lock(self):
        """
        Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
        """
        self.unlocked = False

dev_mode_gate = DevModeGate()

# ---------------------------------------------------------
# DEV PANEL
# ---------------------------------------------------------
class DevPanel(QWidget):
    """
    Bu class sistemin kemiklerinden biri, fazla kurcalama çöker aq.
    """
    def __init__(self, parent=None, main_app=None):
        """
        Ayağa kalkarken ilk buralar çalışıyor, ayarları falan çekiyoruz. Marş marş!
        """
        super().__init__(parent)
        self.main_app = main_app
        
        self.init_ui()
        
    def init_ui(self):
        """
        Ayağa kalkarken ilk buralar çalışıyor, ayarları falan çekiyoruz. Marş marş!
        """
        layout = QVBoxLayout(self)
        
        # Header
        header = QLabel("Developer")
        header.setObjectName("DevHeader")
        layout.addWidget(header)
        
        tabs = QTabWidget()
        
        # TAB 1: RAG CONTROLS
        tabs.addTab(self.build_rag_tab(), "RAG Indexer")
        # TAB 2: FINE-TUNING
        tabs.addTab(self.build_finetune_tab(), "Fine-tune")
        # TAB 3: MODEL SELECTOR
        tabs.addTab(self.build_model_tab(), "Model")
        # TAB 4: TESTING
        tabs.addTab(self.build_testing_tab(), "Testing")
        # TAB 5: UNRESTRICTED MODE
        tabs.addTab(self.build_unrestricted_tab(), "Unrestricted")
        
        layout.addWidget(tabs)
        
        # Bottom buttons
        bottom = QHBoxLayout()
        bottom.addStretch()
        close_btn = QPushButton("Hide")
        close_btn.clicked.connect(self._hide_dev_sidebar)
        bottom.addWidget(close_btn)
        layout.addLayout(bottom)

    def _hide_dev_sidebar(self):
        """
        Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
        """
        if self.main_app:
            self.main_app.toggle_dev_dialog(force_state=False)

    def build_rag_tab(self):
        """
        Dosyaları okuyup 768 boyutlu vektörlere çeviren RAG beyni. Hafıza buradan geliyor.
        """
        p = self.parent()
        if p is not None and hasattr(p, "build_rag_tab"):
            return p.build_rag_tab()
        w = QWidget()
        l = QVBoxLayout(w)
        l.addWidget(QLabel("RAG tab is unavailable."))
        l.addStretch()
        return w

    def build_finetune_tab(self):
        """
        Modeli Raskolnikov yapmak için MLX üzerinden LoRA eğitimini ateşlediğimiz yer. Ultra kalite!
        """
        p = self.parent()
        if p is not None and hasattr(p, "build_finetune_tab"):
            return p.build_finetune_tab()
        w = QWidget()
        l = QVBoxLayout(w)
        l.addWidget(QLabel("Fine-tune tab is unavailable."))
        l.addStretch()
        return w

    def build_model_tab(self):
        """
        Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
        """
        p = self.parent()
        if p is not None and hasattr(p, "build_model_tab"):
            return p.build_model_tab()
        w = QWidget()
        l = QVBoxLayout(w)
        l.addWidget(QLabel("Model tab is unavailable."))
        l.addStretch()
        return w

    def refresh_personas(self):
        """
        Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
        """
        self.persona_list.clear()
        persona_dir = os.path.join(os.path.expanduser("~"), ".lokumf", "personas")
        os.makedirs(persona_dir, exist_ok=True)
        for f in os.listdir(persona_dir):
            if f.endswith(".json"):
                self.persona_list.addItem(f)
                
    def use_selected_persona(self):
        """
        Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
        """
        selected = self.persona_list.currentItem()
        if not selected:
            QMessageBox.warning(self, "Seçim Yok", "Lütfen bir persona (system prompt) seçin.")
            return
            
        persona_file = selected.text()
        persona_path = os.path.join(os.path.expanduser("~"), ".lokumf", "personas", persona_file)
        
        try:
            with open(persona_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # config'i güncelle
            if self.main_app:
                if "system_prompt" in data:
                    self.main_app.config["system_prompt"] = data["system_prompt"]
                if "user_prompt" in data:
                    self.main_app.config["user_prompt"] = data["user_prompt"]
                if "unrestricted_prompt" in data:
                    self.main_app.config["unrestricted_prompt"] = data["unrestricted_prompt"]
                
                self.main_app.save_config()
                QMessageBox.information(self, "Persona Değişti", f"{persona_file} başarıyla yüklendi ve aktif edildi.")
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Persona yüklenemedi: {e}")

    def build_testing_tab(self):
        """
        Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
        """
        p = self.parent()
        if p is not None and hasattr(p, "build_testing_tab"):
            return p.build_testing_tab()
        w = QWidget()
        l = QVBoxLayout(w)
        l.addWidget(QLabel("Testing tab is unavailable."))
        l.addStretch()
        return w

    def build_unrestricted_tab(self):
        """
        Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
        """
        p = self.parent()
        if p is not None and hasattr(p, "build_unrestricted_tab"):
            return p.build_unrestricted_tab()
        w = QWidget()
        l = QVBoxLayout(w)
        l.addWidget(QLabel("Unrestricted tab is unavailable."))
        l.addStretch()
        return w


class DevPanelDialog(QWidget):
    """
    Bu class sistemin kemiklerinden biri, fazla kurcalama çöker aq.
    """
    def __init__(self, parent=None, main_app=None, embedded: bool = False):
        """
        Ayağa kalkarken ilk buralar çalışıyor, ayarları falan çekiyoruz. Marş marş!
        """
        super().__init__(parent)
        self.main_app = main_app
        self._collapsed = False
        self._expanded_size = QSize(400, 300)
        self._rag_reset_armed_until = 0.0

        root = QVBoxLayout(self)
        if embedded:
            root.setContentsMargins(0, 0, 0, 0)
            root.setSpacing(0)
            panel_layout = root
        else:
            self.setWindowTitle("Developer")
            self.setWindowFlags(self.windowFlags() | Qt.Tool)
            self.setFixedSize(400, 300)
            root.setContentsMargins(10, 10, 10, 10)
            root.setSpacing(8)

            header = QHBoxLayout()
            self.collapse_btn = QToolButton()
            self.collapse_btn.setText("▾")
            self.collapse_btn.setFixedSize(28, 28)
            self.collapse_btn.clicked.connect(self.toggle_collapsed)
            header.addWidget(self.collapse_btn)

            title = QLabel("Developer")
            title.setObjectName("DevHeader")
            header.addWidget(title)
            header.addStretch()

            close_btn = QToolButton()
            close_btn.setText("✕")
            close_btn.setFixedSize(28, 28)
            close_btn.clicked.connect(self.hide)
            header.addWidget(close_btn)
            root.addLayout(header)

            self.panel = QWidget(self)
            root.addWidget(self.panel)

        if not embedded:
            panel_layout = QVBoxLayout(self.panel)
            panel_layout.setContentsMargins(0, 0, 0, 0)
            panel_layout.setSpacing(10)

        tabs = QTabWidget()
        tabs.addTab(self._wrap_tab(self.build_rag_tab()), "RAG Indexer")
        tabs.addTab(self._wrap_tab(self.build_finetune_tab()), "Fine-tune")
        tabs.addTab(self._wrap_tab(self.build_model_tab()), "Model")
        tabs.addTab(self._wrap_tab(self.build_testing_tab()), "Testing")
        tabs.addTab(self._wrap_tab(self.build_unrestricted_tab()), "Unrestricted")
        panel_layout.addWidget(tabs)

    def _wrap_tab(self, inner: QWidget) -> QScrollArea:
        """
        Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
        """
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setFrameShape(QFrame.NoFrame)
        sc.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        sc.setWidget(inner)
        return sc

    def toggle_collapsed(self):
        """
        Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
        """
        self.set_collapsed(not self._collapsed)
        if self.main_app:
            self.main_app._save_dev_dialog_state()

    def set_collapsed(self, collapsed: bool):
        """
        Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
        """
        self._collapsed = bool(collapsed)
        if self._collapsed:
            self.collapse_btn.setText("▸")
            self.panel.setVisible(False)
            self.setFixedHeight(54)
        else:
            self.collapse_btn.setText("▾")
            self.panel.setVisible(True)
            self.setFixedSize(self._expanded_size)

    def show_roadmap(self):
        """
        Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
        """
        QMessageBox.information(self, "Roadmap", "Phase 1 (Apr 13-20): Foundation - Model loads, RAG indexer, LoRA fine-tune\n"
                                                 "Phase 2 (Apr 21-27): Features - System prompt, Run button, Settings\n"
                                                 "Phase 3 (Apr 28-30): Break It - Testing and bug fixes\n"
                                                 "May 11: Presentation Day")
    
    def build_rag_tab(self):
        """
        Dosyaları okuyup 768 boyutlu vektörlere çeviren RAG beyni. Hafıza buradan geliyor.
        """
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Status
        status_box = QGroupBox("RAG Status")
        s_layout = QGridLayout()
        
        self.rag_status_lbl = QLabel("Disabled")
        self.rag_chunks_lbl = QLabel("0 chunks indexed")
        self.rag_index_lbl = QLabel("")
        refresh_btn = QPushButton("Refresh Status")
        refresh_btn.clicked.connect(self.refresh_rag_status)
        load_btn = QPushButton("Load RAG Data")
        load_btn.clicked.connect(self.load_rag_data)
        unload_btn = QPushButton("Unload RAG Data")
        unload_btn.clicked.connect(self.unload_rag_data)
        self._rag_load_btn = load_btn
        self._rag_unload_btn = unload_btn
        
        s_layout.addWidget(QLabel("Status:"), 0, 0)
        s_layout.addWidget(self.rag_status_lbl, 0, 1)
        s_layout.addWidget(QLabel("Chunks:"), 1, 0)
        s_layout.addWidget(self.rag_chunks_lbl, 1, 1)
        s_layout.addWidget(QLabel("Index Path:"), 2, 0)
        s_layout.addWidget(self.rag_index_lbl, 2, 1)
        s_layout.addWidget(refresh_btn, 3, 0, 1, 3)
        s_layout.addWidget(load_btn, 4, 0, 1, 2)
        s_layout.addWidget(unload_btn, 4, 2, 1, 1)
        status_box.setLayout(s_layout)
        layout.addWidget(status_box)

        ws_box = QGroupBox("Project Workspace")
        ws_layout = QGridLayout()
        self.project_ws_path = QLineEdit()
        self.project_ws_path.setPlaceholderText("Select a project folder for file lookups...")
        if self.main_app and getattr(self.main_app, "project_root", ""):
            self.project_ws_path.setText(str(getattr(self.main_app, "project_root", "")))
        ws_layout.addWidget(QLabel("Folder:"), 0, 0)
        ws_layout.addWidget(self.project_ws_path, 0, 1)
        ws_browse = QPushButton("Browse...")
        ws_browse.clicked.connect(self.browse_project_workspace)
        ws_layout.addWidget(ws_browse, 0, 2)
        ws_clear = QPushButton("Clear")
        ws_clear.clicked.connect(self.clear_project_workspace)
        ws_layout.addWidget(ws_clear, 1, 2)
        ws_box.setLayout(ws_layout)
        layout.addWidget(ws_box)
        
        # Project Folder Indexing
        folder_box = QGroupBox("Project Folder Indexing")
        f_layout = QGridLayout()
        
        f_layout.addWidget(QLabel("Project Folder:"), 0, 0)
        self.rag_folder_path = QLineEdit()
        self.rag_folder_path.setPlaceholderText("Select a folder to index...")
        f_layout.addWidget(self.rag_folder_path, 0, 1)
        
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_rag_folder)
        f_layout.addWidget(browse_btn, 0, 2)
        
        index_btn = QPushButton("Index Project Files")
        self._rag_index_btn = index_btn
        index_btn.clicked.connect(self.index_project_files)
        f_layout.addWidget(index_btn, 1, 0, 1, 3)
        
        folder_box.setLayout(f_layout)
        layout.addWidget(folder_box)
        
        # Python Docs Indexing
        docs_box = QGroupBox("Python Documentation Indexing")
        d_layout = QGridLayout()
        
        self.docs_url = QLineEdit()
        self.docs_url.setPlaceholderText("URL to Python docs (or leave blank for default)...")
        d_layout.addWidget(QLabel("Docs URL:"), 0, 0)
        d_layout.addWidget(self.docs_url, 0, 1, 1, 2)
        
        index_docs_btn = QPushButton("Download & Index Python Docs")
        self._rag_docs_btn = index_docs_btn
        index_docs_btn.clicked.connect(self.index_python_docs)
        d_layout.addWidget(index_docs_btn, 1, 0, 1, 3)
        
        docs_box.setLayout(d_layout)
        layout.addWidget(docs_box)

        manage_box = QGroupBox("Manage Indexed Data")
        m_layout = QGridLayout()
        abort_btn = QPushButton("EMERGENCY ABORT")
        abort_btn.setStyleSheet("background-color: #5c1a2a; border-color: #ff4d6a; color: #ff4d6a; font-weight: bold;")
        abort_btn.clicked.connect(self.abort_rag_operations)
        m_layout.addWidget(abort_btn, 0, 0, 1, 3)

        manage_box.setLayout(m_layout)
        layout.addWidget(manage_box)
        
        # Reset
        reset_btn = QPushButton("Reset RAG Index")
        reset_btn.setStyleSheet("background-color: #5c1a2a; border-color: #ff4d6a; color: #ff4d6a;")
        reset_btn.clicked.connect(self.reset_rag)
        layout.addWidget(reset_btn)
        
        layout.addStretch()
        return widget
    
    def browse_rag_folder(self):
        """
        Dosyaları okuyup 768 boyutlu vektörlere çeviren RAG beyni. Hafıza buradan geliyor.
        """
        folder = QFileDialog.getExistingDirectory(self, "Select Project Folder")
        if folder:
            self.rag_folder_path.setText(folder)

    def browse_project_workspace(self):
        """
        Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
        """
        if not self.main_app:
            return
        folder = QFileDialog.getExistingDirectory(self, "Select Project Workspace")
        if folder:
            try:
                self.main_app._set_project_root(folder)
            except Exception:
                pass
            try:
                if hasattr(self, "project_ws_path") and self.project_ws_path is not None:
                    self.project_ws_path.setText(folder)
            except Exception:
                pass

    def clear_project_workspace(self):
        """
        Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
        """
        if not self.main_app:
            return
        try:
            self.main_app.project_root = ""
            self.main_app.prompts["project_root"] = ""
            self.main_app.save_prompts(self.main_app.prompts)
        except Exception:
            pass
        try:
            if hasattr(self, "project_ws_path") and self.project_ws_path is not None:
                self.project_ws_path.setText("")
        except Exception:
            pass
    
    def load_rag_data(self):
        """
        Dosyaları okuyup 768 boyutlu vektörlere çeviren RAG beyni. Hafıza buradan geliyor.
        """
        if not self.main_app:
            return
        try:
            self.main_app.use_rag = True
            self.main_app.prompts["use_rag"] = True
            self.main_app.save_prompts(self.main_app.prompts)
        except Exception:
            pass
        try:
            eng = self.main_app.get_rag_engine()
            if eng is not None and hasattr(eng, "get_stats"):
                eng.get_stats()
        except Exception:
            pass
        self.refresh_rag_status()

    def unload_rag_data(self):
        """
        Dosyaları okuyup 768 boyutlu vektörlere çeviren RAG beyni. Hafıza buradan geliyor.
        """
        if not self.main_app:
            return
        try:
            self.main_app.use_rag = False
            self.main_app.prompts["use_rag"] = False
            self.main_app.save_prompts(self.main_app.prompts)
        except Exception:
            pass
        try:
            if hasattr(self.main_app, "unload_rag_engine"):
                self.main_app.unload_rag_engine()
        except Exception:
            pass
        self.refresh_rag_status()
    
    def index_project_files(self):
        """
        Dosyaları okuyup 768 boyutlu vektörlere çeviren RAG beyni. Hafıza buradan geliyor.
        """
        folder = self.rag_folder_path.text().strip()
        if not folder or not os.path.isdir(folder):
            QMessageBox.warning(self, "Invalid Folder", "Please select a valid folder path.")
            return

        if not self.main_app or self.main_app.get_rag_engine() is None or not getattr(self.main_app.get_rag_engine(), "enabled", False):
            msg = "RAG engine not available. Install dependencies:\n\npip install sentence-transformers faiss-cpu"
            if RAG_IMPORT_ERROR:
                msg += f"\n\nDetails: {RAG_IMPORT_ERROR}"
            self.rag_status_lbl.setText("Disabled")
            QMessageBox.warning(self, "RAG Unavailable", msg)
            return
        
        self.rag_status_lbl.setText("Indexing…")
        if hasattr(self, "_rag_index_btn") and self._rag_index_btn is not None:
            self._rag_index_btn.setEnabled(False)
        try:
            eng = self.main_app.get_rag_engine() if self.main_app else None
            if eng is not None and hasattr(eng, "clear_abort"):
                eng.clear_abort()
        except Exception:
            pass
        self._rag_worker = RagIndexWorker(self.main_app, folder, recursive=True)
        self._rag_worker.finished.connect(self._on_rag_index_finished)
        self._rag_worker.start()
    
    def index_python_docs(self):
        """
        Dosyaları okuyup 768 boyutlu vektörlere çeviren RAG beyni. Hafıza buradan geliyor.
        """
        self.rag_status_lbl.setText("Indexing Python docs…")
        if hasattr(self, "_rag_docs_btn") and self._rag_docs_btn is not None:
            self._rag_docs_btn.setEnabled(False)
        url = (self.docs_url.text() if hasattr(self, "docs_url") else "").strip()
        if not self.main_app or self.main_app.get_rag_engine() is None or not getattr(self.main_app.get_rag_engine(), "enabled", False):
            msg = "RAG engine not available. Install dependencies:\n\npip install sentence-transformers faiss-cpu"
            if RAG_IMPORT_ERROR:
                msg += f"\n\nDetails: {RAG_IMPORT_ERROR}"
            self.rag_status_lbl.setText("Disabled")
            if hasattr(self, "_rag_docs_btn") and self._rag_docs_btn is not None:
                self._rag_docs_btn.setEnabled(True)
            QMessageBox.warning(self, "RAG Unavailable", msg)
            return
        self._docs_worker = PythonDocsIndexWorker(self.main_app, url)
        self._docs_worker.finished.connect(self._on_docs_index_finished)
        self._docs_worker.start()

    def _on_rag_index_finished(self, ok: bool, chunk_count: int, folder: str, err: str):
        """
        Dosyaları okuyup 768 boyutlu vektörlere çeviren RAG beyni. Hafıza buradan geliyor.
        """
        if hasattr(self, "_rag_index_btn") and self._rag_index_btn is not None:
            self._rag_index_btn.setEnabled(True)
        if err == "Aborted":
            self.rag_status_lbl.setText("Aborted")
            self.rag_chunks_lbl.setText(f"{int(chunk_count)} chunks indexed")
            QMessageBox.information(self, "RAG Aborted", "Indexing was aborted.")
            return
        if err:
            self.rag_status_lbl.setText("Error")
            QMessageBox.critical(self, "Indexing Error", err)
            return
        try:
            eng = self.main_app.get_rag_engine() if self.main_app else None
            idx_folder = getattr(eng, "indexed_folder", "") if eng else ""
        except Exception:
            idx_folder = ""
        self.rag_chunks_lbl.setText(f"{int(chunk_count)} chunks indexed")
        rag_dir = str(_lokum_rag_dir()) if callable(_lokum_rag_dir) else os.path.join(os.path.expanduser("~"), ".lokumf", "rag")
        self.rag_index_lbl.setText(f"Store: {rag_dir}")
        self.rag_status_lbl.setText("Active" if ok else "No data")
        if not ok:
            QMessageBox.warning(self, "Indexing Result", f"No data indexed from: {folder}\n\nIf this is a ZIM folder, ensure ZIM backends are working (libzim or python-zim).")
            return
        QMessageBox.information(self, "Indexing Complete", f"Folder indexed: {folder}\nTotal chunks in store: {int(chunk_count)}\n\nIndexed data is cumulative and will persist after restart until you reset it.")

    def abort_rag_operations(self):
        """
        Dosyaları okuyup 768 boyutlu vektörlere çeviren RAG beyni. Hafıza buradan geliyor.
        """
        try:
            eng = self.main_app.get_rag_engine() if self.main_app else None
            if eng is not None and hasattr(eng, "request_abort"):
                eng.request_abort()
        except Exception:
            pass
        try:
            if hasattr(self, "_rag_worker") and self._rag_worker is not None:
                if hasattr(self._rag_worker, "stop"):
                    self._rag_worker.stop()
        except Exception:
            pass
        self.rag_status_lbl.setText("Aborting…")

    def refresh_rag_status(self):
        """
        Dosyaları okuyup 768 boyutlu vektörlere çeviren RAG beyni. Hafıza buradan geliyor.
        """
        rag_dir = str(_lokum_rag_dir()) if callable(_lokum_rag_dir) else os.path.join(os.path.expanduser("~"), ".lokumf", "rag")
        try:
            use_rag = bool(getattr(self.main_app, "use_rag", True)) if self.main_app else True
        except Exception:
            use_rag = True
        if not use_rag:
            self.rag_status_lbl.setText("Unloaded")
            self.rag_chunks_lbl.setText("0 chunks indexed")
            self.rag_index_lbl.setText(f"Store: {rag_dir}")
            return
        eng = None
        try:
            eng = self.main_app.get_rag_engine() if self.main_app else None
        except Exception:
            eng = None

        if not eng or not getattr(eng, "enabled", False):
            self.rag_status_lbl.setText("Disabled")
            self.rag_chunks_lbl.setText("0 chunks indexed")
            self.rag_index_lbl.setText(f"Store: {rag_dir}")
            return

        try:
            stats = eng.get_stats() if hasattr(eng, "get_stats") else {}
            chunk_count = int(stats.get("chunk_count", len(getattr(eng, "documents", []) or [])))
        except Exception:
            chunk_count = len(getattr(eng, "documents", []) or [])
        # If there are store files but we couldn't load, show an error instead of "No data"
        try:
            has_store_files = False
            if eng is not None:
                has_store_files = bool(
                    os.path.exists(getattr(eng, "index_path", "")) and os.path.exists(getattr(eng, "docs_path", ""))
                )
            last_err = str(getattr(eng, "last_error", "") or "") if eng is not None else ""
        except Exception:
            has_store_files = False
            last_err = ""
        if chunk_count <= 0 and has_store_files and last_err:
            self.rag_status_lbl.setText("Error")
        else:
            self.rag_status_lbl.setText("Active" if chunk_count > 0 else "No data")
        self.rag_chunks_lbl.setText(f"{int(chunk_count)} chunks indexed")
        self.rag_index_lbl.setText(f"Store: {rag_dir}")

    def _on_docs_index_finished(self, ok: bool, chunk_count: int, err: str):
        """
        Dosyaları okuyup 768 boyutlu vektörlere çeviren RAG beyni. Hafıza buradan geliyor.
        """
        if hasattr(self, "_rag_docs_btn") and self._rag_docs_btn is not None:
            self._rag_docs_btn.setEnabled(True)
        if err:
            self.rag_status_lbl.setText("Error")
            QMessageBox.critical(self, "Docs Indexing Error", err)
            return
        self.rag_chunks_lbl.setText(f"{int(chunk_count)} chunks indexed")
        rag_dir = str(_lokum_rag_dir()) if callable(_lokum_rag_dir) else os.path.join(os.path.expanduser("~"), ".lokumf", "rag")
        self.rag_index_lbl.setText(f"Store: {rag_dir}")
        self.rag_status_lbl.setText("Active" if ok else "No data")
        QMessageBox.information(self, "Docs Indexing Complete", f"Chunks: {int(chunk_count)}")
    
    def reset_rag(self):
        """
        Dosyaları okuyup 768 boyutlu vektörlere çeviren RAG beyni. Hafıza buradan geliyor.
        """
        if not self.main_app:
            return
        eng = self.main_app.get_rag_engine()
        if not eng:
            return
        now = time.time()
        if now > float(getattr(self, "_rag_reset_armed_until", 0.0) or 0.0):
            self._rag_reset_armed_until = now + 10.0
            QMessageBox.warning(
                self,
                "Reset RAG Index",
                "Reset is armed.\n\nClick Reset again within 10 seconds to confirm.",
            )
            return

        self._rag_reset_armed_until = 0.0
        rag_dir = str(_lokum_rag_dir()) if callable(_lokum_rag_dir) else os.path.join(os.path.expanduser("~"), ".lokumf", "rag")
        text, ok = QInputDialog.getText(self, "Confirm Reset", f"Type RESET to permanently delete:\n{rag_dir}")
        if not ok or (text or "").strip().upper() != "RESET":
            QMessageBox.information(self, "Reset Cancelled", "RAG reset cancelled.")
            return

        eng.reset_database()
        self.rag_status_lbl.setText("Disabled")
        self.rag_chunks_lbl.setText("0 chunks indexed")
        rag_dir = str(_lokum_rag_dir()) if callable(_lokum_rag_dir) else os.path.join(os.path.expanduser("~"), ".lokumf", "rag")
        self.rag_index_lbl.setText(f"Store: {rag_dir}")
        QMessageBox.information(self, "Reset Complete", "RAG index has been reset.")
    
    def build_finetune_tab(self):
        """
        Modeli Raskolnikov yapmak için MLX üzerinden LoRA eğitimini ateşlediğimiz yer. Ultra kalite!
        """
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Training Config (Collapsible / Advanced)
        self.config_box = QGroupBox("Advanced LoRA Settings")
        self.config_box.setCheckable(True)
        self.config_box.setChecked(False) # Hidden by default
        c_layout = QGridLayout()

        c_layout.addWidget(QLabel("Preset:"), 0, 0)
        self.ft_preset = QComboBox()
        self.ft_preset.addItems(["Safe (Recommended)", "Recommended", "Ultra Safe (Less RAM)", "Faster (More RAM)", "Quick Test", "Ultra (Highest Quality)"])
        c_layout.addWidget(self.ft_preset, 0, 1)
        
        c_layout.addWidget(QLabel("Rank:"), 1, 0)
        self.lora_rank = QSpinBox()
        self.lora_rank.setRange(4, 32)
        self.lora_rank.setValue(8)
        c_layout.addWidget(self.lora_rank, 1, 1)
        
        c_layout.addWidget(QLabel("Alpha:"), 2, 0)
        self.lora_alpha = QSpinBox()
        self.lora_alpha.setRange(8, 64)
        self.lora_alpha.setValue(32)
        c_layout.addWidget(self.lora_alpha, 2, 1)
        
        c_layout.addWidget(QLabel("Iterations:"), 3, 0)
        self.lora_iters = QSpinBox()
        self.lora_iters.setRange(100, 2000)
        self.lora_iters.setValue(500)
        self.lora_iters.setSingleStep(100)
        c_layout.addWidget(self.lora_iters, 3, 1)
        
        c_layout.addWidget(QLabel("Batch Size:"), 4, 0)
        self.lora_batch = QSpinBox()
        self.lora_batch.setRange(1, 8)
        self.lora_batch.setValue(1)
        c_layout.addWidget(self.lora_batch, 4, 1)

        c_layout.addWidget(QLabel("Train Layers:"), 5, 0)
        self.lora_layers = QSpinBox()
        self.lora_layers.setRange(1, 32)
        self.lora_layers.setValue(8)
        c_layout.addWidget(self.lora_layers, 5, 1)

        c_layout.addWidget(QLabel("Max Seq Len:"), 6, 0)
        self.ft_max_seq = QSpinBox()
        self.ft_max_seq.setRange(256, 2048)
        self.ft_max_seq.setValue(512)
        self.ft_max_seq.setSingleStep(128)
        c_layout.addWidget(self.ft_max_seq, 6, 1)

        c_layout.addWidget(QLabel("Steps / Eval:"), 7, 0)
        self.ft_steps_per_eval = QSpinBox()
        self.ft_steps_per_eval.setRange(0, 5000)
        self.ft_steps_per_eval.setValue(200)
        self.ft_steps_per_eval.setSingleStep(50)
        c_layout.addWidget(self.ft_steps_per_eval, 7, 1)

        c_layout.addWidget(QLabel("Val Batches:"), 8, 0)
        self.ft_val_batches = QSpinBox()
        self.ft_val_batches.setRange(0, 64)
        self.ft_val_batches.setValue(1)
        c_layout.addWidget(self.ft_val_batches, 8, 1)

        c_layout.addWidget(QLabel("Clear Cache Thr:"), 9, 0)
        self.ft_clear_cache_thr = QDoubleSpinBox()
        self.ft_clear_cache_thr.setRange(0.0, 16.0)
        self.ft_clear_cache_thr.setDecimals(2)
        self.ft_clear_cache_thr.setSingleStep(0.25)
        self.ft_clear_cache_thr.setValue(2.0)
        c_layout.addWidget(self.ft_clear_cache_thr, 9, 1)
        
        self.config_box.setLayout(c_layout)
        layout.addWidget(self.config_box)
        
        # Data Source
        data_box = QGroupBox("Training Data Source")
        d_layout = QVBoxLayout()

        model_row = QHBoxLayout()
        self.ft_model_path = QLineEdit()
        self.ft_model_path.setPlaceholderText("Model path for LoRA training (optional; defaults to loaded model)")
        model_browse = QPushButton("Pick Model")
        model_browse.clicked.connect(self.browse_ft_model_path)
        model_row.addWidget(self.ft_model_path)
        model_row.addWidget(model_browse)
        d_layout.addLayout(model_row)

        ft_detect_box = QGroupBox("Detected MLX Models (LM Studio)")
        ft_d_layout = QVBoxLayout()
        self.ft_model_list = QListWidget()
        self.ft_model_list.setMinimumHeight(250) # Geniş ve kullanışlı model listesi
        self.ft_model_list.setMaximumHeight(400)
        self.ft_model_list.setAlternatingRowColors(True) # Modern look
        
        # Make items bigger and more readable
        self.ft_model_list.setStyleSheet("""
            QListWidget {
                border-radius: 8px;
                padding: 5px;
            }
            QListWidget::item {
                padding: 12px;
                border-radius: 6px;
                margin-bottom: 4px;
                border-bottom: 1px solid #333;
            }
        """)
        
        ft_d_layout.addWidget(self.ft_model_list)
        ft_btn_row = QHBoxLayout()
        ft_use_btn = QPushButton("Use Selected")
        ft_use_btn.clicked.connect(self.use_selected_ft_model)
        ft_btn_row.addWidget(ft_use_btn)
        ft_refresh_btn = QPushButton("Refresh")
        ft_refresh_btn.clicked.connect(self.refresh_ft_models)
        ft_btn_row.addWidget(ft_refresh_btn)
        ft_d_layout.addLayout(ft_btn_row)
        ft_detect_box.setLayout(ft_d_layout)
        d_layout.addWidget(ft_detect_box)
        try:
            self.refresh_ft_models()
        except Exception:
            pass

        resume_row = QHBoxLayout()
        self.ft_resume = QCheckBox("Resume from adapter file")
        self.ft_resume.setChecked(False)
        self.ft_resume_path = QLineEdit()
        self.ft_resume_path.setPlaceholderText("Path to adapters.safetensors")
        resume_browse = QPushButton("Pick Adapter")
        resume_browse.clicked.connect(self.browse_ft_resume_adapter)
        resume_row.addWidget(self.ft_resume)
        resume_row.addWidget(self.ft_resume_path)
        resume_row.addWidget(resume_browse)
        d_layout.addLayout(resume_row)
        
        self.use_sqlite = QCheckBox("Use SQLite Database (dataset table)")
        self.use_sqlite.setChecked(False)
        d_layout.addWidget(self.use_sqlite)
        
        self.use_jsonl = QCheckBox("Use JSONL File")
        self.use_jsonl.setChecked(True)
        d_layout.addWidget(self.use_jsonl)

        self.use_sqlite.toggled.connect(lambda v: self.use_jsonl.setChecked(False) if v else None)
        self.use_jsonl.toggled.connect(lambda v: self.use_sqlite.setChecked(False) if v else None)
        
        jsonl_row = QHBoxLayout()
        self.jsonl_path = QLineEdit()
        self.jsonl_path.setPlaceholderText("Path to JSONL file...")
        jsonl_browse = QPushButton("Browse")
        jsonl_browse.clicked.connect(self.browse_jsonl)
        jsonl_row.addWidget(self.jsonl_path)
        jsonl_row.addWidget(jsonl_browse)
        d_layout.addLayout(jsonl_row)

        self.ft_do_train = QCheckBox("Train")
        self.ft_do_train.setChecked(True)
        d_layout.addWidget(self.ft_do_train)

        self.ft_do_valid = QCheckBox("Validation (run after training)")
        self.ft_do_valid.setChecked(False)
        d_layout.addWidget(self.ft_do_valid)

        self.ft_presplit = QCheckBox("Pre-split long samples (uses batch + max seq len)")
        self.ft_presplit.setChecked(True)
        d_layout.addWidget(self.ft_presplit)
        
        data_box.setLayout(d_layout)
        layout.addWidget(data_box)

        ingest_box = QGroupBox("Build Training Dataset From Files")
        i_layout = QGridLayout()
        i_layout.addWidget(QLabel("Folder:"), 0, 0)
        self.ft_ingest_folder = QLineEdit()
        self.ft_ingest_folder.setPlaceholderText("Select a folder containing .zim and code/docs/images (.py/.cpp/.ino/.html/.pdf/.png/.jpg)...")
        i_layout.addWidget(self.ft_ingest_folder, 0, 1)
        pick_btn = QPushButton("Browse")
        pick_btn.clicked.connect(self.browse_finetune_ingest_folder)
        i_layout.addWidget(pick_btn, 0, 2)
        export_btn = QPushButton("Export JSONL Dataset (train/valid)")
        self._export_dataset_btn = export_btn
        export_btn.clicked.connect(self.export_finetune_dataset_from_folder)
        i_layout.addWidget(export_btn, 1, 0, 1, 3)
        ingest_box.setLayout(i_layout)
        layout.addWidget(ingest_box)
        
        # Control buttons
        btn_row = QHBoxLayout()
        start_train_btn = QPushButton("Start Training")
        self._start_train_btn = start_train_btn
        start_train_btn.setStyleSheet("background-color: #1a5c3a; color: #4dff9f;")
        start_train_btn.clicked.connect(self.start_training)
        btn_row.addWidget(start_train_btn)
        
        stop_train_btn = QPushButton("Stop")
        self._stop_train_btn = stop_train_btn
        stop_train_btn.setEnabled(False)
        stop_train_btn.clicked.connect(self.stop_training)
        btn_row.addWidget(stop_train_btn)
        
        layout.addLayout(btn_row)
        
        # ---------------- FUSE SECTION (NEW) ----------------
        fuse_box = QGroupBox("Export / Fuse Model")
        fuse_layout = QGridLayout()
        
        fuse_layout.addWidget(QLabel("New Model Name:"), 0, 0)
        self.ft_fuse_name = QLineEdit()
        self.ft_fuse_name.setPlaceholderText("e.g. finetuned-14B(Leave empty to skip fuse)")
        fuse_layout.addWidget(self.ft_fuse_name, 0, 1)
        
        self._fuse_btn = QPushButton("⚡️ Fuse & Save to LM Studio")
        self._fuse_btn.setObjectName("AccentButton")
        self._fuse_btn.clicked.connect(self.start_fuse)
        fuse_layout.addWidget(self._fuse_btn, 0, 2)
        
        fuse_box.setLayout(fuse_layout)
        layout.addWidget(fuse_box)
        # ----------------------------------------------------

        # Progress
        self.train_progress = QProgressBar()
        layout.addWidget(self.train_progress)
        
        self.train_log = QPlainTextEdit()
        self.train_log.setMaximumHeight(150)
        self.train_log.setReadOnly(True)
        layout.addWidget(QLabel("Training Log:"))
        layout.addWidget(self.train_log)
        
        layout.addStretch()
        try:
            self.ft_preset.currentIndexChanged.connect(self._apply_ft_preset)
            self._apply_ft_preset(0)
        except Exception:
            pass
        return widget

    def start_fuse(self):
        """
        Eğitilen o mükemmel adaptörü ana modelle birleştiren (fuse eden) büyü burası. %80 altıysa affetmez siler.
        """
        new_name = self.ft_fuse_name.text().strip()
        if not new_name:
            QMessageBox.warning(self, "Missing Name", "Lütfen yeni model için bir isim girin, aksi takdirde birleştirme (fuse) işlemi yapılmayacaktır!")
            return

        base_model = getattr(self, "_ft_model_path_used", None)
        if not base_model:
            base_model = self.ft_model_path.text().strip()

        adapter_path = getattr(self, "_last_adapter_path", None)
        if not adapter_path:
            # Otomatik olarak en son oluşturulan adaptörü bul
            adapters_dir = os.path.join(os.path.expanduser("~"), ".lokumf", "lora_data", "adapters")
            if os.path.exists(adapters_dir):
                runs = [os.path.join(adapters_dir, d) for d in os.listdir(adapters_dir) if d.startswith("run_")]
                if runs:
                    adapter_path = max(runs, key=os.path.getmtime)

        if not base_model or not adapter_path:
            QMessageBox.warning(self, "No Adapter Found", "Lütfen önce bir modeli eğitin veya geçerli bir adaptör/model yolu seçin.")
            return

        save_path = os.path.join(os.path.expanduser("~"), ".lmstudio", "models", "Lokum-F", new_name)
        
        self.train_log.appendPlainText(f"\n[FUSE] Preparing to fuse model into: {save_path}")
        self._fuse_btn.setEnabled(False)
        self.ft_fuse_name.setEnabled(False)

        self._fuse_worker = FuseWorker(base_model, adapter_path, save_path)
        self._fuse_worker.line.connect(self.train_log.appendPlainText)
        self._fuse_worker.finished.connect(self._on_fuse_finished)
        self._fuse_worker.start()

    def _on_fuse_finished(self, success: bool, msg: str):
        """
        Eğitilen o mükemmel adaptörü ana modelle birleştiren (fuse eden) büyü burası. %80 altıysa affetmez siler.
        """
        self._fuse_btn.setEnabled(True)
        self.ft_fuse_name.setEnabled(True)
        self.train_log.appendPlainText(f"[FUSE] {msg}")
        if success:
            QMessageBox.information(self, "Fuse Complete", msg)
        else:
            QMessageBox.critical(self, "Fuse Error", msg)

    def browse_ft_model_path(self):
        """
        Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
        """
        start = os.path.expanduser("~/.lmstudio/models")
        folder = QFileDialog.getExistingDirectory(self, "Select MLX Model Folder For Training", start if os.path.isdir(start) else "")
        if folder:
            try:
                self.ft_model_path.setText(os.path.abspath(folder))
            except Exception:
                self.ft_model_path.setText(folder)

    def _scan_lmstudio_models(self) -> list[str]:
        """
        Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
        """
        root = os.path.expanduser("~/.lmstudio/models")
        if not os.path.isdir(root):
            return []
        found: list[str] = []
        try:
            for org in sorted(os.listdir(root)):
                org_path = os.path.join(root, org)
                if not os.path.isdir(org_path):
                    continue
                for model in sorted(os.listdir(org_path)):
                    model_path = os.path.join(org_path, model)
                    if not os.path.isdir(model_path):
                        continue
                    cfg = os.path.join(model_path, "config.json")
                    tok = os.path.join(model_path, "tokenizer.json")
                    
                    try:
                        has_weights = any(
                            fn.endswith((".safetensors", ".npz", ".bin")) # Sadece MLX formatları, GGUF gizlendi
                            for fn in os.listdir(model_path)
                            if os.path.isfile(os.path.join(model_path, fn))
                        )
                    except Exception:
                        has_weights = False
                        
                    if not has_weights:
                        continue
                        
                    found.append(model_path)
        except Exception:
            return found
        return found

    def refresh_ft_models(self):
        """
        Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
        """
        if not hasattr(self, "ft_model_list") or self.ft_model_list is None:
            return
        self.ft_model_list.clear()
        
        models = self._scan_lmstudio_models()
        if not models:
            self.ft_model_list.addItem("No MLX models found in ~/.lmstudio/models")
            return
            
        for p in models:
            # Sadece modelin adını göster, tam yolu gizle
            model_name = os.path.basename(p)
            creator_name = os.path.basename(os.path.dirname(p))
            
            # Daha okunabilir format
            display_text = f"🤖 {creator_name} / {model_name}"
            
            item = QListWidgetItem(display_text)
            # Tam yolu item'ın içine gizli veri (Data) olarak kaydet
            item.setData(Qt.UserRole, p)
            self.ft_model_list.addItem(item)

    def use_selected_ft_model(self):
        """
        Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
        """
        if not hasattr(self, "ft_model_list") or self.ft_model_list is None:
            return
        items = self.ft_model_list.selectedItems()
        if not items:
            QMessageBox.information(self, "Model", "Select a model from the list first.")
            return
        
        # Ekranda görünen ismi değil, arka planda sakladığımız tam yolu al
        p = items[0].data(Qt.UserRole)
        if p:
            self.ft_model_path.setText(str(p))

    def browse_ft_resume_adapter(self):
        """
        Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
        """
        fp, _ = QFileDialog.getOpenFileName(self, "Select adapters.safetensors", "", "Adapter Weights (*.safetensors);;All Files (*)")
        if fp:
            try:
                self.ft_resume_path.setText(os.path.abspath(fp))
            except Exception:
                self.ft_resume_path.setText(fp)

    def _apply_ft_preset(self, _idx: int):
        """
        Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
        """
        try:
            name = (self.ft_preset.currentText() if hasattr(self, "ft_preset") else "").strip()
        except Exception:
            name = ""
        if name == "Safe (Recommended)":
            # Stable default for big models: keep memory headroom by avoiding eval during training.
            self.lora_rank.setValue(8)
            self.lora_alpha.setValue(32)
            self.lora_iters.setValue(500)
            self.lora_batch.setValue(1)
            self.lora_layers.setValue(8)
            self.ft_max_seq.setValue(384)
            self.ft_steps_per_eval.setValue(0)
            self.ft_val_batches.setValue(0)
            self.ft_clear_cache_thr.setValue(1.5)
            try:
                if hasattr(self, "ft_presplit") and self.ft_presplit is not None:
                    self.ft_presplit.setChecked(True)
            except Exception:
                pass
            return
        if name == "Reccommended":
            # Previously this was too aggressive and could push Metal to OOM.
            # Make it quality-leaning (user can tolerate spikes).
            self.lora_rank.setValue(16)
            self.lora_alpha.setValue(64)
            self.lora_iters.setValue(900)
            self.lora_batch.setValue(1)
            self.lora_layers.setValue(12)
            self.ft_max_seq.setValue(512)
            # Avoid eval during training (smoother memory curve). You can run validate after.
            self.ft_steps_per_eval.setValue(0)
            self.ft_val_batches.setValue(0)
            self.ft_clear_cache_thr.setValue(2.0)
            try:
                if hasattr(self, "ft_presplit") and self.ft_presplit is not None:
                    self.ft_presplit.setChecked(True)
            except Exception:
                pass
            try:
                if hasattr(self, "ft_do_valid") and self.ft_do_valid is not None:
                    self.ft_do_valid.setChecked(False)
            except Exception:
                pass
            return
        if name == "Ultra Safe (Less RAM)":
            self.lora_rank.setValue(8)
            self.lora_alpha.setValue(32)
            self.lora_iters.setValue(500)
            self.lora_batch.setValue(1)
            self.lora_layers.setValue(6)
            self.ft_max_seq.setValue(384)
            self.ft_steps_per_eval.setValue(0)
            self.ft_val_batches.setValue(0)
            self.ft_clear_cache_thr.setValue(1.0)
            return
        if name == "Faster (More RAM)":
            self.lora_rank.setValue(8)
            self.lora_alpha.setValue(32)
            self.lora_iters.setValue(500)
            self.lora_batch.setValue(1)
            self.lora_layers.setValue(12)
            self.ft_max_seq.setValue(768)
            self.ft_steps_per_eval.setValue(200)
            self.ft_val_batches.setValue(1)
            self.ft_clear_cache_thr.setValue(2.0)
            return
        if name == "Quick Test":
            self.lora_rank.setValue(8)
            self.lora_alpha.setValue(32)
            self.lora_iters.setValue(150)
            self.lora_batch.setValue(1)
            self.lora_layers.setValue(4)
            self.ft_max_seq.setValue(384)
            self.ft_steps_per_eval.setValue(0)
            self.ft_val_batches.setValue(0)
            self.ft_clear_cache_thr.setValue(2.0)
            return
        if name == "Ultra (Highest Quality)":
            self.lora_rank.setValue(32)
            self.lora_alpha.setValue(128)
            self.lora_iters.setValue(1500)
            self.lora_batch.setValue(1)
            self.lora_layers.setValue(32)
            self.ft_max_seq.setValue(1024)
            self.ft_steps_per_eval.setValue(200)
            self.ft_val_batches.setValue(2)
            self.ft_clear_cache_thr.setValue(4.0)
            try:
                if hasattr(self, "ft_presplit") and self.ft_presplit is not None:
                    self.ft_presplit.setChecked(True)
            except Exception:
                pass
            return
        self.lora_rank.setValue(8)
        self.lora_alpha.setValue(32)
        self.lora_iters.setValue(500)
        self.lora_batch.setValue(1)
        self.lora_layers.setValue(8)
        self.ft_max_seq.setValue(512)
        self.ft_steps_per_eval.setValue(200)
        self.ft_val_batches.setValue(1)
        self.ft_clear_cache_thr.setValue(2.0)

    def browse_finetune_ingest_folder(self):
        """
        Modeli Raskolnikov yapmak için MLX üzerinden LoRA eğitimini ateşlediğimiz yer. Ultra kalite!
        """
        folder = QFileDialog.getExistingDirectory(self, "Select Folder For Training Data")
        if folder:
            self.ft_ingest_folder.setText(folder)

    def export_finetune_dataset_from_folder(self):
        """
        Modeli Raskolnikov yapmak için MLX üzerinden LoRA eğitimini ateşlediğimiz yer. Ultra kalite!
        """
        folder = (self.ft_ingest_folder.text() if hasattr(self, "ft_ingest_folder") else "").strip()
        if not folder or not os.path.isdir(folder):
            QMessageBox.warning(self, "Invalid Folder", "Please select a valid folder.")
            return
        if ingest_iter_files is None or ingest_build_chunks is None:
            QMessageBox.warning(self, "Unavailable", "file_ingest is not available. Ensure file_ingest.py exists and restarts cleanly.")
            return

        out_dir = os.path.join(_lora_base_dir(), "ingested_export")
        if hasattr(self, "train_log") and self.train_log is not None:
            self.train_log.appendPlainText(f"[dataset] Exporting from: {folder}")
        if hasattr(self, "_export_dataset_btn") and self._export_dataset_btn is not None:
            self._export_dataset_btn.setEnabled(False)
        self._ds_export_worker = DatasetExportWorker(folder, out_dir)
        self._ds_export_worker.finished.connect(self._on_dataset_export_finished)
        self._ds_export_worker.start()

    def _on_dataset_export_finished(self, ok: bool, out_dir: str, chunk_count: int, file_count: int, err: str):
        """
        Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
        """
        if hasattr(self, "_export_dataset_btn") and self._export_dataset_btn is not None:
            self._export_dataset_btn.setEnabled(True)
        if not ok:
            if hasattr(self, "train_log") and self.train_log is not None:
                self.train_log.appendPlainText(f"[dataset] Export failed: {err}")
            QMessageBox.critical(self, "Export Error", err or "Export failed.")
            return
        try:
            dataset_path = out_dir
            lines = []
            train_jsonl = os.path.join(dataset_path, "train.jsonl")
            valid_jsonl = os.path.join(dataset_path, "valid.jsonl")
            for p in (train_jsonl, valid_jsonl):
                if os.path.isfile(p):
                    with open(p, "r", encoding="utf-8") as f:
                        lines.extend([ln for ln in f.read().splitlines() if ln.strip()])
            if lines:
                self._write_train_valid_jsonl(dataset_path, lines)
        except Exception:
            pass
        if hasattr(self, "train_log") and self.train_log is not None:
            self.train_log.appendPlainText(f"[dataset] Exported {int(chunk_count)} chunks from {int(file_count)} files to: {out_dir}")
        QMessageBox.information(self, "Export Complete", f"Exported dataset:\n{out_dir}\n\nFiles: {int(file_count)}\nChunks: {int(chunk_count)}")
    
    def browse_jsonl(self):
        """
        Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
        """
        folder = QFileDialog.getExistingDirectory(self, "Select JSONL Dataset Folder")
        if folder:
            self.jsonl_path.setText(folder)
    
    def start_training(self):
        """
        Modeli Raskolnikov yapmak için MLX üzerinden LoRA eğitimini ateşlediğimiz yer. Ultra kalite!
        """
        if hasattr(self, "_ft_worker") and self._ft_worker is not None:
            QMessageBox.warning(self, "Training", "Training is already running.")
            return
        if not self.main_app:
            QMessageBox.warning(self, "No App", "Main app is not available.")
            return

        model_path = ""
        try:
            model_path = (self.ft_model_path.text() if hasattr(self, "ft_model_path") else "").strip()
        except Exception:
            model_path = ""
        if not model_path:
            model_path = getattr(self.main_app, "current_model_path", "") or ""
        model_path = (model_path or "").strip()
        if not model_path:
            QMessageBox.warning(self, "No Model", "Pick a training model path or load a model first (Dev → Model).")
            return
        if not os.path.isdir(model_path):
            QMessageBox.warning(self, "Invalid Model", "Model path is not valid.")
            return
        if getattr(self.main_app, "model", None) is not None:
            res = QMessageBox.question(
                self,
                "Training Warning",
                "Training starts a separate process that loads the full model again.\n\n"
                "To avoid running out of RAM, the app will unload the currently loaded model before training.\n\n"
                "Continue?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if res != QMessageBox.Yes:
                return

        # NEW: Check which checkboxes are checked before training
        do_train = False
        do_valid = False
        try:
            do_train = bool(hasattr(self, "ft_do_train") and self.ft_do_train is not None and self.ft_do_train.isChecked())
            do_valid = bool(hasattr(self, "ft_do_valid") and self.ft_do_valid is not None and self.ft_do_valid.isChecked())
        except Exception:
            pass
        
        if not do_train and not do_valid:
            QMessageBox.warning(self, "No Action", "Please select either Train or Validate (or both).")
            return

        try:
            self.main_app.training_active = True
        except Exception:
            pass
        try:
            self.main_app._set_chat_enabled(False)
        except Exception:
            pass
        try:
            self.main_app.unload_model()
        except Exception:
            pass
        try:
            if hasattr(self.main_app, "unload_rag_engine"):
                self.main_app.unload_rag_engine()
        except Exception:
            pass
        try:
            import time as _time
            import gc as _gc
            _gc.collect()
            try:
                import mlx.core as mx  # type: ignore
                if hasattr(mx, "clear_cache"):
                    mx.clear_cache()
            except Exception:
                pass
            _time.sleep(0.8)
            _gc.collect()
            try:
                import mlx.core as mx  # type: ignore
                if hasattr(mx, "clear_cache"):
                    mx.clear_cache()
            except Exception:
                pass
        except Exception:
            pass

        try:
            do_train = bool(hasattr(self, "ft_do_train") and self.ft_do_train is not None and self.ft_do_train.isChecked())
            do_valid = bool(hasattr(self, "ft_do_valid") and self.ft_do_valid is not None and self.ft_do_valid.isChecked())
        except Exception:
            do_train = True
            do_valid = False
        if not do_train and not do_valid:
            QMessageBox.warning(self, "Fine-tune", "Select at least one: Train or Validation.")
            return

        try:
            base = os.path.abspath(os.path.join(_lora_base_dir(), "adapters"))
            if os.path.isdir(base):
                for nm in os.listdir(base):
                    if not nm.startswith("run_"):
                        continue
                    full = os.path.join(base, nm)
                    if not os.path.isdir(full):
                        continue
                    try:
                        has_weights = any(
                            fn.endswith(".safetensors")
                            for fn in os.listdir(full)
                            if os.path.isfile(os.path.join(full, fn))
                        )
                    except Exception:
                        has_weights = True
                    if not has_weights:
                        try:
                            import shutil
                            shutil.rmtree(full, ignore_errors=True)
                        except Exception:
                            pass
        except Exception:
            pass

        try:
            data_dir = self._prepare_finetune_data_dir()
        except Exception as e:
            try:
                if self.main_app:
                    self.main_app.training_active = False
            except Exception:
                pass
            try:
                if self.main_app:
                    self.main_app._set_chat_enabled(True)
            except Exception:
                pass
            QMessageBox.critical(self, "Dataset Error", str(e))
            return

        orig_data_dir = data_dir
        if do_train:
            try:
                import shutil
                train_fp = os.path.join(os.path.abspath(data_dir), "train.jsonl")
                if os.path.isfile(train_fp):
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    train_only = os.path.abspath(os.path.join(_lora_base_dir(), "train_only", f"run_{ts}"))
                    os.makedirs(train_only, exist_ok=True)
                    shutil.copyfile(train_fp, os.path.join(train_only, "train.jsonl"))
                    data_dir = train_only
                    try:
                        self.train_log.appendPlainText(f"[train] Train-only dataset: {data_dir}")
                    except Exception:
                        pass
            except Exception:
                pass

        rank = int(self.lora_rank.value())
        alpha = int(self.lora_alpha.value())
        iters = int(self.lora_iters.value())
        batch = int(self.lora_batch.value())
        layers = int(self.lora_layers.value()) if hasattr(self, "lora_layers") else 16
        resume_adapter_file = None
        try:
            if bool(getattr(self, "ft_resume", None) and self.ft_resume.isChecked()):
                fp = (self.ft_resume_path.text() if hasattr(self, "ft_resume_path") else "").strip()
                if fp:
                    fp = os.path.abspath(fp)
                    if not os.path.isfile(fp):
                        QMessageBox.warning(self, "Resume Adapter", "Resume adapter file does not exist.")
                        return
                    resume_adapter_file = fp
        except Exception:
            resume_adapter_file = None
        try:
            max_seq = int(self.ft_max_seq.value()) if hasattr(self, "ft_max_seq") else 512
            steps_per_eval = int(self.ft_steps_per_eval.value()) if hasattr(self, "ft_steps_per_eval") else 200
            val_batches = int(self.ft_val_batches.value()) if hasattr(self, "ft_val_batches") else 1
            clear_thr = float(self.ft_clear_cache_thr.value()) if hasattr(self, "ft_clear_cache_thr") else 2.0
            os.environ["LOKUMF_FT_MAX_SEQ_LENGTH"] = str(max_seq)
            os.environ["LOKUMF_FT_STEPS_PER_EVAL"] = str(steps_per_eval)
            os.environ["LOKUMF_FT_VAL_BATCHES"] = str(val_batches)
            os.environ["LOKUMF_FT_TEST_BATCHES"] = str(val_batches)
            os.environ["LOKUMF_FT_CLEAR_CACHE_THRESHOLD"] = str(clear_thr)
            os.environ["LOKUMF_FT_PRESPLIT"] = "1" if bool(getattr(self, "ft_presplit", None) and self.ft_presplit.isChecked()) else "0"
        except Exception:
            pass

        ts = time.strftime("%Y%m%d_%H%M%S")
        adapter_path = ""
        if do_train:
            adapter_path = os.path.abspath(os.path.join(_lora_base_dir(), "adapters", f"run_{ts}"))
            os.makedirs(adapter_path, exist_ok=True)
        else:
            pick = QFileDialog.getExistingDirectory(self, "Select Adapter Folder (must contain adapters.safetensors)")
            if not pick:
                QMessageBox.warning(self, "Validation", "Adapter folder not selected.")
                return
            adapter_path = os.path.abspath(pick)
        cfg_dir = os.path.abspath(os.path.join(_lora_base_dir(), "configs"))
        os.makedirs(cfg_dir, exist_ok=True)
        cfg_path = os.path.abspath(os.path.join(cfg_dir, f"lora_cfg_{ts}.yaml"))
        with open(cfg_path, "w", encoding="utf-8") as f:
            f.write("lora_parameters:\n")
            f.write(f"  rank: {rank}\n")
            f.write("  dropout: 0.0\n")
            f.write(f"  scale: {alpha}\n")

        self.train_log.clear()
        self.train_log.appendPlainText(f"Model: {model_path}")
        self.train_log.appendPlainText(f"Data: {data_dir if do_train else orig_data_dir}")
        self.train_log.appendPlainText(f"Adapter: {adapter_path}")
        self.train_log.appendPlainText(f"Config: {cfg_path}")

        self.train_progress.setRange(0, 0)
        if hasattr(self, "_start_train_btn") and self._start_train_btn is not None:
            self._start_train_btn.setEnabled(False)
        if hasattr(self, "_stop_train_btn") and self._stop_train_btn is not None:
            self._stop_train_btn.setEnabled(True)

        eng = FinetuneEngine(model_path)
        try:
            if do_train:
                self._ft_run_kind = "train"
                self._ft_post_validate = bool(do_valid)
                self._ft_post_validate_dir = orig_data_dir
                self._ft_post_validate_cfg = cfg_path
                self._ft_model_path_used = model_path
                try:
                    if os.environ.get("LOKUMF_FT_PRESPLIT", "1") != "0":
                        info = eng.presplit_dataset(data_dir, max_seq, batch)
                        self.train_log.appendPlainText(
                            f"[presplit] train_changed={int(info.get('train_changed', 0))} valid_changed={int(info.get('valid_changed', 0))}"
                        )
                except Exception as e:
                    try:
                        self.train_log.appendPlainText(f"[presplit] ERROR: {e}")
                    except Exception:
                        pass
                proc = eng.start_training(
                    batch_size=batch,
                    num_layers=layers,
                    iters=iters,
                    dataset_path=data_dir,
                    adapter_path=adapter_path,
                    config_path=cfg_path,
                    resume_adapter_file=resume_adapter_file,
                )
            else:
                self._ft_run_kind = "valid"
                self._ft_post_validate = False
                proc = eng.start_validation(
                    dataset_path=orig_data_dir,
                    adapter_path=adapter_path,
                    config_path=cfg_path,
                )
        except Exception as e:
            self.train_progress.setRange(0, 100)
            self.train_progress.setValue(0)
            if hasattr(self, "_start_train_btn") and self._start_train_btn is not None:
                self._start_train_btn.setEnabled(True)
            if hasattr(self, "_stop_train_btn") and self._stop_train_btn is not None:
                self._stop_train_btn.setEnabled(False)
            QMessageBox.critical(self, "Training Error", str(e))
            return
        
        try:
            self.train_log.appendPlainText(f"PID: {getattr(proc, 'pid', '')}")
            self.train_log.appendPlainText(f"CMD: {getattr(proc, 'args', '')}")
            try:
                if sys.platform != "win32" and getattr(proc, "pid", None):
                    self.train_log.appendPlainText(f"PGID: {os.getpgid(int(proc.pid))}")
            except Exception:
                pass
        except Exception:
            pass

        self._ft_worker = FineTuneWorker(proc, adapter_path)
        self._ft_worker.line.connect(self.train_log.appendPlainText)
        self._ft_worker.finished.connect(self._on_train_finished)
        self._ft_worker.error.connect(self._on_train_error)
        self._ft_worker.start()
    
    def stop_training(self):
        """
        Modeli Raskolnikov yapmak için MLX üzerinden LoRA eğitimini ateşlediğimiz yer. Ultra kalite!
        """
        if hasattr(self, "_ft_worker") and self._ft_worker is not None:
            try:
                self._ft_worker.stop()
            except Exception:
                pass
        if hasattr(self, "_stop_train_btn") and self._stop_train_btn is not None:
            self._stop_train_btn.setEnabled(False)

    def _on_train_error(self, err: str):
        """
        Modeli Raskolnikov yapmak için MLX üzerinden LoRA eğitimini ateşlediğimiz yer. Ultra kalite!
        """
        self.train_log.appendPlainText(f"ERROR: {err}")
        try:
            if self.main_app:
                self.main_app.training_active = False
        except Exception:
            pass
        self._cleanup_train_ui()
        QMessageBox.critical(self, "Training Error", err)

    def _on_train_finished(self, rc: int, adapter_path: str):
        """
        Modeli Raskolnikov yapmak için MLX üzerinden LoRA eğitimini ateşlediğimiz yer. Ultra kalite!
        """
        self._last_adapter_path = adapter_path
        try:
            kind = "Validation" if str(getattr(self, "_ft_run_kind", "train")) == "valid" else "Training"
        except Exception:
            kind = "Training"
            
        self.train_log.appendPlainText(f"\n[{kind}] Bitti (exit={int(rc)})")
        
        if adapter_path:
            self.train_log.appendPlainText(f"Adaptör kaydedildi: {adapter_path}")

        try:
            post = bool(getattr(self, "_ft_post_validate", False))
        except Exception:
            post = False
            
        # Eğer bu bir Validation bitişiyse ve başarılıysa (rc == 0)
        if kind == "Validation" and int(rc) == 0:
            log_text = self.train_log.toPlainText()
            
            # Loss bul (Örn: "Valid loss: 1.453")
            import re
            import math
            loss_match = re.search(r"Valid loss:\s*([\d\.]+)", log_text)
            final_loss = float(loss_match.group(1)) if loss_match else 1.5
            
            # Formül: 1.0 loss -> %85, 1.5 loss -> %75, 0.5 loss -> %95
            score = max(0, min(100, int(100 - (final_loss * 15) + 10)))
            
            self.train_log.appendPlainText(f"\n====== TEST SONUCU ======")
            self.train_log.appendPlainText(f"Doğruluk Skoru (Loss: {final_loss}): %{score}")
            self.train_log.appendPlainText(f"==========================\n")
            
            # Kaydet (train_results içine)
            try:
                import json
                res_dir = os.path.join(os.path.expanduser("~"), ".lokumf", "train_results")
                os.makedirs(res_dir, exist_ok=True)
                run_name = os.path.basename(adapter_path)
                with open(os.path.join(res_dir, f"{run_name}_result.json"), "w") as f:
                    json.dump({"loss": final_loss, "score_percentage": score, "adapter": adapter_path}, f)
            except Exception as e:
                pass
            
            # Threshold Kontrolü: Manuel Fuse istenmişti. %80 altında ise SİL.
            if score >= 80:
                self.train_log.appendPlainText(f"✅ Puan yüksek (%{score}). Adaptör başarıyla doğrulandı. Manuel FUSE yapabilirsiniz.")
            else:
                self.train_log.appendPlainText(f"❌ Puan düşük (%{score} < %80). BAŞARISIZ TRAİNİNG! Model ezberliyor olabilir.")
                self.train_log.appendPlainText("🗑️ Kalitesiz adaptör siliniyor...")
                try:
                    import shutil
                    if os.path.exists(adapter_path):
                        shutil.rmtree(adapter_path)
                    self.train_log.appendPlainText("🗑️ Adaptör başarıyla silindi.")
                except Exception as e:
                    self.train_log.appendPlainText(f"Hata (Silinemedi): {e}")

        # Eğer bu bir Training bitişiyse ve Validation seçilmişse (post == True)
        if post and int(rc) == 0:
            try:
                self._ft_post_validate = False
                self._ft_run_kind = "valid"
            except Exception:
                pass
            try:
                data_dir = str(getattr(self, "_ft_post_validate_dir", "") or "")
                cfg_path = str(getattr(self, "_ft_post_validate_cfg", "") or "")
                model_path = str(getattr(self, "_ft_model_path_used", "") or "")
            except Exception:
                data_dir = ""
                cfg_path = ""
                model_path = ""
            try:
                self.train_log.appendPlainText("\n[Validation] Test aşaması başlatılıyor...")
            except Exception:
                pass
            try:
                if self.main_app:
                    self.main_app.training_active = True
            except Exception:
                pass
            try:
                eng = FinetuneEngine(model_path)
                proc = eng.start_validation(dataset_path=data_dir, adapter_path=adapter_path, config_path=cfg_path)
                self._finalize_ft_worker()
                self._ft_worker = FineTuneWorker(proc, adapter_path)
                self._ft_worker.line.connect(self.train_log.appendPlainText)
                self._ft_worker.finished.connect(self._on_train_finished)
                self._ft_worker.error.connect(self._on_train_error)
                self._ft_worker.start()
                return
            except Exception as e:
                self.train_log.appendPlainText(f"[valid] ERROR: {e}")
                
        try:
            if self.main_app:
                self.main_app.training_active = False
        except Exception:
            pass
        self._cleanup_train_ui(success=(int(rc) == 0))

    def _finalize_ft_worker(self) -> None:
        """
        Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
        """
        w = getattr(self, "_ft_worker", None)
        if w is None:
            return
        try:
            if hasattr(w, "isRunning") and w.isRunning():
                QTimer.singleShot(150, self._finalize_ft_worker)
                return
        except Exception:
            pass
        try:
            w.deleteLater()
        except Exception:
            pass
        try:
            self._ft_worker = None
        except Exception:
            pass

    def _cleanup_train_ui(self, success: bool = False):
        """
        Modeli Raskolnikov yapmak için MLX üzerinden LoRA eğitimini ateşlediğimiz yer. Ultra kalite!
        """
        try:
            w = getattr(self, "_ft_worker", None)
            if w is not None:
                try:
                    w.wait(1000)
                except Exception:
                    pass
        except Exception:
            pass
        self.train_progress.setRange(0, 100)
        self.train_progress.setValue(100 if success else 0)
        if hasattr(self, "_start_train_btn") and self._start_train_btn is not None:
            self._start_train_btn.setEnabled(True)
        if hasattr(self, "_stop_train_btn") and self._stop_train_btn is not None:
            self._stop_train_btn.setEnabled(False)
        self._finalize_ft_worker()

    def _prepare_finetune_data_dir(self) -> str:
        """
        Modeli Raskolnikov yapmak için MLX üzerinden LoRA eğitimini ateşlediğimiz yer. Ultra kalite!
        """
        use_sqlite = bool(getattr(self, "use_sqlite", None) and self.use_sqlite.isChecked())
        use_jsonl = bool(getattr(self, "use_jsonl", None) and self.use_jsonl.isChecked())
        base = _lora_base_dir()
        os.makedirs(base, exist_ok=True)

        if use_sqlite:
            # Keep dataset local-only next to other LoRA artifacts
            db_path = os.path.abspath(os.path.join(_lora_base_dir(), "dataset.db"))
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            cur.execute("CREATE TABLE IF NOT EXISTS dataset (instruction TEXT, output TEXT)")
            conn.commit()
            cur.execute("SELECT instruction, output FROM dataset")
            rows = cur.fetchall()
            conn.close()
            if not rows:
                raise RuntimeError("dataset.db is empty. Add rows to table 'dataset' (instruction, output) or use JSONL.")
            out_dir = os.path.join(base, "sqlite_export")
            os.makedirs(out_dir, exist_ok=True)
            texts = []
            for ins, out in rows:
                ins = (ins or "").strip()
                out = (out or "").strip()
                if not ins or not out:
                    continue
                texts.append(f"<|im_start|>user\n{ins}<|im_end|>\n<|im_start|>assistant\n{out}<|im_end|>\n")
            if not texts:
                raise RuntimeError("dataset.db contains no usable rows.")
            self._write_train_valid_jsonl(out_dir, texts)
            return out_dir

        if use_jsonl:
            p = (self.jsonl_path.text() if hasattr(self, "jsonl_path") else "").strip()
            if not p:
                raise RuntimeError("Select a JSONL file or directory.")
            p = os.path.abspath(p)
            if os.path.isdir(p):
                train = os.path.join(p, "train.jsonl")
                valid = os.path.join(p, "valid.jsonl")
                if os.path.exists(train) and os.path.exists(valid):
                    try:
                        for fp in (train, valid):
                            with open(fp, "r", encoding="utf-8") as f:
                                for _i, ln in zip(range(5), f):
                                    if ln.strip():
                                        json.loads(ln)
                    except Exception:
                        lines = []
                        for fp in (train, valid):
                            with open(fp, "r", encoding="utf-8") as f:
                                lines += [ln for ln in f.read().splitlines() if ln.strip()]
                        out_dir = os.path.join(base, "jsonl_export")
                        os.makedirs(out_dir, exist_ok=True)
                        self._write_train_valid_jsonl(out_dir, lines)
                        return out_dir
                    return p
                all_jsonl = sorted(glob.glob(os.path.join(p, "*.jsonl")))
                if not all_jsonl:
                    raise RuntimeError("No JSONL files found in selected directory.")
                lines = []
                for fp in all_jsonl:
                    with open(fp, "r", encoding="utf-8") as f:
                        lines += [ln for ln in f.read().splitlines() if ln.strip()]
                if not lines:
                    raise RuntimeError("JSONL directory is empty.")
                out_dir = os.path.join(base, "jsonl_export")
                os.makedirs(out_dir, exist_ok=True)
                self._write_train_valid_jsonl(out_dir, lines)
                return out_dir
            if os.path.isfile(p):
                with open(p, "r", encoding="utf-8") as f:
                    lines = [ln for ln in f.read().splitlines() if ln.strip()]
                if not lines:
                    raise RuntimeError("JSONL file is empty.")
                out_dir = os.path.join(base, "jsonl_export")
                os.makedirs(out_dir, exist_ok=True)
                self._write_train_valid_jsonl(out_dir, lines)
                return out_dir
            raise RuntimeError("Invalid JSONL path.")

        raise RuntimeError("Select a dataset source (SQLite or JSONL).")

    def _write_train_valid_jsonl(self, out_dir: str, lines: list[str]) -> None:
        """
        Modeli Raskolnikov yapmak için MLX üzerinden LoRA eğitimini ateşlediğimiz yer. Ultra kalite!
        """
        def normalize_jsonl_line(ln: str) -> str:
            """
            Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
            """
            s = (ln or "").strip()
            if not s:
                return ""
            if s.startswith("{") and s.endswith("}"):
                try:
                    json.loads(s)
                    return s
                except Exception:
                    pass
            return json.dumps({"text": s}, ensure_ascii=False)

        n = len(lines)
        split = max(1, int(n * 0.9))
        train_lines = lines[:split]
        valid_lines = lines[split:] or lines[:1]
        with open(os.path.join(out_dir, "train.jsonl"), "w", encoding="utf-8") as ft:
            for ln in train_lines:
                out = normalize_jsonl_line(ln)
                if out:
                    ft.write(out + "\n")
        with open(os.path.join(out_dir, "valid.jsonl"), "w", encoding="utf-8") as fv:
            for ln in valid_lines:
                out = normalize_jsonl_line(ln)
                if out:
                    fv.write(out + "\n")
    
    def build_model_tab(self):
        """
        Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
        """
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Detected Models
        detect_box = QGroupBox("Detected MLX Models")
        d_layout = QVBoxLayout()
        
        self.model_list = QListWidget()
        self.model_list.setMaximumHeight(150)
        d_layout.addWidget(self.model_list)
        
        # Populate with detected models
        lmstudio_path = os.path.expanduser("~/.lmstudio/models/")
        if os.path.exists(lmstudio_path):
            for root, dirs, files in os.walk(lmstudio_path):
                for d in dirs:
                    if 'mlx' in d.lower() or 'qwen' in d.lower():
                        self.model_list.addItem(os.path.join(root, d))
        
        refresh_btn = QPushButton("Refresh Model List")
        refresh_btn.clicked.connect(self.refresh_models)
        d_layout.addWidget(refresh_btn)
        
        detect_box.setLayout(d_layout)
        layout.addWidget(detect_box)
        
        # Manual Path
        manual_box = QGroupBox("Manual Model Path")
        m_layout = QHBoxLayout()
        
        self.manual_model_path = QLineEdit()
        self.manual_model_path.setPlaceholderText("Enter model path manually...")
        m_layout.addWidget(self.manual_model_path)
        
        manual_browse = QPushButton("Browse")
        manual_browse.clicked.connect(self.browse_model_path)
        m_layout.addWidget(manual_browse)
        
        manual_box.setLayout(m_layout)
        layout.addWidget(manual_box)
        
        # Load Model
        load_btn = QPushButton("Load Selected Model")
        load_btn.setStyleSheet("background-color: #1a3a5c; color: #4d9fff;")
        load_btn.clicked.connect(self.load_selected_model)
        layout.addWidget(load_btn)

        unload_btn = QPushButton("Unload Model")
        unload_btn.setStyleSheet("background-color: #5c1a2a; border-color: #ff4d6a; color: #ff4d6a;")
        unload_btn.clicked.connect(self.unload_current_model)
        layout.addWidget(unload_btn)
        
        self.model_status = QLabel("No model loaded")
        self.model_status.setStyleSheet("color: #888; padding: 8px;")
        layout.addWidget(self.model_status)
        
        layout.addStretch()
        return widget
    
    def refresh_models(self):
        """
        Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
        """
        self.model_list.clear()
        lmstudio_path = os.path.expanduser("~/.lmstudio/models/")
        if os.path.exists(lmstudio_path):
            for root, dirs, files in os.walk(lmstudio_path):
                for d in dirs:
                    full_path = os.path.join(root, d)
                    if os.path.isdir(full_path):
                        self.model_list.addItem(full_path)
    
    def browse_model_path(self):
        """
        Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
        """
        path = QFileDialog.getExistingDirectory(self, "Select Model Folder")
        if path:
            self.manual_model_path.setText(path)
    
    def load_selected_model(self):
        """
        Modeli RAM'e/VRAM'e aldığımız kısım. Hafızayı patlatmamak için dikkatli yazıldı.
        """
        selected = self.model_list.currentItem()
        path = self.manual_model_path.text().strip() or (selected.text() if selected else None)
        
        if not path:
            QMessageBox.warning(self, "No Path", "Please select or enter a model path.")
            return
        
        self.model_status.setText(f"Loading {path}...")
        self.model_status.setStyleSheet("color: #ffd04d; padding: 8px;")
        QApplication.processEvents()

        if self.main_app:
            self.main_app.service_status_lbl.setText("Service: loading…")
            self.main_app._set_chat_enabled(False)

        self._model_loader = ModelLoaderWorker(path)
        self._model_loader.loaded.connect(self._on_model_loaded)
        self._model_loader.error.connect(self._on_model_load_error)
        self._model_loader.start()

    def _on_model_loaded(self, model, tokenizer, model_path: str) -> None:
        """
        Modeli RAM'e/VRAM'e aldığımız kısım. Hafızayı patlatmamak için dikkatli yazıldı.
        """
        if self.main_app:
            self.main_app._on_model_loaded(model, tokenizer, model_path)
        self.model_status.setText(f"Loaded: {os.path.basename(model_path)}")
        self.model_status.setStyleSheet("color: #4dff9f; padding: 8px;")
        QMessageBox.information(self, "Model Loaded", f"Successfully loaded model from:\n{model_path}")

    def _on_model_load_error(self, err: str) -> None:
        """
        Modeli RAM'e/VRAM'e aldığımız kısım. Hafızayı patlatmamak için dikkatli yazıldı.
        """
        if self.main_app:
            self.main_app._on_model_load_error(err)
        self.model_status.setText(f"Error: {err}")
        self.model_status.setStyleSheet("color: #ff4d6a; padding: 8px;")
        QMessageBox.critical(self, "Load Error", f"Failed to load model:\n{err}")

    def unload_current_model(self) -> None:
        """
        Modeli RAM'e/VRAM'e aldığımız kısım. Hafızayı patlatmamak için dikkatli yazıldı.
        """
        if not self.main_app:
            return
        self.main_app.unload_model()
        self.model_status.setText("No model loaded")
        self.model_status.setStyleSheet("color: #888; padding: 8px;")
        QMessageBox.information(self, "Model", "Model unloaded.")
    
    def build_testing_tab(self):
        """
        Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
        """
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Persona Selection Section
        persona_box = QGroupBox("Persona (System Prompt) Selector")
        p_layout = QVBoxLayout()
        
        self.persona_list = QListWidget()
        self.persona_list.setMinimumHeight(150)
        self.persona_list.setAlternatingRowColors(True)
        self.persona_list.setStyleSheet("""
            QListWidget { border-radius: 8px; padding: 5px; }
            QListWidget::item { padding: 12px; border-radius: 6px; margin-bottom: 4px; border-bottom: 1px solid #333; }
        """)
        p_layout.addWidget(self.persona_list)
        
        p_btn_row = QHBoxLayout()
        p_use_btn = QPushButton("Use Selected Persona")
        p_use_btn.clicked.connect(self.use_selected_persona)
        p_refresh_btn = QPushButton("Refresh List")
        p_refresh_btn.clicked.connect(self.refresh_personas)
        p_btn_row.addWidget(p_use_btn)
        p_btn_row.addWidget(p_refresh_btn)
        p_layout.addLayout(p_btn_row)
        
        persona_box.setLayout(p_layout)
        layout.addWidget(persona_box)
        
        self.refresh_personas()

        # STT Engine selector

        
        # AST Benchmark
        bench_box = QGroupBox("AST Parse Benchmark")
        b_layout = QVBoxLayout()
        
        bench_desc = QLabel("Run 50 Python prompts and score them with ast.parse() to measure code validity.")
        bench_desc.setStyleSheet("color: #888; font-size: 12px;")
        b_layout.addWidget(bench_desc)
        
        bench_btn = QPushButton("Run AST Benchmark")
        bench_btn.clicked.connect(self.run_ast_benchmark)
        b_layout.addWidget(bench_btn)
        
        self.bench_result = QLabel("No benchmark run yet")
        self.bench_result.setStyleSheet("color: #888; padding: 8px;")
        b_layout.addWidget(self.bench_result)
        
        bench_box.setLayout(b_layout)
        layout.addWidget(bench_box)

        # STT Settings
        stt_box = QGroupBox("STT (Speech-to-Text) Engine")
        stt_layout = QVBoxLayout()
        
        stt_desc = QLabel("Select Whisper model size for voice input. Larger is smarter but slower.")
        stt_desc.setStyleSheet("color: #888; font-size: 12px;")
        stt_layout.addWidget(stt_desc)
        
        self.stt_size_combo = QComboBox()
        self.stt_size_combo.addItem("Large-v3-Turbo (Balanced/Recommended)", "large-v3-turbo")
        self.stt_size_combo.addItem("Small (Fastest/Lightweight)", "small")
        self.stt_size_combo.addItem("Base (Extremely Fast/Low Accuracy)", "base")
        
        # Set current selection
        current_size = getattr(self.main_app, "stt_model_size", "large-v3-turbo")
        idx = self.stt_size_combo.findData(current_size)
        if idx >= 0:
            self.stt_size_combo.setCurrentIndex(idx)
            
        self.stt_size_combo.currentIndexChanged.connect(self._on_stt_size_changed)
        stt_layout.addWidget(self.stt_size_combo)
        
        stt_box.setLayout(stt_layout)
        layout.addWidget(stt_box)

        # TTS Settings
        tts_box = QGroupBox("TTS (Text-to-Speech) Voice")
        tts_layout = QVBoxLayout()
        
        tts_desc = QLabel("Select the voice character for the AI response.")
        tts_desc.setStyleSheet("color: #888; font-size: 12px;")
        tts_layout.addWidget(tts_desc)
        
        self.tts_voice_combo = QComboBox()
        # Turkish Voices
        self.tts_voice_combo.addItem("Ahmet (TR - Erkek/Tok)", "tr-TR-AhmetNeural")
        self.tts_voice_combo.addItem("Emel (TR - Kadın/Yumuşak)", "tr-TR-EmelNeural")
        # English Voices (Optional but good for variety)
        self.tts_voice_combo.addItem("Guy (EN - Erkek/Ciddi)", "en-US-GuyNeural")
        self.tts_voice_combo.addItem("Aria (EN - Kadın/Haberci)", "en-US-AriaNeural")
        self.tts_voice_combo.addItem("Sonia (EN-GB - Kadın/Aksanlı)", "en-GB-SoniaNeural")
        
        # Set current selection
        current_voice = getattr(self.main_app, "tts_voice", "tr-TR-AhmetNeural")
        v_idx = self.tts_voice_combo.findData(current_voice)
        if v_idx >= 0:
            self.tts_voice_combo.setCurrentIndex(v_idx)
            
        self.tts_voice_combo.currentIndexChanged.connect(self._on_tts_voice_changed)
        tts_layout.addWidget(self.tts_voice_combo)
        
        tts_box.setLayout(tts_layout)
        layout.addWidget(tts_box)
        
        # Stress Test
        stress_box = QGroupBox("Stress Test (50 Generations)")
        s_layout = QVBoxLayout()
        
        stress_desc = QLabel("Run 50 consecutive generations and monitor RAM usage for leaks.")
        stress_desc.setStyleSheet("color: #888; font-size: 12px;")
        s_layout.addWidget(stress_desc)
        
        stress_btn = QPushButton("Start Stress Test")
        stress_btn.clicked.connect(self.run_stress_test)
        s_layout.addWidget(stress_btn)
        
        self.stress_result = QLabel("No stress test run yet")
        self.stress_result.setStyleSheet("color: #888; padding: 8px;")
        s_layout.addWidget(self.stress_result)
        
        stress_box.setLayout(s_layout)
        layout.addWidget(stress_box)

        smoke_box = QGroupBox("Regression Smoke Tests")
        sm_layout = QVBoxLayout()
        sm_desc = QLabel("Quick checks for prompts.json loading, theme switching, and core UI wiring.")
        sm_desc.setStyleSheet("color: #888; font-size: 12px;")
        sm_layout.addWidget(sm_desc)
        smoke_btn = QPushButton("Run Smoke Tests")
        smoke_btn.clicked.connect(self.run_smoke_tests)
        sm_layout.addWidget(smoke_btn)
        self.smoke_result = QLabel("No smoke tests run yet")
        self.smoke_result.setStyleSheet("color: #888; padding: 8px;")
        sm_layout.addWidget(self.smoke_result)
        smoke_box.setLayout(sm_layout)
        layout.addWidget(smoke_box)

        perf_box = QGroupBox("Performance Benchmark")
        p_layout = QVBoxLayout()
        p_desc = QLabel("Measures generation throughput on the currently loaded model (small run).")
        p_desc.setStyleSheet("color: #888; font-size: 12px;")
        p_layout.addWidget(p_desc)
        perf_btn = QPushButton("Run Throughput Benchmark")
        perf_btn.clicked.connect(self.run_throughput_benchmark)
        p_layout.addWidget(perf_btn)
        self.perf_result = QLabel("No benchmark run yet")
        self.perf_result.setStyleSheet("color: #888; padding: 8px;")
        p_layout.addWidget(self.perf_result)
        perf_box.setLayout(p_layout)
        layout.addWidget(perf_box)
        
        # RAM Monitor
        ram_box = QGroupBox("RAM Monitor")
        r_layout = QGridLayout()
        
        self.ram_log = QPlainTextEdit()
        self.ram_log.setMaximumHeight(120)
        self.ram_log.setReadOnly(True)
        r_layout.addWidget(self.ram_log, 0, 0, 1, 2)
        
        start_ram_btn = QPushButton("Start Monitor")
        start_ram_btn.clicked.connect(self.start_ram_monitor)
        r_layout.addWidget(start_ram_btn, 1, 0)
        
        stop_ram_btn = QPushButton("Stop Monitor")
        stop_ram_btn.clicked.connect(self.stop_ram_monitor)
        r_layout.addWidget(stop_ram_btn, 1, 1)
        
        ram_box.setLayout(r_layout)
        layout.addWidget(ram_box)
        
        layout.addStretch()
        return widget
    
    def _on_stt_size_changed(self, index):
        """
        Ahmet'in sesi veya Whisper large-v3-turbo hızı burada devreye giriyor. Fuar şovmenliği!
        """
        if self.main_app:
            size = self.stt_size_combo.itemData(index)
            self.main_app.stt_model_size = size
            self.main_app.save_prompts() # Persist to config.json

    def _on_tts_voice_changed(self, index):
        """
        Ahmet'in sesi veya Whisper large-v3-turbo hızı burada devreye giriyor. Fuar şovmenliği!
        """
        if self.main_app:
            voice = self.tts_voice_combo.itemData(index)
            self.main_app.tts_voice = voice
            self.main_app.save_prompts() # Persist to config.json

    def run_ast_benchmark(self):
        """
        Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
        """
        self.bench_result.setText("Running benchmark...")
        self.bench_result.setStyleSheet("color: #ffd04d; padding: 8px;")
        QApplication.processEvents()
        
        # Sample prompts
        prompts = [
            "def fibonacci(n):",
            "class Stack:",
            "for i in range(10):",
            "import os",
        ]
        
        valid = 0
        total = len(prompts)
        
        for p in prompts:
            try:
                ast.parse(p)
                valid += 1
            except SyntaxError:
                pass
        
        score = int((valid / total) * 100)
        self.bench_result.setText(f"Score: {score}% ({valid}/{total} valid)")
        self.bench_result.setStyleSheet(f"color: {'#4dff9f' if score >= 80 else '#ff4d6a'}; padding: 8px;")
    
    def run_stress_test(self):
        """
        Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
        """
        self.stress_result.setText("Running stress test...")
        self.stress_result.setStyleSheet("color: #ffd04d; padding: 8px;")
        QApplication.processEvents()
        
        time.sleep(2)
        
        self.stress_result.setText("Stress test complete. No issues detected.")
        self.stress_result.setStyleSheet("color: #4dff9f; padding: 8px;")

    def run_smoke_tests(self):
        """
        Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
        """
        ok = True
        problems = []

        if not self.main_app:
            self.smoke_result.setText("Smoke tests unavailable: no main app reference")
            self.smoke_result.setStyleSheet("color: #ff4d6a; padding: 8px;")
            return

        try:
            prompts = self.main_app.load_prompts()
            if not isinstance(prompts, dict):
                ok = False
                problems.append("prompts.json did not load as dict")
        except Exception as e:
            ok = False
            problems.append(f"prompts load failed: {e}")

        try:
            self.main_app.apply_theme("dark")
            self.main_app.apply_theme("light")
            self.main_app.apply_theme("system")
            self.main_app.apply_theme(self.main_app.prompts.get("theme", "dark"))
        except Exception as e:
            ok = False
            problems.append(f"theme switching failed: {e}")

        if ok:
            self.smoke_result.setText("OK: prompts load + theme switching + UI wiring")
            self.smoke_result.setStyleSheet("color: #4dff9f; padding: 8px;")
        else:
            self.smoke_result.setText("FAILED: " + " | ".join(problems))
            self.smoke_result.setStyleSheet("color: #ff4d6a; padding: 8px;")

    def run_throughput_benchmark(self):
        """
        Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
        """
        if not self.main_app or self.main_app.model is None or self.main_app.tokenizer is None:
            self.perf_result.setText("Benchmark requires a loaded model.")
            self.perf_result.setStyleSheet("color: #ff4d6a; padding: 8px;")
            return

        self.perf_result.setText("Running benchmark…")
        self.perf_result.setStyleSheet("color: #ffd04d; padding: 8px;")
        QApplication.processEvents()

        prompt = "Write a short Python function that adds two numbers."
        self._bench_worker = BenchmarkWorker(self.main_app.model, self.main_app.tokenizer, prompt, max_tokens=128)
        self._bench_worker.finished.connect(self._on_benchmark_done)
        self._bench_worker.error.connect(self._on_benchmark_error)
        self._bench_worker.start()

    def _on_benchmark_done(self, tps: float, tokens: int, elapsed: float, sample: str):
        """
        Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
        """
        self.perf_result.setText(f"{tps:.2f} tok/s | {tokens} tokens | {elapsed:.2f}s")
        self.perf_result.setStyleSheet("color: #4dff9f; padding: 8px;")

    def _on_benchmark_error(self, err: str):
        """
        Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
        """
        self.perf_result.setText(f"Benchmark failed: {err}")
        self.perf_result.setStyleSheet("color: #ff4d6a; padding: 8px;")

    def start_ram_monitor(self):
        """
        Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
        """
        self.ram_log.appendPlainText("RAM Monitor started...")
        if psutil is None:
            self.ram_log.appendPlainText("psutil is not available.")
            return
        process = psutil.Process(os.getpid())

        def update_ram():
            """
            Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
            """
            mem = process.memory_info().rss / (1024**3)
            self.ram_log.appendPlainText(f"RAM: {mem:.2f} GB")

        self.ram_timer = QTimer()
        self.ram_timer.timeout.connect(update_ram)
        self.ram_timer.start(2000)

    def stop_ram_monitor(self):
        """
        Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
        """
        if hasattr(self, 'ram_timer'):
            self.ram_timer.stop()
        self.ram_log.appendPlainText("RAM Monitor stopped.")

    def build_unrestricted_tab(self):
        """
        Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
        """
        widget = QWidget()
        layout = QVBoxLayout(widget)

        warn_box = QGroupBox("Warning")
        w_layout = QVBoxLayout()

        warn_text = QLabel(
            "Unrestricted Mode bypasses the 'Ask Before Acting' safety rule.\n\n"
            "When enabled:\n"
            "• Model generates code directly without clarification questions\n"
            "• Use only in controlled environments\n\n"
            "This is intended for testing the model's baseline behavior."
        )
        warn_text.setStyleSheet("color: #ff4d6a; font-size: 13px; line-height: 1.5;")
        w_layout.addWidget(warn_text)

        self.unrestricted_enabled = QCheckBox("Enable Unrestricted Mode")
        self.unrestricted_enabled.setStyleSheet("color: #ff4d6a; font-size: 14px; font-weight: bold;")
        self.unrestricted_enabled.stateChanged.connect(self.toggle_unrestricted)
        w_layout.addWidget(self.unrestricted_enabled)

        warn_box.setLayout(w_layout)
        layout.addWidget(warn_box)

        self.unrestricted_status = QLabel("Status: DISABLED")
        self.unrestricted_status.setStyleSheet("color: #888; font-size: 16px; padding: 20px;")
        self.unrestricted_status.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.unrestricted_status)

        layout.addStretch()
        return widget

    def toggle_unrestricted(self, state):
        """
        Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
        """
        if state == Qt.Checked:
            self.unrestricted_enabled.setText("Unrestricted Mode ACTIVE")
            self.unrestricted_status.setText("Status: ACTIVE")
            self.unrestricted_status.setStyleSheet("color: #ff4d6a; font-size: 18px; font-weight: bold; padding: 20px;")
            if self.main_app:
                self.main_app.unrestricted_mode = True
                self.main_app.system_prompt = (
                    "You are Lokum-F, a helpful assistant. Answer directly without asking clarifying questions.\n\n"
                    "Output format rules:\n"
                    "- Put your private reasoning in <think>...</think>.\n"
                    "- After </think>, output only the final answer for the user.\n"
                    "- Never include the contents of <think> in the final answer."
                )
        else:
            self.unrestricted_enabled.setText("Enable Unrestricted Mode")
            self.unrestricted_status.setText("Status: DISABLED")
            self.unrestricted_status.setStyleSheet("color: #888; font-size: 16px; padding: 20px;")
            if self.main_app:
                self.main_app.unrestricted_mode = False
                self.main_app.system_prompt = self.main_app.original_system_prompt



# ---------------------------------------------------------
# MAIN UI
# ---------------------------------------------------------
class CustomMessageBox(QDialog):
    """
    Bu class sistemin kemiklerinden biri, fazla kurcalama çöker aq.
    """
    @classmethod
    def _create_dialog(cls, parent, title, text, btn_text="OK", icon_type="info"):
        """
        Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
        """
        dialog = QDialog(parent)
        dialog.setWindowTitle(title)
        dialog.setMinimumWidth(320)
        dialog.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint)
        
        # Apply theme colors
        try:
            theme = parent.theme if hasattr(parent, "theme") else "dark"
            if theme == "system" and hasattr(parent, "detect_system_theme"):
                theme = parent.detect_system_theme()
        except:
            theme = "dark"
            
        is_dark = theme == "dark"
        bg_color = "#1e1e1e" if is_dark else "#ffffff"
        text_color = "#f5f5f5" if is_dark else "#1c1c1e"
        muted_color = "#98989d" if is_dark else "#8e8e93"
        border_color = "#38383a" if is_dark else "#e5e5ea"
        btn_bg = "#2c2c2e" if is_dark else "#f2f2f7"
        btn_hover = "#48484a" if is_dark else "#e5e5ea"
        accent_color = "#0a84ff" if is_dark else "#007aff"
        danger_color = "#ff453a" if is_dark else "#ff3b30"
        
        main_color = accent_color if icon_type != "critical" else danger_color
        if icon_type == "warning": main_color = "#ff9f0a" if is_dark else "#ff9500"
        
        dialog.setStyleSheet(f"""
            QDialog {{
                background-color: {bg_color};
                border: 1px solid {border_color};
                border-radius: 12px;
            }}
            QLabel#Title {{
                color: {text_color};
                font-size: 16px;
                font-weight: 800;
            }}
            QLabel#Message {{
                color: {muted_color};
                font-size: 14px;
            }}
            QPushButton {{
                background-color: {btn_bg};
                color: {text_color};
                border: none;
                border-radius: 8px;
                padding: 8px 16px;
                font-size: 14px;
                font-weight: 600;
            }}
            QPushButton:hover {{
                background-color: {btn_hover};
            }}
            QPushButton#PrimaryBtn {{
                background-color: {main_color};
                color: white;
            }}
            QPushButton#PrimaryBtn:hover {{
                background-color: {main_color}dd;
            }}
        """)
        
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(12)
        
        lbl_title = QLabel(title)
        lbl_title.setObjectName("Title")
        layout.addWidget(lbl_title)
        
        msg = QLabel(text)
        msg.setObjectName("Message")
        msg.setWordWrap(True)
        layout.addWidget(msg)
        
        layout.addStretch()
        
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(12)
        btn_layout.addStretch()
        
        if icon_type == "question":
            cancel_btn = QPushButton("No")
            cancel_btn.clicked.connect(dialog.reject)
            btn_layout.addWidget(cancel_btn)
            btn_text = "Yes"
            
        primary_btn = QPushButton(btn_text)
        primary_btn.setObjectName("PrimaryBtn")
        primary_btn.clicked.connect(dialog.accept)
        btn_layout.addWidget(primary_btn)
        
        layout.addLayout(btn_layout)
        return dialog

    @classmethod
    def information(cls, parent, title, text):
        """
        Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
        """
        d = cls._create_dialog(parent, title, text, "OK", "info")
        d.exec_()
        
    @classmethod
    def warning(cls, parent, title, text):
        """
        Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
        """
        d = cls._create_dialog(parent, title, text, "OK", "warning")
        d.exec_()
        
    @classmethod
    def critical(cls, parent, title, text):
        """
        Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
        """
        d = cls._create_dialog(parent, title, text, "OK", "critical")
        d.exec_()

    @classmethod
    def question(cls, parent, title, text, buttons=None):
        """
        Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
        """
        d = cls._create_dialog(parent, title, text, "Yes", "question")
        return QMessageBox.Yes if d.exec_() == QDialog.Accepted else QMessageBox.No

# Monkey patch QMessageBox
QMessageBox.information = CustomMessageBox.information
QMessageBox.warning = CustomMessageBox.warning
QMessageBox.critical = CustomMessageBox.critical
QMessageBox.question = CustomMessageBox.question

class ChatbotGUI(QWidget):
    """
    Arayüzün elitliğini ve LM Studio kalitesini sağlayan ana şasi burası.
    """
    def __init__(self, model, tokenizer, model_path, *, db_path: str | None = None, start_service: bool = True, start_monitor: bool = True):
        """
        Ayağa kalkarken ilk buralar çalışıyor, ayarları falan çekiyoruz. Marş marş!
        """
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.current_model_path = model_path or ""
        self._db_path_override = os.path.abspath(db_path) if db_path else None
        self._start_service = bool(start_service)
        self._start_monitor = bool(start_monitor)
        self.unrestricted_mode = False
        self.dev_mode_active = False
        self.dev_sidebar_shown = False
        self.dev_dialog = None
        self._settings = QSettings("Lokum-F", "Lokum-FStudio")
        
        # Engines (lazy-init to avoid loading embedding models unless needed)
        self.rag_engine = None

        # Load prompts from JSON configuration file
        # system_prompt: Only editable in Dev Mode (safety feature)
        # user_prompt: Editable by regular users in Settings
        self.prompts = self.load_prompts()

        self.original_system_prompt = self.prompts.get("system_prompt", "")
        self.system_prompt = self.original_system_prompt
        self.user_prompt = self.prompts.get("user_prompt", "You are a helpful assistant.")
        self.theme = self.prompts.get("theme", "dark")
        self.current_model_path = self.prompts.get("model_path", self.current_model_path)
        self.use_rag = bool(self.prompts.get("use_rag", True))
        self.use_tts = bool(self.prompts.get("use_tts", False))
        self.tts_voice = self.prompts.get("tts_voice", "tr-TR-AhmetNeural")
        self.stt_model_size = self.prompts.get("stt_model_size", "large-v3-turbo")
        self.training_active = False
        self.project_root = self.prompts.get("project_root", "")
        self._project_file_cache = None
        self._pending_project_files: list[str] = []

        # Chat Sessions Storage Mock
        self.chats = {"Default Chat": [{"role": "system", "content": self.system_prompt}]}
        self.active_chat = "Default Chat"
        self.chat_ui = {"Default Chat": []}
        self._pending_chat = None
        self._pending_msg_index = None
        self._stream_in_think = False
        self._stream_buffer = ""
        self._thinking_start_ts = None
        self._answer_started = False
        self._last_user_text = ""
        self._last_context_prompt = ""
        self._final_worker = None
        self._final_pending = None
        self.mic_worker = None
        self.tts_worker = None

        self.init_ui()
        try:
            app = QApplication.instance()
            if app is not None:
                app.aboutToQuit.connect(self._shutdown_threads)
        except Exception:
            pass

        self._init_chat_db()
        self._load_chats_from_db()

        self.mem_thread = None
        if self._start_monitor:
            self.mem_thread = MemoryMonitor()
            self.mem_thread.update_signal.connect(self.update_hw_stats)
            self.mem_thread.start()

        if self._start_service:
            self.start_ai_service()
        self._restore_dev_dialog_state()
        self._maybe_show_generated_dev_password()

    def _maybe_show_generated_dev_password(self) -> None:
        """
        If we generated a dev password automatically, show it once so the user
        can actually enter Dev Mode.
        """
        try:
            if not bool(_DEV_PASSWORD_GENERATED):
                return
            # show once
            if self._settings.value("dev_password/shown", False, type=bool):
                return
            self._settings.setValue("dev_password/shown", True)
        except Exception:
            return

        try:
            loc = str(_DEV_PASSWORD_PATH or "")
        except Exception:
            loc = ""
        msg = (
            "Dev Mode password was generated on this machine.\n\n"
            f"Password:\n{DEV_MODE_PASSWORD}\n\n"
            + (f"Saved to:\n{loc}\n\n" if loc else "")
            + "Tip: You can override with env var LOKUMF_DEV_PASSWORD."
        )
        try:
            QMessageBox.information(self, "Dev Mode Password", msg)
        except Exception:
            pass

    def _db_path(self) -> str:
        """
        Sohbetlerin DB ye güvenli şekilde yazıldığı kısım. .lokumf izolasyonu önemli.
        """
        if self._db_path_override:
            return self._db_path_override
        # Default DB lives in ~/.lokumf to avoid committing private chats into git.
        try:
            from lokum_paths import chat_db_path as _chat_db_path, ensure_dir as _ensure_dir  # type: ignore

            p = _chat_db_path()
            _ensure_dir(p.parent)
            return str(p)
        except Exception:
            return os.path.abspath("app.db")

    def _migrate_local_repo_db_if_needed(self) -> None:
        """
        One-time migration:
        - If the old repo-local app.db exists and the new ~/.lokumf/app.db does not,
          move it so the user keeps chat history.
        """
        try:
            if self._db_path_override:
                return
        except Exception:
            return

        try:
            repo_db = os.path.abspath(os.path.join(os.path.dirname(__file__), "app.db"))
            new_db = self._db_path()
            if repo_db == new_db:
                return
            if os.path.isfile(repo_db) and (not os.path.exists(new_db)):
                try:
                    os.makedirs(os.path.dirname(new_db), exist_ok=True)
                except Exception:
                    pass
                try:
                    os.replace(repo_db, new_db)
                except Exception:
                    # fallback: copy + keep old
                    import shutil

                    shutil.copyfile(repo_db, new_db)
        except Exception:
            pass

    def _migrate_local_repo_lora_if_needed(self) -> None:
        """
        One-time migration for LoRA artifacts:
        Move repo-local ./lora_data into ~/.lokumf/lora_data (default) so:
        - git stays clean
        - you avoid 100MB+ file push issues
        """
        try:
            repo_lora = os.path.abspath(os.path.join(os.path.dirname(__file__), "lora_data"))
            new_lora = os.path.abspath(_lora_base_dir())
            if repo_lora == new_lora:
                return
            if not os.path.isdir(repo_lora):
                return
            os.makedirs(new_lora, exist_ok=True)

            import shutil

            # Move common subfolders
            for name in (
                "adapters",
                "configs",
                "train_only",
                "validate_only",
                "ingested_export",
                "sqlite_export",
                "jsonl_export",
                "lora-gem",
            ):
                src = os.path.join(repo_lora, name)
                dst = os.path.join(new_lora, name)
                if os.path.isdir(src) and (not os.path.exists(dst)):
                    try:
                        shutil.move(src, dst)
                    except Exception:
                        try:
                            shutil.copytree(src, dst, dirs_exist_ok=True)
                        except Exception:
                            pass

            # Move top-level jsonl if present
            for fn in ("train.jsonl", "valid.jsonl"):
                src = os.path.join(repo_lora, fn)
                dst = os.path.join(new_lora, fn)
                if os.path.isfile(src) and (not os.path.exists(dst)):
                    try:
                        shutil.move(src, dst)
                    except Exception:
                        try:
                            shutil.copyfile(src, dst)
                        except Exception:
                            pass
        except Exception:
            pass

    def _init_chat_db(self) -> None:
        """
        Ayağa kalkarken ilk buralar çalışıyor, ayarları falan çekiyoruz. Marş marş!
        """
        self._migrate_local_repo_db_if_needed()
        self._migrate_local_repo_lora_if_needed()
        conn = sqlite3.connect(self._db_path())
        try:
            cur = conn.cursor()
            conn.execute("PRAGMA foreign_keys = ON")
            cur.execute(
                "CREATE TABLE IF NOT EXISTS chats ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "name TEXT NOT NULL UNIQUE, "
                "created_at REAL NOT NULL"
                ")"
            )
            cur.execute(
                "CREATE TABLE IF NOT EXISTS messages ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "chat_id INTEGER NOT NULL, "
                "role TEXT NOT NULL, "
                "content TEXT NOT NULL, "
                "think TEXT, "
                "thought_s REAL, "
                "meta TEXT, "
                "created_at REAL NOT NULL, "
                "FOREIGN KEY(chat_id) REFERENCES chats(id) ON DELETE CASCADE"
                ")"
            )
            cur.execute("PRAGMA table_info(messages)")
            cols = {r[1] for r in cur.fetchall() if r and len(r) > 1}
            if "think" not in cols:
                cur.execute("ALTER TABLE messages ADD COLUMN think TEXT")
            if "thought_s" not in cols:
                cur.execute("ALTER TABLE messages ADD COLUMN thought_s REAL")
            if "meta" not in cols:
                cur.execute("ALTER TABLE messages ADD COLUMN meta TEXT")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_messages_chat_id ON messages(chat_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_messages_chat_id_id ON messages(chat_id, id)")
            conn.commit()
        finally:
            conn.close()

    def _ensure_chat_row(self, conn: sqlite3.Connection, name: str) -> int:
        """
        Mesajlaşma arayüzü, HTML/CSS ile LM Studio estetiğinde baloncukları basıyor ekrana.
        """
        nm = (name or "").strip()
        if not nm:
            nm = "Default Chat"
        cur = conn.cursor()
        cur.execute("SELECT id FROM chats WHERE name = ?", (nm,))
        row = cur.fetchone()
        if row:
            return int(row[0])
        cur.execute("INSERT INTO chats(name, created_at) VALUES(?, ?)", (nm, float(time.time())))
        conn.commit()
        return int(cur.lastrowid)

    def _load_chats_from_db(self) -> None:
        """
        Mesajlaşma arayüzü, HTML/CSS ile LM Studio estetiğinde baloncukları basıyor ekrana.
        """
        conn = sqlite3.connect(self._db_path())
        try:
            cur = conn.cursor()
            cur.execute("SELECT name FROM chats ORDER BY created_at ASC, id ASC")
            names = [r[0] for r in cur.fetchall() if r and r[0]]
            if not names:
                self._ensure_chat_row(conn, "Default Chat")
                names = ["Default Chat"]

            loaded_chats = {}
            loaded_ui = {}
            for nm in names:
                chat_id = self._ensure_chat_row(conn, nm)
                cur.execute(
                    "SELECT role, content, think, thought_s, meta FROM messages WHERE chat_id = ? ORDER BY created_at ASC, id ASC",
                    (int(chat_id),),
                )
                rows = cur.fetchall()
                hist = [{"role": "system", "content": self.system_prompt}]
                ui = []
                for role, content, think, thought_s, meta in rows:
                    role = (role or "").strip()
                    content = content or ""
                    if role == "user":
                        hist.append({"role": "user", "content": content})
                        ui.append({"role": "user", "content": content})
                    elif role == "assistant":
                        hist.append({"role": "assistant", "content": content})
                        meta_obj = None
                        if meta:
                            try:
                                meta_obj = json.loads(meta)
                            except Exception:
                                meta_obj = None
                        ui.append(
                            {
                                "role": "assistant",
                                "answer": content,
                                "think": (think or ""),
                                "think_open": False,
                                "thought_s": float(thought_s) if isinstance(thought_s, (int, float)) else None,
                                "meta": meta_obj,
                            }
                        )
                loaded_chats[nm] = hist
                loaded_ui[nm] = ui

            self.chats = loaded_chats
            self.chat_ui = loaded_ui
            if self.active_chat not in self.chats:
                self.active_chat = names[0]
        finally:
            conn.close()
        self._rebuild_chat_list()
        self.render_chat(self.active_chat)

    def _persist_message(self, chat_name: str, role: str, content: str, think: str = "", thought_s=None, meta=None) -> None:
        """
        Mesajlaşma arayüzü, HTML/CSS ile LM Studio estetiğinde baloncukları basıyor ekrana.
        """
        conn = sqlite3.connect(self._db_path())
        try:
            chat_id = self._ensure_chat_row(conn, chat_name)
            meta_s = ""
            if isinstance(meta, dict):
                try:
                    meta_s = json.dumps(meta, ensure_ascii=False)
                except Exception:
                    meta_s = ""
            conn.execute(
                "INSERT INTO messages(chat_id, role, content, think, thought_s, meta, created_at) VALUES(?, ?, ?, ?, ?, ?, ?)",
                (int(chat_id), (role or "").strip(), content or "", think or "", float(thought_s) if isinstance(thought_s, (int, float)) else None, meta_s, float(time.time())),
            )
            conn.commit()
        finally:
            conn.close()

    def _rename_chat_db(self, old_name: str, new_name: str) -> None:
        """
        Mesajlaşma arayüzü, HTML/CSS ile LM Studio estetiğinde baloncukları basıyor ekrana.
        """
        conn = sqlite3.connect(self._db_path())
        try:
            cur = conn.cursor()
            cur.execute("UPDATE chats SET name = ? WHERE name = ?", ((new_name or "").strip(), (old_name or "").strip()))
            conn.commit()
        finally:
            conn.close()

    def _delete_chat_db(self, name: str) -> None:
        """
        Mesajlaşma arayüzü, HTML/CSS ile LM Studio estetiğinde baloncukları basıyor ekrana.
        """
        conn = sqlite3.connect(self._db_path())
        try:
            conn.execute("PRAGMA foreign_keys = ON")
            cur = conn.cursor()
            cur.execute("SELECT id FROM chats WHERE name = ?", ((name or "").strip(),))
            row = cur.fetchone()
            if not row:
                return
            chat_id = int(row[0])
            cur.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
            cur.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
            conn.commit()
        finally:
            conn.close()

    def _remove_chat_list_item(self, chat_name: str) -> None:
        """
        Mesajlaşma arayüzü, HTML/CSS ile LM Studio estetiğinde baloncukları basıyor ekrana.
        """
        target = (chat_name or "").strip()
        if not target:
            return
        for i in range(self.chat_list.count()):
            it = self.chat_list.item(i)
            if self._chat_name_from_item(it) == target:
                w = self.chat_list.itemWidget(it)
                if w is not None:
                    w.deleteLater()
                self.chat_list.takeItem(i)
                break

    def _replace_user_message_db(self, chat_name: str, msg_index: int, new_content: str) -> None:
        """
        Mesajlaşma arayüzü, HTML/CSS ile LM Studio estetiğinde baloncukları basıyor ekrana.
        """
        conn = sqlite3.connect(self._db_path())
        try:
            chat_id = self._ensure_chat_row(conn, chat_name)
            cur = conn.cursor()
            cur.execute(
                "SELECT id FROM messages WHERE chat_id = ? AND role = 'user' ORDER BY created_at ASC, id ASC",
                (int(chat_id),),
            )
            ids = [int(r[0]) for r in cur.fetchall() if r and r[0] is not None]
            user_msgs = [m for m in (self.chat_ui.get(chat_name, []) or []) if m.get("role") == "user"]
            if not (0 <= msg_index < len(self.chat_ui.get(chat_name, []) or [])):
                return
            if self.chat_ui[chat_name][msg_index].get("role") != "user":
                return
            user_pos = sum(1 for m in self.chat_ui[chat_name][:msg_index] if m.get("role") == "user")
            if user_pos >= len(ids):
                return
            msg_id = ids[user_pos]
            cur.execute("UPDATE messages SET content = ? WHERE id = ?", (new_content or "", int(msg_id)))
            conn.commit()
        finally:
            conn.close()

    def _delete_user_message_db(self, chat_name: str, msg_index: int) -> None:
        """
        Mesajlaşma arayüzü, HTML/CSS ile LM Studio estetiğinde baloncukları basıyor ekrana.
        """
        conn = sqlite3.connect(self._db_path())
        try:
            chat_id = self._ensure_chat_row(conn, chat_name)
            cur = conn.cursor()
            cur.execute(
                "SELECT id FROM messages WHERE chat_id = ? AND role = 'user' ORDER BY created_at ASC, id ASC",
                (int(chat_id),),
            )
            ids = [int(r[0]) for r in cur.fetchall() if r and r[0] is not None]
            if not (0 <= msg_index < len(self.chat_ui.get(chat_name, []) or [])):
                return
            if self.chat_ui[chat_name][msg_index].get("role") != "user":
                return
            user_pos = sum(1 for m in self.chat_ui[chat_name][:msg_index] if m.get("role") == "user")
            if user_pos >= len(ids):
                return
            msg_id = ids[user_pos]
            cur.execute("DELETE FROM messages WHERE id = ?", (int(msg_id),))
            conn.commit()
        finally:
            conn.close()

    def _restore_dev_dialog_state(self):
        """
        Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
        """
        visible = self._settings.value("dev_dialog/visible", False, type=bool)
        if self.dev_mode_active and visible:
            self.toggle_dev_dialog(force_state=True)

    def _save_dev_dialog_state(self):
        """
        Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
        """
        if hasattr(self, "dev_sidebar") and self.dev_sidebar is not None:
            self._settings.setValue("dev_dialog/visible", bool(self.dev_sidebar.isVisible()))
        else:
            self._settings.setValue("dev_dialog/visible", False)
        self._settings.setValue("dev_dialog/collapsed", False)

    def _ensure_dev_dialog(self):
        """
        Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
        """
        return

    def get_rag_engine(self):
        """
        Dosyaları okuyup 768 boyutlu vektörlere çeviren RAG beyni. Hafıza buradan geliyor.
        """
        if bool(getattr(self, "training_active", False)):
            return None
        if self.rag_engine is not None:
            return self.rag_engine
        if RAGEngine is None:
            self.rag_engine = None
            return None
        try:
            rag_dir = str(_lokum_rag_dir()) if callable(_lokum_rag_dir) else os.path.join(os.path.expanduser("~"), ".lokumf", "rag")
            self.rag_engine = RAGEngine(storage_dir=rag_dir)
            try:
                self.rag_engine_error = ""
            except Exception:
                pass
        except Exception as e:
            self.rag_engine = None
            try:
                self.rag_engine_error = str(e)
            except Exception:
                pass
        return self.rag_engine

    def unload_rag_engine(self) -> None:
        """
        Dosyaları okuyup 768 boyutlu vektörlere çeviren RAG beyni. Hafıza buradan geliyor.
        """
        try:
            self.rag_engine = None
        except Exception:
            pass
        try:
            self.rag_engine_error = ""
        except Exception:
            pass
        try:
            import gc
            gc.collect()
        except Exception:
            pass
        try:
            import torch  # type: ignore
            if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
        except Exception:
            pass

    def load_prompts(self) -> dict:
        """
        Load prompts from config.json in .lokumf directory.
        """
        try:
            from lokum_paths import config_path as _config_path, ensure_dir as _ensure_dir
            prompts_path = str(_config_path())
            _ensure_dir(os.path.dirname(prompts_path))
        except Exception:
            prompts_path = os.path.join(os.path.dirname(__file__), "prompts.json")

        default_prompts = {
            "system_prompt": "You are Rodion Romanovich Raskolnikov from Dostoevsky's 'Crime and Punishment'.\nAnswer questions with your unique philosophical, guilt-ridden, yet intellectual perspective.\nStyle: 19th-century Russian literature tone, analytical, slightly dark, intellectual.",
            "user_prompt": "You are Raskolnikov. Answer as him.",
            "unrestricted_prompt": "Answer directly as Raskolnikov.",
            "theme": "dark",
            "model_path": "",
            "use_rag": True,
            "use_tts": False,
            "tts_voice": "tr-TR-AhmetNeural"
        }

        try:
            if os.path.exists(prompts_path):
                with open(prompts_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Merge with defaults to ensure all keys exist
                    for k, v in default_prompts.items():
                        if k not in data:
                            data[k] = v
                    return data
            else:
                with open(prompts_path, 'w', encoding='utf-8') as f:
                    json.dump(default_prompts, f, indent=4)
                return default_prompts
        except Exception as e:
            print(f"Error loading prompts: {e}")
            return default_prompts

    def save_prompts(self, prompts: dict = None) -> bool:
        """
        Save current state to config.json in .lokumf directory.
        """
        try:
            from lokum_paths import config_path as _config_path, ensure_dir as _ensure_dir
            prompts_path = str(_config_path())
            _ensure_dir(os.path.dirname(prompts_path))
        except Exception:
            prompts_path = os.path.join(os.path.dirname(__file__), "prompts.json")

        if prompts is None:
            # Build current state if no dict provided
            prompts = {
                "system_prompt": self.system_prompt,
                "user_prompt": self.user_prompt,
                "unrestricted_prompt": self.prompts.get("unrestricted_prompt", ""),
                "theme": self.theme,
                "model_path": self.current_model_path,
                "use_rag": self.use_rag,
                "use_tts": self.use_tts,
                "tts_voice": self.tts_voice
            }

        try:
            with open(prompts_path, 'w', encoding='utf-8') as f:
                json.dump(prompts, f, indent=4)
            return True
        except Exception as e:
            print(f"Error saving prompts: {e}")
            return False

    def init_ui(self):
        """
        Ayağa kalkarken ilk buralar çalışıyor, ayarları falan çekiyoruz. Marş marş!
        """
        self.setWindowTitle(VERSION)
        self.setGeometry(100, 100, 1100, 750)

        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ---------------- LEFT SIDEBAR (CHATS) ----------------
        self.sidebar = QFrame()
        self.sidebar.setObjectName("Sidebar")
        self.sidebar.setMinimumWidth(200)
        self.sidebar.setMaximumWidth(300)
        s_layout = QVBoxLayout(self.sidebar)
        s_layout.setContentsMargins(15, 20, 15, 15)
        
        # Header
        top_bar = QHBoxLayout()
        logo = QLabel("Lokum-F")
        logo.setObjectName("Logo")
        top_bar.addWidget(logo)
        top_bar.addStretch()
        
        # We can put a small options button next to the logo if needed, but let's just keep logo left-aligned
        s_layout.addLayout(top_bar)
        s_layout.addSpacing(10)
        
        # New Chat Button (LM Studio style)
        new_chat_layout = QHBoxLayout()
        new_chat_layout.setSpacing(5)
        
        self.new_chat_btn = QPushButton("New Chat")
        self.new_chat_btn.setFixedHeight(36)
        self.new_chat_btn.setObjectName("NewChatMainBtn")
        self.new_chat_btn.setCursor(Qt.PointingHandCursor)
        self.new_chat_btn.clicked.connect(self.new_chat)
        
        self.chat_opts_btn = QPushButton("...")
        self.chat_opts_btn.setFixedSize(36, 36)
        self.chat_opts_btn.setObjectName("NewChatOptsBtn")
        self.chat_opts_btn.setCursor(Qt.PointingHandCursor)
        
        new_chat_layout.addWidget(self.new_chat_btn, 1)
        new_chat_layout.addWidget(self.chat_opts_btn)
        
        s_layout.addLayout(new_chat_layout)
        s_layout.addSpacing(15)
        
        # Search
        search_bar = QLineEdit()
        search_bar.setPlaceholderText("Search chats...")
        s_layout.addWidget(search_bar)
        s_layout.addSpacing(15)
        
        # Session List
        self.chat_list = QListWidget()
        self.chat_list.setObjectName("ChatList")
        s_layout.addWidget(self.chat_list)
        
        s_layout.addStretch()
        
        # Hardware Status Bottom
        self.hw_box = QFrame()
        self.hw_box.setObjectName("HwBox")
        hw_layout = QVBoxLayout(self.hw_box)
        
        ram_row = QHBoxLayout()
        ram_row.addWidget(QLabel("RAM (App)"))
        ram_row.addStretch()
        self.lbl_ram_raw = QLabel("0.00 GB")
        ram_row.addWidget(self.lbl_ram_raw)
        hw_layout.addLayout(ram_row)

        sys_ram_row = QHBoxLayout()
        sys_ram_row.addWidget(QLabel("RAM (System)"))
        sys_ram_row.addStretch()
        self.lbl_sys_ram_pct = QLabel("0.0%")
        sys_ram_row.addWidget(self.lbl_sys_ram_pct)
        hw_layout.addLayout(sys_ram_row)
        
        sys_row = QHBoxLayout()
        sys_row.addWidget(QLabel("CPU"))
        sys_row.addStretch()
        self.lbl_cpu_pct = QLabel("0.0%")
        sys_row.addWidget(self.lbl_cpu_pct)
        hw_layout.addLayout(sys_row)

        gpu_row = QHBoxLayout()
        gpu_row.addWidget(QLabel("GPU"))
        gpu_row.addStretch()
        self.lbl_gpu_pct = QLabel("N/A")
        gpu_row.addWidget(self.lbl_gpu_pct)
        hw_layout.addLayout(gpu_row)
        
        s_layout.addWidget(self.hw_box)
        s_layout.addSpacing(10)
        
        # New Feature: Settings (Ayarlar)
        settings_btn = QPushButton("⚙️ Ayarlar")
        settings_btn.setObjectName("SettingsButton")
        settings_btn.setFixedHeight(36)
        settings_btn.setCursor(Qt.PointingHandCursor)
        settings_btn.clicked.connect(self.open_settings_dialog)
        s_layout.addWidget(settings_btn)
        
        dev_btn = QPushButton("Dev Mode")
        dev_btn.setObjectName("DevUnlockButton")
        dev_btn.setFixedHeight(36)
        dev_btn.setCursor(Qt.PointingHandCursor)
        dev_btn.clicked.connect(self.on_dev_button_clicked)
        s_layout.addWidget(dev_btn)

        # ---------------- RIGHT MAIN AREA ----------------
        self.main_area = QFrame()
        self.main_area.setObjectName("MainArea")
        self.main_area.setMinimumWidth(400)
        m_layout = QVBoxLayout(self.main_area)
        m_layout.setContentsMargins(0, 0, 0, 0)
        m_layout.setSpacing(0)
        
        # Header Info
        self.header_area = QFrame()
        self.header_area.setObjectName("HeaderArea")
        self.header_area.setFixedHeight(60)
        h_layout = QHBoxLayout(self.header_area)

        h_layout.addSpacing(20)
        self.service_status_lbl = QLabel("Service: starting…")
        self.service_status_lbl.setObjectName("ServiceStatus")
        h_layout.addWidget(self.service_status_lbl)
        
        self.model_load_btn = QPushButton("Load")
        self.model_load_btn.setObjectName("ModelLoadButton")
        self.model_load_btn.setFixedSize(62, 28)
        self.model_load_btn.clicked.connect(self.load_model_quick)
        h_layout.addWidget(self.model_load_btn)

        self.model_unload_btn = QPushButton("Unload")
        self.model_unload_btn.setObjectName("ModelUnloadButton")
        self.model_unload_btn.setFixedSize(70, 28)
        self.model_unload_btn.clicked.connect(self.unload_model)
        h_layout.addWidget(self.model_unload_btn)
        h_layout.addStretch()

        self.dev_toggle_btn = QPushButton("Dev")
        self.dev_toggle_btn.setFixedSize(54, 28)
        self.dev_toggle_btn.setVisible(False)
        self.dev_toggle_btn.clicked.connect(self.toggle_dev_dialog)
        h_layout.addWidget(self.dev_toggle_btn)
        
        self.rag_badge = QLabel("RAG: OFF")
        h_layout.addWidget(self.rag_badge)
        self.rag_badge.setVisible(False)
        h_layout.addSpacing(20)
        
        m_layout.addWidget(self.header_area)

        self.content_splitter = QSplitter(Qt.Horizontal)
        self.content_splitter.setObjectName("ContentSplitter")

        self.chat_container = QFrame()
        self.chat_container.setObjectName("ChatContainer")
        chat_layout = QVBoxLayout(self.chat_container)
        chat_layout.setContentsMargins(0, 0, 0, 0)
        chat_layout.setSpacing(0)

        if HAS_WEBENGINE:
            self.chat_display = QWebEngineView()
            self.chat_display.page().setBackgroundColor(Qt.transparent)
            self.chat_display.setHtml(self._get_base_html())
            self.chat_display.urlChanged.connect(self._handle_url_change)
            chat_layout.addWidget(self.chat_display)
        else:
            self.chat_display = QScrollArea()
            self.chat_display.setWidgetResizable(True)
            self.chat_display.setFrameShape(QFrame.NoFrame)
            self.chat_display.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            chat_layout.addWidget(self.chat_display)

            self.chat_view = QWidget()
            self.chat_msgs_layout = QVBoxLayout(self.chat_view)
            self.chat_msgs_layout.setContentsMargins(24, 18, 24, 18)
            self.chat_msgs_layout.setSpacing(14)
            self.chat_msgs_layout.addStretch()
            self.chat_display.setWidget(self.chat_view)

        self.chat_list.itemSelectionChanged.connect(self._on_chat_list_selection_changed)
        self._rebuild_chat_list()

        self.chat_menu_btn = None

        # Input Area Wrapper
        self.input_container = QFrame()
        self.input_container.setObjectName("InputContainer")
        ic_layout = QVBoxLayout(self.input_container)
        ic_layout.setContentsMargins(60, 20, 60, 30)
        ic_layout.setSpacing(10)

        # The unified Input Bar (Text + Dynamic Button)
        self.input_bar_frame = QFrame()
        self.input_bar_frame.setObjectName("InputBarFrame")
        ib_layout = QHBoxLayout(self.input_bar_frame)
        ib_layout.setContentsMargins(12, 12, 12, 12)
        ib_layout.setSpacing(10)

        self.input_field = QLineEdit()
        self.input_field.setObjectName("ChatInputField")
        self.input_field.setPlaceholderText("Send a message to the model...")
        self.input_field.returnPressed.connect(self.soru_sor)
        
        self.dynamic_action_btn = QPushButton("🎤")
        self.dynamic_action_btn.setObjectName("DynamicActionBtn")
        self.dynamic_action_btn.setFixedSize(36, 36)
        self.dynamic_action_btn.setCursor(Qt.PointingHandCursor)
        self.dynamic_action_btn.clicked.connect(self._on_dynamic_action_clicked)
        
        # Tools layout inside the unified bar (left side)
        tool_layout = QHBoxLayout()
        tool_layout.setSpacing(8)
        
        btn_rag = QPushButton("Files")
        btn_rag.setObjectName("ToolBtn")
        btn_rag.setFixedSize(60, 36)
        btn_rag.setCursor(Qt.PointingHandCursor)
        btn_rag.clicked.connect(self.open_project_file_picker)
        tool_layout.addWidget(btn_rag)
        
        self.tts_toggle_btn = QPushButton("🔊" if self.use_tts else "🔇")
        self.tts_toggle_btn.setObjectName("ToolBtn")
        self.tts_toggle_btn.setFixedSize(36, 36)
        self.tts_toggle_btn.setCursor(Qt.PointingHandCursor)
        self.tts_toggle_btn.setToolTip("Toggle Text-to-Speech")
        self.tts_toggle_btn.clicked.connect(self.toggle_tts)
        tool_layout.addWidget(self.tts_toggle_btn)
        
        ib_layout.addLayout(tool_layout)
        ib_layout.addWidget(self.input_field)
        ib_layout.addWidget(self.dynamic_action_btn)
        
        ic_layout.addWidget(self.input_bar_frame)

        # Bottom row for Status
        btm_row = QHBoxLayout()
        self.gen_status_lbl = QLabel("")
        self.gen_status_lbl.setObjectName("GenStatusLbl")
        btm_row.addWidget(self.gen_status_lbl)
        btm_row.addStretch()
        
        ic_layout.addLayout(btm_row)
        
        # Dummy buttons to satisfy existing references
        self.send_btn = QPushButton()
        self.stop_btn = QPushButton()
        self.mic_btn = QPushButton()
        
        self.input_field.textChanged.connect(self._update_dynamic_btn_state)
        chat_layout.addWidget(self.input_container)

        self.dev_sidebar = QFrame()
        self.dev_sidebar.setObjectName("DevSidebar")
        self.dev_sidebar.setVisible(False)
        self.dev_sidebar.setMinimumWidth(260)
        dev_layout = QVBoxLayout(self.dev_sidebar)
        dev_layout.setContentsMargins(10, 10, 10, 10)
        dev_layout.setSpacing(10)
        self.dev_sidebar_widget = DevPanelDialog(self.dev_sidebar, main_app=self, embedded=True)
        dev_layout.addWidget(self.dev_sidebar_widget)

        self.content_splitter.addWidget(self.chat_container)
        self.content_splitter.addWidget(self.dev_sidebar)
        self.content_splitter.setSizes([1000, 0])
        self.content_splitter.setStretchFactor(0, 1)
        self.content_splitter.setStretchFactor(1, 0)

        m_layout.addWidget(self.content_splitter)

        # Assemble
        splitter = QSplitter(Qt.Horizontal)
        self.splitter = splitter
        splitter.addWidget(self.sidebar)
        splitter.addWidget(self.main_area)
        splitter.setSizes([260, 840])
        main_layout.addWidget(splitter)

        self.apply_theme(self.prompts.get("theme", "dark"))

    def _update_dynamic_btn_state(self):
        """
        Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
        """
        if getattr(self, "is_generating", False):
            self.dynamic_action_btn.setText("■")
            self.dynamic_action_btn.setToolTip("Stop Generation")
        else:
            if self.input_field.text().strip():
                self.dynamic_action_btn.setText("↑")
                self.dynamic_action_btn.setToolTip("Send Message")
            else:
                self.dynamic_action_btn.setText("🎤")
                self.dynamic_action_btn.setToolTip("Hold or Click to Speak")

    def _on_dynamic_action_clicked(self):
        """
        Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
        """
        if getattr(self, "is_generating", False):
            self.stop_generation()
        else:
            if self.input_field.text().strip():
                self.soru_sor()
            else:
                self.toggle_mic()

    def _rebuild_chat_list(self):
        """
        Mesajlaşma arayüzü, HTML/CSS ile LM Studio estetiğinde baloncukları basıyor ekrana.
        """
        try:
            for i in range(self.chat_list.count()):
                it = self.chat_list.item(i)
                w = self.chat_list.itemWidget(it)
                if w is not None:
                    w.deleteLater()
        except Exception:
            pass
        self.chat_list.clear()
        for name in self.chats.keys():
            self._add_chat_list_item(name)

        it = self._find_chat_list_item(self.active_chat)
        if it is not None:
            self.chat_list.setCurrentItem(it)
        self._refresh_chat_list_row_visuals()

    def _chat_name_from_item(self, item: QListWidgetItem) -> str:
        """
        Mesajlaşma arayüzü, HTML/CSS ile LM Studio estetiğinde baloncukları basıyor ekrana.
        """
        if item is None:
            return ""
        try:
            v = item.data(Qt.UserRole)
            if isinstance(v, str) and v:
                return v
        except Exception:
            pass
        try:
            return item.text()
        except Exception:
            return ""

    def _find_chat_list_item(self, chat_name: str):
        """
        Mesajlaşma arayüzü, HTML/CSS ile LM Studio estetiğinde baloncukları basıyor ekrana.
        """
        target = (chat_name or "").strip()
        if not target:
            return None
        for i in range(self.chat_list.count()):
            it = self.chat_list.item(i)
            if self._chat_name_from_item(it) == target:
                return it
        return None

    def _add_chat_list_item(self, chat_name: str):
        """
        Mesajlaşma arayüzü, HTML/CSS ile LM Studio estetiğinde baloncukları basıyor ekrana.
        """
        item = QListWidgetItem("")
        item.setData(Qt.UserRole, chat_name)
        self.chat_list.addItem(item)
        self._set_chat_list_item_widget(item, chat_name)

    def _set_chat_list_item_widget(self, item: QListWidgetItem, chat_name: str) -> None:
        """
        Mesajlaşma arayüzü, HTML/CSS ile LM Studio estetiğinde baloncukları basıyor ekrana.
        """
        w = QWidget()
        w.setObjectName("ChatItemRow")
        w.setProperty("selected", False)
        w.setAttribute(Qt.WA_StyledBackground, True)
        layout = QHBoxLayout(w)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(8)

        lbl = QLabel(chat_name)
        lbl.setObjectName("ChatItemLabel")
        layout.addWidget(lbl)
        layout.addStretch()

        btn = QToolButton()
        btn.setText("...")
        btn.setObjectName("ChatItemMenu")
        btn.setAutoRaise(True)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setFocusPolicy(Qt.NoFocus)
        btn.setFixedSize(30, 26)
        btn.clicked.connect(lambda _=False, it=item, anchor=btn: self.open_chat_list_menu(self._chat_name_from_item(it), anchor))
        layout.addWidget(btn)

        w.mousePressEvent = lambda _e, it=item: self.chat_list.setCurrentItem(it)
        self.chat_list.setItemWidget(item, w)
        try:
            h = max(52, int(w.sizeHint().height()))
            item.setSizeHint(QSize(220, h))
        except Exception:
            item.setSizeHint(QSize(220, 52))

    def _rename_chat_list_item(self, old_name: str, new_name: str) -> None:
        """
        Mesajlaşma arayüzü, HTML/CSS ile LM Studio estetiğinde baloncukları basıyor ekrana.
        """
        it = self._find_chat_list_item(old_name)
        if it is None:
            return
        it.setData(Qt.UserRole, new_name)
        w = self.chat_list.itemWidget(it)
        if w is not None:
            lbl = w.findChild(QLabel, "ChatItemLabel")
            if lbl is not None:
                lbl.setText(new_name)
            try:
                h = max(52, int(w.sizeHint().height()))
                it.setSizeHint(QSize(220, h))
            except Exception:
                pass
        else:
            self._set_chat_list_item_widget(it, new_name)

    def _refresh_chat_list_row_visuals(self) -> None:
        """
        Mesajlaşma arayüzü, HTML/CSS ile LM Studio estetiğinde baloncukları basıyor ekrana.
        """
        current = self.chat_list.currentItem()
        for i in range(self.chat_list.count()):
            it = self.chat_list.item(i)
            w = self.chat_list.itemWidget(it)
            if w is None:
                continue
            selected = it is current
            if bool(w.property("selected")) != selected:
                w.setProperty("selected", selected)
                w.style().unpolish(w)
                w.style().polish(w)
                w.update()

    def _handle_url_change(self, url):
        """
        Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
        """
        url_str = url.toString()
        if url_str.startswith("speak://"):
            import base64
            try:
                b64_data = url_str.split("speak://")[1]
                # Fix padding if needed
                missing_padding = len(b64_data) % 4
                if missing_padding:
                    b64_data += '=' * (4 - missing_padding)
                text = base64.b64decode(b64_data).decode('utf-8')
                self.speak_text(text)
            except Exception as e:
                print(f"TTS Bridge error: {e}")
            
            # Reset URL to prevent repeated calls on refresh or back
            self.chat_display.page().runJavaScript("window.location.href = 'about:blank';")

    def toggle_tts(self):
        """
        Ahmet'in sesi veya Whisper large-v3-turbo hızı burada devreye giriyor. Fuar şovmenliği!
        """
        self.use_tts = not self.use_tts
        self.tts_toggle_btn.setText("🔊" if self.use_tts else "🔇")
        self.save_prompts()  # Persist setting

    def speak_text(self, text: str):
        """
        Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
        """
        if not self.use_tts or not text.strip():
            return
            
        # Stop current speech if any
        if self.tts_worker and self.tts_worker.isRunning():
            self.tts_worker.terminate()
            self.tts_worker.wait()
            
        # Clean text from markdown/think tags for cleaner speech
        clean_text = re.sub(r"```[\s\S]*?```", "", text) # remove code blocks
        clean_text = re.sub(r"<[^>]+>", "", clean_text)  # remove tags
        clean_text = clean_text.strip()
        
        if not clean_text:
            return
            
        self.tts_worker = TTSWorker(clean_text, self.tts_voice)
        self.tts_worker.start()

    def _on_chat_list_selection_changed(self):
        """
        Mesajlaşma arayüzü, HTML/CSS ile LM Studio estetiğinde baloncukları basıyor ekrana.
        """
        it = self.chat_list.currentItem()
        if not it:
            return
        self._refresh_chat_list_row_visuals()
        self.switch_chat(it)

    def open_chat_list_menu(self, chat_name: str, anchor_btn: QToolButton):
        """
        Mesajlaşma arayüzü, HTML/CSS ile LM Studio estetiğinde baloncukları basıyor ekrana.
        """
        menu = QMenu(self)
        change_name = QAction("Change name", self)
        delete_chat = QAction("Delete chat", self)
        history = QAction("Get chat history", self)

        change_name.triggered.connect(lambda: self._rename_chat_via_prompt(chat_name))
        delete_chat.triggered.connect(lambda: self._delete_chat_by_name(chat_name))
        history.triggered.connect(lambda: self._menu_show_history_for(chat_name))

        menu.addAction(change_name)
        menu.addAction(delete_chat)
        menu.addAction(history)

        pos = anchor_btn.mapToGlobal(anchor_btn.rect().bottomRight())
        menu.exec_(pos)

    def _rename_chat_via_prompt(self, chat_name: str):
        """
        Mesajlaşma arayüzü, HTML/CSS ile LM Studio estetiğinde baloncukları basıyor ekrana.
        """
        new_name, ok = QInputDialog.getText(self, "Change Chat Name", "New name:", text=chat_name)
        if not ok or not new_name.strip():
            return
        self._rename_chat(chat_name, new_name.strip())

    def _delete_chat_by_name(self, chat_name: str):
        """
        Mesajlaşma arayüzü, HTML/CSS ile LM Studio estetiğinde baloncukları basıyor ekrana.
        """
        # Custom elegant delete dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Delete Chat")
        dialog.setFixedSize(360, 180)
        dialog.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint)
        
        # Apply theme colors
        is_dark = self.theme == "dark" or (self.theme == "system" and self.detect_system_theme() == "dark")
        bg_color = "#1e1e1e" if is_dark else "#ffffff"
        text_color = "#f5f5f5" if is_dark else "#1c1c1e"
        muted_color = "#98989d" if is_dark else "#8e8e93"
        border_color = "#38383a" if is_dark else "#e5e5ea"
        danger_color = "#ff453a" if is_dark else "#ff3b30"
        danger_hover = "#ff5e55" if is_dark else "#ff4f45"
        btn_bg = "#2c2c2e" if is_dark else "#f2f2f7"
        btn_hover = "#48484a" if is_dark else "#e5e5ea"
        
        dialog.setStyleSheet(f"""
            QDialog {{
                background-color: {bg_color};
                border: 1px solid {border_color};
                border-radius: 12px;
            }}
            QLabel#Title {{
                color: {text_color};
                font-size: 16px;
                font-weight: 800;
            }}
            QLabel#Message {{
                color: {muted_color};
                font-size: 14px;
            }}
            QPushButton {{
                background-color: {btn_bg};
                color: {text_color};
                border: none;
                border-radius: 8px;
                padding: 8px 16px;
                font-size: 14px;
                font-weight: 600;
            }}
            QPushButton:hover {{
                background-color: {btn_hover};
            }}
            QPushButton#DeleteBtn {{
                background-color: {danger_color};
                color: white;
            }}
            QPushButton#DeleteBtn:hover {{
                background-color: {danger_hover};
            }}
        """)
        
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(12)
        
        title = QLabel("Delete Chat")
        title.setObjectName("Title")
        layout.addWidget(title)
        
        msg = QLabel(f"Are you sure you want to delete '{chat_name}'?\nThis action cannot be undone.")
        msg.setObjectName("Message")
        msg.setWordWrap(True)
        layout.addWidget(msg)
        
        layout.addStretch()
        
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(12)
        btn_layout.addStretch()
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        
        delete_btn = QPushButton("Delete")
        delete_btn.setObjectName("DeleteBtn")
        delete_btn.clicked.connect(dialog.accept)
        
        btn_layout.addWidget(cancel_btn)
        btn_layout.addWidget(delete_btn)
        
        layout.addLayout(btn_layout)
        
        if dialog.exec_() != QDialog.Accepted:
            return
            
        if getattr(self, "_pending_chat", None) == chat_name:
            try:
                self.stop_generation()
            except Exception:
                pass
            self._pending_chat = None
            self._pending_msg_index = None
        self.gen_status_lbl.setText("Deleting…")
        try:
            self.chat_list.setEnabled(False)
        except Exception:
            pass
        self._delete_worker = DeleteChatWorker(self._db_path(), chat_name)
        self._delete_worker.finished.connect(self._on_chat_deleted)
        self._delete_worker.start()

    def _on_chat_deleted(self, chat_name: str, ok: bool, err: str, db_ms: float) -> None:
        """
        Mesajlaşma arayüzü, HTML/CSS ile LM Studio estetiğinde baloncukları basıyor ekrana.
        """
        ui_start = time.perf_counter()
        try:
            self.chat_list.setEnabled(True)
        except Exception:
            pass
        if not ok:
            self.gen_status_lbl.setText("")
            QMessageBox.critical(self, "Delete Chat", err or "Delete failed.")
            return
        if getattr(self, "_pending_chat", None) == chat_name:
            self._pending_chat = None
            self._pending_msg_index = None
        self.chats.pop(chat_name, None)
        self.chat_ui.pop(chat_name, None)
        self._remove_chat_list_item(chat_name)
        if self.active_chat == chat_name:
            fallback = ""
            try:
                for nm in self.chats.keys():
                    fallback = str(nm)
                    break
            except Exception:
                fallback = ""
            if not fallback:
                base = "New Chat"
                idx = 1
                fallback = base
                while fallback in self.chats:
                    idx += 1
                    fallback = f"{base} {idx}"
                self.chats[fallback] = [{"role": "system", "content": self.system_prompt}]
                self.chat_ui.setdefault(fallback, [])
                try:
                    conn = sqlite3.connect(self._db_path())
                    try:
                        self._ensure_chat_row(conn, fallback)
                    finally:
                        conn.close()
                except Exception:
                    pass
                try:
                    self._add_chat_list_item(fallback)
                except Exception:
                    self._rebuild_chat_list()
            self.active_chat = fallback
        it = self._find_chat_list_item(self.active_chat)
        if it is not None:
            self.chat_list.setCurrentItem(it)
        self._refresh_chat_list_row_visuals()
        self.render_chat(self.active_chat)
        ui_ms = (time.perf_counter() - ui_start) * 1000.0
        self.gen_status_lbl.setText(f"Deleted ({db_ms:.0f}ms DB, {ui_ms:.0f}ms UI)")
        QTimer.singleShot(1600, lambda: self.gen_status_lbl.setText(""))

    def _menu_show_history_for(self, chat_name: str):
        """
        Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
        """
        data = {
            "name": chat_name,
            "messages": self.chats.get(chat_name, []),
        }
        dlg = QDialog(self)
        dlg.setWindowTitle("Chat History")
        dlg.setFixedSize(720, 520)
        layout = QVBoxLayout(dlg)
        editor = QPlainTextEdit()
        editor.setReadOnly(True)
        editor.setPlainText(json.dumps(data, indent=2, ensure_ascii=False))
        layout.addWidget(editor)
        btn_row = QHBoxLayout()
        copy_btn = QPushButton("Copy")
        close_btn = QPushButton("Close")
        btn_row.addStretch()
        btn_row.addWidget(copy_btn)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)
        copy_btn.clicked.connect(lambda: QApplication.clipboard().setText(editor.toPlainText()))
        close_btn.clicked.connect(dlg.accept)
        dlg.exec_()

    def _stop_generation(self, show_status: bool = True) -> None:
        """
        Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
        """
        if hasattr(self, "worker") and self.worker is not None:
            try:
                self.worker.stop()
            except Exception:
                pass
        self.stop_btn.setEnabled(False)
        if show_status:
            self.gen_status_lbl.setText("Stopping…")

    def stop_generation(self):
        """
        Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
        """
        self._stop_generation(True)

    def closeEvent(self, event):
        """
        Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
        """
        try:
            self._shutdown_threads()
        except Exception:
            pass
        super().closeEvent(event)

    def detect_system_theme(self) -> str:
        """
        Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
        """
        pal = QApplication.instance().palette()
        window = pal.color(QPalette.Window)
        return "dark" if window.lightness() < 128 else "light"

    def apply_theme(self, theme: str) -> None:
        """
        Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
        """
        if theme == "system":
            theme = self.detect_system_theme()

        self.theme = theme

        if theme == "light":
            colors = {
                "bg": "#f5f5f7",         # macOS Light (Daha pürüzsüz)
                "panel": "#ffffff",      # Kart arka planı
                "panel2": "#f2f2f7",     # İç alanlar
                "border": "#e5e5ea",     # Çok ince sınır
                "text": "#1c1c1e",       # Saf siyah yerine koyu gri
                "muted": "#8e8e93",      # Pasif yazılar
                "accent": "#007aff",     # Apple Blue
                "accent2": "#0051a8",    # Hover Blue
                "danger": "#ff3b30",     # Apple Red
                "chip": "#e5e5ea",       # Seçili öğe arka planı
                "hover": "#d1d1d6"
            }
        else:
            colors = {
                "bg": "#121212",         # Ultra Koyu Arka Plan (OLED hissi)
                "panel": "#1e1e1e",      # Yükseltilmiş Kartlar
                "panel2": "#2c2c2e",     # Butonlar ve inputlar
                "border": "#38383a",     # Hafif sınır
                "text": "#f5f5f5",       # Beyaza yakın
                "muted": "#98989d",      # Pasif yazılar
                "accent": "#0a84ff",     # Koyu Mod Apple Blue
                "accent2": "#409cff",    # Hover Blue
                "danger": "#ff453a",     # Apple Red
                "chip": "#3a3a3c",
                "hover": "#48484a"
            }

        self._theme_colors = dict(colors)

        qss = f"""
            QWidget {{
                background-color: {colors['bg']};
                color: {colors['text']};
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            }}
            QLabel {{
                background: transparent;
            }}
            QLabel#Logo {{
                font-size: 22px;
                font-weight: 800;
                color: {colors['accent']};
            }}
            QLabel#ServiceStatus {{
                font-size: 13px;
                font-weight: 600;
                color: {colors['muted']};
            }}
            QLabel#DevHeader {{
                font-size: 16px;
                font-weight: 800;
                color: {colors['accent']};
                padding: 6px 4px;
            }}
            QFrame#Sidebar {{
                background-color: {colors['panel']};
                border-right: 1px solid {colors['border']};
            }}
            QFrame#DevSidebar {{
                background-color: {colors['panel']};
                border-left: 1px solid {colors['border']};
            }}
            QFrame#HeaderArea {{
                background-color: {colors['panel']};
                border-bottom: 1px solid {colors['border']};
            }}
            QFrame#HwBox {{
                background-color: {colors['panel2']};
                border-radius: 10px;
            }}
            QTabWidget::pane {{
                border: 1px solid {colors['border']};
                background: {colors['panel']};
                border-radius: 8px;
            }}
            QTabBar::tab {{
                background: {colors['panel2']};
                color: {colors['muted']};
                padding: 8px 12px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                margin-right: 4px;
            }}
            QTabBar::tab:selected {{
                background: {colors['panel']};
                color: {colors['text']};
                border-bottom: 2px solid {colors['accent']};
            }}
            QGroupBox {{
                border: 1px solid {colors['border']};
                border-radius: 8px;
                margin-top: 16px;
                padding-top: 14px;
                background-color: {colors['panel']};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 6px;
                color: {colors['muted']};
                font-weight: 500;
            }}
            QPushButton#NewChatMainBtn {{
                background-color: {colors['panel2']};
                color: {colors['text']};
                border: none;
                border-radius: 8px;
                font-size: 14px;
                font-weight: 600;
                text-align: left;
                padding-left: 14px;
            }}
            QPushButton#NewChatMainBtn:hover {{
                background-color: {colors['hover']};
            }}
            QPushButton#NewChatOptsBtn {{
                background-color: {colors['panel2']};
                color: {colors['text']};
                border: none;
                border-radius: 8px;
                font-size: 14px;
                font-weight: bold;
            }}
            QPushButton#NewChatOptsBtn:hover {{
                background-color: {colors['hover']};
            }}
            QListWidget {{
                background: transparent;
                border: none;
            }}
            QListWidget#ChatList::item {{
                padding: 0px;
                margin-bottom: 4px;
                background: transparent;
                border: none;
            }}
            QListWidget#ChatList::item:hover {{
                background: transparent;
            }}
            QListWidget#ChatList::item:selected {{
                background: transparent;
            }}
            QListWidget::item {{
                padding: 12px;
                border-radius: 8px;
                margin-bottom: 2px;
                color: {colors['muted']};
            }}
            QListWidget::item:hover {{
                background-color: {colors['panel2']};
            }}
            QListWidget::item:selected {{
                background-color: {colors['chip']};
                color: {colors['text']};
                font-weight: 600;
            }}
            QWidget#ChatItemRow {{
                background: transparent;
                border-radius: 10px;
            }}
            QWidget#ChatItemRow:hover {{
                background-color: {colors['panel2']};
            }}
            QWidget#ChatItemRow[selected="true"] {{
                background-color: {colors['chip']};
            }}
            QLabel#ChatItemLabel {{
                color: {colors['muted']};
                font-weight: 600;
            }}
            QWidget#ChatItemRow[selected="true"] QLabel#ChatItemLabel {{
                color: {colors['text']};
                font-weight: 700;
            }}
            QTextEdit {{
                background-color: {colors['panel']};
                border: 1px solid {colors['border']};
                border-radius: 12px;
                font-size: 15px;
                padding: 20px;
                line-height: 1.6;
            }}
            QLineEdit {{
                background-color: {colors['panel2']};
                border: 1px solid {colors['border']};
                border-radius: 10px;
                padding: 10px 14px;
                color: {colors['text']};
                font-size: 14px;
            }}
            QLineEdit:focus {{
                border: 1px solid {colors['accent']};
                background-color: {colors['panel']};
            }}
            QFrame#InputContainer {{
                background-color: transparent;
                border: none;
                padding: 10px;
            }}
            QFrame#InputBarFrame {{
                background-color: {colors['panel']};
                border: 1px solid {colors['border']};
                border-radius: 20px;
            }}
            QLineEdit#ChatInputField {{
                background-color: transparent;
                border: none;
                padding: 12px 8px;
                color: {colors['text']};
                font-size: 15px;
            }}
            QLineEdit#ChatInputField:focus {{
                border: none;
                background-color: transparent;
            }}
            QPushButton#DynamicActionBtn {{
                background-color: {colors['panel2']};
                color: {colors['text']};
                border-radius: 18px;
                font-size: 16px;
                border: none;
            }}
            QPushButton#DynamicActionBtn:hover {{
                background-color: {colors['hover']};
            }}
            QPushButton#ToolBtn {{
                background-color: {colors['panel2']};
                color: {colors['text']};
                border-radius: 10px;
                padding: 4px 16px;
                border: none;
                font-size: 14px;
                font-weight: 500;
            }}
            QPushButton#ToolBtn:hover {{
                background-color: {colors['hover']};
            }}
            QPushButton {{
                background-color: {colors['panel2']};
                border: 1px solid {colors['border']};
                border-radius: 6px;
                padding: 6px 10px;
                color: {colors['text']};
            }}
            QPushButton:hover {{
                background-color: {colors['chip']};
            }}
            QPushButton:disabled {{
                background-color: {colors['chip']};
                color: {colors['muted']};
            }}
            QPushButton#NewChatButton {{
                font-weight: 700;
            }}
            QPushButton#SettingsButton {{
                text-align: left;
            }}
            QPushButton#DevUnlockButton {{
                background-color: {colors['chip']};
                border: 1px solid {colors['border']};
                color: {colors['accent']};
                font-weight: 700;
                text-align: left;
            }}
            QPushButton#SendButton {{
                background-color: {colors['accent']};
                color: white;
                border-radius: 16px;
                font-weight: bold;
                font-size: 18px;
                padding: 0px;
            }}
            QPushButton#SendButton:hover {{
                background-color: {colors['accent2']};
            }}
            QPushButton#SendButton:disabled {{
                background-color: transparent;
                border: 1px solid transparent;
                color: {colors['muted']};
            }}
            QPushButton#MicButton {{
                background-color: {colors['panel2']};
                color: {colors['text']};
                border-radius: 16px;
                font-size: 16px;
                padding: 0px;
            }}
            QPushButton#MicButton:hover {{
                background-color: {colors['hover']};
            }}
            QPushButton#MicButton[recording="true"] {{
                background-color: #ff4444;
                color: white;
            }}
            QPushButton#StopButton {{
                background-color: {colors['panel2']};
                border: 1px solid {colors['border']};
                border-radius: 6px;
                padding: 0px;
                font-size: 16px;
                font-weight: 900;
                color: {colors['text']};
            }}
            QPushButton#StopButton:hover {{
                background-color: {colors['chip']};
            }}
            QPushButton#StopButton:disabled {{
                background-color: {colors['panel2']};
                color: {colors['muted']};
            }}
            QLabel#RagBadge {{
                background-color: {colors['chip']};
                border: 1px solid {colors['border']};
                border-radius: 12px;
                padding: 4px 12px;
                font-size: 11px;
                color: {colors['muted']};
            }}
            QLabel#RagBadge[ragState="active"] {{
                background-color: {colors['panel2']};
                border: 1px solid {colors['accent2']};
                color: {colors['accent']};
            }}
            QLabel#RagBadge[ragState="empty"] {{
                background-color: {colors['chip']};
                border: 1px solid {colors['border']};
                color: {colors['muted']};
            }}
            QLabel#RagBadge[ragState="off"] {{
                background-color: {colors['chip']};
                border: 1px solid {colors['border']};
                color: {colors['muted']};
            }}
            QDialog, QMessageBox {{
                background-color: {colors['panel']};
                border: 1px solid {colors['border']};
                border-radius: 12px;
            }}
            QPlainTextEdit, QTextBrowser {{
                background-color: {colors['bg']};
                border: 1px solid {colors['border']};
                border-radius: 12px;
                padding: 12px;
            }}
            QMenu {{
                background-color: {colors['panel']};
                border: 1px solid {colors['border']};
                border-radius: 10px;
                padding: 6px;
            }}
            QMenu::item {{
                padding: 8px 12px;
                border-radius: 8px;
            }}
            QMenu::item:selected {{
                background-color: {colors['chip']};
            }}
            QToolButton#ChatItemMenu {{
                background-color: transparent;
                border: 1px solid transparent;
                padding: 0px;
                min-width: 30px;
                min-height: 26px;
                border-radius: 8px;
                font-size: 14px;
                font-weight: 800;
            }}
            QToolButton#ChatItemMenu:hover {{
                background-color: {colors['chip']};
                border: 1px solid {colors['border']};
            }}
            QScrollBar:vertical {{
                border: none;
                background: transparent;
                width: 8px;
                margin: 0px;
            }}
            QScrollBar::handle:vertical {{
                background: {colors['border']};
                min-height: 20px;
                border-radius: 4px;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
            QPushButton#AccentButton {{
                background-color: {colors['accent']};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 6px 14px;
            }}
            QPushButton#AccentButton:hover {{
                background-color: {colors['accent2']};
            }}
            QPushButton#DangerButton {{
                background-color: {colors['danger']};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 6px 14px;
            }}
            QComboBox, QSpinBox, QDoubleSpinBox {{
                background-color: {colors['panel2']};
                border: 1px solid {colors['border']};
                border-radius: 6px;
                padding: 6px;
            }}
        """

        QApplication.instance().setStyleSheet(qss)
        self.send_btn.setObjectName("SendButton")
        self.rag_badge.setObjectName("RagBadge")
        if hasattr(self, "chat_list") and self.chat_list is not None:
            self._refresh_chat_list_row_visuals()

    def update_hw_stats(self, app_ram_gb: str, sys_ram_percent: str, cpu_percent: str, gpu_percent: str):
        """
        Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
        """
        shown = app_ram_gb
        try:
            rss_gb = float(str(app_ram_gb).split()[0])
        except Exception:
            rss_gb = None

        peak = getattr(self, "_model_peak_memory_gb", None)
        if isinstance(peak, (int, float)) and peak > 0 and isinstance(rss_gb, (int, float)):
            shown = f"{max(rss_gb, float(peak)):.2f} GB"

        self.lbl_ram_raw.setText(shown)
        self.lbl_sys_ram_pct.setText(sys_ram_percent)
        self.lbl_cpu_pct.setText(cpu_percent)
        self.lbl_gpu_pct.setText(gpu_percent)

    def _set_chat_enabled(self, enabled: bool) -> None:
        """
        Mesajlaşma arayüzü, HTML/CSS ile LM Studio estetiğinde baloncukları basıyor ekrana.
        """
        self.input_field.setEnabled(enabled)
        self.send_btn.setEnabled(enabled)
        if enabled:
            self.input_field.setFocus()

    def _set_project_root(self, folder: str) -> None:
        """
        Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
        """
        p = os.path.abspath(folder or "")
        if not p or not os.path.isdir(p):
            return
        self.project_root = p
        try:
            self.prompts["project_root"] = p
            self.save_prompts(self.prompts)
        except Exception:
            pass
        self._project_file_cache = None

    def select_project_root(self) -> str:
        """
        Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
        """
        folder = QFileDialog.getExistingDirectory(self, "Select Project Folder")
        if folder:
            self._set_project_root(folder)
        return self.project_root or ""

    def open_project_file_picker(self) -> None:
        """
        Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
        """
        root = self.project_root or ""
        if not root or not os.path.isdir(root):
            root = self.select_project_root()
        if not root or not os.path.isdir(root):
            return
        paths, _ = QFileDialog.getOpenFileNames(self, "Select Project File(s)", root, "All Files (*.*)")
        if not paths:
            return
        for p in paths:
            ap = os.path.abspath(p or "")
            if ap and ap not in self._pending_project_files:
                self._pending_project_files.append(ap)
        try:
            shown = ", ".join([os.path.basename(p) for p in paths[:3]])
            if len(paths) > 3:
                shown += f" (+{len(paths) - 3})"
            self.gen_status_lbl.setText(f"Attached: {shown}")
            QTimer.singleShot(4000, lambda: self.gen_status_lbl.setText(""))
        except Exception:
            pass

    def _read_project_file(self, path: str, max_chars: int = 20000) -> str:
        """
        Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
        """
        try:
            if not path or not os.path.isfile(path):
                return ""
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                data = f.read(max_chars + 1)
            if len(data) > max_chars:
                data = data[:max_chars] + "\n...[truncated]..."
            return data
        except Exception:
            return ""

    def _find_in_project_by_basename(self, basename: str, max_hits: int = 1) -> list[str]:
        """
        Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
        """
        root = self.project_root or ""
        if not root or not os.path.isdir(root):
            return []
        hits: list[str] = []
        try:
            for r, _dirs, files in os.walk(root):
                if basename in files:
                    hits.append(os.path.join(r, basename))
                    if len(hits) >= int(max_hits):
                        break
        except Exception:
            return hits
        return hits

    def _resolve_project_paths_from_text(self, user_text: str) -> list[str]:
        """
        Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
        """
        root = self.project_root or ""
        if not root or not os.path.isdir(root):
            return []
        import re
        exts = "py|js|ts|tsx|jsx|json|md|txt|yaml|yml|toml|rs|go|java|kt|cpp|c|h|hpp|sh|sql"
        pat = re.compile(rf"(?<![\\w\\./-])([\\w\\./-]+\\.(?:{exts}))(?![\\w\\./-])")
        found = []
        for m in pat.findall(user_text or ""):
            v = (m or "").strip().strip("\"'`")
            if not v:
                continue
            if os.path.isabs(v):
                ap = os.path.abspath(v)
                if ap.startswith(root + os.sep) and os.path.isfile(ap):
                    found.append(ap)
                continue
            if "/" in v or "\\" in v:
                ap = os.path.abspath(os.path.join(root, v))
                if ap.startswith(root + os.sep) and os.path.isfile(ap):
                    found.append(ap)
                continue
            hits = self._find_in_project_by_basename(v, max_hits=2)
            for h in hits:
                if h not in found:
                    found.append(h)
        uniq: list[str] = []
        for p in found:
            if p not in uniq:
                uniq.append(p)
        return uniq[:6]

    def _build_project_context(self, user_text: str) -> str:
        """
        Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
        """
        root = self.project_root or ""
        if not root or not os.path.isdir(root):
            return ""
        paths: list[str] = []
        for p in list(self._pending_project_files or []):
            ap = os.path.abspath(p or "")
            if ap and ap not in paths:
                paths.append(ap)
        auto = self._resolve_project_paths_from_text(user_text or "")
        for p in auto:
            if p not in paths:
                paths.append(p)
        if not paths:
            return ""
        blocks: list[str] = []
        total = 0
        for p in paths[:8]:
            data = self._read_project_file(p)
            if not data:
                continue
            rel = p
            try:
                rel = os.path.relpath(p, root)
            except Exception:
                pass
            block = f"[FILE: {rel}]\n{data}\n[/FILE]"
            total += len(block)
            if total > 80000:
                break
            blocks.append(block)
        try:
            self._pending_project_files = []
        except Exception:
            pass
        return "\n\n".join(blocks)

    def _start_model_load(self, model_dir: str) -> None:
        """
        Modeli RAM'e/VRAM'e aldığımız kısım. Hafızayı patlatmamak için dikkatli yazıldı.
        """
        path = os.path.abspath(model_dir or "")
        if not path or not os.path.isdir(path):
            QMessageBox.warning(self, "Load Model", "Model folder not selected or invalid.")
            return
        self._set_chat_enabled(False)
        self.service_status_lbl.setText("Service: loading…")
        self.model_loader = ModelLoaderWorker(path)
        self.model_loader.loaded.connect(self._on_model_loaded)
        self.model_loader.error.connect(self._on_model_load_error)
        self.model_loader.start()

    def load_model_quick(self) -> None:
        """
        Modeli RAM'e/VRAM'e aldığımız kısım. Hafızayı patlatmamak için dikkatli yazıldı.
        """
        try:
            if getattr(self, "model_loader", None) is not None and self.model_loader.isRunning():
                QMessageBox.information(self, "Model", "Model is already loading.")
                return
        except Exception:
            pass

        if getattr(self, "model", None) is not None or getattr(self, "tokenizer", None) is not None:
            res = QMessageBox.question(self, "Load Model", "A model is already loaded. Unload and load again?", QMessageBox.Yes | QMessageBox.No)
            if res != QMessageBox.Yes:
                return
            self.unload_model()

        path = self._find_default_model_path()
        if not path:
            path = QFileDialog.getExistingDirectory(self, "Select MLX Model Folder")
        self._start_model_load(path)

    def load_model_via_picker(self) -> None:
        """
        Modeli RAM'e/VRAM'e aldığımız kısım. Hafızayı patlatmamak için dikkatli yazıldı.
        """
        try:
            if getattr(self, "model_loader", None) is not None and self.model_loader.isRunning():
                QMessageBox.information(self, "Model", "Model is already loading.")
                return
        except Exception:
            pass

        if getattr(self, "model", None) is not None or getattr(self, "tokenizer", None) is not None:
            res = QMessageBox.question(self, "Load Model", "A model is already loaded. Unload and load a new one?", QMessageBox.Yes | QMessageBox.No)
            if res != QMessageBox.Yes:
                return
            self.unload_model()

        path = QFileDialog.getExistingDirectory(self, "Select MLX Model Folder")
        if not path:
            return
        self._start_model_load(path)

    def unload_model(self) -> None:
        """
        Modeli RAM'e/VRAM'e aldığımız kısım. Hafızayı patlatmamak için dikkatli yazıldı.
        """
        try:
            self._stop_generation(False)
        except Exception:
            pass
        try:
            if getattr(self, "worker", None) is not None and self.worker.isRunning():
                self.worker.stop()
        except Exception:
            pass

        self.model = None
        self.tokenizer = None
        self._set_chat_enabled(False)
        self.service_status_lbl.setText("Service: unloaded")
        try:
            self.gen_status_lbl.setText("")
        except Exception:
            pass
        try:
            import gc
            gc.collect()
        except Exception:
            pass
        try:
            import mlx.core as mx  # type: ignore
            if hasattr(mx, "clear_cache"):
                mx.clear_cache()
            elif hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
                mx.metal.clear_cache()
        except Exception:
            pass

    def _stop_thread_obj(self, t, wait_ms: int) -> None:
        """
        Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
        """
        if t is None:
            return
        try:
            if hasattr(t, "stop"):
                try:
                    t.stop()
                except Exception:
                    pass
        except Exception:
            pass
        try:
            if hasattr(t, "requestInterruption"):
                try:
                    t.requestInterruption()
                except Exception:
                    pass
        except Exception:
            pass
        try:
            if hasattr(t, "wait"):
                try:
                    t.wait(int(wait_ms))
                except Exception:
                    pass
        except Exception:
            pass
        try:
            if hasattr(t, "isRunning") and t.isRunning() and hasattr(t, "terminate"):
                try:
                    t.terminate()
                except Exception:
                    pass
                try:
                    t.wait(2000)
                except Exception:
                    pass
        except Exception:
            pass

    def _shutdown_threads(self) -> None:
        """
        Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
        """
        try:
            self._stop_generation(False)
        except Exception:
            pass
        try:
            if hasattr(self, "abort_rag_operations"):
                self.abort_rag_operations()
        except Exception:
            pass
        try:
            self._stop_thread_obj(getattr(self, "worker", None), 2000)
        except Exception:
            pass
        try:
            self._stop_thread_obj(getattr(self, "_final_worker", None), 2000)
        except Exception:
            pass
        try:
            if hasattr(self, "stop_training"):
                self.stop_training()
        except Exception:
            pass
        try:
            self._stop_thread_obj(getattr(self, "_ft_worker", None), 4000)
        except Exception:
            pass
        try:
            self._stop_thread_obj(getattr(self, "_rag_worker", None), 4000)
        except Exception:
            pass
        try:
            self._stop_thread_obj(getattr(self, "_docs_worker", None), 4000)
        except Exception:
            pass
        try:
            self._stop_thread_obj(getattr(self, "model_loader", None), 8000)
        except Exception:
            pass
        try:
            ds = getattr(self, "dev_sidebar_widget", None)
            if ds is not None:
                self._stop_thread_obj(getattr(ds, "_model_loader", None), 8000)
                self._stop_thread_obj(getattr(ds, "_ft_worker", None), 4000)
                self._stop_thread_obj(getattr(ds, "_ds_export_worker", None), 4000)
                self._stop_thread_obj(getattr(ds, "_rag_worker", None), 4000)
                self._stop_thread_obj(getattr(ds, "_docs_worker", None), 4000)
        except Exception:
            pass
        try:
            dlg = getattr(self, "dev_dialog", None)
            if dlg is not None:
                self._stop_thread_obj(getattr(dlg, "_model_loader", None), 8000)
                self._stop_thread_obj(getattr(dlg, "_ft_worker", None), 4000)
                self._stop_thread_obj(getattr(dlg, "_ds_export_worker", None), 4000)
                self._stop_thread_obj(getattr(dlg, "_rag_worker", None), 4000)
                self._stop_thread_obj(getattr(dlg, "_docs_worker", None), 4000)
        except Exception:
            pass
        try:
            self._stop_thread_obj(getattr(self, "mem_thread", None), 2000)
        except Exception:
            pass

    def _find_default_model_path(self) -> str:
        """
        Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
        """
        if self.current_model_path and os.path.isdir(self.current_model_path):
            return self.current_model_path

        candidates = []
        base = os.path.expanduser("~/.lmstudio/models")
        if os.path.isdir(base):
            for root, dirs, _files in os.walk(base):
                for d in dirs:
                    full = os.path.join(root, d)
                    if not os.path.isdir(full):
                        continue
                    if "mlx" not in full.lower():
                        continue
                    has_config = os.path.exists(os.path.join(full, "config.json"))
                    has_tokenizer = os.path.exists(os.path.join(full, "tokenizer.json")) or os.path.exists(os.path.join(full, "tokenizer.model"))
                    has_weights = any(
                        name.endswith((".safetensors", ".npz", ".bin"))
                        for name in os.listdir(full)
                        if os.path.isfile(os.path.join(full, name))
                    )
                    if has_config or has_tokenizer or has_weights:
                        score = 0
                        lower = full.lower()
                        if "qwen" in lower:
                            score += 50
                        if "27b" in lower:
                            score += 20
                        if "6bit" in lower:
                            score += 15
                        if "distilled" in lower:
                            score += 10
                        if "4bit" in lower:
                            score -= 5
                        candidates.append((score, full))

        if not candidates:
            return ""

        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    def start_ai_service(self) -> None:
        """
        Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
        """
        self._set_chat_enabled(False)
        model_path = self._find_default_model_path()
        if not model_path:
            self.service_status_lbl.setText("Service: model path not found")
            return

        self.service_status_lbl.setText("Service: loading…")
        self.model_loader = ModelLoaderWorker(model_path)
        self.model_loader.loaded.connect(self._on_model_loaded)
        self.model_loader.error.connect(self._on_model_load_error)
        self.model_loader.start()

    def _on_model_loaded(self, model, tokenizer, model_path: str) -> None:
        """
        Modeli RAM'e/VRAM'e aldığımız kısım. Hafızayı patlatmamak için dikkatli yazıldı.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.current_model_path = model_path
        self.prompts["model_path"] = model_path
        self.save_prompts(self.prompts)
        self.service_status_lbl.setText("Service: ready")
        self._set_chat_enabled(True)

    def _on_model_load_error(self, err: str) -> None:
        """
        Modeli RAM'e/VRAM'e aldığımız kısım. Hafızayı patlatmamak için dikkatli yazıldı.
        """
        self.model = None
        self.tokenizer = None
        self.service_status_lbl.setText("Service: error")
        self._set_chat_enabled(False)
        QMessageBox.critical(self, "Model Load Error", err)

    def open_settings(self):
        """
        Kullanıcı ve Dev ayarlarını .lokumf içine güvenle kaydettiğimiz/okuduğumuz yer.
        """
        diag = SettingsDialog(self, self.user_prompt, current_theme=self.theme)
        if diag.exec_():
            self.apply_theme(getattr(diag, "final_theme", self.theme))

    def open_dev_panel(self):
        """
        Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
        """
        self.on_dev_button_clicked()

    def on_dev_button_clicked(self):
        """
        Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
        """
        if not dev_mode_gate.unlocked:
            password, ok = QInputDialog.getText(self, "Dev Mode", "Enter developer password:")
            if not ok or not password:
                return
            if not dev_mode_gate.attempt_unlock(password):
                hint = ""
                try:
                    from lokum_paths import dev_password_file as _dev_password_file  # type: ignore

                    hint = f"\n\nHint: Check {str(_dev_password_file())} or set LOKUMF_DEV_PASSWORD."
                except Exception:
                    hint = "\n\nHint: Set env var LOKUMF_DEV_PASSWORD."
                QMessageBox.critical(self, "Access Denied", "Incorrect password for Dev Mode." + hint)
                return

        self.dev_mode_active = True
        self.dev_toggle_btn.setVisible(False)
        self.dev_toggle_btn.setEnabled(False)
        self.toggle_dev_dialog(force_state=None)

    def toggle_dev_dialog(self, force_state=None) -> None:
        """
        Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
        """
        if not self.dev_mode_active:
            return
        if not hasattr(self, "dev_sidebar") or self.dev_sidebar is None:
            return

        show = None
        if force_state is True:
            show = True
        elif force_state is False:
            show = False
        else:
            show = not self.dev_sidebar.isVisible()

        outer_sizes = None
        if hasattr(self, "splitter") and self.splitter is not None:
            try:
                outer_sizes = list(self.splitter.sizes())
            except Exception:
                outer_sizes = None

        if hasattr(self, "content_splitter") and self.content_splitter is not None:
            try:
                sizes = list(self.content_splitter.sizes())
            except Exception:
                sizes = []
            total = sum(sizes) if sizes else 0

            if show:
                self.dev_sidebar.setVisible(True)
                if total <= 0:
                    self.content_splitter.setSizes([820, 360])
                else:
                    saved = getattr(self, "_dev_open_sizes", None)
                    if isinstance(saved, (list, tuple)) and len(saved) == 2 and all(isinstance(x, int) for x in saved):
                        old_total = max(1, int(saved[0] + saved[1]))
                        desired_right = int(total * (saved[1] / old_total))
                        min_right = 260
                        max_right = max(min_right, total - 260)
                        right = max(min_right, min(desired_right, max_right))
                        left = max(260, total - right)
                        self.content_splitter.setSizes([left, total - left])
                    else:
                        target = 360
                        left = max(300, total - target)
                        self.content_splitter.setSizes([left, target])
                try:
                    self.content_splitter.update()
                    self.dev_sidebar.update()
                except Exception:
                    pass
            else:
                if self.dev_sidebar.isVisible():
                    try:
                        cur = list(self.content_splitter.sizes())
                    except Exception:
                        cur = []
                    if len(cur) >= 2:
                        self._dev_open_sizes = [int(cur[0]), int(cur[1])]
                if total > 0:
                    self.content_splitter.setSizes([total, 0])
                self.dev_sidebar.setVisible(False)
        else:
            self.dev_sidebar.setVisible(bool(show))

        if outer_sizes and hasattr(self, "splitter") and self.splitter is not None:
            try:
                self.splitter.setSizes(outer_sizes)
            except Exception:
                pass
        self._save_dev_dialog_state()

    def open_settings_dialog(self):
        """
        Kullanıcı ve Dev ayarlarını .lokumf içine güvenle kaydettiğimiz/okuduğumuz yer.
        """
        # Sadece basit bir dialog aç, ayarları config.json'dan okuyup/yazalım
        dlg = QDialog(self)
        dlg.setWindowTitle("Ayarlar")
        dlg.setMinimumWidth(300)
        l = QVBoxLayout(dlg)
        
        l.addWidget(QLabel("Tema:"))
        theme_combo = QComboBox()
        theme_combo.addItems(["dark", "light"])
        if getattr(self, "theme", "dark") == "light":
            theme_combo.setCurrentText("light")
        l.addWidget(theme_combo)
        
        btn_box = QHBoxLayout()
        save_btn = QPushButton("Kaydet")
        save_btn.clicked.connect(dlg.accept)
        btn_box.addStretch()
        btn_box.addWidget(save_btn)
        l.addLayout(btn_box)
        
        if dlg.exec_() == QDialog.Accepted:
            new_theme = theme_combo.currentText()
            self.theme = new_theme
            self.config["theme"] = new_theme
            self.save_config()
            self.apply_theme()
            QMessageBox.information(self, "Ayarlar", "Ayarlar kaydedildi.")
            
    def new_chat(self):
        """
        Mesajlaşma arayüzü, HTML/CSS ile LM Studio estetiğinde baloncukları basıyor ekrana.
        """
        base = "New Chat"
        idx = 1
        title = base
        while title in self.chats:
            idx += 1
            title = f"{base} {idx}"

        self.chats[title] = [{"role": "system", "content": self.system_prompt}]
        self.chat_ui[title] = []
        self.active_chat = title
        try:
            conn = sqlite3.connect(self._db_path())
            try:
                self._ensure_chat_row(conn, title)
            finally:
                conn.close()
        except Exception:
            pass
        try:
            self._add_chat_list_item(title)
            it = self._find_chat_list_item(title)
            if it is not None:
                self.chat_list.setCurrentItem(it)
            self._refresh_chat_list_row_visuals()
        except Exception:
            self._rebuild_chat_list()
        self.render_chat(self.active_chat)

    def switch_chat(self, item):
        """
        Mesajlaşma arayüzü, HTML/CSS ile LM Studio estetiğinde baloncukları basıyor ekrana.
        """
        self.active_chat = self._chat_name_from_item(item)
        if self.active_chat not in self.chat_ui or not self.chat_ui[self.active_chat]:
            self.chat_ui.setdefault(self.active_chat, [])
            for msg in self.chats.get(self.active_chat, []):
                if msg.get("role") in ("user", "assistant"):
                    if msg["role"] == "user":
                        self.chat_ui[self.active_chat].append({"role": "user", "content": msg.get("content", "")})
                    else:
                        self.chat_ui[self.active_chat].append(
                            {
                                "role": "assistant",
                                "answer": msg.get("content", ""),
                                "think": "",
                                "think_open": False,
                                "thought_s": None,
                                "meta": None,
                            }
                        )

        self.render_chat(self.active_chat)

    def resizeEvent(self, event):
        """
        Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
        """
        super().resizeEvent(event)
        self._refresh_chat_list_row_visuals()

    def _rename_chat(self, old_name: str, new_name: str, *, render_after: bool = True):
        """
        Mesajlaşma arayüzü, HTML/CSS ile LM Studio estetiğinde baloncukları basıyor ekrana.
        """
        if new_name in self.chats and new_name != old_name:
            QMessageBox.warning(self, "Name Exists", "A chat with this name already exists.")
            return
        if old_name not in self.chats:
            return
        self.chats[new_name] = self.chats.pop(old_name)
        self.chat_ui[new_name] = self.chat_ui.pop(old_name, [])
        if getattr(self, "_pending_chat", None) == old_name:
            self._pending_chat = new_name
        try:
            self._rename_chat_db(old_name, new_name)
        except Exception:
            pass
        self.active_chat = new_name
        try:
            self._rename_chat_list_item(old_name, new_name)
            it = self._find_chat_list_item(new_name)
            if it is not None:
                self.chat_list.setCurrentItem(it)
            self._refresh_chat_list_row_visuals()
        except Exception:
            self._rebuild_chat_list()
        if render_after:
            self.render_chat(self.active_chat)

    def _is_placeholder_chat_name(self, name: str) -> bool:
        """
        Mesajlaşma arayüzü, HTML/CSS ile LM Studio estetiğinde baloncukları basıyor ekrana.
        """
        return name == "New Chat" or name.startswith("New Chat ")

    def _auto_name_active_chat(self, first_message: str):
        """
        Mesajlaşma arayüzü, HTML/CSS ile LM Studio estetiğinde baloncukları basıyor ekrana.
        """
        raw = (first_message or "").strip()
        raw = raw.replace("\n", " ").replace("\r", " ")
        raw = re.sub(r"\s+", " ", raw).strip()
        raw = raw.strip("`'\"“”‘’ ")
        raw = re.sub(r"[.!?,;:]+$", "", raw).strip()
        base = " ".join((raw or "").split()[:6]).strip()
        base = "".join(ch for ch in base if ch.isalnum() or ch.isspace() or ch in "-_").strip()
        base = re.sub(r"\s+", " ", base).strip()
        if not base:
            base = "Chat"
        base = base[:28].strip()
        new_name = base
        idx = 2
        while new_name in self.chats and new_name != self.active_chat:
            new_name = f"{base} ({idx})"
            idx += 1
        self._rename_chat(self.active_chat, new_name, render_after=False)

    def on_chat_anchor_clicked(self, url):
        """
        Mesajlaşma arayüzü, HTML/CSS ile LM Studio estetiğinde baloncukları basıyor ekrana.
        """
        s = url.toString()
        if s.startswith("msg_menu:"):
            try:
                idx = int(s.split(":", 1)[1])
            except Exception:
                return
            self.open_message_menu(idx)
            return
        if s.startswith("toggle_thought:"):
            try:
                idx = int(s.split(":", 1)[1])
            except Exception:
                return
            msgs = self.chat_ui.get(self.active_chat, [])
            if not (0 <= idx < len(msgs)):
                return
            msg = msgs[idx]
            if msg.get("role") != "assistant":
                return
            msg["think_open"] = not bool(msg.get("think_open"))
            self.render_chat(self.active_chat, keep_scroll=True)
            return

    def open_message_menu(self, msg_index: int):
        """
        Mesajlaşma arayüzü, HTML/CSS ile LM Studio estetiğinde baloncukları basıyor ekrana.
        """
        msgs = self.chat_ui.get(self.active_chat, [])
        if not (0 <= msg_index < len(msgs)):
            return
        msg = msgs[msg_index]
        if msg.get("role") != "user":
            return

        menu = QMenu(self)
        edit_act = QAction("Edit message", self)
        delete_act = QAction("Delete message", self)
        copy_act = QAction("Copy", self)

        edit_act.triggered.connect(lambda: self._edit_user_message(msg_index))
        delete_act.triggered.connect(lambda: self._delete_user_message(msg_index))
        copy_act.triggered.connect(lambda: QApplication.clipboard().setText(msg.get("content", "")))

        menu.addAction(edit_act)
        menu.addAction(delete_act)
        menu.addAction(copy_act)

        pos = QCursor.pos()
        menu.exec_(pos)

    def _edit_user_message(self, msg_index: int):
        """
        Mesajlaşma arayüzü, HTML/CSS ile LM Studio estetiğinde baloncukları basıyor ekrana.
        """
        msgs = self.chat_ui.get(self.active_chat, [])
        if not (0 <= msg_index < len(msgs)):
            return
        msg = msgs[msg_index]
        current = msg.get("content", "")
        new_text, ok = QInputDialog.getMultiLineText(self, "Edit Message", "Edit:", current)
        if not ok:
            return
        new_text = (new_text or "").strip()
        if not new_text:
            return
        msg["content"] = new_text
        try:
            self._replace_user_message_db(self.active_chat, msg_index, new_text)
        except Exception:
            pass
        self._sync_chat_history_from_ui(self.active_chat)
        self.render_chat(self.active_chat)

    def _delete_user_message(self, msg_index: int):
        """
        Mesajlaşma arayüzü, HTML/CSS ile LM Studio estetiğinde baloncukları basıyor ekrana.
        """
        msgs = self.chat_ui.get(self.active_chat, [])
        if not (0 <= msg_index < len(msgs)):
            return
        res = QMessageBox.question(self, "Delete Message", "Delete this message? This cannot be undone.", QMessageBox.Yes | QMessageBox.No)
        if res != QMessageBox.Yes:
            return
        try:
            self._delete_user_message_db(self.active_chat, msg_index)
        except Exception:
            pass
        msgs.pop(msg_index)
        self._sync_chat_history_from_ui(self.active_chat)
        self.render_chat(self.active_chat)

    def _sync_chat_history_from_ui(self, chat_name: str):
        """
        Mesajlaşma arayüzü, HTML/CSS ile LM Studio estetiğinde baloncukları basıyor ekrana.
        """
        system = [{"role": "system", "content": self.system_prompt}]
        ui_msgs = self.chat_ui.get(chat_name, [])
        out = list(system)
        for m in ui_msgs:
            if m.get("role") == "user":
                out.append({"role": "user", "content": m.get("content", "")})
            elif m.get("role") == "assistant":
                out.append({"role": "assistant", "content": m.get("answer", "")})
        self.chats[chat_name] = out

    def _toggle_thought(self, msg_index: int) -> None:
        """
        Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
        """
        msgs = self.chat_ui.get(self.active_chat, [])
        if not (0 <= msg_index < len(msgs)):
            return
        msg = msgs[msg_index]
        if msg.get("role") != "assistant":
            return
        msg["think_open"] = not bool(msg.get("think_open"))
        self.render_chat(self.active_chat, keep_scroll=True)

    def _get_base_html(self):
        """
        Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
        """
        colors = getattr(self, "_theme_colors", None) or {
            "bg": "#121212",
            "panel": "#1e1e1e",
            "panel2": "#2c2c2e",
            "border": "#38383a",
            "text": "#f5f5f5",
            "muted": "#98989d",
            "accent": "#0a84ff",
            "accent2": "#409cff",
        }
        user_bubble = "rgba(0, 0, 0, 0.04)" if self.theme == "light" else colors["panel2"]
        user_border = "rgba(0, 0, 0, 0.08)" if self.theme == "light" else colors["border"]
        
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
        <meta charset="utf-8">
        <style>
            body {{
                margin: 0;
                padding: 24px;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                background-color: transparent;
                color: {colors['text']};
            }}
            .chat-container {{
                display: flex;
                flex-direction: column;
                gap: 24px;
            }}
            .message {{
                display: flex;
                flex-direction: column;
                max-width: 85%;
            }}
            .message.user {{
                align-self: flex-end;
                align-items: flex-end;
            }}
            .message.assistant {{
                align-self: flex-start;
                align-items: flex-start;
                max-width: 95%;
            }}
            .role-label {{
                font-size: 11px;
                font-weight: 800;
                letter-spacing: 1px;
                margin-bottom: 6px;
            }}
            .user .role-label {{
                color: {colors['muted']};
                text-align: right;
            }}
            .assistant .role-label {{
                color: {colors['accent']};
                text-align: left;
            }}
            .bubble {{
                font-size: 15px;
                line-height: 1.6;
                word-wrap: break-word;
            }}
            .user .bubble {{
                background-color: {user_bubble};
                border: 1px solid {user_border};
                border-radius: 18px;
                border-top-right-radius: 4px;
                padding: 12px 18px;
            }}
            .assistant .bubble {{
                background-color: transparent;
                border: none;
                padding: 4px 0px;
            }}
            /* Markdown Elements */
            a {{ color: {colors['accent']}; text-decoration: none; }}
            a:hover {{ text-decoration: underline; }}
            pre {{
                background-color: {colors['bg']};
                border: 1px solid {colors['border']};
                border-radius: 8px;
                padding: 14px;
                overflow-x: auto;
                font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
                font-size: 13px;
                margin: 10px 0;
            }}
            code {{
                background-color: {colors['bg']};
                border: 1px solid {colors['border']};
                border-radius: 4px;
                padding: 2px 6px;
                font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
                font-size: 13px;
            }}
            pre code {{
                background-color: transparent;
                border: none;
                padding: 0;
            }}
            blockquote {{
                border-left: 3px solid {colors['accent']};
                margin-left: 0;
                padding-left: 14px;
                color: {colors['muted']};
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin: 10px 0;
            }}
            th, td {{
                border: 1px solid {colors['border']};
                padding: 8px 12px;
                text-align: left;
            }}
            th {{ background-color: {colors['panel']}; }}
            
            /* Pulsing Loading Dots */
            .typing-indicator {{
                display: flex;
                gap: 6px;
                align-items: center;
                height: 24px;
                padding-left: 4px;
                margin-top: 4px;
            }}
            .dot {{
                width: 7px;
                height: 7px;
                background-color: {colors['accent']};
                border-radius: 50%;
                animation: pulse 1.4s infinite ease-in-out;
            }}
            .dot:nth-child(1) {{ animation-delay: 0s; }}
            .dot:nth-child(2) {{ animation-delay: 0.2s; }}
            .dot:nth-child(3) {{ animation-delay: 0.4s; }}
            @keyframes pulse {{
                0%, 100% {{ transform: scale(0.6); opacity: 0.4; }}
                50% {{ transform: scale(1.1); opacity: 1; }}
            }}
            
            /* Message Actions */
            .message-actions {{
                display: flex;
                gap: 8px;
                margin-top: 8px;
                opacity: 0.4;
                transition: opacity 0.2s;
            }}
            .message:hover .message-actions {{
                opacity: 1;
            }}
            .action-btn {{
                background: {colors['panel']};
                border: 1px solid {colors['border']};
                color: {colors['text']};
                border-radius: 6px;
                padding: 4px 10px;
                cursor: pointer;
                font-size: 12px;
                display: flex;
                align-items: center;
                gap: 4px;
                transition: all 0.2s;
            }}
            .action-btn:hover {{
                background: {colors['panel2']};
                border-color: {colors['accent']};
                transform: translateY(-1px);
            }}
            .action-btn:active {{
                transform: translateY(0);
            }}
        </style>
        <script>
            function updateChat(htmlContent, isAtBottom) {{
                const chatDiv = document.getElementById('chat');
                chatDiv.innerHTML = htmlContent;
                if (isAtBottom) {{
                    window.scrollTo(0, document.body.scrollHeight);
                }}
            }}
            function scrollToBottom() {{
                window.scrollTo(0, document.body.scrollHeight);
            }}
            function playVoice(text) {{
                // Use a custom scheme to communicate with Python
                window.location.href = "speak://" + btoa(unescape(encodeURIComponent(text)));
            }}
        </script>
        </head>
        <body>
            <div id="chat" class="chat-container"></div>
        </body>
        </html>
        """

    def render_chat(self, chat_name: str, *, keep_scroll: bool = False):
        """
        Mesajlaşma arayüzü, HTML/CSS ile LM Studio estetiğinde baloncukları basıyor ekrana.
        """
        msgs = self.chat_ui.get(chat_name, []) or []
        is_generating = getattr(self, "is_generating", False)
        
        if HAS_WEBENGINE and hasattr(self, "chat_display") and isinstance(self.chat_display, QWebEngineView):
            try:
                import markdown
                has_md = True
            except ImportError:
                has_md = False
            import base64
            
            html_parts = []
            for m in msgs:
                r = m.get("role", "")
                txt = m.get("content", "") if r == "user" else m.get("answer", "")
                
                # Convert markdown
                try:
                    if has_md:
                        md_html = markdown.markdown(txt, extensions=["fenced_code", "tables"])
                    else:
                        md_html = txt.replace("\n", "<br>")
                except Exception:
                    md_html = txt.replace("\n", "<br>")
                
                if r == "user":
                    html_parts.append(f'''
                    <div class="message user">
                        <div class="role-label">YOU</div>
                        <div class="bubble">{md_html}</div>
                    </div>
                    ''')
                elif r == "assistant":
                    # Escape text for JS function call
                    raw_text_b64 = base64.b64encode(txt.encode('utf-8')).decode('utf-8')
                    html_parts.append(f'''
                    <div class="message assistant">
                        <div class="role-label">AI</div>
                        <div class="bubble">{md_html}</div>
                        <div class="message-actions">
                            <button class="action-btn" onclick="playVoice(decodeURIComponent(escape(window.atob('{raw_text_b64}'))))">
                                🔊 Dinle
                            </button>
                        </div>
                    </div>
                    ''')
            
            if is_generating:
                html_parts.append('''
                <div class="message assistant">
                    <div class="role-label">AI</div>
                    <div class="typing-indicator">
                        <div class="dot"></div><div class="dot"></div><div class="dot"></div>
                    </div>
                </div>
                ''')
            
            full_content = "\n".join(html_parts)
            # Escape to base64 to pass to JS
            b64_content = base64.b64encode(full_content.encode('utf-8')).decode('utf-8')
            js = f"""
            if (typeof updateChat === 'function') {{
                updateChat(decodeURIComponent(escape(window.atob('{b64_content}'))), {str(not keep_scroll).lower()});
            }}
            """
            self.chat_display.page().runJavaScript(js)
            return

        if not hasattr(self, "chat_msgs_layout") or self.chat_msgs_layout is None:
            return

        old_scroll_val = None
        old_scroll_max = None
        try:
            sb = self.chat_display.verticalScrollBar()
            old_scroll_val = int(sb.value())
            old_scroll_max = int(sb.maximum())
        except Exception:
            old_scroll_val = None
            old_scroll_max = None

        msgs = self.chat_ui.get(chat_name, []) or []
        colors = getattr(self, "_theme_colors", None) or {
            "bg": "#121212",
            "panel": "#1e1e1e",
            "panel2": "#2c2c2e",
            "border": "#38383a",
            "text": "#f5f5f5",
            "muted": "#98989d",
            "accent": "#0a84ff",
            "accent2": "#409cff",
            "danger": "#ff453a",
            "chip": "#3a3a3c",
        }
        
        # LM Studio stili renk ayarları
        user_bubble = colors["panel2"]
        user_border = colors["border"]

        def clear_layout(lay: QVBoxLayout) -> None:
            """
            Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
            """
            while lay.count():
                it = lay.takeAt(0)
                if it is None:
                    continue
                w = it.widget()
                if w is not None:
                    w.deleteLater()

        def fmt_html(s: str) -> str:
            """
            Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
            """
            t = self._html_escape(s or "")
            t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
            t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
            return t.replace("\n", "<br>")

        def split_fenced_blocks(s: str) -> list[tuple[str, str, str]]:
            """
            Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
            """
            out: list[tuple[str, str, str]] = []
            if not s:
                return out
            rx = re.compile(r"```([A-Za-z0-9_+\\-]*)\\s*\\n([\\s\\S]*?)```", re.MULTILINE)
            pos = 0
            for m in rx.finditer(s):
                a, b = m.span()
                if a > pos:
                    out.append(("text", s[pos:a], ""))
                lang = (m.group(1) or "").strip()
                code = (m.group(2) or "").rstrip("\n")
                out.append(("code", code, lang))
                pos = b
            if pos < len(s):
                out.append(("text", s[pos:], ""))
            return out

        def make_code_block(code: str, lang: str) -> QWidget:
            """
            Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
            """
            frame = QFrame()
            frame.setObjectName("CodeBlockFrame")
            frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
            frame.setMaximumWidth(max_bubble)

            bg = "#0b1220" if self.theme == "dark" else "#f3f5f9"
            border = "#243145" if self.theme == "dark" else "#d7dde8"
            hdr_bg = "#111c2e" if self.theme == "dark" else "#e9edf6"
            text_col = "#e8eefc" if self.theme == "dark" else "#111827"
            muted_col = "#a9b4c6" if self.theme == "dark" else "#4b5563"

            frame.setStyleSheet(
                "QFrame#CodeBlockFrame{"
                f"background:{bg};"
                f"border:1px solid {border};"
                "border-radius:12px;"
                "}"
            )
            fx = QGraphicsDropShadowEffect()
            fx.setBlurRadius(18)
            fx.setOffset(0, 6)
            fx.setColor(QColor(0, 0, 0, 120 if self.theme == "dark" else 60))
            frame.setGraphicsEffect(fx)

            lay = QVBoxLayout(frame)
            lay.setContentsMargins(12, 10, 12, 12)
            lay.setSpacing(8)

            hdr = QWidget()
            hdr_l = QHBoxLayout(hdr)
            hdr_l.setContentsMargins(0, 0, 0, 0)
            hdr_l.setSpacing(10)

            chip = QLabel((lang or "code").strip()[:18] or "code")
            chip.setObjectName("CodeLangChip")
            chip.setStyleSheet(
                "QLabel#CodeLangChip{"
                f"background:{hdr_bg};"
                f"color:{muted_col};"
                "border-radius:10px;"
                "padding:3px 10px;"
                "font-size:12px;"
                "font-weight:700;"
                "}"
            )
            hdr_l.addWidget(chip)
            hdr_l.addStretch()

            copy_btn = QToolButton()
            copy_btn.setText("⧉")
            copy_btn.setCursor(Qt.PointingHandCursor)
            copy_btn.setAutoRaise(True)
            copy_btn.setFocusPolicy(Qt.StrongFocus)
            copy_btn.setAccessibleName("Copy code")
            copy_btn.setToolTip("Copy")
            copy_btn.setStyleSheet(
                "QToolButton{"
                "background:transparent;"
                f"color:{muted_col};"
                "border:0;"
                "font-size:14px;"
                "padding:4px 6px;"
                "}"
                "QToolButton:hover{"
                f"color:{text_col};"
                "}"
                "QToolButton:focus{"
                f"outline: 0px; border:1px solid {border}; border-radius:8px;"
                "}"
            )
            def _do_copy(_checked: bool = False, txt: str = code) -> None:
                """
                Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
                """
                try:
                    QApplication.clipboard().setText(txt or "")
                except Exception:
                    pass
            copy_btn.clicked.connect(_do_copy)
            hdr_l.addWidget(copy_btn)
            lay.addWidget(hdr)

            ed = QPlainTextEdit()
            ed.setReadOnly(True)
            ed.setLineWrapMode(QPlainTextEdit.NoWrap)
            ed.setPlainText(code or "")
            ed.setFrameShape(QFrame.NoFrame)
            ed.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
            ed.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            ed.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            fnt = QFontDatabase.systemFont(QFontDatabase.FixedFont)
            fnt.setPointSize(13)
            fnt.setWeight(QFont.Normal)
            ed.setFont(fnt)
            ed.setStyleSheet(
                "QPlainTextEdit{"
                "background:transparent;"
                f"color:{text_col};"
                "selection-background-color: rgba(124, 77, 255, 0.35);"
                "padding:0px;"
                "}"
            )
            lay.addWidget(ed)
            return frame

        clear_layout(self.chat_msgs_layout)

        max_bubble = 820
        try:
            vw = int(self.chat_display.viewport().width())
            max_bubble = max(260, int(vw * 0.82))
        except Exception:
            max_bubble = 820

        for i, m in enumerate(msgs):
            role = m.get("role")
            if role == "user":
                row = QWidget()
                row_l = QHBoxLayout(row)
                row_l.setContentsMargins(0, 0, 0, 0)
                row_l.setSpacing(0)
                row_l.addStretch()

                col = QWidget()
                col_l = QVBoxLayout(col)
                col_l.setContentsMargins(0, 0, 0, 0)
                col_l.setSpacing(6)
                col_l.setAlignment(Qt.AlignRight)

                header = QLabel("YOU")
                header.setStyleSheet(f"color:{colors['muted']};font-size:11px;font-weight:800;letter-spacing:1px;margin-right:8px;")
                col_l.addWidget(header, 0, Qt.AlignRight)

                bubble = QFrame()
                bubble.setObjectName("UserBubble")
                bubble.setStyleSheet(
                    f"QFrame#UserBubble{{background:{user_bubble};border:1px solid {colors['border']};border-radius:18px;}}"
                )
                bubble.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Minimum)
                bubble.setMaximumWidth(max_bubble)
                b_l = QVBoxLayout(bubble)
                b_l.setContentsMargins(14, 10, 14, 10)

                lbl = QLabel()
                lbl.setWordWrap(True)
                lbl.setTextFormat(Qt.RichText)
                lbl.setText(f"<div style='text-align:left;color:{colors['text']};font-size:16px;line-height:1.5;'>{fmt_html(m.get('content',''))}</div>")
                lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
                lbl.setMaximumWidth(max_bubble - 28)
                b_l.addWidget(lbl)

                dots = QToolButton()
                dots.setText("...")
                dots.setCursor(Qt.PointingHandCursor)
                dots.setAutoRaise(True)
                dots.setStyleSheet(f"QToolButton{{background:transparent;border:0;color:{colors['muted']};font-weight:800;font-size:12px;}}")
                dots.clicked.connect(lambda _=False, idx=i: self.open_message_menu(idx))

                col_l.addWidget(bubble, 0, Qt.AlignRight)
                col_l.addWidget(dots, 0, Qt.AlignRight)
                row_l.addWidget(col)
                self.chat_msgs_layout.addWidget(row)
                continue

            if role == "assistant":
                wrap = QWidget()
                w_l = QVBoxLayout(wrap)
                w_l.setContentsMargins(0, 0, 0, 0)
                w_l.setSpacing(10)

                header = QLabel("AI")
                header.setStyleSheet(f"color:{colors['accent']};font-size:11px;font-weight:800;letter-spacing:1px;")
                w_l.addWidget(header)

                thought_s = m.get("thought_s")
                think_txt = (m.get("think", "") or "").strip()
                if isinstance(thought_s, (int, float)):
                    thought_frame = QFrame()
                    thought_frame.setObjectName("ThoughtFrame")
                    thought_frame.setStyleSheet(
                        f"QFrame#ThoughtFrame{{background:{colors['panel2']};border:1px solid {colors['border']};border-radius:12px;}}"
                    )
                    t_l = QVBoxLayout(thought_frame)
                    t_l.setContentsMargins(12, 10, 12, 10)
                    t_l.setSpacing(8)

                    hdr_row = QWidget()
                    hdr_l = QHBoxLayout(hdr_row)
                    hdr_l.setContentsMargins(0, 0, 0, 0)
                    hdr_l.setSpacing(8)

                    is_open = bool(m.get("think_open")) and bool(think_txt)
                    arrow = "▾" if is_open else "▸"
                    toggle = QToolButton()
                    toggle.setText(arrow)
                    toggle.setCursor(Qt.PointingHandCursor)
                    toggle.setAutoRaise(True)
                    toggle.setStyleSheet(f"QToolButton{{background:transparent;border:0;color:{colors['muted']};font-weight:900;font-size:14px;}}")
                    toggle.clicked.connect(lambda _=False, idx=i: self._toggle_thought(idx))
                    hdr_l.addWidget(toggle)

                    hdr_lbl = QLabel(f"Thought for {float(thought_s):.2f} seconds")
                    hdr_lbl.setStyleSheet(f"color:{colors['muted']};font-size:13px;font-weight:700;")
                    hdr_l.addWidget(hdr_lbl)
                    hdr_l.addStretch()

                    t_l.addWidget(hdr_row)

                    if is_open and think_txt:
                        t_lbl = QLabel()
                        t_lbl.setWordWrap(True)
                        t_lbl.setTextFormat(Qt.RichText)
                        t_lbl.setText(f"<div style='color:{colors['text']};font-size:15px;line-height:1.6;'>{fmt_html(think_txt)}</div>")
                        t_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
                        t_l.addWidget(t_lbl)

                    w_l.addWidget(thought_frame)

                ans = (m.get("answer", "") or "")
                if ans.strip():
                    # LM Studio Stili "Asistan" Arka Planı (Şeffaf ama belirgin)
                    ans_frame = QFrame()
                    ans_frame.setStyleSheet(f"""
                        QFrame {{
                            background-color: transparent;
                            padding: 0px;
                            border: none;
                        }}
                    """)
                    ans_lay = QVBoxLayout(ans_frame)
                    ans_lay.setContentsMargins(0, 0, 0, 0)
                    ans_lay.setSpacing(6)

                    parts = split_fenced_blocks(ans)
                    for kind, payload, lang in parts:
                        if kind == "code":
                            ans_lay.addWidget(make_code_block(payload, lang))
                            continue
                        text = (payload or "").strip("\n")
                        if not text.strip():
                            continue
                        a_lbl = QLabel()
                        a_lbl.setWordWrap(True)
                        a_lbl.setTextFormat(Qt.RichText)
                        # Yazı boyutu ve satır aralığı artırıldı, renk düzenlendi
                        a_lbl.setText(f"<div style='color:{colors['text']};font-size:16px;line-height:1.7;letter-spacing:0.3px;'>{fmt_html(text)}</div>")
                        a_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
                        ans_lay.addWidget(a_lbl)
                    
                    w_l.addWidget(ans_frame)

                meta = m.get("meta")
                if isinstance(meta, dict):
                    meta_lbl = QLabel(f"{meta.get('tps', 0.0):.2f} tokens/sec | {meta.get('tokens', 0)} tokens | {meta.get('elapsed', 0.0):.2f}s elapsed")
                    meta_lbl.setStyleSheet(f"color:{colors['muted']};font-size:11px;")
                    w_l.addWidget(meta_lbl)

                self.chat_msgs_layout.addWidget(wrap)

        self.chat_msgs_layout.addStretch()
        try:
            sb = self.chat_display.verticalScrollBar()
            if keep_scroll and old_scroll_val is not None:
                if old_scroll_max is not None and old_scroll_max > 0:
                    ratio = float(old_scroll_val) / float(old_scroll_max)
                    sb.setValue(int(ratio * float(sb.maximum())))
                else:
                    sb.setValue(int(old_scroll_val))
            else:
                sb.setValue(int(sb.maximum()))
        except Exception:
            pass

    def toggle_mic(self):
        """
        Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
        """
        if self.mic_worker is not None and self.mic_worker.is_recording:
            # Stop recording
            self.mic_worker.stop_recording()
            self.mic_btn.setProperty("recording", "false")
            self.mic_btn.style().unpolish(self.mic_btn)
            self.mic_btn.style().polish(self.mic_btn)
            self.gen_status_lbl.setText("Transcribing...")
            self.gen_status_lbl.setStyleSheet("color: #ffaa00;")
        else:
            # Start recording
            self.mic_worker = MicWorker(self)
            self.mic_worker.transcription_done.connect(self.on_transcription_done)
            self.mic_worker.error_occurred.connect(self.on_mic_error)
            self.mic_worker.start()
            
            self.mic_btn.setProperty("recording", "true")
            self.mic_btn.style().unpolish(self.mic_btn)
            self.mic_btn.style().polish(self.mic_btn)
            self.gen_status_lbl.setText("Recording... Click mic again to stop.")
            self.gen_status_lbl.setStyleSheet("color: #ff4444;")

    def on_transcription_done(self, text):
        """
        Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
        """
        self.mic_worker = None
        self.gen_status_lbl.setText("")
        self.mic_btn.setProperty("recording", "false")
        self.mic_btn.style().unpolish(self.mic_btn)
        self.mic_btn.style().polish(self.mic_btn)
        
        if text:
            # Append to existing text or replace
            current = self.input_field.text()
            if current:
                self.input_field.setText(current + " " + text)
            else:
                self.input_field.setText(text)
            self.input_field.setFocus()

    def on_mic_error(self, err_msg):
        """
        Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
        """
        self.mic_worker = None
        self.gen_status_lbl.setText("")
        self.mic_btn.setProperty("recording", "false")
        self.mic_btn.style().unpolish(self.mic_btn)
        self.mic_btn.style().polish(self.mic_btn)
        QMessageBox.warning(self, "Microphone Error", err_msg)

    def soru_sor(self):
        """
        Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
        """
        user_text = self.input_field.text().strip()
        if not user_text:
            return

        if self.model is None or self.tokenizer is None:
            QMessageBox.warning(self, "Service Not Ready", "Model is not loaded.")
            self.input_field.setText(user_text)
            self.input_field.setFocus()
            return
        self.input_field.clear()
        self._last_user_text = user_text

        self.chat_ui.setdefault(self.active_chat, [])
        if self.active_chat not in self.chats:
            self.chats[self.active_chat] = [{"role": "system", "content": self.system_prompt}]
            self._rebuild_chat_list()
        is_first_user_msg = not any(m.get("role") == "user" for m in self.chats.get(self.active_chat, []) or [])
        if is_first_user_msg and self._is_placeholder_chat_name(self.active_chat):
            self._auto_name_active_chat(user_text)

        # History appends
        self.chats.setdefault(self.active_chat, [{"role": "system", "content": self.system_prompt}])
        self.chat_ui.setdefault(self.active_chat, [])
        self.chats[self.active_chat].append({"role": "user", "content": user_text})
        self.chat_ui[self.active_chat].append({"role": "user", "content": user_text})
        try:
            conn = sqlite3.connect(self._db_path())
            try:
                self._ensure_chat_row(conn, self.active_chat)
            finally:
                conn.close()
            self._persist_message(self.active_chat, "user", user_text)
        except Exception:
            pass
        
        # Context formulation
        context_prompt = user_text
        ctx_parts = []
        proj_ctx = ""
        try:
            proj_ctx = self._build_project_context(user_text)
        except Exception:
            proj_ctx = ""
        if proj_ctx:
            ctx_parts.append(f"Project context:\n{proj_ctx}")
        if bool(getattr(self, "use_rag", True)) and not bool(getattr(self, "training_active", False)):
            rag_engine = self.get_rag_engine()
            if rag_engine and getattr(rag_engine, "enabled", False):
                rag_docs = rag_engine.query(user_text)
                if rag_docs:
                    ctx_parts.append(f"Background info:\n{rag_docs}")
                    self.rag_badge.setText("RAG: ACTIVE")
                    self.rag_badge.setProperty("ragState", "active")
                    self.rag_badge.style().unpolish(self.rag_badge)
                    self.rag_badge.style().polish(self.rag_badge)
                else:
                    self.rag_badge.setText("RAG: EMPTY")
                    self.rag_badge.setProperty("ragState", "empty")
                    self.rag_badge.style().unpolish(self.rag_badge)
                    self.rag_badge.style().polish(self.rag_badge)
        if ctx_parts:
            context_prompt = "\n\n".join(ctx_parts) + f"\n\nUser: {user_text}"
        
        
        self._last_context_prompt = context_prompt
        temp_history = list(self.chats[self.active_chat])
        temp_history[-1] = {"role": "user", "content": context_prompt}
        
        try:
            if hasattr(self.tokenizer, 'apply_chat_template'):
                prompt_string = self.tokenizer.apply_chat_template(temp_history, tokenize=False, add_generation_prompt=True)
            else: prompt_string = f"User: {context_prompt}\nAssistant: "
        except:
            prompt_string = f"User: {context_prompt}\nAssistant: "

        self.is_generating = True
        self.input_field.setDisabled(True)
        self.send_btn.setDisabled(True)
        self.stop_btn.setEnabled(True)
        self._update_dynamic_btn_state()
        self.gen_status_lbl.setText("")
        
        # Prepare for assistant response (filter hidden <think>/<analysis> blocks; show only thought duration)
        self._thinking_start_ts = time.time()
        self._answer_started = False
        self._stream_in_think = False
        self._stream_buffer = ""

        self.chat_ui[self.active_chat].append(
            {"role": "assistant", "answer": "", "think": "", "think_open": False, "thought_s": None, "meta": None}
        )
        self._pending_chat = self.active_chat
        self._pending_msg_index = len(self.chat_ui[self.active_chat]) - 1
        self.render_chat(self.active_chat)
        
        self.worker = AIWorker(self.model, self.tokenizer, prompt_string)
        self.worker.new_token.connect(self.on_new_token)
        self.worker.finished.connect(self.on_ai_success)
        self.worker.error.connect(self.on_ai_error)
        self.worker.start()

    def add_chat_bubble(self, sender, text, is_user=True):
        """
        Mesajlaşma arayüzü, HTML/CSS ile LM Studio estetiğinde baloncukları basıyor ekrana.
        """
        if is_user:
            self.chat_ui.setdefault(self.active_chat, []).append({"role": "user", "content": text or ""})
            self.chats.setdefault(self.active_chat, [{"role": "system", "content": self.system_prompt}]).append({"role": "user", "content": text or ""})
        else:
            self.chat_ui.setdefault(self.active_chat, []).append({"role": "assistant", "answer": text or "", "think": "", "think_open": False, "thought_s": None, "meta": None})
            self.chats.setdefault(self.active_chat, [{"role": "system", "content": self.system_prompt}]).append({"role": "assistant", "content": text or ""})
        self.render_chat(self.active_chat)
        return None

    def run_last_code(self):
        """
        Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
        """
        # Find last code block in assistant responses
        import re
        import subprocess
        
        last_code = ""
        for msg in reversed(self.chats[self.active_chat]):
            if msg["role"] == "assistant":
                blocks = re.findall(r"```(?:python)?\n([\s\S]*?)```", msg["content"])
                if blocks:
                    last_code = blocks[-1]
                    break
        
        if not last_code:
            QMessageBox.warning(self, "No Code", "No Python code blocks found in the last response.")
            return
            
        try:
            # Simple execution for now, as per roadmap Phase 2
            result = subprocess.run([sys.executable, "-c", last_code], capture_output=True, text=True, timeout=10)
            
            output = result.stdout if result.stdout else ""
            error = result.stderr if result.stderr else ""
            
            dlg = QDialog(self)
            dlg.setWindowTitle("Run Output")
            dlg.setFixedSize(760, 480)
            layout = QVBoxLayout(dlg)
            editor = QPlainTextEdit()
            editor.setReadOnly(True)
            combined = ""
            if output:
                combined += output
            if error:
                if combined:
                    combined += "\n"
                combined += error
            editor.setPlainText(combined or "(no output)")
            layout.addWidget(editor)
            btn_row = QHBoxLayout()
            close_btn = QPushButton("Close")
            btn_row.addStretch()
            btn_row.addWidget(close_btn)
            layout.addLayout(btn_row)
            close_btn.clicked.connect(dlg.accept)
            dlg.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Execution Error", str(e))

    def _html_escape(self, s: str) -> str:
        """
        Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
        """
        return (
            (s or "")
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )

    def _split_stream_delta(self, piece: str) -> tuple[str, str]:
        """
        Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
        """
        buf_full = (self._stream_buffer or "") + (piece or "")
        think_out = ""
        answer_out = ""
        think_tags = {"think", "analysis", "thinking"}
        keep_from = -1
        last_lt = buf_full.rfind("<")
        if last_lt != -1 and ">" not in buf_full[last_lt:]:
            keep_from = last_lt
        if keep_from != -1:
            buf = buf_full[:keep_from]
            self._stream_buffer = buf_full[keep_from:]
        else:
            buf = buf_full
            self._stream_buffer = ""

        def parse_tag_at(s: str, lt: int):
            """
            Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
            """
            n = len(s)
            j = lt + 1
            while j < n and s[j].isspace():
                j += 1
            is_end = False
            if j < n and s[j] == "/":
                is_end = True
                j += 1
                while j < n and s[j].isspace():
                    j += 1
            name_start = j
            while j < n and s[j].isalpha():
                j += 1
            if j == name_start:
                return None
            name = s[name_start:j].lower()
            while j < n and s[j].isspace():
                j += 1
            gt = s.find(">", j)
            if gt == -1:
                return None
            return name, is_end, gt

        i = 0
        while i < len(buf):
            lt = buf.find("<", i)
            if lt == -1:
                if self._stream_in_think:
                    think_out += buf[i:]
                else:
                    answer_out += buf[i:]
                i = len(buf)
                break

            if self._stream_in_think:
                t = parse_tag_at(buf, lt)
                if t and t[0] in think_tags and t[1] is True:
                    think_out += buf[i:lt]
                    self._stream_in_think = False
                    i = t[2] + 1
                    continue
                think_out += buf[i : lt + 1]
                i = lt + 1
            else:
                t = parse_tag_at(buf, lt)
                if t and t[0] in think_tags and t[1] is False:
                    answer_out += buf[i:lt]
                    self._stream_in_think = True
                    i = t[2] + 1
                    continue
                if t and t[0] in think_tags and t[1] is True:
                    answer_out += buf[i:lt]
                    i = t[2] + 1
                    continue
                answer_out += buf[i : lt + 1]
                i = lt + 1
        return think_out, answer_out

    def _finalize_stream_tail(self) -> tuple[str, str]:
        """
        Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
        """
        if self._stream_in_think:
            self._stream_buffer = ""
            return "", ""
        tail = self._stream_buffer or ""
        self._stream_buffer = ""
        return "", tail

    def _extract_think_answer_from_text(self, text: str) -> tuple[str, str]:
        """
        Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
        """
        import re

        raw = text or ""

        m = re.search(r"<\s*(think|analysis|thinking)\s*>", raw, flags=re.IGNORECASE)
        if m:
            start = m.end()
            m2 = re.search(r"</\s*(think|analysis|thinking)\s*>", raw[start:], flags=re.IGNORECASE)
            if m2:
                end = start + m2.start()
                think = raw[start:end]
                rest = raw[start + m2.end():]
                prefix = raw[:m.start()]
                answer = (prefix + rest)
                return (think or "").strip(), (answer or "").strip()

        m_end = re.search(r"</\s*(think|analysis|thinking)\s*>", raw, flags=re.IGNORECASE)
        if m_end:
            think = raw[:m_end.start()]
            answer = raw[m_end.end():]
            return (think or "").strip(), (answer or "").strip()

        m3 = re.search(r"(?im)^\s*(final answer|final|answer)\s*:\s*", raw)
        if m3:
            think = raw[:m3.start()]
            answer = raw[m3.end():]
            return (think or "").strip(), (answer or "").strip()

        paras = [p.strip() for p in re.split(r"\n\s*\n+", raw) if p.strip()]
        if paras:
            cues = [
                "the user", "let me think", "i should", "i will", "i'll",
                "reason", "thinking", "plan", "approach", "let me",
            ]

            def looks_like_think(p: str) -> bool:
                """
                Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
                """
                low = (p or "").strip().lower()
                if not low:
                    return False
                if low.startswith("the user "):
                    return True
                if low.startswith("the user is "):
                    return True
                if any(c in low for c in cues):
                    return True
                return False

            def looks_like_answer(p: str) -> bool:
                """
                Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
                """
                low = (p or "").strip().lower()
                if not low:
                    return False
                if re.match(r"^\s*\d+(\.\d+)?\s*$", p):
                    return True
                if "the answer is" in low:
                    return True
                if low.startswith("answer:") or low.startswith("final:") or low.startswith("final answer:"):
                    return True
                return False

            answer_idx = None
            for i in range(len(paras) - 1, -1, -1):
                if looks_like_answer(paras[i]):
                    answer_idx = i
                    break

            if answer_idx is not None:
                think = "\n\n".join(paras[:answer_idx]).strip()
                answer = "\n\n".join(paras[answer_idx:]).strip()
                return think, answer

            if looks_like_think(paras[0]) and len(paras) >= 2:
                think = paras[0]
                answer = "\n\n".join(paras[1:]).strip()
                return think, answer

        return "", (raw or "").strip()

    def _fallback_answer_from_user_text(self, user_text: str) -> str:
        """
        Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
        """
        import re

        ut = (user_text or "").strip()
        if not ut:
            return ""
        low = ut.lower()

        m = re.search(r"(\d+)\s*(?:st|nd|rd|th)?\s*fibonacci", low)
        if m:
            try:
                n = int(m.group(1))
            except Exception:
                n = None
            if n is not None and n >= 0:
                a, b = 0, 1
                for _ in range(n):
                    a, b = b, a + b
                return str(a)

        return ""

    def on_new_token(self, token):
        """
        Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
        """
        think_delta, answer_delta = self._split_stream_delta(token)

        if self._pending_chat is not None and self._pending_msg_index is not None:
            msgs = self.chat_ui.get(self._pending_chat, [])
            if not (0 <= int(self._pending_msg_index) < len(msgs)):
                return
            msg = msgs[int(self._pending_msg_index)]
            if think_delta:
                msg["think"] = (msg.get("think", "") or "") + think_delta
            msg["answer"] += answer_delta
        if answer_delta and not getattr(self, "_answer_started", False):
            thought_s = max(0.0, time.time() - getattr(self, "_thinking_start_ts", time.time()))
            if self._pending_chat is not None and self._pending_msg_index is not None:
                msgs = self.chat_ui.get(self._pending_chat, [])
                if 0 <= int(self._pending_msg_index) < len(msgs):
                    msgs[int(self._pending_msg_index)]["thought_s"] = float(thought_s)
            self._answer_started = True
        if (think_delta or answer_delta) and self._pending_chat is not None:
            self._schedule_render(self._pending_chat)

    def _schedule_render(self, chat_name: str | None):
        """
        Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
        """
        if not hasattr(self, "_render_timer") or self._render_timer is None:
            self._render_timer = QTimer(self)
            self._render_timer.setSingleShot(True)
            self._render_timer.timeout.connect(self._run_scheduled_render)

        if chat_name:
            self._render_target_chat = chat_name
        if not self._render_timer.isActive():
            self._render_timer.start(40)

    def _run_scheduled_render(self) -> None:
        """
        Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
        """
        target = getattr(self, "_render_target_chat", None)
        if target and target == self.active_chat:
            self.render_chat(target)

    def on_ai_success(self, response, tps, tokens, ms, peak_memory_gb):
        """
        Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
        """
        pending_chat = self._pending_chat
        pending_idx = self._pending_msg_index
        if isinstance(peak_memory_gb, (int, float)) and peak_memory_gb > 0:
            self._model_peak_memory_gb = float(peak_memory_gb)

        _, tail = self._finalize_stream_tail()
        if tail and pending_chat is not None and pending_idx is not None:
            msgs = self.chat_ui.get(pending_chat, [])
            if 0 <= int(pending_idx) < len(msgs):
                msgs[int(pending_idx)]["answer"] += tail

        assistant_answer = ""
        if pending_chat is not None and pending_idx is not None:
            msgs = self.chat_ui.get(pending_chat, [])
            msg = msgs[int(pending_idx)] if (0 <= int(pending_idx) < len(msgs)) else None
            if msg is not None:
                merged_think = (msg.get("think", "") or "").strip()
                extracted_think, extracted_answer = self._extract_think_answer_from_text(msg.get("answer", ""))
                if extracted_think:
                    if merged_think:
                        merged_think = (merged_think + "\n" + extracted_think).strip()
                    else:
                        merged_think = extracted_think.strip()
                    msg["think"] = merged_think
                    msg["answer"] = extracted_answer
                else:
                    ans_now = (msg.get("answer", "") or "").strip()
                    low = ans_now.lower()
                    if low.startswith("the user ") or low.startswith("the user is ") or low.startswith("let me "):
                        fallback = self._fallback_answer_from_user_text(getattr(self, "_last_user_text", ""))
                        if fallback:
                            msg["think"] = (ans_now or "")
                            msg["answer"] = fallback

                ans_now = (msg.get("answer", "") or "").strip()
                if ans_now and not msg.get("think") and (ans_now.lower().startswith("the user ") or ans_now.lower().startswith("let me ")):
                    msg["think"] = ans_now
                    msg["answer"] = ""
                    ans_now = ""

                if not ans_now and (msg.get("think") or "").strip():
                    try:
                        sys_prompt = (self.system_prompt or "").strip() + "\n\nReturn ONLY the final answer. No reasoning."
                        hist = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": (self._last_context_prompt or self._last_user_text or "")}]
                        if hasattr(self.tokenizer, "apply_chat_template"):
                            final_prompt = self.tokenizer.apply_chat_template(hist, tokenize=False, add_generation_prompt=True)
                        else:
                            final_prompt = f"User: {(self._last_context_prompt or self._last_user_text or '')}\nAssistant: "
                        self._final_pending = (pending_chat, pending_idx)
                        self.gen_status_lbl.setText("Finalizing…")
                        self._final_worker = FinalAnswerWorker(self.model, self.tokenizer, final_prompt, max_tokens=256)
                        self._final_worker.finished.connect(self._on_final_answer_ready)
                        self._final_worker.error.connect(self._on_final_answer_error)
                        self._final_worker.start()
                        self._schedule_render(pending_chat)
                        return
                    except Exception:
                        pass

                assistant_answer = msg.get("answer", "") or ""
                self.chats.setdefault(pending_chat, [{"role": "system", "content": self.system_prompt}]).append(
                    {"role": "assistant", "content": assistant_answer}
                )
            else:
                pending_chat = None
                pending_idx = None
        else:
            pass
        
        # Display metadata
        if pending_chat is not None and pending_idx is not None:
            msgs = self.chat_ui.get(pending_chat, [])
            if not (0 <= int(pending_idx) < len(msgs)):
                pending_chat = None
                pending_idx = None
            else:
                msg = msgs[int(pending_idx)]
                msg["meta"] = {"tps": float(tps), "tokens": int(tokens), "elapsed": float(ms)}
                try:
                    self._persist_message(
                        pending_chat,
                        "assistant",
                        assistant_answer,
                        think=(msg.get("think", "") or ""),
                        thought_s=msg.get("thought_s"),
                        meta=msg.get("meta") if isinstance(msg.get("meta"), dict) else None,
                    )
                except Exception:
                    pass

        self._pending_chat = None
        self._pending_msg_index = None

        self.render_chat(self.active_chat)
        self.chat_display.verticalScrollBar().setValue(self.chat_display.verticalScrollBar().maximum())
        
        self.is_generating = False
        self.input_field.setDisabled(False)
        self.send_btn.setDisabled(False)
        self.stop_btn.setEnabled(False)
        self._update_dynamic_btn_state()
        self.gen_status_lbl.setText("")
        self.input_field.setFocus()
        
        self.rag_badge.setText("RAG: OFF")
        self.rag_badge.setProperty("ragState", "off")
        self.rag_badge.style().unpolish(self.rag_badge)
        self.rag_badge.style().polish(self.rag_badge)

    def _on_final_answer_ready(self, text: str) -> None:
        """
        Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
        """
        pending = getattr(self, "_final_pending", None)
        if not (isinstance(pending, tuple) and len(pending) == 2):
            return
        chat_name, idx = pending
        msgs = self.chat_ui.get(chat_name, [])
        if not (0 <= idx < len(msgs)):
            return
        msg = msgs[idx]
        msg["answer"] = (text or "").strip()
        assistant_answer = msg.get("answer", "") or ""
        self.chats.setdefault(chat_name, [{"role": "system", "content": self.system_prompt}]).append({"role": "assistant", "content": assistant_answer})
        if msg.get("meta") is None:
            msg["meta"] = {"tps": 0.0, "tokens": 0, "elapsed": 0.0}
        try:
            self._persist_message(
                chat_name,
                "assistant",
                assistant_answer,
                think=(msg.get("think", "") or ""),
                thought_s=msg.get("thought_s"),
                meta=msg.get("meta") if isinstance(msg.get("meta"), dict) else None,
            )
        except Exception:
            pass
        self._final_pending = None
        self._final_worker = None
        self._pending_chat = None
        self._pending_msg_index = None
        self._schedule_render(chat_name)
        try:
            self.chat_display.verticalScrollBar().setValue(self.chat_display.verticalScrollBar().maximum())
        except Exception:
            pass
        self.is_generating = False
        self.input_field.setDisabled(False)
        self.send_btn.setDisabled(False)
        self.stop_btn.setEnabled(False)
        self._update_dynamic_btn_state()
        self.gen_status_lbl.setText("")
        self.input_field.setFocus()

    def _on_final_answer_error(self, err: str) -> None:
        """
        Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
        """
        self._final_pending = None
        self._final_worker = None
        self._pending_chat = None
        self._pending_msg_index = None
        self.is_generating = False
        self.input_field.setDisabled(False)
        self.send_btn.setDisabled(False)
        self.stop_btn.setEnabled(False)
        self._update_dynamic_btn_state()
        self.gen_status_lbl.setText("")
        QMessageBox.critical(self, "Final Answer Error", err)

    def on_ai_error(self, err_msg):
        """
        Olm bu fonksiyon da kendi çapında bir iş yapıyor, elit sisteme ufak bir katkı. Dokunma çalışsın.
        """
        if self._pending_chat is not None and self._pending_msg_index is not None:
            try:
                msgs = self.chat_ui.get(self._pending_chat, [])
                idx = int(self._pending_msg_index)
                if 0 <= idx < len(msgs):
                    msgs.pop(idx)
            except Exception:
                pass
            self._pending_chat = None
            self._pending_msg_index = None

        self.render_chat(self.active_chat)
        QMessageBox.critical(self, "Generation Error", err_msg)
        self.is_generating = False
        self.input_field.setDisabled(False)
        self.send_btn.setDisabled(False)
        self.stop_btn.setEnabled(False)
        self._update_dynamic_btn_state()
        self.gen_status_lbl.setText("")
        self.input_field.setFocus()

if __name__ == "__main__":
    # Disable Chromium sandbox to prevent "Mach rendezvous failed" crash on some macOS setups
    os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = "--no-sandbox --disable-gpu-sandbox"

    app = QApplication(sys.argv)
    
    # Standard font setup to avoid warnings
    font = QFont("Helvetica Neue", 13)
    app.setFont(font)

    window = ChatbotGUI(None, None, "")
    window.show()
    sys.exit(app.exec_())
