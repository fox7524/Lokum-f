# Lokum-F (Lokum Fine-Tuning & RAG Studio)

Lokum-F is an advanced, standalone desktop application built for Apple Silicon (M-Series) Macs. It provides a comprehensive GUI for local Large Language Model (LLM) fine-tuning, Retrieval-Augmented Generation (RAG) indexing, and model evaluation—all leveraging the native power of the Apple MLX framework.

## 🚀 Key Features

*   **Local LoRA Fine-Tuning (MLX):** Effortlessly train and adapt large language models entirely on-device using Apple's MLX architecture. Configure advanced parameters (Rank, Alpha, Batch Size, Layers) via an intuitive UI without writing any CLI commands.
*   **Intelligent Adapter Management:** Automatically tracks and organizes fine-tuned adapters with deterministic hashing (`R16_A32_B1_L16_A7B2C`) for pristine developer experience (DX).
*   **One-Click Model Fusion:** Merge your trained LoRA adapters into base models seamlessly with a single click. Includes auto-cleanup to delete heavy adapter artifacts post-fusion, saving valuable disk space.
*   **Smart RAG Engine (FAISS):** Build robust contextual memories for your models. The engine supports dynamic re-indexing, chunk compaction, and active garbage collection to prevent index bloat and ensure fast retrieval.
*   **Developer-Friendly Diagnostics:** Real-time MLX log parsing that filters out Apple Metal API spam ("God-Mode" limits) and provides actionable, clear hints for common hardware errors (e.g., Layer mismatch, Out-Of-Memory exceptions).
*   **Speech-to-Text Integration:** Built-in `mlx-whisper` integration for instant audio transcription directly within the chat interface.

## 🧠 Core Architecture

*   **Apple Silicon Native:** Fully optimized for M-series unified memory architecture.
*   **PyQt6 GUI:** A modern, responsive desktop interface with built-in presets (Ultra, Good, Mid, Low, Custom) scaling to your available VRAM.
*   **Vector Database:** FAISS-based fast semantic search using optimized embeddings.

## 🛠 Installation & Usage

1. **Create a virtual environment:**
   ```bash
   python3 -m venv .venv
   ```
2. **Activate the environment:**
   ```bash
   source .venv/bin/activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt mlx-whisper sounddevice scipy numpy
   ```
4. **Run the application:**
   ```bash
   python3 main.py
   ```

## ⚙️ Advanced Fine-Tuning Guidelines

When using the "Custom" preset, remember the Golden Rule for Apple Silicon memory stability:
*   **Rank (r):** Defines the learning capacity. Keep it an even power of 2 (8, 16, 32).
*   **Alpha:** The scaling factor. Should be `2 * Rank` or at least equal to Rank.
*   **Train Layers:** Do not exceed your model's physical layer count. Reducing this (e.g., to 16 or 8) drastically reduces VRAM consumption.

---
*Built for local AI development on macOS.*