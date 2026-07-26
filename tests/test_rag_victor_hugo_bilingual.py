import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import rag_engine


class _StubEmbedder:
    def encode(self, texts, batch_size=32, show_progress_bar=False):
        return np.ones((len(texts), 3), dtype="float32")


class _StubIndex:
    def __init__(self, dim: int):
        self.d = dim
        self.ntotal = 0
        self._rows: list[np.ndarray] = []

    def add(self, emb: np.ndarray) -> None:
        for row in emb:
            self._rows.append(row)
        self.ntotal = len(self._rows)

    def search(self, vec: np.ndarray, k: int):
        count = min(k, self.ntotal)
        distances = np.array([[0.99 - (i * 0.01) for i in range(count)]], dtype="float32")
        indices = np.array([[i for i in range(count)]], dtype="int64")
        return distances, indices


class _StubFaiss:
    @staticmethod
    def normalize_L2(arr: np.ndarray) -> None:
        return None

    @staticmethod
    def IndexFlatIP(dim: int) -> _StubIndex:
        return _StubIndex(dim)


def _make_engine(tmp_path: Path, monkeypatch) -> rag_engine.RAGEngine:
    monkeypatch.setattr(rag_engine, "faiss", _StubFaiss)
    eng = rag_engine.RAGEngine.__new__(rag_engine.RAGEngine)
    eng.enabled = True
    eng.embedding_model = _StubEmbedder()
    eng.index = None
    eng.documents = []
    eng.chunk_meta = []
    eng.state = {"version": 1, "files": {}}
    eng.last_error = ""
    eng._abort = False
    eng.storage_dir = str(tmp_path / "ragstore")
    eng.index_path = str(tmp_path / "ragstore" / "faiss_index.bin")
    eng.docs_path = str(tmp_path / "ragstore" / "docs_metadata.npy")
    eng.meta_path = str(tmp_path / "ragstore" / "rag_meta.json")
    eng.chunks_meta_path = str(tmp_path / "ragstore" / "chunks_meta.npy")
    eng.state_path = str(tmp_path / "ragstore" / "rag_state.json")
    eng.staging_dir = str(tmp_path / "ragstore" / "staging")
    Path(eng.staging_dir).mkdir(parents=True, exist_ok=True)
    eng.embed_batch_size = 4

    eng._set_last_error = rag_engine.RAGEngine._set_last_error.__get__(eng, rag_engine.RAGEngine)
    eng._check_abort = rag_engine.RAGEngine._check_abort.__get__(eng, rag_engine.RAGEngine)
    eng._file_id_for = rag_engine.RAGEngine._file_id_for.__get__(eng, rag_engine.RAGEngine)
    eng._load_jsonl_chunks = rag_engine.RAGEngine._load_jsonl_chunks.__get__(eng, rag_engine.RAGEngine)
    eng._extract_content = rag_engine.RAGEngine._extract_content.__get__(eng, rag_engine.RAGEngine)
    eng._is_chunk_active = rag_engine.RAGEngine._is_chunk_active.__get__(eng, rag_engine.RAGEngine)
    eng.query_with_sources = rag_engine.RAGEngine.query_with_sources.__get__(eng, rag_engine.RAGEngine)
    eng._load_state = lambda: None
    eng._validate_or_quarantine_existing_store = lambda: None
    eng._checkpoint_policy = lambda: (5000, 30.0)
    eng.save_index = lambda: None
    eng.compact_index = lambda: 0
    return eng


def test_rag_ingests_prebuilt_victor_hugo_jsonl_with_metadata(tmp_path: Path, monkeypatch) -> None:
    eng = _make_engine(tmp_path, monkeypatch)
    jsonl_path = tmp_path / "victor_hugo_rag_chunks.jsonl"
    rows = [
        {
            "chunk_id": "les_miserables_en__0001",
            "doc_path": "sources/raw_texts/les_miserables_en.txt",
            "doc_title": "Les Misérables",
            "category": "sources",
            "language_guess": "en",
            "text": "Les Misérables is a novel about poverty, justice, and mercy.",
        },
        {
            "chunk_id": "bridges__0001",
            "doc_path": "metadata/victor_hugo_tr_en_title_aliases.md",
            "doc_title": "Victor Hugo title aliases",
            "category": "metadata",
            "language_guess": "tr",
            "text": "Sefiller, Les Misérables eserinin Türkçede yaygın adıdır.",
        },
    ]
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    assert rag_engine.RAGEngine.ingest_documents(eng, [str(jsonl_path)]) is True
    assert len(eng.documents) == 2
    assert eng.chunk_meta[0]["doc_title"] == "Les Misérables"
    assert eng.chunk_meta[1]["doc_path"] == "metadata/victor_hugo_tr_en_title_aliases.md"

    result = rag_engine.RAGEngine.query_with_sources(eng, "Sefiller ne anlatır?", k=2)
    assert result["count"] == 2
    assert result["sources"][0]["doc_title"] == "Les Misérables"
    assert result["sources"][1]["language_guess"] == "tr"
