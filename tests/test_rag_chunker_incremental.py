import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import file_ingest
import rag_engine
from rag.chunker import TextChunk, chunk_signature, chunk_text
from rag.normalize import normalize_text
from rag.state_store import build_file_state


class TestSharedRagChunker(unittest.TestCase):
    def test_normalize_text_collapses_whitespace(self):
        raw = " alpha\t beta\r\n\r\ngamma \r delta \n\n\n epsilon "
        self.assertEqual(normalize_text(raw), "alpha beta\n\ngamma\ndelta\n\nepsilon")

    def test_chunk_signature_is_stable_for_same_text(self):
        self.assertEqual(chunk_signature("same text"), chunk_signature("same text"))

    def test_chunk_text_preserves_content_order(self):
        text = "alpha beta gamma delta epsilon zeta eta theta"
        chunks = chunk_text(text, chunk_size=18, overlap=4)
        self.assertTrue(chunks)
        self.assertTrue(all(isinstance(chunk, TextChunk) for chunk in chunks))
        self.assertTrue(chunks[0].text.startswith("alpha"))
        self.assertTrue(chunks[-1].text.endswith("theta"))

    def test_chunk_text_rejects_invalid_overlap(self):
        with self.assertRaisesRegex(ValueError, "overlap"):
            chunk_text("abc", chunk_size=4, overlap=4)

    def test_build_file_state_contains_chunk_signatures(self):
        state = build_file_state(
            source_path="/tmp/a.txt",
            raw_text="alpha beta gamma",
            chunk_size=8,
            overlap=2,
        )
        self.assertEqual(state["source_path"], "/tmp/a.txt")
        self.assertGreaterEqual(state["chunk_count"], 1)
        self.assertEqual(len(state["chunk_signatures"]), state["chunk_count"])
        self.assertEqual(state["file_signature"], chunk_signature("alpha beta gamma"))


class TestLegacyCallersUseSharedNormalization(unittest.TestCase):
    def test_file_ingest_chunk_text_returns_strings(self):
        chunks = file_ingest.chunk_text(" alpha\t beta ", chunk_size=50, overlap=0)
        self.assertEqual(chunks, ["alpha beta"])

    def test_rag_engine_chunk_text_returns_strings(self):
        eng = rag_engine.RAGEngine.__new__(rag_engine.RAGEngine)
        chunks = eng.chunk_text(" alpha\t beta ", chunk_size=50, overlap=0)
        self.assertEqual(chunks, ["alpha beta"])
