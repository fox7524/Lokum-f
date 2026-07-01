import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.query_service import build_context_block


class TestQueryService(unittest.TestCase):
    def test_build_context_block_joins_chunks_with_separator(self):
        block = build_context_block(["one", "two", "three"])
        self.assertIn("one", block)
        self.assertIn("\n---\n", block)
        self.assertTrue(block.endswith("three"))

    def test_build_context_block_skips_empty_chunks(self):
        block = build_context_block(["one", "", "  ", "two"])
        self.assertEqual(block, "one\n\n---\n\ntwo")
