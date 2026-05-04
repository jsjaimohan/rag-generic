"""Tests for Step 7 semantic chunking helpers."""

from __future__ import annotations

import unittest

from backend.ingestion.semantic_chunker import pack_segments, segment_section_lines, split_by_headings


class TestSemanticChunker(unittest.TestCase):
    def test_split_by_headings(self) -> None:
        md = "# A\n\nx\n\n## B\n\ny"
        sections = split_by_headings(md)
        self.assertGreaterEqual(len(sections), 2)

    def test_segment_preserves_fence(self) -> None:
        lines = ["Intro", "", "```py", "x=1", "```", "", "Outro"]
        segs = segment_section_lines(lines)
        self.assertTrue(any("```" in s for s in segs))

    def test_pack_respects_max(self) -> None:
        segs = ["a" * 100, "b" * 100, "c" * 100]
        chunks = pack_segments(segs, max_chars=150, overlap_chars=20)
        self.assertGreaterEqual(len(chunks), 2)
        self.assertTrue(all(len(c) <= 155 for c in chunks))


if __name__ == "__main__":
    unittest.main()
