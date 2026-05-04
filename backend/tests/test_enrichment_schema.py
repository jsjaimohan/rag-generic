"""Tests for Step 6 enrichment schema normalization."""

from __future__ import annotations

import unittest

from backend.ingestion.enrichment_schema import EnrichmentPayload


class TestEnrichmentPayload(unittest.TestCase):
    def test_tags_capped(self) -> None:
        raw = {"tags": [f"t{i}" for i in range(20)]}
        payload = EnrichmentPayload.model_validate(raw)
        self.assertLessEqual(len(payload.tags), 12)

    def test_entities_skips_empty_names(self) -> None:
        raw = {"entities": [{"name": "  ", "type": "x"}, {"name": "OK", "type": "org"}]}
        payload = EnrichmentPayload.model_validate(raw)
        self.assertEqual(len(payload.entities), 1)
        self.assertEqual(payload.entities[0].name, "OK")


if __name__ == "__main__":
    unittest.main()
