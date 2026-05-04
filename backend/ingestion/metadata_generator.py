"""Metadata generation utilities for ingestion."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from urllib.parse import urlparse

from backend.utils.logger import get_logger

logger = get_logger(__name__)


class MetadataGenerator:
    """Generate structured metadata from markdown content."""

    def generate_metadata(
        self,
        source_url: str,
        page_title: str,
        markdown_content: str,
    ) -> dict:
        """Build metadata payload for ingestion and future retrieval."""
        logger.info(
            "metadata_generator.start source_url=%s markdown_length=%s",
            source_url,
            len(markdown_content),
        )
        heading_lines = self._extract_headings(markdown_content=markdown_content)
        word_count = self._count_words(markdown_content=markdown_content)
        domain = urlparse(source_url).netloc
        metadata = {
            "source_url": source_url,
            "source_domain": domain,
            "title": page_title,
            "word_count": word_count,
            "headings": heading_lines,
            "summary": self._build_summary(markdown_content=markdown_content),
            "ingested_at_utc": datetime.now(UTC).isoformat(),
            "tags": self._generate_tags(
                source_url=source_url,
                page_title=page_title,
                headings=heading_lines,
            ),
        }
        logger.info(
            "metadata_generator.done source_url=%s word_count=%s headings=%s tags=%s",
            source_url,
            metadata["word_count"],
            len(metadata["headings"]),
            len(metadata["tags"]),
        )
        return metadata

    @staticmethod
    def _extract_headings(markdown_content: str) -> list[str]:
        heading_lines: list[str] = []
        for line in markdown_content.splitlines():
            if line.startswith("#"):
                heading_lines.append(line.lstrip("# ").strip())
        return heading_lines[:20]

    @staticmethod
    def _count_words(markdown_content: str) -> int:
        word_matches = re.findall(r"\b\w+\b", markdown_content)
        return len(word_matches)

    @staticmethod
    def _build_summary(markdown_content: str) -> str:
        lines = [line.strip() for line in markdown_content.splitlines() if line.strip()]
        paragraph_candidates = [line for line in lines if not line.startswith("#")]
        if not paragraph_candidates:
            return ""
        first_paragraph = paragraph_candidates[0]
        return first_paragraph[:280]

    @staticmethod
    def _generate_tags(source_url: str, page_title: str, headings: list[str]) -> list[str]:
        seed_text = " ".join([source_url, page_title, *headings]).lower()
        candidate_keywords = [
            "api",
            "authentication",
            "oauth",
            "guide",
            "quickstart",
            "reference",
            "sdk",
            "webhook",
        ]
        generated_tags = [tag for tag in candidate_keywords if tag in seed_text]
        return generated_tags[:8]
