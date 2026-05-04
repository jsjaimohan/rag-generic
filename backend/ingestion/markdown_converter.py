"""Markdown conversion utilities for ingestion."""

from __future__ import annotations

import re

from markdownify import markdownify as html_to_markdown

from backend.utils.logger import get_logger

logger = get_logger(__name__)


class MarkdownConverter:
    """Convert cleaned HTML content to markdown."""

    def convert_html_to_markdown(self, cleaned_html: str) -> str:
        """Generate markdown while preserving headings, lists, and links."""
        logger.info("markdown_converter.start cleaned_html_length=%s", len(cleaned_html))
        markdown_content = html_to_markdown(
            cleaned_html,
            heading_style="ATX",
            bullets="-",
            strip=["img"],
        )
        normalized_markdown = self._normalize_markdown(markdown_content=markdown_content)
        logger.info("markdown_converter.done markdown_length=%s", len(normalized_markdown))
        return normalized_markdown

    @staticmethod
    def _normalize_markdown(markdown_content: str) -> str:
        normalized_lines = []
        for line in markdown_content.splitlines():
            collapsed_spaces = re.sub(r"[ \t]+", " ", line).rstrip()
            normalized_lines.append(collapsed_spaces)

        normalized_text = "\n".join(normalized_lines).strip()
        # Keep readable paragraph spacing while preventing noisy vertical whitespace.
        normalized_text = re.sub(r"\n{3,}", "\n\n", normalized_text)
        return f"{normalized_text}\n" if normalized_text else ""
