"""HTML cleaning utilities for ingestion."""

from __future__ import annotations

from bs4 import BeautifulSoup

from backend.utils.logger import get_logger

logger = get_logger(__name__)


class HtmlCleaner:
    """Clean raw HTML and keep high-signal content."""

    def clean_html_content(self, raw_html: str) -> str:
        """Remove low-signal tags and normalize whitespace."""
        logger.info("html_cleaner.start raw_length=%s", len(raw_html))
        soup = BeautifulSoup(raw_html, "html.parser")
        self._remove_low_signal_nodes(soup=soup)
        self._unwrap_non_semantic_tags(soup=soup)
        cleaned_html = str(soup)
        normalized_html = self._normalize_whitespace(html_content=cleaned_html)
        logger.info("html_cleaner.done cleaned_length=%s", len(normalized_html))
        return normalized_html

    @staticmethod
    def _remove_low_signal_nodes(soup: BeautifulSoup) -> None:
        removable_tags = [
            "script",
            "style",
            "noscript",
            "iframe",
            "svg",
            "canvas",
            "form",
            "footer",
            "nav",
        ]
        for removable_tag in removable_tags:
            for node in soup.find_all(removable_tag):
                node.decompose()

    @staticmethod
    def _unwrap_non_semantic_tags(soup: BeautifulSoup) -> None:
        for node in soup.find_all(["span", "font"]):
            node.unwrap()

    @staticmethod
    def _normalize_whitespace(html_content: str) -> str:
        normalized_lines = [line.rstrip() for line in html_content.splitlines()]
        return "\n".join(normalized_lines).strip()
