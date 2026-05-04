"""Crawler implementation for GenericRAG."""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse

import httpx

from backend.settings import settings


@dataclass
class CrawlResult:
    """Normalized crawl output returned by the crawler service."""

    url: str
    status_code: int
    title: str
    html: str


class CrawlerService:
    """Small HTTP crawler for initial foundation phase."""

    def crawl_url(self, url: str) -> CrawlResult:
        """Fetch one URL and return a normalized payload."""
        self._validate_url(url=url)
        response = httpx.get(
            url,
            timeout=settings.crawler_timeout_seconds,
            headers={"User-Agent": settings.crawler_user_agent},
            follow_redirects=True,
        )
        response.raise_for_status()
        html_content = response.text
        page_title = self._extract_title(html_content=html_content)
        return CrawlResult(
            url=str(response.url),
            status_code=response.status_code,
            title=page_title,
            html=html_content,
        )

    @staticmethod
    def _extract_title(html_content: str) -> str:
        title_open = "<title>"
        title_close = "</title>"
        lower_html = html_content.lower()
        start_index = lower_html.find(title_open)
        end_index = lower_html.find(title_close)
        if start_index == -1 or end_index == -1 or end_index <= start_index:
            return ""
        raw_title = html_content[start_index + len(title_open) : end_index]
        return raw_title.strip()

    @staticmethod
    def _validate_url(url: str) -> None:
        parsed_url = urlparse(url)
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            raise ValueError("Invalid URL. Use a valid http/https URL.")
