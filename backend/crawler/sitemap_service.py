"""Sitemap loading and URL grouping utilities."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from urllib.parse import urlparse
import xml.etree.ElementTree as ET

import httpx

from backend.settings import settings


@dataclass
class SitemapDiscoveryResult:
    """Flattened sitemap discovery output."""

    urls: list[str]
    discovered_sitemap_urls: list[str]


@dataclass
class SitemapGroup:
    """Grouped URLs under one logical bucket."""

    group_id: str
    group_name: str
    urls: list[str]
    source_sitemaps: list[str]


class SitemapService:
    """Load sitemap XML and build URL groups for selective ingestion."""

    def discover_urls_from_sitemap(self, sitemap_url: str) -> SitemapDiscoveryResult:
        """Fetch sitemap(s) and return deduplicated URLs discovered recursively."""
        self._validate_http_url(url=sitemap_url)

        visited_sitemaps: set[str] = set()
        sitemap_queue: list[str] = [sitemap_url]
        discovered_sitemap_urls: list[str] = []
        url_entries: list[str] = []
        url_to_sitemap: dict[str, str] = {}

        while sitemap_queue:
            current_sitemap_url = sitemap_queue.pop(0)
            if current_sitemap_url in visited_sitemaps:
                continue

            visited_sitemaps.add(current_sitemap_url)
            discovered_sitemap_urls.append(current_sitemap_url)
            xml_content = self._fetch_sitemap_xml(sitemap_url=current_sitemap_url)
            parsed_result = self._parse_sitemap_xml(xml_content=xml_content)

            for child_sitemap_url in parsed_result["sitemap_urls"]:
                if child_sitemap_url not in visited_sitemaps:
                    sitemap_queue.append(child_sitemap_url)

            for page_url in parsed_result["page_urls"]:
                url_entries.append(page_url)
                if page_url not in url_to_sitemap:
                    url_to_sitemap[page_url] = current_sitemap_url

        if not url_entries:
            raise ValueError("No URL entries found in sitemap.")

        deduplicated_urls = list(dict.fromkeys(url_entries))
        self._url_to_sitemap_map = url_to_sitemap
        return SitemapDiscoveryResult(
            urls=deduplicated_urls,
            discovered_sitemap_urls=discovered_sitemap_urls,
        )

    def group_urls(self, urls: list[str]) -> list[SitemapGroup]:
        """Group URLs by normalized path prefix."""
        grouped_urls: dict[str, list[str]] = defaultdict(list)
        grouped_source_sitemaps: dict[str, set[str]] = defaultdict(set)
        for url in urls:
            group_key = self._build_group_key(url=url)
            grouped_urls[group_key].append(url)
            source_sitemap = self._url_to_sitemap_map.get(url, "")
            if source_sitemap:
                grouped_source_sitemaps[group_key].add(source_sitemap)

        sitemap_groups: list[SitemapGroup] = []
        for group_key in sorted(grouped_urls):
            urls_for_group = grouped_urls[group_key]
            sitemap_groups.append(
                SitemapGroup(
                    group_id=group_key,
                    group_name=group_key.replace("-", " ").title(),
                    urls=urls_for_group,
                    source_sitemaps=sorted(grouped_source_sitemaps[group_key]),
                )
            )
        return sitemap_groups

    def __init__(self) -> None:
        self._url_to_sitemap_map: dict[str, str] = {}

    def _fetch_sitemap_xml(self, sitemap_url: str) -> str:
        response = httpx.get(
            sitemap_url,
            timeout=settings.crawler_timeout_seconds,
            headers={"User-Agent": settings.crawler_user_agent},
            follow_redirects=True,
        )
        response.raise_for_status()
        return response.text

    @staticmethod
    def _parse_sitemap_xml(xml_content: str) -> dict[str, list[str]]:
        try:
            root = ET.fromstring(xml_content)
        except ET.ParseError as exc:
            raise ValueError("Invalid sitemap XML.") from exc

        namespace_prefix = ""
        if root.tag.startswith("{") and "}" in root.tag:
            namespace_prefix = root.tag.split("}")[0] + "}"

        page_urls: list[str] = []
        sitemap_urls: list[str] = []
        if root.tag.endswith("urlset"):
            page_urls = SitemapService._extract_loc_values(
                root=root,
                namespace_prefix=namespace_prefix,
            )
        elif root.tag.endswith("sitemapindex"):
            sitemap_urls = SitemapService._extract_loc_values(
                root=root,
                namespace_prefix=namespace_prefix,
            )
        else:
            loc_values = SitemapService._extract_loc_values(
                root=root,
                namespace_prefix=namespace_prefix,
            )
            page_urls = loc_values

        return {
            "page_urls": page_urls,
            "sitemap_urls": sitemap_urls,
        }

    @staticmethod
    def _extract_loc_values(root: ET.Element, namespace_prefix: str) -> list[str]:
        loc_tag = f"{namespace_prefix}loc"
        return [node.text.strip() for node in root.iter(loc_tag) if node.text]

    @staticmethod
    def _build_group_key(url: str) -> str:
        parsed_url = urlparse(url)
        path_parts = [part for part in parsed_url.path.split("/") if part]
        if not path_parts:
            return "root"

        if path_parts[0] in {"docs", "documentation"} and len(path_parts) >= 2:
            return f"{path_parts[0]}-{path_parts[1]}"
        return path_parts[0]

    @staticmethod
    def _validate_http_url(url: str) -> None:
        parsed_url = urlparse(url)
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            raise ValueError("Invalid sitemap URL. Use a valid http/https URL.")
