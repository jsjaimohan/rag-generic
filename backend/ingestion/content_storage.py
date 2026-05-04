"""Filesystem storage for crawl and ingestion artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha1
import json
from pathlib import Path
from urllib.parse import urlparse

from backend.settings import resolve_data_path, settings
from backend.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class StoredIngestionPaths:
    """Saved file paths generated during one ingestion item."""

    raw_html_path: str
    cleaned_html_path: str
    markdown_path: str
    metadata_path: str


@dataclass
class ExistingIngestionPaths:
    """Previously saved file paths for the same URL."""

    raw_html_path: str | None
    cleaned_html_path: str | None
    markdown_path: str | None
    metadata_path: str | None


class ContentStorageService:
    """Persist raw, cleaned, and processed artifacts to local disk."""

    def __init__(self) -> None:
        self.raw_dir = resolve_data_path(settings.data_raw_dir)
        self.clean_dir = resolve_data_path(settings.data_clean_dir)
        self.processing_dir = resolve_data_path(settings.data_processing_dir)
        self.enriched_dir = resolve_data_path(settings.data_enriched_dir)
        self.semantic_chunks_dir = resolve_data_path(settings.data_semantic_chunks_dir)
        self._ensure_directories()

    def save_ingestion_artifacts(
        self,
        source_url: str,
        raw_html: str,
        cleaned_html: str,
        markdown_content: str,
        metadata: dict,
    ) -> StoredIngestionPaths:
        """Save all pipeline artifacts for one URL."""
        logger.info("content_storage.save.start source_url=%s", source_url)
        file_stem = self._build_file_stem(source_url=source_url)
        raw_html_path = self.raw_dir / f"{file_stem}.html"
        cleaned_html_path = self.clean_dir / f"{file_stem}.clean.html"
        markdown_path = self.processing_dir / f"{file_stem}.md"
        metadata_path = self.processing_dir / f"{file_stem}.metadata.json"

        raw_html_path.write_text(raw_html, encoding="utf-8")
        cleaned_html_path.write_text(cleaned_html, encoding="utf-8")
        markdown_path.write_text(markdown_content, encoding="utf-8")
        metadata_path.write_text(self._serialize_metadata(metadata=metadata), encoding="utf-8")

        logger.info(
            "content_storage.save.done source_url=%s raw=%s clean=%s md=%s metadata=%s",
            source_url,
            raw_html_path,
            cleaned_html_path,
            markdown_path,
            metadata_path,
        )

        return StoredIngestionPaths(
            raw_html_path=str(raw_html_path),
            cleaned_html_path=str(cleaned_html_path),
            markdown_path=str(markdown_path),
            metadata_path=str(metadata_path),
        )

    def get_existing_artifacts(self, source_url: str) -> ExistingIngestionPaths | None:
        """Return latest stored artifacts for URL if available."""
        url_digest = self._build_url_digest(source_url=source_url)
        raw_html_path = self._latest_file_for_pattern(self.raw_dir, f"*__{url_digest}.html")
        cleaned_html_path = self._latest_file_for_pattern(
            self.clean_dir,
            f"*__{url_digest}.clean.html",
        )
        markdown_path = self._latest_file_for_pattern(
            self.processing_dir,
            f"*__{url_digest}.md",
        )
        metadata_path = self._latest_file_for_pattern(
            self.processing_dir,
            f"*__{url_digest}.metadata.json",
        )
        if not any([raw_html_path, cleaned_html_path, markdown_path, metadata_path]):
            return None
        return ExistingIngestionPaths(
            raw_html_path=str(raw_html_path) if raw_html_path else None,
            cleaned_html_path=str(cleaned_html_path) if cleaned_html_path else None,
            markdown_path=str(markdown_path) if markdown_path else None,
            metadata_path=str(metadata_path) if metadata_path else None,
        )

    def get_storage_counts(self) -> dict:
        """Return file counts for data directories."""
        enrichment_count = len(list(self.enriched_dir.glob("*.enrichment.json")))
        chunks_count = len(list(self.semantic_chunks_dir.glob("*.chunks.json")))
        return {
            "raw_count": self._count_files(self.raw_dir),
            "clean_count": self._count_files(self.clean_dir),
            "processing_count": self._count_files(self.processing_dir),
            "enrichment_count": enrichment_count,
            "chunks_count": chunks_count,
            "raw_dir": str(self.raw_dir),
            "clean_dir": str(self.clean_dir),
            "processing_dir": str(self.processing_dir),
            "enriched_dir": str(self.enriched_dir),
            "semantic_chunks_dir": str(self.semantic_chunks_dir),
        }

    def cleanup_storage(
        self,
        clear_raw: bool,
        clear_clean: bool,
        clear_processing: bool,
        clear_enriched: bool = False,
        clear_semantic_chunks: bool = False,
    ) -> dict:
        """Delete files from selected storage locations."""
        logger.info(
            "Clean — START clear_raw=%s clear_clean=%s clear_processing=%s clear_enriched=%s clear_semantic_chunks=%s",
            clear_raw,
            clear_clean,
            clear_processing,
            clear_enriched,
            clear_semantic_chunks,
        )
        deleted_raw = self._delete_files(self.raw_dir) if clear_raw else 0
        deleted_clean = self._delete_files(self.clean_dir) if clear_clean else 0
        deleted_processing = self._delete_files(self.processing_dir) if clear_processing else 0
        deleted_enriched = self._delete_files(self.enriched_dir) if clear_enriched else 0
        deleted_semantic_chunks = (
            self._delete_files(self.semantic_chunks_dir) if clear_semantic_chunks else 0
        )
        logger.info(
            "Clean — END deleted_raw=%s deleted_clean=%s deleted_processing=%s deleted_enriched=%s deleted_semantic_chunks=%s",
            deleted_raw,
            deleted_clean,
            deleted_processing,
            deleted_enriched,
            deleted_semantic_chunks,
        )
        return {
            "deleted_raw": deleted_raw,
            "deleted_clean": deleted_clean,
            "deleted_processing": deleted_processing,
            "deleted_enriched": deleted_enriched,
            "deleted_semantic_chunks": deleted_semantic_chunks,
            "counts_after_cleanup": self.get_storage_counts(),
        }

    def _ensure_directories(self) -> None:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.clean_dir.mkdir(parents=True, exist_ok=True)
        self.processing_dir.mkdir(parents=True, exist_ok=True)
        self.enriched_dir.mkdir(parents=True, exist_ok=True)
        self.semantic_chunks_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _build_file_stem(source_url: str) -> str:
        parsed_url = urlparse(source_url)
        host = parsed_url.netloc.replace(":", "_")
        path = parsed_url.path.strip("/").replace("/", "_") or "root"
        digest = ContentStorageService._build_url_digest(source_url=source_url)
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        return f"{host}__{path}__{timestamp}__{digest}"

    @staticmethod
    def _build_url_digest(source_url: str) -> str:
        return sha1(source_url.encode("utf-8"), usedforsecurity=False).hexdigest()[:10]

    @staticmethod
    def _latest_file_for_pattern(directory: Path, pattern: str) -> Path | None:
        matched_files = list(directory.glob(pattern))
        if not matched_files:
            return None
        return max(matched_files, key=lambda candidate: candidate.stat().st_mtime)

    @staticmethod
    def _count_files(directory: Path) -> int:
        return len([candidate for candidate in directory.iterdir() if candidate.is_file()])

    @staticmethod
    def _delete_files(directory: Path) -> int:
        deleted_count = 0
        for candidate in directory.iterdir():
            if candidate.is_file():
                candidate.unlink()
                deleted_count += 1
        return deleted_count

    @staticmethod
    def _serialize_metadata(metadata: dict) -> str:
        return json.dumps(metadata, indent=2, ensure_ascii=False) + "\n"
