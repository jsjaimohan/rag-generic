"""LLM-backed enrichment for ingestion (Step 6)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from backend.ingestion.enrichment_schema import EnrichmentArtifact, EnrichmentPayload
from backend.llm.qwen_service import LlmClientConfig, QwenService
from backend.prompts.enrichment_prompts import (
    ENRICHMENT_SYSTEM,
    build_enrichment_user_message,
)
from backend.settings import resolve_data_path, settings
from backend.utils.logger import get_logger

logger = get_logger(__name__)


def enrichment_artifact_path_for_markdown(markdown_path: Path) -> Path:
    """Resolved `{stem}.enrichment.json` under the dedicated enriched data directory."""
    out_dir = resolve_data_path(settings.data_enriched_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{markdown_path.stem}.enrichment.json"


def _strip_json_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.split("\n")
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return stripped


def _extract_json_object(text: str) -> dict | None:
    """Parse a JSON object from model output; tolerate stray wrapping text."""
    cleaned = _strip_json_fences(text)
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        parsed = json.loads(cleaned[start : end + 1])
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def _truncate_markdown(markdown_content: str, max_chars: int) -> str:
    if len(markdown_content) <= max_chars:
        return markdown_content
    return markdown_content[:max_chars] + "\n\n[... excerpt truncated for enrichment ...]"


class EnrichmentService:
    """Produce and persist validated enrichment JSON under `data/enriched`."""

    def __init__(self) -> None:
        self._qwen = QwenService(
            config=LlmClientConfig(
                base_url=settings.enrichment_llm_base_url,
                model=settings.enrichment_llm_model,
                api_key=settings.enrichment_llm_api_key,
                disable_thinking=settings.enrichment_llm_disable_thinking,
                http_attempts=settings.enrichment_llm_http_attempts,
                retry_backoff_seconds=settings.enrichment_llm_retry_backoff_seconds,
                read_timeout_seconds=settings.enrichment_llm_read_timeout_seconds,
            )
        )

    def build_fallback_payload(self, base_metadata: dict) -> EnrichmentPayload:
        """Structural fallback when the LLM fails or returns invalid JSON."""
        tags = base_metadata.get("tags") if isinstance(base_metadata.get("tags"), list) else []
        safe_tags = [str(t).strip() for t in tags[:12] if str(t).strip()]
        summary = str(base_metadata.get("summary", "")).strip()
        return EnrichmentPayload(
            short_summary=summary[:600],
            tags=safe_tags,
            entities=[],
            questions_answered=[],
            relationships=[],
            content_intents=[],
        )

    def enrich_from_markdown(
        self,
        markdown_content: str,
        base_metadata: dict,
    ) -> EnrichmentArtifact:
        """Call LLM and return a validated artifact (fallback on any failure)."""
        source_url = str(base_metadata.get("source_url", ""))
        title = str(base_metadata.get("title", ""))
        excerpt = _truncate_markdown(
            markdown_content,
            max_chars=settings.enrichment_max_input_chars,
        )
        user_message = build_enrichment_user_message(
            markdown_excerpt=excerpt,
            page_title=title,
            source_url=source_url,
        )
        enriched_at = datetime.now(UTC).isoformat()
        model_id = settings.enrichment_llm_model

        try:
            raw = self._qwen.complete_chat(
                messages=[
                    {"role": "system", "content": ENRICHMENT_SYSTEM},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.1,
            )
            data = _extract_json_object(raw)
            if not data:
                raise ValueError("No JSON object in model response.")
            payload = EnrichmentPayload.model_validate(data)
            return EnrichmentArtifact(
                source_url=source_url,
                title=title,
                enriched_at_utc=enriched_at,
                model=model_id,
                fallback=False,
                enrichment=payload,
            )
        except Exception as exc:
            logger.warning(
                "enrichment.llm_failed source_url=%s error=%s",
                source_url,
                exc,
            )
            return EnrichmentArtifact(
                source_url=source_url,
                title=title,
                enriched_at_utc=enriched_at,
                model=model_id,
                fallback=True,
                enrichment=self.build_fallback_payload(base_metadata),
            )

    def persist_enrichment(
        self,
        markdown_path: Path,
        base_metadata: dict,
        markdown_content: str | None = None,
    ) -> Path:
        """Write `{stem}.enrichment.json` under the configured enriched directory."""
        markdown_path = markdown_path.resolve()
        text = (
            markdown_content
            if markdown_content is not None
            else markdown_path.read_text(encoding="utf-8", errors="ignore")
        )
        artifact = self.enrich_from_markdown(
            markdown_content=text,
            base_metadata=base_metadata,
        )
        out_path = enrichment_artifact_path_for_markdown(markdown_path)
        out_path.write_text(
            artifact.model_dump_json(indent=2, exclude_none=True) + "\n",
            encoding="utf-8",
        )
        logger.info(
            "enrichment.saved path=%s fallback=%s",
            out_path,
            artifact.fallback,
        )
        return out_path

    def enrich_all_processing_markdown(
        self,
        processing_dir: Path,
        *,
        force: bool = False,
        limit: int | None = None,
    ) -> dict:
        """Batch-enrich `*.md` files; skip if enrichment exists unless force."""
        files = sorted(processing_dir.glob("*.md"))
        if limit is not None:
            files = files[: max(0, limit)]
        enriched = 0
        skipped = 0
        failed = 0
        paths: list[str] = []

        for md_path in files:
            enrichment_path = enrichment_artifact_path_for_markdown(md_path)
            if enrichment_path.is_file() and not force:
                skipped += 1
                continue
            meta_path = md_path.with_suffix(".metadata.json")
            if not meta_path.is_file():
                failed += 1
                continue
            try:
                raw_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                failed += 1
                continue
            try:
                out = self.persist_enrichment(
                    markdown_path=md_path,
                    base_metadata=raw_meta if isinstance(raw_meta, dict) else {},
                )
                paths.append(str(out))
                enriched += 1
            except Exception:
                logger.exception("enrichment.batch_failed path=%s", md_path)
                failed += 1

        return {
            "enriched_count": enriched,
            "skipped_count": skipped,
            "failed_count": failed,
            "enrichment_paths": paths,
        }
