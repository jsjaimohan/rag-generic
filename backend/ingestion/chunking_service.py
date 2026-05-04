"""Orchestrate Step 7 chunk file generation."""

from __future__ import annotations

import json
import threading
from pathlib import Path

from backend.ingestion.chunking_jobs import complete_job, fail_job, update_job
from backend.ingestion.enrichment_service import enrichment_artifact_path_for_markdown
from backend.ingestion.semantic_chunker import build_chunk_manifest
from backend.settings import resolve_data_path, settings
from backend.utils.logger import get_logger

logger = get_logger(__name__)


def chunks_manifest_path_for_markdown(markdown_path: Path) -> Path:
    """Resolved `{stem}.chunks.json` under the semantic-chunks data directory."""
    out_dir = resolve_data_path(settings.data_semantic_chunks_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{markdown_path.stem}.chunks.json"


def _load_json_dict(path: Path) -> dict:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    data = json.loads(raw)
    return data if isinstance(data, dict) else {}


def _enrichment_relationships_from_path(enrichment_path: Path) -> list[dict] | None:
    if not enrichment_path.is_file():
        return None
    try:
        data = _load_json_dict(enrichment_path)
    except (OSError, json.JSONDecodeError):
        return None
    enrichment = data.get("enrichment")
    if not isinstance(enrichment, dict):
        return None
    rel = enrichment.get("relationships")
    return rel if isinstance(rel, list) else None


def write_chunks_for_markdown(
    *,
    md_path: Path,
    force: bool = False,
) -> tuple[bool, str | None]:
    """Return (wrote_file, chunks_json_path or None if skipped)."""
    md_path = md_path.resolve()
    out_path = chunks_manifest_path_for_markdown(md_path)
    if out_path.is_file() and not force:
        return False, None

    markdown_content = md_path.read_text(encoding="utf-8", errors="ignore")
    meta_path = md_path.with_suffix(".metadata.json")
    source_url = ""
    page_title = ""
    if meta_path.is_file():
        try:
            meta = _load_json_dict(meta_path)
            source_url = str(meta.get("source_url", ""))
            page_title = str(meta.get("title", ""))
        except (OSError, json.JSONDecodeError):
            pass

    enrichment_path = enrichment_artifact_path_for_markdown(md_path)
    enr_rels = _enrichment_relationships_from_path(enrichment_path)

    manifest = build_chunk_manifest(
        markdown_path=str(md_path),
        markdown_content=markdown_content,
        source_url=source_url,
        page_title=page_title,
        enrichment_relationships=enr_rels,
    )
    out_path.write_text(
        manifest.model_dump_json(indent=2, exclude_none=True) + "\n",
        encoding="utf-8",
    )
    logger.info(
        "chunking.wrote path=%s chunks=%s",
        out_path,
        manifest.quality.chunk_count,
    )
    return True, str(out_path)


def run_chunking_job(
    job_id: str,
    markdown_files: list[Path],
    force: bool,
) -> None:
    """Background worker: update job progress as each file is processed."""
    logger.info(
        "Semantic chunking — START (worker) job_id=%s total_files=%s force=%s",
        job_id,
        len(markdown_files),
        force,
    )
    try:
        written = 0
        skipped = 0
        failed = 0
        chunks_total = 0

        for idx, md_path in enumerate(markdown_files):
            update_job(job_id, processed_files=idx, current_file=md_path.name)
            try:
                did_write, _ = write_chunks_for_markdown(md_path=md_path, force=force)
                if did_write:
                    written += 1
                    chunks_json = chunks_manifest_path_for_markdown(md_path)
                    if chunks_json.is_file():
                        data = json.loads(chunks_json.read_text(encoding="utf-8"))
                        chunks_total += int(data.get("quality", {}).get("chunk_count", 0))
                else:
                    skipped += 1
            except Exception:
                logger.exception("chunking.file_failed path=%s", md_path)
                failed += 1
            update_job(job_id, processed_files=idx + 1, current_file="")

        complete_job(
            job_id,
            {
                "files_written": written,
                "files_skipped": skipped,
                "files_failed": failed,
                "chunks_total": chunks_total,
            },
        )
        logger.info(
            "Semantic chunking — END (worker) job_id=%s files_written=%s files_skipped=%s files_failed=%s chunks_total=%s",
            job_id,
            written,
            skipped,
            failed,
            chunks_total,
        )
    except Exception as exc:
        logger.exception("chunking.job_failed job_id=%s", job_id)
        logger.info("Semantic chunking — END (worker) job_id=%s status=failed error=%s", job_id, exc)
        fail_job(job_id, str(exc))


def start_chunking_background(
    job_id: str,
    markdown_files: list[Path],
    force: bool,
) -> None:
    thread = threading.Thread(
        target=run_chunking_job,
        args=(job_id, markdown_files, force),
        daemon=True,
    )
    thread.start()
