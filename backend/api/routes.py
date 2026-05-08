"""API routes for GenericRAG."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, HttpUrl
from starlette.concurrency import run_in_threadpool

from backend.crawler.crawler import CrawlerService
from backend.crawler.sitemap_service import SitemapGroup, SitemapService
from backend.evaluation.eval_service import EvaluationCase, EvaluationService
from backend.ingestion.chunking_jobs import create_job, get_job, job_to_response
from backend.ingestion.chunking_service import start_chunking_background
from backend.ingestion.content_storage import ContentStorageService
from backend.ingestion.indexing_jobs import (
    create_indexing_job,
    get_indexing_job,
    indexing_job_to_response,
)
from backend.ingestion.indexing_service import start_indexing_background
from backend.ingestion.enrichment_service import EnrichmentService
from backend.ingestion.html_cleaner import HtmlCleaner
from backend.ingestion.markdown_converter import MarkdownConverter
from backend.ingestion.metadata_generator import MetadataGenerator
from backend.llm.qwen_service import QwenService
from backend.rag.minirag_service import MiniRagService
from backend.settings import settings
from backend.utils.logger import get_logger
from backend.vectordb.qdrant_service import qdrant_service

logger = get_logger(__name__)

router = APIRouter()
crawler_service = CrawlerService()
sitemap_service = SitemapService()
html_cleaner = HtmlCleaner()
markdown_converter = MarkdownConverter()
metadata_generator = MetadataGenerator()
content_storage_service = ContentStorageService()
enrichment_service = EnrichmentService()
qwen_service = QwenService()
minirag_service = MiniRagService()
evaluation_service = EvaluationService()


class CrawlRequest(BaseModel):
    """Request payload for crawling one URL."""

    url: HttpUrl


class IngestRequest(BaseModel):
    """Request payload for crawl + markdown pipeline."""

    url: HttpUrl
    skip_scrape_if_exists: bool = False


class SitemapLoadRequest(BaseModel):
    """Request payload for loading sitemap and generating groups."""

    sitemap_url: HttpUrl


class GroupIngestRequest(BaseModel):
    """Request payload for ingesting selected sitemap groups."""

    sitemap_url: HttpUrl
    selected_group_ids: list[str] = []
    selected_urls: list[HttpUrl] = []
    skip_scrape_if_exists: bool = False


class ChatRequest(BaseModel):
    """Request payload for chatbot API."""

    query: str
    top_k: int = 4


class DataCleanupRequest(BaseModel):
    """Request payload for cleaning stored data artifacts."""

    clear_raw: bool = False
    clear_clean: bool = False
    clear_processing: bool = False
    clear_enriched: bool = False
    clear_semantic_chunks: bool = False


class EvaluationCaseRequest(BaseModel):
    """One evaluation case payload."""

    query: str
    expected_terms: list[str]


class EvaluationRunRequest(BaseModel):
    """Batch evaluation payload."""

    test_cases: list[EvaluationCaseRequest]
    top_k: int = 4


class RunEnrichmentRequest(BaseModel):
    """Batch Step 6 enrichment over existing markdown in processing."""

    force: bool = False
    limit: int | None = None


class RunChunkingRequest(BaseModel):
    """Step 7 semantic chunking over markdown in processing."""

    force: bool = False
    limit: int | None = None


class RunIndexingRequest(BaseModel):
    """Step 8: embed chunk manifests and upsert into Qdrant."""

    recreate_collection: bool = False
    limit: int | None = None


@router.get("/health")
def health() -> dict:
    return {"ok": True}


@router.post("/crawl")
def crawl(request: CrawlRequest) -> dict:
    try:
        crawl_result = crawler_service.crawl_url(url=str(request.url))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "url": crawl_result.url,
        "status_code": crawl_result.status_code,
        "title": crawl_result.title,
        "content_length": len(crawl_result.html),
    }


@router.get("/qdrant/health")
def qdrant_health() -> dict:
    try:
        return qdrant_service.health_check()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/llm/health")
def llm_health() -> dict:
    try:
        return qwen_service.health_check()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/chat")
async def chat(request: ChatRequest) -> dict:
    """Chat backed by MiniRAG; CPU/GPU-bound work runs in a thread pool (non-blocking event loop)."""
    try:
        if not request.query.strip():
            raise ValueError("Query must not be empty.")
        return await run_in_threadpool(
            minirag_service.chat,
            request.query,
            request.top_k,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """
    Same retrieval as ``POST /chat``, then stream LM Studio tokens as Server-Sent Events.

    Each line is ``data: <json>\\n\\n``. Event types:

    - ``meta`` — retrieval fields (no ``answer``; ``performance`` lacks ``llm_generation_s`` / ``chat_total_s``).
    - ``token`` — ``{"type":"token","delta":"..."}`` (many events).
    - ``done`` — full ``answer`` and final ``performance`` (adds ``llm_generation_s``, ``chat_total_s``).
    - ``error`` — ``{"type":"error","detail":"..."}`` then the stream ends.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty.")

    async def event_generator():
        try:
            meta_partial, messages = await run_in_threadpool(
                minirag_service.prepare_chat_context,
                request.query,
                request.top_k,
            )
            yield (
                "data: "
                + json.dumps({"type": "meta", "payload": meta_partial}, ensure_ascii=False)
                + "\n\n"
            )

            t_llm = time.perf_counter()
            parts: list[str] = []
            async for delta in minirag_service.qwen_service.astream_complete_chat(
                messages=messages,
                temperature=0.1,
            ):
                parts.append(delta)
                yield (
                    "data: "
                    + json.dumps({"type": "token", "delta": delta}, ensure_ascii=False)
                    + "\n\n"
                )

            answer = "".join(parts).strip()
            llm_generation_s = time.perf_counter() - t_llm
            perf = meta_partial["performance"]
            retrieval_wall_s = perf["retrieval_wall_s"]
            prompt_build_s = perf["prompt_build_s"]
            chat_total_s = retrieval_wall_s + prompt_build_s + llm_generation_s
            perf["llm_generation_s"] = round(llm_generation_s, 4)
            perf["chat_total_s"] = round(chat_total_s, 4)

            done_payload = {
                "type": "done",
                "answer": answer,
                "performance": perf,
            }
            yield "data: " + json.dumps(done_payload, ensure_ascii=False) + "\n\n"
        except Exception as exc:
            logger.warning("chat_stream.failed error=%s", exc, exc_info=True)
            err_payload = {"type": "error", "detail": str(exc)}
            yield "data: " + json.dumps(err_payload, ensure_ascii=False) + "\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/evaluation/run")
async def run_evaluation(request: EvaluationRunRequest) -> dict:
    """Run retrieval and hallucination evaluation (blocking pipeline offloaded to thread pool)."""
    try:
        logger.info(
            "api.evaluation.run accepted cases=%s top_k=%s",
            len(request.test_cases),
            request.top_k,
        )
        evaluation_cases = [
            EvaluationCase(query=item.query, expected_terms=item.expected_terms)
            for item in request.test_cases
        ]
        return await run_in_threadpool(
            evaluation_service.run,
            evaluation_cases,
            request.top_k,
        )
    except Exception as exc:
        logger.warning("api.evaluation.run failed: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/data/status")
def data_status() -> dict:
    """Return storage paths, file counts, and Qdrant collection point count."""
    counts = content_storage_service.get_storage_counts()
    counts["qdrant_collection"] = settings.qdrant_collection_name
    try:
        counts["qdrant_points_count"] = qdrant_service.count_points()
    except Exception as exc:
        logger.warning("data_status.qdrant_count_failed error=%s", exc)
        counts["qdrant_points_count"] = None
        counts["qdrant_error"] = str(exc)
    return counts


@router.post("/admin/data/cleanup")
def cleanup_data(request: DataCleanupRequest) -> dict:
    """Delete selected local storage artifacts."""
    if not any(
        [
            request.clear_raw,
            request.clear_clean,
            request.clear_processing,
            request.clear_enriched,
            request.clear_semantic_chunks,
        ]
    ):
        raise HTTPException(status_code=400, detail="Select at least one data area to clean.")
    return content_storage_service.cleanup_storage(
        clear_raw=request.clear_raw,
        clear_clean=request.clear_clean,
        clear_processing=request.clear_processing,
        clear_enriched=request.clear_enriched,
        clear_semantic_chunks=request.clear_semantic_chunks,
    )


@router.post("/admin/data/run-clean")
def run_clean_step() -> dict:
    """Run HTML cleaning for files currently present in raw storage."""
    logger.info("Processing — Run clean — START")
    raw_dir = Path(content_storage_service.raw_dir)
    clean_dir = Path(content_storage_service.clean_dir)
    deleted_clean_before_run = content_storage_service.cleanup_storage(
        clear_raw=False,
        clear_clean=True,
        clear_processing=False,
    )["deleted_clean"]
    cleaned_count = 0
    failed_count = 0

    for raw_file in raw_dir.glob("*.html"):
        file_stem = raw_file.stem
        clean_file = clean_dir / f"{file_stem}.clean.html"
        try:
            raw_html = raw_file.read_text(encoding="utf-8", errors="ignore")
            cleaned_html = html_cleaner.clean_html_content(raw_html=raw_html)
            clean_file.write_text(cleaned_html, encoding="utf-8")
            cleaned_count += 1
        except Exception:
            failed_count += 1

    logger.info(
        "Processing — Run clean — END cleaned_count=%s failed_count=%s deleted_clean_before_run=%s",
        cleaned_count,
        failed_count,
        deleted_clean_before_run,
    )
    return {
        "deleted_clean_before_run": deleted_clean_before_run,
        "cleaned_count": cleaned_count,
        "skipped_count": 0,
        "failed_count": failed_count,
    }


@router.post("/admin/data/run-processing")
def run_processing_step() -> dict:
    """Run markdown + metadata generation for cleaned HTML files."""
    logger.info("Processing — Run processing — START")
    clean_dir = Path(content_storage_service.clean_dir)
    processing_dir = Path(content_storage_service.processing_dir)
    deleted_processing_before_run = content_storage_service.cleanup_storage(
        clear_raw=False,
        clear_clean=False,
        clear_processing=True,
    )["deleted_processing"]
    processed_count = 0
    failed_count = 0

    for clean_file in clean_dir.glob("*.clean.html"):
        file_stem = clean_file.name.removesuffix(".clean.html")
        markdown_file = processing_dir / f"{file_stem}.md"
        metadata_file = processing_dir / f"{file_stem}.metadata.json"
        try:
            cleaned_html = clean_file.read_text(encoding="utf-8", errors="ignore")
            markdown_content = markdown_converter.convert_html_to_markdown(
                cleaned_html=cleaned_html
            )
            source_url = f"local://{file_stem}"
            generated_metadata = metadata_generator.generate_metadata(
                source_url=source_url,
                page_title=file_stem,
                markdown_content=markdown_content,
            )
            markdown_file.write_text(markdown_content, encoding="utf-8")
            metadata_file.write_text(
                content_storage_service._serialize_metadata(metadata=generated_metadata),
                encoding="utf-8",
            )
            try:
                enrichment_service.persist_enrichment(
                    markdown_path=markdown_file,
                    base_metadata=generated_metadata,
                    markdown_content=markdown_content,
                )
            except Exception as exc:
                logger.warning(
                    "run_processing.enrichment_failed stem=%s error=%s",
                    file_stem,
                    exc,
                )
            processed_count += 1
        except Exception:
            failed_count += 1

    logger.info(
        "Processing — Run processing — END processed_count=%s failed_count=%s deleted_processing_before_run=%s",
        processed_count,
        failed_count,
        deleted_processing_before_run,
    )
    return {
        "deleted_processing_before_run": deleted_processing_before_run,
        "processed_count": processed_count,
        "skipped_count": 0,
        "failed_count": failed_count,
    }


@router.post("/admin/data/run-enrichment")
def run_enrichment_step(request: RunEnrichmentRequest) -> dict:
    """Step 6: LLM enrichment for markdown files in processing (skips existing unless force)."""
    processing_dir = Path(content_storage_service.processing_dir)
    logger.info(
        "Enrichment — START force=%s limit=%s processing_dir=%s",
        request.force,
        request.limit,
        processing_dir,
    )
    try:
        result = enrichment_service.enrich_all_processing_markdown(
            processing_dir,
            force=request.force,
            limit=request.limit,
        )
        logger.info(
            "Enrichment — END enriched_count=%s skipped_count=%s failed_count=%s",
            result["enriched_count"],
            result["skipped_count"],
            result["failed_count"],
        )
        return result
    except Exception as exc:
        logger.info("Enrichment — END status=failed error=%s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/admin/data/run-chunking")
def start_chunking_job(request: RunChunkingRequest) -> dict:
    """Step 7: build `{stem}.chunks.json` under semantic-chunks dir for each markdown (async job)."""
    processing_dir = Path(content_storage_service.processing_dir)
    markdown_files = sorted(processing_dir.glob("*.md"))
    if request.limit is not None:
        markdown_files = markdown_files[: max(0, request.limit)]
    job_id = create_job(len(markdown_files))
    logger.info(
        "Semantic chunking — START (job accepted) job_id=%s total_files=%s force=%s",
        job_id,
        len(markdown_files),
        request.force,
    )
    start_chunking_background(job_id, markdown_files, request.force)
    return {"job_id": job_id, "total_files": len(markdown_files)}


def _chunking_unknown_event_payload(job_id: str) -> dict:
    return job_to_response(
        {
            "job_id": job_id,
            "status": "unknown",
            "processed_files": 0,
            "total_files": 0,
            "current_file": "",
            "error": "Job not found (API restart or reload clears in-memory job state).",
            "result": None,
        }
    )


@router.get("/admin/data/chunking-jobs/{job_id}")
def chunking_job_status(job_id: str) -> dict:
    """Poll chunking job progress (optional; UI uses SSE ``/chunking-events``)."""
    job = get_job(job_id)
    if not job:
        return _chunking_unknown_event_payload(job_id)
    return job_to_response(job)


@router.get("/admin/data/chunking-events/{job_id}")
async def chunking_job_event_stream(job_id: str) -> StreamingResponse:
    """Server-Sent Events stream of chunking job progress (replaces browser polling)."""

    async def event_generator():
        while True:
            job = get_job(job_id)
            if not job:
                yield f"data: {json.dumps(_chunking_unknown_event_payload(job_id))}\n\n"
                break
            payload = job_to_response(job)
            yield f"data: {json.dumps(payload)}\n\n"
            if payload["status"] in ("completed", "failed", "unknown"):
                break
            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/admin/data/run-indexing")
def start_indexing_job(request: RunIndexingRequest) -> dict:
    """Step 8: dense-embed chunks from ``*.chunks.json`` and upsert into Qdrant (async job)."""
    semantic_dir = Path(content_storage_service.semantic_chunks_dir)
    manifest_paths = sorted(semantic_dir.glob("*.chunks.json"))
    if request.limit is not None:
        manifest_paths = manifest_paths[: max(0, request.limit)]
    job_id = create_indexing_job(len(manifest_paths))
    logger.info(
        "Step 8 — Persist vector embeddings: START (job accepted, worker will embed + upsert) job_id=%s "
        "manifests=%s recreate_collection=%s",
        job_id,
        len(manifest_paths),
        request.recreate_collection,
    )
    start_indexing_background(job_id, manifest_paths, request.recreate_collection)
    return {"job_id": job_id, "total_files": len(manifest_paths)}


def _indexing_unknown_event_payload(job_id: str) -> dict:
    return indexing_job_to_response(
        {
            "job_id": job_id,
            "status": "unknown",
            "phase": "unknown",
            "processed_files": 0,
            "total_files": 0,
            "points_indexed": 0,
            "current_file": "",
            "error": "Job not found (API restart or reload clears in-memory job state).",
            "result": None,
        }
    )


@router.get("/admin/data/indexing-jobs/{job_id}")
def indexing_job_status(job_id: str) -> dict:
    """Poll Step 8 indexing job progress (optional; UI uses SSE ``/indexing-events``)."""
    job = get_indexing_job(job_id)
    if not job:
        return _indexing_unknown_event_payload(job_id)
    return indexing_job_to_response(job)


@router.get("/admin/data/indexing-events/{job_id}")
async def indexing_job_event_stream(job_id: str) -> StreamingResponse:
    """Server-Sent Events stream of Step 8 indexing job progress (replaces browser polling)."""

    async def event_generator():
        while True:
            job = get_indexing_job(job_id)
            if not job:
                yield f"data: {json.dumps(_indexing_unknown_event_payload(job_id))}\n\n"
                break
            payload = indexing_job_to_response(job)
            yield f"data: {json.dumps(payload)}\n\n"
            if payload["status"] in ("completed", "failed", "unknown"):
                break
            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/admin/data/delete-embeddings")
def delete_embeddings() -> dict:
    """Drop all vectors in the configured Qdrant collection (same schema after recreate)."""
    count_before: int | None = None
    try:
        count_before = qdrant_service.count_points()
    except Exception as exc:
        logger.warning("delete_embeddings.count_before_failed error=%s", exc)
    try:
        qdrant_service.ensure_collection(recreate=True)
    except Exception as exc:
        logger.exception("delete_embeddings.failed")
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    logger.info(
        "delete_embeddings.done collection=%s removed_points=%s",
        settings.qdrant_collection_name,
        count_before,
    )
    return {
        "ok": True,
        "removed_points": count_before,
        "qdrant_points_count": 0,
        "collection": settings.qdrant_collection_name,
    }


@router.post("/ingest")
def ingest(request: IngestRequest) -> dict:
    """Step 2 pipeline: crawl -> clean HTML -> markdown -> metadata."""
    try:
        logger.info(
            "ingest.single.start url=%s skip_scrape_if_exists=%s",
            request.url,
            request.skip_scrape_if_exists,
        )
        ingest_result = _run_single_url_ingestion(
            url=str(request.url),
            skip_scrape_if_exists=request.skip_scrape_if_exists,
        )
        logger.info(
            "ingest.single.done url=%s status=%s",
            request.url,
            ingest_result.get("status"),
        )
    except Exception as exc:
        logger.exception("ingest.single.failed url=%s error=%s", request.url, exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return ingest_result


@router.post("/sitemap/load")
def load_sitemap(request: SitemapLoadRequest) -> dict:
    """Discover sitemap URLs and return selectable URL groups."""
    try:
        discovery_result = sitemap_service.discover_urls_from_sitemap(
            sitemap_url=str(request.sitemap_url)
        )
        groups = sitemap_service.group_urls(urls=discovery_result.urls)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    grouped_response = []
    for group in groups:
        grouped_response.append(
            {
                "group_id": group.group_id,
                "group_name": group.group_name,
                "url_count": len(group.urls),
                "sample_urls": group.urls[:3],
                "source_sitemaps_count": len(group.source_sitemaps),
            }
        )

    return {
        "sitemap_url": str(request.sitemap_url),
        "total_urls": len(discovery_result.urls),
        "discovered_sitemaps_count": len(discovery_result.discovered_sitemap_urls),
        "sample_sitemaps": discovery_result.discovered_sitemap_urls[:5],
        "groups": grouped_response,
    }


@router.post("/ingest/groups")
def ingest_groups(request: GroupIngestRequest) -> dict:
    """Ingest selected sitemap groups or directly selected URLs.

    This endpoint is fault-tolerant: if one URL fails (404, timeout, parsing),
    ingestion continues for remaining URLs and reports failures in response.
    """
    try:
        logger.info(
            "ingest.groups.start sitemap_url=%s selected_groups=%s skip_scrape_if_exists=%s",
            request.sitemap_url,
            len(request.selected_group_ids),
            request.skip_scrape_if_exists,
        )
        discovery_result = sitemap_service.discover_urls_from_sitemap(
            sitemap_url=str(request.sitemap_url)
        )
        sitemap_groups = sitemap_service.group_urls(urls=discovery_result.urls)
        urls_to_ingest = _select_urls_for_group_ingestion(
            sitemap_groups=sitemap_groups,
            selected_group_ids=request.selected_group_ids,
            selected_urls=[str(url) for url in request.selected_urls],
        )
    except Exception as exc:
        logger.exception("ingest.groups.preparation_failed sitemap_url=%s error=%s", request.sitemap_url, exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    ingested_items: list[dict] = []
    failed_items: list[dict] = []
    skipped_items: list[dict] = []
    for url in urls_to_ingest:
        try:
            logger.info("ingest.groups.url.start url=%s", url)
            ingest_item = _run_single_url_ingestion(
                url=url,
                skip_scrape_if_exists=request.skip_scrape_if_exists,
            )
            if ingest_item.get("status") == "skipped":
                logger.info("ingest.groups.url.skipped url=%s reason=%s", url, ingest_item.get("reason"))
                skipped_items.append(ingest_item)
            else:
                logger.info("ingest.groups.url.ingested url=%s", url)
                ingested_items.append(ingest_item)
        except Exception as exc:
            logger.warning("ingest.groups.url.failed url=%s error=%s", url, exc)
            failed_items.append(
                {
                    "url": url,
                    "error": str(exc),
                }
            )

    logger.info(
        "ingest.groups.done sitemap_url=%s total=%s ingested=%s skipped=%s failed=%s",
        request.sitemap_url,
        len(urls_to_ingest),
        len(ingested_items),
        len(skipped_items),
        len(failed_items),
    )
    return {
        "sitemap_url": str(request.sitemap_url),
        "selected_group_ids": request.selected_group_ids,
        "selected_urls_count": len(urls_to_ingest),
        "ingested_count": len(ingested_items),
        "skipped_count": len(skipped_items),
        "failed_count": len(failed_items),
        "items": ingested_items,
        "skipped_items": skipped_items,
        "failed_items": failed_items,
    }


def _run_single_url_ingestion(url: str, skip_scrape_if_exists: bool = False) -> dict:
    """Run crawl + cleaning + markdown + metadata for one URL."""
    logger.info("ingest.pipeline.start url=%s", url)
    if skip_scrape_if_exists:
        logger.info("ingest.pipeline.skip_check url=%s", url)
        existing_paths = content_storage_service.get_existing_artifacts(source_url=url)
        if existing_paths and existing_paths.markdown_path:
            logger.info("ingest.pipeline.skip_hit url=%s markdown_path=%s", url, existing_paths.markdown_path)
            return {
                "source_url": url,
                "status": "skipped",
                "reason": "existing_artifacts_found",
                "stored_paths": {
                    "raw_html_path": existing_paths.raw_html_path,
                    "cleaned_html_path": existing_paths.cleaned_html_path,
                    "markdown_path": existing_paths.markdown_path,
                    "metadata_path": existing_paths.metadata_path,
                },
            }

    logger.info("ingest.pipeline.crawl.start url=%s", url)
    crawl_result = crawler_service.crawl_url(url=url)
    logger.info("ingest.pipeline.crawl.done url=%s status_code=%s", crawl_result.url, crawl_result.status_code)
    logger.info("ingest.pipeline.clean.start url=%s", crawl_result.url)
    cleaned_html = html_cleaner.clean_html_content(raw_html=crawl_result.html)
    logger.info("ingest.pipeline.clean.done url=%s cleaned_length=%s", crawl_result.url, len(cleaned_html))
    logger.info("ingest.pipeline.markdown.start url=%s", crawl_result.url)
    markdown_content = markdown_converter.convert_html_to_markdown(cleaned_html=cleaned_html)
    logger.info(
        "ingest.pipeline.markdown.done url=%s markdown_length=%s",
        crawl_result.url,
        len(markdown_content),
    )
    logger.info("ingest.pipeline.metadata.start url=%s", crawl_result.url)
    generated_metadata = metadata_generator.generate_metadata(
        source_url=crawl_result.url,
        page_title=crawl_result.title,
        markdown_content=markdown_content,
    )
    logger.info("ingest.pipeline.metadata.done url=%s", crawl_result.url)
    logger.info("ingest.pipeline.storage.start url=%s", crawl_result.url)
    stored_paths = content_storage_service.save_ingestion_artifacts(
        source_url=crawl_result.url,
        raw_html=crawl_result.html,
        cleaned_html=cleaned_html,
        markdown_content=markdown_content,
        metadata=generated_metadata,
    )
    logger.info("ingest.pipeline.storage.done url=%s", crawl_result.url)
    enrichment_path: str | None = None
    try:
        enrichment_path = str(
            enrichment_service.persist_enrichment(
                markdown_path=Path(stored_paths.markdown_path),
                base_metadata=generated_metadata,
                markdown_content=markdown_content,
            )
        )
    except Exception as exc:
        logger.warning("ingest.pipeline.enrichment_failed url=%s error=%s", crawl_result.url, exc)

    return {
        "source_url": crawl_result.url,
        "status": "ingested",
        "title": crawl_result.title,
        "metadata": generated_metadata,
        "markdown_preview": markdown_content[:1200],
        "markdown_length": len(markdown_content),
        "stored_paths": {
            "raw_html_path": stored_paths.raw_html_path,
            "cleaned_html_path": stored_paths.cleaned_html_path,
            "markdown_path": stored_paths.markdown_path,
            "metadata_path": stored_paths.metadata_path,
            "enrichment_path": enrichment_path,
        },
    }


def _select_urls_for_group_ingestion(
    sitemap_groups: list[SitemapGroup],
    selected_group_ids: list[str],
    selected_urls: list[str],
) -> list[str]:
    """Resolve ingest target URLs from selected group IDs and explicit URLs."""
    collected_urls: list[str] = []
    groups_by_id = {group.group_id: group for group in sitemap_groups}

    for group_id in selected_group_ids:
        group = groups_by_id.get(group_id)
        if group is None:
            raise ValueError(f"Unknown group_id: {group_id}")
        collected_urls.extend(group.urls)

    collected_urls.extend(selected_urls)
    deduplicated_urls = list(dict.fromkeys(collected_urls))
    if not deduplicated_urls:
        raise ValueError("No URLs selected for ingestion.")
    return deduplicated_urls
