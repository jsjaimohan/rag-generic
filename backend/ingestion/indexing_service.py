"""Step 8: embed semantic chunk manifests and upsert into Qdrant."""

from __future__ import annotations

import threading
import time
from pathlib import Path

from qdrant_client.models import PointStruct

from backend.embeddings.embedding_service import embedding_service
from backend.ingestion.chunk_schema import ChunkManifest, ChunkRecord
from backend.ingestion.indexing_jobs import (
    complete_indexing_job,
    fail_indexing_job,
    update_indexing_job,
)
from backend.settings import settings
from backend.utils.logger import get_logger
from backend.vectordb.qdrant_service import qdrant_service

logger = get_logger(__name__)


def _manifest_stem(manifest_path: Path) -> str:
    name = manifest_path.name
    if name.endswith(".chunks.json"):
        return name[: -len(".chunks.json")]
    return manifest_path.stem


def _chunk_payload(manifest: ChunkManifest, record: ChunkRecord, manifest_path: Path) -> dict:
    return {
        "chunk_id": record.chunk_id,
        "document_id": manifest.document_id,
        "source_url": manifest.source_url,
        "title": manifest.title,
        "section_title": record.section_title,
        "chunk_index": record.chunk_index,
        "heading_level": record.heading_level,
        "text": record.text,
        "manifest_stem": _manifest_stem(manifest_path),
        "chunking_policy_version": manifest.chunking_policy.version,
        "relationships": [rel.model_dump() for rel in record.relationships],
    }


def run_indexing_job(
    job_id: str,
    manifest_paths: list[Path],
    recreate_collection: bool,
) -> None:
    logger.info(
        "Step 8 — Persist vector embeddings: worker START job_id=%s manifests=%s recreate_collection=%s",
        job_id,
        len(manifest_paths),
        recreate_collection,
    )
    try:
        # 1) Disk: snapshot to HF cache (resume-friendly). 2) RAM: SentenceTransformer. 3) Qdrant. 4) Upsert.
        update_indexing_job(
            job_id,
            phase="downloading_model",
            current_file=settings.embedding_model_name,
        )
        logger.info(
            "Step 8 — (1/4) Download / verify model weights on disk job_id=%s repo=%s",
            job_id,
            settings.embedding_model_name,
        )
        ev_download = threading.Event()

        def _heartbeat_download() -> None:
            start = time.monotonic()
            while not ev_download.wait(timeout=45):
                logger.info(
                    "Step 8 — Still downloading model files job_id=%s elapsed_s=%s",
                    job_id,
                    int(time.monotonic() - start),
                )

        threading.Thread(target=_heartbeat_download, daemon=True).start()
        try:
            embedding_service.ensure_weights_on_disk()
        finally:
            ev_download.set()

        update_indexing_job(
            job_id,
            phase="loading_model",
            current_file=settings.embedding_model_name,
        )
        logger.info("Step 8 — (2/4) Load model into memory job_id=%s", job_id)
        ev_load = threading.Event()

        def _heartbeat_load() -> None:
            start = time.monotonic()
            while not ev_load.wait(timeout=45):
                logger.info(
                    "Step 8 — Still loading model into memory job_id=%s elapsed_s=%s",
                    job_id,
                    int(time.monotonic() - start),
                )

        threading.Thread(target=_heartbeat_load, daemon=True).start()
        try:
            dim = embedding_service.get_embedding_dimension()
        finally:
            ev_load.set()
        if dim != settings.qdrant_vector_size:
            raise ValueError(
                f"Embedding dimension {dim} does not match QDRANT_VECTOR_SIZE={settings.qdrant_vector_size} "
                f"(model {settings.embedding_model_name})."
            )

        update_indexing_job(job_id, phase="qdrant", current_file="ensure_collection")
        logger.info(
            "Step 8 — (3/4) Ensure Qdrant collection job_id=%s recreate_collection=%s",
            job_id,
            recreate_collection,
        )
        qdrant_service.ensure_collection(recreate=recreate_collection)

        logger.info(
            "Step 8 — (4/4) Encode + upsert job_id=%s model=%s dim=%s manifest_files=%s",
            job_id,
            settings.embedding_model_name,
            dim,
            len(manifest_paths),
        )
        update_indexing_job(job_id, phase="embedding", current_file="", points_indexed=0)

        manifests_ok = 0
        manifests_failed = 0
        points_total = 0
        points_buffer: list[PointStruct] = []
        batch_max = settings.qdrant_upsert_batch_size

        def flush_buffer() -> None:
            nonlocal points_buffer, points_total
            if not points_buffer:
                return
            qdrant_service.upsert_points(points_buffer)
            points_total += len(points_buffer)
            update_indexing_job(job_id, points_indexed=points_total)
            points_buffer = []

        for idx, mpath in enumerate(manifest_paths):
            update_indexing_job(job_id, processed_files=idx, current_file=mpath.name)
            try:
                raw = mpath.read_text(encoding="utf-8", errors="ignore")
                manifest = ChunkManifest.model_validate_json(raw)
            except Exception:
                logger.exception("indexing.manifest_invalid path=%s", mpath)
                manifests_failed += 1
                update_indexing_job(job_id, processed_files=idx + 1, current_file="")
                continue

            texts = [c.text for c in manifest.chunks]
            if not texts:
                manifests_ok += 1
                update_indexing_job(job_id, processed_files=idx + 1, current_file="")
                continue

            try:
                vectors = embedding_service.encode(texts)
            except Exception:
                logger.exception("indexing.encode_failed path=%s", mpath)
                manifests_failed += 1
                update_indexing_job(job_id, processed_files=idx + 1, current_file="")
                continue

            if len(vectors) != len(manifest.chunks):
                logger.error(
                    "indexing.encode_mismatch path=%s chunks=%s vectors=%s",
                    mpath,
                    len(manifest.chunks),
                    len(vectors),
                )
                manifests_failed += 1
                update_indexing_job(job_id, processed_files=idx + 1, current_file="")
                continue

            for record, vec in zip(manifest.chunks, vectors, strict=True):
                points_buffer.append(
                    PointStruct(
                        id=record.chunk_id,
                        vector=vec,
                        payload=_chunk_payload(manifest, record, mpath),
                    )
                )
                if len(points_buffer) >= batch_max:
                    flush_buffer()

            manifests_ok += 1
            update_indexing_job(job_id, processed_files=idx + 1, current_file="")

        flush_buffer()

        logger.info(
            "Step 8 — Embedding: END (success) job_id=%s points_upserted=%s manifests_ok=%s manifests_failed=%s",
            job_id,
            points_total,
            manifests_ok,
            manifests_failed,
        )

        complete_indexing_job(
            job_id,
            {
                "manifests_processed": manifests_ok,
                "manifests_failed": manifests_failed,
                "points_upserted": points_total,
            },
        )
        logger.info(
            "Step 8 — Persist vector embeddings: worker END (completed) job_id=%s",
            job_id,
        )
    except KeyboardInterrupt:
        msg = "Interrupted (Ctrl+C or server shutdown) during model download or indexing."
        logger.warning("indexing.job_interrupted job_id=%s", job_id)
        logger.info("Step 8 — Embedding: END (interrupted) job_id=%s", job_id)
        logger.info("Step 8 — Persist vector embeddings: worker END (interrupted) job_id=%s", job_id)
        fail_indexing_job(job_id, msg)
    except Exception as exc:
        logger.exception("indexing.job_failed job_id=%s", job_id)
        logger.info(
            "Step 8 — Embedding: END (failed) job_id=%s error=%s",
            job_id,
            exc,
        )
        logger.info(
            "Step 8 — Persist vector embeddings: worker END (failed) job_id=%s",
            job_id,
        )
        fail_indexing_job(job_id, str(exc))


def start_indexing_background(
    job_id: str,
    manifest_paths: list[Path],
    recreate_collection: bool,
) -> None:
    thread = threading.Thread(
        target=run_indexing_job,
        args=(job_id, manifest_paths, recreate_collection),
        daemon=True,
    )
    thread.start()
