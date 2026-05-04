"""Step 9 — hybrid retrieval (dense + keyword on pool), expansion, rerank, ContextBundle."""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field

from backend.embeddings.embedding_service import embedding_service
from backend.ingestion.chunk_schema import ChunkManifest
from backend.rag.context_bundle import (
    ContextBundleItem,
    RetrievalConfidenceInputs,
    RetrievalTraceEntry,
    RetrievalTraces,
)
from backend.reranker.reranker_service import reranker_service
from backend.settings import resolve_data_path, settings
from backend.utils.logger import get_logger
from backend.vectordb.qdrant_service import QdrantSearchHit, qdrant_service

logger = get_logger(__name__)

# Match legacy MiniRAG filename stem boost.
PATH_BOOST_PER_TERM = 5


@dataclass
class RetrievalOutcome:
    """Result of Step 9 retrieval for MiniRAG /chat."""

    items: list[ContextBundleItem]
    mode: str
    traces: RetrievalTraces
    confidence_inputs: RetrievalConfidenceInputs
    # Seconds per sub-step (perf_counter), for /chat performance debugging.
    timings_s: dict[str, float] = field(default_factory=dict)


def _extract_terms(query: str) -> list[str]:
    return [term for term in re.findall(r"\w+", query.lower()) if len(term) >= 3]


def _path_boost_score(file_stem: str, terms: list[str]) -> int:
    boost = 0
    for term in terms:
        if term and term in file_stem:
            boost += PATH_BOOST_PER_TERM
    return boost


def _keyword_overlap_score(text: str, terms: list[str]) -> int:
    if not terms:
        return 0
    lowered = text.lower()
    return sum(lowered.count(term) for term in terms)


def _rerank_input_cap(top_k: int, candidate_count: int) -> int:
    """Max passages to score with the cross-encoder; never below ``top_k``."""
    limit = max(1, settings.retrieval_rerank_input_max)
    return min(candidate_count, max(top_k, limit))


def _passages_for_rerank(items: list[ContextBundleItem]) -> list[str]:
    return [f"{item.section_title}\n{item.text}"[:4000] for item in items]


def _try_cross_encoder_rerank(query: str, items: list[ContextBundleItem]) -> bool:
    """
    Score and re-order ``items`` with the cross-encoder.

    Returns True if reranking ran successfully. On load/inference failure, logs a warning
    and returns False so callers keep the existing order (hybrid / keyword).
    """
    if not settings.retrieval_enable_reranker or len(items) <= 1:
        return True
    try:
        r_scores = reranker_service.predict_scores(query, _passages_for_rerank(items))
        for item, rs in zip(items, r_scores, strict=True):
            item.reranker_score = rs
        items.sort(
            key=lambda x: x.reranker_score if x.reranker_score is not None else 0.0,
            reverse=True,
        )
        return True
    except Exception as exc:
        logger.warning(
            "retrieval.rerank_failed — continuing without cross-encoder scores: %s",
            exc,
        )
        return False


def _normalize_scores(raw: list[float]) -> list[float]:
    if not raw:
        return []
    lo, hi = min(raw), max(raw)
    if hi <= lo:
        return [1.0 for _ in raw]
    return [(x - lo) / (hi - lo) for x in raw]


def _payload_relationship_targets(payload: dict) -> list[str]:
    targets: list[str] = []
    for rel in payload.get("relationships") or []:
        if not isinstance(rel, dict):
            continue
        tid = rel.get("target_chunk_id")
        if tid:
            targets.append(str(tid))
    return targets


def _hit_to_bundle_item(
    *,
    point_id: str,
    payload: dict,
    dense_score: float | None,
    keyword_score: float | None,
    hybrid_score: float | None,
    reranker_score: float | None,
    relationship_expansion: bool,
) -> ContextBundleItem | None:
    text = (payload.get("text") or "").strip()
    if not text:
        return None
    chunk_id = str(payload.get("chunk_id") or point_id)
    return ContextBundleItem(
        chunk_id=chunk_id,
        source_url=str(payload.get("source_url") or ""),
        document_id=str(payload.get("document_id") or ""),
        manifest_stem=str(payload.get("manifest_stem") or ""),
        section_title=str(payload.get("section_title") or ""),
        text=text,
        dense_score=dense_score,
        keyword_score=keyword_score,
        hybrid_score=hybrid_score,
        reranker_score=reranker_score,
        relationship_expansion=relationship_expansion,
    )


def _merge_hybrid_scores(
    hits: list[QdrantSearchHit],
    terms: list[str],
) -> list[tuple[QdrantSearchHit, float, float, float]]:
    """Return tuples of (hit, dense, kw_raw, hybrid) for each hit."""
    dense_raw = [h.score for h in hits]
    dense_norm = _normalize_scores(dense_raw)
    kw_raw: list[int] = []
    for h in hits:
        payload = h.payload
        blob = f"{payload.get('section_title') or ''}\n{payload.get('text') or ''}"
        kw_raw.append(_keyword_overlap_score(blob, terms))
    kw_norm = _normalize_scores([float(x) for x in kw_raw])
    w = settings.retrieval_hybrid_keyword_weight
    merged: list[tuple[QdrantSearchHit, float, float, float]] = []
    for idx, h in enumerate(hits):
        hybrid = (1.0 - w) * dense_norm[idx] + w * kw_norm[idx]
        merged.append((h, dense_raw[idx], float(kw_raw[idx]), hybrid))
    merged.sort(key=lambda x: x[3], reverse=True)
    return merged


def _expand_with_relationships(
    merged: list[tuple[QdrantSearchHit, float, float, float]],
    *,
    rerank_pool_cap: int,
) -> list[tuple[str, dict, float, float, float, bool]]:
    """
    Produce rows: point_id, payload, dense, kw, hybrid, rel_expansion.
    Starts from hybrid-sorted hits, takes top rerank_pool_cap, fetches related points.
    """
    seen: set[str] = set()
    rows: list[tuple[str, dict, float, float, float, bool]] = []

    top_slice = merged[:rerank_pool_cap]
    for h, d_raw, k_raw, hyb in top_slice:
        pid = h.point_id
        if pid in seen:
            continue
        seen.add(pid)
        rows.append((pid, h.payload, d_raw, k_raw, hyb, False))

    max_expand = settings.retrieval_relationship_expand_max
    parent_hybrid_by_target: dict[str, float] = {}
    for h, _, _, hyb in top_slice:
        for tid in _payload_relationship_targets(h.payload):
            if not tid or tid in seen:
                continue
            prev = parent_hybrid_by_target.get(tid)
            if prev is None or hyb > prev:
                parent_hybrid_by_target[tid] = hyb

    ordered_targets = sorted(
        parent_hybrid_by_target.keys(),
        key=lambda t: parent_hybrid_by_target[t],
        reverse=True,
    )
    to_fetch = ordered_targets[:max_expand]
    if to_fetch:
        fetched = qdrant_service.retrieve_payloads_by_ids(to_fetch)
        for rec in fetched:
            if rec.point_id in seen:
                continue
            hyb = parent_hybrid_by_target.get(rec.point_id, 0.0) * 0.92
            seen.add(rec.point_id)
            rows.append((rec.point_id, rec.payload, None, None, hyb, True))

    return rows


def retrieve_for_query(query: str, top_k: int) -> RetrievalOutcome:
    """
    Primary: dense search (bounded by ``retrieval_vector_pool_size``) → hybrid scores →
    relationship expansion → sort by hybrid → **cross-encoder on at most**
    ``retrieval_rerank_input_max`` (and never below ``top_k``) → return ``top_k``.

    Fallback keyword paths use the same CE cap. See settings: ``RETRIEVAL_RERANK_INPUT_MAX``.
    """
    t_rq = time.perf_counter()
    terms = _extract_terms(query)
    if not terms and query.strip():
        terms = [query.lower().strip()]

    t_ct = time.perf_counter()
    try:
        n_points = qdrant_service.count_points()
    except Exception as exc:
        logger.warning("retrieval.qdrant_count_failed error=%s", exc)
        n_points = 0
    count_points_s = time.perf_counter() - t_ct

    if n_points > 0:
        try:
            out = _retrieve_qdrant_path(query, top_k, terms)
            out.timings_s["qdrant_count_points_s"] = round(count_points_s, 4)
            out.timings_s["retrieve_for_query_total_s"] = round(
                time.perf_counter() - t_rq, 4
            )
            return out
        except Exception as exc:
            logger.warning(
                "retrieval.vector_pipeline_failed — falling back to keyword manifests "
                "(embedding/Qdrant error; reranker errors are handled inside the vector path): %s",
                exc,
            )
    out = _retrieve_keyword_manifests_path(query, top_k, terms)
    out.timings_s["qdrant_count_points_s"] = round(count_points_s, 4)
    out.timings_s["retrieve_for_query_total_s"] = round(time.perf_counter() - t_rq, 4)
    return out


def _retrieve_qdrant_path(
    query: str,
    top_k: int,
    terms: list[str],
) -> RetrievalOutcome:
    timings: dict[str, float] = {}
    path_t0 = time.perf_counter()
    traces = RetrievalTraces()

    t = time.perf_counter()
    qvec = embedding_service.encode([query])[0]
    timings["embed_query_s"] = round(time.perf_counter() - t, 4)

    t = time.perf_counter()
    hits = qdrant_service.search_dense(
        qvec, limit=settings.retrieval_vector_pool_size
    )
    timings["qdrant_dense_search_s"] = round(time.perf_counter() - t, 4)

    if not hits:
        out = _retrieve_keyword_manifests_path(query, top_k, terms)
        out.timings_s = {
            **timings,
            **out.timings_s,
            "retrieve_vector_path_inner_s": round(time.perf_counter() - path_t0, 4),
            "branch": "empty_hits_keyword_fallback",
        }
        return out

    t = time.perf_counter()
    merged = _merge_hybrid_scores(hits, terms)
    timings["hybrid_merge_s"] = round(time.perf_counter() - t, 4)
    trace_cap = settings.retrieval_trace_max_entries
    traces.pre_rerank = [
        RetrievalTraceEntry(
            chunk_id=str(h.payload.get("chunk_id") or h.point_id),
            score=hyb,
            source_url=str(h.payload.get("source_url") or ""),
            manifest_stem=str(h.payload.get("manifest_stem") or ""),
        )
        for h, _, _, hyb in merged[:trace_cap]
    ]

    t = time.perf_counter()
    rows = _expand_with_relationships(
        merged,
        rerank_pool_cap=settings.retrieval_rerank_pool_size,
    )
    timings["relationship_expand_s"] = round(time.perf_counter() - t, 4)

    items_intermediate: list[ContextBundleItem] = []

    t = time.perf_counter()
    for point_id, payload, d_raw, k_raw, hyb, rel_x in rows:
        item = _hit_to_bundle_item(
            point_id=point_id,
            payload=payload,
            dense_score=float(d_raw) if d_raw is not None else None,
            keyword_score=float(k_raw) if k_raw is not None else None,
            hybrid_score=hyb,
            reranker_score=None,
            relationship_expansion=rel_x,
        )
        if item is None:
            continue
        items_intermediate.append(item)
    timings["bundle_candidates_s"] = round(time.perf_counter() - t, 4)

    if not items_intermediate:
        out = _retrieve_keyword_manifests_path(query, top_k, terms)
        out.timings_s = {
            **timings,
            **out.timings_s,
            "retrieve_vector_path_inner_s": round(time.perf_counter() - path_t0, 4),
            "branch": "no_bundle_items_keyword_fallback",
        }
        return out

    conf = RetrievalConfidenceInputs(max_hybrid_score=merged[0][3] if merged else None)

    items_intermediate.sort(
        key=lambda x: x.hybrid_score if x.hybrid_score is not None else 0.0,
        reverse=True,
    )
    ce_cap = _rerank_input_cap(top_k, len(items_intermediate))
    to_rerank = items_intermediate[:ce_cap]

    rerank_applied = False
    t = time.perf_counter()
    if settings.retrieval_enable_reranker and len(to_rerank) > 1:
        rerank_applied = _try_cross_encoder_rerank(query, to_rerank)
    timings["cross_encoder_rerank_s"] = round(time.perf_counter() - t, 4)
    ranked_for_trace = to_rerank

    traces.post_rerank = [
        RetrievalTraceEntry(
            chunk_id=item.chunk_id,
            score=float(item.reranker_score or item.hybrid_score or 0.0),
            source_url=item.source_url,
            manifest_stem=item.manifest_stem,
        )
        for item in ranked_for_trace[:trace_cap]
    ]

    final_items = ranked_for_trace[:top_k]
    if final_items:
        scores = [float(i.reranker_score or i.hybrid_score or 0.0) for i in final_items]
        conf.max_rerank_score = max(scores)
        if len(scores) >= 2:
            top2 = sorted(scores, reverse=True)
            conf.score_margin_top1_top2 = top2[0] - top2[1]

    mode = (
        "vector_hybrid_rerank"
        if settings.retrieval_enable_reranker and rerank_applied
        else "vector_hybrid"
    )
    timings["retrieve_vector_path_inner_s"] = round(time.perf_counter() - path_t0, 4)
    timings["branch"] = "vector_qdrant"
    return RetrievalOutcome(
        items=final_items,
        mode=mode,
        traces=traces,
        confidence_inputs=conf,
        timings_s=timings,
    )


def _legacy_markdown_chunk_id(source_path: str, section_title: str) -> str:
    digest = hashlib.sha256(
        f"{source_path}|{section_title}".encode("utf-8", errors="ignore")
    ).hexdigest()[:22]
    return f"kw-md-{digest}"


def _dedupe_max_per_file(
    items: list[ContextBundleItem],
    *,
    max_items: int,
    max_per_file: int = 2,
) -> list[ContextBundleItem]:
    """Prefer diversity across sources (same behavior as pre-Step-9 MiniRAG)."""
    picked: list[ContextBundleItem] = []
    per_file: dict[str, int] = {}
    for item in items:
        if len(picked) >= max_items:
            break
        file_key = item.document_id
        count = per_file.get(file_key, 0)
        if count >= max_per_file:
            continue
        per_file[file_key] = count + 1
        picked.append(item)
    return picked


def _retrieve_processing_markdown_legacy(
    query: str,
    top_k: int,
    terms: list[str],
) -> RetrievalOutcome:
    """Last-resort keyword retrieval over `data/processing/*.md` (synthetic chunk ids)."""
    t_in = time.perf_counter()
    processing_dir = resolve_data_path(settings.data_processing_dir)
    markdown_files = sorted(processing_dir.glob("*.md"))
    traces = RetrievalTraces()
    if not markdown_files:
        return RetrievalOutcome(
            items=[],
            mode="keyword_markdown",
            traces=traces,
            confidence_inputs=RetrievalConfidenceInputs(),
            timings_s={
                "keyword_markdown_inner_s": round(time.perf_counter() - t_in, 4),
                "branch": "keyword_markdown_empty_dir",
            },
        )

    scored: list[ContextBundleItem] = []
    for markdown_file in markdown_files:
        path_boost = _path_boost_score(markdown_file.stem.lower(), terms)
        content = markdown_file.read_text(encoding="utf-8", errors="ignore")
        source_path = str(markdown_file.resolve())
        stem = markdown_file.stem
        for section_title, chunk_text in _iter_heading_chunks_processing(content):
            chunk_score = _keyword_overlap_score(chunk_text, terms)
            total = float(chunk_score + path_boost)
            if total <= 0:
                continue
            cid = _legacy_markdown_chunk_id(source_path, section_title)
            scored.append(
                ContextBundleItem(
                    chunk_id=cid,
                    source_url="",
                    document_id=source_path,
                    manifest_stem=stem,
                    section_title=section_title,
                    text=chunk_text,
                    dense_score=None,
                    keyword_score=total,
                    hybrid_score=total,
                    reranker_score=None,
                    relationship_expansion=False,
                )
            )

    scored.sort(key=lambda x: x.keyword_score or 0.0, reverse=True)
    trace_cap = settings.retrieval_trace_max_entries
    traces.pre_rerank = [
        RetrievalTraceEntry(
            chunk_id=x.chunk_id,
            score=float(x.keyword_score or 0.0),
            source_url=x.source_url,
            manifest_stem=x.manifest_stem,
        )
        for x in scored[:trace_cap]
    ]

    if not scored:
        return RetrievalOutcome(
            items=[],
            mode="keyword_markdown",
            traces=traces,
            confidence_inputs=RetrievalConfidenceInputs(),
            timings_s={
                "keyword_markdown_inner_s": round(time.perf_counter() - t_in, 4),
                "branch": "keyword_markdown_no_scores",
            },
        )

    head_n = min(len(scored), settings.retrieval_vector_pool_size)
    head = scored[:head_n]
    dedupe_budget = _rerank_input_cap(top_k, len(head))
    deduped = _dedupe_max_per_file(head, max_items=dedupe_budget, max_per_file=2)

    rerank_applied = False
    t_ce = time.perf_counter()
    if settings.retrieval_enable_reranker and len(deduped) > 1:
        ce_cap = _rerank_input_cap(top_k, len(deduped))
        to_rerank = deduped[:ce_cap]
        rerank_applied = _try_cross_encoder_rerank(query, to_rerank)
        final_items = _dedupe_max_per_file(to_rerank, max_items=top_k, max_per_file=2)
    else:
        final_items = _dedupe_max_per_file(head, max_items=top_k, max_per_file=2)
    ce_s = time.perf_counter() - t_ce

    traces.post_rerank = [
        RetrievalTraceEntry(
            chunk_id=item.chunk_id,
            score=float(item.reranker_score or item.keyword_score or 0.0),
            source_url=item.source_url,
            manifest_stem=item.manifest_stem,
        )
        for item in final_items[:trace_cap]
    ]
    scores = [float(i.reranker_score or i.keyword_score or 0.0) for i in final_items]
    conf = RetrievalConfidenceInputs(
        max_hybrid_score=float(scored[0].keyword_score) if scored else None,
        max_rerank_score=max(scores) if scores else None,
        score_margin_top1_top2=(sorted(scores, reverse=True)[0] - sorted(scores, reverse=True)[1])
        if len(scores) >= 2
        else None,
    )
    mode = "keyword_markdown_rerank" if rerank_applied else "keyword_markdown"
    return RetrievalOutcome(
        items=final_items,
        mode=mode,
        traces=traces,
        confidence_inputs=conf,
        timings_s={
            "keyword_markdown_inner_s": round(time.perf_counter() - t_in, 4),
            "cross_encoder_rerank_s": round(ce_s, 4),
            "branch": "keyword_markdown",
        },
    )


def _split_into_heading_chunks(content: str) -> list[tuple[str, str]]:
    lines = content.split("\n")
    chunks: list[tuple[str, str]] = []
    buffer: list[str] = []
    section_title = ""

    def flush_section(title: str) -> None:
        body = "\n".join(buffer).strip()
        buffer.clear()
        if body:
            chunks.append((title, body))

    for line in lines:
        heading_match = re.match(r"^(#{1,6})\s+(.*)$", line)
        if heading_match:
            flush_section(section_title)
            section_title = heading_match.group(2).strip()
            buffer = [line]
        else:
            buffer.append(line)

    flush_section(section_title)
    if not chunks and content.strip():
        return [("", content.strip())]
    return chunks


def _strip_low_signal_lines(text: str) -> str:
    kept: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            kept.append(line)
            continue
        if len(stripped) <= 2 and stripped in {"-", "*", "•"}:
            continue
        kept.append(line)
    return "\n".join(kept).strip()


def _iter_heading_chunks_processing(content: str):
    for section_title, chunk_text in _split_into_heading_chunks(content):
        stripped = _strip_low_signal_lines(chunk_text)
        if stripped:
            yield section_title, stripped


def _retrieve_keyword_manifests_path(
    query: str,
    top_k: int,
    terms: list[str],
) -> RetrievalOutcome:
    t_in = time.perf_counter()
    chunks_dir = resolve_data_path(settings.data_semantic_chunks_dir)
    paths = sorted(chunks_dir.glob("*.chunks.json"))
    scored: list[ContextBundleItem] = []

    for path in paths:
        try:
            manifest = ChunkManifest.model_validate_json(
                path.read_text(encoding="utf-8", errors="ignore")
            )
        except Exception:
            continue
        manifest_stem = path.name[: -len(".chunks.json")] if path.name.endswith(
            ".chunks.json"
        ) else path.stem
        for record in manifest.chunks:
            blob = f"{record.section_title}\n{record.text}"
            kw = float(_keyword_overlap_score(blob, terms))
            if kw <= 0:
                continue
            item = _hit_to_bundle_item(
                point_id=record.chunk_id,
                payload={
                    "chunk_id": record.chunk_id,
                    "source_url": manifest.source_url,
                    "document_id": manifest.document_id,
                    "manifest_stem": manifest_stem,
                    "section_title": record.section_title,
                    "text": record.text,
                },
                dense_score=None,
                keyword_score=kw,
                hybrid_score=kw,
                reranker_score=None,
                relationship_expansion=False,
            )
            if item:
                scored.append(item)

    keyword_manifest_disk_scan_s = round(time.perf_counter() - t_in, 4)
    scored.sort(key=lambda x: x.keyword_score or 0.0, reverse=True)
    trace_cap = settings.retrieval_trace_max_entries
    traces = RetrievalTraces()
    traces.pre_rerank = [
        RetrievalTraceEntry(
            chunk_id=x.chunk_id,
            score=float(x.keyword_score or 0.0),
            source_url=x.source_url,
            manifest_stem=x.manifest_stem,
        )
        for x in scored[:trace_cap]
    ]

    if not scored:
        out = _retrieve_processing_markdown_legacy(query, top_k, terms)
        out.timings_s["keyword_manifest_disk_scan_s"] = keyword_manifest_disk_scan_s
        return out

    head_n = min(len(scored), settings.retrieval_vector_pool_size)
    head = scored[:head_n]
    dedupe_budget = _rerank_input_cap(top_k, len(head))
    pool = _dedupe_max_per_file(head, max_items=dedupe_budget, max_per_file=2)

    rerank_applied = False
    t_ce = time.perf_counter()
    if settings.retrieval_enable_reranker and len(pool) > 1:
        ce_cap = _rerank_input_cap(top_k, len(pool))
        to_rerank = pool[:ce_cap]
        rerank_applied = _try_cross_encoder_rerank(query, to_rerank)
        final_items = _dedupe_max_per_file(to_rerank, max_items=top_k, max_per_file=2)
    else:
        final_items = _dedupe_max_per_file(head, max_items=top_k, max_per_file=2)
    ce_s = time.perf_counter() - t_ce

    traces.post_rerank = [
        RetrievalTraceEntry(
            chunk_id=item.chunk_id,
            score=float(item.reranker_score or item.keyword_score or 0.0),
            source_url=item.source_url,
            manifest_stem=item.manifest_stem,
        )
        for item in final_items[:trace_cap]
    ]

    scores = [float(i.reranker_score or i.keyword_score or 0.0) for i in final_items]
    conf = RetrievalConfidenceInputs(
        max_hybrid_score=float(scored[0].keyword_score) if scored else None,
        max_rerank_score=max(scores) if scores else None,
        score_margin_top1_top2=(sorted(scores, reverse=True)[0] - sorted(scores, reverse=True)[1])
        if len(scores) >= 2
        else None,
    )
    mode = "keyword_manifests_rerank" if rerank_applied else "keyword_manifests"
    inner_total = round(time.perf_counter() - t_in, 4)
    return RetrievalOutcome(
        items=final_items,
        mode=mode,
        traces=traces,
        confidence_inputs=conf,
        timings_s={
            "keyword_manifest_disk_scan_s": keyword_manifest_disk_scan_s,
            "cross_encoder_rerank_s": round(ce_s, 4),
            "keyword_manifest_inner_s": inner_total,
            "branch": "keyword_manifests_rerank"
            if rerank_applied
            else "keyword_manifests",
        },
    )
