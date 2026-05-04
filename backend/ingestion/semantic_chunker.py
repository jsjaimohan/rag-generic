"""Section-aware semantic chunking with code/table cohesion and overlap (Step 7)."""

from __future__ import annotations

import re
import uuid
from datetime import UTC, datetime

from backend.ingestion.chunk_schema import (
    ChunkManifest,
    ChunkQualityMetrics,
    ChunkRecord,
    ChunkRelationship,
    ChunkingPolicy,
)
from backend.settings import settings


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


def split_by_headings(content: str) -> list[tuple[int | None, str, list[str]]]:
    """Split markdown into (heading_level, section_title, lines_including_heading)."""
    lines = content.split("\n")
    sections: list[tuple[int | None, str, list[str]]] = []
    buffer: list[str] = []
    section_title = ""
    level: int | None = None

    def flush_section(title: str, lvl: int | None) -> None:
        if buffer:
            sections.append((lvl, title, list(buffer)))

    for line in lines:
        heading_match = re.match(r"^(#{1,6})\s+(.*)$", line)
        if heading_match:
            flush_section(section_title, level)
            level = len(heading_match.group(1))
            section_title = heading_match.group(2).strip()
            buffer = [line]
        else:
            buffer.append(line)

    flush_section(section_title, level)
    if not sections and content.strip():
        return [(None, "", content.split("\n"))]
    return sections


def segment_section_lines(lines: list[str]) -> list[str]:
    """Atomic segments: fenced blocks, pipe tables, prose paragraphs."""
    segments: list[str] = []
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith("```"):
            block_lines = [line]
            i += 1
            while i < n:
                block_lines.append(lines[i])
                if lines[i].strip().startswith("```") and len(block_lines) > 1:
                    i += 1
                    break
                i += 1
            text = "\n".join(block_lines).strip()
            if text:
                segments.append(text)
            continue
        if stripped.startswith("|") and stripped.count("|") >= 2:
            block_lines = [line]
            i += 1
            while i < n and lines[i].strip() and "|" in lines[i]:
                block_lines.append(lines[i])
                i += 1
            text = "\n".join(block_lines).strip()
            if text:
                segments.append(text)
            continue
        block_lines: list[str] = []
        while i < n and lines[i].strip():
            block_lines.append(lines[i])
            i += 1
        while i < n and not lines[i].strip():
            i += 1
        text = "\n".join(block_lines).strip()
        if text:
            segments.append(text)
    return segments


def _hard_split_oversized(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text] if text else []
    parts: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        if end < n:
            newline_break = text.rfind("\n", start, end)
            if newline_break != -1 and newline_break - start > max_chars // 2:
                end = newline_break + 1
        piece = text[start:end].strip()
        if piece:
            parts.append(piece)
        if end >= n:
            break
        start = max(start + 1, end - overlap_chars)
    return parts


def pack_segments(
    segments: list[str],
    max_chars: int,
    overlap_chars: int,
) -> list[str]:
    """Pack atomic segments into chunks without splitting segments when possible."""
    chunks: list[str] = []
    buffer: list[str] = []
    buffer_len = 0

    def flush() -> None:
        nonlocal buffer, buffer_len
        if not buffer:
            return
        chunks.append("\n\n".join(buffer))
        buffer = []
        buffer_len = 0

    for seg in segments:
        for piece in _hard_split_oversized(seg, max_chars, overlap_chars):
            sep = 2 if buffer else 0
            if buffer_len + sep + len(piece) <= max_chars:
                buffer.append(piece)
                buffer_len += sep + len(piece)
            else:
                flush()
                overlap_prefix = ""
                if chunks and overlap_chars > 0:
                    prev = chunks[-1]
                    if len(prev) > overlap_chars:
                        overlap_prefix = prev[-overlap_chars:].lstrip()
                if overlap_prefix and len(overlap_prefix) + 2 + len(piece) <= max_chars:
                    buffer = [overlap_prefix, piece]
                    buffer_len = len(overlap_prefix) + 2 + len(piece)
                else:
                    buffer = [piece]
                    buffer_len = len(piece)
    flush()
    return [c for c in chunks if c.strip()]


def stable_document_id(source_url: str, markdown_path: str, policy_version: str) -> str:
    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"doc|{source_url}|{markdown_path}|{policy_version}",
        )
    )


def stable_chunk_id(document_id: str, chunk_index: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"chunk|{document_id}|{chunk_index}"))


def _resolve_hint_to_chunk_id(
    hint: str,
    chunks: list[ChunkRecord],
) -> str | None:
    hint_lower = hint.lower().strip()
    if len(hint_lower) < 2:
        return None
    title_hits = [
        c.chunk_id
        for c in chunks
        if hint_lower in (c.section_title or "").lower() and (c.section_title or "").strip()
    ]
    if len(title_hits) == 1:
        return title_hits[0]
    text_hits = [
        c.chunk_id
        for c in chunks
        if hint_lower in c.text.lower()[: min(2000, len(c.text))]
    ]
    if len(text_hits) == 1:
        return text_hits[0]
    return None


def build_chunk_manifest(
    *,
    markdown_path: str,
    markdown_content: str,
    source_url: str,
    page_title: str,
    enrichment_relationships: list[dict] | None,
    max_chunk_chars: int | None = None,
    overlap_chars: int | None = None,
    policy_version: str | None = None,
) -> ChunkManifest:
    """Build full manifest: headings → segments → packed chunks + graph hints."""
    max_c = max_chunk_chars if max_chunk_chars is not None else settings.chunk_max_chars
    overlap = overlap_chars if overlap_chars is not None else settings.chunk_overlap_chars
    policy_ver = policy_version or settings.chunking_policy_version

    cleaned = _strip_low_signal_lines(markdown_content)
    sections = split_by_headings(cleaned)

    document_id = stable_document_id(
        source_url=source_url,
        markdown_path=markdown_path,
        policy_version=policy_ver,
    )

    chunk_records: list[ChunkRecord] = []
    chunk_index = 0

    for heading_level, section_title, section_lines in sections:
        if not section_lines:
            continue
        segments = segment_section_lines(section_lines)
        if not segments:
            continue
        packed_texts = pack_segments(segments, max_chars=max_c, overlap_chars=overlap)
        for text in packed_texts:
            chunk_id = stable_chunk_id(document_id, chunk_index)
            chunk_records.append(
                ChunkRecord(
                    chunk_id=chunk_id,
                    chunk_index=chunk_index,
                    section_title=section_title,
                    heading_level=heading_level,
                    text=text,
                    relationships=[],
                )
            )
            chunk_index += 1

    for idx, record in enumerate(chunk_records):
        if idx < len(chunk_records) - 1:
            nxt = chunk_records[idx + 1]
            record.relationships.append(
                ChunkRelationship(
                    relation="continuation",
                    target_chunk_id=nxt.chunk_id,
                )
            )

    if enrichment_relationships and chunk_records:
        for edge in enrichment_relationships:
            if not isinstance(edge, dict):
                continue
            rel = str(edge.get("relation", "")).strip()
            hint = str(edge.get("target_hint", "")).strip()
            if not rel or not hint:
                continue
            target_id = _resolve_hint_to_chunk_id(hint, chunk_records)
            source_idx = None
            hint_lower = hint.lower()
            for i, chunk in enumerate(chunk_records):
                if hint_lower in chunk.text.lower() or hint_lower in (chunk.section_title or "").lower():
                    source_idx = i
                    break
            if source_idx is None:
                source_idx = 0
            chunk_records[source_idx].relationships.append(
                ChunkRelationship(
                    relation=rel,
                    target_chunk_id=target_id,
                    target_hint=None if target_id else hint,
                )
            )

    edge_count = sum(len(c.relationships) for c in chunk_records)
    lengths = [len(c.text) for c in chunk_records]
    orphan_count = sum(
        1
        for c in chunk_records
        if not (c.section_title or "").strip() and c.heading_level is None
    )

    quality = ChunkQualityMetrics(
        chunk_count=len(chunk_records),
        avg_chunk_chars=sum(lengths) / len(lengths) if lengths else 0.0,
        min_chunk_chars=min(lengths) if lengths else 0,
        max_chunk_chars=max(lengths) if lengths else 0,
        orphan_chunk_count=orphan_count,
        relationship_edge_count=edge_count,
        relationship_density=(edge_count / len(chunk_records)) if chunk_records else 0.0,
    )

    return ChunkManifest(
        markdown_path=markdown_path,
        source_url=source_url,
        title=page_title,
        document_id=document_id,
        chunking_policy=ChunkingPolicy(
            version=policy_ver,
            max_chunk_chars=max_c,
            overlap_chars=overlap,
        ),
        generated_at_utc=datetime.now(UTC).isoformat(),
        chunks=chunk_records,
        quality=quality,
    )
