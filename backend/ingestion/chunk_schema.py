"""Step 7 — validated schema for semantic chunks and graph hints."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChunkRelationship(BaseModel):
    """Directed edge for retrieval expansion (same-doc or soft cross-ref)."""

    relation: str = Field(..., min_length=1, max_length=64)
    target_chunk_id: str | None = None
    target_hint: str | None = Field(default=None, max_length=400)


class ChunkRecord(BaseModel):
    """One semantic chunk with stable id and traceable section context."""

    chunk_id: str
    chunk_index: int = Field(ge=0)
    section_title: str = ""
    heading_level: int | None = None
    text: str
    relationships: list[ChunkRelationship] = Field(default_factory=list)


class ChunkingPolicy(BaseModel):
    """Versioned parameters for reproducible re-chunking."""

    version: str
    max_chunk_chars: int
    overlap_chars: int


class ChunkQualityMetrics(BaseModel):
    """Lightweight quality signals for tuning chunking."""

    chunk_count: int = 0
    avg_chunk_chars: float = 0.0
    min_chunk_chars: int = 0
    max_chunk_chars: int = 0
    orphan_chunk_count: int = 0
    relationship_edge_count: int = 0
    relationship_density: float = 0.0


class ChunkManifest(BaseModel):
    """Per-markdown chunk dataset persisted as `{stem}.chunks.json` under `data/semantic-chunks`."""

    schema_version: int = 1
    markdown_path: str
    source_url: str = ""
    title: str = ""
    document_id: str
    chunking_policy: ChunkingPolicy
    generated_at_utc: str
    chunks: list[ChunkRecord]
    quality: ChunkQualityMetrics
