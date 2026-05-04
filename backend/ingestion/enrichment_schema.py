"""Validated schema for Step 6 AI enrichment artifacts."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class EntityRef(BaseModel):
    """Named entity or key concept grounded in page content."""

    name: str = Field(..., min_length=1, max_length=200)
    type: str = Field(default="concept", max_length=64)


class RelationshipEdge(BaseModel):
    """Soft link between this page and another topic (hint text only)."""

    relation: str = Field(..., min_length=1, max_length=64)
    target_hint: str = Field(..., min_length=1, max_length=300)


class EnrichmentPayload(BaseModel):
    """LLM-generated retrieval metadata; must stay auxiliary to source markdown."""

    short_summary: str = Field(default="", max_length=600)
    tags: list[str] = Field(default_factory=list)
    entities: list[EntityRef] = Field(default_factory=list)
    questions_answered: list[str] = Field(default_factory=list)
    relationships: list[RelationshipEdge] = Field(default_factory=list)
    content_intents: list[str] = Field(default_factory=list)

    @field_validator("tags", "questions_answered", "content_intents", mode="before")
    @classmethod
    def _cap_string_lists(cls, value: object) -> list[str]:
        if not isinstance(value, list):
            return []
        out: list[str] = []
        for item in value[:16]:
            if isinstance(item, str) and (s := item.strip()):
                out.append(s[:200])
        return out[:12]

    @field_validator("entities", mode="before")
    @classmethod
    def _normalize_entities(cls, value: object) -> list[dict]:
        if not isinstance(value, list):
            return []
        out: list[dict] = []
        for item in value[:16]:
            if isinstance(item, dict):
                name = str(item.get("name", "")).strip()
                if not name:
                    continue
                type_raw = str(item.get("type", "concept")).strip() or "concept"
                out.append({"name": name[:200], "type": type_raw[:64]})
        return out[:16]

    @field_validator("relationships", mode="before")
    @classmethod
    def _normalize_relationships(cls, value: object) -> list[dict]:
        if not isinstance(value, list):
            return []
        out: list[dict] = []
        for item in value[:16]:
            if isinstance(item, dict):
                rel = str(item.get("relation", "")).strip()
                hint = str(item.get("target_hint", "")).strip()
                if rel and hint:
                    out.append({"relation": rel[:64], "target_hint": hint[:300]})
        return out[:16]


class EnrichmentArtifact(BaseModel):
    """On-disk enrichment record under `data/enriched` (paired by markdown stem)."""

    schema_version: int = 1
    source_url: str = ""
    title: str = ""
    enriched_at_utc: str
    model: str = ""
    fallback: bool = False
    enrichment: EnrichmentPayload
