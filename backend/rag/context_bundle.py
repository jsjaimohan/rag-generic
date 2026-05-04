"""Step 9 — structured retrieval outputs for grounded generation (ContextBundle)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class RetrievalTraceEntry:
    """One line in pre/post rerank traces (debug / Step 10 validation)."""

    chunk_id: str
    score: float
    source_url: str = ""
    manifest_stem: str = ""


@dataclass
class RetrievalTraces:
    pre_rerank: list[RetrievalTraceEntry] = field(default_factory=list)
    post_rerank: list[RetrievalTraceEntry] = field(default_factory=list)


@dataclass
class RetrievalConfidenceInputs:
    """Signals for Step 10 confidence aggregation (reranker-primary for now)."""

    max_rerank_score: float | None = None
    score_margin_top1_top2: float | None = None
    max_hybrid_score: float | None = None


@dataclass
class ContextBundleItem:
    """One machine-addressable chunk passed to the generator."""

    chunk_id: str
    source_url: str
    document_id: str
    manifest_stem: str
    section_title: str
    text: str
    dense_score: float | None = None
    keyword_score: float | None = None
    hybrid_score: float | None = None
    reranker_score: float | None = None
    relationship_expansion: bool = False

    def to_prompt_context_line(self, *, index: int) -> str:
        """Single block for the grounded prompt (Step 9); Step 10 will use numbered sources."""
        section_note = f" | Section: {self.section_title}" if self.section_title else ""
        return (
            f"[{index}] chunk_id={self.chunk_id} url={self.source_url} "
            f"doc={self.manifest_stem}{section_note}\n{self.text[:2000]}"
        )


def traces_to_serializable(traces: RetrievalTraces) -> dict:
    return {
        "pre_rerank": [asdict(x) for x in traces.pre_rerank],
        "post_rerank": [asdict(x) for x in traces.post_rerank],
    }


def confidence_to_serializable(conf: RetrievalConfidenceInputs) -> dict:
    return asdict(conf)


def bundle_items_to_serializable(items: list[ContextBundleItem]) -> list[dict]:
    return [asdict(x) for x in items]
