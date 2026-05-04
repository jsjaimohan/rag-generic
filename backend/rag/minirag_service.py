"""MiniRAG orchestration: Step 9 retrieval + grounded LLM response."""

from __future__ import annotations

import re
import time

from backend.llm.qwen_service import QwenService
from backend.prompts.prompts import build_grounded_chat_prompt
from backend.rag.context_bundle import (
    bundle_items_to_serializable,
    confidence_to_serializable,
    traces_to_serializable,
)
from backend.rag.retrieval_service import retrieve_for_query
from backend.settings import settings
from backend.utils.logger import get_logger

logger = get_logger(__name__)


def _listing_style_query(query: str) -> bool:
    """Heuristic: user wants an exhaustive listing (not a single fact)."""
    q = query.strip()
    if not q:
        return False
    if re.search(r"\b(list|listing|enumerate)\b", q, re.I):
        return True
    if re.search(r"\blist\s+all\b", q, re.I):
        return True
    if re.search(r"\bfull\s+(list|listing|catalogue|catalog)\b", q, re.I):
        return True
    if re.search(r"\bcomplete\s+(list|listing)\b", q, re.I):
        return True
    if re.search(r"\ball\b", q, re.I) and re.search(
        r"\b(diplomas?|programmes?|programs?|courses?|certificates?|offerings?)\b",
        q,
        re.I,
    ):
        return True
    if re.search(
        r"\b(every|each)\s+(diploma|programme|program|course|certificate)\b",
        q,
        re.I,
    ):
        return True
    return False


def effective_top_k_for_query(user_query: str, requested_top_k: int) -> int:
    """Raise top_k for listing-style queries so catalogue answers get enough chunks."""
    k = max(1, requested_top_k)
    if not settings.retrieval_listing_query_boost:
        return min(k, settings.retrieval_rerank_input_max)
    listing_floor = max(1, settings.retrieval_listing_query_top_k)
    if _listing_style_query(user_query):
        k = max(k, listing_floor)
    return min(k, settings.retrieval_rerank_input_max)


class MiniRagService:
    """Hybrid / vector retrieval with reranking, then LM Studio text generation."""

    def __init__(self) -> None:
        self.qwen_service = QwenService()

    def chat(self, user_query: str, top_k: int = 4) -> dict:
        """Retrieve ContextBundles then generate an answer."""
        effective_k = effective_top_k_for_query(user_query, top_k)
        if effective_k != top_k:
            logger.info(
                "minirag.top_k_boost requested=%s effective=%s",
                top_k,
                effective_k,
            )
        t_retrieval_start = time.perf_counter()
        outcome = retrieve_for_query(user_query, effective_k)
        retrieval_wall_s = time.perf_counter() - t_retrieval_start

        t_prompt_start = time.perf_counter()
        context_blocks = [
            item.to_prompt_context_line(index=i + 1)
            for i, item in enumerate(outcome.items)
        ]
        prompt = build_grounded_chat_prompt(
            user_question=user_query,
            context_blocks=context_blocks,
        )
        prompt_build_s = time.perf_counter() - t_prompt_start

        t_llm_start = time.perf_counter()
        answer = self.qwen_service.generate_chat_response(prompt=prompt)
        llm_generation_s = time.perf_counter() - t_llm_start

        chat_total_s = retrieval_wall_s + prompt_build_s + llm_generation_s

        sources = [
            item.source_url if item.source_url else item.manifest_stem
            for item in outcome.items
        ]

        result: dict = {
            "answer": answer,
            "retrieval_top_k_used": effective_k,
            "retrieved_count": len(outcome.items),
            "sources": sources,
            "chunk_ids": [item.chunk_id for item in outcome.items],
            "retrieval_mode": outcome.mode,
            "retrieval_confidence_inputs": confidence_to_serializable(
                outcome.confidence_inputs
            ),
            "performance": {
                "retrieval_wall_s": round(retrieval_wall_s, 4),
                "prompt_build_s": round(prompt_build_s, 4),
                "llm_generation_s": round(llm_generation_s, 4),
                "chat_total_s": round(chat_total_s, 4),
                "retrieval_timings_s": outcome.timings_s,
            },
        }
        if settings.retrieval_include_trace_in_response:
            result["retrieval_trace"] = traces_to_serializable(outcome.traces)
            result["context_bundles"] = bundle_items_to_serializable(outcome.items)

        logger.info(
            "minirag.performance retrieval_wall_s=%.3f prompt_build_s=%.3f llm_generation_s=%.3f "
            "chat_total_s=%.3f mode=%s retrieved=%s",
            retrieval_wall_s,
            prompt_build_s,
            llm_generation_s,
            chat_total_s,
            outcome.mode,
            len(outcome.items),
        )
        logger.info(
            "minirag.chat mode=%s retrieved=%s top_rerank=%s margin=%s",
            outcome.mode,
            len(outcome.items),
            outcome.confidence_inputs.max_rerank_score,
            outcome.confidence_inputs.score_margin_top1_top2,
        )
        return result
