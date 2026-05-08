"""MiniRAG orchestration: Step 9 retrieval + grounded LLM response."""

from __future__ import annotations

import re
import time

from backend.llm.qwen_service import LlmClientConfig, QwenService
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
        self.qwen_service = QwenService(
            config=LlmClientConfig(
                base_url=settings.chat_llm_base_url,
                model=settings.chat_llm_model,
                api_key=settings.chat_llm_api_key,
                disable_thinking=settings.chat_llm_disable_thinking,
                http_attempts=settings.chat_llm_http_attempts,
                retry_backoff_seconds=settings.chat_llm_retry_backoff_seconds,
                read_timeout_seconds=settings.chat_llm_read_timeout_seconds,
            )
        )

    def prepare_chat_context(self, user_query: str, top_k: int = 4) -> tuple[dict, list[dict[str, str]]]:
        """
        Run retrieval and build LM Studio ``messages`` (shared by buffered and streaming chat).

        Returns ``(meta_partial, messages)`` where ``meta_partial`` matches the non-streaming
        JSON fields except ``answer`` and ``performance.llm_generation_s`` / ``chat_total_s``.
        """
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

        sources = [
            item.source_url if item.source_url else item.manifest_stem
            for item in outcome.items
        ]

        meta_partial: dict = {
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
                "retrieval_timings_s": outcome.timings_s,
            },
        }
        if settings.retrieval_include_trace_in_response:
            meta_partial["retrieval_trace"] = traces_to_serializable(outcome.traces)
            meta_partial["context_bundles"] = bundle_items_to_serializable(outcome.items)

        messages = self.qwen_service.grounded_chat_messages_for_prompt(prompt)
        return meta_partial, messages

    def chat(self, user_query: str, top_k: int = 4) -> dict:
        """Retrieve ContextBundles then generate an answer (buffered JSON response)."""
        meta_partial, messages = self.prepare_chat_context(user_query, top_k)
        outcome_mode = meta_partial["retrieval_mode"]
        retrieved_n = meta_partial["retrieved_count"]

        t_llm_start = time.perf_counter()
        answer = self.qwen_service.complete_chat(
            messages=messages,
            temperature=0.1,
        )
        llm_generation_s = time.perf_counter() - t_llm_start

        perf = meta_partial["performance"]
        retrieval_wall_s = perf["retrieval_wall_s"]
        prompt_build_s = perf["prompt_build_s"]
        chat_total_s = retrieval_wall_s + prompt_build_s + llm_generation_s
        perf["llm_generation_s"] = round(llm_generation_s, 4)
        perf["chat_total_s"] = round(chat_total_s, 4)

        result = {**meta_partial, "answer": answer}

        logger.info(
            "minirag.performance retrieval_wall_s=%.3f prompt_build_s=%.3f llm_generation_s=%.3f "
            "chat_total_s=%.3f mode=%s retrieved=%s",
            retrieval_wall_s,
            prompt_build_s,
            llm_generation_s,
            chat_total_s,
            outcome_mode,
            retrieved_n,
        )
        logger.info(
            "minirag.chat mode=%s retrieved=%s top_rerank=%s margin=%s",
            outcome_mode,
            retrieved_n,
            meta_partial["retrieval_confidence_inputs"].get("max_rerank_score"),
            meta_partial["retrieval_confidence_inputs"].get("score_margin_top1_top2"),
        )
        return result
