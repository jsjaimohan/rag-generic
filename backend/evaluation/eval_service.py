"""Evaluation service for retrieval and grounding quality."""

from __future__ import annotations

import time
from dataclasses import dataclass
from statistics import mean

from backend.rag.minirag_service import MiniRagService
from backend.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EvaluationCase:
    """One evaluation query with expected terms."""

    query: str
    expected_terms: list[str]


class EvaluationService:
    """Runs lightweight evaluation over the current RAG pipeline."""

    def __init__(self) -> None:
        self.minirag_service = MiniRagService()

    def run(self, test_cases: list[EvaluationCase], top_k: int = 4) -> dict:
        """Run evaluation and return aggregate metrics."""
        if not test_cases:
            raise ValueError("At least one test case is required.")

        t_total = time.perf_counter()
        logger.info(
            "evaluation.start cases=%s top_k=%s",
            len(test_cases),
            top_k,
        )

        case_results: list[dict] = []
        retrieval_hits: list[int] = []
        grounded_hits: list[int] = []

        for index, test_case in enumerate(test_cases):
            t_case = time.perf_counter()
            chat_result = self.minirag_service.chat(user_query=test_case.query, top_k=top_k)
            case_sec = time.perf_counter() - t_case
            answer = chat_result.get("answer", "")

            retrieval_hit = 1 if int(chat_result.get("retrieved_count", 0)) > 0 else 0
            matched_term = self._first_matching_expected_term(answer, test_case.expected_terms)
            grounded_hit = 1 if matched_term is not None else 0

            retrieval_hits.append(retrieval_hit)
            grounded_hits.append(grounded_hit)

            case_results.append(
                {
                    "query": test_case.query,
                    "retrieved_count": chat_result.get("retrieved_count", 0),
                    "retrieval_hit": retrieval_hit,
                    "grounded_hit": grounded_hit,
                    "matched_term": matched_term,
                    "expected_terms": test_case.expected_terms,
                    "answer_preview": answer[:300],
                    "sources": chat_result.get("sources", []),
                    "retrieval_mode": chat_result.get("retrieval_mode"),
                    "chunk_ids": chat_result.get("chunk_ids", []),
                }
            )
            logger.info(
                "evaluation.case index=%s/%s duration_s=%.2f retrieved=%s mode=%s "
                "retrieval_hit=%s grounded_hit=%s query_preview=%r",
                index + 1,
                len(test_cases),
                case_sec,
                chat_result.get("retrieved_count", 0),
                chat_result.get("retrieval_mode"),
                retrieval_hit,
                grounded_hit,
                test_case.query[:120],
            )

        retrieval_hit_rate = mean(retrieval_hits)
        grounded_hit_rate = mean(grounded_hits)
        hallucination_rate = 1.0 - grounded_hit_rate

        logger.info(
            "evaluation.done total_duration_s=%.2f retrieval_hit_rate=%.4f grounded_hit_rate=%.4f",
            time.perf_counter() - t_total,
            retrieval_hit_rate,
            grounded_hit_rate,
        )

        return {
            "cases_count": len(test_cases),
            "retrieval_hit_rate": round(retrieval_hit_rate, 4),
            "grounded_hit_rate": round(grounded_hit_rate, 4),
            "hallucination_rate_estimate": round(hallucination_rate, 4),
            "cases": case_results,
        }

    @staticmethod
    def _first_matching_expected_term(answer: str, expected_terms: list[str]) -> str | None:
        """First expected term found as substring in answer (case-insensitive), or None."""
        lowered_answer = answer.lower()
        for term in expected_terms:
            if term.lower() in lowered_answer:
                return term
        return None
