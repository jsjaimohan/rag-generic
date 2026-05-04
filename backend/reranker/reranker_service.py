"""Cross-encoder reranking (BGE reranker) for Step 9."""

from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError

import torch

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from starlette.concurrency import run_in_threadpool

from backend.hf_hub_utils import resolve_hub_snapshot
from backend.settings import settings
from backend.utils.logger import get_logger

logger = get_logger(__name__)


def _reranker_effective_device_lower() -> str:
    return (settings.reranker_device or settings.embedding_device or "").strip().lower()


class RerankerService:
    """Lazy-loaded CrossEncoder for (query, passage) relevance scores."""

    def __init__(self) -> None:
        self._model = None
        self._model_lock = threading.Lock()
        self._local_snapshot_path: str | None = None
        self._warmed_up = False

    def _get_model(self):
        if self._model is not None:
            return self._model
        with self._model_lock:
            if self._model is None:
                from sentence_transformers import CrossEncoder

                device = settings.reranker_device or settings.embedding_device or None
                if not self._local_snapshot_path:
                    self._local_snapshot_path = resolve_hub_snapshot(
                        settings.reranker_model_name
                    )
                path = self._local_snapshot_path
                logger.info(
                    "reranker.load local_path=%s device=%s",
                    path,
                    device or "default",
                )
                self._model = CrossEncoder(
                    path,
                    device=device,
                    trust_remote_code=True,
                    local_files_only=settings.st_local_files_only,
                    backend="torch",
                )
            return self._model

    def _mps_warmup_extra_predicts(self, model) -> None:
        if _reranker_effective_device_lower() != "mps" or not torch.backends.mps.is_available():
            return
        rounds = settings.reranker_mps_warmup_rounds
        if rounds <= 0:
            return
        # Small multi-pair batch to approximate first real rerank shape.
        passage = "Warmup passage text " * 12
        pairs = [("retrieval warmup query", passage)] * 4
        logger.info("reranker.mps_warmup_extra_start rounds=%s", rounds)
        for i in range(rounds):
            t_round = time.perf_counter()
            with torch.inference_mode():
                model.predict(pairs, show_progress_bar=False)
            torch.mps.synchronize()
            logger.info(
                "reranker.mps_warmup_extra round=%s/%s duration_s=%.3f",
                i + 1,
                rounds,
                time.perf_counter() - t_round,
            )
        logger.info("reranker.mps_warmup_extra_done")

    def warmup(self) -> None:
        """Load the cross-encoder and run dummy predict passes (MPS graph / Metal warmup)."""
        if self._warmed_up:
            return
        model = self._get_model()
        with torch.inference_mode():
            model.predict([("warmup", "warmup")], show_progress_bar=False)
        if _reranker_effective_device_lower() == "mps" and torch.backends.mps.is_available():
            torch.mps.synchronize()
        self._mps_warmup_extra_predicts(model)
        self._warmed_up = True
        logger.info("reranker.warmup_done")

    def predict_scores(self, query: str, passages: list[str]) -> list[float]:
        """Higher score = more relevant (model-dependent scale)."""
        if not passages:
            return []

        timeout_s = settings.reranker_predict_timeout_seconds
        t0 = time.perf_counter()

        def _run_predict() -> list[float]:
            load_t0 = time.perf_counter()
            model = self._get_model()
            load_elapsed = time.perf_counter() - load_t0
            if load_elapsed > 2.0:
                logger.info(
                    "reranker.model_ready duration_s=%.2f (snapshot load + CrossEncoder init)",
                    load_elapsed,
                )
            batch_size = max(1, settings.reranker_batch_size or 16)
            scores: list[float] = []
            for i in range(0, len(passages), batch_size):
                batch = passages[i : i + batch_size]
                pairs = [(query, p) for p in batch]
                with torch.inference_mode():
                    batch_scores = model.predict(pairs, show_progress_bar=False)
                scores.extend(float(s) for s in batch_scores)
            return scores

        if timeout_s and timeout_s > 0:
            with ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(_run_predict)
                try:
                    scores = fut.result(timeout=timeout_s)
                except FuturesTimeoutError as exc:
                    raise RuntimeError(
                        f"CrossEncoder predict exceeded {timeout_s}s "
                        "(set RERANKER_PREDICT_TIMEOUT_SECONDS=0 to disable, "
                        "or RETRIEVAL_ENABLE_RERANKER=false to skip reranking)"
                    ) from exc
        else:
            scores = _run_predict()

        total = time.perf_counter() - t0
        logger.info(
            "reranker.predict_done passages=%s duration_s=%.3f timeout_cap_s=%s",
            len(passages),
            total,
            timeout_s if timeout_s and timeout_s > 0 else "none",
        )
        return scores

    async def predict_scores_async(self, query: str, passages: list[str]) -> list[float]:
        """Same as ``predict_scores`` but runs in a worker thread (non-blocking event loop)."""
        return await run_in_threadpool(self.predict_scores, query, passages)


reranker_service = RerankerService()
