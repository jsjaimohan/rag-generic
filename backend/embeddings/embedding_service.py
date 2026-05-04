"""Dense embedding generation (BGE-M3) for Step 8 Qdrant ingestion."""

from __future__ import annotations

import os
import threading
import time

import torch

# Avoid tqdm/Hugging Face download bars on stderr mixing with uvicorn access logs.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from backend.hf_hub_utils import hub_offline_mode, resolve_hub_snapshot
from backend.settings import settings
from backend.utils.logger import get_logger

logger = get_logger(__name__)


def _embedding_device_is_mps() -> bool:
    return (settings.embedding_device or "").strip().lower() == "mps"


class EmbeddingService:
    """Lazy-loaded sentence-transformers model for normalized dense vectors."""

    def __init__(self) -> None:
        self._model = None
        self._model_lock = threading.Lock()
        self._local_snapshot_path: str | None = None
        self._warmed_up = False
        # Filled by encode() for retrieval perf (get_model vs forward).
        self._last_encode_timings: dict[str, float] = {}

    @property
    def last_encode_timings(self) -> dict[str, float]:
        """Sub-step seconds from the most recent encode() (get_model vs forward)."""
        return dict(self._last_encode_timings)

    def ensure_weights_on_disk(self) -> str:
        """Download model repo into the Hugging Face cache if missing; return local snapshot directory."""
        logger.info(
            "embeddings.snapshot.start repo=%s",
            settings.embedding_model_name,
        )
        path = resolve_hub_snapshot(settings.embedding_model_name)
        self._local_snapshot_path = path
        logger.info("embeddings.snapshot.done local_dir=%s", path)
        return path

    def _forward_encode(self, model, texts: list[str]):
        """Run bi-encoder forward under inference_mode (avoids autograd overhead)."""
        with torch.inference_mode():
            return model.encode(
                texts,
                batch_size=settings.embedding_batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )

    def _mps_warmup_extra_rounds(self) -> None:
        """
        Apple MPS: first real forwards often pay graph compilation; extra rounds + synchronize
        move that cost to startup instead of the first user /chat.
        """
        if not _embedding_device_is_mps() or not torch.backends.mps.is_available():
            if _embedding_device_is_mps() and not torch.backends.mps.is_available():
                logger.warning(
                    "EMBEDDING_DEVICE=mps but torch.backends.mps.is_available() is false; "
                    "skipping MPS warmup rounds"
                )
            return
        rounds = settings.embedding_mps_warmup_rounds
        if rounds <= 0:
            return
        model = self._get_model()
        samples = [
            "What accreditations apply to accounting diplomas at LSBF Singapore?",
            "Represent this sentence for retrieval: " + ("word " * 80),
        ]
        logger.info("embeddings.mps_warmup_extra_start rounds=%s", rounds)
        for i in range(rounds):
            t_round = time.perf_counter()
            sample = samples[i % len(samples)]
            self._forward_encode(model, [sample])
            torch.mps.synchronize()
            logger.info(
                "embeddings.mps_warmup_extra round=%s/%s duration_s=%.3f",
                i + 1,
                rounds,
                time.perf_counter() - t_round,
            )
        logger.info("embeddings.mps_warmup_extra_done")

    def warmup(self) -> None:
        """
        Ensure HF snapshot exists, load SentenceTransformer, run one encode.

        Verifies cache layout early (especially with ``ST_LOCAL_FILES_ONLY`` / ``HF_HUB_OFFLINE``).
        """
        if self._warmed_up:
            return
        t0 = time.perf_counter()
        logger.info(
            "embeddings.warmup.start repo=%s hf_hub_cache=%s st_local_files_only=%s hf_hub_offline=%s",
            settings.embedding_model_name,
            settings.huggingface_hub_cache or "(env default)",
            settings.st_local_files_only,
            hub_offline_mode(),
        )
        self.ensure_weights_on_disk()
        dim = self.get_embedding_dimension()
        self.encode(["warmup"])
        self._mps_warmup_extra_rounds()
        self._warmed_up = True
        logger.info(
            "embeddings.warmup_done dimension=%s duration_s=%.2f snapshot=%s",
            dim,
            time.perf_counter() - t0,
            self._local_snapshot_path,
        )

    def _get_model(self):
        if self._model is not None:
            return self._model
        with self._model_lock:
            if self._model is None:
                from sentence_transformers import SentenceTransformer

                device = settings.embedding_device or None
                if not self._local_snapshot_path:
                    self._local_snapshot_path = resolve_hub_snapshot(
                        settings.embedding_model_name
                    )
                path = self._local_snapshot_path
                logger.info(
                    "embeddings.load local_path=%s device=%s",
                    path,
                    device or "default",
                )
                self._model = SentenceTransformer(
                    path,
                    device=device,
                    trust_remote_code=True,
                    local_files_only=settings.st_local_files_only,
                    backend="torch",
                )
            return self._model

    def get_embedding_dimension(self) -> int:
        return int(self._get_model().get_sentence_embedding_dimension())

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            self._last_encode_timings = {}
            return []
        t0 = time.perf_counter()
        model = self._get_model()
        t1 = time.perf_counter()
        vectors = self._forward_encode(model, texts)
        t2 = time.perf_counter()
        self._last_encode_timings = {
            "embed_get_model_s": round(t1 - t0, 4),
            "embed_forward_s": round(t2 - t1, 4),
            "embed_encode_total_s": round(t2 - t0, 4),
        }
        return vectors.tolist()


embedding_service = EmbeddingService()
