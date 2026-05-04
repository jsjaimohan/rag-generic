"""Dense embedding generation (BGE-M3) for Step 8 Qdrant ingestion."""

from __future__ import annotations

import os
import time

# Avoid tqdm/Hugging Face download bars on stderr mixing with uvicorn access logs.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from backend.hf_hub_utils import hub_offline_mode, resolve_hub_snapshot
from backend.settings import settings
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    """Lazy-loaded sentence-transformers model for normalized dense vectors."""

    def __init__(self) -> None:
        self._model = None
        self._local_snapshot_path: str | None = None
        self._warmed_up = False

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
        self._warmed_up = True
        logger.info(
            "embeddings.warmup_done dimension=%s duration_s=%.2f snapshot=%s",
            dim,
            time.perf_counter() - t0,
            self._local_snapshot_path,
        )

    def _get_model(self):
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
            return []
        model = self._get_model()
        vectors = model.encode(
            texts,
            batch_size=settings.embedding_batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.tolist()


embedding_service = EmbeddingService()
