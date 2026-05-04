"""Global settings for GenericRAG."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / "backend" / ".env")
except ImportError:
    pass


def _configure_hf_hub_cache() -> None:
    """
    Pin Hugging Face Hub file cache before transformers / sentence-transformers run.

    - If ``HF_HUB_CACHE`` is set, it wins (resolved path; directory created if missing).
    - Else use ``MODELS_HUGGINGFACE_DIR`` (default: ``backend/models/huggingface`` under the
      project root). Models are stored persistently there and reused across runs.

    To keep using the OS default cache instead (~/.cache/huggingface/hub), set e.g.:
    ``HF_HUB_CACHE=/Users/you/.cache/huggingface/hub``
    """
    explicit = (os.environ.get("HF_HUB_CACHE") or "").strip()
    if explicit:
        path = Path(explicit).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HUB_CACHE"] = str(path)
        return
    raw = (os.getenv("MODELS_HUGGINGFACE_DIR") or "backend/models/huggingface").strip()
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    else:
        path = path.resolve()
    path.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HUB_CACHE"] = str(path)


_configure_hf_hub_cache()


def _env_bool(key: str, default: bool) -> bool:
    value = os.getenv(key)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _env_float(key: str, default: float) -> float:
    value = os.getenv(key)
    if value is None or not value.strip():
        return default
    return float(value)


@dataclass(frozen=True)
class Settings:
    """Application configuration loaded from environment variables."""

    app_name: str = os.getenv("APP_NAME", "GenericRAG")
    app_env: str = os.getenv("APP_ENV", "development")
    app_host: str = os.getenv("APP_HOST", "0.0.0.0")
    app_port: int = int(os.getenv("APP_PORT", "8000"))

    qdrant_host: str = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port: int = int(os.getenv("QDRANT_PORT", "6333"))
    qdrant_collection_name: str = os.getenv("QDRANT_COLLECTION_NAME", "minirag_docs")
    qdrant_vector_size: int = int(os.getenv("QDRANT_VECTOR_SIZE", "1024"))
    qdrant_upsert_batch_size: int = int(os.getenv("QDRANT_UPSERT_BATCH_SIZE", "128"))

    embedding_model_name: str = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
    embedding_batch_size: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
    embedding_device: str | None = os.getenv("EMBEDDING_DEVICE") or None
    # At FastAPI startup: snapshot_download (if needed) + load BGE + one encode (avoids first-request stall).
    embedding_warmup_on_startup: bool = _env_bool(
        "EMBEDDING_WARMUP_ON_STARTUP", True
    )
    # When EMBEDDING_DEVICE=mps: extra encode rounds at startup + mps.synchronize() to stabilize Metal (reduces first /chat stalls).
    embedding_mps_warmup_rounds: int = max(
        0, int(os.getenv("EMBEDDING_MPS_WARMUP_ROUNDS", "4"))
    )

    # Resolved at import time by _configure_hf_hub_cache(); BGE + reranker snapshots live here.
    huggingface_hub_cache: str = os.environ.get("HF_HUB_CACHE", "")

    # After snapshot_download, load ST models from disk only (no per-request Hub revision/onnx/img probes).
    st_local_files_only: bool = _env_bool("ST_LOCAL_FILES_ONLY", True)

    lmstudio_base_url: str = os.getenv(
        "LMSTUDIO_BASE_URL", "http://localhost:1234/v1"
    )
    lmstudio_model: str = os.getenv("LMSTUDIO_MODEL", "phi-2-mlx")
    lmstudio_api_key: str = os.getenv("LMSTUDIO_API_KEY", "lm-studio")
    # Qwen3-only: send chat_template_kwargs.enable_thinking=false. Phi/Llama/Gemma etc. should keep this false
    # (otherwise LM Studio may return 400; code retries without these kwargs on 400).
    lmstudio_chat_disable_thinking: bool = _env_bool("LMSTUDIO_CHAT_DISABLE_THINKING", False)
    # HTTP retries when LM Studio is starting up or briefly unavailable (connection / 502 / 503 / 504 / 429).
    lmstudio_http_attempts: int = max(
        1, int(os.getenv("LMSTUDIO_HTTP_ATTEMPTS", "3"))
    )
    lmstudio_retry_backoff_seconds: float = float(
        os.getenv("LMSTUDIO_RETRY_BACKOFF_SECONDS", "0.5")
    )
    # /v1/chat/completions read timeout (token stream). Local models can be slow; use 0 for no read limit.
    lmstudio_chat_read_timeout_seconds: int = int(
        os.getenv("LMSTUDIO_CHAT_READ_TIMEOUT_SECONDS", "300")
    )

    crawler_user_agent: str = os.getenv(
        "CRAWLER_USER_AGENT", "GenericRAGBot/1.0 (+https://local.dev)"
    )
    crawler_timeout_seconds: int = int(os.getenv("CRAWLER_TIMEOUT_SECONDS", "20"))
    data_raw_dir: str = os.getenv("DATA_RAW_DIR", "backend/data/raw")
    data_clean_dir: str = os.getenv("DATA_CLEAN_DIR", "backend/data/clean")
    data_processing_dir: str = os.getenv("DATA_PROCESSING_DIR", "backend/data/processing")
    data_enriched_dir: str = os.getenv("DATA_ENRICHED_DIR", "backend/data/enriched")
    data_semantic_chunks_dir: str = os.getenv(
        "DATA_SEMANTIC_CHUNKS_DIR", "backend/data/semantic-chunks"
    )

    enrichment_max_input_chars: int = int(
        os.getenv("ENRICHMENT_MAX_INPUT_CHARS", "12000")
    )

    chunk_max_chars: int = int(os.getenv("CHUNK_MAX_CHARS", "1800"))
    chunk_overlap_chars: int = int(os.getenv("CHUNK_OVERLAP_CHARS", "200"))
    chunking_policy_version: str = os.getenv("CHUNKING_POLICY_VERSION", "7.1")

    # Step 9 — hybrid retrieval + reranking
    # Dense vector candidates (Stage A; typical 50–100).
    retrieval_vector_pool_size: int = int(os.getenv("RETRIEVAL_VECTOR_POOL_SIZE", "72"))
    retrieval_hybrid_keyword_weight: float = _env_float(
        "RETRIEVAL_HYBRID_KEYWORD_WEIGHT", 0.35
    )
    # Top hybrid hits used as primary rows + relationship expansion seeds (before CE cap).
    retrieval_rerank_pool_size: int = int(os.getenv("RETRIEVAL_RERANK_POOL_SIZE", "48"))
    # Hard cap on passages sent to the cross-encoder (Stage B); final context is top_k (e.g. 5–10).
    retrieval_rerank_input_max: int = int(os.getenv("RETRIEVAL_RERANK_INPUT_MAX", "64"))
    retrieval_enable_reranker: bool = _env_bool("RETRIEVAL_ENABLE_RERANKER", True)
    retrieval_relationship_expand_max: int = int(
        os.getenv("RETRIEVAL_RELATIONSHIP_EXPAND_MAX", "6")
    )
    retrieval_include_trace_in_response: bool = _env_bool(
        "RETRIEVAL_INCLUDE_TRACE_IN_RESPONSE", False
    )
    retrieval_trace_max_entries: int = int(os.getenv("RETRIEVAL_TRACE_MAX_ENTRIES", "16"))
    # Broad “list all / every programme” questions: raise effective top_k (capped by RETRIEVAL_RERANK_INPUT_MAX).
    retrieval_listing_query_boost: bool = _env_bool(
        "RETRIEVAL_LISTING_QUERY_BOOST", True
    )
    retrieval_listing_query_top_k: int = int(
        os.getenv("RETRIEVAL_LISTING_QUERY_TOP_K", "10")
    )
    reranker_model_name: str = os.getenv(
        "RERANKER_MODEL_NAME", "BAAI/bge-reranker-v2-m3"
    )
    reranker_batch_size: int = int(os.getenv("RERANKER_BATCH_SIZE", "16"))
    reranker_device: str | None = os.getenv("RERANKER_DEVICE") or None
    # Wall-clock cap for an entire predict_scores call (load + all batches). 0 = no limit.
    reranker_predict_timeout_seconds: int = int(
        os.getenv("RERANKER_PREDICT_TIMEOUT_SECONDS", "300")
    )
    # At startup (after embedding warmup): load cross-encoder and run dummy predict (and extra MPS rounds when on mps).
    reranker_warmup_on_startup: bool = _env_bool("RERANKER_WARMUP_ON_STARTUP", True)
    reranker_mps_warmup_rounds: int = max(
        0, int(os.getenv("RERANKER_MPS_WARMUP_ROUNDS", "2"))
    )


settings = Settings()


def resolve_data_path(path_value: str) -> Path:
    """Resolve data directory path from project root."""
    candidate_path = Path(path_value).expanduser()
    if candidate_path.is_absolute():
        return candidate_path
    return (PROJECT_ROOT / candidate_path).resolve()
