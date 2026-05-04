"""Application entrypoint for GenericRAG."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

# Set before any import that may load huggingface_hub / transformers (e.g. API routes).
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

# Configure HF_HUB_CACHE via backend.settings (must run before routes import hub).
from backend.settings import settings
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from backend.api.routes import router as api_router
from backend.utils.logger import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)
logger.info(
    "startup log_level=%s huggingface_hub_cache=%s",
    os.environ.get("LOG_LEVEL", "INFO"),
    settings.huggingface_hub_cache,
)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    if settings.embedding_warmup_on_startup:
        from backend.embeddings.embedding_service import embedding_service

        try:
            await run_in_threadpool(embedding_service.warmup)
        except Exception:
            logger.exception(
                "embeddings.warmup_failed — fix HF cache or set EMBEDDING_WARMUP_ON_STARTUP=false"
            )
            raise
    yield


app = FastAPI(title=settings.app_name, lifespan=_lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)
