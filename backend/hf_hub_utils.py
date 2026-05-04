"""Resolve Hugging Face Hub repos to a local snapshot path (project cache via HF_HUB_CACHE)."""

from __future__ import annotations

import os

from huggingface_hub import snapshot_download
from huggingface_hub.utils import disable_progress_bars

from backend.settings import settings
from backend.utils.logger import get_logger

logger = get_logger(__name__)


def hub_offline_mode() -> bool:
    return (os.getenv("HF_HUB_OFFLINE") or "").strip().lower() in ("1", "true", "yes")


def resolve_hub_snapshot(repo_id: str) -> str:
    """
    Download if needed, then return the snapshot directory path.

    Loading ``SentenceTransformer`` / ``CrossEncoder`` from this path avoids treating the
    argument as a remote repo id (fewer Hub metadata requests than passing ``repo_id`` alone).

    Set ``HF_HUB_OFFLINE=1`` only after the snapshot exists under ``HF_HUB_CACHE`` to skip network.
    """
    disable_progress_bars()
    offline = hub_offline_mode()
    kwargs: dict = {
        "repo_id": repo_id,
        "local_files_only": offline,
    }
    if settings.huggingface_hub_cache:
        kwargs["cache_dir"] = settings.huggingface_hub_cache
    path = snapshot_download(**kwargs)
    logger.info("hf_hub.snapshot repo=%s path=%s offline=%s", repo_id, path, offline)
    return path
