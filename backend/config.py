"""Configuration utilities for GenericRAG."""

from __future__ import annotations

from backend.settings import settings


def get_app_bind_address() -> tuple[str, int]:
    """Return host/port tuple for server startup."""
    return settings.app_host, settings.app_port


def get_qdrant_url() -> str:
    """Build the Qdrant HTTP URL."""
    return f"http://{settings.qdrant_host}:{settings.qdrant_port}"
