"""Logging utilities for GenericRAG."""

from __future__ import annotations

import logging
import os


def setup_logging() -> None:
    """Configure app-wide logging format (``LOG_LEVEL`` env: DEBUG, INFO, WARNING, …)."""
    level_name = (os.getenv("LOG_LEVEL") or "INFO").strip().upper()
    level = getattr(logging, level_name, logging.INFO)
    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    logging.basicConfig(level=level, format=fmt)
    for log_name in ("uvicorn", "uvicorn.access", "uvicorn.error", "httpx"):
        logging.getLogger(log_name).setLevel(level)


def get_logger(name: str) -> logging.Logger:
    """Return a logger for the given module name."""
    return logging.getLogger(name)
