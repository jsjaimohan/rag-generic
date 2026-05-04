"""In-memory indexing job progress (single-process dev server).

Job state is lost when the process restarts (e.g. ``uvicorn --reload``). Poll endpoints
return ``status: unknown`` instead of HTTP 404 so the UI can stop polling and explain.
"""

from __future__ import annotations

import threading
import uuid
from typing import Any

_lock = threading.Lock()
_jobs: dict[str, dict[str, Any]] = {}


def create_indexing_job(total_files: int) -> str:
    job_id = str(uuid.uuid4())
    with _lock:
        _jobs[job_id] = {
            "job_id": job_id,
            "status": "running",
            "phase": "queued",
            "processed_files": 0,
            "total_files": total_files,
            "points_indexed": 0,
            "current_file": "",
            "error": None,
            "result": None,
        }
    return job_id


def update_indexing_job(job_id: str, **kwargs: Any) -> None:
    with _lock:
        if job_id in _jobs:
            _jobs[job_id].update(kwargs)


def complete_indexing_job(job_id: str, result: dict[str, Any]) -> None:
    with _lock:
        if job_id in _jobs:
            _jobs[job_id].update(
                status="completed",
                phase="completed",
                result=result,
                current_file="",
                processed_files=_jobs[job_id]["total_files"],
            )


def fail_indexing_job(job_id: str, error: str) -> None:
    update_indexing_job(job_id, status="failed", phase="failed", error=error, current_file="")


def get_indexing_job(job_id: str) -> dict[str, Any] | None:
    with _lock:
        job = _jobs.get(job_id)
        return dict(job) if job else None


def indexing_job_to_response(job: dict[str, Any]) -> dict[str, Any]:
    total = max(1, int(job.get("total_files") or 1))
    processed = int(job.get("processed_files") or 0)
    percent = min(100.0, round(100.0 * processed / total, 1))
    return {**job, "percent": percent}
