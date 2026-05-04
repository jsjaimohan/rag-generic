"""Qdrant vector database: connectivity, collections, search, and upserts (Steps 8–9)."""

from __future__ import annotations

from dataclasses import dataclass

import httpx
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from backend.config import get_qdrant_url
from backend.settings import settings


@dataclass(frozen=True)
class QdrantSearchHit:
    """One dense search result with payload."""

    point_id: str
    score: float
    payload: dict


@dataclass(frozen=True)
class QdrantRetrievedPoint:
    """Point fetched by id (e.g. relationship expansion)."""

    point_id: str
    payload: dict


class QdrantService:
    """HTTP health checks plus gRPC/HTTP client for collection lifecycle and upserts."""

    def __init__(self) -> None:
        self._client = QdrantClient(url=get_qdrant_url(), timeout=120.0)

    @property
    def client(self) -> QdrantClient:
        return self._client

    def health_check(self) -> dict:
        """Lightweight REST check (no heavy client assumptions)."""
        qdrant_url = get_qdrant_url()
        response = httpx.get(f"{qdrant_url}/collections", timeout=5)
        response.raise_for_status()
        payload = response.json()
        return {
            "ok": True,
            "collections_count": len(payload.get("result", {}).get("collections", [])),
        }

    def ensure_collection(self, *, recreate: bool = False) -> None:
        """Create collection if missing, or recreate when ``recreate`` is True."""
        name = settings.qdrant_collection_name
        size = settings.qdrant_vector_size
        params = VectorParams(size=size, distance=Distance.COSINE)
        if recreate:
            self._client.recreate_collection(collection_name=name, vectors_config=params)
            return
        try:
            self._client.get_collection(name)
        except Exception:
            self._client.create_collection(collection_name=name, vectors_config=params)

    def upsert_points(self, points: list[PointStruct], *, wait: bool = True) -> None:
        if not points:
            return
        self._client.upsert(
            collection_name=settings.qdrant_collection_name,
            points=points,
            wait=wait,
        )

    def count_points(self) -> int:
        result = self._client.count(
            collection_name=settings.qdrant_collection_name,
            exact=True,
        )
        return int(result.count)

    def search_dense(
        self,
        query_vector: list[float],
        *,
        limit: int,
        score_threshold: float | None = None,
    ) -> list[QdrantSearchHit]:
        """Cosine similarity search over the default dense vector."""
        if limit <= 0:
            return []
        response = self._client.query_points(
            collection_name=settings.qdrant_collection_name,
            query=query_vector,
            limit=limit,
            score_threshold=score_threshold,
            with_payload=True,
        )
        out: list[QdrantSearchHit] = []
        for scored in response.points:
            pid = scored.id
            point_id = str(pid) if not isinstance(pid, str) else pid
            payload = scored.payload or {}
            if not isinstance(payload, dict):
                payload = dict(payload)
            out.append(
                QdrantSearchHit(
                    point_id=point_id,
                    score=float(scored.score),
                    payload=payload,
                )
            )
        return out

    def retrieve_payloads_by_ids(self, point_ids: list[str]) -> list[QdrantRetrievedPoint]:
        """Fetch payloads for known point ids (chunk_id UUIDs)."""
        if not point_ids:
            return []
        records = self._client.retrieve(
            collection_name=settings.qdrant_collection_name,
            ids=point_ids,
            with_payload=True,
            with_vectors=False,
        )
        out: list[QdrantRetrievedPoint] = []
        for record in records:
            pid = record.id
            point_id = str(pid) if not isinstance(pid, str) else pid
            payload = record.payload or {}
            if not isinstance(payload, dict):
                payload = dict(payload)
            out.append(QdrantRetrievedPoint(point_id=point_id, payload=payload))
        return out


qdrant_service = QdrantService()
