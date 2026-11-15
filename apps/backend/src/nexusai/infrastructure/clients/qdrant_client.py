"""Qdrant vector database client wrapper."""

from __future__ import annotations

from typing import Any

from qdrant_client import QdrantClient as QdrantSDK
from qdrant_client.models import Distance, PointStruct, VectorParams

from nexusai.core.config import get_settings

settings = get_settings()


class QdrantClient:
    """Wrapper for Qdrant operations."""

    def __init__(self) -> None:
        self._url = settings.QDRANT_URL
        self._api_key = settings.QDRANT_API_KEY or None
        self._collection = settings.QDRANT_COLLECTION
        self._client = QdrantSDK(url=self._url, api_key=self._api_key)

    async def ensure_collection(self, vector_size: int = 1024) -> None:
        """Create collection if it does not exist."""
        collections = self._client.get_collections().collections
        if any(c.name == self._collection for c in collections):
            return
        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

    async def upsert(self, vectors: list[dict[str, Any]]) -> None:
        """Upsert vectors into the collection."""
        if not vectors:
            return
        points = [
            PointStruct(
                id=v["id"],
                vector=v["vector"],
                payload=v.get("payload", {}),
            )
            for v in vectors
        ]
        self._client.upsert(collection_name=self._collection, points=points)

    async def search(self, query_vector: list[float], limit: int = 10) -> list[dict[str, Any]]:
        """Search for similar vectors."""
        results = self._client.query_points(
            collection_name=self._collection,
            query=query_vector,
            limit=limit,
        )
        return [
            {
                "id": r.id,
                "score": r.score,
                "payload": r.payload or {},
            }
            for r in results.points
        ]

    async def delete(self, source_id: str) -> None:
        """Delete all vectors for a source."""
        self._client.delete(
            collection_name=self._collection,
            points_selector={"filter": {"must": [{"key": "source_id", "match": {"value": source_id}}]}},
        )

    async def health(self) -> dict[str, Any]:
        """Check Qdrant health."""
        try:
            self._client.get_collections()
            return {"status": "ok"}
        except Exception as exc:  # noqa: BLE001
            return {"status": "error", "detail": str(exc)}
