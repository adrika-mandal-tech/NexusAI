"""Vector store port — abstract interface for vector database operations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
from uuid import UUID


class VectorStorePort(ABC):
    """Abstract interface for vector store operations."""

    @abstractmethod
    async def ensure_collection(self, vector_size: int = 1024) -> None:
        """Create or validate collection compatibility for `vector_size`."""

    @abstractmethod
    async def upsert_chunks(
        self,
        chunks: list[dict[str, Any]],
    ) -> None:
        """Upsert chunks with dense (and optional sparse) embeddings into the collection."""

    @abstractmethod
    async def search(
        self,
        query_vector: list[float],
        limit: int = 10,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search for similar dense vectors."""

    async def search_sparse(
        self,
        sparse_vector: dict[str, Any],
        limit: int = 10,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search using lexical sparse vectors when supported by the collection."""
        return []

    async def get_document_chunks(
        self,
        document_id: UUID | str,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        """Retrieve stored chunks belonging to a specific document."""
        return []

    @abstractmethod
    async def delete_document_chunks(self, document_id: UUID) -> None:
        """Delete all chunks for a document."""

    @abstractmethod
    async def health(self) -> dict[str, Any]:
        """Check vector store health."""
