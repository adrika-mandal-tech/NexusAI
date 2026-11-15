"""Graph store port — abstract interface for knowledge graph operations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
from uuid import UUID


class GraphStorePort(ABC):
    """Abstract interface for graph store operations."""

    @abstractmethod
    async def create_document_node(
        self,
        document_id: UUID,
        filename: str,
        source_type: str,
        user_id: UUID,
    ) -> None:
        """Create a Document node in Neo4j."""

    @abstractmethod
    async def create_chunk_nodes(
        self,
        chunks: list[dict[str, Any]],
    ) -> None:
        """Create Chunk nodes linked to Document."""

    @abstractmethod
    async def create_entities_and_relationships(
        self,
        chunk_id: UUID,
        document_id: UUID,
        entities: list[dict[str, Any]],
        relationships: list[dict[str, Any]],
    ) -> None:
        """Create Entity nodes and RELATED edges for a chunk."""

    async def create_batch_entities_and_relationships(
        self,
        document_id: UUID,
        chunk_extractions: list[dict[str, Any]],
    ) -> None:
        """Batch-persist entities and relationships for multiple chunks."""
        for item in chunk_extractions:
            await self.create_entities_and_relationships(
                chunk_id=item["chunk_id"],
                document_id=document_id,
                entities=item.get("entities", []),
                relationships=item.get("relationships", []),
            )

    @abstractmethod
    async def get_chunk_entities(self, chunk_ids: list[UUID]) -> list[dict[str, Any]]:
        """Get entities connected to the given chunk IDs (1-hop)."""

    async def search_graph_context(
        self,
        query_terms: list[str],
        limit: int = 15,
        document_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Search both Document entities/chunks and CSV-ingested domain graph nodes."""
        return {"chunks": [], "triples": [], "entities": []}

    async def verify_document_chunks(self, document_id: UUID) -> dict[str, int]:
        """Verify persisted Neo4j chunk and entity counts for cross-DB consistency."""
        return {"chunk_count": 0, "entity_count": 0}

    @abstractmethod
    async def get_graph(self, document_id: UUID | None = None) -> dict[str, Any]:
        """Get graph visualization data `{nodes: [...], edges: [...]}`."""

    @abstractmethod
    async def delete_document_graph(self, document_id: UUID) -> None:
        """Delete all graph data exclusively owned by a document."""

    @abstractmethod
    async def health(self) -> dict[str, Any]:
        """Check graph store health."""
