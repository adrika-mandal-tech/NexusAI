"""Knowledge repository port — abstract interface for ingestion and retrieval.

The domain defines the contract; infrastructure implements it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from uuid import UUID

from nexusai.domain.entities.knowledge import (
    Conversation,
    IngestionSource,
    KnowledgeGraph,
    Message,
)


class KnowledgeRepositoryPort(ABC):
    """Abstract interface for knowledge persistence operations."""

    @abstractmethod
    async def create_source(self, source: IngestionSource) -> IngestionSource:
        """Persist a new ingestion source."""

    @abstractmethod
    async def get_source(self, source_id: UUID) -> IngestionSource | None:
        """Fetch a source by ID."""

    @abstractmethod
    async def list_sources(self, user_id: UUID, limit: int, offset: int) -> tuple[list[IngestionSource], int]:
        """List sources for a user."""

    @abstractmethod
    async def update_source(self, source: IngestionSource) -> IngestionSource:
        """Update source status/metadata."""

    @abstractmethod
    async def delete_source(self, source_id: UUID) -> None:
        """Delete a source and its chunks."""

    @abstractmethod
    async def create_conversation(self, conversation: Conversation) -> Conversation:
        """Create a new conversation."""

    @abstractmethod
    async def get_conversation(self, conversation_id: UUID) -> Conversation | None:
        """Fetch a conversation with messages."""

    @abstractmethod
    async def list_conversations(self, user_id: UUID, limit: int, offset: int) -> tuple[list[Conversation], int]:
        """List conversations for a user."""

    @abstractmethod
    async def add_message(self, message: Message) -> Message:
        """Append a message to a conversation."""

    @abstractmethod
    async def get_graph(self, user_id: UUID) -> KnowledgeGraph | None:
        """Fetch the knowledge graph for a user."""

    async def find_source_by_hash(self, user_id: UUID, content_hash: str) -> IngestionSource | None:
        """Find an existing non-failed source matching SHA-256 content_hash."""
        sources, _ = await self.list_sources(user_id=user_id, limit=500, offset=0)
        for src in sources:
            if src.metadata.get("content_hash") == content_hash and src.status.value != "failed":
                return src
        return None

    async def delete_conversation(self, conversation_id: UUID) -> None:
        """Delete a conversation and its messages."""
        return None
