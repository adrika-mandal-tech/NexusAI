"""Knowledge service — orchestrates ingestion, chat, and graph operations.

This is the application layer: it coordinates domain entities and infrastructure
ports without containing business rules itself.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from fastapi import HTTPException, status

from nexusai.core.logging import get_logger
from nexusai.domain.entities.knowledge import (
    Conversation,
    ConversationRole,
    IngestionSource,
    KnowledgeGraph,
    Message,
    SourceStatus,
)
from nexusai.domain.ports.knowledge_repository import KnowledgeRepositoryPort

logger = get_logger(__name__)


class KnowledgeService:
    """Handles document ingestion, chat, and graph retrieval."""

    def __init__(self, knowledge_repo: KnowledgeRepositoryPort) -> None:
        self._repo = knowledge_repo

    # ── Sources / Ingestion ────────────────────────────────────────

    async def create_source(
        self,
        user_id: uuid.UUID,
        title: str,
        source_type: str,
        file_path: str | None = None,
        url: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> IngestionSource:
        source = IngestionSource(
            user_id=user_id,
            title=title,
            source_type=source_type,
            file_path=file_path,
            url=url,
            metadata=metadata or {},
        )
        return await self._repo.create_source(source)

    async def get_source(self, source_id: uuid.UUID) -> IngestionSource | None:
        return await self._repo.get_source(source_id)

    async def list_sources(self, user_id: uuid.UUID, limit: int, offset: int) -> tuple[list[IngestionSource], int]:
        return await self._repo.list_sources(user_id, limit, offset)

    async def update_source_status(
        self,
        source_id: uuid.UUID,
        status: SourceStatus,
        chunk_count: int | None = None,
        error_message: str | None = None,
    ) -> IngestionSource | None:
        source = await self._repo.get_source(source_id)
        if not source:
            return None
        source.status = status
        if chunk_count is not None:
            source.chunk_count = chunk_count
        if error_message is not None:
            source.error_message = error_message
        source.updated_at = datetime.utcnow()
        return await self._repo.update_source(source)

    async def delete_source(self, source_id: uuid.UUID) -> None:
        await self._repo.delete_source(source_id)

    async def trigger_ingestion(self, source_id: uuid.UUID) -> dict[str, Any]:
        """Trigger background ingestion for a source."""
        source = await self.get_source(source_id)
        if not source:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Source not found")
        source.status = SourceStatus.PROCESSING
        await self._repo.update_source(source)
        return {"task_id": str(uuid.uuid4()), "source_id": str(source_id), "status": "processing"}

    # ── Conversations ──────────────────────────────────────────────

    async def create_conversation(self, user_id: uuid.UUID, title: str | None = None) -> Conversation:
        conversation = Conversation(user_id=user_id, title=title)
        return await self._repo.create_conversation(conversation)

    async def get_conversation(self, conversation_id: uuid.UUID) -> Conversation | None:
        return await self._repo.get_conversation(conversation_id)

    async def list_conversations(self, user_id: uuid.UUID, limit: int, offset: int) -> tuple[list[Conversation], int]:
        return await self._repo.list_conversations(user_id, limit, offset)

    async def add_message(
        self,
        conversation_id: uuid.UUID,
        role: ConversationRole,
        content: str,
        citations: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Message:
        message = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            citations=citations or [],
            metadata=metadata or {},
        )
        return await self._repo.add_message(message)

    # ── Graph ──────────────────────────────────────────────────────

    async def get_graph(self, user_id: uuid.UUID) -> KnowledgeGraph:
        return await self._repo.get_graph(user_id) or KnowledgeGraph()
