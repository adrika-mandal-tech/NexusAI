"""In-memory repository implementation for development and demo purposes."""

from __future__ import annotations

import uuid
from datetime import datetime

from nexusai.domain.entities.knowledge import (
    Conversation,
    IngestionSource,
    KnowledgeGraph,
    Message,
)


class InMemoryKnowledgeRepository:
    """In-memory implementation of KnowledgeRepositoryPort."""

    def __init__(self) -> None:
        self._sources: dict[uuid.UUID, IngestionSource] = {}
        self._conversations: dict[uuid.UUID, Conversation] = {}
        self._messages: dict[uuid.UUID, Message] = {}
        self._user_sources: dict[uuid.UUID, list[uuid.UUID]] = {}
        self._user_conversations: dict[uuid.UUID, list[uuid.UUID]] = {}

    async def create_source(self, source: IngestionSource) -> IngestionSource:
        source.id = uuid.uuid4()
        source.created_at = datetime.utcnow()
        source.updated_at = source.created_at
        self._sources[source.id] = source
        self._user_sources.setdefault(source.user_id, []).append(source.id)
        return source

    async def get_source(self, source_id: uuid.UUID) -> IngestionSource | None:
        return self._sources.get(source_id)

    async def list_sources(self, user_id: uuid.UUID, limit: int, offset: int) -> tuple[list[IngestionSource], int]:
        ids = self._user_sources.get(user_id, [])
        items = [self._sources[i] for i in ids if i in self._sources]
        items.sort(key=lambda x: x.created_at, reverse=True)
        total = len(items)
        return items[offset : offset + limit], total

    async def update_source(self, source: IngestionSource) -> IngestionSource:
        source.updated_at = datetime.utcnow()
        self._sources[source.id] = source
        return source

    async def delete_source(self, source_id: uuid.UUID) -> None:
        source = self._sources.pop(source_id, None)
        if source:
            self._user_sources.get(source.user_id, []).remove(source_id)

    async def create_conversation(self, conversation: Conversation) -> Conversation:
        conversation.id = uuid.uuid4()
        conversation.created_at = datetime.utcnow()
        conversation.updated_at = conversation.created_at
        self._conversations[conversation.id] = conversation
        self._user_conversations.setdefault(conversation.user_id, []).append(conversation.id)
        return conversation

    async def get_conversation(self, conversation_id: uuid.UUID) -> Conversation | None:
        return self._conversations.get(conversation_id)

    async def list_conversations(self, user_id: uuid.UUID, limit: int, offset: int) -> tuple[list[Conversation], int]:
        ids = self._user_conversations.get(user_id, [])
        items = [self._conversations[i] for i in ids if i in self._conversations]
        items.sort(key=lambda x: x.created_at, reverse=True)
        total = len(items)
        return items[offset : offset + limit], total

    async def add_message(self, message: Message) -> Message:
        message.id = uuid.uuid4()
        message.created_at = datetime.utcnow()
        self._messages[message.id] = message
        conversation = self._conversations.get(message.conversation_id)
        if conversation:
            conversation.messages.append(message)
            conversation.updated_at = datetime.utcnow()
        return message

    async def get_graph(self, user_id: uuid.UUID) -> KnowledgeGraph | None:
        return KnowledgeGraph()
