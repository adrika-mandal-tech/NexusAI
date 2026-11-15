"""Core domain entities for NexusAI."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


def _utc_now_naive() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class SourceType(StrEnum):
    PDF = "pdf"
    DOCX = "docx"
    PPTX = "pptx"
    XLSX = "xlsx"
    HTML = "html"
    CSV = "csv"
    TXT = "txt"
    MD = "md"
    JSON = "json"
    IMAGE = "image"
    WEB = "web"
    GITHUB = "github"


class SourceStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class IngestionSource(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    user_id: uuid.UUID
    title: str
    source_type: SourceType
    status: SourceStatus = SourceStatus.PENDING
    file_path: str | None = None
    url: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    chunk_count: int = 0
    error_message: str | None = None
    created_at: datetime = Field(default_factory=_utc_now_naive)
    updated_at: datetime = Field(default_factory=_utc_now_naive)


class ConversationRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


class Message(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    conversation_id: uuid.UUID
    role: ConversationRole
    content: str
    citations: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utc_now_naive)


class Conversation(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    user_id: uuid.UUID
    title: str | None = None
    messages: list[Message] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_utc_now_naive)
    updated_at: datetime = Field(default_factory=_utc_now_naive)


class GraphNode(BaseModel):
    id: str
    label: str
    type: str
    description: str
    confidence: float
    citations: int
    updated_at: datetime
    taxonomy: list[str] = Field(default_factory=list)


class GraphEdge(BaseModel):
    source: str
    target: str
    label: str
    weight: float = 1.0


class KnowledgeGraph(BaseModel):
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)
