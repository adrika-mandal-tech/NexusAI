"""Pydantic schemas for knowledge endpoints."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class SourceCreate(BaseModel):
    title: str = Field(..., max_length=512)
    source_type: str = Field(..., max_length=20)
    file_path: str | None = Field(default=None, max_length=1024)
    url: str | None = Field(default=None, max_length=2048)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SourceResponse(BaseModel):
    id: UUID
    user_id: UUID
    title: str
    source_type: str
    status: str
    file_path: str | None
    url: str | None
    metadata: dict[str, Any]
    chunk_count: int
    error_message: str | None
    created_at: str
    updated_at: str

    model_config = {"from_attributes": True}


class ConversationCreate(BaseModel):
    title: str | None = Field(default=None, max_length=512)


class ConversationResponse(BaseModel):
    id: UUID
    user_id: UUID
    title: str | None
    created_at: str
    updated_at: str

    model_config = {"from_attributes": True}


class MessageCreate(BaseModel):
    role: str = Field(..., max_length=20)
    content: str
    citations: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MessageResponse(BaseModel):
    id: UUID
    conversation_id: UUID
    role: str
    content: str
    citations: list[dict[str, Any]]
    metadata: dict[str, Any]
    created_at: str

    model_config = {"from_attributes": True}


class SearchResponse(BaseModel):
    query: str
    answer: str
    sources: list[dict[str, Any]]
    citations: list[dict[str, Any]]

