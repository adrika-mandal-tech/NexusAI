"""Shared types for the knowledge graph module.

Defines datasource descriptors, ingestion options, and graph output formats
that are consumed by the ingestion engine, Neo4j loader, and visualizer.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SourceType(str, Enum):
    CSV = "csv"
    JSON = "json"
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    MD = "md"
    XLSX = "xlsx"
    HTML = "html"
    URL = "url"
    GITHUB = "github"
    TEXT = "text"


class Datasource(BaseModel):
    """A single source of data for the knowledge graph pipeline.

    Exactly one of ``file``, ``url``, or ``github`` should be supplied.
    """

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    title: str
    source_type: SourceType
    description: str | None = None
    filename: str | None = None

    # Multipart file upload (originating from client)
    file_bytes: bytes | None = None
    file_name: str | None = None

    # URL or file:// path fetch (for web/HTML/structured data)
    url: str | None = None
    github_repo: str | None = None
    github_path: str | None = None

    # Commons / generic options
    base_url: str | None = None  # for relative links in HTML/docs
    depth: int = 1  # depth for github repo crawling

    # Ingestion options
    chunk_size: int = Field(default=1000, ge=100)
    chunk_overlap: int = Field(default=200, ge=0)
    min_chunk_size: int = Field(default=50, ge=0)
    top_k_entities: int = Field(default=20, ge=0)
    top_k_relationships: int = Field(default=15, ge=0)
    use_llm_extraction: bool = Field(default=True)
    llm_model: str | None = None

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class Chunk(BaseModel):
    """A chunk of text with metadata for graph extraction."""

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    index: int
    text: str
    page_number: int | None = None
    section_title: str | None = None
    source_title: str
    source_file: str | None = None
    source_type: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Entity(BaseModel):
    """A knowledge-graph entity extracted from a chunk."""

    id: str  # e.g. "<uuid>:<name>:<type>"
    name: str
    entity_type: str
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    description: str | None = None
    confidence: float = Field(default=0.85, ge=0.0, le=1.0)
    source_text: str | None = None  # original text the entity was extracted from


class Relationship(BaseModel):
    """A knowledge-graph relationship between two entities."""

    id: str
    source_id: str
    target_id: str
    relationship_type: str
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    description: str | None = None
    weight: float = Field(default=1.0, ge=0.0)
    confidence: float = Field(default=0.85, ge=0.0, le=1.0)
    source_text: str | None = None


class ExtractionResult(BaseModel):
    """Full extraction output for a single chunk."""

    chunk_id: uuid.UUID
    entities: list[Entity] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)
    raw_response: str | None = None
    confidence_score: float = Field(default=0.0)


class UploadResult(BaseModel):
    """Result returned to the API client after a full knowledge-graph build."""

    source_id: uuid.UUID
    source_type: str
    title: str
    chunk_count: int
    entity_count: int
    relationship_count: int
    node_count: int
    edge_count: int
    status: str  # "completed" | "failed"
    error_message: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    graph_url: str | None = None  # optional visualization URL


class GraphQuery(BaseModel):
    """Parameters for querying the knowledge graph."""

    source_id: uuid.UUID | None = None
    source_title: str | None = None
    entity_type: str | None = None
    relationship_type: str | None = None
    page_size: int = Field(default=100, ge=1, le=1000)
    offset: int = Field(default=0, ge=0)
    min_confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class GraphVisualization(BaseModel):
    """Neo4j-style visualization payload (nodes + edges + layout)."""

    nodes: list[dict[str, Any]] = Field(default_factory=list)
    edges: list[dict[str, Any]] = Field(default_factory=list)
    layout: dict[str, Any] = Field(default_factory=dict)
    summary: dict[str, Any] = Field(default_factory=dict)
    thumbnail: str | None = None  # base64 PNG of the force-directed layout
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class Neo4jConfig(BaseModel):
    """Neo4j connection configuration."""

    url: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = "changeme"
    database: str = "neo4j"
    max_connection_lifetime: int = 60
    max_connection_pool_size: int = 50


class IngestionStatistics(BaseModel):
    """Aggregated ingestion statistics."""

    total_sources: int = 0
    total_chunks: int = 0
    total_entities: int = 0
    total_relationships: int = 0
    total_nodes: int = 0
    total_edges: int = 0
    per_source: dict[str, dict[str, int]] = Field(default_factory=dict)
    generated_at: datetime = Field(default_factory=datetime.utcnow)


__all__ = [
    "SourceType", "Datasource", "Chunk", "Entity", "Relationship",
    "ExtractionResult", "UploadResult", "GraphQuery", "GraphVisualization",
    "Neo4jConfig", "IngestionStatistics",
]
