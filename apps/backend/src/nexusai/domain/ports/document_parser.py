"""Document parser port — abstract interface for document parsing."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
from uuid import UUID


class ParsedChunk:
    """A chunk of parsed document content with metadata."""

    def __init__(
        self,
        chunk_id: UUID,
        document_id: UUID,
        index: int,
        text: str,
        page_number: int | None = None,
        section_title: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self.chunk_id = chunk_id
        self.document_id = document_id
        self.index = index
        self.text = text
        self.page_number = page_number
        self.section_title = section_title
        self.metadata = metadata or {}


class ParsedDocument:
    """Result of parsing a document."""

    def __init__(
        self,
        document_id: UUID,
        filename: str,
        source_type: str,
        chunks: list[ParsedChunk],
        metadata: dict[str, Any] | None = None,
    ):
        self.document_id = document_id
        self.filename = filename
        self.source_type = source_type
        self.chunks = chunks
        self.metadata = metadata or {}


class DocumentParserPort(ABC):
    """Abstract interface for document parsing."""

    @abstractmethod
    async def parse(
        self,
        file_path: str,
        document_id: UUID,
        filename: str,
        source_type: str,
    ) -> ParsedDocument:
        """Parse a document file and return structured chunks with metadata."""

    @abstractmethod
    def supported_types(self) -> list[str]:
        """Return list of supported file extensions (e.g., ['.pdf', '.docx'])."""
