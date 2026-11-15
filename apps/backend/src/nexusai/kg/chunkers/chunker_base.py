"""Text chunking strategies for the knowledge graph pipeline.

Provides several chunkers that split parsed text into smaller segments
suitable for LLM entity extraction and graph construction.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum

from nexusai.kg.types import Chunk

_DEFAULT_CHUNK_SIZE = 1000
_DEFAULT_CHUNK_OVERLAP = 200


class ChunkStrategy(str, Enum):
    """Available chunking strategies."""

    SEMANTIC = "semantic"
    SENTENCE = "sentence"
    HEADER = "header"
    FIXED = "fixed"
    AUTO = "auto"


class ChunkStrategyError(ValueError):
    """Raised when an unsupported chunking strategy is requested."""


class BaseChunker(ABC):
    """Abstract base for all chunkers."""

    def __init__(
        self,
        chunk_size: int = _DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = _DEFAULT_CHUNK_OVERLAP,
        min_chunk_size: int = 50,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    @abstractmethod
    def chunk(self, text: str, source_title: str, source_file: str | None = None, source_type: str | None = None) -> list[Chunk]:
        """Split text into a list of chunks."""


class FixedSizeChunker(BaseChunker):
    """Split text into fixed-size overlapping windows."""

    def chunk(self, text: str, source_title: str, source_file: str | None = None, source_type: str | None = None) -> list[Chunk]:
        chunks: list[Chunk] = []
        start = 0
        text_len = len(text)
        index = 0

        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            chunk_text = text[start:end].strip()

            if not chunk_text:
                start = end
                continue

            # If we hit the end of the text, back off to a sentence boundary
            if end == text_len and len(chunk_text) < self.chunk_size * 0.8:
                cut = self._backoff_cut(chunk_text)
                if cut > 0:
                    chunk_text = chunk_text[:cut].rstrip()
                    end = start + cut

            chunks.append(
                Chunk(
                    id=self._new_uuid(),
                    index=index,
                    text=chunk_text,
                    source_title=source_title,
                    source_file=source_file,
                    source_type=source_type,
                    metadata={},
                )
            )
            index += 1

            if end == text_len:
                break

            start = end - self.chunk_overlap

        return chunks

    def _backoff_cut(self, text: str) -> int:
        """Find a reasonable cut point near the end of a short chunk."""
        for sep in (". ", "! ", "? ", "\n\n", "\n", ", "):
            idx = text.rfind(sep)
            if idx > 0:
                return idx + len(sep)
        last_space = text.rfind(" ")
        if last_space > 0:
            return last_space
        return 0

    @staticmethod
    def _new_uuid() -> __import__("uuid").UUID:
        return __import__("uuid").uuid4()


class SentenceChunker(BaseChunker):
    """Chunk by sentence boundaries, then group into fixed windows."""

    def chunk(self, text: str, source_title: str, source_file: str | None = None, source_type: str | None = None) -> list[Chunk]:
        import re

        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks: list[Chunk] = []
        current: list[str] = []
        current_words = 0
        index = 0

        for sentence in sentences:
            words = len(sentence.split())
            if current and current_words + words > self.chunk_size:
                chunks.append(
                    Chunk(
                        id=self._new_uuid(),
                        index=index,
                        text=" ".join(current).strip(),
                        source_title=source_title,
                        source_file=source_file,
                        source_type=source_type,
                        metadata={},
                    )
                )
                index += 1
                current = []
                current_words = 0

            if words > self.chunk_size:
                tokens = sentence.split()
                for i in range(0, len(tokens), self.chunk_size):
                    window = tokens[i : i + self.chunk_size]
                    chunks.append(
                        Chunk(
                            id=self._new_uuid(),
                            index=index,
                            text=" ".join(window),
                            source_title=source_title,
                            source_file=source_file,
                            source_type=source_type,
                            metadata={},
                        )
                    )
                    index += 1
            else:
                current.append(sentence)
                current_words += words

        if current:
            chunks.append(
                Chunk(
                    id=self._new_uuid(),
                    index=index,
                    text=" ".join(current).strip(),
                    source_title=source_title,
                    source_file=source_file,
                    source_type=source_type,
                    metadata={},
                )
            )

        return chunks

    @staticmethod
    def _new_uuid() -> __import__("uuid").UUID:
        return __import__("uuid").uuid4()
