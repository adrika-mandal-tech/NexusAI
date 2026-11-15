"""Embedding port — abstract interface for embedding generation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class EmbeddingPort(ABC):
    """Abstract interface for embedding generation."""

    @abstractmethod
    def get_dimension(self) -> int:
        """Return the embedding dimension."""

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts."""

    @abstractmethod
    def embed_query(self, query: str) -> list[float]:
        """Generate embedding for a single query."""

    @abstractmethod
    def embed_sparse(self, texts: list[str]) -> list[dict[str, Any]]:
        """Generate sparse embeddings (for hybrid search).

        Returns list of dicts with 'indices' and 'values' for each text.
        """
