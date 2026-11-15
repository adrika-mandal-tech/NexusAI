"""Embedding client using BGE-M3 via sentence-transformers."""

from __future__ import annotations

import os

from nexusai.core.config import get_settings

settings = get_settings()

if settings.HF_API_KEY:
    os.environ.setdefault("HF_TOKEN", settings.HF_API_KEY)


class EmbeddingClient:
    """Wrapper for embedding generation."""

    def __init__(self) -> None:
        self._model_name = "BAAI/bge-m3"
        self._model = None
        self._dimension = 1024

    def _load(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
            self._dimension = self._model.get_sentence_embedding_dimension()

    def get_dimension(self) -> int:
        if self._model is None:
            self._load()
        return self._dimension

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts."""
        self._load()
        embeddings = self._model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Generate embedding for a single query."""
        return self.embed_texts([query])[0]
