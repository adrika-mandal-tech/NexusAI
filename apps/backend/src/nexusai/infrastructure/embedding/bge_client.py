"""Backward-compatible BGE-M3 embedding client delegating to ConfigurableEmbeddingService."""

from __future__ import annotations

from nexusai.infrastructure.embedding.provider import ConfigurableEmbeddingService


class BGEEmbeddingClient(ConfigurableEmbeddingService):
    """BGE-M3 embedding client sharing the process-wide cached model instance."""

    def __init__(self) -> None:
        super().__init__(provider="bge-m3", model_name="BAAI/bge-m3")
