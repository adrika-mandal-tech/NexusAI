"""Factory for instantiating chunkers by name."""

from __future__ import annotations

from nexusai.kg.chunkers.chunker_base import BaseChunker, ChunkStrategy, ChunkStrategyError
from nexusai.kg.chunkers.header_chunker import HeaderChunker


class ChunkerFactory:
    """Factory for creating chunkers by name."""

    _registry: dict[str, type[BaseChunker]] = {
        ChunkStrategy.SEMANTIC.value: __import__("nexusai.kg.chunkers.chunker_base", fromlist=["SentenceChunker"]).SentenceChunker,
        ChunkStrategy.SENTENCE.value: __import__("nexusai.kg.chunkers.chunker_base", fromlist=["SentenceChunker"]).SentenceChunker,
        ChunkStrategy.HEADER.value: HeaderChunker,
        ChunkStrategy.FIXED.value: __import__("nexusai.kg.chunkers.chunker_base", fromlist=["FixedSizeChunker"]).FixedSizeChunker,
        ChunkStrategy.AUTO.value: __import__("nexusai.kg.chunkers.chunker_base", fromlist=["SentenceChunker"]).SentenceChunker,
    }

    @classmethod
    def create(cls, strategy: str | ChunkStrategy = ChunkStrategy.AUTO, **kwargs) -> BaseChunker:
        if isinstance(strategy, ChunkStrategy):
            key = strategy.value
        else:
            key = strategy.lower()

        chunker_cls = cls._registry.get(key)
        if chunker_cls is None:
            raise ChunkStrategyError(f"Unknown chunking strategy: {key!r}")

        return chunker_cls(**kwargs)

    @classmethod
    def registered_strategies(cls) -> list[str]:
        return list(cls._registry.keys())
