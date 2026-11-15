"""Query knowledge base use case — upgraded to run HybridRetrievalEngine + NexusSingleAgentAssistant."""

from __future__ import annotations

from typing import Any, AsyncGenerator
from uuid import UUID

from nexusai.application.agent.assistant import (
    AssistantTurnResult,
    NexusSingleAgentAssistant,
)
from nexusai.application.retrieval.hybrid_retriever import HybridRetrievalEngine
from nexusai.domain.ports.embedding_port import EmbeddingPort
from nexusai.domain.ports.graph_store import GraphStorePort
from nexusai.domain.ports.knowledge_repository import KnowledgeRepositoryPort
from nexusai.domain.ports.llm_port import LLMPort
from nexusai.domain.ports.vector_store import VectorStorePort
from nexusai.infrastructure.llm.provider import LLMProviderRegistry


class QueryKnowledgeUseCase:
    """Orchestrates Hybrid RAG query execution while preserving backward-compatible signatures."""

    def __init__(
        self,
        embedder: EmbeddingPort,
        vector_store: VectorStorePort,
        graph_store: GraphStorePort | None,
        gemini_llm: LLMPort | None = None,
        groq_llm: LLMPort | None = None,
        knowledge_repo: KnowledgeRepositoryPort | None = None,
        retriever: HybridRetrievalEngine | None = None,
        assistant: NexusSingleAgentAssistant | None = None,
        llm_registry: LLMProviderRegistry | None = None,
    ) -> None:
        from nexusai.infrastructure.repositories import InMemoryKnowledgeRepository

        self._embedder = embedder
        self._vector_store = vector_store
        self._graph_store = graph_store
        self._gemini_llm = gemini_llm
        self._groq_llm = groq_llm
        self._repo = knowledge_repo or InMemoryKnowledgeRepository()
        self._registry = llm_registry or LLMProviderRegistry()
        self._retriever = retriever or HybridRetrievalEngine(
            embedder=self._embedder,
            vector_store=self._vector_store,
            graph_store=self._graph_store,
        )
        self._assistant = assistant or NexusSingleAgentAssistant(
            retriever=self._retriever,
            llm_registry=self._registry,
            knowledge_repo=self._repo,
        )

    def _resolve_override(self, provider: str | None) -> LLMPort | None:
        selected = (provider or "").lower()
        if selected == "groq" and self._groq_llm is not None and getattr(self._groq_llm, "is_configured", False):
            return self._groq_llm
        if selected == "gemini" and self._gemini_llm is not None and getattr(self._gemini_llm, "is_configured", False):
            return self._gemini_llm
        return None

    async def execute_detailed(
        self,
        query: str,
        user_id: UUID,
        top_k: int = 8,
        provider: str | None = None,
        model: str | None = None,
        conversation_id: UUID | None = None,
        mode: str | None = None,
        fusion_strategy: str | None = None,
        enable_reranking: bool | None = None,
        include_web: bool = False,
        filter_metadata: dict[str, Any] | None = None,
    ) -> AssistantTurnResult:
        """Execute full hybrid RAG query and return answer, citations, and diagnostics."""
        return await self._assistant.answer(
            query=query,
            user_id=user_id,
            conversation_id=conversation_id,
            top_k=top_k,
            mode=mode,
            fusion_strategy=fusion_strategy,
            enable_reranking=enable_reranking,
            include_web=include_web,
            provider=provider,
            model=model,
            filter_metadata=filter_metadata,
            llm_override=self._resolve_override(provider),
        )

    async def execute(
        self,
        query: str,
        user_id: UUID,
        top_k: int = 8,
        provider: str | None = "gemini",
        model: str | None = None,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Backward-compatible tuple return `(answer, citations)`."""
        res = await self.execute_detailed(
            query=query,
            user_id=user_id,
            top_k=top_k,
            provider=provider,
            model=model,
        )
        return res.answer, res.citations

    async def stream_execute(
        self,
        query: str,
        user_id: UUID,
        top_k: int = 8,
        provider: str | None = "gemini",
        model: str | None = None,
        conversation_id: UUID | None = None,
        mode: str | None = None,
        fusion_strategy: str | None = None,
        enable_reranking: bool | None = None,
        include_web: bool = False,
        filter_metadata: dict[str, Any] | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Stream SSE events (`sources`, `data`, `done`) with full retrieval diagnostics."""
        async for event in self._assistant.stream_answer(
            query=query,
            user_id=user_id,
            conversation_id=conversation_id,
            top_k=top_k,
            mode=mode,
            fusion_strategy=fusion_strategy,
            enable_reranking=enable_reranking,
            include_web=include_web,
            provider=provider,
            model=model,
            filter_metadata=filter_metadata,
            llm_override=self._resolve_override(provider),
        ):
            yield event
