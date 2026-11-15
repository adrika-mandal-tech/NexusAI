"""Process-wide singleton service container and resilient ingestion dispatcher."""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any
from uuid import UUID

from fastapi import BackgroundTasks

from nexusai.application.agent.assistant import NexusSingleAgentAssistant
from nexusai.application.retrieval.hybrid_retriever import HybridRetrievalEngine
from nexusai.application.services.knowledge_service import KnowledgeService
from nexusai.application.use_cases.ingest_document import IngestDocumentUseCase
from nexusai.application.use_cases.query_knowledge import QueryKnowledgeUseCase
from nexusai.core.config import get_settings
from nexusai.core.logging import get_logger
from nexusai.domain.entities.knowledge import SourceStatus
from nexusai.infrastructure.db.repositories import ResilientKnowledgeRepository
from nexusai.infrastructure.embedding.provider import ConfigurableEmbeddingService
from nexusai.infrastructure.graph_stores.neo4j_readonly_store import Neo4jReadOnlyStore
from nexusai.infrastructure.graph_stores.neo4j_store import Neo4jGraphStore
from nexusai.infrastructure.llm.provider import LLMProviderRegistry
from nexusai.infrastructure.parsers.router_parser import SmartDocumentRouter
from nexusai.infrastructure.search.tavily_service import TavilySearchService
from nexusai.infrastructure.vector_stores.qdrant_store import QdrantVectorStore

logger = get_logger(__name__)


@functools.lru_cache(maxsize=1)
def get_embedder() -> ConfigurableEmbeddingService:
    return ConfigurableEmbeddingService()


@functools.lru_cache(maxsize=1)
def get_document_router() -> SmartDocumentRouter:
    return SmartDocumentRouter()


@functools.lru_cache(maxsize=1)
def get_vector_store() -> QdrantVectorStore:
    embedder = get_embedder()
    return QdrantVectorStore(
        embedding_model=embedder.model_name,
        expected_dimension=embedder.get_dimension(),
        auto_scope_by_model=True,
    )


def reset_embedding_and_vector_caches() -> None:
    """Clear singleton embedder, vector store, and router caches after runtime settings update."""
    get_embedder.cache_clear()
    get_vector_store.cache_clear()
    get_document_router.cache_clear()


def get_graph_store() -> Neo4jGraphStore:
    return Neo4jGraphStore()


def get_readonly_graph_store() -> Neo4jReadOnlyStore:
    return Neo4jReadOnlyStore()


@functools.lru_cache(maxsize=1)
def get_knowledge_repo() -> ResilientKnowledgeRepository:
    return ResilientKnowledgeRepository()


@functools.lru_cache(maxsize=1)
def get_tavily_service() -> TavilySearchService:
    return TavilySearchService()


@functools.lru_cache(maxsize=1)
def get_llm_registry() -> LLMProviderRegistry:
    return LLMProviderRegistry()


def get_hybrid_retriever() -> HybridRetrievalEngine:
    return HybridRetrievalEngine(
        embedder=get_embedder(),
        vector_store=get_vector_store(),
        graph_store=get_graph_store(),
        tavily_service=get_tavily_service(),
    )


def get_assistant() -> NexusSingleAgentAssistant:
    return NexusSingleAgentAssistant(
        retriever=get_hybrid_retriever(),
        llm_registry=get_llm_registry(),
        knowledge_repo=get_knowledge_repo(),
        readonly_graph_store=None,
    )


def get_ingest_use_case() -> IngestDocumentUseCase:
    registry = get_llm_registry()
    _, active_llm = registry.resolve()
    return IngestDocumentUseCase(
        parser=get_document_router(),
        embedder=get_embedder(),
        vector_store=get_vector_store(),
        graph_store=get_graph_store(),
        llm=active_llm,
        knowledge_repo=get_knowledge_repo(),
    )


def get_query_use_case() -> QueryKnowledgeUseCase:
    return QueryKnowledgeUseCase(
        embedder=get_embedder(),
        vector_store=get_vector_store(),
        graph_store=get_graph_store(),
        knowledge_repo=get_knowledge_repo(),
        retriever=get_hybrid_retriever(),
        assistant=get_assistant(),
        llm_registry=get_llm_registry(),
    )


def get_knowledge_service() -> KnowledgeService:
    return KnowledgeService(knowledge_repo=get_knowledge_repo())


def has_active_celery_worker() -> bool:
    """Return True ONLY when Celery mode is enabled AND at least one live worker responds to ping."""
    settings = get_settings()
    mode = (settings.USE_CELERY_WORKERS or "auto").strip().lower()
    if mode in ("false", "0", "no", "off", "background"):
        return False
    try:
        from nexusai.workers.celery_app import celery_app

        inspector = celery_app.control.inspect(timeout=0.35)
        pong = inspector.ping()
        return bool(pong)
    except Exception:
        return False


async def run_ingestion_job_safe(
    source_id: UUID,
    file_path: str,
    user_id: UUID,
    cleanup_file: bool = False,
) -> dict[str, Any]:
    """Execute `IngestDocumentUseCase` safely inside FastAPI `BackgroundTasks` or inline mode."""
    use_case = get_ingest_use_case()
    repo = get_knowledge_repo()
    try:
        await use_case.execute(
            source_id=source_id,
            file_path=file_path,
            user_id=user_id,
        )
        updated = await repo.get_source(source_id)
        status_val = updated.status.value if updated else "ready"
        return {
            "source_id": str(source_id),
            "status": status_val,
            "chunks": updated.chunk_count if updated else 0,
        }
    except Exception as exc:
        logger.error("Background ingestion failed for source %s: %s", source_id, exc)
        try:
            current = await repo.get_source(source_id)
            if current is not None:
                current.status = SourceStatus.FAILED
                current.error_message = str(exc)
                await repo.update_source(current)
        except Exception as update_exc:
            logger.warning("Failed updating source %s status to FAILED: %s", source_id, update_exc)
        return {
            "source_id": str(source_id),
            "status": "failed",
            "chunks": 0,
            "error": str(exc),
        }
    finally:
        if cleanup_file:
            Path(file_path).unlink(missing_ok=True)


async def dispatch_document_ingestion(
    source_id: UUID,
    file_path: str,
    user_id: UUID,
    background_tasks: BackgroundTasks | None = None,
    wait_inline: bool = False,
) -> dict[str, Any]:
    """Dispatch ingestion to an active Celery worker, FastAPI `BackgroundTasks`, or inline await."""
    if wait_inline:
        return await run_ingestion_job_safe(source_id, file_path, user_id, cleanup_file=False)

    if has_active_celery_worker():
        from nexusai.workers.tasks.ingestion import ingest_document_task

        ingest_document_task.delay(
            source_id=str(source_id),
            file_path=file_path,
            user_id=str(user_id),
        )
        logger.info("Dispatched source %s to live Celery worker", source_id)
        return {"source_id": str(source_id), "status": "processing", "chunks": 0, "dispatch": "celery"}

    if background_tasks is not None:
        background_tasks.add_task(run_ingestion_job_safe, source_id, file_path, user_id, False)
        logger.info("Scheduled source %s via FastAPI BackgroundTasks (no Celery worker running)", source_id)
        return {"source_id": str(source_id), "status": "processing", "chunks": 0, "dispatch": "background_task"}

    return await run_ingestion_job_safe(source_id, file_path, user_id, cleanup_file=False)
