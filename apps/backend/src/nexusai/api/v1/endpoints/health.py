"""Health check & live dependency status endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from nexusai.core.container import (
    get_embedder,
    get_graph_store,
    get_llm_registry,
    get_tavily_service,
    get_vector_store,
)
from nexusai.core.logging import get_logger
from nexusai.core.observability import configure_langsmith_env
from nexusai.infrastructure.db.session import check_db_health

logger = get_logger(__name__)

router = APIRouter(prefix="/health", tags=["health"])


@router.get("")
async def health_check() -> dict[str, Any]:
    """Liveness probe — returns 200 ok when process is responsive."""
    return {"status": "ok"}


@router.get("/dependencies")
@router.get("/deep")
async def health_dependencies() -> dict[str, Any]:
    """Deep readiness probe inspecting Qdrant, Neo4j, PostgreSQL, Embeddings, LLMs, Tavily, & LangSmith."""
    qdrant_info = await get_vector_store().health()
    neo4j_info = await get_graph_store().health()
    pg_info = await check_db_health()
    embedder = get_embedder()
    llm_info = get_llm_registry().status()
    tavily_info = get_tavily_service().health()
    langsmith_info = configure_langsmith_env()

    core_ok = (
        qdrant_info.get("status") == "ok"
        and neo4j_info.get("status") == "ok"
    )
    return {
        "status": "ok" if core_ok else "degraded",
        "dependencies": {
            "qdrant": qdrant_info,
            "neo4j": neo4j_info,
            "database": pg_info,
            "embeddings": {
                "status": "ok",
                "provider": embedder.provider,
                "model": embedder.model_name,
                "dimension": embedder.get_dimension(),
            },
            "llm": llm_info,
            "tavily": tavily_info,
            "langsmith": langsmith_info,
        },
    }
