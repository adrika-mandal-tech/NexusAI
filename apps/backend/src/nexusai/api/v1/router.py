"""API v1 router — aggregates all endpoint routers under /api/v1 (and /api alias)."""

from fastapi import APIRouter

from nexusai.api.v1.endpoints import auth, graph, health, knowledge, rag_platform

api_v1_router = APIRouter()
api_v1_router.include_router(auth.router)
api_v1_router.include_router(health.router)
api_v1_router.include_router(knowledge.router)
api_v1_router.include_router(graph.router)
api_v1_router.include_router(rag_platform.router)
