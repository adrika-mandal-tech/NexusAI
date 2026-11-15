"""FastAPI application factory.

Creates the app, mounts the v1 & /api routers, registers CORS middleware,
LangSmith observability configuration, exception handlers, and structured logging.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from nexusai.api.v1.router import api_v1_router
from nexusai.core.config import get_settings
from nexusai.core.logging import (
    generate_request_id,
    get_logger,
    request_id_ctx,
    setup_logging,
)
from nexusai.core.observability import configure_langsmith_env, set_request_id
from nexusai.infrastructure.db.session import close_db, init_db

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan — startup and shutdown hooks."""
    setup_logging()
    settings = get_settings()
    if (
        settings.ENVIRONMENT.lower() == "production"
        and settings.AUTH_SECRET_KEY == "nexusai-dev-secret-change-in-production"
    ):
        logger.warning(
            "Default AUTH_SECRET_KEY is active in production mode; configure AUTH_SECRET_KEY in environment."
        )
    ls_state = configure_langsmith_env()
    logger.info(
        "NexusAI backend starting up (langsmith_active=%s, project=%s)",
        ls_state["tracing_active"],
        ls_state["project"],
    )
    from nexusai.infrastructure.graph_stores.neo4j_driver_factory import (
        close_neo4j_drivers,
        init_neo4j_drivers,
    )

    try:
        init_neo4j_drivers()
    except Exception as exc:
        logger.warning("Neo4j driver init failed: %s", exc)
    try:
        await init_db()
        logger.info("Database initialized")
    except Exception as exc:
        logger.warning("Database init skipped: %s", exc)
    yield
    logger.info("NexusAI backend shutting down")
    try:
        await close_neo4j_drivers()
    except Exception:
        pass
    try:
        await close_db()
    except Exception:
        pass


def create_app() -> FastAPI:
    """Build and return the FastAPI application."""
    settings = get_settings()

    application = FastAPI(
        title=settings.APP_NAME,
        version="0.2.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # ── CORS ──────────────────────────────────────────────────
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.BACKEND_CORS_ORIGINS,
        allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Request-ID middleware ─────────────────────────────────
    @application.middleware("http")
    async def inject_request_id(request: Request, call_next):  # noqa: ANN001
        rid = request.headers.get("X-Request-ID", generate_request_id())
        request_id_ctx.set(rid)
        set_request_id(rid)
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        return response

    # ── Global exception handler (catches only UNHANDLED errors) ──
    @application.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        from fastapi import HTTPException

        if isinstance(exc, HTTPException):
            raise exc
        logger.error("Unhandled exception: %s", exc, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"},
        )

    # ── Routers (/api/v1 canonical + /api alias) ──────────────
    application.include_router(api_v1_router, prefix=settings.API_V1_PREFIX)
    if settings.API_V1_PREFIX != "/api":
        application.include_router(api_v1_router, prefix="/api", include_in_schema=False)

    # ── Root Health Check ─────────────────────────────────────
    @application.get("/health", tags=["Health"])
    async def health_check() -> dict[str, str]:
        return {
            "status": "ok",
            "service": "NexusAI Backend",
            "version": "0.2.0",
        }

    return application


app = create_app()
