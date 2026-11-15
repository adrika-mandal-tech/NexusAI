"""Database session management with event-loop-safe async engine pooling."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from sqlalchemy import text

from nexusai.core.config import get_settings
from nexusai.core.logging import get_logger
from nexusai.infrastructure.db.models import Base

logger = get_logger(__name__)

_engine: Any = None
_async_session_maker: Any = None
_engine_loop_id: int | None = None
_db_ready: bool = False


def _current_loop_id() -> int | None:
    try:
        return id(asyncio.get_running_loop())
    except RuntimeError:
        return None


def _get_engine() -> tuple[Any, Any]:
    global _engine, _async_session_maker, _engine_loop_id
    loop_id = _current_loop_id()
    if _engine is None or (_engine is not False and loop_id is not None and _engine_loop_id != loop_id):
        try:
            from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

            settings = get_settings()
            _engine = create_async_engine(
                settings.DATABASE_URL,
                echo=False,
                pool_pre_ping=True,
                pool_size=settings.DATABASE_POOL_SIZE,
                max_overflow=settings.DATABASE_MAX_OVERFLOW,
            )
            _async_session_maker = async_sessionmaker(
                _engine,
                class_=AsyncSession,
                expire_on_commit=False,
            )
            _engine_loop_id = loop_id
        except Exception as exc:
            logger.warning("Database engine not initialized: %s", exc)
            _engine = False
            _async_session_maker = False
            _engine_loop_id = loop_id
    if _engine is False:
        raise RuntimeError("Database is not configured or not available")
    return _engine, _async_session_maker


async def init_db() -> bool:
    """Initialize database tables and recover stuck processing records."""
    global _db_ready
    try:
        engine, _ = _get_engine()
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        _db_ready = True
        return True
    except Exception as exc:
        _db_ready = False
        logger.warning("Database init skipped (falling back if needed): %s", exc)
        return False


async def check_db_health() -> dict[str, Any]:
    """Ping PostgreSQL for deep health checks."""
    try:
        engine, _ = _get_engine()
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return {"status": "healthy", "backend": "postgresql"}
    except Exception as exc:
        return {"status": "degraded", "backend": "local_json_fallback", "detail": str(exc)}


def is_db_ready() -> bool:
    return _db_ready


async def close_db() -> None:
    """Close database connections."""
    global _engine, _async_session_maker, _engine_loop_id, _db_ready
    try:
        if _engine and _engine is not False:
            await _engine.dispose()
    except Exception:
        pass
    finally:
        _engine = None
        _async_session_maker = None
        _engine_loop_id = None
        _db_ready = False


@asynccontextmanager
async def get_session() -> AsyncGenerator[Any, None]:
    """Get a database session (auto-initializes schema once if needed)."""
    if not _db_ready:
        await init_db()
    _, session_maker = _get_engine()
    async with session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def get_db_session() -> AsyncGenerator[Any, None]:
    """FastAPI dependency for database session."""
    async with get_session() as session:
        yield session
