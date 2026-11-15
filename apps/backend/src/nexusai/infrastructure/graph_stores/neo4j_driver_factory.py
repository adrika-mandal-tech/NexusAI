"""Process-wide Neo4j driver lifecycle.

Two drivers are maintained:

* **writer** — the administrative account. Used for ingestion, schema DDL and
  destructive maintenance. Never used to serve a user-facing natural-language
  query.
* **reader** — a dedicated read-only account. Used exclusively by the
  natural-language graph query path. When it is not configured the NL path is
  **disabled** (fail-closed); it must never silently fall back to the writer.

Drivers own a connection pool. Creating one per request exhausts sockets, so
they are created once per process and closed on shutdown.
"""

from __future__ import annotations

import threading
from typing import Any

from neo4j import AsyncDriver, AsyncGraphDatabase

from nexusai.core.config import get_settings
from nexusai.core.logging import get_logger

logger = get_logger(__name__)

_lock = threading.Lock()
_writer_driver: AsyncDriver | None = None
_reader_driver: AsyncDriver | None = None


def _build_driver(user: str, password: str) -> AsyncDriver:
    settings = get_settings()
    return AsyncGraphDatabase.driver(
        settings.NEO4J_URL,
        auth=(user, password),
        max_connection_pool_size=settings.NEO4J_POOL_SIZE,
        max_connection_lifetime=settings.NEO4J_CONNECTION_LIFETIME,
    )


def init_neo4j_drivers() -> None:
    """Create the writer and (optionally) the read-only driver.

    Idempotent: calling this more than once is a no-op. The Celery worker
    calls it too, because the worker is a separate process.
    """
    global _writer_driver, _reader_driver
    with _lock:
        if _writer_driver is not None:
            return

        settings = get_settings()
        _writer_driver = _build_driver(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
        logger.info("Neo4j writer driver initialised")

        if settings.NEO4J_READONLY_USER and settings.NEO4J_READONLY_PASSWORD:
            _reader_driver = _build_driver(
                settings.NEO4J_READONLY_USER, settings.NEO4J_READONLY_PASSWORD
            )
            logger.info("Neo4j read-only driver initialised")
        else:
            _reader_driver = None
            logger.warning(
                "NEO4J_READONLY_USER / NEO4J_READONLY_PASSWORD are not configured — "
                "the natural-language graph query path is DISABLED."
            )


async def close_neo4j_drivers() -> None:
    """Close both drivers and clear the cached instances."""
    global _writer_driver, _reader_driver
    with _lock:
        for name, driver in (("reader", _reader_driver), ("writer", _writer_driver)):
            if driver is None:
                continue
            try:
                await driver.close()
                logger.info("Neo4j %s driver closed", name)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Error closing Neo4j %s driver: %s", name, exc)
        _writer_driver = None
        _reader_driver = None


def get_writer_driver() -> AsyncDriver:
    """Return the administrative driver, creating it on first use."""
    if _writer_driver is None:
        init_neo4j_drivers()
    assert _writer_driver is not None, "Neo4j writer driver failed to initialise"
    return _writer_driver


def get_reader_driver() -> AsyncDriver | None:
    """Return the read-only driver, or ``None`` when it is not configured.

    ``None`` means the natural-language query path must refuse to run.
    """
    if _writer_driver is None:
        init_neo4j_drivers()
    return _reader_driver


def reader_is_configured() -> bool:
    """Whether the read-only account is available."""
    return get_reader_driver() is not None


def get_database_name() -> str:
    """The Neo4j database to operate on."""
    return get_settings().NEO4J_DATABASE


async def check_health() -> dict[str, Any]:
    """Health probe for both drivers. Never raises."""
    results: dict[str, Any] = {}
    for name, getter in (("writer", get_writer_driver), ("reader", get_reader_driver)):
        driver = getter()
        if driver is None:
            results[name] = {"status": "not_configured"}
            continue
        try:
            async with driver.session(database=get_database_name()) as session:
                await session.run("RETURN 1")
            results[name] = {"status": "ok"}
        except Exception as exc:  # noqa: BLE001
            results[name] = {"status": "error", "detail": str(exc)}
    return results
