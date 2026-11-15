"""Celery tasks for document ingestion using SmartDocumentRouter & ConfigurableEmbeddingService."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from nexusai.core.logging import get_logger
from nexusai.workers.celery_app import celery_app

logger = get_logger(__name__)


@celery_app.task(bind=True, max_retries=3, default_retry_delay=30)
def ingest_document_task(
    self: Any,
    source_id: str,
    file_path: str,
    user_id: str,
) -> dict[str, Any]:
    """Celery task to ingest a document asynchronously."""
    import asyncio

    from nexusai.core.container import get_ingest_use_case
    from nexusai.infrastructure.db.session import init_db
    from nexusai.infrastructure.graph_stores.neo4j_driver_factory import (
        init_neo4j_drivers,
    )

    async def _run_ingestion() -> dict[str, Any]:
        await init_db()
        init_neo4j_drivers()
        use_case = get_ingest_use_case()
        await use_case.execute(
            source_id=UUID(source_id),
            file_path=file_path,
            user_id=UUID(user_id),
        )
        return {"status": "completed", "source_id": source_id}

    try:
        return asyncio.run(_run_ingestion())
    except Exception as exc:
        logger.error("Ingestion task failed for %s: %s", source_id, exc)
        try:
            raise self.retry(exc=exc)
        except Exception:
            return {"status": "failed", "source_id": source_id, "error": str(exc)}
