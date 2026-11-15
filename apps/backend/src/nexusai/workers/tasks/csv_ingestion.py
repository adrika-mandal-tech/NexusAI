"""Celery task for CSV-to-graph ingestion.

The ingestion is CPU- and I/O-bound and can run for a long time on a large
file, so it is dispatched to a worker rather than run in the request.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nexusai.core.logging import get_logger
from nexusai.workers.celery_app import celery_app

logger = get_logger(__name__)


@celery_app.task(bind=True, max_retries=3, default_retry_delay=60)
def ingest_csv_task(
    self,
    proposal_id: str,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Ingest an approved CSV proposal into Neo4j."""
    import asyncio

    from nexusai.application.graph.csv.service import CsvGraphService
    from nexusai.infrastructure.graph_stores.neo4j_driver_factory import (
        init_neo4j_drivers,
    )
    from nexusai.infrastructure.graph_stores.neo4j_store import Neo4jGraphStore

    async def _run() -> dict[str, Any]:
        # The worker is its own process, so it owns its own drivers.
        init_neo4j_drivers()
        store = Neo4jGraphStore()
        service = CsvGraphService(store)

        proposal = await service.load_proposal(proposal_id)
        files = _resolve_files(proposal)
        report = await service.ingest(proposal, files, dry_run=dry_run)

        await store.run_read(
            "MATCH (j:IngestionJob {id: $job_id}) SET j.report = $report",
            {"job_id": str(report.job_id), "report": _as_json(report)},
        )
        return report.model_dump(mode="json")

    try:
        return asyncio.run(_run())
    except Exception as exc:  # noqa: BLE001
        logger.error("CSV ingestion task failed: %s", exc)
        raise self.retry(exc=exc)


def _resolve_files(proposal: Any) -> dict[str, tuple[Path, str, str]]:
    """Locate the stored uploads for a proposal's files."""
    from nexusai.core.config import get_settings

    upload_dir = Path(get_settings().UPLOAD_DIR)
    files: dict[str, tuple[Path, str, str]] = {}
    for profile in proposal.files:
        path = upload_dir / f"{profile.file_id}.csv"
        if not path.exists():
            alternative = next(iter(upload_dir.glob(f"{profile.file_id}.*")), None)
            if alternative is None:
                raise FileNotFoundError(
                    f"The uploaded file for {profile.filename} is no longer available."
                )
            path = alternative
        files[profile.filename] = (path, profile.encoding, profile.delimiter)
    return files


def _as_json(report: Any) -> str:
    """The report as a JSON string — a Neo4j property holds primitives only."""
    import json

    return json.dumps(report.model_dump(mode="json"))
