"""Orchestration for the CSV-to-knowledge-graph workflow.

The service owns the state machine:

    upload -> profile -> discover -> propose -> approve -> plan -> ingest -> validate

The proposal is persisted as a ``:GraphSchemaProposal`` node in Neo4j, next to
the graph it describes. That keeps the review state and the data in one
place, and means the workflow does not depend on the metadata database being
available.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nexusai.application.graph.csv.cypher_planner import (
    IngestionPlan,
    build_plan,
)
from nexusai.application.graph.csv.ingestor import Ingestor
from nexusai.application.graph.csv.key_inference import discover_relationships
from nexusai.application.graph.csv.profiler import (
    CsvFile,
    profile_csv,
    read_rows,
)
from nexusai.application.graph.csv.schema_proposer import build_proposal
from nexusai.application.graph.csv.validator import validate_ingestion
from nexusai.core.logging import get_logger
from nexusai.domain.entities.graph_schema import (
    FileProfile,
    GraphSchemaProposal,
    IngestionReport,
    ProposalStatus,
)
from nexusai.infrastructure.csv.csv_loader import StoredCsv

logger = get_logger(__name__)


class ProposalNotFound(LookupError):
    """Raised when a proposal id does not exist."""


class ProposalNotApproved(RuntimeError):
    """Raised when an ingestion is attempted without an approved proposal."""


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class CsvGraphService:
    """Coordinates the CSV-to-graph workflow."""

    def __init__(self, store: Any) -> None:
        self._store = store

    # ── Proposal creation ───────────────────────────────────────────────────

    async def create_proposal(
        self,
        stored: list[StoredCsv],
        include_pii: bool = False,
        created_by: str = "",
    ) -> GraphSchemaProposal:
        """Profile the uploads, discover relationships and build a proposal."""
        profiles: list[FileProfile] = []
        for item in stored:
            csv_file = CsvFile(
                file_id=item.file_id,
                filename=item.filename,
                path=item.path,
                source_id=item.source_id,
                size_bytes=item.size_bytes,
                encoding=item.encoding,
                delimiter=item.delimiter,
                content_hash=item.content_hash,
            )
            profile = profile_csv(csv_file)
            profile.warnings.extend(item.warnings)
            profiles.append(profile)

        distinct_values = self._collect_distinct_values(stored, profiles)
        candidates = discover_relationships(profiles, distinct_values)
        proposal = build_proposal(
            profiles, candidates, include_pii=include_pii, created_by=created_by
        )
        await self._save_proposal(proposal)
        logger.info(
            "Created proposal %s with %d node types and %d relationship types",
            proposal.proposal_id,
            len(proposal.nodes),
            len(proposal.relationships),
        )
        return proposal

    def _collect_distinct_values(
        self,
        stored: list[StoredCsv],
        profiles: list[FileProfile],
    ) -> dict[tuple[str, str], set[str]]:
        """The distinct values of every column, read by streaming."""
        by_name = {item.filename: item for item in stored}
        result: dict[tuple[str, str], set[str]] = {}
        for profile in profiles:
            item = by_name.get(profile.filename)
            if item is None:
                continue
            for column in profile.columns:
                values: set[str] = set()
                try:
                    for row in read_rows(item.path, item.encoding, item.delimiter):
                        raw = (row.get(column.name) or "").strip()
                        if raw:
                            values.add(raw)
                except OSError as exc:
                    logger.warning("Could not read %s: %s", item.path, exc)
                result[(profile.filename, column.name)] = values
        return result

    # ── Persistence ─────────────────────────────────────────────────────────

    async def _save_proposal(self, proposal: GraphSchemaProposal) -> None:
        """Persist a proposal.

        The payload is stored as a JSON string because a Neo4j property can
        only hold a primitive or an array of primitives — a nested map is
        rejected by the server.
        """
        query = """
        MERGE (p:GraphSchemaProposal {id: $proposal_id})
        SET p.version = $version,
            p.status = $status,
            p.created_at = $created_at,
            p.created_by = $created_by,
            p.source_ids = $source_ids,
            p.payload = $payload
        """
        write_fn = getattr(self._store, "run_write", self._store.run_read)
        await write_fn(
            query,
            {
                "proposal_id": str(proposal.proposal_id),
                "version": proposal.version,
                "status": proposal.status.value,
                "created_at": proposal.created_at,
                "created_by": proposal.created_by,
                "source_ids": [str(s) for s in proposal.source_ids],
                "payload": json.dumps(proposal.model_dump(mode="json")),
            },
        )

    async def load_proposal(self, proposal_id: str | uuid.UUID) -> GraphSchemaProposal:
        """Load a stored proposal."""
        records = await self._store.run_read(
            "MATCH (p:GraphSchemaProposal {id: $proposal_id}) RETURN p.payload AS payload",
            {"proposal_id": str(proposal_id)},
        )
        if not records:
            raise ProposalNotFound(f"No proposal with id {proposal_id}")
        payload = records[0]["payload"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        return GraphSchemaProposal.model_validate(payload)

    async def update_proposal(self, proposal: GraphSchemaProposal) -> GraphSchemaProposal:
        """Persist an edited proposal as a new version."""
        proposal.version += 1
        proposal.status = ProposalStatus.PENDING_REVIEW
        proposal.approved_at = None
        proposal.approved_by = None
        await self._save_proposal(proposal)
        return proposal

    async def approve_proposal(
        self, proposal_id: str | uuid.UUID, approved_by: str
    ) -> GraphSchemaProposal:
        """Approve a proposal. This is the only thing that authorises a write."""
        proposal = await self.load_proposal(proposal_id)
        if proposal.status == ProposalStatus.REJECTED:
            raise ProposalNotApproved("A rejected proposal cannot be approved.")
        proposal.status = ProposalStatus.APPROVED
        proposal.approved_at = _utcnow_iso()
        proposal.approved_by = approved_by
        await self._save_proposal(proposal)
        return proposal

    async def reject_proposal(
        self, proposal_id: str | uuid.UUID, reason: str, rejected_by: str
    ) -> GraphSchemaProposal:
        """Reject a proposal."""
        proposal = await self.load_proposal(proposal_id)
        proposal.status = ProposalStatus.REJECTED
        proposal.rejected_at = _utcnow_iso()
        proposal.rejection_reason = reason
        await self._save_proposal(proposal)
        return proposal

    # ── Planning and ingestion ──────────────────────────────────────────────

    def build_plan_for(self, proposal: GraphSchemaProposal) -> IngestionPlan:
        """Build the Cypher plan for an approved proposal."""
        if proposal.status != ProposalStatus.APPROVED:
            raise ProposalNotApproved(
                f"The proposal is {proposal.status.value!r}; it must be approved first."
            )
        return build_plan(proposal)

    async def ingest(
        self,
        proposal: GraphSchemaProposal,
        files: dict[str, tuple[Path, str, str]],
        dry_run: bool = False,
    ) -> IngestionReport:
        """Run the ingestion for an approved proposal."""
        if proposal.status != ProposalStatus.APPROVED:
            raise ProposalNotApproved(
                f"The proposal is {proposal.status.value!r}; it must be approved first."
            )

        ingestor = Ingestor(
            store=self._store,
            proposal=proposal,
            job_id=uuid.uuid4(),
            dry_run=dry_run,
        )
        report = await ingestor.run(files)
        validation = await validate_ingestion(self._store, proposal, report, files)
        report.validation = validation

        if report.status.value == "completed":
            if validation.failed:
                report.status = type(report.status).COMPLETED_WITH_ERRORS
            elif validation.warned:
                report.status = type(report.status).COMPLETED_WITH_WARNINGS

        write_fn = getattr(self._store, "run_write", self._store.run_read)
        await write_fn(
            "MATCH (p:GraphSchemaProposal {id: $proposal_id}) "
            "SET p.status = $status, p.last_job_id = $job_id",
            {
                "proposal_id": str(proposal.proposal_id),
                "status": ProposalStatus.INGESTED.value
                if report.status.value.startswith("completed")
                else proposal.status.value,
                "job_id": str(report.job_id),
            },
        )
        return report

    async def load_report(self, job_id: str | uuid.UUID) -> dict[str, Any] | None:
        """Load a stored ingestion report."""
        records = await self._store.run_read(
            "MATCH (j:IngestionJob {id: $job_id}) RETURN j.report AS report",
            {"job_id": str(job_id)},
        )
        if not records:
            return None
        report = records[0]["report"]
        if isinstance(report, str):
            report = json.loads(report)
        return report
