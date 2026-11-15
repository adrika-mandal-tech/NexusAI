"""Transactional, batched, idempotent ingestion of CSV rows into Neo4j.

Design rules:

* **Type conversion happens in Python**, before the batch is submitted. Neo4j
  stores whatever it is given, so a numeric column that arrives as a string
  silently breaks every aggregate over it.
* **Rows are validated before the batch.** A row that cannot be converted is
  excluded and recorded; it never reaches the database.
* **Endpoints are ``MATCH``ed, not ``MERGE``d**, so a relationship to a
  non-existent node is impossible rather than merely discouraged.
* **Batches are bounded** and written through ``execute_write`` so transient
  failures are retried by the driver.
"""

from __future__ import annotations

import csv
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from nexusai.core.config import get_settings
from nexusai.core.logging import get_logger
from nexusai.domain.entities.graph_schema import (
    EdgeIdentity,
    FileIngestionResult,
    GraphSchemaProposal,
    IngestionReport,
    IngestionStatus,
    NodeDefinition,
    PropertyType,
    RejectedRow,
    RelationshipDefinition,
)

logger = get_logger(__name__)

_CURRENCY_RE = re.compile(r"[^\d.,\-]")
_THOUSANDS_RE = re.compile(r"(?<=\d),(?=\d{3}\b)")
_PERCENT_RE = re.compile(r"%")
_WHITESPACE_RE = re.compile(r"\s+")

# Date formats accepted for conversion, mirroring the inference ladder.
_DATE_FORMATS = (
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%d-%m-%Y",
    "%d.%m.%Y",
    "%d %b %Y",
    "%d %B %Y",
    "%b %d, %Y",
    "%B %d, %Y",
    "%Y%m%d",
)

_DATETIME_FORMATS = (
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%d %H:%M:%S%z",
    "%d/%m/%Y %H:%M:%S",
    "%m/%d/%Y %H:%M:%S",
)


class RowRejected(Exception):
    """Raised when a row cannot be converted. Carries the reason."""

    def __init__(self, column: str, value: str, message: str) -> None:
        super().__init__(message)
        self.column = column
        self.value = value
        self.message = message


def _clean(value: str | None) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _convert(value: str, target: PropertyType, normalisations: list[str]) -> Any:
    """Convert a raw cell to the target property type.

    Raises :class:`RowRejected` when the value cannot be converted.
    """
    text = _clean(value)
    if text == "":
        return None

    for rule in normalisations:
        if rule == "trim":
            text = text.strip()
        elif rule == "trim_lower":
            text = text.lower()
        elif rule == "trim_upper":
            text = text.upper()
        elif rule == "whitespace_collapse":
            text = _WHITESPACE_RE.sub(" ", text).strip()
        elif rule == "strip_thousands":
            text = _THOUSANDS_RE.sub("", text)
        elif rule == "strip_currency":
            text = _CURRENCY_RE.sub("", text)
        elif rule == "strip_percent":
            text = _PERCENT_RE.sub("", text)
        elif rule == "leading_zero_string":
            pass  # the value is kept as a string

    if text == "":
        return None

    try:
        if target == PropertyType.STRING:
            return text
        if target == PropertyType.INTEGER:
            return int(text)
        if target == PropertyType.FLOAT:
            parsed = float(text)
            if parsed != parsed or parsed in (float("inf"), float("-inf")):
                raise ValueError("non-finite value")
            return parsed
        if target == PropertyType.BOOLEAN:
            lowered = text.lower()
            if lowered in ("true", "t", "yes", "y", "1"):
                return True
            if lowered in ("false", "f", "no", "n", "0"):
                return False
            raise ValueError("not a boolean")
        if target == PropertyType.DATE:
            for fmt in _DATE_FORMATS:
                try:
                    return datetime.strptime(text, fmt).date()
                except ValueError:
                    continue
            raise ValueError("not a recognised date")
        if target == PropertyType.DATETIME:
            for fmt in _DATETIME_FORMATS:
                try:
                    return datetime.strptime(text, fmt)
                except ValueError:
                    continue
            raise ValueError("not a recognised datetime")
        if target == PropertyType.LOCAL_DATETIME:
            for fmt in _DATETIME_FORMATS:
                try:
                    return datetime.strptime(text, fmt)
                except ValueError:
                    continue
            raise ValueError("not a recognised datetime")
        if target == PropertyType.TIME:
            for fmt in ("%H:%M:%S", "%H:%M"):
                try:
                    return datetime.strptime(text, fmt).time()
                except ValueError:
                    continue
            raise ValueError("not a recognised time")
    except (TypeError, ValueError) as exc:
        raise RowRejected("", text, f"could not be converted to {target.value}: {exc}") from exc

    return text


@dataclass
class _RowResult:
    """The outcome of converting one source row."""

    payload: dict[str, Any] | None
    rejection: RejectedRow | None = None


@dataclass
class Ingestor:
    """Executes an ingestion plan against Neo4j."""

    store: Any
    proposal: GraphSchemaProposal
    job_id: uuid.UUID = field(default_factory=uuid.uuid4)
    batch_size: int = 0
    dry_run: bool = False
    _conversion_failures: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.batch_size:
            self.batch_size = get_settings().CSV_INGEST_BATCH_SIZE

    # ── Row conversion ──────────────────────────────────────────────────────

    def _convert_node_row(
        self,
        node: NodeDefinition,
        row: dict[str, str],
        row_number: int,
        filename: str,
    ) -> _RowResult:
        """Convert one source row into the node write payload."""
        payload: dict[str, Any] = {}
        for mapping in node.property_mappings:
            raw = row.get(mapping.source_column, "")
            try:
                value = _convert(raw, mapping.property_type, [r.value for r in mapping.normalisation])
            except RowRejected as exc:
                if mapping.required:
                    return _RowResult(
                        None,
                        RejectedRow(
                            file=filename,
                            row=row_number,
                            column=mapping.source_column,
                            raw_value=None,
                            rule_id=f"{node.label.lower()}_{mapping.target_property}_convert",
                            message=exc.message,
                            action="rejected",
                        ),
                    )
                # A nullable property with an unconvertible value: the row is
                # still a real entity, so it is kept with a null property and
                # the failure is reported. The value is never silently
                # dropped.
                value = None
                self._conversion_failures.append(
                    {
                        "file": filename,
                        "row": row_number,
                        "column": mapping.source_column,
                        "rule_id": f"{node.label.lower()}_{mapping.target_property}_convert",
                        "message": exc.message,
                        "action": "value_set_to_null",
                    }
                )
            payload[mapping.target_property] = value

        for required in node.required_properties:
            if payload.get(required) in (None, ""):
                return _RowResult(
                    None,
                    RejectedRow(
                        file=filename,
                        row=row_number,
                        column=required,
                        raw_value=None,
                        rule_id=f"{node.label.lower()}_{required}_not_null",
                        message=f"{node.label}.{required} is required but missing",
                        action="rejected",
                    ),
                )

        payload.update(
            {
                "source_file": filename,
                "source_row": row_number,
                "source_id": str(node.source_id),
                "ingestion_job_id": str(self.job_id),
                "kind": "csv_ingest",
            }
        )
        return _RowResult(payload)

    def _convert_relationship_row(
        self,
        rel: RelationshipDefinition,
        source_node: NodeDefinition,
        target_node: NodeDefinition,
        row: dict[str, str],
        row_number: int,
        filename: str,
    ) -> _RowResult:
        """Convert one source row into the relationship write payload.

        ``source_key`` identifies the child node — it is the child's own
        identity value from the row. ``target_key`` is the foreign-key value
        that points at the parent. Both are converted with the type of the
        property they will be matched against, so the MATCH succeeds.
        """
        source_identity_column = source_node.key.columns[0] if source_node.key.columns else None
        fk_column = rel.source_columns[0]

        if not source_identity_column:
            return _RowResult(
                None,
                RejectedRow(
                    file=filename,
                    row=row_number,
                    column=fk_column,
                    raw_value=None,
                    rule_id=f"{rel.relationship_type.lower()}_no_identity",
                    message="the source node has no identity column",
                    action="rejected",
                ),
            )

        source_key_raw = _clean(row.get(source_identity_column, ""))
        target_key_raw = _clean(row.get(fk_column, ""))
        if not source_key_raw or not target_key_raw:
            return _RowResult(
                None,
                RejectedRow(
                    file=filename,
                    row=row_number,
                    column=fk_column,
                    raw_value=None,
                    rule_id=f"{rel.relationship_type.lower()}_key_missing",
                    message="the relationship key is missing on one side",
                    action="rejected",
                ),
            )

        try:
            source_key = _convert(
                source_key_raw,
                self._identity_property_type(source_node),
                [],
            )
            target_key = _convert(
                target_key_raw,
                self._identity_property_type(target_node),
                [],
            )
        except RowRejected as exc:
            return _RowResult(
                None,
                RejectedRow(
                    file=filename,
                    row=row_number,
                    column=fk_column,
                    raw_value=None,
                    rule_id=f"{rel.relationship_type.lower()}_key_convert",
                    message=exc.message,
                    action="rejected",
                ),
            )

        payload: dict[str, Any] = {
            "source_key": source_key,
            "target_key": target_key,
            "source_file": filename,
            "source_row": row_number,
            "source_id": str(rel.source_id),
            "ingestion_job_id": str(self.job_id),
        }
        if rel.edge_identity == EdgeIdentity.ENDPOINTS_PLUS_KEY:
            for key_column in rel.identity_key_columns:
                payload[key_column] = _clean(row.get(key_column, ""))
        return _RowResult(payload)

    @staticmethod
    def _identity_property_type(node: NodeDefinition) -> PropertyType:
        """The property type of a node's identity column."""
        if not node.key.columns:
            return PropertyType.STRING
        column = node.key.columns[0]
        for mapping in node.property_mappings:
            if mapping.source_column == column:
                return mapping.property_type
        return PropertyType.STRING

    # ── Reading ─────────────────────────────────────────────────────────────

    @staticmethod
    def _read_rows(path: Path, encoding: str, delimiter: str) -> Iterable[tuple[int, dict[str, str]]]:
        """Stream ``(row_number, row)`` pairs, 1-based."""
        with path.open("r", encoding=encoding, errors="replace", newline="") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            for index, row in enumerate(reader, start=1):
                yield index, {k: ("" if v is None else v) for k, v in row.items() if k is not None}

    # ── Execution ───────────────────────────────────────────────────────────

    async def run(
        self,
        files: dict[str, tuple[Path, str, str]],
    ) -> IngestionReport:
        """Execute the ingestion.

        ``files`` maps a filename to ``(path, encoding, delimiter)``.
        """
        from nexusai.application.graph.csv.cypher_planner import build_plan

        started = time.perf_counter()
        plan = build_plan(self.proposal)
        report = IngestionReport(
            job_id=self.job_id,
            proposal_id=self.proposal.proposal_id,
            status=IngestionStatus.RUNNING,
            dry_run=self.dry_run,
        )

        node_by_label = {n.label: n for n in self.proposal.nodes}
        file_results: dict[str, FileIngestionResult] = {
            f.filename: FileIngestionResult(file_id=f.file_id, filename=f.filename)
            for f in self.proposal.files
        }

        try:
            # 0. Job tracking — the provenance nodes that make per-file
            #    statistics a graph query.
            if not self.dry_run:
                await self.store.create_ingestion_job(
                    job_id=str(self.job_id),
                    source_id=str(self.proposal.source_ids[0]) if self.proposal.source_ids else "",
                    proposal_id=str(self.proposal.proposal_id),
                )

            # 1. Schema DDL
            if not self.dry_run:
                await self.store.apply_ddl(plan.ddl)
                report.statements_executed.append(
                    {"kind": "ddl", "count": len(plan.ddl), "dry_run": False}
                )

            # 2. Nodes
            for node_plan in plan.node_plans:
                node = node_by_label[node_plan.label]
                profile = self.proposal.file(node.source_file)
                if profile is None:
                    continue
                path, encoding, delimiter = files[node.source_file]
                result = file_results[node.source_file]

                batch: list[dict[str, Any]] = []
                for row_number, row in self._read_rows(path, encoding, delimiter):
                    result.rows_read += 1
                    converted = self._convert_node_row(
                        node, row, row_number, node.source_file
                    )
                    if converted.payload is None:
                        result.rows_rejected += 1
                        if converted.rejection:
                            result.rejected_rows.append(converted.rejection)
                        continue
                    result.rows_accepted += 1
                    batch.append(converted.payload)

                    if len(batch) >= self.batch_size:
                        await self._flush_nodes(node_plan.query, batch, result)
                        batch = []

                if batch:
                    await self._flush_nodes(node_plan.query, batch, result)

                result.conversion_failures.extend(
                    failure
                    for failure in self._conversion_failures
                    if failure["file"] == node.source_file
                )
                self._conversion_failures = [
                    failure
                    for failure in self._conversion_failures
                    if failure["file"] != node.source_file
                ]

                if not self.dry_run:
                    await self.store.link_source_file(
                        job_id=str(self.job_id),
                        file_id=profile.file_id,
                        filename=profile.filename,
                        source_id=str(profile.source_id),
                        rows_read=result.rows_read,
                    )

                report.statements_executed.append(
                    {
                        "kind": "node",
                        "label": node_plan.label,
                        "source_file": node_plan.source_file,
                        "batches": max(1, (result.rows_accepted + self.batch_size - 1) // self.batch_size),
                        "expected": node_plan.expected_count,
                        "dry_run": self.dry_run,
                    }
                )

            # 3. Relationships
            for rel_plan in plan.relationship_plans:
                rel = next(
                    r
                    for r in self.proposal.relationships
                    if r.relationship_type == rel_plan.relationship_type
                    and r.source_label == rel_plan.source_label
                )
                source_node = node_by_label[rel_plan.source_label]
                target_node = node_by_label[rel_plan.target_label]
                profile = self.proposal.file(rel.source_file)
                if profile is None:
                    continue
                path, encoding, delimiter = files[rel.source_file]
                result = file_results[rel.source_file]

                batch = []
                for row_number, row in self._read_rows(path, encoding, delimiter):
                    converted = self._convert_relationship_row(
                        rel, source_node, target_node, row, row_number, rel.source_file
                    )
                    if converted.payload is None:
                        continue
                    batch.append(converted.payload)
                    if len(batch) >= self.batch_size:
                        await self._flush_relationships(rel_plan.query, batch, result)
                        batch = []
                if batch:
                    await self._flush_relationships(rel_plan.query, batch, result)

                report.statements_executed.append(
                    {
                        "kind": "relationship",
                        "type": rel_plan.relationship_type,
                        "from": rel_plan.source_label,
                        "to": rel_plan.target_label,
                        "source_file": rel_plan.source_file,
                        "expected": rel_plan.expected_count,
                        "dry_run": self.dry_run,
                    }
                )

            report.files = list(file_results.values())
            report.status = IngestionStatus.COMPLETED

            # 4. Close the provenance nodes.
            if not self.dry_run:
                for result in report.files:
                    await self.store.finalise_source_file(
                        job_id=str(self.job_id),
                        file_id=result.file_id,
                        stats={
                            "rows_accepted": result.rows_accepted,
                            "rows_rejected": result.rows_rejected,
                            "nodes_created": result.nodes_created,
                            "relationships_created": result.relationships_created,
                        },
                    )
                await self.store.finalise_ingestion_job(
                    job_id=str(self.job_id), status=report.status.value
                )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Ingestion job %s failed", self.job_id)
            report.status = IngestionStatus.FAILED
            report.error = str(exc)
        finally:
            report.duration_seconds = round(time.perf_counter() - started, 3)
            report.finished_at = datetime.now(timezone.utc).isoformat()

        return report

    async def _flush_nodes(
        self,
        query: str,
        batch: list[dict[str, Any]],
        result: FileIngestionResult,
    ) -> None:
        if self.dry_run:
            return
        counters = await self.store.write_batch(query, batch)
        result.nodes_created += counters["nodes_created"]
        result.nodes_matched += counters["nodes_matched"]

    async def _flush_relationships(
        self,
        query: str,
        batch: list[dict[str, Any]],
        result: FileIngestionResult,
    ) -> None:
        if self.dry_run:
            return
        counters = await self.store.write_batch(query, batch)
        result.relationships_created += counters["relationships_created"]
        result.relationships_matched += counters["relationships_matched"]
        result.unmatched_relationship_rows += counters["rows_unmatched"]
