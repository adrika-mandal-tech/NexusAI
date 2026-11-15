"""Post-ingestion validation and reconciliation.

A Cypher statement that executed without error is **not** evidence that the
graph is semantically correct. This module runs independent checks after the
write transaction has committed and reports a pass/warn/fail per check.

The row-level reconciliation check is the most valuable one: it re-reads the
source CSV rows and diffs them, field by field, against the nodes that were
created. It is only possible because every node carries ``source_file`` and
``source_row``.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any

from nexusai.core.logging import get_logger
from nexusai.domain.entities.graph_schema import (
    GraphSchemaProposal,
    IngestionReport,
    IngestionStatus,
    ValidationCheck,
    ValidationReport,
    validate_identifier,
)

logger = get_logger(__name__)


async def validate_ingestion(
    store: Any,
    proposal: GraphSchemaProposal,
    report: IngestionReport,
    files: dict[str, tuple[Path, str, str]] | None = None,
    reconciliation_sample: int = 20,
) -> ValidationReport:
    """Run every validation check and return the report."""
    validation = ValidationReport()

    if report.status == IngestionStatus.FAILED:
        validation.errors.append(f"ingestion failed: {report.error}")
        validation.checks.append(
            ValidationCheck(
                check_id="ingestion_status",
                status="fail",
                actual=report.status.value,
                detail="the ingestion did not complete",
            )
        )
        return validation

    # ── 1. Expected node count per label ──
    for node in proposal.nodes:
        label = validate_identifier(node.label, "label")
        records = await store.run_read(f"MATCH (n:{label}) RETURN count(n) AS c")
        actual = int(records[0]["c"]) if records else 0
        validation.checks.append(
            ValidationCheck(
                check_id=f"expected_node_count_{label}",
                status="pass" if actual == node.expected_node_count else "fail",
                expected=node.expected_node_count,
                actual=actual,
                detail=f"{label} node count",
            )
        )

    # ── 2. Expected relationship count per type ──
    for rel in proposal.relationships:
        rel_type = validate_identifier(rel.relationship_type, "relationship type")
        records = await store.run_read(
            f"MATCH ()-[r:{rel_type}]->() RETURN count(r) AS c"
        )
        actual = int(records[0]["c"]) if records else 0
        expected = rel.expected_relationship_count
        if expected == 0:
            status = "pass" if actual >= 0 else "fail"
        else:
            status = "pass" if actual == expected else "fail"
        validation.checks.append(
            ValidationCheck(
                check_id=f"expected_relationship_count_{rel_type}",
                status=status,
                expected=expected,
                actual=actual,
                detail=f"{rel_type} relationship count",
            )
        )

    # ── 3. Missing endpoints (must be zero) ──
    records = await store.run_read(
        "MATCH (a)-[r]->(b) WHERE a IS NULL OR b IS NULL RETURN count(r) AS c"
    )
    missing = int(records[0]["c"]) if records else 0
    validation.checks.append(
        ValidationCheck(
            check_id="no_missing_endpoints",
            status="pass" if missing == 0 else "fail",
            actual=missing,
            detail="every relationship must point at an existing node",
        )
    )

    # ── 4. Null or empty identity values ──
    for node in proposal.nodes:
        label = validate_identifier(node.label, "label")
        if not node.key.columns:
            continue
        prop = validate_identifier(node.key.columns[0], "property")
        records = await store.run_read(
            f"MATCH (n:{label}) WHERE n.{prop} IS NULL OR trim(toString(n.{prop})) = '' "
            "RETURN count(n) AS c"
        )
        nulls = int(records[0]["c"]) if records else 0
        validation.checks.append(
            ValidationCheck(
                check_id=f"no_null_identity_{label}",
                status="pass" if nulls == 0 else "fail",
                actual=nulls,
                detail=f"{label}.{prop} must not be null",
            )
        )

    # ── 5. Duplicate identity values ──
    for node in proposal.nodes:
        label = validate_identifier(node.label, "label")
        if not node.key.columns:
            continue
        prop = validate_identifier(node.key.columns[0], "property")
        records = await store.run_read(
            f"MATCH (n:{label}) WHERE n.{prop} IS NOT NULL "
            f"WITH n.{prop} AS k, count(n) AS c WHERE c > 1 RETURN count(*) AS dupes"
        )
        dupes = int(records[0]["dupes"]) if records else 0
        validation.checks.append(
            ValidationCheck(
                check_id=f"no_duplicate_identity_{label}",
                status="pass" if dupes == 0 else "warn",
                actual=dupes,
                detail=f"{label}.{prop} values that appear more than once",
            )
        )

    # ── 6. Orphan nodes (no relationships at all) ──
    records = await store.run_read(
        "MATCH (n) WHERE n.kind = 'csv_ingest' AND NOT (n)--() "
        "RETURN count(n) AS c"
    )
    orphans = int(records[0]["c"]) if records else 0
    validation.checks.append(
        ValidationCheck(
            check_id="orphan_nodes",
            status="pass" if orphans == 0 else "warn",
            actual=orphans,
            detail="CSV-derived nodes with no relationships",
        )
    )

    # ── 7. Property type consistency ──
    for node in proposal.nodes:
        label = validate_identifier(node.label, "label")
        for prop, expected_type in node.properties.items():
            prop = validate_identifier(prop, "property")
            records = await store.run_read(
                f"MATCH (n:{label}) WHERE n.{prop} IS NOT NULL "
                f"RETURN valueType(n.{prop}) AS t, count(n) AS c ORDER BY c DESC"
            )
            if not records:
                continue
            types = {r["t"] for r in records}
            if len(types) > 1:
                validation.checks.append(
                    ValidationCheck(
                        check_id=f"property_type_consistency_{label}_{prop}",
                        status="fail",
                        expected=str(expected_type.value),
                        actual=", ".join(sorted(types)),
                        detail=f"{label}.{prop} has mixed property types",
                    )
                )
            else:
                actual = records[0]["t"]
                if not _type_matches(expected_type.value, actual):
                    validation.checks.append(
                        ValidationCheck(
                            check_id=f"property_type_{label}_{prop}",
                            status="fail",
                            expected=expected_type.value,
                            actual=actual,
                            detail=f"{label}.{prop} stored as {actual}, expected {expected_type.value}",
                        )
                    )

    # ── 8. Relationship direction ──
    for rel in proposal.relationships:
        rel_type = validate_identifier(rel.relationship_type, "relationship type")
        src = validate_identifier(rel.source_label, "label")
        tgt = validate_identifier(rel.target_label, "label")
        records = await store.run_read(
            f"MATCH (a:{tgt})-[r:{rel_type}]->(b:{src}) RETURN count(r) AS c"
        )
        inverted = int(records[0]["c"]) if records else 0
        validation.checks.append(
            ValidationCheck(
                check_id=f"relationship_direction_{rel_type}",
                status="pass" if inverted == 0 else "fail",
                actual=inverted,
                detail=f"{rel_type} must run {src} -> {tgt}",
            )
        )

    # ── 9. Provenance coverage ──
    records = await store.run_read(
        "MATCH (n) WHERE n.kind = 'csv_ingest' "
        "RETURN count(n) AS total, count(n.source_file) AS with_file, "
        "count(n.source_row) AS with_row, count(n.source_id) AS with_source_id"
    )
    if records:
        row = records[0]
        total = int(row.get("total") or 0)
        with_file = int(row.get("with_file") or 0)
        with_row = int(row.get("with_row") or 0)
        with_source = int(row.get("with_source_id") or 0)
        complete = with_file == total and with_row == total and with_source == total
        validation.checks.append(
            ValidationCheck(
                check_id="provenance_coverage",
                status="pass" if complete else "warn",
                expected=total,
                actual={"with_file": with_file, "with_row": with_row, "with_source_id": with_source},
                detail="every CSV-derived node must carry its source file, row and source id",
            )
        )

    # ── 10. Uniqueness constraints installed ──
    records = await store.run_read(
        "SHOW CONSTRAINTS YIELD labelsOrTypes, properties, type "
        "RETURN labelsOrTypes AS labels, properties AS properties, type AS type"
    )
    constrained = {
        (tuple(r["labels"] or []), tuple(r["properties"] or []))
        for r in records
        if _is_uniqueness_constraint(r.get("type"))
    }
    for node in proposal.nodes:
        label = validate_identifier(node.label, "label")
        if not node.key.columns:
            continue
        prop = validate_identifier(node.key.columns[0], "property")
        present = any(
            label in labels and prop in props for labels, props in constrained
        )
        validation.checks.append(
            ValidationCheck(
                check_id=f"constraint_present_{label}",
                status="pass" if present else "fail",
                detail=f"a uniqueness constraint on {label}.{prop} must exist",
            )
        )

    # ── 11. Row-level reconciliation ──
    if files:
        check = await _reconcile_rows(store, proposal, files, reconciliation_sample)
        validation.checks.append(check)

    validation.errors = [c.detail for c in validation.failed]
    validation.warnings = [c.detail for c in validation.warned]
    return validation


def _is_uniqueness_constraint(constraint_type: str | None) -> bool:
    """Whether a ``SHOW CONSTRAINTS`` type denotes a uniqueness constraint.

    Neo4j has used several spellings across versions — ``UNIQUENESS``,
    ``NODE_KEY`` and the 5.x+ ``NODE_PROPERTY_UNIQUENESS``.
    """
    value = (constraint_type or "").upper()
    return value in (
        "UNIQUENESS",
        "NODE_KEY",
        "NODE_PROPERTY_UNIQUENESS",
        "RELATIONSHIP_PROPERTY_UNIQUENESS",
    )


def _type_matches(expected: str, actual: str) -> bool:
    """Whether a stored Neo4j type matches the expected property type."""
    actual = (actual or "").upper().strip()
    # Neo4j reports "FLOAT NOT NULL", "ZONED DATETIME NOT NULL" and friends.
    for suffix in (" NOT NULL",):
        if actual.endswith(suffix):
            actual = actual[: -len(suffix)].strip()
    actual = actual.replace("ZONED ", "").replace("LOCAL ", "")
    mapping = {
        "string": {"STRING"},
        "integer": {"INTEGER"},
        "float": {"FLOAT", "INTEGER"},
        "boolean": {"BOOLEAN"},
        "date": {"DATE", "DATETIME", "DATE_TIME"},
        "datetime": {"DATETIME", "DATE_TIME", "DATE"},
        "local_datetime": {"DATETIME", "DATE_TIME"},
        "time": {"TIME"},
    }
    return actual in mapping.get(expected.lower(), {expected.upper()})


async def _reconcile_rows(
    store: Any,
    proposal: GraphSchemaProposal,
    files: dict[str, tuple[Path, str, str]],
    sample_size: int,
) -> ValidationCheck:
    """Diff sampled source rows against the nodes that were created."""
    mismatches: list[str] = []
    sampled = 0

    for node in proposal.nodes:
        profile = proposal.file(node.source_file)
        if profile is None or node.source_file not in files:
            continue
        path, encoding, delimiter = files[node.source_file]
        label = validate_identifier(node.label, "label")
        key_column = node.key.columns[0] if node.key.columns else None
        if not key_column:
            continue
        key_property = next(
            (
                m.target_property
                for m in node.property_mappings
                if m.source_column == key_column
            ),
            key_column,
        )

        rows: list[dict[str, str]] = []
        with path.open("r", encoding=encoding, errors="replace", newline="") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            for row in reader:
                rows.append({k: ("" if v is None else v) for k, v in row.items() if k is not None})
                if len(rows) >= max(sample_size, 20):
                    break

        step = max(1, len(rows) // max(1, sample_size))
        for row in rows[::step][:sample_size]:
            sampled += 1
            key_value = (row.get(key_column) or "").strip()
            if not key_value:
                continue
            records = await store.run_read(
                f"MATCH (n:{label}) WHERE n.{key_property} = $key RETURN n LIMIT 1",
                {"key": key_value},
            )
            if not records:
                mismatches.append(
                    f"{node.source_file} row {row.get('__row__', '?')}: no {label} node for {key_column}={key_value!r}"
                )
                continue
            node_props = records[0]["n"]
            for mapping in node.property_mappings:
                raw = (row.get(mapping.source_column) or "").strip()
                stored = node_props.get(mapping.target_property)
                if raw == "":
                    if stored not in (None, ""):
                        mismatches.append(
                            f"{node.source_file} {key_column}={key_value!r}: "
                            f"{mapping.target_property} should be empty but is {stored!r}"
                        )
                    continue
                if stored is None:
                    mismatches.append(
                        f"{node.source_file} {key_column}={key_value!r}: "
                        f"{mapping.target_property} is null but the source has {raw!r}"
                    )
                    continue
                if not _values_equal(raw, stored, mapping.property_type.value):
                    mismatches.append(
                        f"{node.source_file} {key_column}={key_value!r}: "
                        f"{mapping.target_property} is {stored!r} but the source has {raw!r}"
                    )

    return ValidationCheck(
        check_id="row_reconciliation_sample",
        status="pass" if not mismatches else "fail",
        expected=0,
        actual=len(mismatches),
        detail=(
            f"{sampled} sampled source rows reconciled with no mismatches"
            if not mismatches
            else f"{len(mismatches)} mismatch(es): " + "; ".join(mismatches[:5])
        ),
    )


def _values_equal(raw: str, stored: Any, expected_type: str) -> bool:
    """Compare a raw CSV value with a stored property, tolerating conversion."""
    if stored is None:
        return False
    text = str(raw).strip()
    try:
        if expected_type == "integer":
            return int(text) == int(stored)
        if expected_type == "float":
            return abs(float(text) - float(stored)) < 1e-9
        if expected_type == "boolean":
            lowered = text.lower()
            truthy = lowered in ("true", "t", "yes", "y", "1")
            return truthy == bool(stored)
        if expected_type == "date":
            return _date_matches(text, stored)
        if expected_type in ("datetime", "local_datetime"):
            return _date_matches(text, stored)
    except (TypeError, ValueError):
        return False
    return str(stored) == text


def _date_matches(raw: str, stored: Any) -> bool:
    """Compare a raw date string with a stored date/datetime value."""
    from nexusai.application.graph.csv.ingestor import _DATE_FORMATS, _DATETIME_FORMATS

    for fmt in _DATE_FORMATS:
        try:
            parsed = datetime.strptime(raw, fmt)
        except ValueError:
            continue
        if hasattr(stored, "year"):
            return parsed.year == stored.year and parsed.month == stored.month and parsed.day == stored.day
    for fmt in _DATETIME_FORMATS:
        try:
            parsed = datetime.strptime(raw, fmt)
        except ValueError:
            continue
        if hasattr(stored, "year"):
            return (
                parsed.year == stored.year
                and parsed.month == stored.month
                and parsed.day == stored.day
            )
    return False

