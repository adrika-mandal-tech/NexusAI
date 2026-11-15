"""Cross-file foreign-key discovery.

A relationship is proposed only when **all four gates** pass:

1. **Structural candidacy** — the column is not the file's own key, its name
   relates to the parent's key, its type is compatible, and its cardinality
   cannot exceed the parent's.
2. **Value containment** — the decisive test. A share of the child's distinct
   values must appear in the parent's key values. This is what stops two
   columns with similar names from ever becoming a relationship.
3. **Cardinality** — one-to-one, many-to-one or many-to-many, derived from the
   data.
4. **Direction** — the child holds the foreign key, so the edge runs from the
   child to the parent.

A perfect name match with poor value overlap is **rejected**. That is the
single most important behaviour in this module.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from nexusai.application.graph.csv.type_inference import best_key_column
from nexusai.core.config import get_settings
from nexusai.core.logging import get_logger
from nexusai.domain.entities.graph_schema import (
    Confidence,
    FileProfile,
    InferredType,
)

logger = get_logger(__name__)

# Types that can be compared across files.
_COMPATIBLE_TYPES: dict[InferredType, set[InferredType]] = {
    InferredType.STRING: {InferredType.STRING},
    InferredType.CATEGORICAL: {InferredType.STRING, InferredType.CATEGORICAL},
    InferredType.FREE_TEXT: {InferredType.STRING},
    InferredType.INTEGER: {InferredType.INTEGER},
    InferredType.FLOAT: {InferredType.FLOAT, InferredType.INTEGER},
    InferredType.BOOLEAN: {InferredType.BOOLEAN},
    InferredType.DATE: {InferredType.DATE},
    InferredType.DATETIME: {InferredType.DATETIME},
    InferredType.TIMESTAMP: {InferredType.DATETIME},
}

# Relationship verbs, keyed by the *target* label. The target is the thing
# being referenced, so it is the more informative signal: an order that
# references a seller is SOLD_BY, not PLACED.
_VERB_BY_TARGET: tuple[tuple[str, str], ...] = (
    ("seller", "SOLD_BY"),
    ("vendor", "SOLD_BY"),
    ("merchant", "SOLD_BY"),
    ("supplier", "SUPPLIED_BY"),
    ("customer", "PLACED"),
    ("client", "PLACED"),
    ("account", "PLACED"),
    ("buyer", "PLACED"),
    ("product", "CONTAINS"),
    ("item", "CONTAINS"),
    ("employee", "CREATED_BY"),
    ("staff", "CREATED_BY"),
    ("user", "CREATED_BY"),
    ("author", "CREATED_BY"),
    ("manager", "MANAGED_BY"),
    ("parent", "CHILD_OF"),
)

# Fallback verbs, keyed by a pattern on the foreign-key column name.
_VERB_BY_SUFFIX: tuple[tuple[str, str], ...] = (
    ("_by", "CREATED_BY"),
    ("_from", "CONNECTED_TO"),
    ("_to", "CONNECTED_TO"),
    ("_parent", "CHILD_OF"),
)

# Fallback verbs, keyed by the child label.
_VERB_BY_CHILD: tuple[tuple[str, str], ...] = (
    ("order", "PLACED"),
    ("purchase", "PLACED"),
    ("invoice", "ISSUED"),
    ("payment", "SETTLED"),
    ("ticket", "RAISED"),
    ("shipment", "SHIPPED"),
    ("booking", "BOOKED"),
    ("subscription", "SUBSCRIBES"),
)


@dataclass
class RelationshipCandidate:
    """A proposed relationship, with the evidence that justifies it."""

    relationship_type: str
    source_label: str
    target_label: str
    source_file: str
    source_columns: list[str]
    target_columns: list[str]
    direction: str = "outbound"
    match_ratio: float = 0.0
    unmatched_values: list[str] = field(default_factory=list)
    cardinality: str = "many_to_one"
    confidence: Confidence = Confidence.MEDIUM
    name_match: float = 0.0
    evidence: list[str] = field(default_factory=list)
    ambiguities: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _normalise_value(value: str) -> str:
    return (value or "").strip()


def _name_relation(fk_name: str, parent_key: str) -> float:
    """How strongly a foreign-key column name relates to a parent key name.

    Returns 0.0 (no relation), 0.5 (weak) or 1.0 (strong).
    """
    fk = re.sub(r"[^0-9a-z]+", "_", fk_name.lower()).strip("_")
    pk = re.sub(r"[^0-9a-z]+", "_", parent_key.lower()).strip("_")
    if not fk or not pk:
        return 0.0
    if fk == pk:
        return 1.0
    if fk == f"{pk}_id" or fk == f"{pk}_key" or fk == f"{pk}_fk":
        return 1.0
    if fk.endswith(f"_{pk}"):
        return 1.0
    if pk.endswith(f"_{fk}") or fk.endswith("_id") and pk.rstrip("_id") in fk:
        return 0.5
    # The child's column may be named after the parent without the _id suffix.
    if fk.startswith(f"{pk}_") or fk.endswith(f"_{pk}"):
        return 0.5
    return 0.0


def _types_compatible(child: InferredType, parent: InferredType) -> bool:
    allowed = _COMPATIBLE_TYPES.get(child, set())
    return parent in allowed


def _pick_relationship_type(
    fk_column: str,
    child_label: str,
    parent_label: str,
) -> str:
    """Choose a relationship type name for the edge.

    The target label is consulted first, because the target is what the edge
    points at: an order that references a seller is ``SOLD_BY``, while the
    same order referencing a customer is ``PLACED``.
    """
    parent_lower = parent_label.lower()
    for keyword, verb in _VERB_BY_TARGET:
        if keyword in parent_lower:
            return verb

    lowered = fk_column.lower()
    for suffix, verb in _VERB_BY_SUFFIX:
        if lowered.endswith(suffix):
            return verb

    child_lower = child_label.lower()
    for keyword, verb in _VERB_BY_CHILD:
        if keyword in child_lower:
            return verb

    # Fall back to a readable, deterministic name.
    return f"RELATES_TO_{parent_label.upper()}"[:60]


def _determine_cardinality(
    fk_distinct: int,
    child_rows: int,
    parent_key_distinct: int,
    parent_rows: int,
) -> str:
    if fk_distinct == child_rows and parent_key_distinct == parent_rows:
        return "one_to_one"
    if fk_distinct == child_rows:
        return "many_to_one"
    if parent_key_distinct != parent_rows:
        return "many_to_many"
    return "many_to_one"


def discover_relationships(
    profiles: list[FileProfile],
    distinct_values: dict[tuple[str, str], set[str]] | None = None,
) -> list[RelationshipCandidate]:
    """Find foreign-key relationships across a set of profiled files.

    ``distinct_values`` maps ``(filename, column)`` to the column's distinct
    values. When omitted, the columns' sampled values are used — the caller
    should supply it when precise containment matters.
    """
    settings = get_settings()
    distinct_values = distinct_values or {}

    def values_for(filename: str, column: str) -> set[str]:
        key = (filename, column)
        if key in distinct_values:
            return distinct_values[key]
        profile = next((p for p in profiles if p.filename == filename), None)
        if profile is None:
            return set()
        col = profile.column(column)
        if col is None:
            return set()
        return {str(v) for v in col.sample_values}

    # Parent keys: every candidate key in every file.
    parent_keys: list[tuple[FileProfile, Any]] = []
    for profile in profiles:
        for column in profile.columns:
            if column.is_candidate_key:
                parent_keys.append((profile, column))

    candidates: list[RelationshipCandidate] = []

    for child in profiles:
        # Only the file's *own* identity column is excluded from foreign-key
        # candidacy. A column may legitimately be both a key and a reference.
        child_identity = best_key_column(child.columns)
        for child_column in child.columns:
            if child_identity is not None and child_column.name == child_identity.name:
                continue
            if child_column.inferred_type in (InferredType.EMPTY, InferredType.MIXED):
                continue

            for parent, parent_column in parent_keys:
                if parent.filename == child.filename:
                    continue

                # ── Gate 1: structural candidacy ──
                name_match = _name_relation(child_column.name, parent_column.name)
                if name_match == 0.0:
                    continue
                if not _types_compatible(
                    child_column.inferred_type, parent_column.inferred_type
                ):
                    continue
                if child_column.distinct_count > parent_column.distinct_count:
                    # A foreign key cannot have more distinct values than the
                    # key it references.
                    continue

                # ── Gate 2: value containment (decisive) ──
                child_values = {
                    _normalise_value(v)
                    for v in values_for(child.filename, child_column.name)
                    if _normalise_value(v)
                }
                parent_values = {
                    _normalise_value(v)
                    for v in values_for(parent.filename, parent_column.name)
                    if _normalise_value(v)
                }
                if not child_values or not parent_values:
                    continue

                overlap = len(child_values & parent_values) / len(child_values)
                if overlap < settings.CSV_FK_OVERLAP_MIN:
                    # Rejected even with a perfect name match. This is the
                    # behaviour that stops similar-looking columns becoming
                    # relationships.
                    logger.debug(
                        "Rejecting FK candidate %s.%s -> %s.%s: overlap %.2f below threshold",
                        child.filename,
                        child_column.name,
                        parent.filename,
                        parent_column.name,
                        overlap,
                    )
                    continue

                unmatched = sorted(child_values - parent_values)

                # ── Gate 3: cardinality ──
                cardinality = _determine_cardinality(
                    child_column.distinct_count,
                    child.total_rows,
                parent_column.distinct_count,
                parent.total_rows,
                )

                # ── Gate 4: direction ──
                direction = "outbound"

                rel_type = _pick_relationship_type(
                    child_column.name, child.proposed_label, parent.proposed_label
                )

                overlap_score = min(1.0, overlap / settings.CSV_FK_OVERLAP_STRONG)
                confidence_score = (
                    0.35 * name_match + 0.45 * overlap_score + 0.20 * (1.0 if cardinality != "many_to_many" else 0.0)
                )
                if confidence_score >= 0.8:
                    confidence = Confidence.HIGH
                elif confidence_score >= 0.5:
                    confidence = Confidence.MEDIUM
                else:
                    confidence = Confidence.LOW

                ambiguities: list[str] = []
                warnings: list[str] = []
                if unmatched:
                    # Some foreign-key values do not resolve, so the
                    # relationship is only probable, never certain.
                    confidence = Confidence.MEDIUM if confidence == Confidence.HIGH else confidence
                    warnings.append(
                        f"{len(unmatched)} value(s) in {child_column.name} do not resolve "
                        f"against {parent.proposed_label}.{parent_column.name}: "
                        f"{', '.join(unmatched[:5])}"
                    )
                if cardinality == "many_to_many":
                    ambiguities.append(
                        f"{parent.proposed_label}.{parent_column.name} is not unique; "
                        "the edge may need a business key to stay distinct"
                    )

                candidates.append(
                    RelationshipCandidate(
                        relationship_type=rel_type,
                        source_label=child.proposed_label,
                        target_label=parent.proposed_label,
                        source_file=child.filename,
                        source_columns=[child_column.name],
                        target_columns=[parent_column.name],
                        direction=direction,
                        match_ratio=round(overlap, 4),
                        unmatched_values=unmatched,
                        cardinality=cardinality,
                        confidence=confidence,
                        name_match=name_match,
                        evidence=[
                            f"{child.filename}.{child_column.name} -> "
                            f"{parent.filename}.{parent_column.name}",
                            f"name match {name_match:.1f}, value overlap {overlap:.1%}",
                            f"cardinality {cardinality}",
                        ],
                        ambiguities=ambiguities,
                        warnings=warnings,
                    )
                )

    # Keep the strongest candidate per (child column, parent label) pair.
    best: dict[tuple[str, str, str], RelationshipCandidate] = {}
    for candidate in candidates:
        key = (
            candidate.source_file,
            candidate.source_columns[0],
            candidate.target_label,
        )
        existing = best.get(key)
        if existing is None or candidate.match_ratio > existing.match_ratio:
            best[key] = candidate

    return sorted(
        best.values(),
        key=lambda c: (c.source_file, c.source_columns[0], c.target_label),
    )


def collect_distinct_values(
    profiles: list[FileProfile],
    read_values: Any,
) -> dict[tuple[str, str], set[str]]:
    """Gather the distinct values of every column, using a streaming reader.

    ``read_values`` is a callable ``(filename, column) -> set[str]``.
    """
    result: dict[tuple[str, str], set[str]] = {}
    for profile in profiles:
        for column in profile.columns:
            result[(profile.filename, column.name)] = set(read_values(profile.filename, column.name))
    return result
