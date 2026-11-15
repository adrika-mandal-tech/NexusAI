"""Build a reviewable :class:`GraphSchemaProposal` from profiled CSV files.

The proposer is a pure function of the uploaded files. It never touches
Neo4j, which is what makes the result reviewable, re-derivable and safe to
approve.

The LLM is not in this path. Relationship *types* and *labels* are derived
deterministically; the user can edit them in the review step.
"""

from __future__ import annotations

import uuid
from typing import Any

from nexusai.application.graph.csv.type_inference import INFERRED_TO_PROPERTY
from nexusai.core.logging import get_logger
from nexusai.domain.entities.graph_schema import (
    Confidence,
    ConstraintDefinition,
    EdgeIdentity,
    FileProfile,
    GraphSchemaProposal,
    IndexDefinition,
    InferredType,
    KeyDefinition,
    NodeDefinition,
    Normalisation,
    PropertyMapping,
    PropertyType,
    RelationshipDefinition,
    ValidationRule,
    ValidationSeverity,
)

logger = get_logger(__name__)

# Columns that carry the provenance contract.
PROVENANCE_PROPERTIES: dict[str, PropertyType] = {
    "source_file": PropertyType.STRING,
    "source_row": PropertyType.INTEGER,
    "source_id": PropertyType.STRING,
    "ingestion_job_id": PropertyType.STRING,
    "ingested_at": PropertyType.DATETIME,
    "kind": PropertyType.STRING,
}

# The discriminator that keeps CSV-derived nodes out of the document-RAG
# label space.
CSV_NODE_KIND = "csv_ingest"


def _normalisation_for(
    inferred: Any,
    column_name: str,
    may_contain_pii: bool,
) -> list[Normalisation]:
    """The normalisation rules to apply to a column's values."""
    rules: list[Normalisation] = []
    name = column_name.lower()

    if inferred in ("integer", "float"):
        rules.append(Normalisation.STRIP_THOUSANDS)
    if inferred in ("float",) or "amount" in name or "price" in name or "cost" in name:
        rules.append(Normalisation.STRIP_CURRENCY)
    if "percent" in name or "%" in name:
        rules.append(Normalisation.STRIP_PERCENT)
    if inferred == "date":
        rules.append(Normalisation.PARSE_DATE)
    if inferred == "datetime":
        rules.append(Normalisation.PARSE_DATETIME)
    if inferred == "string" and may_contain_pii and ("email" in name or "mail" in name):
        rules.append(Normalisation.TRIM_LOWER)
    if any(flag == "leading_zero_ids" for flag in ()):
        rules.append(Normalisation.LEADING_ZERO_STRING)
    return rules


def _property_name_for(column: str) -> str:
    """A safe Cypher property name derived from a column name."""
    import re

    cleaned = re.sub(r"[^0-9a-zA-Z_]+", "_", column.strip()).strip("_")
    if not cleaned:
        cleaned = "value"
    if cleaned[0].isdigit():
        cleaned = f"c_{cleaned}"
    return cleaned[:120]


def _choose_identity(
    profile: FileProfile, include_pii: bool = False
) -> tuple[list[str], KeyDefinition, Confidence, list[str]]:
    """Pick the identity column(s) for a node.

    The selection rule lives in ``type_inference.best_key_column`` so that
    the proposer and the relationship discovery always agree on which column
    is a file's own identity.

    When PII is excluded, a non-PII key is preferred. If the only key is
    PII it is still used — the identity cannot be dropped, because the node
    would have no key to merge on — and the choice is reported.
    """
    from nexusai.application.graph.csv.type_inference import best_key_column

    warnings: list[str] = []
    best = best_key_column(profile.columns)

    if best is not None and best.may_contain_pii and not include_pii:
        alternatives = [
            column
            for column in profile.columns
            if column.is_candidate_key and not column.may_contain_pii
        ]
        if alternatives:
            alternatives.sort(
                key=lambda c: (int(c.name.lower().endswith("_id")), c.distinct_count),
                reverse=True,
            )
            best = alternatives[0]
        else:
            warnings.append(
                f"the only candidate key, {best.name!r}, looks like PII. It is used as "
                "the node identity because a node cannot exist without a key — review "
                "this before approving."
            )

    if best is None:
        # Fall back to the first column, so the node is still addressable.
        fallback = profile.columns[0] if profile.columns else None
        if fallback is None:
            raise ValueError(f"{profile.filename} has no columns to use as an identity")
        key = KeyDefinition(
            name=_property_name_for(fallback.name),
            columns=[fallback.name],
            confidence=Confidence.LOW,
            evidence="no unique column was found; the first column is used as a fallback identity",
        )
        return [fallback.name], key, Confidence.LOW, warnings

    evidence = "; ".join(best.notes) or "unique column"
    key = KeyDefinition(
        name=_property_name_for(best.name),
        columns=[best.name],
        confidence=best.key_confidence,
        evidence=evidence,
    )
    confidence = best.key_confidence
    alternatives = [
        c for c in profile.columns if c.is_candidate_key and c.name != best.name
    ]
    if alternatives:
        evidence += " (alternatives: " + ", ".join(c.name for c in alternatives) + ")"
    return [best.name], key, confidence, warnings


def _build_node(
    profile: FileProfile,
    include_pii: bool,
) -> tuple[NodeDefinition, list[str]]:
    """Build a node definition for one file."""
    warnings: list[str] = []
    identity_columns, key, key_confidence, identity_warnings = _choose_identity(
        profile, include_pii=include_pii
    )
    warnings.extend(identity_warnings)

    mappings: list[PropertyMapping] = []
    properties: dict[str, PropertyType] = {}
    required: list[str] = []

    for column in profile.columns:
        is_identity = column.name in identity_columns
        if column.may_contain_pii and not include_pii and not is_identity:
            # A PII column is left out of the mapping. The identity column is
            # never left out, because the node would then have no key.
            warnings.append(
                f"column {column.name!r} looks like PII and was excluded from the mapping"
            )
            continue
        if column.inferred_type == "empty" and not is_identity:
            warnings.append(f"column {column.name!r} is entirely empty and was skipped")
            continue
        if column.may_contain_pii and not include_pii and is_identity:
            warnings.append(
                f"column {column.name!r} looks like PII but is used as the node identity, "
                "so it is stored"
            )

        prop_name = _property_name_for(column.name)
        prop_type = column.property_type or PropertyType.STRING
        notes: list[str] = []
        if column.inferred_type == InferredType.MIXED and column.dominant_type:
            # A mostly-date column with one bad value is still a date column.
            # The good values keep their type; the bad ones become null and
            # are reported.
            prop_type = INFERRED_TO_PROPERTY.get(column.dominant_type, prop_type)
            notes.append(
                f"the column mixes types; values are stored as {prop_type.value} and "
                "unconvertible values become null"
            )
        is_identity = column.name in identity_columns

        mappings.append(
            PropertyMapping(
                source_column=column.name,
                target_property=prop_name,
                property_type=prop_type,
                normalisation=_normalisation_for(
                    column.inferred_type, column.name, column.may_contain_pii
                ),
                required=is_identity,
                nullable=not is_identity,
                conversion_error_policy="reject_row" if is_identity else "null_value",
                confidence=column.key_confidence if is_identity else Confidence.MEDIUM,
                notes=notes,
            )
        )
        properties[prop_name] = prop_type
        if is_identity:
            required.append(prop_name)

    properties.update(PROVENANCE_PROPERTIES)

    node = NodeDefinition(
        label=profile.proposed_label,
        source_file=profile.filename,
        source_id=profile.source_id,
        identity_columns=identity_columns,
        key=key,
        property_mappings=mappings,
        properties=properties,
        required_properties=required,
        confidence=key_confidence,
        expected_node_count=profile.total_rows,
        warnings=warnings,
    )
    return node, warnings


def _build_relationship(
    candidate: Any,
    child_node: NodeDefinition,
    parent_node: NodeDefinition,
) -> RelationshipDefinition:
    """Build a relationship definition from a discovered candidate."""
    return RelationshipDefinition(
        relationship_type=candidate.relationship_type,
        source_label=child_node.label,
        target_label=parent_node.label,
        source_file=candidate.source_file,
        source_id=child_node.source_id,
        source_columns=list(candidate.source_columns),
        target_columns=list(candidate.target_columns),
        direction=candidate.direction,
        edge_identity=EdgeIdentity.ENDPOINTS,
        properties=dict(PROVENANCE_PROPERTIES),
        cardinality=candidate.cardinality,
        match_ratio=candidate.match_ratio,
        unmatched_value_count=len(candidate.unmatched_values),
        confidence=candidate.confidence,
        expected_relationship_count=0,
        requires_direction_confirmation=bool(candidate.ambiguities),
        unresolved_ambiguities=list(candidate.ambiguities),
        warnings=list(candidate.warnings),
    )


def _build_constraints(nodes: list[NodeDefinition]) -> list[ConstraintDefinition]:
    """One uniqueness constraint per node, on its identity property."""
    constraints: list[ConstraintDefinition] = []
    for node in nodes:
        if not node.key.columns:
            continue
        prop = _property_name_for(node.key.columns[0])
        constraints.append(
            ConstraintDefinition(
                statement=(
                    f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{node.label}) "
                    f"REQUIRE n.{prop} IS UNIQUE"
                ),
                label=node.label,
                properties=[prop],
                kind="uniqueness",
                rationale=(
                    f"Prevents duplicate {node.label} nodes and makes MERGE-based "
                    "ingestion safe under concurrent runs."
                ),
            )
        )
    return constraints


def _build_indexes(
    nodes: list[NodeDefinition],
    relationships: list[RelationshipDefinition],
) -> list[IndexDefinition]:
    """Indexes for the provenance statistics and the FK lookups."""
    indexes: list[IndexDefinition] = []
    for node in nodes:
        indexes.append(
            IndexDefinition(
                statement=f"CREATE INDEX IF NOT EXISTS FOR (n:{node.label}) ON (n.source_file)",
                label=node.label,
                properties=["source_file"],
                kind="range",
                rationale="Supports the per-file node statistics and selective deletion.",
            )
        )
    for rel in relationships:
        for column in rel.source_columns:
            prop = _property_name_for(column)
            if prop in rel.source_label:
                continue
            indexes.append(
                IndexDefinition(
                    statement=(
                        f"CREATE INDEX IF NOT EXISTS FOR (n:{rel.source_label}) ON (n.{prop})"
                    ),
                    label=rel.source_label,
                    properties=[prop],
                    kind="range",
                    rationale=(
                        f"Supports the {rel.relationship_type} relationship MATCH and the "
                        "missing-relationship validation query."
                    ),
                )
            )
    return indexes


def _build_validation_rules(nodes: list[NodeDefinition]) -> list[ValidationRule]:
    """Default row-level validation rules."""
    rules: list[ValidationRule] = []
    for node in nodes:
        for column in node.identity_columns:
            prop = _property_name_for(column)
            rules.append(
                ValidationRule(
                    rule_id=f"{node.label.lower()}_{prop}_not_null",
                    name=f"{node.label}.{prop} must not be null",
                    scope="node",
                    applies_to=node.label,
                    column=column,
                    condition=f"{prop} IS NOT NULL AND trim({prop}) <> ''",
                    severity=ValidationSeverity.ERROR,
                    enabled=True,
                    error_message=f"Missing {prop} for {node.label}",
                )
            )
    return rules


def build_proposal(
    profiles: list[FileProfile],
    candidates: list[Any] | None = None,
    include_pii: bool = False,
    created_by: str = "",
) -> GraphSchemaProposal:
    """Assemble a complete schema proposal from profiled files.

    ``candidates`` are :class:`~nexusai.application.graph.csv.key_inference.RelationshipCandidate`
    instances. When omitted, no relationships are proposed.
    """
    if not profiles:
        raise ValueError("Cannot build a proposal without at least one profiled file")

    ambiguities: list[str] = []
    warnings: list[str] = []

    # Label collisions are a data-corruption risk — never merge silently.
    labels: dict[str, list[str]] = {}
    for profile in profiles:
        labels.setdefault(profile.proposed_label, []).append(profile.filename)
    for label, filenames in labels.items():
        if len(filenames) > 1:
            ambiguities.append(
                f"label {label!r} is proposed for multiple files ({', '.join(filenames)}). "
                "Rename them before approving — merging different files into one label "
                "would corrupt the graph."
            )

    nodes: list[NodeDefinition] = []
    node_by_file: dict[str, NodeDefinition] = {}
    for profile in profiles:
        if profile.proposed_label in labels and len(labels[profile.proposed_label]) > 1:
            # Skip building a node for an ambiguous label; it cannot be approved.
            continue
        node, node_warnings = _build_node(profile, include_pii=include_pii)
        nodes.append(node)
        node_by_file[profile.filename] = node
        warnings.extend(node_warnings)

    relationships: list[RelationshipDefinition] = []
    node_by_label = {node.label: node for node in nodes}
    for candidate in candidates or []:
        child = node_by_file.get(candidate.source_file)
        parent = node_by_label.get(candidate.target_label)
        if child is None or parent is None:
            # Never drop a candidate silently — record why.
            warnings.append(
                f"a discovered relationship ({candidate.source_file} -> "
                f"{candidate.target_label}) was not included because "
                + ("the source file has no node definition" if child is None
                   else f"the target label {candidate.target_label!r} is not defined")
            )
            continue
        rel = _build_relationship(candidate, child, parent)
        relationships.append(rel)
        ambiguities.extend(rel.unresolved_ambiguities)
        warnings.extend(rel.warnings)

    # A relationship type must identify exactly one label pair. Where two
    # pairs would share a type, disambiguate deterministically and report it
    # rather than silently merging two different edges.
    pairs_by_type: dict[str, set[tuple[str, str]]] = {}
    for rel in relationships:
        pairs_by_type.setdefault(rel.relationship_type, set()).add(
            (rel.source_label, rel.target_label)
        )
    for rel in relationships:
        pairs = pairs_by_type.get(rel.relationship_type, set())
        if len(pairs) > 1:
            rel.relationship_type = f"{rel.relationship_type}_{rel.source_label.upper()}"[:60]
            rel.unresolved_ambiguities.append(
                f"the relationship type was made unique by appending the source label "
                f"({rel.relationship_type})"
            )
            ambiguities.append(
                f"relationship type {rel.relationship_type!r} would connect more than one "
                "label pair; the types were disambiguated automatically — rename them if "
                "you prefer."
            )

    constraints = _build_constraints(nodes)
    indexes = _build_indexes(nodes, relationships)
    rules = _build_validation_rules(nodes)

    confidences = {n.confidence for n in relationships} | {n.confidence for n in nodes}
    if Confidence.LOW in confidences:
        overall = Confidence.LOW
    elif Confidence.MEDIUM in confidences:
        overall = Confidence.MEDIUM
    else:
        overall = Confidence.HIGH

    return GraphSchemaProposal(
        proposal_id=uuid.uuid4(),
        source_ids=[p.source_id for p in profiles],
        files=profiles,
        nodes=nodes,
        relationships=relationships,
        constraints=constraints,
        indexes=indexes,
        validation_rules=rules,
        overall_confidence=overall,
        ambiguities=ambiguities,
        warnings=warnings,
        created_by=created_by,
        requires_approval=True,
    )
