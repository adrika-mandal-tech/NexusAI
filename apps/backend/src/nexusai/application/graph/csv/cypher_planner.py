"""Generate parameterised Cypher from an approved graph schema.

Two hard rules govern everything in this module:

1. **Only validated identifiers are interpolated.** Labels, relationship types
   and property names cannot be parameterised in Cypher, so they are
   interpolated — but each one passes :func:`validate_identifier` first.
2. **Every value is a parameter.** No CSV value ever reaches the query text.

A consequence of rule 2 is that type conversion happens in Python *before*
the batch is submitted, not in Cypher. Cypher would happily store the string
``"1250.50"`` in a numeric property, and a graph with mixed property types
silently breaks every aggregate over it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from nexusai.domain.entities.graph_schema import (
    EdgeIdentity,
    GraphSchemaProposal,
    NodeDefinition,
    RelationshipDefinition,
    validate_identifier,
)


class CypherPlanError(ValueError):
    """Raised when a proposal cannot be turned into a safe plan."""


@dataclass
class NodeWritePlan:
    """How to write one node type."""

    label: str
    identity_property: str
    identity_column: str
    property_bindings: list[tuple[str, str]]  # (target_property, source_column)
    query: str
    expected_count: int = 0
    source_file: str = ""


@dataclass
class RelationshipWritePlan:
    """How to write one relationship type."""

    relationship_type: str
    source_label: str
    target_label: str
    source_identity_property: str
    source_identity_column: str
    target_identity_property: str
    target_identity_column: str
    edge_identity: EdgeIdentity
    identity_key_columns: list[str]
    property_bindings: list[tuple[str, str]]
    query: str
    expected_count: int = 0
    source_file: str = ""


@dataclass
class IngestionPlan:
    """The complete, previewable ingestion plan."""

    proposal_id: str
    ddl: list[str] = field(default_factory=list)
    node_plans: list[NodeWritePlan] = field(default_factory=list)
    relationship_plans: list[RelationshipWritePlan] = field(default_factory=list)
    expected_nodes: int = 0
    expected_relationships: int = 0
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "ddl_statements": len(self.ddl),
            "node_types": [
                {"label": p.label, "expected": p.expected_count, "source_file": p.source_file}
                for p in self.node_plans
            ],
            "relationship_types": [
                {
                    "type": p.relationship_type,
                    "from": p.source_label,
                    "to": p.target_label,
                    "expected": p.expected_count,
                    "source_file": p.source_file,
                }
                for p in self.relationship_plans
            ],
            "expected_nodes": self.expected_nodes,
            "expected_relationships": self.expected_relationships,
            "warnings": self.warnings,
        }

    def all_statements(self) -> list[dict[str, Any]]:
        """Every statement, for the preview UI and the audit log."""
        statements: list[dict[str, Any]] = []
        for ddl in self.ddl:
            statements.append({"kind": "ddl", "text": ddl})
        for plan in self.node_plans:
            statements.append(
                {
                    "kind": "node",
                    "label": plan.label,
                    "source_file": plan.source_file,
                    "expected": plan.expected_count,
                    "text": plan.query,
                }
            )
        for plan in self.relationship_plans:
            statements.append(
                {
                    "kind": "relationship",
                    "type": plan.relationship_type,
                    "source_file": plan.source_file,
                    "expected": plan.expected_count,
                    "text": plan.query,
                }
            )
        return statements


def _identity_property(node: NodeDefinition) -> tuple[str, str]:
    """The (property, source column) used as the MERGE key for a node."""
    if not node.key.columns:
        raise CypherPlanError(f"Node {node.label!r} has no identity column")
    column = node.key.columns[0]
    prop = None
    for mapping in node.property_mappings:
        if mapping.source_column == column:
            prop = mapping.target_property
            break
    if prop is None:
        prop = validate_identifier(column, "identity property")
    return validate_identifier(prop, "identity property"), column


def _node_write_query(node: NodeDefinition) -> str:
    """Build the parameterised node write query."""
    label = validate_identifier(node.label, "label")
    identity_property, _ = _identity_property(node)

    set_clauses = [
        f"n.{validate_identifier(prop, 'property')} = row.{validate_identifier(prop, 'property')}"
        for prop, _ in _property_bindings(node)
    ]
    # Provenance is always written.
    for prop in ("source_file", "source_row", "source_id", "ingestion_job_id", "kind"):
        set_clauses.append(f"n.{prop} = row.{prop}")
    set_clauses.append("n.ingested_at = datetime()")

    body = ",\n        ".join(set_clauses)
    return (
        "UNWIND $rows AS row\n"
        f"MERGE (n:{label} {{{identity_property}: row.{identity_property}}})\n"
        f"SET {body}"
    )


def _property_bindings(node: NodeDefinition) -> list[tuple[str, str]]:
    """(target_property, source_column) pairs for a node's data properties."""
    bindings: list[tuple[str, str]] = []
    for mapping in node.property_mappings:
        bindings.append(
            (
                validate_identifier(mapping.target_property, "property"),
                mapping.source_column,
            )
        )
    return bindings


def _relationship_write_query(
    rel: RelationshipDefinition,
    source_node: NodeDefinition,
    target_node: NodeDefinition,
) -> str:
    """Build the parameterised relationship write query.

    The endpoints are ``MATCH``ed, never ``MERGE``d. ``MERGE``ing an endpoint
    would silently create a node for every unmatched foreign key, inventing
    entities that do not exist in the source data.

    ``row.source_key`` always identifies the ``source_label`` node and
    ``row.target_key`` always identifies the ``target_label`` node. Only the
    direction of the edge changes with ``rel.direction``.
    """
    rel_type = validate_identifier(rel.relationship_type, "relationship type")
    src_prop, _ = _identity_property(source_node)
    tgt_prop, _ = _identity_property(target_node)

    src_label = validate_identifier(source_node.label, "label")
    tgt_label = validate_identifier(target_node.label, "label")

    merge_pattern = f"(s)-[r:{rel_type}]->(t)"
    if rel.edge_identity == EdgeIdentity.ENDPOINTS_PLUS_KEY and rel.identity_key_columns:
        key_props = [validate_identifier(p, "property") for p in rel.identity_key_columns]
        merge_pattern = (
            f"(s)-[r:{rel_type} {{{', '.join(f'{p}: row.{p}' for p in key_props)}}}]->(t)"
        )

    if rel.direction == "inbound":
        merge_pattern = merge_pattern.replace("(s)-[r:", "(t)-[r:").replace("]->(t)", "]->(s)")

    set_clauses = [
        "r.source_file = row.source_file",
        "r.source_row = row.source_row",
        "r.source_id = row.source_id",
        "r.ingestion_job_id = row.ingestion_job_id",
        "r.ingested_at = datetime()",
    ]

    set_body = ",\n        ".join(set_clauses)
    return (
        "UNWIND $rows AS row\n"
        f"MATCH (s:{src_label} {{{src_prop}: row.source_key}})\n"
        f"MATCH (t:{tgt_label} {{{tgt_prop}: row.target_key}})\n"
        f"MERGE {merge_pattern}\n"
        f"SET {set_body}\n"
        # The processed count reveals how many rows matched both endpoints,
        # which is how unmatched foreign keys are detected.
        "RETURN count(r) AS processed"
    )


def build_plan(proposal: GraphSchemaProposal) -> IngestionPlan:
    """Turn an approved proposal into a complete, executable plan."""
    if not proposal.nodes:
        raise CypherPlanError("The proposal defines no nodes")

    plan = IngestionPlan(proposal_id=str(proposal.proposal_id))
    plan.ddl.extend(constraint.statement for constraint in proposal.constraints)
    plan.ddl.extend(index.statement for index in proposal.indexes)

    node_by_label = {node.label: node for node in proposal.nodes}

    for node in proposal.nodes:
        identity_property, identity_column = _identity_property(node)
        plan.node_plans.append(
            NodeWritePlan(
                label=node.label,
                identity_property=identity_property,
                identity_column=identity_column,
                property_bindings=_property_bindings(node),
                query=_node_write_query(node),
                expected_count=node.expected_node_count,
                source_file=node.source_file,
            )
        )
        plan.expected_nodes += node.expected_node_count

    for rel in proposal.relationships:
        source_node = node_by_label.get(rel.source_label)
        target_node = node_by_label.get(rel.target_label)
        if source_node is None or target_node is None:
            raise CypherPlanError(
                f"Relationship {rel.relationship_type!r} references a label that is not "
                f"defined in the proposal ({rel.source_label} -> {rel.target_label})"
            )
        _, source_col = _identity_property(source_node)
        _, target_col = _identity_property(target_node)
        plan.relationship_plans.append(
            RelationshipWritePlan(
                relationship_type=rel.relationship_type,
                source_label=source_node.label,
                target_label=target_node.label,
                source_identity_property=_identity_property(source_node)[0],
                source_identity_column=source_col,
                target_identity_property=_identity_property(target_node)[0],
                target_identity_column=target_col,
                edge_identity=rel.edge_identity,
                identity_key_columns=list(rel.identity_key_columns),
                property_bindings=[],
                query=_relationship_write_query(rel, source_node, target_node),
                expected_count=rel.expected_relationship_count,
                source_file=rel.source_file,
            )
        )
        plan.expected_relationships += rel.expected_relationship_count

    if not plan.ddl:
        plan.warnings.append("The proposal defines no constraints or indexes")

    return plan


def render_plan_for_review(plan: IngestionPlan) -> str:
    """Render the plan as a ``.cypher`` file for download and review."""
    lines: list[str] = [
        "-- NexusAI CSV-to-graph ingestion plan",
        f"-- proposal_id: {plan.proposal_id}",
        "-- All values are supplied as parameters. Only validated identifiers",
        "-- appear in the query text.",
        "",
    ]
    for statement in plan.all_statements():
        label = statement.get("label") or statement.get("type") or statement["kind"]
        lines.append(f"-- ── {statement['kind']}: {label} ──")
        lines.append(statement["text"].rstrip() + ";")
        lines.append("")
    return "\n".join(lines)
