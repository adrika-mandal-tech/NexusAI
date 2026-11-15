"""Tests for the Cypher planner.

The generated statements must be parameterised: no CSV value may appear in
the query text, and only validated identifiers may be interpolated.
"""

from __future__ import annotations

import re

import pytest

from nexusai.application.graph.csv.cypher_planner import (
    CypherPlanError,
    build_plan,
    render_plan_for_review,
)
from nexusai.application.graph.csv.key_inference import discover_relationships
from nexusai.application.graph.csv.schema_proposer import build_proposal
from nexusai.domain.entities.graph_schema import (
    EdgeIdentity,
    GraphSchemaProposal,
    KeyDefinition,
    NodeDefinition,
    PropertyType,
    RelationshipDefinition,
)
from tests.graph.conftest import distinct_values_of, profile_of


@pytest.fixture
def proposal() -> GraphSchemaProposal:
    profiles = [profile_of(name) for name in ("customers.csv", "orders.csv", "sellers.csv")]
    values = {}
    for profile in profiles:
        for column, distinct in distinct_values_of(profile.filename).items():
            values[(profile.filename, column)] = distinct
    return build_proposal(profiles, discover_relationships(profiles, values))


def test_the_plan_contains_a_node_write_per_label(proposal: GraphSchemaProposal) -> None:
    plan = build_plan(proposal)
    labels = {p.label for p in plan.node_plans}
    assert labels == {"Customer", "Order", "Seller"}


def test_a_node_write_uses_merge_on_the_identity_property(
    proposal: GraphSchemaProposal,
) -> None:
    plan = build_plan(proposal)
    customer = next(p for p in plan.node_plans if p.label == "Customer")
    assert "MERGE (n:Customer {customer_id: row.customer_id})" in customer.query


def test_a_node_write_binds_every_value_as_a_parameter(
    proposal: GraphSchemaProposal,
) -> None:
    """No CSV value may appear in the query text."""
    plan = build_plan(proposal)
    for node_plan in plan.node_plans:
        for _, source_column in node_plan.property_bindings:
            assert f"row.{source_column}" in node_plan.query
        # A quoted literal would mean data had reached the query text.
        assert not re.search(r"'[^']*'", node_plan.query)


def test_a_relationship_write_matches_both_endpoints(
    proposal: GraphSchemaProposal,
) -> None:
    """Endpoints are MATCHed, never MERGEd — otherwise orphan nodes appear."""
    plan = build_plan(proposal)
    for rel_plan in plan.relationship_plans:
        assert rel_plan.query.count("MATCH") == 2
        assert "MERGE (s)-[r:" in rel_plan.query
        # The only MERGE is the edge itself.
        assert rel_plan.query.count("MERGE") == 1


def test_a_relationship_write_uses_the_child_and_parent_identities(
    proposal: GraphSchemaProposal,
) -> None:
    plan = build_plan(proposal)
    placed = next(
        p for p in plan.relationship_plans if p.relationship_type == "PLACED"
    )
    assert "MATCH (s:Order {order_id: row.source_key})" in placed.query
    assert "MATCH (t:Customer {customer_id: row.target_key})" in placed.query


def test_the_plan_creates_a_uniqueness_constraint_per_node(
    proposal: GraphSchemaProposal,
) -> None:
    plan = build_plan(proposal)
    statements = " ".join(plan.ddl)
    assert "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Customer) REQUIRE n.customer_id IS UNIQUE" in statements
    assert "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Order) REQUIRE n.order_id IS UNIQUE" in statements


def test_the_plan_creates_the_provenance_index(proposal: GraphSchemaProposal) -> None:
    plan = build_plan(proposal)
    statements = " ".join(plan.ddl)
    assert "CREATE INDEX IF NOT EXISTS FOR (n:Customer) ON (n.source_file)" in statements


def test_the_plan_writes_the_provenance_properties(
    proposal: GraphSchemaProposal,
) -> None:
    plan = build_plan(proposal)
    for node_plan in plan.node_plans:
        for prop in ("source_file", "source_row", "source_id", "ingestion_job_id"):
            assert f"n.{prop} = row.{prop}" in node_plan.query


def test_an_edge_identity_with_a_key_includes_the_key(
    proposal: GraphSchemaProposal,
) -> None:
    """A business key on the edge keeps distinct rows as distinct edges."""
    rel = proposal.relationships[0]
    rel.edge_identity = EdgeIdentity.ENDPOINTS_PLUS_KEY
    rel.identity_key_columns = ["order_id"]
    plan = build_plan(proposal)
    target = next(
        p
        for p in plan.relationship_plans
        if p.relationship_type == rel.relationship_type
        and p.source_label == rel.source_label
    )
    assert "row.order_id" in target.query


def test_a_proposal_without_nodes_is_rejected() -> None:
    with pytest.raises(CypherPlanError):
        build_plan(GraphSchemaProposal())


def test_a_relationship_to_an_unknown_label_is_rejected() -> None:
    """A proposal that references an undefined label cannot be built."""
    with pytest.raises(ValueError):
        GraphSchemaProposal(
            nodes=[
                NodeDefinition(
                    label="A",
                    source_file="a.csv",
                    source_id="11111111-1111-1111-1111-111111111111",
                    identity_columns=["id"],
                    key=KeyDefinition(name="id", columns=["id"]),
                    properties={"id": PropertyType.STRING},
                )
            ],
            relationships=[
                RelationshipDefinition(
                    relationship_type="REL",
                    source_label="A",
                    target_label="DoesNotExist",
                    source_file="a.csv",
                    source_id="11111111-1111-1111-1111-111111111111",
                )
            ],
        )


def test_the_rendered_plan_is_reviewable(proposal: GraphSchemaProposal) -> None:
    plan = build_plan(proposal)
    rendered = render_plan_for_review(plan)
    assert "-- NexusAI CSV-to-graph ingestion plan" in rendered
    assert "MERGE (n:Customer" in rendered
    assert "parameter" in rendered.lower()
