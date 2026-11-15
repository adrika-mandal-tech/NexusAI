"""Tests for the read-only query guard.

The guard is the security boundary for model-generated Cypher. These tests
cover the policy, the parameter contract, and the bounded execution.
"""

from __future__ import annotations

import pytest

from nexusai.infrastructure.graph_stores.neo4j_readonly_store import (
    Neo4jReadOnlyStore,
    ReadOnlyStoreUnavailable,
    ReadOnlyViolation,
)


@pytest.fixture
def guard() -> Neo4jReadOnlyStore:
    """A guard instance that does not need a live database.

    ``validate_query`` and ``validate_parameters`` are pure, so a store with
    no driver is enough for the policy tests.
    """
    store = Neo4jReadOnlyStore.__new__(Neo4jReadOnlyStore)
    store._driver = None
    store._database = "neo4j"
    return store


# ── Write clauses must be rejected ──────────────────────────────────────────


@pytest.mark.parametrize(
    "query",
    [
        "MATCH (n) DELETE n",
        "MATCH (n) DETACH DELETE n",
        "MATCH (n) SET n.x = 1 RETURN n LIMIT 10",
        "MATCH (n) REMOVE n.x RETURN n LIMIT 10",
        "CREATE (n:Probe {x: 1}) RETURN n LIMIT 10",
        "MERGE (n:Probe {x: 1}) RETURN n LIMIT 10",
        "DROP INDEX my_index",
        "MATCH (n) FOREACH (x IN [1] | SET n.p = x) RETURN n LIMIT 10",
        "LOAD CSV FROM 'file:///etc/passwd' AS line RETURN line LIMIT 10",
        "USE system MATCH (n) RETURN n LIMIT 10",
        "CALL apoc.periodic.iterate('MATCH (n) RETURN n', 'RETURN n', {}) YIELD value RETURN value LIMIT 10",
        "CALL db.labels()",
        "CREATE DATABASE evil",
        "GRANT MATCH ON GRAPH neo4j TO neo4j",
    ],
)
def test_write_queries_are_rejected(guard: Neo4jReadOnlyStore, query: str) -> None:
    with pytest.raises(ReadOnlyViolation):
        guard.validate_query(query)


# ── Multi-statement injection must be rejected ──────────────────────────────


def test_multi_statement_is_rejected(guard: Neo4jReadOnlyStore) -> None:
    with pytest.raises(ReadOnlyViolation):
        guard.validate_query(
            "MATCH (n) RETURN n LIMIT 1; MATCH (n) DETACH DELETE n"
        )


def test_a_semicolon_inside_a_string_is_not_a_statement_separator(
    guard: Neo4jReadOnlyStore,
) -> None:
    """A legitimate query with a semicolon in a literal must not be rejected."""
    guard.validate_query("MATCH (n) WHERE n.name = 'a;b' RETURN n LIMIT 10")


# ── Read queries must be accepted ───────────────────────────────────────────


@pytest.mark.parametrize(
    "query",
    [
        "MATCH (n:Customer) RETURN n LIMIT 10",
        "MATCH (n:Customer) RETURN n.customer_id LIMIT $limit",
        "MATCH (c:Customer)<-[:PLACED]-(o:Order) RETURN c, o ORDER BY o.total_amount DESC LIMIT 25",
        "MATCH (n) WHERE n.source_file = $file RETURN n LIMIT 50",
        "MATCH (n) WITH n.name AS name, count(*) AS c WHERE c > 1 RETURN name, c LIMIT 20",
        "MATCH (a)-[r*1..3]->(b) RETURN a, b LIMIT 100",
        "UNWIND $rows AS row MATCH (n:Customer {customer_id: row.id}) RETURN n LIMIT 10",
        "MATCH (n) RETURN DISTINCT n.country LIMIT 10",
        "MATCH (n) OPTIONAL MATCH (n)-[:PLACED]->(m) RETURN n, m LIMIT 10",
    ],
)
def test_read_queries_are_accepted(guard: Neo4jReadOnlyStore, query: str) -> None:
    guard.validate_query(query)


def test_a_query_without_a_limit_is_rejected(guard: Neo4jReadOnlyStore) -> None:
    """An unbounded result set is not permitted."""
    with pytest.raises(ReadOnlyViolation, match="LIMIT"):
        guard.validate_query("MATCH (n) RETURN n")


def test_an_empty_query_is_rejected(guard: Neo4jReadOnlyStore) -> None:
    with pytest.raises(ReadOnlyViolation):
        guard.validate_query("   ")


def test_a_write_inside_a_comment_is_stripped_and_harmless(
    guard: Neo4jReadOnlyStore,
) -> None:
    """A commented-out clause cannot execute, so the query stays read-only."""
    guard.validate_query("MATCH (n) /* CREATE (x:Probe) */ RETURN n LIMIT 10")
    guard.validate_query("MATCH (n) RETURN n LIMIT 10 // DELETE n")


# ── The parameter contract ──────────────────────────────────────────────────


def test_a_literal_value_is_rejected(guard: Neo4jReadOnlyStore) -> None:
    """Data must not enter the query text."""
    with pytest.raises(ReadOnlyViolation, match="literal"):
        guard.validate_parameters("MATCH (n) WHERE n.id = 'C001' RETURN n LIMIT 1", {})


def test_a_numeric_comparison_is_rejected(guard: Neo4jReadOnlyStore) -> None:
    """A value in a comparison position must be a parameter."""
    with pytest.raises(ReadOnlyViolation, match="literal number"):
        guard.validate_parameters("MATCH (n) WHERE n.total = 42 RETURN n LIMIT 1", {})


def test_a_numeric_limit_is_allowed(guard: Neo4jReadOnlyStore) -> None:
    """A LIMIT number is structural and cannot change the shape of a query."""
    guard.validate_parameters("MATCH (n) RETURN n LIMIT 25", {})


def test_a_parameterised_query_is_accepted(guard: Neo4jReadOnlyStore) -> None:
    guard.validate_parameters(
        "MATCH (n) WHERE n.id = $customer_id RETURN n LIMIT $limit",
        {"customer_id": "C001", "limit": 10},
    )


def test_an_invalid_parameter_name_is_rejected(guard: Neo4jReadOnlyStore) -> None:
    with pytest.raises(ReadOnlyViolation):
        guard.validate_parameters("MATCH (n) RETURN n LIMIT 1", {"bad name": 1})


# ── The store is disabled without a read-only account ───────────────────────


def test_the_store_refuses_to_construct_without_a_reader(monkeypatch) -> None:
    """Without a read-only account the path is disabled, never downgraded."""
    import nexusai.infrastructure.graph_stores.neo4j_readonly_store as module

    monkeypatch.setattr(module, "get_reader_driver", lambda: None)
    with pytest.raises(ReadOnlyStoreUnavailable):
        Neo4jReadOnlyStore()
