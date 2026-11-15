"""Tests for the four-gate foreign-key discovery.

The most important behaviour: two columns with similar values, or even a
perfect name match, must NOT become a relationship unless the values
actually contain one another.
"""

from __future__ import annotations

import pytest

from nexusai.application.graph.csv.key_inference import (
    _name_relation,
    _types_compatible,
    discover_relationships,
)
from nexusai.domain.entities.graph_schema import Confidence, InferredType
from tests.graph.conftest import distinct_values_of, profile_of


def _discover(*names: str):
    """Discover relationships across the named fixtures."""
    profiles = [profile_of(name) for name in names]
    values = {}
    for name in names:
        for column, distinct in distinct_values_of(name).items():
            values[(name, column)] = distinct
    return discover_relationships(profiles, values)


def test_customer_id_is_a_foreign_key_to_customers() -> None:
    candidates = _discover("customers.csv", "orders.csv")
    matches = [
        c
        for c in candidates
        if c.source_file == "orders.csv"
        and c.source_columns[0] == "customer_id"
        and c.target_label == "Customer"
    ]
    assert len(matches) == 1
    assert matches[0].match_ratio == 1.0
    assert matches[0].confidence == Confidence.HIGH


def test_seller_id_is_a_foreign_key_to_sellers() -> None:
    candidates = _discover("customers.csv", "orders.csv", "sellers.csv")
    matches = [
        c
        for c in candidates
        if c.source_file == "orders.csv"
        and c.source_columns[0] == "seller_id"
        and c.target_label == "Seller"
    ]
    assert len(matches) == 1
    assert matches[0].match_ratio == 1.0


def test_similar_values_without_a_relationship_are_rejected() -> None:
    """customers.country and sellers.country share values but are not a FK.

    This is the behaviour the design calls for explicitly: a name match
    alone must never create a relationship.
    """
    candidates = _discover("customers.csv", "sellers.csv")
    assert not [
        c for c in candidates if c.source_columns[0] == "country"
    ]


def test_a_perfect_name_match_with_poor_overlap_is_rejected() -> None:
    """A column called customer_id whose values do not resolve is not a FK."""
    customers = profile_of("customers.csv")
    orders = profile_of("orders.csv")
    values = {
        ("customers.csv", "customer_id"): {"C001", "C002", "C003", "C004", "C005"},
        ("orders.csv", "customer_id"): {"X999", "X888"},
    }
    candidates = discover_relationships([customers, orders], values)
    assert not [
        c for c in candidates if c.source_columns[0] == "customer_id"
    ]


def test_a_probably_foreign_key_reports_its_unmatched_values() -> None:
    customers = profile_of("customers.csv")
    orders = profile_of("orders_orphan_fk.csv")
    values = {
        ("customers.csv", "customer_id"): {"C001", "C002", "C003", "C004", "C005"},
        ("orders_orphan_fk.csv", "customer_id"): {"C001", "C002", "C999"},
    }
    candidates = discover_relationships([customers, orders], values)
    matches = [c for c in candidates if c.source_columns[0] == "customer_id"]
    assert len(matches) == 1
    assert matches[0].match_ratio == pytest.approx(2 / 3, abs=0.01)
    assert matches[0].unmatched_values == ["C999"]
    assert matches[0].confidence == Confidence.MEDIUM


def test_a_foreign_key_below_the_overlap_threshold_is_rejected() -> None:
    """Too little value overlap means it is not a foreign key at all."""
    customers = profile_of("customers.csv")
    orders = profile_of("orders_orphan_fk.csv")
    values = {
        ("customers.csv", "customer_id"): {"C001", "C002", "C003", "C004", "C005"},
        ("orders_orphan_fk.csv", "customer_id"): {"C001", "X1", "X2", "X3", "X4"},
    }
    candidates = discover_relationships([customers, orders], values)
    assert not [c for c in candidates if c.source_columns[0] == "customer_id"]


def test_a_primary_key_is_not_also_a_foreign_key() -> None:
    """customers.customer_id is the key of its own file, so it is not an FK."""
    candidates = _discover("customers.csv", "orders.csv")
    assert not [
        c for c in candidates if c.source_file == "customers.csv"
    ]


def test_relationship_types_are_distinct() -> None:
    """Two edges from one file must not share a relationship type."""
    candidates = _discover("customers.csv", "orders.csv", "sellers.csv")
    types = {c.relationship_type for c in candidates}
    assert len(types) == len(candidates)
    assert "PLACED" in types
    assert "SOLD_BY" in types


def test_name_relation_strength() -> None:
    assert _name_relation("customer_id", "customer_id") == 1.0
    assert _name_relation("customer_id", "seller_id") == 0.0


def test_type_compatibility() -> None:
    assert _types_compatible(InferredType.STRING, InferredType.STRING)
    assert not _types_compatible(InferredType.DATE, InferredType.STRING)
