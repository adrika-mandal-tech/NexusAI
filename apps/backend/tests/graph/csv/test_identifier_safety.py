"""Tests for Cypher identifier safety.

Labels, relationship types and property names are interpolated into Cypher
because Cypher cannot parameterise them. Every identifier that reaches a
query must therefore pass the validator. These tests include a fuzzer, because
the failure mode is silent query injection.
"""

from __future__ import annotations

import random
import string

import pytest

from nexusai.domain.entities.graph_schema import (
    NodeDefinition,
    RelationshipDefinition,
    UnsafeIdentifierError,
    validate_identifier,
)


@pytest.mark.parametrize(
    "value",
    [
        "Cust; DROP CONSTRAINT x--",
        "Cust`",
        "Cust{name}",
        "Cust'",
        'Cust"',
        "Cust n",
        "Cust;DROP",
        "Cust--",
        "Cust/*x*/",
        "1Cust",
        "",
        "Cust\nDROP",
        "Cust) DETACH DELETE (n) //",
    ],
)
def test_dangerous_identifiers_are_rejected(value: str) -> None:
    with pytest.raises(UnsafeIdentifierError):
        validate_identifier(value, "label")


@pytest.mark.parametrize(
    "value",
    ["Customer", "Order_Item", "camelCase", "_private", "A1", "c"],
)
def test_safe_identifiers_are_accepted(value: str) -> None:
    assert validate_identifier(value, "label") == value


def test_fuzzing_never_produces_an_unsafe_identifier() -> None:
    """No random string may yield an identifier that could break a query."""
    alphabet = string.printable
    rng = random.Random(20261020)
    dangerous = set("`'\"{}();\\")
    for _ in range(10_000):
        candidate = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 40)))
        try:
            accepted = validate_identifier(candidate, "label")
        except UnsafeIdentifierError:
            continue
        # If it was accepted, it must contain nothing that could terminate or
        # escape a Cypher identifier.
        assert not (set(accepted) & dangerous)
        assert "\n" not in accepted
        assert " " not in accepted
        assert accepted[0].isalpha() or accepted[0] == "_"


def test_node_label_is_validated_by_the_model() -> None:
    with pytest.raises(ValueError):
        NodeDefinition(
            label="Bad;Label",
            source_file="a.csv",
            source_id="11111111-1111-1111-1111-111111111111",
            identity_columns=["id"],
            key={"name": "id", "columns": ["id"]},
        )


def test_relationship_type_is_validated_by_the_model() -> None:
    with pytest.raises(ValueError):
        RelationshipDefinition(
            relationship_type="REL] //",
            source_label="A",
            target_label="B",
            source_file="a.csv",
            source_id="11111111-1111-1111-1111-111111111111",
        )
