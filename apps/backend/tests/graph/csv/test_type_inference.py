"""Tests for the deterministic type inference and key detection."""

from __future__ import annotations

import pytest

from nexusai.application.graph.csv.type_inference import (
    infer_type,
    normalise_cell,
    score_key_candidate,
)
from nexusai.domain.entities.graph_schema import Confidence, InferredType


def test_integer_column() -> None:
    guess = infer_type(["1", "2", "3"], "count")
    assert guess.inferred_type == InferredType.INTEGER


def test_float_column() -> None:
    guess = infer_type(["1.5", "2.25", "3"], "amount")
    assert guess.inferred_type == InferredType.FLOAT


def test_all_integral_floats_are_integers() -> None:
    guess = infer_type(["42", "7"], "quantity")
    assert guess.inferred_type == InferredType.INTEGER


def test_boolean_column() -> None:
    guess = infer_type(["yes", "no", "true", "false"], "active")
    assert guess.inferred_type == InferredType.BOOLEAN


def test_iso_date_column() -> None:
    guess = infer_type(["2024-01-15", "2024-02-20"], "created_at")
    assert guess.inferred_type == InferredType.DATE
    assert "ambiguous_date_format" not in guess.quality_flags


def test_ambiguous_date_column_is_flagged_not_guessed() -> None:
    """A day/month ambiguity must be flagged, never silently resolved."""
    guess = infer_type(["03/04/2024", "05/06/2024"], "order_date")
    assert guess.inferred_type == InferredType.DATE
    assert "ambiguous_date_format" in guess.quality_flags
    assert len(guess.possible_formats) > 1


def test_unambiguous_date_has_one_format() -> None:
    guess = infer_type(["13/04/2024", "25/12/2024"], "order_date")
    assert guess.inferred_type == InferredType.DATE
    assert "ambiguous_date_format" not in guess.quality_flags


def test_epoch_timestamp_is_not_an_integer() -> None:
    guess = infer_type(["1700000000", "1700003600"], "created_at")
    assert guess.inferred_type == InferredType.TIMESTAMP


def test_an_epoch_value_in_a_non_temporal_column_stays_an_integer() -> None:
    """A numeric identifier must not be mistaken for a timestamp."""
    guess = infer_type(["1700000000", "1700003600"], "reference_number")
    assert guess.inferred_type == InferredType.INTEGER


def test_categorical_column() -> None:
    guess = infer_type(["DE", "US", "FR", "UK"], "country")
    assert guess.inferred_type == InferredType.CATEGORICAL


def test_unique_column_is_not_categorical() -> None:
    """Every value being distinct means an identifier, never a category."""
    guess = infer_type(["C001", "C002", "C003"], "customer_id")
    assert guess.inferred_type == InferredType.STRING


def test_mixed_column() -> None:
    guess = infer_type(["1", "2", "abc", "4"], "value")
    assert guess.inferred_type == InferredType.MIXED
    assert "mixed_types" in guess.quality_flags


def test_empty_column() -> None:
    guess = infer_type(["", "  ", "NA"], "notes")
    assert guess.inferred_type == InferredType.EMPTY


def test_leading_zero_is_flagged() -> None:
    guess = infer_type(["007", "010", "011"], "code")
    assert guess.inferred_type == InferredType.INTEGER
    assert "leading_zero_ids" in guess.quality_flags


@pytest.mark.parametrize(
    "raw",
    ["", "  ", "NA", "n/a", "NULL", "null", "None", "-", "?", "unknown", "#N/A"],
)
def test_null_sentinels_are_recognised(raw: str) -> None:
    assert normalise_cell(raw) == ""


def test_null_sentinels_are_excluded_before_typing() -> None:
    """A null sentinel must not make an otherwise numeric column mixed."""
    guess = infer_type(["1", "NA", "3", ""], "value")
    assert guess.inferred_type == InferredType.INTEGER


# ── Key detection ───────────────────────────────────────────────────────────


def test_unique_id_column_is_a_key() -> None:
    is_key, score, confidence, _ = score_key_candidate(
        "customer_id", ["C001", "C002", "C003"], 3
    )
    assert is_key
    assert score >= 60
    assert confidence in (Confidence.HIGH, Confidence.MEDIUM)


def test_non_unique_column_is_not_a_key() -> None:
    is_key, score, _, _ = score_key_candidate("country", ["DE", "US", "DE"], 3)
    assert not is_key
    assert score < 40


def test_column_with_nulls_is_not_a_key() -> None:
    is_key, score, _, _ = score_key_candidate("customer_id", ["C001", "", "C003"], 3)
    assert not is_key
    assert score < 40


def test_date_column_is_not_nominated_as_a_key() -> None:
    """Uniqueness by coincidence is not a business key."""
    is_key, score, _, evidence = score_key_candidate(
        "created_at", ["2024-01-01", "2024-01-02", "2024-01-03"], 3, InferredType.DATE
    )
    assert not is_key
    assert any("coincidental" in e for e in evidence)


def test_float_column_is_not_nominated_as_a_key() -> None:
    is_key, _, _, _ = score_key_candidate(
        "amount", ["1.5", "2.5", "3.5"], 3, InferredType.FLOAT
    )
    assert not is_key


def test_uuid_shaped_column_scores_higher() -> None:
    values = ["6f1c9a2e-0b3d-4e5f-8a7b-9c0d1e2f3a4b", "7a2d0b3f-1c4e-5f6a-9b8c-0d1e2f3a4b5c"]
    _, with_uuid, _, evidence = score_key_candidate("ref", values, 2)
    assert with_uuid > 0
    assert any("UUID" in e for e in evidence)


def test_small_file_caps_confidence() -> None:
    """A three-row file cannot prove uniqueness."""
    _, _, confidence, evidence = score_key_candidate(
        "customer_id", ["C001", "C002", "C003"], 3
    )
    assert confidence == Confidence.MEDIUM
    assert any("small sample" in e for e in evidence)


def test_small_file_excludes_non_identifier_columns() -> None:
    """In a tiny file, a plain attribute must not become a business key."""
    is_key, _, _, _ = score_key_candidate("country", ["DE", "US"], 2)
    assert not is_key
