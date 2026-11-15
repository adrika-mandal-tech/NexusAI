"""Tests for the streaming CSV profiler."""

from __future__ import annotations

import uuid

import pytest

from nexusai.application.graph.csv.profiler import (
    CsvFile,
    CsvValidationError,
    detect_delimiter,
    normalise_column_name,
    profile_csv,
    propose_label,
    singularise,
)
from tests.graph.conftest import fixture_path, profile_of


def _csv_file(name: str) -> CsvFile:
    path = fixture_path(name)
    return CsvFile(
        file_id=f"f-{name}",
        filename=name,
        path=path,
        source_id=uuid.uuid4(),
        size_bytes=path.stat().st_size,
    )


def test_the_header_is_not_counted_as_a_data_row() -> None:
    profile = profile_of("customers.csv")
    assert profile.total_rows == 5
    assert profile.sampled_rows == 5


def test_the_proposed_label_comes_from_the_filename() -> None:
    assert profile_of("customers.csv").proposed_label == "Customer"
    assert profile_of("orders.csv").proposed_label == "Order"
    assert profile_of("sellers.csv").proposed_label == "Seller"


def test_null_ratios_are_computed() -> None:
    profile = profile_of("customers_malformed.csv")
    name = profile.column("name")
    assert name is not None
    assert name.null_ratio > 0


def test_distinct_counts_are_computed() -> None:
    profile = profile_of("customers.csv")
    country = profile.column("country")
    assert country is not None
    assert country.distinct_count == 3


def test_a_pii_column_is_flagged() -> None:
    profile = profile_of("customers_pii.csv")
    assert profile.column("ssn") is not None
    assert profile.column("ssn").may_contain_pii
    assert profile.column("email") is not None
    assert profile.column("email").may_contain_pii


def test_an_ambiguous_date_is_flagged() -> None:
    profile = profile_of("orders_ambiguous_date.csv")
    column = profile.column("order_date")
    assert column is not None
    assert "ambiguous_date_format" in column.quality_flags


def test_a_duplicate_key_is_detected() -> None:
    profile = profile_of("customers_duplicate_key.csv")
    assert profile.column("customer_id") is not None
    # The repeated id means the column is not unique.
    assert profile.column("customer_id").distinct_count == 2
    assert profile.column("customer_id").is_candidate_key is False


def test_an_empty_file_is_rejected() -> None:
    with pytest.raises(CsvValidationError):
        profile_csv(_csv_file("empty.csv"))


def test_a_small_file_warns_about_low_confidence() -> None:
    profile = profile_of("customers_tiny.csv")
    assert any("fewer than 50 rows" in w for w in profile.warnings)


def test_a_large_sample_is_bounded() -> None:
    """The profiler must not read more than the configured sample size."""
    profile = profile_csv(_csv_file("customers.csv"), sample_limit=2)
    assert profile.sampled_rows == 2
    assert profile.total_rows == 5


def test_a_ragged_row_is_counted() -> None:
    path = fixture_path("customers.csv")
    with path.open("a") as handle:
        handle.write("C006,Short\n")
    try:
        profile = profile_csv(_csv_file("customers.csv"))
        assert profile.ragged_row_count >= 1
    finally:
        path.write_text(
            "customer_id,name,email,country,segment,created_at\n"
            "C001,Acme Corp,info@acme.com,DE,enterprise,2024-01-15\n"
            "C002,Globex,hello@globex.com,US,smb,2024-01-22\n"
            "C003,Initech,contact@initech.com,DE,enterprise,2024-02-05\n"
            "C004,Umbrella,ops@umbrella.com,UK,enterprise,2024-02-11\n"
            "C005,Stark Industries,hello@stark.com,US,smb,2024-03-01\n"
        )


# ── Name helpers ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("plural", "singular"),
    [
        ("customers", "customer"),
        ("addresses", "address"),
        ("boxes", "box"),
        ("status", "status"),
        ("companies", "company"),
        ("cities", "city"),
    ],
)
def test_singularisation(plural: str, singular: str) -> None:
    assert singularise(plural) == singular


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Customer ID", "customer_id"),
        ("customer-id", "customer_id"),
        ("  spaced  ", "spaced"),
        ("123abc", "c_123abc"),
        ("", "column"),
    ],
)
def test_column_names_are_normalised_to_safe_identifiers(raw: str, expected: str) -> None:
    assert normalise_column_name(raw) == expected


def test_a_label_is_always_a_valid_identifier() -> None:
    for name in ("customers.csv", "order items.csv", "2024-sales.csv", "weird!!!.csv"):
        label, _, _ = propose_label(name)
        assert label.isidentifier() or label[0] == "_"


def test_the_delimiter_is_detected() -> None:
    assert detect_delimiter("a,b,c\n1,2,3\n") == ","
    assert detect_delimiter("a;b;c\n1;2;3\n") == ";"
    assert detect_delimiter("a\tb\tc\n1\t2\t3\n") == "\t"
