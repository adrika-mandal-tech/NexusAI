"""Shared fixtures for the CSV-to-graph test suites."""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from nexusai.application.graph.csv.profiler import CsvFile, profile_csv, read_rows

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "csv"


def fixture_path(name: str) -> Path:
    """The path of a CSV fixture."""
    return FIXTURE_DIR / name


def profile_of(name: str) -> "object":
    """Profile a CSV fixture."""
    path = fixture_path(name)
    csv_file = CsvFile(
        file_id=f"f-{name}",
        filename=name,
        path=path,
        source_id=uuid.uuid4(),
        size_bytes=path.stat().st_size,
    )
    return profile_csv(csv_file)


def distinct_values_of(name: str) -> dict[str, set[str]]:
    """The distinct values of every column of a CSV fixture."""
    path = fixture_path(name)
    rows = list(read_rows(path, "utf-8", ","))
    columns = rows[0].keys() if rows else []
    return {
        column: {(row.get(column) or "").strip() for row in rows if (row.get(column) or "").strip()}
        for column in columns
    }


@pytest.fixture
def customers_profile():
    """The profile of customers.csv."""
    return profile_of("customers.csv")


@pytest.fixture
def orders_profile():
    """The profile of orders.csv."""
    return profile_of("orders.csv")


@pytest.fixture
def sellers_profile():
    """The profile of sellers.csv."""
    return profile_of("sellers.csv")


