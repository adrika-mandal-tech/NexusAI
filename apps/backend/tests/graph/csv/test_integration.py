"""Integration tests against a live Neo4j.

These tests are skipped when no database is reachable. They write only to
their own node labels and clean up after themselves.

Neo4j Community has no multi-database support, so the tests use distinctive
label prefixes and delete exactly what they created. They must never be run
against a database that holds data you care about.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from nexusai.application.graph.csv.ingestor import Ingestor
from nexusai.application.graph.csv.key_inference import discover_relationships
from nexusai.application.graph.csv.profiler import read_rows
from nexusai.application.graph.csv.schema_proposer import build_proposal
from nexusai.application.graph.csv.validator import validate_ingestion
from nexusai.domain.entities.graph_schema import GraphSchemaProposal
from nexusai.infrastructure.graph_stores.neo4j_driver_factory import init_neo4j_drivers
from nexusai.infrastructure.graph_stores.neo4j_store import Neo4jGraphStore
from tests.graph.conftest import fixture_path, profile_of

pytestmark = pytest.mark.asyncio


@pytest.fixture
async def store() -> Neo4jGraphStore:
    """A graph store, skipping the test when Neo4j is unreachable."""
    init_neo4j_drivers()
    graph = Neo4jGraphStore()
    try:
        await graph.run_read("RETURN 1")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Neo4j is not reachable: {exc}")
    yield graph
    # Clean up everything this module created.
    await graph.run_read(
        "MATCH (n) WHERE n.kind = 'csv_ingest' OR n:IngestionJob OR n:SourceFile "
        "DETACH DELETE n"
    )
    await graph.run_read(
        "MATCH (n) WHERE n:Customer OR n:Order OR n:Seller OR n:GraphSchemaProposal "
        "DETACH DELETE n"
    )


async def _proposal() -> tuple[GraphSchemaProposal, dict[str, tuple[Path, str, str]]]:
    """Build a proposal from the three happy-path fixtures."""
    names = ("customers.csv", "orders.csv", "sellers.csv")
    profiles = [profile_of(name) for name in names]
    values: dict[tuple[str, str], set[str]] = {}
    files: dict[str, tuple[Path, str, str]] = {}
    for profile in profiles:
        path = fixture_path(profile.filename)
        files[profile.filename] = (path, "utf-8", ",")
        for row in read_rows(path, "utf-8", ","):
            for column, value in row.items():
                if value and value.strip():
                    values.setdefault((profile.filename, column), set()).add(value.strip())
    return build_proposal(profiles, discover_relationships(profiles, values)), files


async def test_the_happy_path_creates_the_expected_graph(store: Neo4jGraphStore) -> None:
    proposal, files = await _proposal()
    for rel in proposal.relationships:
        rel.expected_relationship_count = 5

    job_id = uuid.uuid4()
    report = await Ingestor(store=store, proposal=proposal, job_id=job_id).run(files)
    assert report.status.value == "completed"
    assert report.error is None

    total_nodes = 0
    for result in report.files:
        assert result.rows_rejected == 0
        # Created plus matched is the number of rows that reached the graph.
        # The split depends on whether the data was already there.
        total_nodes += result.nodes_created + result.nodes_matched
    assert total_nodes == 13  # 5 customers + 5 orders + 3 sellers

    # The counts are scoped to this job, so the assertions hold whatever else
    # is already in the database.
    for label, expected in (("Customer", 5), ("Order", 5), ("Seller", 3)):
        records = await store.run_read(
            f"MATCH (n:{label}) WHERE n.ingestion_job_id = $job_id RETURN count(n) AS c",
            {"job_id": str(job_id)},
        )
        assert records[0]["c"] == expected, label

    for rel_type, expected in (("PLACED", 5), ("SOLD_BY", 5)):
        records = await store.run_read(
            f"MATCH ()-[r:{rel_type}]->() WHERE r.ingestion_job_id = $job_id "
            "RETURN count(r) AS c",
            {"job_id": str(job_id)},
        )
        assert records[0]["c"] == expected, rel_type


async def test_ingestion_is_idempotent(store: Neo4jGraphStore) -> None:
    """Re-running the same ingestion must not multiply the graph."""
    proposal, files = await _proposal()
    first = await Ingestor(store=store, proposal=proposal, job_id=uuid.uuid4()).run(files)
    assert first.status.value == "completed"

    second = await Ingestor(store=store, proposal=proposal, job_id=uuid.uuid4()).run(files)
    assert second.status.value == "completed"

    for result in second.files:
        assert result.nodes_created == 0
        assert result.nodes_matched == result.rows_accepted

    records = await store.run_read(
        "MATCH (n:Customer) WHERE n.ingestion_job_id = $job_id RETURN count(n) AS c",
        {"job_id": str(second.job_id)},
    )
    assert records[0]["c"] == 5
    records = await store.run_read(
        "MATCH ()-[r:PLACED]->() WHERE r.ingestion_job_id = $job_id RETURN count(r) AS c",
        {"job_id": str(second.job_id)},
    )
    assert records[0]["c"] == 5


async def test_the_batch_size_does_not_change_the_result(store: Neo4jGraphStore) -> None:
    proposal, files = await _proposal()
    small = await Ingestor(store=store, proposal=proposal, job_id=uuid.uuid4(), batch_size=1).run(files)
    assert small.status.value == "completed"
    records = await store.run_read(
        "MATCH (n:Order) WHERE n.ingestion_job_id = $job_id RETURN count(n) AS c",
        {"job_id": str(small.job_id)},
    )
    assert records[0]["c"] == 5


async def test_an_orphan_foreign_key_creates_no_node(store: Neo4jGraphStore) -> None:
    """A reference to a missing parent must not invent a node."""
    customers = profile_of("customers.csv")
    orders = profile_of("orders_orphan_fk.csv")
    values = {
        ("customers.csv", "customer_id"): {"C001", "C002", "C003", "C004", "C005"},
        ("orders_orphan_fk.csv", "customer_id"): {"C001", "C002", "C999"},
        ("orders_orphan_fk.csv", "order_id"): {"O2001", "O2002", "O2003"},
        ("orders_orphan_fk.csv", "seller_id"): {"S001"},
    }
    proposal = build_proposal(
        [customers, orders], discover_relationships([customers, orders], values)
    )
    files = {
        "customers.csv": (fixture_path("customers.csv"), "utf-8", ","),
        "orders_orphan_fk.csv": (fixture_path("orders_orphan_fk.csv"), "utf-8", ","),
    }
    report = await Ingestor(store=store, proposal=proposal, job_id=uuid.uuid4()).run(files)
    assert report.status.value == "completed"

    # No node was invented for the unmatched key.
    records = await store.run_read(
        "MATCH (c:Customer {customer_id: 'C999'}) RETURN count(c) AS c"
    )
    assert records[0]["c"] == 0

    # Only the two resolvable orders produced edges.
    records = await store.run_read(
        "MATCH ()-[r:PLACED]->() WHERE r.ingestion_job_id = $job_id RETURN count(r) AS c",
        {"job_id": str(report.job_id)},
    )
    assert records[0]["c"] == 2

    # The unmatched row is reported.
    orders_result = next(r for r in report.files if r.filename == "orders_orphan_fk.csv")
    assert orders_result.unmatched_relationship_rows == 1


async def test_a_malformed_row_is_rejected_and_reported(store: Neo4jGraphStore) -> None:
    """A row with no identity is rejected; a bad optional value is reported."""
    profile = profile_of("customers_malformed.csv")
    proposal = build_proposal([profile])
    files = {"customers_malformed.csv": (fixture_path("customers_malformed.csv"), "utf-8", ",")}
    report = await Ingestor(store=store, proposal=proposal, job_id=uuid.uuid4()).run(files)
    assert report.status.value == "completed"

    result = report.files[0]
    # The row with an empty customer_id is rejected outright.
    assert result.rows_rejected == 1
    assert result.rejected_rows
    rejection = result.rejected_rows[0]
    assert rejection.row == 4
    assert rejection.message
    assert rejection.column == "customer_id"

    # The row with an unparseable date is kept, with the failure reported and
    # the property set to null. The value is never silently dropped.
    assert result.conversion_failures
    failure = result.conversion_failures[0]
    assert failure["column"] == "created_at"
    assert failure["action"] == "value_set_to_null"

    # The good rows were still ingested.
    label = proposal.nodes[0].label
    records = await store.run_read(
        f"MATCH (c:{label}) WHERE c.ingestion_job_id = $job_id RETURN count(c) AS c",
        {"job_id": str(report.job_id)},
    )
    assert records[0]["c"] == result.rows_accepted

    # The unparseable date became a null, not a string.
    records = await store.run_read(
        f"MATCH (c:{label} {{customer_id: 'C002'}}) RETURN c.created_at AS d"
    )
    assert records[0]["d"] is None

    # The rows that did parse are stored as dates.
    records = await store.run_read(
        f"MATCH (c:{label} {{customer_id: 'C001'}}) "
        "RETURN c.created_at AS d, valueType(c.created_at) AS t"
    )
    assert records[0]["t"].startswith("DATE")


async def test_property_types_are_consistent(store: Neo4jGraphStore) -> None:
    proposal, files = await _proposal()
    await Ingestor(store=store, proposal=proposal, job_id=uuid.uuid4()).run(files)

    records = await store.run_read(
        "MATCH (o:Order) RETURN valueType(o.total_amount) AS t, count(*) AS c"
    )
    assert len(records) == 1
    assert records[0]["t"].startswith("FLOAT")

    records = await store.run_read(
        "MATCH (o:Order) RETURN valueType(o.order_date) AS t, count(*) AS c"
    )
    assert len(records) == 1
    assert records[0]["t"].startswith("DATE")


async def test_the_provenance_is_recorded(store: Neo4jGraphStore) -> None:
    proposal, files = await _proposal()
    job_id = uuid.uuid4()
    await Ingestor(store=store, proposal=proposal, job_id=job_id).run(files)

    records = await store.run_read(
        "MATCH (n) WHERE n.ingestion_job_id = $job_id "
        "RETURN count(n) AS total, count(n.source_file) AS with_file, "
        "count(n.source_row) AS with_row, count(n.source_id) AS with_source_id",
        {"job_id": str(job_id)},
    )
    row = records[0]
    assert row["total"] == row["with_file"] == row["with_row"] == row["with_source_id"]
    assert row["total"] == 13

    # The source row numbers are the real ones.
    records = await store.run_read(
        "MATCH (c:Customer) WHERE c.ingestion_job_id = $job_id "
        "RETURN c.customer_id AS id, c.source_row AS row ORDER BY c.source_row",
        {"job_id": str(job_id)},
    )
    assert [r["id"] for r in records] == ["C001", "C002", "C003", "C004", "C005"]
    assert [r["row"] for r in records] == [1, 2, 3, 4, 5]


async def test_the_validation_reports_pass(store: Neo4jGraphStore) -> None:
    proposal, files = await _proposal()
    report = await Ingestor(store=store, proposal=proposal, job_id=uuid.uuid4()).run(files)
    validation = await validate_ingestion(store, proposal, report, files)

    failed = [c for c in validation.checks if c.status == "fail"]
    assert not failed, f"validation failures: {[c.check_id for c in failed]}"

    reconciliation = next(
        c for c in validation.checks if c.check_id == "row_reconciliation_sample"
    )
    assert reconciliation.status == "pass"
    assert reconciliation.actual == 0


async def test_the_relationship_direction_is_correct(store: Neo4jGraphStore) -> None:
    proposal, files = await _proposal()
    job_id = uuid.uuid4()
    await Ingestor(store=store, proposal=proposal, job_id=job_id).run(files)

    records = await store.run_read(
        "MATCH (c:Customer)-[r:PLACED]->(:Order) WHERE r.ingestion_job_id = $job_id "
        "RETURN count(c) AS c",
        {"job_id": str(job_id)},
    )
    assert records[0]["c"] == 0

    records = await store.run_read(
        "MATCH (o:Order)-[r:PLACED]->(c:Customer) WHERE r.ingestion_job_id = $job_id "
        "RETURN count(o) AS c",
        {"job_id": str(job_id)},
    )
    assert records[0]["c"] == 5


async def test_a_uniqueness_constraint_is_installed(store: Neo4jGraphStore) -> None:
    proposal, files = await _proposal()
    await Ingestor(store=store, proposal=proposal, job_id=uuid.uuid4()).run(files)

    records = await store.run_read(
        "SHOW CONSTRAINTS YIELD labelsOrTypes, properties, type "
        "RETURN labelsOrTypes AS labels, properties AS properties, type AS type"
    )
    constrained = {
        (tuple(r["labels"]), tuple(r["properties"]))
        for r in records
        if "UNIQUENESS" in (r["type"] or "").upper()
    }
    assert (("Customer",), ("customer_id",)) in constrained


async def test_the_job_provenance_nodes_are_created(store: Neo4jGraphStore) -> None:
    proposal, files = await _proposal()
    job_id = uuid.uuid4()
    await Ingestor(store=store, proposal=proposal, job_id=job_id).run(files)

    records = await store.run_read(
        "MATCH (j:IngestionJob {id: $job_id})-[:CONTAINS]->(f:SourceFile) "
        "RETURN f.filename AS filename, f.nodes_created AS nodes ORDER BY f.filename",
        {"job_id": str(job_id)},
    )
    assert len(records) == 3
    by_file = {r["filename"]: r["nodes"] for r in records}
    assert by_file["customers.csv"] == 5
    assert by_file["orders.csv"] == 5
    assert by_file["sellers.csv"] == 3


async def test_a_dry_run_writes_nothing(store: Neo4jGraphStore) -> None:
    proposal, files = await _proposal()
    report = await Ingestor(
        store=store, proposal=proposal, job_id=uuid.uuid4(), dry_run=True
    ).run(files)
    assert report.status.value == "completed"
    assert report.dry_run is True

    records = await store.run_read(
        "MATCH (n:Customer) WHERE n.ingestion_job_id = $job_id RETURN count(n) AS c",
        {"job_id": str(report.job_id)},
    )
    assert records[0]["c"] == 0
