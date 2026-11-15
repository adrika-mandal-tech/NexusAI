"""API tests for the CSV-to-graph workflow.

The workflow is multi-step, so the tests walk it in order: upload, review,
approve, plan, ingest, then inspect the report and the graph.
"""

from __future__ import annotations

import io
import uuid

import pytest
from httpx import AsyncClient

pytestmark = pytest.mark.asyncio

CSV_CUSTOMERS = (
    "customer_id,name,email,country,segment,created_at\n"
    "C001,Acme Corp,info@acme.com,DE,enterprise,2024-01-15\n"
    "C002,Globex,hello@globex.com,US,smb,2024-01-22\n"
    "C003,Initech,contact@initech.com,DE,enterprise,2024-02-05\n"
)
CSV_ORDERS = (
    "order_id,customer_id,order_date,total_amount,status\n"
    "O1001,C001,2024-01-20,1250.50,shipped\n"
    "O1002,C002,2024-01-22,340.00,pending\n"
    "O1003,C001,2024-02-01,99.99,shipped\n"
)


@pytest.fixture
async def token(client: AsyncClient) -> str:
    """Register a user and return its access token.

    The in-memory user store persists for the life of the process, so each
    test registers a distinct account.
    """
    email = f"graphtest-{uuid.uuid4().hex[:12]}@example.com"
    response = await client.post(
        "/api/v1/auth/register",
        json={
            "email": email,
            "password": "a-strong-password",
            "full_name": "Graph Tester",
        },
    )
    assert response.status_code == 200, response.text
    return response.json()["access_token"]


@pytest.fixture
def auth(token: str) -> dict[str, str]:
    """The authorization header."""
    return {"Authorization": f"Bearer {token}"}


async def test_the_graph_endpoints_require_authentication(client: AsyncClient) -> None:
    """The graph endpoints write to the database, so they must be protected."""
    response = await client.post(
        "/api/v1/graph/csv/upload",
        files={"files": ("customers.csv", io.BytesIO(CSV_CUSTOMERS.encode()), "text/csv")},
    )
    assert response.status_code == 401


async def test_the_old_csv_endpoint_is_gone(client: AsyncClient) -> None:
    """The injection-prone endpoint is retired."""
    response = await client.post(
        "/api/v1/knowledge/graph/csv",
        files={"file": ("customers.csv", io.BytesIO(CSV_CUSTOMERS.encode()), "text/csv")},
    )
    assert response.status_code == 410
    assert "graph/csv/upload" in response.json()["detail"]


async def test_the_old_csv_endpoint_reports_gone_without_a_file(
    client: AsyncClient,
) -> None:
    """It must not report a validation error about the missing upload."""
    response = await client.post("/api/v1/knowledge/graph/csv")
    assert response.status_code == 410


async def test_uploading_a_csv_produces_a_reviewable_proposal(
    client: AsyncClient, auth: dict[str, str]
) -> None:
    response = await client.post(
        "/api/v1/graph/csv/upload",
        headers=auth,
        files=[
            ("files", ("customers.csv", io.BytesIO(CSV_CUSTOMERS.encode()), "text/csv")),
            ("files", ("orders.csv", io.BytesIO(CSV_ORDERS.encode()), "text/csv")),
        ],
    )
    assert response.status_code == 201, response.text
    body = response.json()

    assert body["status"] == "pending_review"
    assert body["requires_approval"] is True
    labels = {node["label"] for node in body["nodes"]}
    assert labels == {"Customer", "Order"}

    # The relationship between them is proposed with its evidence.
    assert len(body["relationships"]) == 1
    relationship = body["relationships"][0]
    assert relationship["source_label"] == "Order"
    assert relationship["target_label"] == "Customer"
    assert relationship["source_columns"] == ["customer_id"]
    assert relationship["match_ratio"] == 1.0

    # The proposal carries the constraints that will be created.
    statements = " ".join(c["statement"] for c in body["constraints"])
    assert "Customer" in statements
    assert "Order" in statements


async def test_a_non_csv_upload_is_rejected(
    client: AsyncClient, auth: dict[str, str]
) -> None:
    response = await client.post(
        "/api/v1/graph/csv/upload",
        headers=auth,
        files={"files": ("evil.exe", io.BytesIO(b"MZ..."), "application/octet-stream")},
    )
    assert response.status_code == 422


async def test_a_zip_named_csv_is_rejected(
    client: AsyncClient, auth: dict[str, str]
) -> None:
    """A spreadsheet container must not reach the CSV path."""
    response = await client.post(
        "/api/v1/graph/csv/upload",
        headers=auth,
        files={"files": ("sneaky.csv", io.BytesIO(b"PK\x03\x04rest"), "text/csv")},
    )
    assert response.status_code == 422


async def test_an_empty_upload_is_rejected(
    client: AsyncClient, auth: dict[str, str]
) -> None:
    response = await client.post(
        "/api/v1/graph/csv/upload",
        headers=auth,
        files={"files": ("empty.csv", io.BytesIO(b""), "text/csv")},
    )
    assert response.status_code == 422


async def test_ingesting_without_approval_is_refused(
    client: AsyncClient, auth: dict[str, str]
) -> None:
    """A proposal must be approved before anything is written."""
    upload = await client.post(
        "/api/v1/graph/csv/upload",
        headers=auth,
        files={"files": ("customers.csv", io.BytesIO(CSV_CUSTOMERS.encode()), "text/csv")},
    )
    proposal_id = upload.json()["proposal_id"]

    response = await client.post(
        "/api/v1/graph/csv/ingest", headers=auth, json={"proposal_id": proposal_id}
    )
    assert response.status_code == 409


async def test_the_plan_is_available_after_approval(
    client: AsyncClient, auth: dict[str, str]
) -> None:
    upload = await client.post(
        "/api/v1/graph/csv/upload",
        headers=auth,
        files={"files": ("customers.csv", io.BytesIO(CSV_CUSTOMERS.encode()), "text/csv")},
    )
    proposal_id = upload.json()["proposal_id"]

    # The plan is not available before approval.
    response = await client.get(f"/api/v1/graph/csv/plan/{proposal_id}", headers=auth)
    assert response.status_code == 409

    approve = await client.post(
        f"/api/v1/graph/csv/proposal/{proposal_id}/approve", headers=auth
    )
    assert approve.status_code == 200
    assert approve.json()["status"] == "approved"

    response = await client.get(f"/api/v1/graph/csv/plan/{proposal_id}", headers=auth)
    assert response.status_code == 200
    body = response.json()
    assert body["statements"]
    cypher = body["cypher"]
    assert "MERGE (n:Customer" in cypher
    # No value is interpolated into the query text.
    assert "Acme" not in cypher
    assert "C001" not in cypher


async def test_a_proposal_can_be_rejected(
    client: AsyncClient, auth: dict[str, str]
) -> None:
    upload = await client.post(
        "/api/v1/graph/csv/upload",
        headers=auth,
        files={"files": ("customers.csv", io.BytesIO(CSV_CUSTOMERS.encode()), "text/csv")},
    )
    proposal_id = upload.json()["proposal_id"]

    response = await client.post(
        f"/api/v1/graph/csv/proposal/{proposal_id}/reject",
        headers=auth,
        json={"reason": "wrong schema"},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "rejected"

    # A rejected proposal cannot be ingested.
    response = await client.post(
        "/api/v1/graph/csv/ingest", headers=auth, json={"proposal_id": proposal_id}
    )
    assert response.status_code == 409


async def test_an_unknown_proposal_is_not_found(
    client: AsyncClient, auth: dict[str, str]
) -> None:
    response = await client.get(
        "/api/v1/graph/csv/proposal/00000000-0000-0000-0000-000000000000", headers=auth
    )
    assert response.status_code == 404


async def test_asking_a_question_requires_a_read_only_account(
    client: AsyncClient, auth: dict[str, str]
) -> None:
    """Without the read-only account the path is disabled, not downgraded."""
    response = await client.post(
        "/api/v1/graph/csv/ask", headers=auth, json={"question": "how many orders?"}
    )
    # Either the account is configured and the question is answered, or the
    # path is refused. It must never fall back to the administrative account.
    assert response.status_code in (200, 503)
    if response.status_code == 503:
        assert "read-only" in response.json()["detail"].lower()
