"""Tests for the natural-language graph query path.

The LLM is stubbed so the tests cover the parts that matter for safety: the
generated query is validated, a write is refused, and the evidence is
returned with its provenance.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from nexusai.application.graph.csv.graph_qa import GraphQaService, _parse_json_response
from nexusai.domain.ports.llm_port import LLMPort
from nexusai.infrastructure.graph_stores.neo4j_readonly_store import (
    Neo4jReadOnlyStore,
)


class StubLLM(LLMPort):
    """An LLM that returns a canned response."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.1,
        max_tokens: int | None = None,
    ) -> str:
        self.prompts.append(prompt)
        return self.response

    async def stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.1,
        max_tokens: int | None = None,
    ):  # noqa: ANN201
        yield self.response

    async def extract_entities(self, text: str, schema: dict[str, Any] | None = None) -> dict[str, Any]:
        return {}


class StubStore(Neo4jReadOnlyStore):
    """A read-only store whose database calls are recorded, not executed."""

    def __init__(self) -> None:  # noqa: D107
        self._driver = None
        self._database = "neo4j"
        self.executed: list[str] = []
        self.records: list[dict[str, Any]] = [
            {"customer_id": "C001", "name": "Acme Corp", "source_file": "customers.csv", "source_row": 1}
        ]

    async def fetch_schema(self) -> dict[str, Any]:
        return {
            "labels": {"Customer": ["customer_id", "name"], "Order": ["order_id", "total_amount"]},
            "relationship_types": {"PLACED": ["source_file"]},
            "topology": [{"from": ["Order"], "rel": "PLACED", "to": ["Customer"]}],
            "constraints": [
                {
                    "labels": ["Customer"],
                    "properties": ["customer_id"],
                    "type": "NODE_PROPERTY_UNIQUENESS",
                }
            ],
        }

    async def _explain(self, cypher: str) -> None:
        self.executed.append(f"EXPLAIN {cypher}")

    async def _run(
        self,
        cypher: str,
        parameters: dict[str, Any] | None = None,
        timeout: int | None = None,
    ) -> list[dict[str, Any]]:
        self.executed.append(cypher)
        return list(self.records)


def _service(response: str) -> tuple[GraphQaService, StubStore, StubLLM]:
    store = StubStore()
    llm = StubLLM(response)
    return GraphQaService(read_only_store=store, llm=llm), store, llm


def test_a_read_query_is_answered_with_evidence() -> None:
    service, store, llm = _service(
        json.dumps(
            {
                "cypher": "MATCH (c:Customer) RETURN c LIMIT $limit",
                "parameters": {"limit": 10},
                "explanation": "All customers",
            }
        )
    )
    result = asyncio.run(service.answer("who are the customers?"))

    assert result.status == "ok"
    assert result.cypher == "MATCH (c:Customer) RETURN c LIMIT $limit"
    assert result.parameters == {"limit": 10}
    assert result.record_count == 1
    assert result.source_files == ["customers.csv"]
    assert result.citations[0]["source_file"] == "customers.csv"
    assert result.citations[0]["source_row"] == 1


def test_a_generated_write_query_is_rejected() -> None:
    """The guard must refuse before anything reaches the database."""
    service, store, llm = _service(
        json.dumps(
            {
                "cypher": "MATCH (n) DETACH DELETE n",
                "parameters": {},
                "explanation": "deleting everything",
            }
        )
    )
    result = asyncio.run(service.answer("delete everything"))

    assert result.status == "rejected"
    assert result.records == []
    # The write was never executed.
    assert not any("DELETE" in q for q in store.executed)


def test_a_multi_statement_write_is_rejected() -> None:
    service, store, llm = _service(
        json.dumps(
            {
                "cypher": "MATCH (n) RETURN n LIMIT 1; MATCH (n) DETACH DELETE n",
                "parameters": {},
                "explanation": "",
            }
        )
    )
    result = asyncio.run(service.answer("sneaky"))
    assert result.status == "rejected"
    assert not any("DELETE" in q for q in store.executed)


def test_a_literal_value_is_rejected() -> None:
    """Data must not enter the query text."""
    service, store, llm = _service(
        json.dumps(
            {
                "cypher": "MATCH (c:Customer) WHERE c.name = 'Acme Corp' RETURN c LIMIT 10",
                "parameters": {},
                "explanation": "",
            }
        )
    )
    result = asyncio.run(service.answer("acme"))
    assert result.status == "rejected"


def test_a_question_with_no_data_reports_empty() -> None:
    service, store, llm = _service(
        json.dumps(
            {
                "cypher": "MATCH (c:Customer) RETURN c LIMIT $limit",
                "parameters": {"limit": 10},
                "explanation": "",
            }
        )
    )
    store.records = []
    result = asyncio.run(service.answer("who?"))
    assert result.status == "empty"
    assert result.record_count == 0


def test_a_malformed_model_response_is_reported() -> None:
    service, store, llm = _service("this is not JSON")
    result = asyncio.run(service.answer("who?"))
    assert result.status == "error"
    assert "JSON" in (result.error or "")


def test_a_fenced_json_response_is_parsed() -> None:
    service, store, llm = _service(
        "```json\n"
        '{"cypher": "MATCH (c:Customer) RETURN c LIMIT 10", "parameters": {}, "explanation": "x"}'
        "\n```"
    )
    result = asyncio.run(service.answer("who?"))
    assert result.cypher == "MATCH (c:Customer) RETURN c LIMIT 10"


def test_the_json_parser_tolerates_surrounding_prose() -> None:
    payload = _parse_json_response(
        'Here is the query:\n{"cypher": "MATCH (n) RETURN n LIMIT 1", "parameters": {}}\nHope that helps.'
    )
    assert payload["cypher"] == "MATCH (n) RETURN n LIMIT 1"


def test_the_schema_is_included_in_the_prompt() -> None:
    service, store, llm = _service(
        json.dumps({"cypher": "MATCH (c:Customer) RETURN c LIMIT 10", "parameters": {}})
    )
    asyncio.run(service.answer("who?"))
    assert "Customer" in llm.prompts[0]
    assert "PLACED" in llm.prompts[0]
