"""Natural-language questions answered from the knowledge graph.

The flow is:

    question
      -> retrieve the live schema
      -> generate Cypher + parameters
      -> validate (read-only policy, parameter contract, schema existence)
      -> execute on the read-only store, bounded
      -> shape the result into evidence records
      -> answer with citations and provenance

Two rules shape the implementation:

* **Values are parameters, never literals.** The model is required to emit a
  separate parameter map, and a literal in the query text is a validation
  failure. Data therefore never enters the query string.
* **Policy violations are never retried.** A rejected query is a final
  answer, not a prompt to try harder. Only a *schema* or *syntax* error gets
  one corrective retry.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from nexusai.core.config import get_settings
from nexusai.core.logging import get_logger
from nexusai.domain.ports.llm_port import LLMPort
from nexusai.infrastructure.graph_stores.neo4j_readonly_store import (
    Neo4jReadOnlyStore,
    ReadOnlyResult,
    ReadOnlyViolation,
)

logger = get_logger(__name__)

SYSTEM_PROMPT = """You translate a question about a knowledge graph into a single read-only Cypher query.

Rules you must follow exactly:
1. Emit exactly one Cypher statement. Never emit more than one.
2. Use only the labels, relationship types and properties listed in the SCHEMA below.
3. Put every literal value in the PARAMETERS map, never in the query text.
   Use $name placeholders in the query and supply the values in "parameters".
4. Always include a LIMIT clause. Use a LIMIT of at most {max_rows}.
5. Read-only only: MATCH, WHERE, WITH, RETURN, ORDER BY, SKIP, LIMIT, UNWIND, UNION.
   Never emit CREATE, MERGE, DELETE, SET, REMOVE, DROP, CALL, LOAD CSV or USE.
6. If the schema does not contain what the question needs, still return the closest
   read-only query you can, and explain the limitation in "explanation".

Respond with JSON only, in exactly this shape:
{{"cypher": "...", "parameters": {{"name": "value"}}, "explanation": "..."}}
"""

ANSWER_PROMPT = """You answer a question using the results of a read-only Cypher query on a knowledge graph.

Use only the provided records. Do not use your own knowledge and do not invent facts.
If the records do not answer the question, say so plainly.
Cite the records you used with [1], [2] and so on, in square brackets, at the end of
each sentence that relies on them. The number is the record number in the RECORDS block.

EXECUTED QUERY:
{cypher}

PARAMETERS:
{parameters}

RECORDS:
{records}

{context}

Question: {question}

Answer:"""


@dataclass
class GraphAnswer:
    """The result of a natural-language graph question."""

    question: str
    answer: str = ""
    cypher: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    explanation: str = ""
    records: list[dict[str, Any]] = field(default_factory=list)
    record_count: int = 0
    truncated: bool = False
    duration_ms: int = 0
    status: str = "ok"
    error: str | None = None
    citations: list[dict[str, Any]] = field(default_factory=list)
    source_files: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "cypher": self.cypher,
            "parameters": self.parameters,
            "explanation": self.explanation,
            "records": self.records,
            "record_count": self.record_count,
            "truncated": self.truncated,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "error": self.error,
            "citations": self.citations,
            "source_files": self.source_files,
        }


def _parse_json_response(text: str) -> dict[str, Any]:
    """Extract the JSON object from a model response.

    Models sometimes wrap the JSON in a code fence or add prose around it.
    """
    if not text:
        raise ValueError("The model returned an empty response.")
    candidate = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", candidate, re.DOTALL)
    if fence:
        candidate = fence.group(1).strip()
    if not candidate.startswith("{"):
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("The model response did not contain a JSON object.")
        candidate = candidate[start : end + 1]
    return json.loads(candidate)


def _format_schema(schema: dict[str, Any]) -> str:
    """Render the live schema for the prompt."""
    lines: list[str] = ["SCHEMA:"]
    labels = schema.get("labels") or {}
    if labels:
        lines.append("  Node labels and their properties:")
        for label, properties in sorted(labels.items()):
            lines.append(f"    - {label} {{{', '.join(properties)}}}")
    else:
        lines.append("    (no node labels)")

    rel_types = schema.get("relationship_types") or {}
    topology = schema.get("topology") or []
    if topology:
        lines.append("  Relationship types actually present:")
        for edge in topology:
            source = "/".join(edge.get("from") or []) or "?"
            target = "/".join(edge.get("to") or []) or "?"
            lines.append(f"    - ({source})-[:{edge.get('rel')}]->({target})")
    elif rel_types:
        lines.append("  Relationship types:")
        for rel_type, properties in sorted(rel_types.items()):
            lines.append(f"    - {rel_type} {{{', '.join(properties)}}}")

    constraints = schema.get("constraints") or []
    if constraints:
        lines.append("  Uniqueness constraints (use these properties as node identities):")
        for constraint in constraints:
            labels_on = ", ".join(constraint.get("labels") or [])
            properties = ", ".join(constraint.get("properties") or [])
            if labels_on and properties:
                lines.append(f"    - {labels_on}.{properties} is unique")
    lines.append("")
    return "\n".join(lines)


class GraphQaService:
    """Answers natural-language questions from the knowledge graph."""

    def __init__(
        self,
        read_only_store: Neo4jReadOnlyStore,
        llm: LLMPort,
    ) -> None:
        self._store = read_only_store
        self._llm = llm

    async def answer(self, question: str) -> GraphAnswer:
        """Answer a question about the graph."""
        settings = get_settings()
        result = GraphAnswer(question=question)

        schema = await self._store.fetch_schema()
        result.explanation = f"Schema with {len(schema.get('labels') or {})} labels."
        if not (schema.get("labels") or {}):
            result.status = "empty"
            result.error = (
                "The graph has no node labels yet. Ingest a CSV before asking questions."
            )
            return result

        system = SYSTEM_PROMPT.format(max_rows=settings.GRAPH_QUERY_MAX_ROWS)
        prompt = f"{_format_schema(schema)}\nQuestion: {question}"

        # ── Generate ──
        try:
            raw = await self._llm.generate(
                prompt=prompt,
                system_prompt=system,
                temperature=0.0,
            )
        except Exception as exc:  # noqa: BLE001
            result.status = "error"
            result.error = f"The language model could not be reached: {exc}"
            return result

        try:
            payload = _parse_json_response(raw)
        except (ValueError, json.JSONDecodeError) as exc:
            result.status = "error"
            result.error = f"The model did not return a usable query: {exc}"
            return result

        cypher = str(payload.get("cypher") or "").strip()
        parameters = payload.get("parameters") or {}
        if not isinstance(parameters, dict):
            parameters = {}
        result.cypher = cypher
        result.parameters = parameters
        result.explanation = str(payload.get("explanation") or "")

        # ── Validate and execute ──
        try:
            execution = await self._execute(cypher, parameters)
        except ReadOnlyViolation as exc:
            result.status = "rejected"
            result.error = str(exc)
            return result
        except Exception as exc:  # noqa: BLE001
            result.status = "error"
            result.error = str(exc)
            return result

        if execution is not None:
            result.records = execution.records
            result.record_count = execution.row_count
            result.truncated = execution.truncated
            result.duration_ms = execution.duration_ms
            if execution.status == "error":
                result.status = "error"
                result.error = execution.message
                return result

        if result.status == "ok" and not result.records:
            result.status = "empty"
            result.error = result.error or "The query returned no matching records."

        # ── Answer ──
        if result.status in ("ok", "empty"):
            result.answer = await self._compose_answer(result, schema)
            result.citations = self._build_citations(result.records)
            result.source_files = await self._collect_source_files(result.records)

        return result

    async def _execute(
        self,
        cypher: str,
        parameters: dict[str, Any],
    ) -> ReadOnlyResult | None:
        """Validate and run the generated query.

        A policy violation is final — it is never fed back to the model.
        """
        try:
            self._store.validate_parameters(cypher, parameters)
            return await self._store.run_read_only(cypher, parameters)
        except ReadOnlyViolation as exc:
            logger.warning("Rejected a generated query: %s", exc)
            raise
        except Exception as exc:  # noqa: BLE001
            logger.warning("A generated query failed: %s", exc)
            raise

    async def _compose_answer(
        self,
        result: GraphAnswer,
        schema: dict[str, Any],
    ) -> str:
        """Generate the final answer from the evidence."""
        context_lines = [
            "  CSV provenance: every record was created from a source CSV.",
        ]
        records_text = json.dumps(result.records, indent=2, default=str)
        prompt = ANSWER_PROMPT.format(
            cypher=result.cypher,
            parameters=json.dumps(result.parameters, default=str),
            records=records_text,
            context="\n".join(context_lines),
            question=result.question,
        )
        try:
            return await self._llm.generate(
                prompt=prompt,
                system_prompt=(
                    "You answer questions about a knowledge graph using only the "
                    "provided records, and cite them with [n] markers."
                ),
                temperature=0.0,
            )
        except Exception as exc:  # noqa: BLE001
            # The evidence is still valuable even when the model is unavailable.
            return (
                "The query ran successfully, but the answer could not be generated "
                f"because the language model was unavailable ({exc}).\n\n"
                f"Executed query:\n{result.cypher}\n\n"
                f"Records:\n{records_text}"
            )

    def _build_citations(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Build citation objects from the evidence records."""
        citations: list[dict[str, Any]] = []
        for index, record in enumerate(records, start=1):
            labels = record.get("_labels")
            if labels is None:
                labels = record.get("labels") or []
            if isinstance(labels, str):
                labels = [labels]
            citations.append(
                {
                    "index": index,
                    "labels": sorted(labels),
                    "source_file": record.get("source_file"),
                    "source_row": record.get("source_row"),
                    "source_id": record.get("source_id"),
                }
            )
        return citations

    async def _collect_source_files(self, records: list[dict[str, Any]]) -> list[str]:
        """The distinct source CSV files behind the evidence."""
        files = {r.get("source_file") for r in records if r.get("source_file")}
        return sorted(files)
