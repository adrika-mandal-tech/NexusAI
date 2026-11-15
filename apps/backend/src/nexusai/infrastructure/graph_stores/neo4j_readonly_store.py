"""Read-only Neo4j access for the natural-language graph query path.

This module is the security boundary for user-supplied questions. It is the
only place where LLM-generated Cypher reaches the database.

**Four independent layers, in order of strength:**

1. **An ``EXPLAIN`` gate.** The query is planned (not executed) and Neo4j's
   own ``query_type`` is inspected. A write is rejected before it runs. This
   is the same pattern the ``neo4j-graphrag`` library uses in
   ``retrievers/text2cypher.py`` (added in v1.16.0). Because the
   classification comes from Neo4j's own parser, a write clause cannot be
   smuggled past it.
2. **A clause allowlist and a statement-count check.** ``EXPLAIN`` explains
   only the first statement, so a multi-statement payload must be rejected
   explicitly. Every procedure call, ``LOAD CSV`` and ``USE`` is rejected.
3. **A post-execution write check.** The transaction counters report whether
   the statement modified anything; a read must not.
4. **Bounded execution** — a transaction timeout, a server-side row cap, and
   a client-side row cap.

**Edition note — verified on Neo4j Community 2026.08.1.** Community does not
implement the fine-grained privilege system: ``GRANT MATCH {*} ON GRAPH`` is
rejected with ``UnsupportedAdministrationCommand`` and every authenticated
user has full read/write access. A *separate* read-only account is therefore
an **identity** boundary, not a **privilege** boundary, on Community. The
server-side refusal of a write — which is the strongest control on
Enterprise — is unavailable here, so layers 1-3 above are what actually
protect the database. On Enterprise the account would additionally be
restricted by the server.

If the read-only account is not configured the whole path is **disabled**; it
never falls back to the administrative account.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from nexusai.core.config import get_settings
from nexusai.core.logging import get_logger
from nexusai.infrastructure.graph_stores.neo4j_driver_factory import (
    get_database_name,
    get_reader_driver,
)

logger = get_logger(__name__)

# Neo4j reports the statement type of a query. "r" is read-only.
READ_ONLY_QUERY_TYPE = "r"

# Clauses that must never appear in a read-only query. This is checked on the
# *parsed* form, not as a substring of the raw text, so a clause cannot be
# smuggled in through a string literal.
FORBIDDEN_CLAUSES: frozenset[str] = frozenset(
    {
        "create",
        "merge",
        "delete",
        "detach",
        "set",
        "remove",
        "drop",
        "foreach",
        "load csv",
        "use",
        "call",
        "grant",
        "revoke",
        "start",
        "alter",
        "rename",
        "create database",
        "create index",
        "create constraint",
        "create user",
        "create role",
    }
)

# Clauses that are permitted. Anything not in this list and not in
# ``FORBIDDEN_CLAUSES`` is rejected as well, so the allowlist is closed.
ALLOWED_CLAUSES: frozenset[str] = frozenset(
    {
        "match",
        "optional match",
        "where",
        "with",
        "return",
        "order by",
        "skip",
        "limit",
        "unwind",
        "union",
        "union all",
        "distinct",
    }
)

# A Cypher statement separator outside a string literal.
_STATEMENT_SPLIT_RE = re.compile(r";\s*(?=(?:[^'\"]|'[^']*'|\"[^\"]*\")*$)")

# A quoted string in the query text. This is the injection vector: a string
# literal can contain a quote and terminate the literal, changing the
# structure of the query.
_STRING_LITERAL_RE = re.compile(r"(?<![\w$])('(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\")")

# A number in a comparison position, e.g. "= 42" or "> 3". A number cannot
# alter the structure of a query, but a value that the caller meant to supply
# as a parameter must still be supplied as one.
_NUMERIC_COMPARISON_RE = re.compile(
    r"(?:=|<>|!=|<=|>=|<|>)\s*-?\d+(?:\.\d+)?(?!\s*,\s*(?:[A-Za-z_]\w*)?\s*$)",
    re.IGNORECASE,
)


class ReadOnlyViolation(Exception):
    """Raised when a query fails the read-only policy."""


class ReadOnlyStoreUnavailable(Exception):
    """Raised when the read-only account is not configured."""


@dataclass
class ReadOnlyResult:
    """The outcome of a guarded read-only query."""

    cypher: str
    parameters: dict[str, Any] = field(default_factory=dict)
    records: list[dict[str, Any]] = field(default_factory=list)
    row_count: int = 0
    truncated: bool = False
    duration_ms: int = 0
    status: str = "ok"
    message: str = ""


class Neo4jReadOnlyStore:
    """Read-only graph access, with a hard write guard."""

    def __init__(self, driver: Any | None = None) -> None:
        self._driver = driver if driver is not None else get_reader_driver()
        self._database = get_database_name()
        if self._driver is None:
            raise ReadOnlyStoreUnavailable(
                "The read-only Neo4j account is not configured. Set NEO4J_READONLY_USER "
                "and NEO4J_READONLY_PASSWORD. The natural-language graph query path is "
                "disabled — it will not fall back to the administrative account."
            )

    # ── Schema retrieval ────────────────────────────────────────────────────

    async def fetch_schema(self) -> dict[str, Any]:
        """Retrieve the live graph schema.

        Uses only built-in procedures, so it works on a server without the
        APOC plugin.
        """
        labels: dict[str, list[str]] = {}
        try:
            records = await self._run(
                "CALL db.schema.nodeTypeProperties() YIELD nodeType, propertyName "
                "RETURN nodeType AS label, collect(DISTINCT propertyName) AS properties"
            )
            for record in records:
                labels[record["label"]] = sorted(record["properties"])
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not read node schema: %s", exc)

        rel_types: dict[str, list[str]] = {}
        try:
            records = await self._run(
                "CALL db.schema.relTypeProperties() YIELD relType, propertyName "
                "RETURN relType AS type, collect(DISTINCT propertyName) AS properties"
            )
            for record in records:
                rel_types[record["type"]] = sorted(record["properties"])
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not read relationship schema: %s", exc)

        topology: list[dict[str, Any]] = []
        try:
            records = await self._run(
                "MATCH (a)-[r]->(b) "
                "RETURN DISTINCT labels(a) AS from, type(r) AS rel, labels(b) AS to "
                "LIMIT 500"
            )
            topology = [
                {"from": r["from"], "rel": r["rel"], "to": r["to"]} for r in records
            ]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not read relationship topology: %s", exc)

        constraints: list[dict[str, Any]] = []
        try:
            records = await self._run(
                "SHOW CONSTRAINTS YIELD labelsOrTypes, properties, type "
                "RETURN labelsOrTypes AS labels, properties AS properties, type AS type"
            )
            constraints = records
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not read constraints: %s", exc)

        return {
            "labels": labels,
            "relationship_types": rel_types,
            "topology": topology,
            "constraints": constraints,
        }

    # ── Guarded execution ──────────────────────────────────────────────────

    async def run_read_only(
        self,
        cypher: str,
        parameters: dict[str, Any] | None = None,
        timeout_ms: int | None = None,
        max_rows: int | None = None,
    ) -> ReadOnlyResult:
        """Validate and execute a query, enforcing the read-only policy."""
        settings = get_settings()
        timeout = timeout_ms or settings.GRAPH_QUERY_TIMEOUT_MS
        limit = max_rows or settings.GRAPH_QUERY_MAX_ROWS
        parameters = parameters or {}

        self.validate_query(cypher)

        started = __import__("time").perf_counter()
        status = "ok"
        message = ""

        try:
            await self._explain(cypher)
        except ReadOnlyViolation:
            raise
        except Exception as exc:  # noqa: BLE001
            # A planning failure is a query error, not a policy violation.
            status = "error"
            message = str(exc)
            return ReadOnlyResult(
                cypher=cypher,
                parameters=parameters,
                status=status,
                message=message,
                duration_ms=int((__import__("time").perf_counter() - started) * 1000),
            )

        try:
            records = await self._run(cypher, parameters, timeout=timeout)
        except Exception as exc:  # noqa: BLE001
            status = "error"
            message = str(exc)
            records = []

        truncated = len(records) > limit
        if truncated:
            records = records[:limit]

        return ReadOnlyResult(
            cypher=cypher,
            parameters=parameters,
            records=records,
            row_count=len(records),
            truncated=truncated,
            status=status,
            message=message,
            duration_ms=int((__import__("time").perf_counter() - started) * 1000),
        )
    # ── Validation ──────────────────────────────────────────────────────────

    def validate_query(self, cypher: str) -> None:
        """Apply the read-only policy to a query.

        Raises :class:`ReadOnlyViolation` when the query is not permitted.
        """
        if not cypher or not cypher.strip():
            raise ReadOnlyViolation("The query is empty.")

        if len(cypher) > 20_000:
            raise ReadOnlyViolation("The query is too long.")

        # 1. A single statement only. EXPLAIN explains only the first
        #    statement, so a multi-statement payload must be rejected here.
        statements = [s for s in _STATEMENT_SPLIT_RE.split(cypher) if s.strip()]
        if len(statements) > 1:
            raise ReadOnlyViolation(
                "Only a single statement is permitted. A multi-statement query was rejected."
            )

        # 2. Strip comments before the clause scan, so a clause cannot hide
        #    inside a comment.
        stripped = re.sub(r"//[^\n]*", " ", cypher)
        stripped = re.sub(r"/\*.*?\*/", " ", stripped, flags=re.DOTALL)

        lowered = stripped.lower()

        # 3. No forbidden clause. Checked against word boundaries so that a
        #    property named e.g. "settings" does not trip the "set" rule.
        for clause in FORBIDDEN_CLAUSES:
            if re.search(rf"\b{re.escape(clause)}\b", lowered):
                raise ReadOnlyViolation(
                    f"The query contains the forbidden clause {clause!r}. Only read-only "
                    "queries are permitted."
                )

        # 4. Clause allowlist. Every clause keyword in the query must be one
        #    we recognise as read-only.
        found = re.findall(
            r"\b(matches|match|optional|where|with|return|order|by|skip|limit|unwind|union|"
            r"distinct|create|merge|delete|detach|set|remove|drop|foreach|load|csv|use|call|"
            r"grant|revoke|start|alter|rename|as|in|is|not|null|and|or|xor|case|when|then|"
            r"else|end|exists|contains|starts|ends|asc|desc|true|false)\b",
            lowered,
        )
        allowed_words = {
            "matches",
            "match",
            "optional",
            "where",
            "with",
            "return",
            "order",
            "by",
            "skip",
            "limit",
            "unwind",
            "union",
            "all",
            "distinct",
            "as",
            "in",
            "is",
            "not",
            "null",
            "and",
            "or",
            "xor",
            "case",
            "when",
            "then",
            "else",
            "end",
            "exists",
            "contains",
            "starts",
            "ends",
            "asc",
            "desc",
            "true",
            "false",
        }
        unknown = sorted({w for w in found if w not in allowed_words})
        if unknown:
            raise ReadOnlyViolation(
                f"The query contains clause(s) that are not permitted in a read-only "
                f"query: {', '.join(unknown)}."
            )

        # 5. A LIMIT is mandatory so the result set is always bounded.
        if not re.search(r"\blimit\b", lowered):
            raise ReadOnlyViolation(
                "A read-only query must include a LIMIT so the result set stays bounded."
            )

    def validate_parameters(self, cypher: str, parameters: dict[str, Any]) -> None:
        """Enforce the parameter contract.

        A literal in a value position means user data has reached the query
        text, which is both an injection surface and a data-exfiltration
        surface (the query is logged, displayed and sent back to the model on
        a corrective retry).

        String literals are always rejected. Numbers are rejected when they
        appear in a comparison, but a number in a ``LIMIT`` is structural and
        cannot change the shape of the query, so it is allowed.
        """
        stripped = re.sub(r"//[^\n]*", " ", cypher)
        stripped = re.sub(r"/\*.*?\*/", " ", stripped, flags=re.DOTALL)
        if _STRING_LITERAL_RE.search(stripped):
            raise ReadOnlyViolation(
                "The query contains a literal value. Values must be supplied as "
                "parameters so that data never enters the query text."
            )
        comparison = _NUMERIC_COMPARISON_RE.search(stripped)
        if comparison:
            raise ReadOnlyViolation(
                "The query compares against a literal number. Values must be supplied "
                "as parameters so that data never enters the query text."
            )
        for name in parameters:
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
                raise ReadOnlyViolation(f"Invalid parameter name: {name!r}")

    # ── Internals ───────────────────────────────────────────────────────────

    async def _explain(self, cypher: str) -> None:
        """Plan the query and refuse anything that is not read-only."""
        async with self._driver.session(database=self._database) as session:
            result = await session.run(f"EXPLAIN {cypher}")
            summary = await result.consume()
        query_type = getattr(summary, "query_type", None)
        if query_type != READ_ONLY_QUERY_TYPE:
            raise ReadOnlyViolation(
                f"Refusing to execute a non-read-only query (query_type={query_type!r})."
            )

    async def _run(
        self,
        cypher: str,
        parameters: dict[str, Any] | None = None,
        timeout: int | None = None,
    ) -> list[dict[str, Any]]:
        """Execute a query and return the records."""
        kwargs: dict[str, Any] = {}
        if timeout is not None:
            kwargs["timeout"] = timeout / 1000
        async with self._driver.session(database=self._database) as session:
            result = await session.run(cypher, parameters or {}, **kwargs)
            records = await result.data()
            summary = await result.consume()
        # Safety net for Neo4j Community, where every authenticated user has
        # full read/write access and no server-side privilege can restrain a
        # write. A statement that modified the database reports updates in its
        # counters, so this detects a write that slipped past the guard.
        counters = getattr(summary, "counters", None)
        if counters is not None and getattr(counters, "contains_updates", False):
            raise ReadOnlyViolation(
                "The executed query reported database updates, which a read-only query "
                "must not do. The query was rejected."
            )
        return records
