#!/usr/bin/env python
"""Provision the read-only Neo4j account used by the graph query path.

Run once, as an administrator, against the target Neo4j instance:

    python -m scripts.setup_neo4j_readonly

The account is granted only ``MATCH`` privileges, so the server refuses any
write from it regardless of the query text. This is the primary security
control for the natural-language graph query path; the application-side
``EXPLAIN`` guard is the second layer.

The password is read from ``NEO4J_READONLY_PASSWORD`` (or ``NEO4J_PASSWORD``
as a fallback) and is never printed. If the account already exists it is
recreated, so the script is idempotent.
"""

from __future__ import annotations

import asyncio
import os
import sys

from neo4j import GraphDatabase

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nexusai.core.config import get_settings  # noqa: E402


async def main() -> int:
    settings = get_settings()
    user = settings.NEO4J_READONLY_USER or "nexusai_reader"
    password = settings.NEO4J_READONLY_PASSWORD or settings.NEO4J_PASSWORD
    database = settings.NEO4J_DATABASE

    if not password:
        print("No password available. Set NEO4J_READONLY_PASSWORD.", file=sys.stderr)
        return 1

    driver = GraphDatabase.driver(
        settings.NEO4J_URL, auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
    )
    try:
        with driver.session(database="system") as session:
            edition = ""
            try:
                record = session.run(
                    "CALL dbms.components() YIELD name, versions, edition "
                    "WHERE name = 'Neo4j Kernel' RETURN versions[0] AS version, edition"
                ).single()
                edition = (record["edition"] if record else "") or ""
            except Exception:
                pass

            session.run(f"DROP USER {user} IF EXISTS").consume()
            session.run(
                f"CREATE USER {user} SET PASSWORD $password CHANGE NOT REQUIRED",
                password=password,
            ).consume()
            print(f"Read-only account {user!r} created on graph {database!r}.")

            # Granular privileges exist on Enterprise only. On Community the
            # statement is rejected, and every authenticated user has full
            # read/write access — so the account is an identity boundary, not
            # a privilege boundary. The application-side guard is what
            # actually enforces read-only behaviour there.
            privilege_error = ""
            for statement in (
                f"GRANT MATCH {{*}} ON GRAPH {database} TO {user}",
                f"GRANT MATCH ON GRAPH {database} TO {user}",
            ):
                try:
                    session.run(statement).consume()
                    print(f"  granted: {statement}")
                    privilege_error = ""
                    break
                except Exception as exc:  # noqa: BLE001
                    privilege_error = str(exc)

            if privilege_error:
                print("")
                print("  NOTE: read-only privileges could not be granted.")
                print(f"  Edition: {edition or 'unknown'}")
                print(f"  Server:  {privilege_error[:200]}")
                print("")
                print("  Neo4j Community does not implement the fine-grained privilege")
                print("  system, so this account is an identity boundary but NOT a")
                print("  privilege boundary. The read-only guarantee is enforced by the")
                print("  application guard instead:")
                print("    - an EXPLAIN gate that rejects any statement Neo4j")
                print("      classifies as a write")
                print("    - a clause allowlist that rejects every procedure call,")
                print("      LOAD CSV and USE")
                print("    - a post-execution check on the transaction counters")
                print("")
                print("  For a server-enforced read-only account, run Neo4j Enterprise,")
                print("  which supports GRANT MATCH {*} ON GRAPH.")
            else:
                print("  The server will refuse any write from this account.")

            print("")
            print("Set NEO4J_READONLY_USER and NEO4J_READONLY_PASSWORD in the app environment.")
    finally:
        driver.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
