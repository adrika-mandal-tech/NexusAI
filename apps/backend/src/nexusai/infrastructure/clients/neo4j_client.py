"""Neo4j knowledge graph client wrapper."""

from __future__ import annotations

from typing import Any

from neo4j import AsyncGraphDatabase

from nexusai.core.config import get_settings

settings = get_settings()


class Neo4jClient:
    """Wrapper for Neo4j operations."""

    def __init__(self) -> None:
        self._uri = settings.NEO4J_URL
        self._user = settings.NEO4J_USER
        self._password = settings.NEO4J_PASSWORD
        self._driver = AsyncGraphDatabase.driver(
            self._uri,
            auth=(self._user, self._password),
        )

    async def close(self) -> None:
        await self._driver.close()

    async def health(self) -> dict[str, Any]:
        try:
            async with self._driver.session() as session:
                await session.run("RETURN 1")
            return {"status": "ok"}
        except Exception as exc:  # noqa: BLE001
            return {"status": "error", "detail": str(exc)}

    async def create_source_node(self, source_id: str, title: str, source_type: str) -> None:
        query = """
        MERGE (s:Source {id: $source_id})
        SET s.title = $title, s.type = $source_type, s.created_at = datetime()
        """
        async with self._driver.session() as session:
            await session.run(query, {"source_id": source_id, "title": title, "source_type": source_type})

    async def create_entity(self, entity_id: str, name: str, entity_type: str, source_id: str) -> None:
        query = """
        MERGE (e:Entity {id: $entity_id})
        SET e.name = $name, e.type = $entity_type, e.source_id = $source_id
        """
        async with self._driver.session() as session:
            await session.run(query, {"entity_id": entity_id, "name": name, "entity_type": entity_type, "source_id": source_id})

    async def create_relationship(self, source_id: str, source_name: str, target_name: str, relation: str) -> None:
        query = """
        MATCH (s:Entity {name: $source_name, source_id: $source_id})
        MERGE (t:Entity {name: $target_name, source_id: $source_id})
        MERGE (s)-[r:RELATED {type: $relation, source_id: $source_id}]->(t)
        """
        async with self._driver.session() as session:
            await session.run(query, {"source_id": source_id, "source_name": source_name, "target_name": target_name, "relation": relation})

    async def get_graph(self, source_id: str | None = None) -> dict[str, Any]:
        match = "MATCH (n)-[r]->(m) RETURN n, r, m"
        params: dict[str, Any] = {}
        if source_id:
            match = "MATCH (n)-[r]->(m) WHERE r.source_id = $source_id RETURN n, r, m"
            params["source_id"] = source_id
        nodes: dict[str, dict[str, Any]] = {}
        edges: list[dict[str, Any]] = []
        async with self._driver.session() as session:
            result = await session.run(match, params)
            async for record in result:
                n = record["n"]
                m = record["m"]
                r = record["r"]
                n_id = n.get("id") or n.get("name") or str(hash(str(dict(n))))[:8]
                m_id = m.get("id") or m.get("name") or str(hash(str(dict(m))))[:8]
                if n_id not in nodes:
                    nodes[n_id] = {"id": n_id, "label": n.get("name") or n.get("id") or n_id, "type": n.get("type", "Entity")}
                if m_id not in nodes:
                    nodes[m_id] = {"id": m_id, "label": m.get("name") or m.get("id") or m_id, "type": m.get("type", "Entity")}
                rel_type = getattr(r, "type", None) or r.get("type", "RELATED") if hasattr(r, "get") else getattr(r, "type", "RELATED")
                edges.append({"source": n_id, "target": m_id, "label": rel_type})
        return {"nodes": list(nodes.values()), "edges": edges}

    async def execute_query(self, query: str, params: dict[str, Any] | None = None) -> Any:
        async with self._driver.session() as session:
            result = await session.run(query, params or {})
            return await result.data()
