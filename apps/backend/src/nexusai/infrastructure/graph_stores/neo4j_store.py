"""Neo4j graph store implementation supporting Document KG, CSV KG, and Shared Identity."""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any
from uuid import UUID

from nexusai.core.config import get_settings
from nexusai.core.logging import get_logger
from nexusai.domain.ports.graph_store import GraphStorePort
from nexusai.infrastructure.graph_stores.neo4j_driver_factory import (
    get_database_name,
    get_writer_driver,
)
from nexusai.infrastructure.graph_stores.neo4j_readonly_store import (
    FORBIDDEN_CLAUSES,
    READ_ONLY_QUERY_TYPE,
    ReadOnlyViolation,
)

logger = get_logger(__name__)

_INTERNAL_LABELS = frozenset({"GraphSchemaProposal", "IngestionJob", "SourceFile", "Document", "Chunk"})
_SAFE_ENTITY_SUBLABELS: dict[str, str] = {
    "TECHNOLOGY": "Technology",
    "DATABASE": "Database",
    "FRAMEWORK": "Framework",
    "MODEL": "Model",
    "ALGORITHM": "Algorithm",
    "ARCHITECTURE": "Architecture",
    "HARDWARE": "Hardware",
    "METRIC": "Metric",
    "ORGANIZATION": "Organization",
    "CONCEPT": "Concept",
    "TOOL": "Tool",
    "LANGUAGE": "Language",
    "SERVICE": "Service",
    "CODESYMBOL": "CodeSymbol",
    "PERSON": "Person",
    "LOCATION": "Location",
    "PRODUCT": "Product",
    "ACRONYM": "Technology",
}
_STATEMENT_SPLIT_RE = re.compile(r";\s*(?=(?:[^'\"]|'[^']*'|\"[^\"]*\")*$)")
_QUOTED_STR_RE = re.compile(r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"")


class Neo4jGraphStore(GraphStorePort):
    """Neo4j implementation of GraphStorePort using the shared process-wide driver pool."""

    def __init__(self, driver: Any | None = None, owns_driver: bool = False) -> None:
        self._driver = driver or get_writer_driver()
        # Never close the shared process-wide singleton driver from neo4j_driver_factory
        self._owns_driver = bool(driver is not None and owns_driver)
        self._database = get_database_name()

    async def close(self) -> None:
        """Close the driver only if owned exclusively by this instance."""
        if self._owns_driver:
            await self._driver.close()

    async def health(self) -> dict[str, Any]:
        """Verify live Neo4j connectivity and return basic node/relationship counts."""
        try:
            await self.run_read("RETURN 1 AS ok")
            node_rows = await self.run_read(
                "MATCH (n) WHERE none(lbl IN labels(n) WHERE lbl IN ['GraphSchemaProposal', 'IngestionJob']) RETURN count(n) AS cnt"
            )
            rel_rows = await self.run_read("MATCH ()-[r]->() RETURN count(r) AS cnt")
            n_cnt = int(node_rows[0]["cnt"]) if node_rows else 0
            r_cnt = int(rel_rows[0]["cnt"]) if rel_rows else 0
            return {
                "status": "ok",
                "database": self._database,
                "node_count": n_cnt,
                "nodes_count": n_cnt,
                "relationships_count": r_cnt,
            }
        except Exception as exc:
            logger.warning("Neo4j health check failed: %s", exc)
            return {"status": "error", "detail": str(exc)}

    async def get_schema_summary(self) -> dict[str, Any]:
        """Return labels, relationship types, constraints, and per-label counts."""
        label_rows = await self.run_read(
            "MATCH (n) UNWIND labels(n) AS lbl RETURN lbl AS label, count(n) AS count ORDER BY count DESC"
        )
        rel_rows = await self.run_read(
            "MATCH ()-[r]->() RETURN type(r) AS type, count(r) AS count ORDER BY count DESC"
        )
        constraint_rows = await self.run_read(
            "SHOW CONSTRAINTS YIELD name, type, labelsOrTypes, properties RETURN name, type, labelsOrTypes, properties"
        )
        return {
            "labels": label_rows,
            "relationship_types": rel_rows,
            "constraints": constraint_rows,
            "total_nodes": sum(int(r.get("count", 0)) for r in label_rows),
            "total_relationships": sum(int(r.get("count", 0)) for r in rel_rows),
        }

    async def get_graph_visualization(
        self,
        source_id: str | None = None,
        limit: int = 250,
    ) -> dict[str, Any]:
        """Return bounded Neo4j-style nodes, typed edges, degrees, and technical metadata across Document & CSV graphs."""
        safe_limit = max(10, min(1000, int(limit)))
        doc_edge_limit = max(25, safe_limit // 3)

        if source_id:
            query = """
            MATCH (d:Document {id: $source_id})-[:HAS_CHUNK]->(c:Chunk)-[:HAS_ENTITY|MENTIONS]->(e:Entity)
            OPTIONAL MATCH (e)-[r]->(e2:Entity)
            RETURN d AS doc_node, labels(d) AS doc_labels, elementId(d) AS doc_eid,
                   e AS n, labels(e) AS n_labels, elementId(e) AS n_eid,
                   r AS rel, coalesce(r.type, type(r)) AS rel_type,
                   e2 AS m, labels(e2) AS m_labels, elementId(e2) AS m_eid
            LIMIT $limit
            """
            params: dict[str, Any] = {"source_id": str(source_id), "limit": safe_limit}
        else:
            query = """
            MATCH (n)-[r]->(m)
            WHERE NOT any(lbl IN labels(n) WHERE lbl IN ['Chunk', 'GraphSchemaProposal', 'IngestionJob'])
              AND NOT any(lbl IN labels(m) WHERE lbl IN ['Chunk', 'GraphSchemaProposal', 'IngestionJob'])
            RETURN null AS doc_node, [] AS doc_labels, null AS doc_eid,
                   n AS n, labels(n) AS n_labels, elementId(n) AS n_eid,
                   r AS rel, coalesce(r.type, type(r)) AS rel_type,
                   m AS m, labels(m) AS m_labels, elementId(m) AS m_eid
            LIMIT $limit
            """
            params = {"limit": safe_limit}

        rows = await self.run_read(query, params)
        nodes: dict[str, dict[str, Any]] = {}
        edges: list[dict[str, Any]] = []
        seen_edges: set[tuple[str, str, str]] = set()

        def _normalize_label(raw_type: str, lbls: list[str]) -> str:
            non_generic = [lbl for lbl in lbls if lbl != "Entity"]
            if non_generic:
                return non_generic[0]
            mapped = _SAFE_ENTITY_SUBLABELS.get(raw_type.upper())
            if mapped:
                return mapped
            if raw_type and raw_type.upper() != "ENTITY":
                return raw_type.title() if raw_type.isupper() else raw_type
            return lbls[0] if lbls else "Entity"

        def _infer_doc_type(p: dict[str, Any]) -> str | None:
            dt = p.get("doc_type") or p.get("source_type")
            if dt:
                return str(dt).upper().lstrip(".")
            fn = str(p.get("filename") or p.get("source_file") or "")
            if "." in fn:
                return fn.rsplit(".", 1)[-1].upper()
            return None

        def _extract_node(props: dict[str, Any] | None, labels: list[str] | None, eid: str | None) -> tuple[str, dict[str, Any]] | None:
            if not props and not eid:
                return None
            p = dict(props or {})
            lbls = [lbl for lbl in (labels or []) if lbl]
            raw_prop_type = str(p.get("type") or "")
            primary_type = _normalize_label(raw_prop_type, lbls)
            if primary_type not in lbls and primary_type:
                lbls = [primary_type, *[lbl for lbl in lbls if lbl != primary_type]]

            nid = str(
                p.get("id")
                or p.get("entity_id")
                or p.get("customer_id")
                or p.get("order_id")
                or p.get("product_id")
                or p.get("supplier_id")
                or eid
                or p.get("name")
            )
            display = str(
                p.get("name")
                or p.get("company_name")
                or p.get("product_name")
                or p.get("contact_name")
                or p.get("filename")
                or p.get("title")
                or p.get("order_id")
                or nid
            )
            doc_type = _infer_doc_type(p)
            clean_props = {
                k: v for k, v in p.items()
                if isinstance(v, (str, int, float, bool)) and len(str(v)) < 280
            }
            if doc_type:
                clean_props.setdefault("doc_type", doc_type)
            return nid, {
                "id": nid,
                "element_id": str(eid or nid),
                "label": display[:68],
                "type": primary_type,
                "labels": lbls,
                "doc_type": doc_type,
                "source_file": p.get("filename") or p.get("source_file"),
                "properties": clean_props,
            }

        def _add_edge(src_id: str, dst_id: str, rel_label: str, rel_props: dict[str, Any] | None = None) -> None:
            if not src_id or not dst_id or not rel_label or src_id == dst_id:
                return
            clean_rel = str(rel_label).strip().upper().replace(" ", "_")
            key = (src_id, dst_id, clean_rel)
            if key in seen_edges:
                return
            seen_edges.add(key)
            rp = {
                k: v
                for k, v in dict(rel_props or {}).items()
                if isinstance(v, (str, int, float, bool)) and len(str(v)) < 200
            }
            edges.append(
                {
                    "id": f"{src_id}->{clean_rel}->{dst_id}",
                    "source": src_id,
                    "target": dst_id,
                    "label": clean_rel,
                    "type": clean_rel,
                    "properties": rp,
                }
            )

        for row in rows:
            doc_info = _extract_node(row.get("doc_node"), row.get("doc_labels"), row.get("doc_eid"))
            if doc_info:
                nodes.setdefault(doc_info[0], doc_info[1])
            n_info = _extract_node(row.get("n"), row.get("n_labels"), row.get("n_eid"))
            if n_info:
                nodes.setdefault(n_info[0], n_info[1])
            if doc_info and n_info:
                _add_edge(doc_info[0], n_info[0], "EXTRACTED_ENTITY", {"doc_type": doc_info[1].get("doc_type") or "DOC"})
            m_info = _extract_node(row.get("m"), row.get("m_labels"), row.get("m_eid"))
            if m_info:
                nodes.setdefault(m_info[0], m_info[1])
            if n_info and m_info and row.get("rel_type"):
                rel_obj = row.get("rel")
                r_props = dict(rel_obj) if isinstance(rel_obj, dict) else {}
                _add_edge(n_info[0], m_info[0], str(row["rel_type"]), r_props)

        # Also bridge ingested `:Document` nodes -> `:Entity` nodes (even when linked via intermediate `:Chunk` nodes)
        # so every ingested document & its format (`PDF`, `DOCX`, `CSV`, `MD`) is clearly visible in the Neo4j graph!
        if not source_id:
            doc_bridge_rows = await self.run_read(
                """
                MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk)-[:HAS_ENTITY|MENTIONS]->(e:Entity)
                WITH d, e, count(DISTINCT c) AS mention_chunks
                ORDER BY mention_chunks DESC
                LIMIT $doc_limit
                RETURN d AS doc_node, labels(d) AS doc_labels, elementId(d) AS doc_eid,
                       e AS ent_node, labels(e) AS ent_labels, elementId(e) AS ent_eid,
                       mention_chunks
                """,
                {"doc_limit": doc_edge_limit},
            )
            for row in doc_bridge_rows:
                d_info = _extract_node(row.get("doc_node"), row.get("doc_labels"), row.get("doc_eid"))
                e_info = _extract_node(row.get("ent_node"), row.get("ent_labels"), row.get("ent_eid"))
                if d_info and e_info:
                    nodes.setdefault(d_info[0], d_info[1])
                    nodes.setdefault(e_info[0], e_info[1])
                    if not e_info[1].get("doc_type") and d_info[1].get("doc_type"):
                        nodes[e_info[0]]["doc_type"] = d_info[1]["doc_type"]
                        nodes[e_info[0]]["properties"].setdefault("doc_type", d_info[1]["doc_type"])
                    if not e_info[1].get("source_file") and d_info[1].get("source_file"):
                        nodes[e_info[0]]["source_file"] = d_info[1]["source_file"]
                        nodes[e_info[0]]["properties"].setdefault("source_file", d_info[1]["source_file"])
                    _add_edge(
                        d_info[0],
                        e_info[0],
                        "MENTIONS",
                        {
                            "mention_chunks": int(row.get("mention_chunks") or 1),
                            "doc_type": d_info[1].get("doc_type") or "DOC",
                        },
                    )

        # If no edges matched (e.g. isolated entities), include top standalone entities
        if not nodes:
            fallback_rows = await self.run_read(
                """
                MATCH (n)
                WHERE NOT any(lbl IN labels(n) WHERE lbl IN ['Chunk', 'GraphSchemaProposal', 'IngestionJob'])
                RETURN n AS n, labels(n) AS n_labels, elementId(n) AS n_eid
                LIMIT $limit
                """,
                {"limit": min(100, safe_limit)},
            )
            for row in fallback_rows:
                n_info = _extract_node(row.get("n"), row.get("n_labels"), row.get("n_eid"))
                if n_info:
                    nodes.setdefault(n_info[0], n_info[1])

        return self._attach_graph_degrees_and_stats(list(nodes.values()), edges)

    @staticmethod
    def _attach_graph_degrees_and_stats(
        node_list: list[dict[str, Any]],
        edge_list: list[dict[str, Any]],
    ) -> dict[str, Any]:
        in_deg: Counter[str] = Counter()
        out_deg: Counter[str] = Counter()
        label_counts: Counter[str] = Counter()
        rel_counts: Counter[str] = Counter()
        doc_types: Counter[str] = Counter()

        for e in edge_list:
            out_deg[str(e["source"])] += 1
            in_deg[str(e["target"])] += 1
            rel_counts[str(e.get("label") or e.get("type") or "RELATED")] += 1

        for n in node_list:
            nid = str(n["id"])
            ind = in_deg[nid]
            outd = out_deg[nid]
            n["in_degree"] = ind
            n["out_degree"] = outd
            n["degree"] = ind + outd
            label_counts[str(n.get("type") or "Entity")] += 1
            if n.get("doc_type"):
                doc_types[str(n["doc_type"])] += 1

        return {
            "nodes": node_list,
            "edges": edge_list,
            "stats": {
                "node_count": len(node_list),
                "edge_count": len(edge_list),
                "label_counts": dict(label_counts.most_common()),
                "relationship_counts": dict(rel_counts.most_common()),
                "ingested_doc_types": dict(doc_types.most_common()),
            },
        }

    async def create_document_node(
        self,
        document_id: UUID,
        filename: str,
        source_type: str,
        user_id: UUID,
    ) -> None:
        """Create or update a Document node in Neo4j with explicit ingested `doc_type` metadata."""
        ext = Path(filename).suffix.lower().lstrip(".")
        doc_type = (ext or source_type or "doc").upper()
        query = """
        MERGE (d:Document {id: $document_id})
        SET d.document_id = $document_id,
            d.name = $filename,
            d.filename = $filename,
            d.source_type = $source_type,
            d.doc_type = $doc_type,
            d.user_id = $user_id,
            d.updated_at = datetime(),
            d.created_at = coalesce(d.created_at, datetime())
        """
        await self.run_write(
            query,
            {
                "document_id": str(document_id),
                "filename": filename,
                "source_type": source_type,
                "doc_type": doc_type,
                "user_id": str(user_id),
            },
        )

    async def create_chunk_nodes(self, chunks: list[dict[str, Any]]) -> None:
        """Batch-create Chunk nodes linked to Document with primitive/JSON-safe properties."""
        if not chunks:
            return

        query = """
        UNWIND $chunks AS chunk
        MATCH (d:Document {id: chunk.document_id})
        MERGE (c:Chunk {id: chunk.chunk_id})
        SET c.chunk_id = chunk.chunk_id,
            c.document_id = chunk.document_id,
            c.index = chunk.index,
            c.text = chunk.text,
            c.page_number = chunk.page_number,
            c.section_title = chunk.section_title,
            c.filename = chunk.filename,
            c.entity_ids = chunk.entity_ids,
            c.metadata_json = chunk.metadata_json
        MERGE (d)-[:HAS_CHUNK]->(c)
        """
        chunk_data = []
        for chunk in chunks:
            meta = chunk.get("metadata") or {}
            chunk_data.append(
                {
                    "chunk_id": str(chunk["chunk_id"]),
                    "document_id": str(chunk["document_id"]),
                    "index": int(chunk.get("chunk_index", chunk.get("index", 0))),
                    "text": chunk["text"],
                    "page_number": chunk.get("page_number"),
                    "section_title": chunk.get("section_title"),
                    "filename": chunk.get("filename", ""),
                    "entity_ids": [str(eid) for eid in (chunk.get("entity_ids") or [])],
                    "metadata_json": json.dumps(meta, default=str),
                }
            )

        await self.run_write(query, {"chunks": chunk_data})
        logger.info("Created %d Chunk nodes in Neo4j", len(chunks))

    async def create_entities_and_relationships(
        self,
        chunk_id: UUID,
        document_id: UUID,
        entities: list[dict[str, Any]],
        relationships: list[dict[str, Any]],
    ) -> None:
        """Persist entities and relationships for a single chunk."""
        await self.create_batch_entities_and_relationships(
            document_id=document_id,
            chunk_extractions=[
                {
                    "chunk_id": chunk_id,
                    "entities": entities,
                    "relationships": relationships,
                }
            ],
        )

    async def create_batch_entities_and_relationships(
        self,
        document_id: UUID,
        chunk_extractions: list[dict[str, Any]],
    ) -> None:
        """Batch-persist deterministic Entity nodes and RELATED edges across all chunks of a document."""
        doc_id_str = str(document_id)
        entity_rows: list[dict[str, Any]] = []
        rel_rows: list[dict[str, Any]] = []

        for item in chunk_extractions:
            cid_str = str(item["chunk_id"])
            for ent in item.get("entities") or []:
                entity_rows.append(
                    {
                        "chunk_id": cid_str,
                        "id": str(ent["id"]),
                        "name": str(ent["name"]),
                        "name_lower": str(ent["name"]).lower(),
                        "type": str(ent.get("type") or "ENTITY"),
                    }
                )
            for rel in item.get("relationships") or []:
                rel_rows.append(
                    {
                        "source_id": str(rel["source_id"]),
                        "target_id": str(rel["target_id"]),
                        "type": str(rel.get("type") or "RELATED"),
                    }
                )

        if entity_rows:
            entity_query = """
            UNWIND $rows AS row
            MERGE (e:Entity {id: row.id})
            SET e.entity_id = row.id,
                e.name = row.name,
                e.name_lower = row.name_lower,
                e.type = row.type,
                e.document_id = coalesce(e.document_id, $document_id),
                e.document_ids = CASE
                    WHEN e.document_ids IS NULL THEN [$document_id]
                    WHEN NOT $document_id IN e.document_ids THEN e.document_ids + $document_id
                    ELSE e.document_ids
                END,
                e.mention_count = coalesce(e.mention_count, 0) + 1,
                e.updated_at = datetime()
            WITH e, row
            OPTIONAL MATCH (d:Document {id: $document_id})
            FOREACH (_ IN CASE WHEN d IS NOT NULL THEN [1] ELSE [] END |
                SET e.doc_type = coalesce(d.doc_type, toUpper(d.source_type), e.doc_type),
                    e.source_file = coalesce(d.filename, e.source_file)
                MERGE (d)-[:EXTRACTED_ENTITY]->(e)
            )
            WITH e, row
            MATCH (c:Chunk {id: row.chunk_id})
            MERGE (c)-[:HAS_ENTITY]->(e)
            """
            await self.run_write(entity_query, {"document_id": doc_id_str, "rows": entity_rows})

            # Apply validated secondary Neo4j domain labels (e.g. :Technology, :Database, :Algorithm)
            by_sublabel: dict[str, list[str]] = {}
            for row in entity_rows:
                sub = _SAFE_ENTITY_SUBLABELS.get(str(row.get("type") or "").upper())
                if sub:
                    by_sublabel.setdefault(sub, []).append(row["id"])
            for sublabel, ids in by_sublabel.items():
                await self.run_write(
                    f"UNWIND $ids AS eid MATCH (e:Entity {{id: eid}}) SET e:{sublabel}",
                    {"ids": list(dict.fromkeys(ids))},
                )

        if rel_rows:
            rel_query = """
            UNWIND $rows AS rel
            MATCH (s:Entity {id: rel.source_id})
            MATCH (t:Entity {id: rel.target_id})
            MERGE (s)-[r:RELATED {type: rel.type}]->(t)
            SET r.document_id = coalesce(r.document_id, $document_id),
                r.updated_at = datetime()
            """
            await self.run_write(rel_query, {"document_id": doc_id_str, "rows": rel_rows})

    async def get_chunk_entities(self, chunk_ids: list[UUID]) -> list[dict[str, Any]]:
        """Get entities connected to the given chunk IDs (1-hop expansion) with accurate relationship labels."""
        if not chunk_ids:
            return []

        chunk_id_strs = [str(cid) for cid in chunk_ids]
        query = """
        MATCH (c:Chunk)-[:HAS_ENTITY]->(e:Entity)
        WHERE c.id IN $chunk_ids
        OPTIONAL MATCH (e)-[r:RELATED]-(other:Entity)
        RETURN c.id AS chunk_id,
               e.id AS entity_id,
               e.name AS entity_name,
               e.type AS entity_type,
               coalesce(r.type, type(r)) AS relationship_type,
               other.id AS related_entity_id,
               other.name AS related_entity_name
        LIMIT 150
        """
        return await self.run_read(query, {"chunk_ids": chunk_id_strs})

    async def search_graph_context(
        self,
        query_terms: list[str],
        limit: int = 15,
        document_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Search both Document `:Entity`/`:Chunk` subgraphs and CSV-ingested domain graph nodes."""
        cleaned_terms = [t.strip().lower() for t in query_terms if len(t.strip()) >= 2][:10]
        if not cleaned_terms:
            return {"chunks": [], "triples": [], "entities": []}

        # 1. Document Knowledge Graph: match Entities -> linked Chunks + 1-hop Entity neighbors
        doc_filter_clause = "AND c.document_id IN $doc_ids" if document_ids else ""
        doc_kg_query = f"""
        MATCH (c:Chunk)-[:HAS_ENTITY]->(e:Entity)
        WHERE any(term IN $terms WHERE toLower(e.name) CONTAINS term)
        {doc_filter_clause}
        OPTIONAL MATCH (e)-[r:RELATED]-(nbr:Entity)
        RETURN c.id AS chunk_id,
               c.document_id AS document_id,
               c.text AS text,
               c.page_number AS page_number,
               c.section_title AS section_title,
               c.filename AS filename,
               e.id AS entity_id,
               e.name AS entity_name,
               e.type AS entity_type,
               coalesce(r.type, type(r)) AS rel_type,
               nbr.name AS neighbor_name,
               nbr.type AS neighbor_type
        LIMIT $limit
        """
        doc_records = await self.run_read(
            doc_kg_query,
            {"terms": cleaned_terms, "doc_ids": document_ids or [], "limit": limit * 2},
        )

        chunks_by_id: dict[str, dict[str, Any]] = {}
        triples: list[dict[str, Any]] = []
        seen_triples: set[tuple[str, str, str]] = set()
        entities_by_id: dict[str, dict[str, Any]] = {}

        for rec in doc_records:
            cid = rec.get("chunk_id")
            if cid and cid not in chunks_by_id:
                chunks_by_id[cid] = {
                    "chunk_id": cid,
                    "document_id": rec.get("document_id"),
                    "text": rec.get("text") or "",
                    "page_number": rec.get("page_number"),
                    "section_title": rec.get("section_title"),
                    "filename": rec.get("filename") or "",
                    "score": 0.85,
                    "retrieval_channel": "graph_document",
                    "entity_ids": [rec["entity_id"]] if rec.get("entity_id") else [],
                    "metadata": {"matched_entity": rec.get("entity_name")},
                }
            elif cid and rec.get("entity_id"):
                eids = chunks_by_id[cid].setdefault("entity_ids", [])
                if rec["entity_id"] not in eids:
                    eids.append(rec["entity_id"])

            eid = rec.get("entity_id")
            if eid and eid not in entities_by_id:
                entities_by_id[eid] = {
                    "entity_id": eid,
                    "name": rec.get("entity_name"),
                    "type": rec.get("entity_type"),
                    "source": "document_kg",
                }

            if rec.get("entity_name") and rec.get("neighbor_name") and rec.get("rel_type"):
                t_key = (str(rec["entity_name"]), str(rec["rel_type"]), str(rec["neighbor_name"]))
                if t_key not in seen_triples:
                    seen_triples.add(t_key)
                    triples.append(
                        {
                            "source": t_key[0],
                            "relationship": t_key[1],
                            "target": t_key[2],
                            "origin": rec.get("filename") or "document_kg",
                        }
                    )

        # 2. CSV Domain Knowledge Graph: search domain nodes (excluding internal metadata labels)
        if not document_ids:
            csv_kg_query = """
            MATCH (n)
            WHERE none(lbl IN labels(n) WHERE lbl IN ['GraphSchemaProposal', 'IngestionJob', 'SourceFile', 'Document', 'Chunk', 'Entity'])
              AND (
                any(term IN $terms WHERE any(lbl IN labels(n) WHERE toLower(lbl) CONTAINS term))
                OR any(k IN keys(n) WHERE any(term IN $terms WHERE toLower(toString(n[k])) CONTAINS term))
              )
            OPTIONAL MATCH (n)-[r]-(m)
            WHERE m IS NULL OR none(lbl IN labels(m) WHERE lbl IN ['GraphSchemaProposal', 'IngestionJob', 'SourceFile', 'Document', 'Chunk'])
            RETURN labels(n) AS n_labels,
                   properties(n) AS n_props,
                   type(r) AS rel_type,
                   startNode(r) = n AS outgoing,
                   labels(m) AS m_labels,
                   properties(m) AS m_props
            LIMIT $limit
            """
            csv_records = await self.run_read(
                csv_kg_query,
                {"terms": cleaned_terms, "limit": limit * 2},
            )

            for idx, rec in enumerate(csv_records):
                n_labels = rec.get("n_labels") or ["Record"]
                n_props = rec.get("n_props") or {}
                n_label = n_labels[0]
                n_display = self._display_prop(n_props, n_label)
                source_file = n_props.get("source_file", "csv_knowledge_graph")
                source_row = n_props.get("source_row")

                m_props = rec.get("m_props") or {}
                m_labels = rec.get("m_labels") or []
                rel_type = rec.get("rel_type")

                rel_line = ""
                if rel_type and m_labels:
                    m_display = self._display_prop(m_props, m_labels[0])
                    if rec.get("outgoing", True):
                        t_key = (f"{n_label}:{n_display}", str(rel_type), f"{m_labels[0]}:{m_display}")
                    else:
                        t_key = (f"{m_labels[0]}:{m_display}", str(rel_type), f"{n_label}:{n_display}")
                    if t_key not in seen_triples:
                        seen_triples.add(t_key)
                        triples.append(
                            {
                                "source": t_key[0],
                                "relationship": t_key[1],
                                "target": t_key[2],
                                "origin": source_file,
                                "source_row": source_row,
                            }
                        )
                    rel_line = f" | Related: ({t_key[0]}) -[:{t_key[1]}]-> ({t_key[2]})"

                synth_id = f"csv:{source_file}:{source_row or idx}:{n_label}:{n_display}"
                if synth_id not in chunks_by_id and len(chunks_by_id) < limit:
                    clean_props = {
                        k: v
                        for k, v in n_props.items()
                        if k not in {"job_id"} and v is not None
                    }
                    props_str = ", ".join(f"{k}={v}" for k, v in list(clean_props.items())[:12])
                    chunks_by_id[synth_id] = {
                        "chunk_id": synth_id,
                        "document_id": f"csv:{source_file}",
                        "text": f"[Knowledge Graph {n_label}] {props_str}{rel_line}",
                        "page_number": int(source_row) if isinstance(source_row, int) else None,
                        "section_title": f"Graph Entity: {n_label}",
                        "filename": str(source_file),
                        "score": 0.82,
                        "retrieval_channel": "graph_csv",
                        "metadata": {
                            "source_file": source_file,
                            "source_row": source_row,
                            "node_label": n_label,
                        },
                    }

        return {
            "chunks": list(chunks_by_id.values())[:limit],
            "triples": triples[: limit * 2],
            "entities": list(entities_by_id.values())[:limit],
        }

    @staticmethod
    def _display_prop(props: dict[str, Any], fallback_label: str) -> str:
        for key in ("name", "company_name", "product_name", "contact_name", "title", "id"):
            if key in props and props[key]:
                return str(props[key])
        for key, val in props.items():
            if key.endswith("_id") and val is not None:
                return f"{key}={val}"
        return fallback_label

    async def verify_document_chunks(self, document_id: UUID) -> dict[str, int]:
        """Return persisted Chunk and Entity counts in Neo4j for cross-DB consistency checks."""
        doc_id_str = str(document_id)
        records = await self.run_read(
            """
            OPTIONAL MATCH (d:Document {id: $doc_id})-[:HAS_CHUNK]->(c:Chunk)
            WITH count(DISTINCT c) AS chunk_count
            OPTIONAL MATCH (c2:Chunk {document_id: $doc_id})-[:HAS_ENTITY]->(e:Entity)
            RETURN chunk_count, count(DISTINCT e) AS entity_count
            """,
            {"doc_id": doc_id_str},
        )
        if not records:
            return {"chunk_count": 0, "entity_count": 0}
        return {
            "chunk_count": int(records[0].get("chunk_count") or 0),
            "entity_count": int(records[0].get("entity_count") or 0),
        }

    async def get_graph(self, document_id: UUID | None = None, limit: int = 250) -> dict[str, Any]:
        """Get unified graph visualization data covering both Document entities and CSV domain entities."""
        params: dict[str, Any] = {"limit": limit}
        if document_id:
            match = """
            MATCH (n)-[r]->(m)
            WHERE (n.document_id = $document_id OR $document_id IN coalesce(n.document_ids, []))
               OR (m.document_id = $document_id OR $document_id IN coalesce(m.document_ids, []))
            RETURN n, r, m
            LIMIT $limit
            """
            params["document_id"] = str(document_id)
        else:
            match = """
            MATCH (n)-[r]->(m)
            WHERE none(lbl IN labels(n) WHERE lbl IN ['GraphSchemaProposal', 'IngestionJob'])
              AND none(lbl IN labels(m) WHERE lbl IN ['GraphSchemaProposal', 'IngestionJob'])
            RETURN n, r, m
            LIMIT $limit
            """

        nodes: dict[str, dict[str, Any]] = {}
        edges: list[dict[str, Any]] = []

        async with self._driver.session(database=self._database) as session:
            result = await session.run(match, params)
            async for record in result:
                n = record["n"]
                m = record["m"]
                r = record["r"]

                n_id, n_obj = self._node_to_viz(n)
                m_id, m_obj = self._node_to_viz(m)
                nodes.setdefault(n_id, n_obj)
                nodes.setdefault(m_id, m_obj)

                rel_type = r.get("type") if "type" in r else r.type
                edges.append(
                    {
                        "source": n_id,
                        "target": m_id,
                        "label": str(rel_type or "RELATED"),
                    }
                )

        # Also include isolated domain/entity nodes if graph has few edges
        if len(nodes) < 20 and not document_id:
            iso_records = await self.run_read(
                """
                MATCH (n)
                WHERE none(lbl IN labels(n) WHERE lbl IN ['GraphSchemaProposal', 'IngestionJob'])
                RETURN n LIMIT 50
                """
            )
            async with self._driver.session(database=self._database) as session:
                res = await session.run(
                    "MATCH (n) WHERE none(lbl IN labels(n) WHERE lbl IN ['GraphSchemaProposal', 'IngestionJob']) RETURN n LIMIT 50"
                )
                async for rec in res:
                    nid, nobj = self._node_to_viz(rec["n"])
                    nodes.setdefault(nid, nobj)
            _ = iso_records

        return {"nodes": list(nodes.values()), "edges": edges}

    def _node_to_viz(self, node: Any) -> tuple[str, dict[str, Any]]:
        props = dict(node)
        labels = list(node.labels) if getattr(node, "labels", None) else ["Entity"]
        primary_label = labels[0]
        node_id = str(
            props.get("id")
            or props.get("chunk_id")
            or f"{primary_label}:{self._display_prop(props, getattr(node, 'element_id', 'node'))}"
        )
        display = (
            props.get("name")
            or props.get("filename")
            or self._display_prop(props, node_id[:16])
        )
        if primary_label == "Chunk":
            idx = props.get("index", 0)
            display = f"Chunk #{idx}"
        node_type = str(props.get("type") or primary_label)
        return node_id, {
            "id": node_id,
            "label": str(display)[:64],
            "type": node_type,
            "category": primary_label,
            "source_file": props.get("source_file") or props.get("filename"),
            "source_row": props.get("source_row"),
        }

    async def delete_document_graph(self, document_id: UUID) -> None:
        """Delete document & chunk nodes and garbage-collect unreferenced Entity nodes."""
        doc_id_str = str(document_id)
        queries = [
            "MATCH (c:Chunk {document_id: $doc_id}) DETACH DELETE c",
            "MATCH (d:Document {id: $doc_id}) DETACH DELETE d",
            """
            MATCH (e:Entity)
            WHERE e.document_id = $doc_id OR $doc_id IN coalesce(e.document_ids, [])
            SET e.document_ids = [x IN coalesce(e.document_ids, []) WHERE x <> $doc_id]
            WITH e
            WHERE NOT (:Chunk)-[:HAS_ENTITY]->(e)
            DETACH DELETE e
            """,
        ]
        for q in queries:
            await self.run_write(q, {"doc_id": doc_id_str})

        logger.info("Deleted graph data for document %s", doc_id_str)

    async def execute_cypher_console(
        self,
        cypher: str,
        parameters: dict[str, Any] | None = None,
        max_rows: int | None = None,
    ) -> dict[str, Any]:
        """Execute a safeguarded, single-statement read-only Cypher query for the Neo4j Browser Console.

        Security layers:
        1. Single-statement enforcement (rejects `;`-separated multi-statement payloads).
        2. Comment & string-literal-stripped forbidden write/admin clause check (`CREATE`, `MERGE`,
           `DELETE`, `DETACH`, `SET`, `REMOVE`, `DROP`, `LOAD CSV`, `CALL`, `GRANT`, `ALTER`, etc.).
        3. Server-side `EXPLAIN` plan gate verifying `summary.query_type == 'r'`.
        4. Managed `session.execute_read()` transaction with 5s timeout, row cap (`<=200`), and
           post-execution `counters.contains_updates == False` assertion.
        """
        raw = (cypher or "").strip()
        if not raw:
            raise ReadOnlyViolation("Cypher query cannot be empty.")
        if len(raw) > 20_000:
            raise ReadOnlyViolation("Cypher query exceeds maximum length (20,000 characters).")

        statements = [s for s in _STATEMENT_SPLIT_RE.split(raw) if s.strip()]
        if len(statements) > 1:
            raise ReadOnlyViolation("Only a single read-only Cypher statement is permitted.")

        clean_stmt = statements[0].strip().rstrip(";").strip()
        no_comments = re.sub(r"//[^\n]*", " ", clean_stmt)
        no_comments = re.sub(r"/\*.*?\*/", " ", no_comments, flags=re.DOTALL)
        no_strings = _QUOTED_STR_RE.sub(" '' ", no_comments).lower()

        for forbidden in FORBIDDEN_CLAUSES:
            if re.search(rf"\b{re.escape(forbidden)}\b", no_strings):
                raise ReadOnlyViolation(
                    f"Forbidden mutating/admin clause '{forbidden.upper()}' rejected. "
                    "The Neo4j Cypher Console permits read-only MATCH/RETURN queries only."
                )

        settings = get_settings()
        row_cap = max(1, min(500, max_rows or settings.GRAPH_QUERY_MAX_ROWS or 200))
        timeout_sec = max(1.0, (settings.GRAPH_QUERY_TIMEOUT_MS or 5000) / 1000.0)
        params = dict(parameters or {})

        # Auto-append bounded LIMIT when user omits LIMIT (like Neo4j Browser auto-limit)
        effective_cypher = (
            clean_stmt
            if re.search(r"\blimit\b", no_strings)
            else f"{clean_stmt}\nLIMIT {row_cap}"
        )

        t0 = time.perf_counter()

        # Layer 3: Neo4j parser EXPLAIN gate
        async with self._driver.session(database=self._database) as session:
            explain_res = await session.run(f"EXPLAIN {effective_cypher}", params)
            explain_summary = await explain_res.consume()
        q_type = getattr(explain_summary, "query_type", None)
        if q_type != READ_ONLY_QUERY_TYPE:
            raise ReadOnlyViolation(
                f"Rejected non-read-only Cypher plan (query_type={q_type!r}). Only read queries ('r') are allowed."
            )

        nodes_by_id: dict[str, dict[str, Any]] = {}
        edges_list: list[dict[str, Any]] = []
        seen_edge_keys: set[tuple[str, str, str]] = set()
        elem_id_to_node_id: dict[str, str] = {}

        def _register_neo4j_node(n_obj: Any) -> dict[str, Any]:
            props = dict(n_obj)
            lbls = list(getattr(n_obj, "labels", ()) or ())
            eid = str(getattr(n_obj, "element_id", "") or "")
            non_gen = [lbl for lbl in lbls if lbl != "Entity"]
            raw_t = str(props.get("type") or "")
            p_type = (
                non_gen[0]
                if non_gen
                else (_SAFE_ENTITY_SUBLABELS.get(raw_t.upper()) or (raw_t.title() if raw_t else (lbls[0] if lbls else "Node")))
            )
            if p_type and p_type not in lbls:
                lbls = [p_type, *lbls]
            nid = str(
                props.get("id")
                or props.get("entity_id")
                or props.get("chunk_id")
                or props.get("customer_id")
                or props.get("order_id")
                or props.get("product_id")
                or props.get("supplier_id")
                or eid
                or props.get("name")
            )
            if eid:
                elem_id_to_node_id[eid] = nid
            order_num = props.get("orderID") or props.get("order_id") or props.get("orderId")
            cust_id = props.get("customerID") or props.get("customer_id") or props.get("customerId")
            supp_name = props.get("companyName") or props.get("company_name")
            prod_name = props.get("productName") or props.get("product_name")

            disp = str(
                props.get("name")
                or supp_name
                or prod_name
                or props.get("contactName")
                or props.get("contact_name")
                or (f"Order #{order_num}" if order_num and "Order" in lbls else None)
                or (f"Customer {cust_id}" if cust_id and "Customer" in lbls else None)
                or props.get("title")
                or props.get("filename")
                or (f"Chunk #{props.get('index', 0)}" if "Chunk" in lbls else "")
                or nid
            )
            dt = props.get("doc_type") or props.get("source_type")
            if not dt and props.get("filename") and "." in str(props.get("filename")):
                dt = str(props["filename"]).rsplit(".", 1)[-1]
            doc_type_str = str(dt).upper().lstrip(".") if dt else None
            clean_p = {
                k: (v if isinstance(v, (str, int, float, bool)) else str(v))
                for k, v in props.items()
                if v is not None and len(str(v)) < 320
            }
            node_dict = {
                "id": nid,
                "element_id": eid or nid,
                "label": disp[:68],
                "type": p_type,
                "labels": lbls,
                "doc_type": doc_type_str,
                "source_file": props.get("filename") or props.get("source_file"),
                "properties": clean_p,
            }
            nodes_by_id.setdefault(nid, node_dict)
            return node_dict

        def _register_neo4j_rel(r_obj: Any) -> dict[str, Any] | None:
            r_props = dict(r_obj) if (hasattr(r_obj, "items") or isinstance(r_obj, dict)) else {}
            r_label = str(r_props.get("type") or getattr(r_obj, "type", "RELATED")).upper()
            s_node = getattr(r_obj, "start_node", None)
            e_node = getattr(r_obj, "end_node", None)

            def _extract_ref_id(ref: Any) -> str:
                if ref is None:
                    return ""
                if hasattr(ref, "labels") or hasattr(ref, "element_id"):
                    try:
                        return _register_neo4j_node(ref)["id"]
                    except Exception:
                        pass
                    eid = str(getattr(ref, "element_id", "") or "")
                    if eid:
                        return elem_id_to_node_id.get(eid, eid)
                s_str = str(ref)
                return elem_id_to_node_id.get(s_str, s_str)

            src_id = _extract_ref_id(s_node)
            dst_id = _extract_ref_id(e_node)

            if not src_id or not dst_id:
                nodes_tuple = getattr(r_obj, "nodes", ()) or ()
                if len(nodes_tuple) == 2:
                    src_id = src_id or _extract_ref_id(nodes_tuple[0])
                    dst_id = dst_id or _extract_ref_id(nodes_tuple[1])

            if not src_id or not dst_id:
                return None
            ekey = (src_id, dst_id, r_label)
            clean_rp = {
                k: (v if isinstance(v, (str, int, float, bool)) else str(v))
                for k, v in r_props.items()
                if v is not None and len(str(v)) < 240
            }
            edge_obj = {
                "id": f"{src_id}->{r_label}->{dst_id}",
                "source": src_id,
                "target": dst_id,
                "label": r_label,
                "type": r_label,
                "properties": clean_rp,
            }
            if ekey not in seen_edge_keys:
                seen_edge_keys.add(ekey)
                edges_list.append(edge_obj)
            return edge_obj

        def _serialize_val(val: Any) -> Any:
            if val is None or isinstance(val, (str, int, float, bool)):
                return val
            if hasattr(val, "isoformat"):
                return val.isoformat()
            # Neo4j Node
            if hasattr(val, "labels") and hasattr(val, "element_id"):
                nd = _register_neo4j_node(val)
                return {
                    "_kind": "node",
                    "id": nd["id"],
                    "label": nd["label"],
                    "labels": nd["labels"],
                    "doc_type": nd["doc_type"],
                    "properties": nd["properties"],
                }
            # Neo4j Path
            if hasattr(val, "nodes") and hasattr(val, "relationships"):
                p_nodes = [_register_neo4j_node(n)["label"] for n in getattr(val, "nodes", ())]
                for r in getattr(val, "relationships", ()):
                    _register_neo4j_rel(r)
                return {"_kind": "path", "hops": max(0, len(p_nodes) - 1), "nodes": p_nodes}
            # Neo4j Relationship
            if hasattr(val, "type") and hasattr(val, "element_id"):
                rel_d = _register_neo4j_rel(val)
                clean_props = (
                    rel_d["properties"]
                    if rel_d
                    else {
                        str(k): (
                            v.isoformat()
                            if hasattr(v, "isoformat")
                            else (v if isinstance(v, (str, int, float, bool)) else str(v))
                        )
                        for k, v in dict(val).items()
                    }
                )
                return {
                    "_kind": "relationship",
                    "type": rel_d["label"] if rel_d else str(getattr(val, "type", "RELATED")),
                    "source": rel_d["source"] if rel_d else None,
                    "target": rel_d["target"] if rel_d else None,
                    "properties": clean_props,
                }
            if isinstance(val, (list, tuple)):
                return [_serialize_val(x) for x in val]
            if isinstance(val, dict):
                return {str(k): _serialize_val(v) for k, v in val.items()}
            return str(val)

        async def _read_tx(tx: Any) -> tuple[list[str], list[Any], Any]:
            res = await tx.run(effective_cypher, params, timeout=timeout_sec)
            cols = list(res.keys())
            recs = [rec async for rec in res]
            summ = await res.consume()
            return cols, recs, summ

        async with self._driver.session(database=self._database) as session:
            columns, raw_records, summary = await session.execute_read(_read_tx)

        counters = getattr(summary, "counters", None)
        if counters is not None and getattr(counters, "contains_updates", False):
            raise ReadOnlyViolation("Rejected query: execution counters reported unexpected write updates.")

        truncated = len(raw_records) > row_cap
        sliced_records = raw_records[:row_cap]

        # Pass 1: register all Node objects first so Relationship endpoints resolve cleanly by element_id
        for rec in sliced_records:
            for col in columns:
                v = rec.get(col)
                if hasattr(v, "labels") and hasattr(v, "element_id"):
                    _register_neo4j_node(v)
                elif hasattr(v, "nodes") and hasattr(v, "relationships"):
                    for pn in getattr(v, "nodes", ()):
                        _register_neo4j_node(pn)

        # Pass 2: serialize all values + relationships
        serialized_rows: list[dict[str, Any]] = []
        for rec in sliced_records:
            serialized_rows.append({col: _serialize_val(rec.get(col)) for col in columns})

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        graph_payload = self._attach_graph_degrees_and_stats(list(nodes_by_id.values()), edges_list)

        return {
            "status": "ok",
            "cypher": effective_cypher,
            "query_type": q_type,
            "columns": columns,
            "rows": serialized_rows,
            "row_count": len(serialized_rows),
            "truncated": truncated,
            "elapsed_ms": elapsed_ms,
            "nodes": graph_payload["nodes"],
            "edges": graph_payload["edges"],
            "stats": graph_payload["stats"],
        }

    # ── CSV-to-graph ingestion & write/read helpers ──────────────────────────

    async def apply_ddl(self, statements: list[str]) -> None:
        """Run schema DDL (constraints, indexes). Each statement is idempotent."""
        async with self._driver.session(database=self._database) as session:
            for statement in statements:
                await session.run(statement)

    async def write_batch(
        self,
        query: str,
        rows: list[dict[str, Any]],
        timeout: float | None = None,
    ) -> dict[str, int]:
        """Write one batch of rows via managed write transaction and return counters."""
        if not rows:
            return {
                "nodes_created": 0,
                "nodes_matched": 0,
                "relationships_created": 0,
                "relationships_matched": 0,
                "rows_processed": 0,
                "rows_unmatched": 0,
            }

        async def _tx(tx: Any) -> Any:
            kwargs = {"timeout": timeout} if timeout else {}
            result = await tx.run(query, rows=rows, **kwargs)
            records = await result.data()
            summary = await result.consume()
            return records, summary

        async with self._driver.session(database=self._database) as session:
            records, summary = await session.execute_write(_tx)

        counters = getattr(summary, "counters", None)
        nodes_created = getattr(counters, "nodes_created", 0) if counters else 0
        rels_created = getattr(counters, "relationships_created", 0) if counters else 0

        processed = len(rows)
        if records:
            try:
                processed = int(records[0].get("processed") or 0)
            except (TypeError, ValueError):
                processed = len(rows)

        return {
            "nodes_created": nodes_created,
            "nodes_matched": max(0, len(rows) - nodes_created),
            "relationships_created": rels_created,
            "relationships_matched": max(0, processed - rels_created),
            "rows_processed": processed,
            "rows_unmatched": max(0, len(rows) - processed),
        }

    async def run_write(self, query: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Execute a parameterized write query against the administrative driver."""
        async with self._driver.session(database=self._database) as session:
            result = await session.run(query, params or {})
            return await result.data()

    async def run_read(self, query: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Run a parameterized read query and return records."""
        async with self._driver.session(database=self._database) as session:
            result = await session.run(query, params or {})
            return await result.data()

    async def count_nodes(self, label: str | None = None) -> int:
        """Count nodes, optionally for a single validated label."""
        from nexusai.domain.entities.graph_schema import validate_identifier

        if label:
            validate_identifier(label, "label")
            query = f"MATCH (n:{label}) RETURN count(n) AS c"
        else:
            query = "MATCH (n) RETURN count(n) AS c"
        records = await self.run_read(query)
        return int(records[0]["c"]) if records else 0

    async def count_relationships(self) -> int:
        """Count all relationships."""
        records = await self.run_read("MATCH ()-[r]->() RETURN count(r) AS c")
        return int(records[0]["c"]) if records else 0

    async def create_ingestion_job(
        self,
        job_id: str,
        source_id: str,
        proposal_id: str,
        status: str = "running",
    ) -> None:
        """Create the job-tracking node for a CSV ingestion run."""
        query = """
        MERGE (j:IngestionJob {id: $job_id})
        SET j.source_id = $source_id,
            j.proposal_id = $proposal_id,
            j.status = $status,
            j.started_at = datetime()
        """
        await self.run_write(
            query,
            {"job_id": job_id, "source_id": source_id, "proposal_id": proposal_id, "status": status},
        )

    async def link_source_file(
        self,
        job_id: str,
        file_id: str,
        filename: str,
        source_id: str,
        rows_read: int,
    ) -> None:
        """Create the per-file provenance node and link it to the job."""
        query = """
        MERGE (f:SourceFile {id: $file_id})
        SET f.filename = $filename,
            f.source_id = $source_id,
            f.rows_read = $rows_read
        WITH f
        MATCH (j:IngestionJob {id: $job_id})
        MERGE (j)-[:CONTAINS]->(f)
        """
        await self.run_write(
            query,
            {
                "job_id": job_id,
                "file_id": file_id,
                "filename": filename,
                "source_id": source_id,
                "rows_read": rows_read,
            },
        )

    async def finalise_source_file(
        self,
        job_id: str,
        file_id: str,
        stats: dict[str, Any],
    ) -> None:
        """Record the per-file ingestion statistics on the provenance node."""
        query = """
        MATCH (j:IngestionJob {id: $job_id})-[:CONTAINS]->(f:SourceFile {id: $file_id})
        SET f.rows_accepted = $rows_accepted,
            f.rows_rejected = $rows_rejected,
            f.nodes_created = $nodes_created,
            f.relationships_created = $relationships_created,
            f.finished_at = datetime()
        """
        await self.run_write(query, {"job_id": job_id, "file_id": file_id, **stats})

    async def finalise_ingestion_job(self, job_id: str, status: str) -> None:
        """Close the job-tracking node."""
        query = """
        MATCH (j:IngestionJob {id: $job_id})
        SET j.status = $status, j.finished_at = datetime()
        """
        await self.run_write(query, {"job_id": job_id, "status": status})
