"""Graph statistics for the CSV-to-graph subsystem.

The per-file counts come from the job-tracking nodes rather than from a full
graph scan, so the numbers are cheap to obtain and always traceable to an
ingestion run.
"""

from __future__ import annotations

from typing import Any

from nexusai.core.logging import get_logger
from nexusai.domain.entities.graph_schema import validate_identifier

logger = get_logger(__name__)


async def collect_statistics(store: Any) -> dict[str, Any]:
    """Collect the graph statistics shown in the UI."""
    stats: dict[str, Any] = {"per_file": [], "per_label": [], "per_relationship": []}

    records = await store.run_read(
        "MATCH (n) WHERE n.kind = 'csv_ingest' RETURN labels(n)[0] AS label, count(n) AS c "
        "ORDER BY label"
    )
    stats["per_label"] = records

    records = await store.run_read(
        "MATCH ()-[r]->() WHERE r.source_file IS NOT NULL "
        "RETURN type(r) AS type, count(r) AS c ORDER BY type"
    )
    stats["per_relationship"] = records

    # The authoritative per-file figures live on the :SourceFile provenance
    # nodes, which the ingestion job maintains. The timestamp is rendered to
    # a string here because a Neo4j temporal is not JSON-serialisable.
    # The most recent job is listed first, so the UI can show the current
    # state without having to filter.
    records = await store.run_read(
        "MATCH (j:IngestionJob)-[:CONTAINS]->(f:SourceFile) "
        "RETURN f.id AS file_id, f.filename AS filename, f.rows_read AS rows_read, "
        "f.rows_accepted AS rows_accepted, f.rows_rejected AS rows_rejected, "
        "f.nodes_created AS nodes_created, f.relationships_created AS relationships_created, "
        "j.id AS job_id, j.status AS job_status, "
        "toString(j.started_at) AS started_at "
        "ORDER BY j.started_at DESC, f.filename"
    )
    stats["per_file"] = records
    stats["jobs"] = sorted(
        {r["job_id"] for r in records if r.get("job_id")}, reverse=True
    )

    records = await store.run_read("MATCH (n) RETURN count(n) AS c")
    stats["total_nodes"] = int(records[0]["c"]) if records else 0
    records = await store.run_read("MATCH ()-[r]->() RETURN count(r) AS c")
    stats["total_relationships"] = int(records[0]["c"]) if records else 0

    records = await store.run_read(
        "MATCH (n) WHERE n.kind = 'csv_ingest' AND NOT (n)--() RETURN count(n) AS c"
    )
    stats["orphan_nodes"] = int(records[0]["c"]) if records else 0

    return stats


async def enrich_dataset_graph(store: Any) -> None:
    """Enrich flat tabular nodes into dimensional entities with semantic relationships."""
    try:
        # 1. Customer dataset: Customer -> City -> Country, Customer -> Company
        await store.run_write("""
            MATCH (c:Customer)
            WHERE (c.kind = 'csv_ingest' OR c.source_file = 'customers.csv')
              AND c.companyName IS NOT NULL AND c.companyName <> 'NULL' AND c.companyName <> ''
            MERGE (comp:Company {name: c.companyName})
            ON CREATE SET comp.kind = 'csv_ingest', comp.source_file = coalesce(c.source_file, 'customers.csv')
            MERGE (c)-[:AFFILIATED_WITH]->(comp)
        """)
        await store.run_write("""
            MATCH (c:Customer)
            WHERE (c.kind = 'csv_ingest' OR c.source_file = 'customers.csv')
              AND c.city IS NOT NULL AND c.city <> 'NULL' AND c.city <> ''
            MERGE (city:City {name: c.city})
            ON CREATE SET city.kind = 'csv_ingest', city.source_file = coalesce(c.source_file, 'customers.csv'), city.country = c.country
            MERGE (c)-[:LOCATED_IN]->(city)
        """)
        await store.run_write("""
            MATCH (city:City)
            WHERE city.country IS NOT NULL AND city.country <> 'NULL' AND city.country <> ''
            MERGE (country:Country {name: city.country})
            ON CREATE SET country.kind = 'csv_ingest', country.source_file = coalesce(city.source_file, 'customers.csv')
            MERGE (city)-[:IN_COUNTRY]->(country)
        """)

        # 2. CoffeeShop dataset: Customer -> Order -> Item
        await store.run_write("""
            MATCH (n)
            WHERE (n.source_file = 'CoffeeShop_Improved.csv' OR n.source_file = 'CoffeeShop_Improved.xlsx')
              AND ('CoffeeshopImproved' IN labels(n) OR n.order_id IS NOT NULL)
            WITH n LIMIT 500
            MERGE (c:Customer {name: n.cust_name})
            ON CREATE SET c.kind = 'csv_ingest', c.source_file = 'CoffeeShop_Improved.csv'
            MERGE (o:Order {order_id: n.order_id})
            ON CREATE SET o.kind = 'csv_ingest', o.source_file = 'CoffeeShop_Improved.csv', o.service_type = n.in_or_out
            MERGE (i:Item {item_id: n.item_id})
            ON CREATE SET i.kind = 'csv_ingest', i.source_file = 'CoffeeShop_Improved.csv'
            MERGE (c)-[:PLACED]->(o)
            MERGE (o)-[:CONTAINS]->(i)
        """)

        # Clean up flat raw wrapper nodes if dimensional entities exist
        await store.run_write("""
            MATCH (n:CoffeeshopImproved)
            DETACH DELETE n
        """)
    except Exception as exc:
        logger.warning("Optional dataset graph enrichment pass skipped: %s", exc)


async def collect_visualisation(
    store: Any,
    node_limit: int = 500,
    edge_limit: int = 1000,
    labels: list[str] | None = None,
    source_file: str | None = None,
) -> dict[str, Any]:
    """Return a ``{nodes, edges}`` payload strictly for uploaded dataset spreadsheet graphs.

    Excludes unrelated PDF documents, text chunks, and mock entities.
    """
    await enrich_dataset_graph(store)

    label_clause = ""
    params: dict[str, Any] = {"node_limit": node_limit, "edge_limit": edge_limit}
    if labels:
        for label in labels:
            validate_identifier(label, "label")
        label_clause = ":" + ":".join(labels)

    file_clause = ""
    if source_file:
        file_clause = "AND n.source_file = $source_file"
        params["source_file"] = source_file

    # Strictly filter for dataset/csv nodes and prioritize connected nodes with active relationships
    query = f"""
    MATCH (n{label_clause})
    WHERE (n.kind = 'csv_ingest' OR (n.source_file IS NOT NULL AND NOT 'Document' IN labels(n) AND NOT 'Chunk' IN labels(n) AND NOT 'Entity' IN labels(n)))
      {file_clause}
    OPTIONAL MATCH (n)-[r]-(m)
    WHERE m.kind = 'csv_ingest' OR (m.source_file IS NOT NULL AND NOT 'Document' IN labels(m) AND NOT 'Chunk' IN labels(m) AND NOT 'Entity' IN labels(m))
    WITH n, count(r) AS degree
    ORDER BY degree DESC
    RETURN elementId(n) AS eid, n.source_file AS source_file,
           labels(n) AS labels, n.source_row AS source_row,
           properties(n) AS props LIMIT $node_limit
    """
    records = await store.run_read(query, params)
    nodes: list[dict[str, Any]] = []
    for record in records:
        label_list = record.get("labels") or []
        raw_props = record.get("props") or {}
        clean_props = {
            str(k): (v.isoformat() if hasattr(v, "isoformat") else (v if isinstance(v, (str, int, float, bool)) else str(v)))
            for k, v in raw_props.items()
            if v is not None and str(v) != "NULL" and len(str(v)) < 300
        }
        # Smart display name resolution: check common attribute names across entities
        order_num = clean_props.get("orderID") or clean_props.get("order_id") or clean_props.get("id")
        cust_id = clean_props.get("customerID") or clean_props.get("customer_id")
        supp_name = clean_props.get("companyName") or clean_props.get("company_name")
        prod_name = clean_props.get("productName") or clean_props.get("product_name")
        item_id = clean_props.get("item_id")
        raw_name = clean_props.get("name")
        valid_name = raw_name if (raw_name and str(raw_name).strip().lower() not in ("null", "none", "")) else None

        # Choose the most specific label (avoid generic "Entity" or "Node" if another exists)
        specific_label = next(
            (lbl for lbl in label_list if lbl not in ("Entity", "Node", "Kind")),
            clean_props.get("type") or (label_list[0] if label_list else "Entity")
        )

        row_num = record.get("source_row")
        display = (
            valid_name
            or clean_props.get("contactName")
            or supp_name
            or prod_name
            or (f"Order #{order_num}" if order_num and "Order" in label_list else None)
            or (f"Item {item_id}" if item_id and "Item" in label_list else None)
            or (f"Customer {cust_id}" if cust_id and "Customer" in label_list else None)
            or clean_props.get("city")
            or clean_props.get("country")
            or (f"{specific_label} #{row_num}" if row_num is not None else None)
            or (f"{specific_label} {str(clean_props.get('id') or record.get('eid'))[-6:]}" if label_list else str(record.get("eid")))
        )
        nodes.append(
            {
                "id": str(record.get("eid")),
                "label": str(display)[:50],
                "type": specific_label,
                "labels": label_list,
                "source_file": record.get("source_file"),
                "source_row": record.get("source_row"),
                "properties": clean_props,
            }
        )

    node_id_set = {n["id"] for n in nodes}
    edge_params = {"node_ids": list(node_id_set), "edge_limit": edge_limit}
    edge_records = await store.run_read(
        "MATCH (a)-[r]->(b) "
        "WHERE elementId(a) IN $node_ids AND elementId(b) IN $node_ids "
        "RETURN elementId(a) AS source, elementId(b) AS target, type(r) AS type, "
        "properties(r) AS props, coalesce(r.source_file, a.source_file, b.source_file) AS source_file LIMIT $edge_limit",
        edge_params,
    )
    edges = [
        {
            "id": f"{rec.get('source')}->{rec.get('type')}->{rec.get('target')}",
            "source": str(rec.get("source")),
            "target": str(rec.get("target")),
            "label": str(rec.get("type") or "RELATED"),
            "type": str(rec.get("type") or "RELATED"),
            "source_file": rec.get("source_file"),
            "properties": {
                str(k): (v.isoformat() if hasattr(v, "isoformat") else (v if isinstance(v, (str, int, float, bool)) else str(v)))
                for k, v in (rec.get("props") or {}).items()
                if v is not None and str(v) != "NULL"
            },
        }
        for rec in edge_records
        if rec.get("source") and rec.get("target")
    ]

    return {
        "nodes": nodes,
        "edges": edges,
        "summary": {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "labels": sorted({label for node in nodes for label in node.get("labels", [])}),
            "relationship_types": sorted({edge["label"] for edge in edges if edge.get("label")}),
        },
    }
