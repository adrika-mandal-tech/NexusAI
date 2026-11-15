"""Knowledge graph pipeline for *document* text.

.. deprecated::
    The CSV-to-graph workflow now lives in
    :mod:`nexusai.application.graph.csv`. The old ``ingest_csv_graph``
    function was removed because it interpolated a relationship type taken
    from a CSV cell into the Cypher text, which allowed Cypher injection
    through the uploaded file. It also created ``:Entity`` nodes with no
    ``document_id``, so the document cleanup path could never delete them.

    The ``neo4j-graphrag`` ``SimpleKGPipeline`` import below was also dead:
    that class now lives in ``neo4j_graphrag.experimental.pipeline.kg_builder``
    and the import at the old path raises ``ImportError`` on the installed
    version, so the text pipeline was permanently disabled.
"""

from __future__ import annotations

from typing import Any

from nexusai.core.logging import get_logger

logger = get_logger(__name__)


def build_graph_from_text(text: str, source_id: str) -> dict[str, Any]:
    """Build a graph from free text.

    Not implemented. The previous implementation depended on
    ``neo4j_graphrag.pipeline.SimpleKGPipeline``, which no longer exists at
    that import path, so it silently did nothing. Rather than keep a
    permanently disabled code path, this now reports the situation plainly.

    Use ``POST /api/v1/graph/csv/upload`` for structured data.
    """
    logger.warning(
        "build_graph_from_text is not implemented; use the CSV-to-graph workflow instead"
    )
    return {
        "status": "not_implemented",
        "source_id": source_id,
        "characters": len(text or ""),
        "message": (
            "Text-based knowledge graph extraction is not implemented. The "
            "CSV-to-graph workflow is available at POST /api/v1/graph/csv/upload."
        ),
    }
