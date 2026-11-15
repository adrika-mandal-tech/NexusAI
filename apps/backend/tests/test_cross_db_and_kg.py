"""Phase 6 Verification Tests — Cross-DB Shared Identity, Rollback Reconciliation, and Northwind CSV KG."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from nexusai.application.graph.csv.service import CsvGraphService
from nexusai.application.use_cases.ingest_document import IngestDocumentUseCase
from nexusai.domain.entities.knowledge import IngestionSource, SourceStatus, SourceType
from nexusai.infrastructure.csv.csv_loader import store_csv
from nexusai.infrastructure.db.repositories import ResilientKnowledgeRepository
from nexusai.infrastructure.db.session import init_db
from nexusai.infrastructure.embedding.provider import ConfigurableEmbeddingService
from nexusai.infrastructure.graph_stores.neo4j_store import Neo4jGraphStore
from nexusai.infrastructure.parsers.router_parser import SmartDocumentRouter
from nexusai.infrastructure.vector_stores.qdrant_store import QdrantVectorStore

REPO_ROOT = Path(__file__).resolve().parents[3]
TEST_FILES_DIR = REPO_ROOT / "Test_Files"


@pytest.mark.asyncio
async def test_document_ingestion_maintains_shared_ids_across_postgres_qdrant_and_neo4j(tmp_path: Path) -> None:
    await init_db()
    repo = ResilientKnowledgeRepository()
    embedder = ConfigurableEmbeddingService(
        provider="sentence-transformers",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    )
    vstore = QdrantVectorStore(
        collection_name="nexusai_pytest_phase6",
        embedding_model=embedder.model_name,
        expected_dimension=384,
        auto_scope_by_model=True,
    )
    gstore = Neo4jGraphStore()
    router = SmartDocumentRouter()

    md_file = tmp_path / "hybrid_rag_spec.md"
    md_file.write_text(
        "# NexusAI Architecture\n\n"
        "FastAPI uses LangChain to orchestrate Qdrant vector search and Neo4j graph traversal.\n\n"
        "## OCR Pipeline\n\n"
        "PaddleOCR and Docling extract tables from scanned documents before BGE-M3 embedding generation.\n",
        encoding="utf-8",
    )

    user_id = uuid.UUID("00000000-0000-0000-0000-000000000000")
    source_id = uuid.uuid4()
    source = await repo.create_source(
        IngestionSource(
            id=source_id,
            user_id=user_id,
            title=md_file.name,
            source_type=SourceType.MD,
            status=SourceStatus.PENDING,
            file_path=str(md_file),
        )
    )

    use_case = IngestDocumentUseCase(
        parser=router,
        embedder=embedder,
        vector_store=vstore,
        graph_store=gstore,
        llm=None,
        knowledge_repo=repo,
    )

    try:
        meta = await use_case.execute(source.id, str(md_file), user_id)
        assert meta["consistency_verified"] is True
        assert meta["qdrant_indexed"] is True
        assert meta["neo4j_indexed"] is True
        assert meta["entity_count"] >= 3

        updated = await repo.get_source(source_id)
        assert updated is not None
        assert updated.status == SourceStatus.READY
        assert updated.chunk_count >= 1

        # Verify Qdrant chunks and Neo4j chunks share identical chunk_id and entity_ids
        qdrant_chunks = await vstore.get_document_chunks(source_id)
        assert len(qdrant_chunks) == updated.chunk_count
        qdrant_chunk_ids = [uuid.UUID(c["chunk_id"]) for c in qdrant_chunks]

        neo_counts = await gstore.verify_document_chunks(source_id)
        assert neo_counts["chunk_count"] == len(qdrant_chunks)
        assert neo_counts["entity_count"] >= 3

        chunk_entities = await gstore.get_chunk_entities(qdrant_chunk_ids)
        assert len(chunk_entities) >= 3
        neo_entity_ids = {rec["entity_id"] for rec in chunk_entities}
        qdrant_entity_ids = {eid for c in qdrant_chunks for eid in c.get("entity_ids", [])}
        assert neo_entity_ids == qdrant_entity_ids

        # Graph search finds entities and linked chunks
        graph_ctx = await gstore.search_graph_context(["Qdrant", "Neo4j"], document_ids=[str(source_id)])
        assert len(graph_ctx["chunks"]) >= 1
        assert len(graph_ctx["entities"]) >= 2
    finally:
        await vstore.delete_document_chunks(source_id)
        await gstore.delete_document_graph(source_id)
        await repo.delete_source(source_id)
        try:
            vstore._client.delete_collection(vstore.collection_name)
        except Exception:
            pass


@pytest.mark.asyncio
async def test_partial_neo4j_failure_rolls_back_qdrant_and_marks_source_failed(tmp_path: Path) -> None:
    await init_db()
    repo = ResilientKnowledgeRepository()
    embedder = ConfigurableEmbeddingService(
        provider="sentence-transformers",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    )
    vstore = QdrantVectorStore(
        collection_name="nexusai_pytest_rollback",
        embedding_model=embedder.model_name,
        expected_dimension=384,
        auto_scope_by_model=True,
    )
    failing_graph = AsyncMock(spec=Neo4jGraphStore)
    failing_graph.delete_document_graph.return_value = None
    failing_graph.create_document_node.side_effect = RuntimeError("Simulated Neo4j write failure")

    txt_file = tmp_path / "rollback_test.txt"
    txt_file.write_text("Qdrant points must roll back if Neo4j indexing fails.", encoding="utf-8")

    user_id = uuid.UUID("00000000-0000-0000-0000-000000000000")
    source_id = uuid.uuid4()
    await repo.create_source(
        IngestionSource(
            id=source_id,
            user_id=user_id,
            title=txt_file.name,
            source_type=SourceType.TXT,
            status=SourceStatus.PENDING,
            file_path=str(txt_file),
        )
    )

    use_case = IngestDocumentUseCase(
        parser=SmartDocumentRouter(),
        embedder=embedder,
        vector_store=vstore,
        graph_store=failing_graph,
        llm=None,
        knowledge_repo=repo,
    )

    try:
        with pytest.raises(RuntimeError, match="Simulated Neo4j write failure"):
            await use_case.execute(source_id, str(txt_file), user_id)

        failed_src = await repo.get_source(source_id)
        assert failed_src is not None
        assert failed_src.status == SourceStatus.FAILED
        assert failed_src.metadata.get("qdrant_indexed") is False
        assert "rolled_back_qdrant_chunks" in failed_src.metadata.get("reconciliation_actions", [])

        # Confirm zero orphaned chunks remain in Qdrant
        orphaned = await vstore.get_document_chunks(source_id)
        assert orphaned == []
    finally:
        await repo.delete_source(source_id)
        try:
            vstore._client.delete_collection(vstore.collection_name)
        except Exception:
            pass


@pytest.mark.asyncio
async def test_real_northwind_csvs_schema_inference_neo4j_ingest_and_graph_search() -> None:
    csv_names = ["customers.csv", "orders.csv", "products.csv", "suppliers.csv"]
    paths = [TEST_FILES_DIR / name for name in csv_names]
    if not all(p.exists() for p in paths):
        pytest.skip("Northwind CSV files missing in Test_Files")

    gstore = Neo4jGraphStore()
    service = CsvGraphService(gstore)
    batch_source_id = uuid.uuid4()

    stored = [
        store_csv(
            content=p.read_bytes(),
            filename=p.name,
            source_id=batch_source_id,
            content_type="text/csv",
        )
        for p in paths
    ]

    proposal = await service.create_proposal(stored, include_pii=False, created_by="pytest")
    labels = {n.label for n in proposal.nodes}
    assert {"Customer", "Order", "Product", "Supplier"}.issubset(labels)
    assert len(proposal.relationships) >= 2

    approved = await service.approve_proposal(proposal.proposal_id, approved_by="pytest")
    files_map: dict[str, tuple[Path, str, str]] = {
        s.filename: (s.path, s.encoding, s.delimiter) for s in stored
    }
    report = await service.ingest(approved, files_map, dry_run=False)
    total_nodes = sum(f.nodes_created + f.nodes_matched for f in report.files)
    total_rels = sum(f.relationships_created + f.relationships_matched for f in report.files)
    assert total_nodes > 50
    assert total_rels > 20

    # Verify unified graph search retrieves CSV domain nodes with source_file & source_row lineage
    csv_ctx: dict[str, Any] = await gstore.search_graph_context(["Chai", "Customer"])
    assert len(csv_ctx["chunks"]) >= 1
    assert any(c["retrieval_channel"] == "graph_csv" for c in csv_ctx["chunks"])
