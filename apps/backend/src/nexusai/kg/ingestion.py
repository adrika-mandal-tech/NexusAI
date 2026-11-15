"""Unified ingestion engine for the knowledge graph pipeline.

This module is the heart of the "any dataset -> knowledge graph" feature.
It takes a Datasource that carries a multipart file upload (PDF, DOCX,
CSV, JSON, TXT, MD, XLSX, HTML), a URL, or a GitHub repository URL. It
runs the full pipeline: fetch -> parse -> chunk -> extract -> ingest
into Neo4j.

Usage:
    from nexusai.kg.types import Datasource, SourceType
    from nexusai.kg.ingestion import KnowledgeGraphIngestionEngine

    engine = KnowledgeGraphIngestionEngine(neo4j_store=my_store, llm=my_llm)
    source = Datasource(
        title="Example Corpus",
        source_type=SourceType.CSV,
        file_bytes=open("customers.csv", "rb").read(),
        file_name="customers.csv",
        chunk_size=1000,
    )
    result = await engine.ingest(source)
"""

from __future__ import annotations

import tempfile
from datetime import datetime
from typing import Any
from uuid import UUID

from nexusai.kg.visualizers.neo4j_visualizer import Neo4jGraphVisualizer

from nexusai.core.config import get_settings
from nexusai.core.logging import get_logger
from nexusai.kg.chunkers import ChunkStrategy
from nexusai.kg.extractors.llm_extractor import LLMEntityExtractor
from nexusai.kg.loaders.loader_utils import _normalize_entity, _normalize_relationship
from nexusai.kg.types import (
    Chunk,
    Datasource,
    Entity,
    ExtractionResult,
    Relationship,
    UploadResult,
)

from ...infrastructure.graph_stores.neo4j_store import Neo4jGraphStore
from ...infrastructure.llm.gemini_client import GeminiLLMClient
from ...infrastructure.llm.groq_client import GroqLLMClient
from ...infrastructure.parsers import DocumentParserPort
from ...infrastructure.parsers.docling_parser import DoclingDocumentParser

logger = get_logger(__name__)


class KnowledgeGraphIngestionEngine:
    """Orchestrate the full knowledge-graph creation pipeline."""

    def __init__(
        self,
        parser: DocumentParserPort | None = None,
        graph_store: Neo4jGraphStore | None = None,
        llm: Any | None = None,
        chunk_strategy: str = ChunkStrategy.SENTENCE.value,
        use_llm_extraction: bool = True,
        llm_provider: str | None = None,
        llm_model: str | None = None,
        top_k_entities: int = 20,
        top_k_relationships: int = 15,
    ) -> None:
        self._parser = parser or DoclingDocumentParser()
        self._graph_store = graph_store or Neo4jGraphStore()
        self._chunk_strategy = chunk_strategy
        self._use_llm_extraction = use_llm_extraction
        self._llm_provider = llm_provider or get_settings().LLM_PROVIDER
        self._llm_model = llm_model or get_settings().DEFAULT_GEMINI_MODEL
        self._top_k_entities = top_k_entities
        self._top_k_relationships = top_k_relationships

        from nexusai.kg.extractors.llm_extractor import ExtractionConfig
        config = ExtractionConfig(
            use_llm=use_llm_extraction,
            llm_provider=llm_provider,
            llm_model=llm_model,
            top_k_entities=top_k_entities,
            top_k_relationships=top_k_relationships,
        )
        self._extractor = LLMEntityExtractor(llm=self._llm_client(), config=config)
        self._visualizer = Neo4jGraphVisualizer()

    def _llm_client(self) -> Any:
        """Return the configured LLM client."""
        if self._llm_provider == "groq":
            return GroqLLMClient()
        return GeminiLLMClient()


    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def ingest(self, datasource: Datasource) -> UploadResult:
        """Run the full ingestion pipeline for a single datasource."""
        logger.info("Starting ingestion for %s (type=%s)", datasource.title, datasource.source_type)

        try:
            parsed_doc = await self._fetch_and_parse(datasource)
            chunks = await self._chunk_document(parsed_doc, datasource)
            extraction = await self._extract_entities(chunks, datasource)
        except Exception as exc:
            logger.error("Ingestion pipeline failed: %s", exc)
            return UploadResult(
                source_id=datasource.id,
                source_type=datasource.source_type.value,
                title=datasource.title,
                chunk_count=0,
                entity_count=0,
                relationship_count=0,
                node_count=0,
                edge_count=0,
                status="failed",
                error_message=str(exc),
            )

        entities, relationships = self._build_entity_relationship_lists(chunks, extraction)

        try:
            await self._persist_to_neo4j(datasource, chunks, entities, relationships)
            node_count, edge_count = self._compute_graph_counts(chunks, entities, relationships)
        except Exception as exc:
            logger.error("Neo4j persistence failed: %s", exc)
            return UploadResult(
                source_id=datasource.id,
                source_type=datasource.source_type.value,
                title=datasource.title,
                chunk_count=len(chunks),
                entity_count=len(entities),
                relationship_count=len(relationships),
                node_count=0,
                edge_count=0,
                status="failed",
                error_message=str(exc),
            )

        visualization = None
        try:
            viz = await self._visualizer.get_graph_visualization(datasource.id)
            visualization = viz
        except Exception:
            pass

        return UploadResult(
            source_id=datasource.id,
            source_type=datasource.source_type.value,
            title=datasource.title,
            chunk_count=len(chunks),
            entity_count=len(entities),
            relationship_count=len(relationships),
            node_count=node_count,
            edge_count=edge_count,
            status="completed",
            created_at=datetime.utcnow(),
            graph_url=visualization.get("thumbnail", "") if visualization else None,
        )


    # ------------------------------------------------------------------
    # Pipeline steps
    # ------------------------------------------------------------------

    async def _fetch_and_parse(self, datasource: Datasource) -> Any:
        """Fetch raw text/content and parse into chunks with metadata."""
        if datasource.file_bytes is not None:
            content = datasource.file_bytes
            filename = datasource.file_name or datasource.title
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as tmp:
                    tmp.write(content)
                    tmp_path = tmp.name
                return await self._parser.parse(
                    file_path=tmp_path,
                    document_id=datasource.id,
                    filename=filename,
                    source_type=datasource.source_type.value,
                )
            finally:
                import os
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        if datasource.url:
            import httpx
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(datasource.url)
                resp.raise_for_status()
                from nexusai.kg.parsers import fetch_document
                parsed = await fetch_document(
                    content=resp.text,
                    mime_type=resp.headers.get("content-type", "text/plain"),
                    source_id=datasource.id,
                    filename=datasource.title,
                    source_type=datasource.source_type.value,
                )
                return parsed

        if datasource.github_repo:
            from nexusai.kg.parsers import fetch_github_repository
            parsed = await fetch_github_repository(
                repo_url=datasource.github_repo,
                path=datasource.github_path,
                depth=datasource.depth,
                source_id=datasource.id,
                filename=datasource.title,
                source_type=datasource.source_type.value,
            )
            return parsed

        raise ValueError("Datasource must provide file_bytes, url, or github_repo")

    async def _chunk_document(self, parsed_doc: Any, datasource: Datasource) -> list[Chunk]:
        """Chunk parsed text using the configured strategy."""
        text = self._parser.extract_text(parsed_doc)
        strategy = ChunkStrategy(self._chunk_strategy)
        return self._parser.chunk_text(
            text,
            strategy=strategy,
            chunk_size=datasource.chunk_size,
            chunk_overlap=datasource.chunk_overlap,
        )

    async def _extract_entities(self, chunks: list[Chunk], datasource: Datasource) -> list[ExtractionResult]:
        """Run LLM extraction over all chunks."""
        results = []
        for chunk in chunks:
            res = self._extractor.extract(
                chunk_text=chunk.text,
                document_id=str(datasource.id),
                chunk_id=str(chunk.id),
            )
            results.append(ExtractionResult(chunk_id=chunk.id, **res))
        return results


    def _build_entity_relationship_lists(self, chunks: list[Chunk], extraction_results: list[ExtractionResult]) -> tuple[list[Entity], list[Relationship]]:
        entities: list[Entity] = []
        relationships: list[Relationship] = []

        for extraction in extraction_results:
            chunk = next((c for c in chunks if c.id == extraction.chunk_id), None)
            if not chunk:
                continue
            for ent in extraction.entities:
                entity_id = f"{ent.entity_type}:{ent.name}"
                eid = f"{chunk.document_id}:{entity_id}"
                entities.append(
                    Entity(
                        id=eid,
                        name=ent.name,
                        entity_type=ent.entity_type,
                        chunk_id=chunk.id,
                        document_id=chunk.document_id,
                        description=ent.description,
                        confidence=ent.confidence,
                    )
                )
            for rel in extraction.relationships:
                relationships.append(
                    Relationship(
                        id=f"{rel.source_id}|{rel.target_id}",
                        source_id=rel.source_id,
                        target_id=rel.target_id,
                        relationship_type=rel.relationship_type,
                        chunk_id=chunk.id,
                        document_id=chunk.document_id,
                        description=rel.description,
                        weight=rel.weight,
                        confidence=rel.confidence,
                    )
                )
        return entities, relationships


    async def _persist_to_neo4j(
        self,
        datasource: Datasource,
        chunks: list[Chunk],
        entities: list[Entity],
        relationships: list[Relationship],
    ) -> None:
        """Write all graph artifacts into Neo4j."""
        # Document node
        await self._graph_store.create_document_node(
            document_id=datasource.id,
            filename=datasource.file_name or datasource.title,
            source_type=datasource.source_type.value,
            user_id=UUID("00000000-0000-0000-0000-000000000000"),
        )

        # Chunk nodes
        if chunks:
            chunk_dicts = [
                {
                    "chunk_id": str(c.id),
                    "document_id": str(c.document_id),
                    "index": c.index,
                    "text": c.text,
                    "page_number": c.page_number,
                    "section_title": c.section_title,
                    "filename": c.source_file or datasource.file_name or datasource.title,
                    "metadata": c.metadata,
                }
                for c in chunks
            ]
            await self._graph_store.create_chunk_nodes(chunk_dicts)

        # Entities & relationships
        if entities or relationships:
            for chunk in chunks:
                chunk_entities = [e for e in entities if e.chunk_id == chunk.id]
                chunk_relationships = [r for r in relationships if r.chunk_id == chunk.id]
                entity_name_to_id = {e.name: e.id for e in chunk_entities}
                ent_norm = [_normalize_entity(e) for e in chunk_entities]
                rel_norm = [
                    norm
                    for r in chunk_relationships
                    if (norm := _normalize_relationship(r, entity_name_to_id)) is not None
                ]
                if ent_norm or rel_norm:
                    await self._graph_store.create_entities_and_relationships(
                        chunk_id=chunk.id,
                        document_id=datasource.id,
                        entities=ent_norm,
                        relationships=rel_norm,
                    )

    @staticmethod
    def _compute_graph_counts(chunks: list[Chunk], entities: list[Entity], relationships: list[Relationship]) -> tuple[int, int]:
        node_count = len(chunks) + 1
        edge_count = len(entities) + len(relationships)
        return node_count, edge_count
