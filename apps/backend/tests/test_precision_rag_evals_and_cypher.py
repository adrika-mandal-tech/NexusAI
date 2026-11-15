"""Verification Tests for Precision Hybrid RAG, PyMuPDF, 10% Chunking, Multi-Embeddings, Live % Eval Metrics, & Neo4j Cypher Console."""

from __future__ import annotations

import uuid
from pathlib import Path

import httpx
import pytest

from nexusai.infrastructure.embedding.provider import ConfigurableEmbeddingService
from nexusai.infrastructure.parsers.chunker import ConfigurableChunker
from nexusai.infrastructure.parsers.file_inspector import (
    FileValidationError,
    get_max_upload_bytes_for_extension,
    validate_and_inspect_file,
)
from nexusai.infrastructure.parsers.pymupdf_parser import PyMuPDFParser
from nexusai.infrastructure.parsers.router_parser import SmartDocumentRouter

REPO_ROOT = Path(__file__).resolve().parents[3]
TEST_FILES_DIR = REPO_ROOT / "Test_Files"


def test_chunker_ten_percent_overlap_and_per_type_limits(tmp_path: Path) -> None:
    chunker = ConfigurableChunker(chunk_size=800)
    assert chunker.chunk_overlap == 80
    assert chunker.overlap_ratio == 0.1

    custom_chunker = ConfigurableChunker(chunk_size=1000)
    assert custom_chunker.chunk_overlap == 100
    assert custom_chunker.overlap_ratio == 0.1

    assert get_max_upload_bytes_for_extension(".pdf") == 100 * 1024 * 1024
    assert get_max_upload_bytes_for_extension(".docx") == 50 * 1024 * 1024
    assert get_max_upload_bytes_for_extension(".png") == 25 * 1024 * 1024

    oversized_txt = tmp_path / "too_big.txt"
    oversized_txt.write_bytes(b"A" * 2048)
    with pytest.raises(FileValidationError, match="exceeds maximum allowed size"):
        validate_and_inspect_file(oversized_txt, max_bytes=1024)


@pytest.mark.asyncio
async def test_pymupdf_parser_and_router_integration(tmp_path: Path) -> None:
    cuda_pdf = TEST_FILES_DIR / "CUDA_C_Best_Practices_Guide.pdf"
    if not cuda_pdf.exists():
        pytest.skip("Sample CUDA PDF not present")

    from pypdf import PdfReader, PdfWriter

    reader = PdfReader(str(cuda_pdf))
    writer = PdfWriter()
    for idx in (15, 16):
        if idx < len(reader.pages):
            writer.add_page(reader.pages[idx])
    sample_pdf = tmp_path / "pymupdf_sample.pdf"
    with open(sample_pdf, "wb") as fh:
        writer.write(fh)

    report = validate_and_inspect_file(sample_pdf, prefer_pymupdf=True)
    assert report.recommended_parser == "pymupdf"

    parser = PyMuPDFParser()
    doc_id = uuid.uuid4()
    parsed = await parser.parse(str(sample_pdf), doc_id, sample_pdf.name, "pdf")
    assert parsed.metadata["parser_used"] == "pymupdf"
    assert len(parsed.chunks) >= 2
    assert all(c.metadata.get("chunk_overlap_ratio") == 0.1 for c in parsed.chunks)

    router = SmartDocumentRouter()
    routed_doc = await router.parse(str(sample_pdf), doc_id, sample_pdf.name, "pdf")
    assert len(routed_doc.chunks) >= 2


def test_huggingface_and_openlm_embedding_providers() -> None:
    hf_bge = ConfigurableEmbeddingService(provider="huggingface", model_name="BAAI/bge-m3")
    assert hf_bge.get_dimension() == 1024
    vec_1024 = hf_bge.embed_query("CUDA coalesced global memory bandwidth")
    assert len(vec_1024) == 1024

    openlm_minilm = ConfigurableEmbeddingService(
        provider="openlm",
        model_name="all-MiniLM-L6-v2",
        openlm_url="",
    )
    assert openlm_minilm.model_name == "sentence-transformers/all-MiniLM-L6-v2"
    assert openlm_minilm.get_dimension() == 384
    vec_384 = openlm_minilm.embed_query("Neo4j knowledge graph Cypher traversal")
    assert len(vec_384) == 384


@pytest.mark.asyncio
async def test_live_eval_metrics_pct_document_scoped_overview_and_cypher_console(
    client: httpx.AsyncClient,
) -> None:
    md_payload = (
        b"# Quantum Tensor Engine Specification\n\n"
        b"## Section 1: Architecture Overview\n"
        b"The Quantum Tensor Engine (QTE-9) achieves 94.5% memory bandwidth efficiency using "
        b"BGE-M3 dense embeddings stored in Qdrant and entity graphs persisted in Neo4j.\n\n"
        b"## Section 2: Latency & Throughput\n"
        b"Asynchronous DMA transfers via `cudaMemcpyAsync` reduce p99 kernel dispatch latency to 1.8 ms.\n"
    )
    up_res = await client.post(
        "/api/v1/documents/upload",
        data={"wait_inline": "true", "force_reindex": "true"},
        files={"file": ("qte9_spec.md", md_payload, "text/markdown")},
    )
    assert up_res.status_code == 201
    doc_id = up_res.json()["document_id"]

    try:
        # 1. Document-scoped content analysis / overview query + 7 live % eval metrics
        chat_res = await client.post(
            "/api/v1/chat",
            json={
                "message": "Analyze and summarize this document, including QTE-9 bandwidth efficiency and p99 latency.",
                "document_id": doc_id,
            },
        )
        assert chat_res.status_code == 200
        chat_data = chat_res.json()
        assert "[KB-" in chat_data["content"]
        # Ensure zero cross-document pollution from unrelated files
        assert all(c["filename"] == "qte9_spec.md" for c in chat_data["citations"])

        eval_m = chat_data["diagnostics"]["eval_metrics"]
        for metric_key in (
            "hallucination_rate_pct",
            "correctness_pct",
            "accuracy_pct",
            "context_precision_pct",
            "context_recall_pct",
            "tool_calling_accuracy_pct",
            "task_success_rate_pct",
        ):
            assert metric_key in eval_m
            assert 0.0 <= float(eval_m[metric_key]) <= 100.0

        assert eval_m["frameworks"] == ["LangSmith", "Ragas", "DeepEval"]
        assert len(eval_m["tools_invoked"]) >= 2

        # 2. Out-of-domain abstention must report 0.0% hallucination
        abstain_res = await client.post(
            "/api/v1/chat",
            json={
                "message": "What was the exact 2198 Jovian atmospheric helium-3 submarine pressure?",
                "document_id": doc_id,
            },
        )
        assert abstain_res.status_code == 200
        abstain_eval = abstain_res.json()["diagnostics"]["eval_metrics"]
        assert abstain_eval["hallucination_rate_pct"] == 0.0

        # 3. Neo4j Cypher Console: valid read-only queries succeed with nodes/rows
        cypher_ok = await client.post(
            "/api/v1/knowledge-graph/cypher",
            json={
                "cypher": "MATCH (d:Document {id: $doc_id}) RETURN d, d.doc_type AS DocType LIMIT 5",
                "params": {"doc_id": doc_id},
            },
        )
        assert cypher_ok.status_code == 200
        c_json = cypher_ok.json()
        assert c_json["status"] == "ok"
        assert c_json["query_type"] == "r"
        assert c_json["row_count"] >= 1
        assert len(c_json["nodes"]) >= 1
        assert c_json["rows"][0]["DocType"] in ("MD", "MARKDOWN", "TXT")

        # 4. Neo4j Cypher Console: mutating / multi-statement queries are strictly rejected (400)
        for bad_cypher in (
            "CREATE (x:Hacker {name: 'pwn'}) RETURN x",
            "MATCH (n) DETACH DELETE n",
            "MATCH (n) SET n.hacked = true RETURN n",
            "MATCH (n) RETURN count(n); MATCH (m) DELETE m",
        ):
            bad_res = await client.post(
                "/api/v1/knowledge-graph/cypher",
                json={"cypher": bad_cypher},
            )
            assert bad_res.status_code == 400
    finally:
        await client.delete(f"/api/v1/documents/{doc_id}")

