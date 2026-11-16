"""Exhaustive Backend API & Hybrid RAG Pipeline Verification Suite.

Verifies every FastAPI router (/health, /api/v1/*, /api/* alias, auth, documents,
knowledge, search/*, chat, chat/stream SSE, conversations, knowledge-graph/*,
graph/csv/*, settings, models, observability, evaluation) plus all 7 Single-Agent
LangChain tools and RAG validation/fusion/reranking/grounding edge cases.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest
from httpx import AsyncClient

from nexusai.core.container import get_assistant, get_vector_store

REPO_ROOT = Path(__file__).resolve().parents[3]
TEST_FILES_DIR = REPO_ROOT / "Test_Files"


@pytest.mark.asyncio
async def test_all_health_observability_models_and_alias_endpoints(client: AsyncClient) -> None:
    """Verify root /health, /api/v1/health, /api/v1/health/deep, /api/health/deep alias, /models, and /observability/summary."""
    r_root = await client.get("/health")
    assert r_root.status_code == 200
    assert r_root.json()["status"] == "ok"
    assert "X-Request-ID" in r_root.headers

    r_v1 = await client.get("/api/v1/health")
    assert r_v1.status_code == 200
    assert r_v1.json()["status"] == "ok"

    r_deep_v1 = await client.get("/api/v1/health/dependencies")
    assert r_deep_v1.status_code == 200
    deep = r_deep_v1.json()
    assert deep["dependencies"]["qdrant"]["status"] == "ok"
    assert deep["dependencies"]["neo4j"]["status"] == "ok"
    assert deep["dependencies"]["database"]["status"] == "healthy"
    assert deep["dependencies"]["embeddings"]["dimension"] == 1024

    # Verify /health/deep alias and /api/* alias parity for frontend compatibility
    r_deep_alias = await client.get("/api/health/deep")
    assert r_deep_alias.status_code == 200
    assert r_deep_alias.json()["dependencies"]["qdrant"]["status"] == "ok"

    r_models = await client.get("/api/v1/knowledge/models")
    assert r_models.status_code == 200
    models_data = r_models.json()
    assert "providers" in models_data
    assert "gemini" in models_data["providers"]
    assert "openlm" in models_data["providers"]

    r_obs = await client.get("/api/v1/observability/summary")
    assert r_obs.status_code == 200
    obs = r_obs.json()
    assert "langsmith" in obs
    assert "total_requests" in obs


@pytest.mark.asyncio
async def test_auth_register_login_and_me_endpoints(client: AsyncClient) -> None:
    """Verify /api/v1/auth/register, /login, /me, duplicate registration (409), and invalid token (401)."""
    unique_email = f"qa-{uuid.uuid4().hex[:8]}@nexusai.local"
    reg = await client.post(
        "/api/v1/auth/register",
        json={"email": unique_email, "password": "StrongPassword!123", "name": "QA Engineer"},
    )
    assert reg.status_code == 200
    token = reg.json()["access_token"]
    assert token

    # Duplicate email returns 409
    dup = await client.post(
        "/api/v1/auth/register",
        json={"email": unique_email, "password": "StrongPassword!123", "name": "QA Engineer"},
    )
    assert dup.status_code == 409

    # Login succeeds
    login = await client.post(
        "/api/v1/auth/login",
        json={"email": unique_email, "password": "StrongPassword!123"},
    )
    assert login.status_code == 200
    assert login.json()["user"]["email"] == unique_email

    # Wrong password returns 401
    bad_login = await client.post(
        "/api/v1/auth/login",
        json={"email": unique_email, "password": "wrong"},
    )
    assert bad_login.status_code == 401

    # /me with bearer token succeeds
    me = await client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200
    assert me.json()["email"] == unique_email

    # /me with invalid token returns 401
    bad_me = await client.get("/api/v1/auth/me", headers={"Authorization": "Bearer invalid-token"})
    assert bad_me.status_code == 401


@pytest.mark.asyncio
async def test_document_upload_validation_errors_empty_corrupt_spoofed_unsupported(client: AsyncClient) -> None:
    """Verify strict upload validation: empty (400), corrupted PDF (422), corrupted DOCX (422), spoofed EXE (422), unsupported ext (422)."""
    # 1. Empty 0-byte file -> 400
    r_empty = await client.post(
        "/api/v1/documents/upload",
        files={"file": ("empty.txt", b"", "text/plain")},
    )
    assert r_empty.status_code == 400

    # 2. Missing file -> 400
    r_none = await client.post("/api/v1/documents/upload")
    assert r_none.status_code == 400

    # 3. Corrupted PDF -> 422
    r_bad_pdf = await client.post(
        "/api/v1/documents/upload",
        files={"file": ("broken.pdf", b"%PDF-1.7\nGARBAGE_STREAM_NO_EOF", "application/pdf")},
    )
    assert r_bad_pdf.status_code == 422

    # 4. Corrupted DOCX ZIP -> 422
    r_bad_docx = await client.post(
        "/api/v1/documents/upload",
        files={
            "file": (
                "broken.docx",
                b"PK\x03\x04CORRUPTED_OOXML_ARCHIVE_DATA",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        },
    )
    assert r_bad_docx.status_code == 422

    # 5. Spoofed Windows PE executable disguised as PDF -> 422
    r_spoof = await client.post(
        "/api/v1/documents/upload",
        files={"file": ("malware.pdf", b"MZ\x90\x00\x03\x00\x00\x00PE_HEADER", "application/pdf")},
    )
    assert r_spoof.status_code == 422

    # 6. Unsupported binary extension -> 422
    r_unsup = await client.post(
        "/api/v1/documents/upload",
        files={"file": ("archive.tar.gz", b"\x1f\x8b\x08\x00binarypayload", "application/gzip")},
    )
    assert r_unsup.status_code == 422


@pytest.mark.asyncio
async def test_multiformat_ingestion_dedup_reindex_all_search_modes_tools_and_chat(client: AsyncClient) -> None:
    """End-to-end verification of TXT, MD, CSV, DOCX ingestion, deduplication, reindex, all search modes, all 7 agent tools, multi-turn chat, SSE stream, and clean deletion."""
    created_doc_ids: list[str] = []
    nonce = uuid.uuid4().hex[:8]

    try:
        # 1. Upload Markdown technical spec (wait_inline=True)
        md_bytes = (
            f"# NexusAI Architecture Specification ({nonce})\n\n"
            "## Memory Coalescing & ECC Protection\n"
            "Global memory loads on CUDA compute capability devices achieve peak bandwidth when 32 threads in a warp "
            "access contiguous 128-byte aligned cache lines. Error-Correcting Code (ECC) memory detects and corrects "
            "single-bit DRAM faults while incurring roughly a 12.5% capacity overhead.\n\n"
            "## Hybrid RRF Retrieval\n"
            "Reciprocal Rank Fusion combines dense BGE-M3 cosine rankings with sparse Okapi BM25 lexical rankings "
            "using constant k=60 so exact technical identifiers and semantic paraphrases both surface in top-k.\n"
        ).encode("utf-8")

        up_md = await client.post(
            "/api/v1/documents/upload",
            files={"file": (f"arch_spec_{nonce}.md", md_bytes, "text/markdown")},
            data={"wait_inline": "true", "force_reindex": "true"},
        )
        assert up_md.status_code == 201, up_md.text
        md_doc = up_md.json()
        assert md_doc["status"] == "ready"
        assert md_doc["chunks"] >= 1
        md_id = md_doc["source_id"]
        created_doc_ids.append(md_id)

        # 2. Idempotent duplicate upload detection (same SHA-256 bytes, force_reindex=False)
        up_dup = await client.post(
            "/api/v1/documents/upload",
            files={"file": (f"arch_spec_{nonce}.md", md_bytes, "text/markdown")},
            data={"wait_inline": "true", "force_reindex": "false"},
        )
        assert up_dup.status_code == 201
        assert up_dup.json()["duplicate"] is True
        assert up_dup.json()["source_id"] == md_id

        # 3. Batch upload CSV + real DOCX (Test_Files/report.docx if present, else TXT)
        csv_bytes = (
            "component_id,subsystem,latency_budget_ms,owner\n"
            f"CMP-{nonce}-01,QdrantVectorEngine,25,VectorTeam\n"
            f"CMP-{nonce}-02,Neo4jGraphEngine,40,GraphTeam\n"
        ).encode("utf-8")

        batch_files: list[tuple[str, tuple[str, bytes, str]]] = [
            ("files", (f"latency_budget_{nonce}.csv", csv_bytes, "text/csv")),
        ]
        docx_path = TEST_FILES_DIR / "report.docx"
        if docx_path.exists():
            batch_files.append(
                (
                    "files",
                    (
                        "report.docx",
                        docx_path.read_bytes(),
                        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    ),
                )
            )

        up_batch = await client.post(
            "/api/v1/knowledge/upload/batch",
            files=batch_files,
            data={"wait_inline": "true", "force_reindex": "true"},
        )
        assert up_batch.status_code == 201, up_batch.text
        batch_json = up_batch.json()
        assert len(batch_json["uploaded"]) == len(batch_files)
        assert batch_json["errors"] == []
        for item in batch_json["uploaded"]:
            assert item["status"] == "ready"
            created_doc_ids.append(item["source_id"])

        # 4. Document listing, detail, and status endpoints
        r_list = await client.get("/api/v1/documents")
        assert r_list.status_code == 200
        listed_ids = {d["id"] for d in r_list.json()}
        assert md_id in listed_ids

        r_detail = await client.get(f"/api/v1/documents/{md_id}")
        assert r_detail.status_code == 200
        assert r_detail.json()["status"] == "ready"

        r_status = await client.get(f"/api/v1/documents/{md_id}/status")
        assert r_status.status_code == 200
        assert r_status.json()["chunk_count"] >= 1

        # 5. Re-index endpoint (/api/v1/documents/{id}/reindex)
        r_reindex = await client.post(f"/api/v1/documents/{md_id}/reindex?wait_inline=true")
        assert r_reindex.status_code == 200
        assert r_reindex.json()["status"] == "ready"

        # Verify Qdrant chunk count for md_id has no uncontrolled duplicates after re-index
        chunks_in_qdrant = await get_vector_store().get_document_chunks(md_id, limit=50)
        chunk_ids = [c["chunk_id"] for c in chunks_in_qdrant]
        assert len(chunk_ids) == len(set(chunk_ids))
        assert len(chunk_ids) >= 1

        # 6. Test ALL Search Endpoints & Fusion Strategies
        # 6a. Empty query validation -> 400
        for ep in ("/api/v1/search/vector", "/api/v1/search/sparse", "/api/v1/search/graph", "/api/v1/search/hybrid", "/api/v1/search/web"):
            bad = await client.post(ep, json={"query": "   "})
            assert bad.status_code == 400

        # 6b. Vector search
        s_vec = await client.post(
            "/api/v1/search/vector",
            json={"query": "128-byte aligned cache lines warp memory coalescing", "top_k": 5, "document_id": md_id},
        )
        assert s_vec.status_code == 200
        assert len(s_vec.json()["results"]) >= 1
        assert s_vec.json()["results"][0]["document_id"] == md_id

        # 6c. Sparse search (exact identifier lookup on CSV!)
        csv_doc_id = batch_json["uploaded"][0]["source_id"]
        s_sp = await client.post(
            "/api/v1/search/sparse",
            json={"query": f"CMP-{nonce}-01 QdrantVectorEngine", "top_k": 5, "document_id": csv_doc_id},
        )
        assert s_sp.status_code == 200
        assert len(s_sp.json()["results"]) >= 1

        # 6d. Graph search
        s_gr = await client.post(
            "/api/v1/search/graph",
            json={"query": "CUDA Reciprocal Rank Fusion BGE-M3", "top_k": 5},
        )
        assert s_gr.status_code == 200
        assert s_gr.json()["graph_status"] in ("ok", "empty")

        # 6e. Hybrid search across all 3 fusion strategies: rrf, normalized_linear, qdrant_native_rrf
        for strategy in ("rrf", "normalized_linear", "qdrant_native_rrf"):
            s_hyb = await client.post(
                "/api/v1/search/hybrid",
                json={
                    "query": "What overhead does ECC memory incur and how does RRF fusion work?",
                    "top_k": 5,
                    "fusion_strategy": strategy,
                    "enable_reranking": True,
                    "document_id": md_id,
                },
            )
            assert s_hyb.status_code == 200, f"Strategy {strategy} failed: {s_hyb.text}"
            h_data = s_hyb.json()
            assert h_data["reranked"] is True
            assert len(h_data["results"]) >= 1
            top_hit = h_data["results"][0]
            assert top_hit["document_id"] == md_id
            assert top_hit["rerank_score"] is not None

        # 6f. Legacy GET /api/v1/knowledge/search endpoint parity
        s_legacy = await client.get(
            "/api/v1/knowledge/search",
            params={"q": "ECC memory single-bit DRAM faults", "top_k": 4},
        )
        assert s_legacy.status_code == 200
        assert len(s_legacy.json()["citations"]) >= 1

        # 6g. Web search endpoint (/api/v1/search/web)
        s_web = await client.post("/api/v1/search/web", json={"query": "CUDA programming guide"})
        assert s_web.status_code == 200
        assert s_web.json()["status"] in ("unconfigured", "ok")

        # 7. Test ALL Single-Agent Assistant LangChain StructuredTools directly
        assistant = get_assistant()
        tools_by_name = {t.name: t for t in assistant.tools}
        expected_tools = {
            "document_search",
            "vector_search",
            "graph_search",
            "cypher_query",
            "web_search",
            "document_summary",
            "source_lookup",
            "dataset_analytics_profiler",
        }
        assert set(tools_by_name.keys()) == expected_tools

        t_doc = await tools_by_name["document_search"].ainvoke({"query": "ECC memory overhead", "top_k": 3})
        assert isinstance(t_doc, list) and len(t_doc) >= 1

        t_vec = await tools_by_name["vector_search"].ainvoke({"query": "128-byte aligned cache lines", "top_k": 3})
        assert isinstance(t_vec, list) and len(t_vec) >= 1

        t_gr = await tools_by_name["graph_search"].ainvoke({"query": "CUDA", "top_k": 3})
        assert "graph_status" in t_gr

        t_cyp = await tools_by_name["cypher_query"].ainvoke({"cypher": "MATCH (n) RETURN count(n) AS c LIMIT 1"})
        assert t_cyp["status"] in ("ok", "rejected")

        t_web = await tools_by_name["web_search"].ainvoke({"query": "Reciprocal Rank Fusion"})
        assert "web_status" in t_web

        t_sum = await tools_by_name["document_summary"].ainvoke({"document_id": md_id})
        assert t_sum["status"] == "ok"
        assert t_sum["document_id"] == md_id

        first_chunk_id = chunks_in_qdrant[0]["chunk_id"]
        t_lookup = await tools_by_name["source_lookup"].ainvoke({"document_id": md_id, "chunk_id": first_chunk_id})
        assert t_lookup["status"] == "ok"
        assert t_lookup["chunk"]["chunk_id"] == first_chunk_id

        # 8. Multi-Turn Conversation CRUD + Grounded Chat + Abstention + SSE Stream
        conv_create = await client.post("/api/v1/conversations", json={"title": "RAG QA Session"})
        assert conv_create.status_code == 201
        conv_id = conv_create.json()["id"]

        # Turn 1: Grounded technical question
        chat_t1 = await client.post(
            "/api/v1/chat",
            json={
                "message": "What capacity overhead does ECC memory incur according to the architecture specification?",
                "conversation_id": conv_id,
                "document_id": md_id,
                "mode": "hybrid",
                "enable_reranking": True,
            },
        )
        assert chat_t1.status_code == 200
        c1 = chat_t1.json()
        assert c1["conversation_id"] == conv_id
        assert "12.5%" in c1["answer"] or "ECC" in c1["answer"]
        assert len(c1["citations"]) >= 1
        assert c1["retrieval_diagnostics"]["reranked"] is True

        # Turn 2: Follow-up question using conversation context
        chat_t2 = await client.post(
            "/api/v1/chat",
            json={
                "message": "And what constant k does it use for Reciprocal Rank Fusion?",
                "conversation_id": conv_id,
                "document_id": md_id,
            },
        )
        assert chat_t2.status_code == 200
        assert "60" in chat_t2.json()["answer"] or "Reciprocal Rank Fusion" in chat_t2.json()["answer"]

        # Verify conversation history persisted 4 messages (2 user + 2 assistant)
        conv_get = await client.get(f"/api/v1/conversations/{conv_id}")
        assert conv_get.status_code == 200
        assert len(conv_get.json()["messages"]) == 4

        conv_list = await client.get("/api/v1/conversations")
        assert conv_list.status_code == 200
        assert any(c["id"] == conv_id for c in conv_list.json())

        # Out-of-domain unanswerable question scoped to md_id -> grounded low-evidence or no fabricated tags
        chat_ood = await client.post(
            "/api/v1/chat",
            json={
                "message": "zzzxxyyqq quantum photosynthesis recipe on Jupiter moons 998877?",
                "document_id": md_id,
            },
        )
        assert chat_ood.status_code == 200
        assert "[KB-99]" not in chat_ood.json()["answer"]

        # SSE Chat Stream verification (/api/v1/chat/stream)
        sse_res = await client.post(
            "/api/v1/chat/stream",
            json={"message": "Explain memory coalescing warp alignment", "document_id": md_id},
        )
        assert sse_res.status_code == 200
        sse_body = sse_res.text
        assert "event: sources" in sse_body
        assert "event: data" in sse_body
        assert "event: done" in sse_body

        # Delete conversation -> 204 then 404
        conv_del = await client.delete(f"/api/v1/conversations/{conv_id}")
        assert conv_del.status_code == 204
        conv_404 = await client.get(f"/api/v1/conversations/{conv_id}")
        assert conv_404.status_code == 404

        # 9. Knowledge Graph Visualization, Schema, Fail-Closed Cypher, and CSV Proposal Upload
        kg_viz = await client.get("/api/v1/knowledge-graph?limit=100")
        assert kg_viz.status_code == 200
        assert "nodes" in kg_viz.json() and "edges" in kg_viz.json()

        kg_legacy = await client.get("/api/v1/knowledge/graph")
        assert kg_legacy.status_code == 200
        assert "nodes" in kg_legacy.json()

        kg_schema = await client.get("/api/v1/knowledge-graph/schema")
        assert kg_schema.status_code == 200
        assert "labels" in kg_schema.json()

        kg_ro = await client.post(
            "/api/v1/knowledge-graph/query",
            json={"cypher": "MATCH (n) RETURN count(n) AS total"},
        )
        # Fails closed with 503 when NEO4J_READONLY_USER is unset, or 200 if configured
        assert kg_ro.status_code in (503, 200)

        # Unauthenticated CSV-to-KG proposal upload returns 401; authenticated returns 201
        csv_unauth = await client.post(
            "/api/v1/graph/csv/upload",
            files=[("files", ("departments.csv", b"dept_id,dept_name\nD1,Research\nD2,Platform\n", "text/csv"))],
        )
        assert csv_unauth.status_code == 401

        auth_reg = await client.post(
            "/api/v1/auth/register",
            json={"email": f"csv-{nonce}@nexusai.local", "password": "StrongPassword!123", "name": "CSV Tester"},
        )
        auth_headers = {"Authorization": f"Bearer {auth_reg.json()['access_token']}"}
        csv_prop = await client.post(
            "/api/v1/graph/csv/upload",
            files=[("files", ("departments.csv", b"dept_id,dept_name\nD1,Research\nD2,Platform\n", "text/csv"))],
            headers=auth_headers,
        )
        assert csv_prop.status_code == 201
        prop_id = csv_prop.json()["proposal_id"]
        prop_get = await client.get(f"/api/v1/graph/csv/proposal/{prop_id}", headers=auth_headers)
        assert prop_get.status_code == 200

        # 10. Settings GET & PATCH (verifying secret immutability)
        s_get = await client.get("/api/v1/settings")
        assert s_get.status_code == 200
        orig_top_k = s_get.json()["retrieval"]["top_k"]

        s_patch = await client.patch(
            "/api/v1/settings",
            json={"RETRIEVAL_TOP_K": 7, "GOOGLE_API_KEYS": "ATTEMPTED_SECRET_INJECTION"},
        )
        assert s_patch.status_code == 200
        assert s_patch.json()["updated"] == {"RETRIEVAL_TOP_K": 7}
        assert "GOOGLE_API_KEYS" not in s_patch.json()["updated"]
        # Restore original setting
        await client.patch("/api/v1/settings", json={"RETRIEVAL_TOP_K": orig_top_k})

        # 11. Comparative Evaluation Benchmark Endpoint (/api/v1/evaluation/run & /evaluation)
        ev_run = await client.post(
            "/api/v1/evaluation/run",
            json={
                "cases": [
                    {
                        "case_id": "qa-ecc",
                        "category": "acronym",
                        "question": "What capacity overhead does ECC memory incur?",
                        "expected_keywords": ["12.5%", "ECC", "DRAM"],
                        "expected_filenames": [f"arch_spec_{nonce}.md"],
                        "expect_abstention": False,
                    },
                    {
                        "case_id": "qa-ood",
                        "category": "no_evidence",
                        "question": "zzzxxyyqq alien warp reactor temperature on Pluto 991122?",
                        "expected_keywords": [],
                        "expect_abstention": True,
                    },
                ],
                "top_k": 4,
                "document_id": md_id,
            },
        )
        assert ev_run.status_code == 200
        ev_json = ev_run.json()
        assert set(ev_json["comparison"].keys()) == {"dense_only", "hybrid_rrf", "hybrid_reranked"}

        ev_latest = await client.get("/api/v1/evaluation")
        assert ev_latest.status_code == 200
        assert len(ev_latest.json()["benchmark_runs"]) >= 1

    finally:
        # 12. Cleanly delete all test documents and verify 404 after deletion
        for doc_id in created_doc_ids:
            del_res = await client.delete(f"/api/v1/documents/{doc_id}")
            assert del_res.status_code == 204
            chk_404 = await client.get(f"/api/v1/documents/{doc_id}")
            assert chk_404.status_code == 404

