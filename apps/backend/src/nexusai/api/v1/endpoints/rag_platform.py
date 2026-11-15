"""Unified Hybrid RAG Platform REST Endpoints: Documents, Chat, Multi-Channel Search, KG Schema/Query, Settings, & Evaluation."""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from fastapi.responses import StreamingResponse

from nexusai.api.v1.endpoints.knowledge import (
    _DEFAULT_USER_ID,
    _extract_filter,
    chat,
    chat_stream,
    create_conversation,
    delete_conversation,
    delete_source,
    get_conversation,
    get_source,
    list_conversations,
    list_sources,
    process_single_upload,
    reindex_source,
)
from nexusai.application.services.knowledge_service import KnowledgeService
from nexusai.application.use_cases.query_knowledge import QueryKnowledgeUseCase
from nexusai.core.config import get_settings
from nexusai.core.container import (
    get_assistant,
    get_embedder,
    get_graph_store,
    get_hybrid_retriever,
    get_knowledge_service,
    get_llm_registry,
    get_query_use_case,
    get_readonly_graph_store,
    get_tavily_service,
    get_vector_store,
    reset_embedding_and_vector_caches,
)
from nexusai.core.observability import get_observability_tracker
from nexusai.evaluation.evaluator import EvaluationCase, RAGEvaluator
from nexusai.infrastructure.graph_stores.neo4j_readonly_store import (
    ReadOnlyStoreUnavailable,
    ReadOnlyViolation,
)

router = APIRouter(tags=["hybrid-rag-platform"])

_EVALUATOR_SINGLETON: RAGEvaluator | None = None


def _get_evaluator() -> RAGEvaluator:
    global _EVALUATOR_SINGLETON
    if _EVALUATOR_SINGLETON is None:
        _EVALUATOR_SINGLETON = RAGEvaluator(
            retriever=get_hybrid_retriever(),
            assistant=get_assistant(),
        )
    return _EVALUATOR_SINGLETON


# ── Documents Endpoints (`/documents/*`) ──────────────────────────


@router.get("/documents")
async def list_documents(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    user_id: str = Query(_DEFAULT_USER_ID),
    service: KnowledgeService = Depends(get_knowledge_service),
) -> list[dict[str, Any]]:
    return await list_sources(limit=limit, offset=offset, user_id=user_id, service=service)


@router.post("/documents/upload", status_code=201)
async def upload_documents(
    background_tasks: BackgroundTasks,
    file: UploadFile | None = File(None),
    files: list[UploadFile] | None = File(None),
    user_id: str = Form(_DEFAULT_USER_ID),
    force_reindex: bool = Form(False),
    wait_inline: bool = Form(False),
    service: KnowledgeService = Depends(get_knowledge_service),
) -> dict[str, Any]:
    incoming = list(files or [])
    if file is not None:
        incoming.insert(0, file)
    if not incoming:
        raise HTTPException(status_code=400, detail="At least one file is required.")

    if len(incoming) == 1:
        return await process_single_upload(
            file=incoming[0],
            user_id_str=user_id,
            source_type_hint="pdf",
            service=service,
            background_tasks=background_tasks,
            force_reindex=force_reindex,
            wait_inline=wait_inline,
        )

    uploaded: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for item in incoming:
        try:
            uploaded.append(
                await process_single_upload(
                    file=item,
                    user_id_str=user_id,
                    source_type_hint="pdf",
                    service=service,
                    background_tasks=background_tasks,
                    force_reindex=force_reindex,
                    wait_inline=wait_inline,
                )
            )
        except HTTPException as exc:
            errors.append({"filename": item.filename, "status_code": exc.status_code, "error": exc.detail})
    return {"uploaded": uploaded, "errors": errors, "total_submitted": len(incoming)}


@router.get("/documents/{document_id}")
async def get_document(
    document_id: uuid.UUID,
    user_id: str = Query(_DEFAULT_USER_ID),
    service: KnowledgeService = Depends(get_knowledge_service),
) -> dict[str, Any]:
    return await get_source(source_id=document_id, user_id=user_id, service=service)


@router.get("/documents/{document_id}/status")
async def get_document_status(
    document_id: uuid.UUID,
    user_id: str = Query(_DEFAULT_USER_ID),
    service: KnowledgeService = Depends(get_knowledge_service),
) -> dict[str, Any]:
    src = await get_source(source_id=document_id, user_id=user_id, service=service)
    meta = src.get("metadata") or {}
    return {
        "document_id": src["id"],
        "filename": src["title"],
        "status": src["status"],
        "stage": meta.get("ingestion_stage", src["status"]),
        "chunk_count": src["chunk_count"],
        "parser_used": meta.get("parser_used") or meta.get("recommended_parser"),
        "error_message": src.get("error_message"),
    }


@router.post("/documents/{document_id}/reindex")
async def reindex_document(
    document_id: uuid.UUID,
    background_tasks: BackgroundTasks,
    wait_inline: bool = Query(False),
    user_id: str = Query(_DEFAULT_USER_ID),
    service: KnowledgeService = Depends(get_knowledge_service),
) -> dict[str, Any]:
    return await reindex_source(
        source_id=document_id,
        background_tasks=background_tasks,
        wait_inline=wait_inline,
        user_id=user_id,
        service=service,
    )


@router.delete("/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_document(
    document_id: uuid.UUID,
    user_id: str = Query(_DEFAULT_USER_ID),
    service: KnowledgeService = Depends(get_knowledge_service),
) -> None:
    await delete_source(source_id=document_id, user_id=user_id, service=service)


# ── Chat & Conversations (`/chat*`, `/conversations*`) ────────────


@router.post("/chat")
async def chat_endpoint(
    body: dict[str, Any],
    use_case: QueryKnowledgeUseCase = Depends(get_query_use_case),
) -> dict[str, Any]:
    return await chat(body=body, use_case=use_case)


@router.post("/chat/stream")
async def chat_stream_endpoint(
    body: dict[str, Any],
    use_case: QueryKnowledgeUseCase = Depends(get_query_use_case),
) -> StreamingResponse:
    return await chat_stream(body=body, use_case=use_case)


@router.get("/conversations")
async def list_conversations_endpoint(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    user_id: str = Query(_DEFAULT_USER_ID),
    service: KnowledgeService = Depends(get_knowledge_service),
) -> list[dict[str, Any]]:
    return await list_conversations(limit=limit, offset=offset, user_id=user_id, service=service)


@router.post("/conversations", status_code=201)
async def create_conversation_endpoint(
    body: dict[str, Any],
    user_id: str = Query(_DEFAULT_USER_ID),
    service: KnowledgeService = Depends(get_knowledge_service),
) -> dict[str, Any]:
    return await create_conversation(body=body, user_id=user_id, service=service)


@router.get("/conversations/{conversation_id}")
async def get_conversation_endpoint(
    conversation_id: uuid.UUID,
    user_id: str = Query(_DEFAULT_USER_ID),
    service: KnowledgeService = Depends(get_knowledge_service),
) -> dict[str, Any]:
    return await get_conversation(conversation_id=conversation_id, user_id=user_id, service=service)


@router.delete("/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation_endpoint(
    conversation_id: uuid.UUID,
    user_id: str = Query(_DEFAULT_USER_ID),
) -> None:
    await delete_conversation(conversation_id=conversation_id, user_id=user_id)


# ── Direct Retrieval & Diagnostics (`/search/*`) ──────────────────


@router.post("/search/vector")
async def search_vector(body: dict[str, Any]) -> dict[str, Any]:
    query = str(body.get("query") or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="query is required")
    res = await get_hybrid_retriever().retrieve(
        query=query,
        top_k=int(body.get("top_k") or 8),
        mode="dense",
        enable_reranking=False,
        filter_metadata=_extract_filter(body),
    )
    return {"query": query, "mode": "dense", "results": res.citations, "diagnostics": res.diagnostics}


@router.post("/search/sparse")
async def search_sparse(body: dict[str, Any]) -> dict[str, Any]:
    query = str(body.get("query") or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="query is required")
    res = await get_hybrid_retriever().retrieve(
        query=query,
        top_k=int(body.get("top_k") or 8),
        mode="sparse",
        enable_reranking=False,
        filter_metadata=_extract_filter(body),
    )
    return {"query": query, "mode": "sparse", "results": res.citations, "diagnostics": res.diagnostics}


@router.post("/search/graph")
async def search_graph(body: dict[str, Any]) -> dict[str, Any]:
    query = str(body.get("query") or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="query is required")
    res = await get_hybrid_retriever().retrieve(
        query=query,
        top_k=int(body.get("top_k") or 8),
        mode="graph",
        enable_reranking=False,
    )
    return {
        "query": query,
        "mode": "graph",
        "graph_status": res.graph_status,
        "entities": res.graph_entities,
        "results": res.citations,
        "diagnostics": res.diagnostics,
    }


@router.post("/search/hybrid")
async def search_hybrid(body: dict[str, Any]) -> dict[str, Any]:
    query = str(body.get("query") or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="query is required")
    res = await get_hybrid_retriever().retrieve(
        query=query,
        top_k=int(body.get("top_k") or 8),
        mode="hybrid",
        fusion_strategy=body.get("fusion_strategy") or "rrf",
        enable_reranking=body.get("enable_reranking"),
        include_web=bool(body.get("include_web", False)),
        filter_metadata=_extract_filter(body),
    )
    return {
        "query": query,
        "mode": "hybrid",
        "fusion_strategy": res.fusion_strategy,
        "reranked": res.reranked,
        "results": res.citations,
        "graph_entities": res.graph_entities,
        "web_results": res.web_results,
        "diagnostics": res.diagnostics,
    }


@router.post("/search/web")
async def search_web_endpoint(body: dict[str, Any]) -> dict[str, Any]:
    query = str(body.get("query") or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="query is required")
    return await get_tavily_service().search(
        query=query,
        max_results=int(body.get("max_results") or 5),
        search_depth=str(body.get("search_depth") or "basic"),
    )


# ── Knowledge Graph Explorer (`/knowledge-graph/*`) ───────────────


@router.get("/knowledge-graph")
async def get_knowledge_graph(
    source_id: str | None = Query(None),
    limit: int = Query(250, ge=10, le=1000),
) -> dict[str, Any]:
    return await get_graph_store().get_graph_visualization(source_id=source_id, limit=limit)


@router.get("/knowledge-graph/schema")
async def get_knowledge_graph_schema() -> dict[str, Any]:
    return await get_graph_store().get_schema_summary()


@router.post("/knowledge-graph/cypher")
async def execute_knowledge_graph_cypher_console(body: dict[str, Any]) -> dict[str, Any]:
    """Execute safeguarded read-only Cypher for the Neo4j Browser Console (`EXPLAIN` `query_type=='r'` + `execute_read`)."""
    cypher = str(body.get("cypher") or "").strip()
    if not cypher:
        raise HTTPException(status_code=400, detail="cypher is required")
    try:
        return await get_graph_store().execute_cypher_console(
            cypher=cypher,
            parameters=body.get("params") or body.get("parameters") or {},
            max_rows=int(body.get("max_rows") or 200),
        )
    except ReadOnlyViolation as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Cypher execution error: {exc}") from exc


@router.post("/knowledge-graph/query")
async def query_knowledge_graph_readonly(body: dict[str, Any]) -> dict[str, Any]:
    """Execute safeguarded read-only Cypher via Neo4jReadOnlyStore (fails closed with 503 if unconfigured)."""
    cypher = str(body.get("cypher") or "").strip()
    if not cypher:
        raise HTTPException(status_code=400, detail="cypher is required")
    try:
        ro_store = get_readonly_graph_store()
        res = await ro_store.run_read_only(cypher, parameters=body.get("params") or {})
        return {"status": res.status, "rows": res.records, "count": res.row_count}
    except ReadOnlyStoreUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ReadOnlyViolation as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# ── Settings & Runtime Inspection (`/settings`) ───────────────────


@router.get("/settings")
async def get_runtime_settings() -> dict[str, Any]:
    settings = get_settings()
    embedder = get_embedder()
    qdrant_health = await get_vector_store().health()
    return {
        "embedding": {
            "provider": embedder.provider,
            "model": embedder.model_name,
            "dimension": embedder.get_dimension(),
            "batch_size": settings.EMBEDDING_BATCH_SIZE,
            "cache_size": settings.EMBEDDING_CACHE_SIZE,
            "active_qdrant_collection": qdrant_health.get("active_collection"),
            "collection_points": qdrant_health.get("points_count", 0),
        },
        "chunking": {
            "strategy": settings.CHUNK_STRATEGY,
            "chunk_size": settings.CHUNK_SIZE,
            "chunk_overlap": settings.CHUNK_OVERLAP,
            "ocr_engine": settings.OCR_ENGINE,
            "ocr_min_chars_per_page": settings.OCR_MIN_CHARS_PER_PAGE,
        },
        "retrieval": {
            "top_k": settings.RETRIEVAL_TOP_K,
            "rrf_k": settings.RETRIEVAL_RRF_K,
            "enable_sparse_retrieval": settings.ENABLE_SPARSE_RETRIEVAL,
            "enable_reranking": settings.ENABLE_RERANKING,
        },
        "llm": get_llm_registry().status(),
        "tavily": get_tavily_service().health(),
        "observability": get_observability_tracker().summary()["langsmith"],
    }


@router.patch("/settings")
async def update_runtime_settings(body: dict[str, Any]) -> dict[str, Any]:
    """Update safe runtime retrieval/chunking/embedding/provider parameters without exposing or mutating secrets."""
    from nexusai.infrastructure.embedding.provider import (
        ConfigurableEmbeddingService,
        EmbeddingConfigurationError,
    )

    settings = get_settings()
    allowed_updates = {
        "CHUNK_STRATEGY",
        "CHUNK_SIZE",
        "CHUNK_OVERLAP",
        "OCR_ENGINE",
        "RETRIEVAL_TOP_K",
        "RETRIEVAL_RRF_K",
        "ENABLE_RERANKING",
        "ENABLE_SPARSE_RETRIEVAL",
        "LLM_PROVIDER",
        "EMBEDDING_PROVIDER",
        "EMBEDDING_MODEL",
    }

    # Pre-validate embedding provider/model pair before mutating process settings
    if "EMBEDDING_PROVIDER" in {k.upper() for k in body} or "EMBEDDING_MODEL" in {k.upper() for k in body}:
        upper_map = {k.upper(): v for k, v in body.items()}
        cand_prov = str(upper_map.get("EMBEDDING_PROVIDER", settings.EMBEDDING_PROVIDER))
        cand_mod = str(upper_map.get("EMBEDDING_MODEL", settings.EMBEDDING_MODEL))
        try:
            ConfigurableEmbeddingService(provider=cand_prov, model_name=cand_mod)
        except EmbeddingConfigurationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    applied: dict[str, Any] = {}
    reset_caches = False
    for k, v in body.items():
        upper_k = k.upper()
        if upper_k in allowed_updates and hasattr(settings, upper_k):
            setattr(settings, upper_k, v)
            applied[upper_k] = v
            if upper_k in {"EMBEDDING_PROVIDER", "EMBEDDING_MODEL", "CHUNK_STRATEGY", "CHUNK_SIZE", "CHUNK_OVERLAP", "OCR_ENGINE"}:
                reset_caches = True

    if reset_caches:
        reset_embedding_and_vector_caches()

    return {"updated": applied, "settings": await get_runtime_settings()}


# ── Observability & RAG Evaluation (`/evaluation*`, `/observability*`) ───────


@router.get("/observability/summary")
async def get_observability_summary() -> dict[str, Any]:
    return get_observability_tracker().summary()


@router.get("/evaluation")
@router.get("/evaluation/latest")
async def get_evaluation_and_telemetry(
    limit: int = Query(40, ge=1, le=200),
) -> dict[str, Any]:
    tracker = get_observability_tracker()
    evaluator = _get_evaluator()
    return {
        "telemetry_summary": tracker.summary(),
        "recent_traces": tracker.recent(limit=limit),
        "benchmark_runs": evaluator.recent_runs,
    }


@router.post("/evaluation/run")
async def run_evaluation_benchmark(
    body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = body or {}
    top_k = int(payload.get("top_k") or 5)
    raw_cases = payload.get("cases")
    parsed_cases: list[EvaluationCase] | None = None
    if isinstance(raw_cases, list) and raw_cases:
        parsed_cases = [
            EvaluationCase(
                case_id=str(c.get("case_id") or f"custom-{idx + 1}"),
                category=str(c.get("category") or "custom"),
                question=str(c.get("question") or ""),
                expected_keywords=list(c.get("expected_keywords") or c.get("expected_terms") or []),
                expected_filenames=list(c.get("expected_filenames") or []),
                expect_abstention=bool(c.get("expect_abstention") or c.get("unanswerable") or False),
            )
            for idx, c in enumerate(raw_cases)
            if isinstance(c, dict) and str(c.get("question") or "").strip()
        ]
    evaluator = _get_evaluator()
    return await evaluator.run_comparative_benchmark(
        cases=parsed_cases,
        top_k=top_k,
        filter_metadata=_extract_filter(payload),
    )
