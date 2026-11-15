"""Knowledge API endpoints — Hybrid RAG Upload, Idempotent Ingestion, Search, Chat, SSE Streaming, & Graph."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any
from uuid import UUID

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

from nexusai.application.services.knowledge_service import KnowledgeService
from nexusai.application.use_cases.query_knowledge import QueryKnowledgeUseCase
from nexusai.core.config import get_settings
from nexusai.core.container import (
    dispatch_document_ingestion,
    get_graph_store,
    get_knowledge_repo,
    get_knowledge_service,
    get_llm_registry,
    get_query_use_case,
    get_vector_store,
)
from nexusai.core.ids import compute_sha256_bytes
from nexusai.core.logging import get_logger
from nexusai.domain.entities.knowledge import ConversationRole, SourceStatus
from nexusai.infrastructure.parsers.file_inspector import (
    FileValidationError,
    validate_and_inspect_file,
)

logger = get_logger(__name__)

router = APIRouter(prefix="/knowledge", tags=["knowledge"])

_DEFAULT_USER_ID = "00000000-0000-0000-0000-000000000000"

FILE_SIZE_LIMITS: dict[str, int] = {
    "pdf": 100 * 1024 * 1024,
    "doc": 50 * 1024 * 1024,
    "docx": 50 * 1024 * 1024,
    "pptx": 50 * 1024 * 1024,
    "xlsx": 100 * 1024 * 1024,
    "html": 20 * 1024 * 1024,
    "htm": 20 * 1024 * 1024,
    "csv": 100 * 1024 * 1024,
    "txt": 10 * 1024 * 1024,
    "md": 10 * 1024 * 1024,
    "json": 10 * 1024 * 1024,
    "png": 25 * 1024 * 1024,
    "jpg": 25 * 1024 * 1024,
    "jpeg": 25 * 1024 * 1024,
    "tiff": 25 * 1024 * 1024,
    "bmp": 25 * 1024 * 1024,
    "webp": 25 * 1024 * 1024,
}

SOURCE_TYPE_MAP: dict[str, str] = {
    "pdf": "pdf",
    "doc": "docx",
    "docx": "docx",
    "pptx": "pptx",
    "xlsx": "xlsx",
    "html": "html",
    "htm": "html",
    "csv": "csv",
    "txt": "txt",
    "md": "md",
    "json": "json",
    "png": "image",
    "jpg": "image",
    "jpeg": "image",
    "tiff": "image",
    "tif": "image",
    "bmp": "image",
    "webp": "image",
}


def _resolve_upload_storage_dir() -> Path:
    settings = get_settings()
    base = Path(settings.UPLOAD_DIR).parent / "documents"
    base.mkdir(parents=True, exist_ok=True)
    return base


async def process_single_upload(
    file: UploadFile,
    user_id_str: str,
    source_type_hint: str,
    service: KnowledgeService,
    background_tasks: BackgroundTasks | None,
    force_reindex: bool = False,
    wait_inline: bool = False,
) -> dict[str, Any]:
    """Validate, deduplicate, persist, and dispatch ingestion for one uploaded document."""
    uid = uuid.UUID(user_id_str)
    content = await file.read()
    filename = Path(file.filename or "document.txt").name
    ext = Path(filename).suffix.lower().lstrip(".")

    if len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Uploaded file '{filename}' is empty (0 bytes).",
        )

    max_size = FILE_SIZE_LIMITS.get(ext, 50 * 1024 * 1024)
    if len(content) > max_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Max size for .{ext} is {max_size // (1024 * 1024)} MB",
        )

    content_hash = compute_sha256_bytes(content)
    repo = get_knowledge_repo()

    # Idempotent duplicate check: avoid uncontrolled duplicate indexing unless force_reindex=True
    if not force_reindex and hasattr(repo, "find_source_by_hash"):
        existing = await repo.find_source_by_hash(uid, content_hash)
        if existing is not None and existing.status == SourceStatus.READY:
            return {
                "source_id": str(existing.id),
                "document_id": str(existing.id),
                "status": "ready",
                "duplicate": True,
                "chunks": existing.chunk_count,
                "filename": existing.title,
            }

    storage_dir = _resolve_upload_storage_dir()
    staged_path = storage_dir / f"{content_hash[:16]}_{filename}"
    staged_path.write_bytes(content)

    # Pre-flight validation (magic bytes, corrupted PDF/ZIP rejection, format support)
    try:
        inspection = validate_and_inspect_file(staged_path, filename=filename, max_bytes=max_size)
    except FileValidationError as exc:
        staged_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    mapped_type = SOURCE_TYPE_MAP.get(ext, source_type_hint if source_type_hint in SOURCE_TYPE_MAP.values() else "txt")

    source = await service.create_source(
        user_id=uid,
        title=filename,
        source_type=mapped_type,
        metadata={
            "filename": filename,
            "size": len(content),
            "content_type": file.content_type,
            "content_hash": content_hash,
            "stored_path": str(staged_path),
            "recommended_parser": inspection.recommended_parser,
            "page_count": inspection.page_count,
        },
    )

    dispatch_info = await dispatch_document_ingestion(
        source_id=source.id,
        file_path=str(staged_path),
        user_id=uid,
        background_tasks=background_tasks,
        wait_inline=wait_inline,
    )
    return {
        "source_id": str(source.id),
        "document_id": str(source.id),
        "filename": filename,
        "duplicate": False,
        "recommended_parser": inspection.recommended_parser,
        **dispatch_info,
    }


# ── Sources & Ingestion ──────────────────────────────────────────


@router.post("/upload", status_code=201)
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    source_type: str = Form("pdf"),
    user_id: str = Form(_DEFAULT_USER_ID),
    force_reindex: bool = Form(False),
    wait_inline: bool = Form(False),
    service: KnowledgeService = Depends(get_knowledge_service),
) -> dict[str, Any]:
    """Upload a file, validate integrity, deduplicate by SHA-256, and run ingestion."""
    return await process_single_upload(
        file=file,
        user_id_str=user_id,
        source_type_hint=source_type,
        service=service,
        background_tasks=background_tasks,
        force_reindex=force_reindex,
        wait_inline=wait_inline,
    )


@router.post("/upload/batch", status_code=201)
async def upload_multiple_files(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    user_id: str = Form(_DEFAULT_USER_ID),
    force_reindex: bool = Form(False),
    wait_inline: bool = Form(False),
    service: KnowledgeService = Depends(get_knowledge_service),
) -> dict[str, Any]:
    """Upload and ingest multiple documents in a single request."""
    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for f in files:
        try:
            res = await process_single_upload(
                file=f,
                user_id_str=user_id,
                source_type_hint="pdf",
                service=service,
                background_tasks=background_tasks,
                force_reindex=force_reindex,
                wait_inline=wait_inline,
            )
            results.append(res)
        except HTTPException as exc:
            errors.append({"filename": f.filename, "status_code": exc.status_code, "error": exc.detail})
    return {"uploaded": results, "errors": errors, "total_submitted": len(files)}


@router.get("/sources", response_model=list[dict[str, Any]])
async def list_sources(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    user_id: str = Query(_DEFAULT_USER_ID),
    service: KnowledgeService = Depends(get_knowledge_service),
) -> list[dict[str, Any]]:
    sources, _ = await service.list_sources(uuid.UUID(user_id), limit, offset)
    return [s.model_dump(mode="json") for s in sources]


@router.get("/sources/{source_id}", response_model=dict[str, Any])
async def get_source(
    source_id: uuid.UUID,
    user_id: str = Query(_DEFAULT_USER_ID),
    service: KnowledgeService = Depends(get_knowledge_service),
) -> dict[str, Any]:
    source = await service.get_source(source_id)
    if not source or source.user_id != uuid.UUID(user_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Source not found")
    return source.model_dump(mode="json")


@router.post("/sources/{source_id}/reindex")
async def reindex_source(
    source_id: uuid.UUID,
    background_tasks: BackgroundTasks,
    wait_inline: bool = Query(False),
    user_id: str = Query(_DEFAULT_USER_ID),
    service: KnowledgeService = Depends(get_knowledge_service),
) -> dict[str, Any]:
    """Re-index an existing document cleanly across Qdrant and Neo4j."""
    uid = uuid.UUID(user_id)
    source = await service.get_source(source_id)
    if not source or source.user_id != uid:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Source not found")

    stored_path = (source.metadata or {}).get("stored_path")
    if not stored_path or not Path(stored_path).exists():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Original uploaded file is not available on disk for re-indexing; please re-upload.",
        )

    await get_vector_store().delete_document_chunks(source_id)
    await get_graph_store().delete_document_graph(source_id)
    return await dispatch_document_ingestion(
        source_id=source_id,
        file_path=str(stored_path),
        user_id=uid,
        background_tasks=background_tasks,
        wait_inline=wait_inline,
    )


@router.delete("/sources/{source_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_source(
    source_id: uuid.UUID,
    user_id: str = Query(_DEFAULT_USER_ID),
    service: KnowledgeService = Depends(get_knowledge_service),
) -> None:
    source = await service.get_source(source_id)
    if not source or source.user_id != uuid.UUID(user_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Source not found")

    await get_vector_store().delete_document_chunks(source_id)
    await get_graph_store().delete_document_graph(source_id)

    stored_path = (source.metadata or {}).get("stored_path")
    if stored_path:
        Path(stored_path).unlink(missing_ok=True)

    await service.delete_source(source_id)


# ── Conversations ────────────────────────────────────────────────


@router.post("/conversations", response_model=dict[str, Any], status_code=201)
async def create_conversation(
    body: dict[str, Any],
    user_id: str = Query(_DEFAULT_USER_ID),
    service: KnowledgeService = Depends(get_knowledge_service),
) -> dict[str, Any]:
    conversation = await service.create_conversation(
        user_id=uuid.UUID(user_id),
        title=body.get("title"),
    )
    return conversation.model_dump(mode="json")


@router.get("/conversations", response_model=list[dict[str, Any]])
async def list_conversations(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    user_id: str = Query(_DEFAULT_USER_ID),
    service: KnowledgeService = Depends(get_knowledge_service),
) -> list[dict[str, Any]]:
    conversations, _ = await service.list_conversations(uuid.UUID(user_id), limit, offset)
    return [c.model_dump(mode="json") for c in conversations]


@router.get("/conversations/{conversation_id}", response_model=dict[str, Any])
async def get_conversation(
    conversation_id: uuid.UUID,
    user_id: str = Query(_DEFAULT_USER_ID),
    service: KnowledgeService = Depends(get_knowledge_service),
) -> dict[str, Any]:
    conversation = await service.get_conversation(conversation_id)
    if not conversation or conversation.user_id != uuid.UUID(user_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    return conversation.model_dump(mode="json")


@router.delete("/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(
    conversation_id: uuid.UUID,
    user_id: str = Query(_DEFAULT_USER_ID),
) -> None:
    repo = get_knowledge_repo()
    conv = await repo.get_conversation(conversation_id)
    if not conv or conv.user_id != uuid.UUID(user_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    await repo.delete_conversation(conversation_id)


@router.post("/conversations/{conversation_id}/messages", response_model=dict[str, Any])
async def add_message(
    conversation_id: uuid.UUID,
    body: dict[str, Any],
    user_id: str = Query(_DEFAULT_USER_ID),
    service: KnowledgeService = Depends(get_knowledge_service),
) -> dict[str, Any]:
    conversation = await service.get_conversation(conversation_id)
    if not conversation or conversation.user_id != uuid.UUID(user_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    message = await service.add_message(
        conversation_id=conversation_id,
        role=body.get("role", ConversationRole.USER),
        content=body.get("content", ""),
        citations=body.get("citations", []),
        metadata=body.get("metadata", {}),
    )
    return message.model_dump(mode="json")


# ── Hybrid RAG Search / Chat ─────────────────────────────────────


def _extract_filter(body: dict[str, Any]) -> dict[str, Any] | None:
    filters: dict[str, Any] = {}
    if body.get("document_ids"):
        filters["document_ids"] = body["document_ids"]
    if body.get("document_id"):
        filters["document_id"] = body["document_id"]
    if body.get("source_type"):
        filters["source_type"] = body["source_type"]
    return filters or None


@router.get("/search", response_model=dict[str, Any])
async def search(
    q: str = Query(..., min_length=1),
    top_k: int = Query(8, ge=1, le=30),
    mode: str | None = Query(None),
    fusion_strategy: str | None = Query(None),
    rerank: bool | None = Query(None),
    include_web: bool = Query(False),
    provider: str | None = Query(None),
    model: str | None = Query(None),
    use_case: QueryKnowledgeUseCase = Depends(get_query_use_case),
) -> dict[str, Any]:
    res = await use_case.execute_detailed(
        query=q,
        user_id=UUID(_DEFAULT_USER_ID),
        top_k=top_k,
        provider=provider,
        model=model,
        mode=mode,
        fusion_strategy=fusion_strategy,
        enable_reranking=rerank,
        include_web=include_web,
    )
    return {
        "request_id": res.request_id,
        "query": q,
        "answer": res.answer,
        "citations": res.citations,
        "diagnostics": res.diagnostics,
    }


@router.post("/chat", response_model=dict[str, Any])
async def chat(
    body: dict[str, Any],
    use_case: QueryKnowledgeUseCase = Depends(get_query_use_case),
) -> dict[str, Any]:
    query = str(body.get("message") or body.get("query") or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="message is required")
    conv_raw = body.get("conversation_id")
    conv_id = UUID(str(conv_raw)) if conv_raw else None

    res = await use_case.execute_detailed(
        query=query,
        user_id=UUID(str(body.get("user_id") or _DEFAULT_USER_ID)),
        top_k=int(body.get("top_k") or 8),
        provider=body.get("provider"),
        model=body.get("model"),
        conversation_id=conv_id,
        mode=body.get("mode"),
        fusion_strategy=body.get("fusion_strategy"),
        enable_reranking=body.get("enable_reranking"),
        include_web=bool(body.get("include_web", False)),
        filter_metadata=_extract_filter(body),
    )
    return {
        "request_id": res.request_id,
        "conversation_id": res.conversation_id,
        "role": "assistant",
        "content": res.answer,
        "answer": res.answer,
        "citations": res.citations,
        "diagnostics": res.diagnostics,
        "retrieval_diagnostics": res.diagnostics,
        "chart": getattr(res, "chart", None),
    }


@router.post("/chat/stream")
async def chat_stream(
    body: dict[str, Any],
    use_case: QueryKnowledgeUseCase = Depends(get_query_use_case),
) -> StreamingResponse:
    query = str(body.get("message") or body.get("query") or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="message is required")
    conv_raw = body.get("conversation_id")
    conv_id = UUID(str(conv_raw)) if conv_raw else None

    async def event_generator():
        try:
            async for event in use_case.stream_execute(
                query=query,
                user_id=UUID(str(body.get("user_id") or _DEFAULT_USER_ID)),
                top_k=int(body.get("top_k") or 8),
                provider=body.get("provider"),
                model=body.get("model"),
                conversation_id=conv_id,
                mode=body.get("mode"),
                fusion_strategy=body.get("fusion_strategy"),
                enable_reranking=body.get("enable_reranking"),
                include_web=bool(body.get("include_web", False)),
                filter_metadata=_extract_filter(body),
            ):
                ev_type = event["event"]
                payload = json.dumps(event["data"], ensure_ascii=False)
                if ev_type == "sources":
                    yield f"event: sources\ndata: {payload}\n\n"
                elif ev_type == "data":
                    yield f"event: data\ndata: {payload}\n\n"
                elif ev_type == "done":
                    yield f"event: done\ndata: {payload}\n\n"
        except Exception as exc:
            logger.error("SSE chat stream error: %s", exc)
            err_payload = json.dumps({"error": str(exc)})
            yield f"event: error\ndata: {err_payload}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/models", response_model=dict[str, Any])
async def list_models() -> dict[str, Any]:
    return get_llm_registry().status()


# ── Graph ────────────────────────────────────────────────────────


@router.get("/graph", response_model=dict[str, Any])
async def get_graph(
    source_id: str | None = Query(None),
    limit: int = Query(250, ge=10, le=1000),
) -> dict[str, Any]:
    return await get_graph_store().get_graph_visualization(source_id=source_id, limit=limit)


@router.post("/graph/csv", status_code=410)
async def upload_graph_csv(file: UploadFile | None = File(None)) -> dict[str, Any]:
    """Deprecated — replaced by the CSV-to-graph workflow."""
    raise HTTPException(
        status_code=410,
        detail=(
            "This endpoint has been removed. Use POST /api/v1/graph/csv/upload to upload "
            "CSVs, then review and approve the schema proposal before ingesting."
        ),
    )


@router.post("/graph/build")
async def build_graph_from_source(
    body: dict[str, Any],
    service: KnowledgeService = Depends(get_knowledge_service),
) -> dict[str, Any]:
    """Extract and synchronize Neo4j entities/relationships from an indexed document's chunks."""
    from nexusai.application.graph.entity_extractor import HybridEntityExtractor

    raw_id = body.get("source_id")
    if not raw_id:
        raise HTTPException(status_code=400, detail="source_id is required")
    sid = UUID(str(raw_id))
    source = await service.get_source(sid)
    if not source:
        raise HTTPException(status_code=404, detail="Source not found")

    chunks = await get_vector_store().get_document_chunks(sid, limit=300)
    if not chunks:
        raise HTTPException(status_code=400, detail="No indexed chunks found for this document; re-index first.")

    _, llm = get_llm_registry().resolve()
    extractor = HybridEntityExtractor(llm=llm, enable_llm=False)
    entities, relationships, _ = await extractor.extract_for_chunks(chunks)

    gstore = get_graph_store()
    await gstore.create_document_node(
        document_id=sid,
        filename=source.title,
        source_type=str(source.source_type),
        user_id=source.user_id,
    )
    await gstore.create_chunk_nodes(chunks)
    if entities:
        await gstore.create_batch_entities_and_relationships(
            entities=entities,
            relationships=relationships,
            document_id=sid,
        )
    return {
        "status": "ok",
        "source_id": str(sid),
        "chunks_processed": len(chunks),
        "entities_extracted": len(entities),
        "relationships_extracted": len(relationships),
    }
