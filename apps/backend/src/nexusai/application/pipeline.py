"""RAG pipeline backward-compatibility facade delegating lazily to `nexusai.core.container`.

Eliminates import-time database/model initialization and fake chunk counts.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, AsyncGenerator
from uuid import UUID

from nexusai.core.container import (
    get_hybrid_retriever,
    get_ingest_use_case,
    get_knowledge_repo,
    get_llm_registry,
)
from nexusai.core.logging import get_logger
from nexusai.infrastructure.llm.provider import synthesize_extractive_grounded_answer

logger = get_logger(__name__)


async def ingest_file(source_id: str, filename: str, content: bytes, source_type: str) -> dict[str, Any]:
    """Run the ingestion pipeline for a single file (backward compatible)."""
    suffix = Path(filename).suffix or f".{source_type}"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        use_case = get_ingest_use_case()
        sid = UUID(source_id)
        await use_case.execute(
            source_id=sid,
            file_path=tmp_path,
            user_id=UUID("00000000-0000-0000-0000-000000000000"),
        )
        src = await get_knowledge_repo().get_source(sid)
        return {
            "source_id": source_id,
            "status": "completed",
            "chunks": src.chunk_count if src else 0,
        }
    except Exception as exc:
        logger.error("Ingestion failed for %s: %s", filename, exc)
        return {"source_id": source_id, "status": "failed", "error": str(exc)}
    finally:
        Path(tmp_path).unlink(missing_ok=True)


async def retrieve_context(query: str, limit: int = 5) -> list[dict[str, Any]]:
    """Retrieve hybrid context chunks from NexusAI."""
    res = await get_hybrid_retriever().retrieve(query=query, top_k=limit)
    return res.chunks


async def generate_answer(
    query: str,
    context: list[dict[str, Any]],
    provider: str | None = None,
    model: str | None = None,
) -> str:
    """Generate a grounded answer using the configured LLM provider or extractive fallback."""
    _, llm = get_llm_registry().resolve(provider)
    if llm is None:
        return synthesize_extractive_grounded_answer(query=query, citations=context)
    context_text = "\n\n".join(
        str(item.get("text") or item.get("payload", {}).get("text", ""))
        for item in context
    )
    prompt = (
        "You are NexusAI. Answer using ONLY the context below. If unsupported, say so.\n\n"
        f"Context:\n{context_text}\n\nQuestion: {query}\nAnswer:"
    )
    return await llm.generate(prompt, temperature=0.1)


async def stream_answer(
    query: str,
    context: list[dict[str, Any]],
    provider: str | None = None,
    model: str | None = None,
) -> AsyncGenerator[str, None]:
    """Stream an answer token-by-token."""
    _, llm = get_llm_registry().resolve(provider)
    if llm is None:
        yield synthesize_extractive_grounded_answer(query=query, citations=context)
        return
    context_text = "\n\n".join(
        str(item.get("text") or item.get("payload", {}).get("text", ""))
        for item in context
    )
    prompt = (
        "You are NexusAI. Answer using ONLY the context below. If unsupported, say so.\n\n"
        f"Context:\n{context_text}\n\nQuestion: {query}\nAnswer:"
    )
    async for delta in llm.stream(prompt, temperature=0.1):
        yield delta
