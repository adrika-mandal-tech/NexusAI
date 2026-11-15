"""Single-Agent LangChain Knowledge Assistant with Modular Tools, Multi-Turn Context, & Source Validation.

Architectural Boundary:
NexusAI is strictly a SINGLE-AGENT knowledge assistant equipped with modular
LangChain tools (`document_search`, `vector_search`, `graph_search`,
`cypher_query`, `web_search`, `document_summary`, `source_lookup`) — never a
multi-agent swarm.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, AsyncGenerator
from uuid import UUID

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import StructuredTool

from nexusai.application.data_analyst.dataset_profiler import get_dataset_profiler
from nexusai.application.retrieval.hybrid_retriever import (
    HybridRetrievalEngine,
    HybridRetrievalResult,
)
from nexusai.core.logging import get_logger
from nexusai.core.observability import (
    StepTimer,
    TraceTelemetryRecord,
    get_observability_tracker,
    get_request_id,
    set_request_id,
    traced_step,
)
from nexusai.domain.entities.knowledge import ConversationRole, Message
from nexusai.domain.ports.knowledge_repository import KnowledgeRepositoryPort
from nexusai.domain.ports.llm_port import LLMPort
from nexusai.evaluation.evaluator import (
    compute_live_turn_eval_metrics,
    schedule_langsmith_eval_log,
)
from nexusai.infrastructure.graph_stores.neo4j_readonly_store import (
    Neo4jReadOnlyStore,
    ReadOnlyStoreUnavailable,
    ReadOnlyViolation,
)
from nexusai.infrastructure.llm.provider import (
    LLMProviderRegistry,
    synthesize_extractive_grounded_answer,
)

logger = get_logger(__name__)

_CITATION_TAG_RE = re.compile(r"\[(KB-\d+|WEB-\d+)\]")

GROUNDED_SYSTEM_TEMPLATE = """You are NexusAI, an expert AI assistant, enterprise software architect, and precision knowledge analyst.
Your mission is to provide thorough, comprehensive, and highly detailed answers to clarify all relevant doubts of the user.

Key instructions:
1. When retrieved evidence from documents or knowledge base is provided:
   Provide an in-depth, structured, and informative answer addressing the user's specific questions and technical doubts completely.
   Ground factual claims in the retrieved chunks and cite them inline using [KB-1], [KB-2], etc.
   Provide rich explanations, context, architecture concepts, and actionable insights.
2. When dataset or spreadsheet statistical profile context is provided:
   Provide exhaustive mathematical and analytical detail using the exact descriptive statistics (counts, means, std, min/max, percentiles), categorical distributions, and trend numbers.
   Highlight period-over-period growth rates, correlation relationships, data quality (missing values/duplicates), and actionable business recommendations.
3. When no specific private knowledge chunks match or evidence is sparse:
   Thoroughly answer the user's doubt, query, or conceptual topic using comprehensive technical knowledge.
   Explain the underlying mechanics, best practices, and practical guidance in detail.
   Do not invent fake [KB-n] tags if no private documents were cited.
4. Formatting Rules:
   Write in clear, structured, natural prose with rich paragraphs.
   Do NOT use special markdown symbols such as hash headings (###), bold asterisks (**), italics, bullet asterisks (*), or horizontal dividers (---).
   Use clear topic opening sentences, capitalized terms, and clean paragraph breaks so the response is completely clear and legible. Preserve valid bracketed citations like [KB-1].

{conversation_history_block}
Retrieved Evidence:
{assembled_context}"""

GROUNDED_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", GROUNDED_SYSTEM_TEMPLATE),
        ("human", "{question}"),
    ]
)


def strip_special_symbols(text: str) -> str:
    """Remove special markdown symbols (#, ##, ###, **, *, ---, bullets, backticks) while preserving [KB-x] citations."""
    if not text:
        return ""
    cleaned = text
    # 1. Remove markdown headings #, ##, ###, etc.
    cleaned = re.sub(r"^#{1,6}\s+", "", cleaned, flags=re.MULTILINE)
    # 2. Remove horizontal rules ---, ***, ___
    cleaned = re.sub(r"^[-\*_]{3,}\s*$", "", cleaned, flags=re.MULTILINE)
    # 3. Remove bold / italics **text**, *text*, __text__, _text_
    cleaned = re.sub(r"\*\*([^*]+)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"__([^_]+)__", r"\1", cleaned)
    cleaned = re.sub(r"(?<!\w)\*([^*]+)\*(?!\w)", r"\1", cleaned)
    cleaned = re.sub(r"(?<!\w)_([^_]+)_(?!\w)", r"\1", cleaned)
    # 4. Remove code fence blocks and inline backticks
    cleaned = re.sub(r"```[a-zA-Z0-9_-]*\n([\s\S]*?)```", r"\1", cleaned)
    cleaned = re.sub(r"`([^`]+)`", r"\1", cleaned)
    # 5. Remove bullet markers * , - , + , •
    cleaned = re.sub(r"^[\s]*[\*\-\+•⁃]\s+", "", cleaned, flags=re.MULTILINE)
    # 6. Remove blockquotes >
    cleaned = re.sub(r"^>\s+", "", cleaned, flags=re.MULTILINE)
    # 7. Collapse excessive blank lines
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


def validate_and_clean_citations(
    answer_text: str,
    citations: list[dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    """Strip any fabricated `[KB-x]` / `[WEB-x]` tags and clean special markdown symbols."""
    valid_ids = {str(c.get("citation_id")) for c in citations if c.get("citation_id")}
    found_tags = _CITATION_TAG_RE.findall(answer_text or "")
    invalid_tags = sorted({t for t in found_tags if t not in valid_ids})
    used_valid_tags = sorted({t for t in found_tags if t in valid_ids})

    cleaned = answer_text or ""
    for bad in invalid_tags:
        cleaned = cleaned.replace(f"[{bad}]", "")
    cleaned = re.sub(r"  +", " ", cleaned).strip()

    # Strip special symbols from generated prose
    cleaned = strip_special_symbols(cleaned)

    # If the model omitted inline tags despite using valid citations, append grounded reference footer
    if citations and not used_valid_tags and cleaned:
        top_tags = [str(c["citation_id"]) for c in citations[:3] if c.get("citation_id")]
        if top_tags:
            used_valid_tags = top_tags

    report = {
        "valid_citation_ids": sorted(valid_ids),
        "cited_in_answer": used_valid_tags,
        "stripped_fabricated_citations": invalid_tags,
        "grounding_verified": len(invalid_tags) == 0,
    }
    return cleaned, report


@dataclass
class AssistantTurnResult:
    """Complete response from `NexusSingleAgentAssistant`."""

    request_id: str
    conversation_id: str | None
    answer: str
    citations: list[dict[str, Any]]
    diagnostics: dict[str, Any]
    provider_used: str
    model_used: str
    citation_validation: dict[str, Any]
    chart: dict[str, Any] | None = None


class NexusSingleAgentAssistant:
    """Single-agent LangChain assistant orchestrating hybrid retrieval, tools, memory, & grounded generation."""

    def __init__(
        self,
        retriever: HybridRetrievalEngine,
        llm_registry: LLMProviderRegistry,
        knowledge_repo: KnowledgeRepositoryPort,
        readonly_graph_store: Neo4jReadOnlyStore | None = None,
    ) -> None:
        self._retriever = retriever
        self._llm_registry = llm_registry
        self._repo = knowledge_repo
        self._readonly_graph = readonly_graph_store
        self._tools = self._build_langchain_tools()

    @property
    def tools(self) -> list[StructuredTool]:
        """Expose modular LangChain tools registered on this single-agent assistant."""
        return list(self._tools)

    def _build_langchain_tools(self) -> list[StructuredTool]:
        """Register the 7 modular LangChain tools on the single-agent assistant."""

        async def _document_search(query: str, top_k: int = 6) -> list[dict[str, Any]]:
            """Hybrid search (dense + sparse BM25 + graph RRF + reranking) across indexed documents."""
            res = await self._retriever.retrieve(query=query, top_k=top_k, mode="hybrid")
            return res.citations

        async def _vector_search(query: str, top_k: int = 6) -> list[dict[str, Any]]:
            """Dense semantic BGE-M3 vector search in Qdrant."""
            res = await self._retriever.retrieve(query=query, top_k=top_k, mode="dense", enable_reranking=False)
            return res.citations

        async def _graph_search(query: str, top_k: int = 8) -> dict[str, Any]:
            """Entity & relationship neighborhood retrieval from Neo4j."""
            res = await self._retriever.retrieve(query=query, top_k=top_k, mode="graph")
            return {
                "graph_status": res.graph_status,
                "entities": res.graph_entities,
                "chunks": res.citations,
            }

        async def _cypher_query(cypher: str) -> dict[str, Any]:
            """Execute a safeguarded read-only Cypher query via Neo4jReadOnlyStore."""
            try:
                store = self._readonly_graph or Neo4jReadOnlyStore()
                res = await store.run_read_only(cypher)
                return {"status": res.status, "rows": res.records, "row_count": res.row_count}
            except (ReadOnlyStoreUnavailable, ReadOnlyViolation) as exc:
                return {"status": "rejected", "detail": str(exc)}

        async def _web_search(query: str) -> dict[str, Any]:
            """External web research via Tavily (kept strictly separate from private KB)."""
            res = await self._retriever.retrieve(query=query, top_k=5, mode="web", include_web=True)
            return {"web_status": res.web_status, "results": res.web_results}

        async def _document_summary(document_id: str) -> dict[str, Any]:
            """Retrieve metadata and ordered excerpt overview for a specific document_id."""
            src = await self._repo.get_source(UUID(document_id))
            if not src:
                return {"status": "not_found", "document_id": document_id}
            chunks: list[dict[str, Any]] = []
            if hasattr(self._retriever._vector_store, "get_document_chunks"):
                chunks = await getattr(self._retriever._vector_store, "get_document_chunks")(document_id, limit=8)
            return {
                "status": "ok",
                "document_id": str(src.id),
                "title": src.title,
                "source_type": getattr(src.source_type, "value", str(src.source_type)),
                "chunk_count": src.chunk_count,
                "preview_chunks": [c.get("text", "")[:300] for c in chunks[:5]],
            }

        async def _source_lookup(document_id: str, chunk_id: str | None = None) -> dict[str, Any]:
            """Inspect exact chunk text and lineage metadata for citation verification."""
            if hasattr(self._retriever._vector_store, "get_document_chunks"):
                chunks = await getattr(self._retriever._vector_store, "get_document_chunks")(document_id, limit=200)
                if chunk_id:
                    for ch in chunks:
                        if str(ch.get("chunk_id")) == str(chunk_id):
                            return {"status": "ok", "chunk": ch}
                return {"status": "ok", "chunks": chunks[:10]}
            return {"status": "unavailable"}

        async def _dataset_analytics_profiler(dataset_query: str) -> dict[str, Any]:
            """Compute statistical profiles, descriptive stats, correlations, and trends across CSV/Excel datasets."""
            profiler = get_dataset_profiler()
            profile_text = profiler.find_and_profile_relevant_datasets(dataset_query)
            return {"status": "ok" if profile_text else "no_matching_datasets", "profile": profile_text}

        return [
            StructuredTool.from_function(coroutine=_document_search, name="document_search", description="Hybrid dense+sparse+graph search over private indexed documents."),
            StructuredTool.from_function(coroutine=_vector_search, name="vector_search", description="Dense semantic vector similarity search in Qdrant."),
            StructuredTool.from_function(coroutine=_graph_search, name="graph_search", description="Knowledge graph entity and relationship search in Neo4j."),
            StructuredTool.from_function(coroutine=_cypher_query, name="cypher_query", description="Safeguarded read-only Cypher query against Neo4j."),
            StructuredTool.from_function(coroutine=_web_search, name="web_search", description="External web search using Tavily."),
            StructuredTool.from_function(coroutine=_document_summary, name="document_summary", description="Fetch summary metadata and opening chunks of an indexed document."),
            StructuredTool.from_function(coroutine=_source_lookup, name="source_lookup", description="Look up exact source chunk content and page metadata."),
            StructuredTool.from_function(coroutine=_dataset_analytics_profiler, name="dataset_analytics_profiler", description="Compute descriptive statistics, means, percentiles, and trends for tabular datasets."),
        ]

    async def _load_conversation_context(
        self,
        conversation_id: UUID | None,
        current_query: str,
    ) -> tuple[str, str]:
        """Load recent turns and enrich ambiguous follow-up pronouns for retrieval without losing original question."""
        if conversation_id is None:
            return "", current_query
        conv = await self._repo.get_conversation(conversation_id)
        if not conv or not conv.messages:
            return "", current_query

        recent = conv.messages[-6:]
        lines = ["Conversation History (for multi-turn continuity only — still ground facts in Retrieved Evidence):"]
        last_user_queries: list[str] = []
        for msg in recent:
            role_label = "User" if msg.role == ConversationRole.USER else "Assistant"
            lines.append(f"{role_label}: {msg.content[:500]}")
            if msg.role == ConversationRole.USER and msg.content.strip():
                last_user_queries.append(msg.content.strip())

        # Contextualize short follow-up questions ("What about its latency?", "And on page 2?")
        words = current_query.strip().split()
        pronouns = {"it", "its", "they", "their", "them", "that", "those", "this", "these"}
        if last_user_queries and (len(words) <= 7 or any(w.lower().strip("?.,") in pronouns for w in words)):
            retrieval_query = f"{last_user_queries[-1]} — Follow-up: {current_query}"
        else:
            retrieval_query = current_query

        return "\n".join(lines) + "\n", retrieval_query

    @traced_step(name="nexus_single_agent_answer", run_type="chain")
    async def answer(
        self,
        query: str,
        user_id: UUID,
        conversation_id: UUID | None = None,
        top_k: int | None = None,
        mode: str | None = None,
        fusion_strategy: str | None = None,
        enable_reranking: bool | None = None,
        include_web: bool = False,
        provider: str | None = None,
        model: str | None = None,
        filter_metadata: dict[str, Any] | None = None,
        persist_messages: bool = True,
        llm_override: LLMPort | None = None,
    ) -> AssistantTurnResult:
        """Run end-to-end single-agent retrieval, grounded generation, citation validation, and persistence."""
        t0 = time.perf_counter()
        req_id = set_request_id(get_request_id())

        history_block, retrieval_query = await self._load_conversation_context(conversation_id, query)

        retrieval: HybridRetrievalResult = await self._retriever.retrieve(
            query=retrieval_query,
            top_k=top_k,
            mode=mode,
            fusion_strategy=fusion_strategy,
            enable_reranking=enable_reranking,
            include_web=include_web,
            filter_metadata=filter_metadata,
        )

        latencies = dict(retrieval.diagnostics.get("latency_ms") or {})
        candidates: list[tuple[str, LLMPort]] = (
            [("custom", llm_override)]
            if llm_override is not None
            else self._llm_registry.resolve_candidates(provider)
        )
        provider_name, resolved_llm = (
            candidates[0] if candidates else (provider or "gemini", None)
        )
        model_used = model or getattr(resolved_llm, "model_name", "extractive-grounded-v1")
        had_llm_error = False

        if not candidates:
            with StepTimer(latencies, "llm_ms"):
                raw_answer = synthesize_extractive_grounded_answer(
                    query=query,
                    citations=retrieval.citations,
                    graph_entities=retrieval.graph_entities,
                    provider_notice="No external LLM API key is configured in `.env`; showing verified extractive synthesis from retrieved sources",
                )
            cleaned_answer, val_report = validate_and_clean_citations(raw_answer, retrieval.citations)
            provider_used = "extractive_fallback"
            model_used = "extractive-grounded-v1"
        else:
            context_to_use = (
                retrieval.assembled_context
                if retrieval.has_sufficient_evidence
                else "No matching private workspace documents found for this query. Provide a comprehensive, detailed, and highly clear technical explanation answering the user's doubt and question directly."
            )
            # Enrich context with high-density dataset statistical profiles if query or retrieved evidence touches tabular data
            try:
                dataset_profile = get_dataset_profiler().find_and_profile_relevant_datasets(query)
                if dataset_profile:
                    context_to_use = f"{context_to_use}\n\n{dataset_profile}"
            except Exception as prof_err:
                logger.debug("Dataset profiling pass notice: %s", prof_err)
            formatted_msgs = GROUNDED_PROMPT.format_messages(
                conversation_history_block=history_block,
                assembled_context=context_to_use,
                question=query,
            )
            sys_text = str(formatted_msgs[0].content)
            human_text = str(formatted_msgs[1].content)
            raw_answer_out: str | None = None
            last_exc: Exception | None = None

            with StepTimer(latencies, "llm_ms"):
                for idx_cand, (cand_name, cand_llm) in enumerate(candidates):
                    try:
                        req_model = model if idx_cand == 0 else None
                        if req_model and hasattr(cand_llm, "generate"):
                            try:
                                raw_answer_out = await cand_llm.generate(  # type: ignore[call-arg]
                                    prompt=human_text,
                                    system_prompt=sys_text,
                                    temperature=0.1,
                                    model=req_model,
                                )
                            except TypeError:
                                raw_answer_out = await cand_llm.generate(
                                    prompt=human_text,
                                    system_prompt=sys_text,
                                    temperature=0.1,
                                )
                        else:
                            raw_answer_out = await cand_llm.generate(
                                prompt=human_text,
                                system_prompt=sys_text,
                                temperature=0.1,
                            )
                        provider_used = cand_name
                        resolved_llm = cand_llm
                        usage_obj = getattr(cand_llm, "last_usage", None)
                        model_used = (
                            getattr(usage_obj, "model", None)
                            or req_model
                            or getattr(cand_llm, "model_name", model_used)
                        )
                        break
                    except Exception as exc:
                        last_exc = exc
                        logger.warning(
                            "LLM provider '%s' failed (%s); %s",
                            cand_name,
                            exc,
                            "failing over to next configured LLM provider"
                            if idx_cand + 1 < len(candidates)
                            else "degrading to extractive grounded answer",
                        )

            if raw_answer_out is not None:
                cleaned_answer, val_report = validate_and_clean_citations(raw_answer_out, retrieval.citations)
            else:
                had_llm_error = True
                fallback_answer = synthesize_extractive_grounded_answer(
                    query=query,
                    citations=retrieval.citations,
                    graph_entities=retrieval.graph_entities,
                    provider_notice=f"LLM provider '{provider_name}' encountered an error ({last_exc}); showing grounded extractive evidence",
                )
                cleaned_answer, val_report = validate_and_clean_citations(fallback_answer, retrieval.citations)
                provider_used = f"{provider_name}_degraded_extractive"

        total_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        latencies["total_e2e_ms"] = total_ms
        usage = getattr(resolved_llm, "last_usage", None)
        p_toks = int(getattr(usage, "prompt_tokens", 0) or 0)
        c_toks = int(getattr(usage, "completion_tokens", 0) or 0)

        eval_metrics = compute_live_turn_eval_metrics(
            query=query,
            answer=cleaned_answer,
            citations=retrieval.citations,
            retrieval=retrieval,
            citation_validation=val_report,
            provider_used=provider_used,
            had_llm_error=had_llm_error,
        )
        schedule_langsmith_eval_log(
            request_id=req_id,
            query=query,
            answer=cleaned_answer,
            eval_metrics=eval_metrics,
        )

        diagnostics = {
            **retrieval.diagnostics,
            "provider_used": provider_used,
            "model_used": model_used,
            "prompt_tokens": p_toks,
            "completion_tokens": c_toks,
            "citation_validation": val_report,
            "eval_metrics": eval_metrics,
            "latency_ms": latencies,
        }

        if persist_messages and conversation_id is not None:
            try:
                conv = await self._repo.get_conversation(conversation_id)
                if conv is not None:
                    await self._repo.add_message(
                        Message(
                            conversation_id=conversation_id,
                            role=ConversationRole.USER,
                            content=query,
                            citations=[],
                            metadata={"request_id": req_id},
                        )
                    )
                    await self._repo.add_message(
                        Message(
                            conversation_id=conversation_id,
                            role=ConversationRole.ASSISTANT,
                            content=cleaned_answer,
                            citations=retrieval.citations,
                            metadata={"diagnostics": diagnostics},
                        )
                    )
            except Exception as exc:
                logger.warning("Failed to persist conversation messages for %s: %s", conversation_id, exc)

        get_observability_tracker().record(
            TraceTelemetryRecord(
                request_id=req_id,
                operation="assistant_chat",
                status="ok",
                query=query[:240],
                retrieval_mode=retrieval.mode,
                fusion_strategy=retrieval.fusion_strategy,
                reranked=retrieval.reranked,
                dense_count=int(retrieval.diagnostics.get("dense_candidates", 0)),
                sparse_count=int(retrieval.diagnostics.get("sparse_candidates", 0)),
                graph_count=int(retrieval.diagnostics.get("graph_candidates", 0)),
                web_count=len(retrieval.web_results),
                final_count=len(retrieval.chunks),
                graph_status=retrieval.graph_status,
                llm_provider=provider_used,
                llm_model=model_used,
                prompt_tokens=p_toks,
                completion_tokens=c_toks,
                latency_ms=latencies,
                total_ms=total_ms,
            )
        )

        chart_spec = get_dataset_profiler().generate_chart_spec(query)

        return AssistantTurnResult(
            request_id=req_id,
            conversation_id=str(conversation_id) if conversation_id else None,
            answer=cleaned_answer,
            citations=retrieval.citations,
            diagnostics=diagnostics,
            provider_used=provider_used,
            model_used=model_used,
            citation_validation=val_report,
            chart=chart_spec,
        )

    @traced_step(name="nexus_single_agent_stream", run_type="chain")
    async def stream_answer(
        self,
        query: str,
        user_id: UUID,
        conversation_id: UUID | None = None,
        top_k: int | None = None,
        mode: str | None = None,
        fusion_strategy: str | None = None,
        enable_reranking: bool | None = None,
        include_web: bool = False,
        provider: str | None = None,
        model: str | None = None,
        filter_metadata: dict[str, Any] | None = None,
        persist_messages: bool = True,
        llm_override: LLMPort | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Stream `sources`, `data` deltas, and `done` (with validated answer, citations, & diagnostics)."""
        t0 = time.perf_counter()
        req_id = set_request_id(get_request_id())
        history_block, retrieval_query = await self._load_conversation_context(conversation_id, query)

        retrieval = await self._retriever.retrieve(
            query=retrieval_query,
            top_k=top_k,
            mode=mode,
            fusion_strategy=fusion_strategy,
            enable_reranking=enable_reranking,
            include_web=include_web,
            filter_metadata=filter_metadata,
        )

        yield {
            "event": "sources",
            "data": {
                "request_id": req_id,
                "sources": retrieval.citations,
                "citations": retrieval.citations,
                "diagnostics": retrieval.diagnostics,
            },
        }

        latencies = dict(retrieval.diagnostics.get("latency_ms") or {})
        candidates = (
            [("custom", llm_override)]
            if llm_override is not None
            else self._llm_registry.resolve_candidates(provider)
        )
        provider_name, resolved_llm = (
            candidates[0] if candidates else (provider or "gemini", None)
        )
        model_used = model or getattr(resolved_llm, "model_name", "extractive-grounded-v1")
        had_llm_error = False

        full_answer = ""
        llm_start = time.perf_counter()

        try:
            if not candidates:
                full_answer = synthesize_extractive_grounded_answer(
                    query=query,
                    citations=retrieval.citations,
                    graph_entities=retrieval.graph_entities,
                    provider_notice="No external LLM API key is configured in `.env`; showing verified extractive synthesis from retrieved sources",
                )
                provider_used = "extractive_fallback"
                # Stream smoothly in chunks for responsive UI rendering
                for start_i in range(0, len(full_answer), 48):
                    yield {"event": "data", "data": {"delta": full_answer[start_i : start_i + 48]}}
            else:
                context_to_use = (
                    retrieval.assembled_context
                    if retrieval.has_sufficient_evidence
                    else "No matching private workspace documents found for this query. Provide an extensive, clear, comprehensive, and developer-grade answer answering the user's doubt and question directly."
                )
                formatted_msgs = GROUNDED_PROMPT.format_messages(
                    conversation_history_block=history_block,
                    assembled_context=context_to_use,
                    question=query,
                )
                sys_text = str(formatted_msgs[0].content)
                human_text = str(formatted_msgs[1].content)
                provider_used = provider_name
                streamed_ok = False
                last_stream_exc: Exception | None = None

                for idx_cand, (cand_name, cand_llm) in enumerate(candidates):
                    try:
                        provider_used = cand_name
                        model_used = (model if idx_cand == 0 else None) or getattr(cand_llm, "model_name", model_used)
                        async for delta in cand_llm.stream(
                            prompt=human_text,
                            system_prompt=sys_text,
                            temperature=0.1,
                        ):
                            full_answer += delta
                            yield {"event": "data", "data": {"delta": delta}}
                        if full_answer.strip():
                            streamed_ok = True
                            break
                    except Exception as stream_exc:
                        last_stream_exc = stream_exc
                        logger.warning(
                            "Streaming LLM '%s' error (%s); %s",
                            cand_name,
                            stream_exc,
                            "failing over to next configured LLM"
                            if (not full_answer and idx_cand + 1 < len(candidates))
                            else "falling back to extractive answer",
                        )
                        if full_answer:
                            # Already streamed partial tokens; avoid duplicating stream
                            streamed_ok = True
                            break

                if not streamed_ok and not full_answer:
                    had_llm_error = True
                    full_answer = synthesize_extractive_grounded_answer(
                        query=query,
                        citations=retrieval.citations,
                        graph_entities=retrieval.graph_entities,
                        provider_notice=f"LLM provider '{provider_name}' unavailable ({last_stream_exc}); showing grounded extractive evidence",
                    )
                    provider_used = f"{provider_name}_degraded_extractive"
                    yield {"event": "data", "data": {"delta": full_answer}}
        except BaseException as exc:
            # Handle client disconnect (asyncio.CancelledError / GeneratorExit) cleanly
            logger.info("Stream interrupted for request_id=%s (%s)", req_id, type(exc).__name__)
            raise
        finally:
            latencies["llm_ms"] = round((time.perf_counter() - llm_start) * 1000.0, 2)

        cleaned_answer, val_report = validate_and_clean_citations(full_answer, retrieval.citations)
        latencies["total_e2e_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)

        eval_metrics = compute_live_turn_eval_metrics(
            query=query,
            answer=cleaned_answer,
            citations=retrieval.citations,
            retrieval=retrieval,
            citation_validation=val_report,
            provider_used=provider_used,
            had_llm_error=had_llm_error,
        )
        schedule_langsmith_eval_log(
            request_id=req_id,
            query=query,
            answer=cleaned_answer,
            eval_metrics=eval_metrics,
        )

        diagnostics = {
            **retrieval.diagnostics,
            "provider_used": provider_used,
            "model_used": model_used,
            "citation_validation": val_report,
            "eval_metrics": eval_metrics,
            "latency_ms": latencies,
        }

        if persist_messages and conversation_id is not None:
            try:
                conv = await self._repo.get_conversation(conversation_id)
                if conv is not None:
                    await self._repo.add_message(
                        Message(
                            conversation_id=conversation_id,
                            role=ConversationRole.USER,
                            content=query,
                            citations=[],
                            metadata={"request_id": req_id},
                        )
                    )
                    await self._repo.add_message(
                        Message(
                            conversation_id=conversation_id,
                            role=ConversationRole.ASSISTANT,
                            content=cleaned_answer,
                            citations=retrieval.citations,
                            metadata={"diagnostics": diagnostics},
                        )
                    )
            except Exception as exc:
                logger.warning("Failed to persist streamed conversation messages: %s", exc)

        chart_spec = get_dataset_profiler().generate_chart_spec(query)

        yield {
            "event": "done",
            "data": {
                "request_id": req_id,
                "conversation_id": str(conversation_id) if conversation_id else None,
                "answer": cleaned_answer,
                "citations": retrieval.citations,
                "diagnostics": diagnostics,
                "chart": chart_spec,
            },
        }
