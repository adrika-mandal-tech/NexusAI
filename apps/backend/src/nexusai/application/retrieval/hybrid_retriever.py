"""Complete Hybrid Retrieval Engine: Query Understanding, Dense, Sparse/BM25, Neo4j Graph, Fusion, Reranking, & Context Assembly."""

from __future__ import annotations

import asyncio
import math
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Literal
from uuid import UUID

from nexusai.core.config import get_settings
from nexusai.core.logging import get_logger
from nexusai.core.observability import (
    StepTimer,
    TraceTelemetryRecord,
    get_observability_tracker,
    get_request_id,
    traced_step,
)
from nexusai.domain.ports.embedding_port import EmbeddingPort
from nexusai.domain.ports.graph_store import GraphStorePort
from nexusai.domain.ports.vector_store import VectorStorePort
from nexusai.infrastructure.embedding.provider import build_lexical_sparse_vector
from nexusai.infrastructure.parsers.chunker import estimate_token_count
from nexusai.infrastructure.search.tavily_service import (
    TavilySearchService,
    sanitize_untrusted_text,
)

logger = get_logger(__name__)

RetrievalMode = Literal["hybrid", "dense", "sparse", "graph", "web"]
FusionStrategy = Literal["rrf", "normalized_linear", "qdrant_native_rrf"]

_ACRONYM_EXPANSIONS: dict[str, str] = {
    "rag": "retrieval augmented generation",
    "kg": "knowledge graph",
    "llm": "large language model",
    "ocr": "optical character recognition",
    "rrf": "reciprocal rank fusion",
    "bm25": "best matching 25 lexical ranking",
    "ecc": "error correcting code memory",
    "sm": "streaming multiprocessor",
    "simt": "single instruction multiple threads",
    "pcie": "peripheral component interconnect express bus",
    "dma": "direct memory access",
    "cuda": "compute unified device architecture",
    "vram": "video random access memory",
    "fk": "foreign key relationship",
    "pk": "primary key identifier",
    "pii": "personally identifiable information",
    "sse": "server sent events streaming",
    "jwt": "json web token authentication",
    "api": "application programming interface",
}

_STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does",
        "for", "from", "how", "i", "in", "is", "it", "of", "on", "or", "that",
        "the", "their", "this", "to", "was", "what", "when", "where", "which",
        "who", "why", "with", "about", "explain", "describe", "tell", "me",
    }
)

# Conversational / meta-document words that must NOT be used as Neo4j entity substring filters
_GENERIC_ENTITY_STOPWORDS = frozenset(
    {
        "analyze", "analysis", "summarize", "summary", "document", "documents",
        "file", "files", "content", "contents", "paper", "report", "guide",
        "page", "pages", "section", "sections", "chapter", "please", "give",
        "show", "list", "detail", "details", "main", "point", "points",
        "overview", "takeaway", "takeaways", "information", "explain", "describe",
        "question", "answer", "according", "uploaded", "indexed", "text", "data",
    }
)

_OVERVIEW_QUERY_RE = re.compile(
    r"\b(?:summarize|summary|overview|analyze|analysis|main\s+points|key\s+(?:points|takeaways|findings)|"
    r"what\s+is\s+this\s+(?:document|file|pdf|report|paper|guide)\s+about|explain\s+this\s+(?:document|file|pdf))\b",
    re.IGNORECASE,
)

_LEX_TOKEN_RE = re.compile(r"[A-Za-z0-9_+#.-]{2,}")
_QUOTED_RE = re.compile(r'"([^"]{2,64})"|\'([^\']{2,64})\'')
_CAMEL_OR_ACRONYM_RE = re.compile(r"\b(?:[A-Z]{2,}[A-Za-z0-9_-]*|[A-Z][a-z]+(?:[A-Z][A-Za-z0-9]+)+|[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,}){0,2})\b")
_WEB_HINT_RE = re.compile(
    r"\b(?:latest\s+news|current\s+events|search\s+the\s+web|on\s+the\s+internet|today'?s|202[567]\s+release)\b",
    re.IGNORECASE,
)
_GRAPH_HINT_RE = re.compile(
    r"\b(?:relationship|connected\s+to|supplied\s+by|placed\s+by|orders?\s+for|customers?\s+in|suppliers?\s+of|graph|neo4j|entity|entities)\b",
    re.IGNORECASE,
)
_CONVERSATIONAL_GREETING_RE = re.compile(
    r"^\s*(?:hi+|hello+|hey+|good\s+(?:morning|afternoon|evening)|how\s+are\s+you|who\s+are\s+you|"
    r"what\s+can\s+you\s+do|thanks?(?:\s+you)?|what(?:'?s|\s+is)\s+(?:up|wrong)|ok(?:ay)?)[!.?\s]*$",
    re.IGNORECASE,
)


def tokenize_lexical(text: str) -> list[str]:
    """Tokenize preserving technical identifiers like `cudaMemcpyAsync`, `BGE-M3`, `C++`, `PP-OCRv6`."""
    if not text:
        return []
    raw = _LEX_TOKEN_RE.findall(text)
    tokens: list[str] = []
    for tok in raw:
        low = tok.lower().strip(".-")
        if len(low) >= 2 and low not in _STOPWORDS:
            tokens.append(low)
            # Also split camelCase / snake_case / hyphenated technical tokens additively
            parts = re.findall(r"[A-Z]?[a-z0-9]+|[A-Z]+(?=[A-Z]|$)", tok)
            if len(parts) > 1:
                for p in parts:
                    pl = p.lower()
                    if len(pl) >= 2 and pl not in _STOPWORDS and pl != low:
                        tokens.append(pl)
    return tokens


@dataclass
class QueryAnalysis:
    """Output of query understanding stage."""

    original_query: str
    normalized_query: str
    semantic_query: str
    expanded_lexical_query: str
    query_tokens: list[str]
    expanded_acronyms: dict[str, str] = field(default_factory=dict)
    extracted_entities: list[str] = field(default_factory=list)
    sub_queries: list[str] = field(default_factory=list)
    recommended_mode: RetrievalMode = "hybrid"


def understand_query(
    query: str,
    requested_mode: str | None = None,
    include_web: bool = False,
) -> QueryAnalysis:
    """Normalize query, expand technical acronyms additively, and extract graph entities."""
    original = query or ""
    normalized = re.sub(r"\s+", " ", original).strip()

    # Preserve original query for semantic dense search (do not rewrite simple queries!)
    semantic_query = normalized

    # Additive acronym expansion for lexical/sparse/BM25 retrieval
    words = [w.strip(".,;:?!()[]{}\"'").lower() for w in normalized.split()]
    expanded_acronyms: dict[str, str] = {}
    expansion_phrases: list[str] = []
    for w in words:
        if w in _ACRONYM_EXPANSIONS and w not in expanded_acronyms:
            expanded_acronyms[w] = _ACRONYM_EXPANSIONS[w]
            expansion_phrases.append(_ACRONYM_EXPANSIONS[w])

    expanded_lexical = (
        f"{normalized} {' '.join(expansion_phrases)}".strip()
        if expansion_phrases
        else normalized
    )

    # Extract entity mentions for Neo4j graph retrieval
    entities: list[str] = []
    seen_ent: set[str] = set()

    def _add_ent(val: str) -> None:
        v = val.strip()
        low = v.lower()
        if (
            len(v) >= 2
            and low not in _STOPWORDS
            and low not in _GENERIC_ENTITY_STOPWORDS
            and low not in seen_ent
        ):
            seen_ent.add(low)
            entities.append(v)

    for m in _QUOTED_RE.finditer(normalized):
        _add_ent(m.group(1) or m.group(2) or "")
    for m in _CAMEL_OR_ACRONYM_RE.finditer(normalized):
        _add_ent(m.group(0))
    # Fallback: add top distinctive non-stopword tokens >= 4 chars if fewer than 3 entities
    for tok in tokenize_lexical(normalized):
        if len(entities) >= 8:
            break
        if len(tok) >= 4:
            _add_ent(tok)

    # Multi-part sub-query decomposition only for genuinely compound questions
    sub_queries: list[str] = []
    q_parts = [p.strip() for p in re.split(r"\?\s+|\b(?:and also|as well as|versus|vs\.)\b", normalized, flags=re.IGNORECASE) if len(p.strip()) >= 12]
    if len(q_parts) > 1:
        sub_queries = q_parts[:3]

    valid_modes = {"hybrid", "dense", "sparse", "graph", "web"}
    if requested_mode and requested_mode.lower() in valid_modes:
        rec_mode: RetrievalMode = requested_mode.lower()  # type: ignore[assignment]
    elif requested_mode and requested_mode.lower() == "vector":
        rec_mode = "dense"
    elif include_web and _WEB_HINT_RE.search(normalized) and not entities:
        rec_mode = "web"
    else:
        rec_mode = "hybrid"

    return QueryAnalysis(
        original_query=original,
        normalized_query=normalized,
        semantic_query=semantic_query,
        expanded_lexical_query=expanded_lexical,
        query_tokens=tokenize_lexical(expanded_lexical),
        expanded_acronyms=expanded_acronyms,
        extracted_entities=entities[:8],
        sub_queries=sub_queries,
        recommended_mode=rec_mode,
    )


class BM25LexicalScorer:
    """Okapi BM25 ($k_1=1.5, b=0.75$) + exact technical phrase/acronym matcher over candidate chunks."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b

    def score_documents(
        self,
        query: str,
        query_tokens: list[str],
        candidates: list[dict[str, Any]],
    ) -> list[float]:
        if not candidates or not query_tokens:
            return [0.0 for _ in candidates]

        doc_tokens_list: list[list[str]] = []
        df: Counter[str] = Counter()
        unique_q = list(dict.fromkeys(query_tokens))

        for item in candidates:
            header = f"{item.get('filename', '')} {item.get('section_title', '')}"
            toks = tokenize_lexical(f"{header} {item.get('text', '')}")
            doc_tokens_list.append(toks)
            for ut in set(toks):
                df[ut] += 1

        n_docs = len(candidates)
        avg_dl = sum(len(t) for t in doc_tokens_list) / max(1, n_docs)
        raw_q_lower = query.lower().strip()

        scores: list[float] = []
        for idx, toks in enumerate(doc_tokens_list):
            if not toks:
                scores.append(0.0)
                continue
            tf = Counter(toks)
            dl = len(toks)
            score = 0.0
            for qt in unique_q:
                f = tf.get(qt, 0)
                if f == 0:
                    continue
                n_q = df.get(qt, 0)
                idf = math.log(1.0 + (n_docs - n_q + 0.5) / (n_q + 0.5))
                denom = f + self.k1 * (1.0 - self.b + self.b * (dl / max(1.0, avg_dl)))
                score += idf * ((f * (self.k1 + 1.0)) / max(1e-9, denom))

            # Exact technical phrase / rare identifier bonus
            text_low = str(candidates[idx].get("text", "")).lower()
            if len(raw_q_lower) >= 4 and raw_q_lower in text_low:
                score *= 1.25
            scores.append(round(score, 6))
        return scores


class HybridReranker:
    """Reranks fused candidates using calibrated lexical/phrase/coverage cross-features (and optional local CrossEncoder)."""

    def __init__(self, cross_encoder_model: str | None = None) -> None:
        self._cross_encoder_name = cross_encoder_model
        self._cross_encoder: Any | None = None
        self._cross_encoder_checked = False
        self._bm25 = BM25LexicalScorer()

    def _get_optional_cross_encoder(self) -> Any | None:
        if self._cross_encoder_checked:
            return self._cross_encoder
        self._cross_encoder_checked = True
        if not self._cross_encoder_name:
            return None
        try:
            from sentence_transformers import CrossEncoder

            self._cross_encoder = CrossEncoder(
                self._cross_encoder_name,
                local_files_only=True,
            )
            logger.info("Loaded local CrossEncoder reranker '%s'", self._cross_encoder_name)
        except Exception as exc:
            logger.info("Local CrossEncoder '%s' not cached (%s); using calibrated feature reranker", self._cross_encoder_name, exc)
            self._cross_encoder = None
        return self._cross_encoder

    def rerank(
        self,
        analysis: QueryAnalysis,
        candidates: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Rerank fused candidates while preserving `dense_score`, `sparse_score`, and `fusion_score`."""
        if not candidates:
            return []

        ce = self._get_optional_cross_encoder()
        if ce is not None:
            pairs = [(analysis.original_query, str(c.get("text", ""))[:1500]) for c in candidates]
            raw_ce_scores = ce.predict(pairs)
            reranked = []
            for cand, sc in zip(candidates, raw_ce_scores):
                updated = dict(cand)
                updated["rerank_score"] = round(float(sc), 6)
                reranked.append(updated)
            reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
            for i, item in enumerate(reranked[:top_k], start=1):
                item["final_rank"] = i
            return reranked[:top_k]

        # Deterministic calibrated cross-feature reranker (zero extra model download overhead)
        orig_tokens = tokenize_lexical(analysis.original_query)
        bm25_scores = self._bm25.score_documents(
            analysis.original_query,
            analysis.query_tokens or orig_tokens,
            candidates,
        )
        max_bm25 = max(bm25_scores) if bm25_scores and max(bm25_scores) > 0 else 1.0
        max_fusion = max((float(c.get("fusion_score") or 0.0) for c in candidates), default=1.0) or 1.0

        orig_bigrams = [
            f"{orig_tokens[i]} {orig_tokens[i + 1]}"
            for i in range(len(orig_tokens) - 1)
        ]

        reranked_items: list[dict[str, Any]] = []
        for cand, bm25_val in zip(candidates, bm25_scores):
            text_str = str(cand.get("text") or "")
            section_str = str(cand.get("section_title") or "")
            doc_tok_set = set(tokenize_lexical(f"{section_str} {text_str}"))

            fusion_norm = float(cand.get("fusion_score") or 0.0) / max_fusion
            bm25_norm = bm25_val / max_bm25

            # Query term coverage in [0, 1]
            if orig_tokens:
                matched_terms = sum(1 for t in set(orig_tokens) if t in doc_tok_set)
                coverage = matched_terms / max(1, len(set(orig_tokens)))
            else:
                coverage = 0.0

            # Bigram / ordered phrase proximity bonus
            text_norm_join = " ".join(tokenize_lexical(text_str))
            if orig_bigrams:
                bigram_hits = sum(1 for bg in orig_bigrams if bg in text_norm_join)
                phrase_score = bigram_hits / len(orig_bigrams)
            else:
                phrase_score = 0.0

            # Section heading alignment bonus
            sec_tokens = set(tokenize_lexical(section_str))
            heading_bonus = (
                0.08
                if (orig_tokens and sec_tokens and any(t in sec_tokens for t in orig_tokens))
                else 0.0
            )

            # Multi-channel consensus bonus (matched in >=2 independent channels)
            channels_matched = sum(
                1
                for rk in (cand.get("dense_rank"), cand.get("sparse_rank"), cand.get("graph_rank"))
                if rk is not None
            )
            consensus_bonus = 0.05 * max(0, channels_matched - 1)

            calibrated_score = (
                0.48 * fusion_norm
                + 0.24 * bm25_norm
                + 0.16 * coverage
                + 0.12 * phrase_score
                + heading_bonus
                + consensus_bonus
            )

            updated = dict(cand)
            updated["bm25_score"] = round(bm25_val, 6)
            updated["rerank_score"] = round(calibrated_score, 6)
            reranked_items.append(updated)

        reranked_items.sort(
            key=lambda x: (
                float(x.get("rerank_score") or 0.0),
                float(x.get("fusion_score") or 0.0),
            ),
            reverse=True,
        )
        final_slice = reranked_items[:top_k]
        for idx, item in enumerate(final_slice, start=1):
            item["final_rank"] = idx
        return final_slice


@dataclass
class HybridRetrievalResult:
    """Complete structured output from HybridRetrievalEngine including diagnostics & citations."""

    request_id: str
    query: str
    analysis: QueryAnalysis
    mode: RetrievalMode
    fusion_strategy: FusionStrategy
    reranked: bool
    chunks: list[dict[str, Any]]
    citations: list[dict[str, Any]]
    web_results: list[dict[str, Any]]
    graph_entities: list[dict[str, Any]]
    graph_status: str
    web_status: str
    assembled_context: str
    has_sufficient_evidence: bool
    diagnostics: dict[str, Any]


class HybridRetrievalEngine:
    """Combines Dense BGE-M3, Sparse/BM25 Lexical, Neo4j Graph, and optional Tavily Web search."""

    def __init__(
        self,
        embedder: EmbeddingPort,
        vector_store: VectorStorePort,
        graph_store: GraphStorePort | None = None,
        tavily_service: TavilySearchService | None = None,
        reranker: HybridReranker | None = None,
    ) -> None:
        self._embedder = embedder
        self._vector_store = vector_store
        self._graph_store = graph_store
        self._tavily = tavily_service or TavilySearchService()
        self._reranker = reranker or HybridReranker()
        self._bm25 = BM25LexicalScorer()

    @traced_step(name="hybrid_retrieve", run_type="retriever")
    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        dense_top_k: int | None = None,
        sparse_top_k: int | None = None,
        graph_top_k: int | None = None,
        mode: str | None = None,
        fusion_strategy: str | None = None,
        rrf_k: int | None = None,
        enable_reranking: bool | None = None,
        include_web: bool = False,
        filter_metadata: dict[str, Any] | None = None,
        max_context_tokens: int = 3600,
        min_relevance_threshold: float = 0.08,
    ) -> HybridRetrievalResult:
        """Execute full hybrid retrieval pipeline with latency & rank diagnostics."""
        t0 = time.perf_counter()
        req_id = get_request_id()
        settings = get_settings()

        final_top_k = max(1, top_k or settings.RETRIEVAL_TOP_K or 8)
        d_top_k = max(final_top_k, dense_top_k or final_top_k * 3)
        s_top_k = max(final_top_k, sparse_top_k or final_top_k * 3)
        g_top_k = max(4, graph_top_k or final_top_k * 2)
        k_rrf = max(1, rrf_k or settings.RETRIEVAL_RRF_K or 60)
        do_rerank = settings.ENABLE_RERANKING if enable_reranking is None else bool(enable_reranking)

        valid_fusions = {"rrf", "normalized_linear", "qdrant_native_rrf"}
        chosen_fusion: FusionStrategy = (
            fusion_strategy.lower()  # type: ignore[assignment]
            if (fusion_strategy and fusion_strategy.lower() in valid_fusions)
            else "rrf"
        )

        latencies: dict[str, float] = {}
        with StepTimer(latencies, "query_understanding_ms"):
            analysis = understand_query(query, requested_mode=mode, include_web=include_web)
        effective_mode = analysis.recommended_mode

        if not analysis.normalized_query or (
            not include_web
            and not self._extract_scoped_doc_ids(filter_metadata)
            and _CONVERSATIONAL_GREETING_RE.match(analysis.normalized_query)
        ):
            return HybridRetrievalResult(
                request_id=req_id,
                query=query,
                analysis=analysis,
                mode=effective_mode,
                fusion_strategy=chosen_fusion,
                reranked=False,
                chunks=[],
                citations=[],
                web_results=[],
                graph_entities=[],
                graph_status="skipped",
                web_status="skipped",
                assembled_context="",
                has_sufficient_evidence=False,
                diagnostics={
                    "latency_ms": latencies,
                    "reason": "empty_query" if not analysis.normalized_query else "conversational_greeting",
                },
            )

        run_dense = effective_mode in ("hybrid", "dense")
        run_sparse = effective_mode in ("hybrid", "sparse") and settings.ENABLE_SPARSE_RETRIEVAL
        run_graph = effective_mode in ("hybrid", "graph") and self._graph_store is not None
        run_web = bool(include_web or effective_mode == "web")

        # Execute independent retrieval channels concurrently
        dense_task = (
            self._retrieve_dense(analysis.semantic_query, d_top_k, filter_metadata, latencies)
            if run_dense
            else self._noop_list()
        )
        sparse_task = (
            self._retrieve_sparse(analysis, s_top_k, filter_metadata, latencies)
            if run_sparse
            else self._noop_list()
        )
        graph_task = (
            self._retrieve_graph(analysis.extracted_entities, g_top_k, latencies, filter_metadata=filter_metadata)
            if run_graph
            else self._noop_graph("disabled" if self._graph_store is not None else "unavailable")
        )
        web_task = (
            self._retrieve_web(analysis.normalized_query, latencies)
            if run_web
            else self._noop_web("disabled")
        )

        dense_hits, sparse_hits, graph_bundle, web_bundle = await asyncio.gather(
            dense_task,
            sparse_task,
            graph_task,
            web_task,
        )

        graph_chunks: list[dict[str, Any]] = graph_bundle["chunks"]
        graph_entities: list[dict[str, Any]] = graph_bundle["entities"]
        graph_status: str = graph_bundle["status"]
        graph_error: str | None = graph_bundle.get("error")

        # Optional Qdrant native server-side RRF mode
        native_hits: list[dict[str, Any]] = []
        if chosen_fusion == "qdrant_native_rrf" and run_dense and run_sparse and hasattr(self._vector_store, "search_hybrid_native"):
            with StepTimer(latencies, "qdrant_native_rrf_ms"):
                try:
                    q_vec = await self._embed_query_async(analysis.semantic_query)
                    s_vec = build_lexical_sparse_vector(analysis.expanded_lexical_query)
                    native_hits = await getattr(self._vector_store, "search_hybrid_native")(
                        query_vector=q_vec,
                        sparse_vector=s_vec,
                        limit=max(d_top_k, s_top_k),
                        prefetch_limit=max(d_top_k, s_top_k),
                        filter_metadata=filter_metadata,
                    )
                except Exception as exc:
                    logger.warning("Qdrant native RRF degraded: %s", exc)
                    native_hits = []

        # Enrich sparse branch with Okapi BM25 re-scoring over union of dense+sparse candidates
        # so exact keywords/acronyms inside dense candidates also receive accurate sparse ranks!
        if run_sparse:
            sparse_hits = self._enrich_sparse_with_bm25(analysis, sparse_hits, dense_hits, s_top_k)

        # Fuse dense + sparse + graph candidates with chunk_id deduplication
        with StepTimer(latencies, "fusion_ms"):
            fused_candidates = self._fuse_candidates(
                dense_hits=dense_hits,
                sparse_hits=sparse_hits,
                graph_hits=graph_chunks,
                native_hits=native_hits,
                strategy=chosen_fusion,
                rrf_k=k_rrf,
            )

        # 1-hop graph expansion on top fused document chunks if Neo4j is healthy
        if run_graph and graph_status in ("ok", "empty") and fused_candidates and self._graph_store is not None:
            with StepTimer(latencies, "graph_expansion_ms"):
                top_uuids: list[UUID] = []
                for c in fused_candidates[: min(10, len(fused_candidates))]:
                    try:
                        top_uuids.append(UUID(str(c["chunk_id"])))
                    except Exception:
                        continue
                if top_uuids:
                    try:
                        hop_entities = await self._graph_store.get_chunk_entities(top_uuids)
                        seen_e_keys = {
                            (e.get("entity_id"), e.get("related_entity_name"), e.get("relationship_type"))
                            for e in graph_entities
                        }
                        for he in hop_entities:
                            key = (he.get("entity_id"), he.get("related_entity_name"), he.get("relationship_type"))
                            if he.get("entity_name") and key not in seen_e_keys:
                                seen_e_keys.add(key)
                                graph_entities.append(he)
                        if graph_entities and graph_status == "empty":
                            graph_status = "ok"
                    except Exception as exc:
                        logger.warning("Neo4j 1-hop chunk entity expansion failed: %s", exc)

        # Holistic document-overview enrichment: when user asks to summarize/analyze a scoped document,
        # pull opening structural chunks (chunk_index 0..3) so overview answers are 100% grounded in the document.
        is_overview_query = bool(_OVERVIEW_QUERY_RE.search(analysis.original_query))
        scoped_doc_ids = self._extract_scoped_doc_ids(filter_metadata)
        if is_overview_query and hasattr(self._vector_store, "get_document_chunks"):
            target_docs = list(scoped_doc_ids)
            if not target_docs and dense_hits:
                top_doc_id = str(dense_hits[0].get("document_id") or "")
                if top_doc_id and not top_doc_id.startswith("csv:"):
                    target_docs = [top_doc_id]
            for doc_id_str in target_docs[:2]:
                try:
                    lead_chunks = await getattr(self._vector_store, "get_document_chunks")(doc_id_str, limit=4)
                    existing_cids = {str(h.get("chunk_id") or "") for h in dense_hits}
                    for rank_offset, lch in enumerate(lead_chunks, start=1):
                        lcid = str(lch.get("chunk_id") or "")
                        if lcid and lcid not in existing_cids:
                            enriched = dict(lch)
                            enriched["dense_rank"] = len(dense_hits) + rank_offset
                            enriched["dense_score"] = max(float(lch.get("score") or 0.0), 0.62)
                            enriched["is_overview_lead"] = True
                            dense_hits.append(enriched)
                            existing_cids.add(lcid)
                        elif lcid:
                            for dh in dense_hits:
                                if str(dh.get("chunk_id") or "") == lcid:
                                    dh["is_overview_lead"] = True
                except Exception as exc:
                    logger.debug("Overview lead chunk lookup skipped for %s: %s", doc_id_str, exc)

        # Relevance gating: avoid feeding pure noise when neither semantic nor lexical terms match
        embedder_model = str(getattr(self._embedder, "model_name", "") or "").lower()
        if do_rerank and fused_candidates:
            with StepTimer(latencies, "rerank_ms"):
                final_chunks = self._reranker.rerank(
                    analysis=analysis,
                    candidates=fused_candidates,
                    top_k=final_top_k,
                )
        else:
            final_chunks = fused_candidates[:final_top_k]
            for idx, item in enumerate(final_chunks, start=1):
                item["final_rank"] = idx

        # Re-merge any overview lead chunks that weren't in fused_candidates yet
        if is_overview_query:
            present_ids = {str(c.get("chunk_id") or "") for c in final_chunks}
            for dh in dense_hits:
                if dh.get("is_overview_lead") and str(dh.get("chunk_id") or "") not in present_ids:
                    if len(final_chunks) < final_top_k:
                        final_chunks.append(dh)
                        present_ids.add(str(dh.get("chunk_id") or ""))

        gated_chunks = self._filter_irrelevant_noise(
            analysis=analysis,
            chunks=final_chunks,
            min_threshold=min_relevance_threshold,
            embedder_model=embedder_model,
            effective_mode=effective_mode,
            has_doc_scope=bool(scoped_doc_ids),
        )

        web_results: list[dict[str, Any]] = web_bundle["results"]
        web_status: str = web_bundle["status"]

        with StepTimer(latencies, "context_assembly_ms"):
            assembled_context, citations = self.assemble_context_and_citations(
                kb_chunks=gated_chunks,
                graph_entities=graph_entities,
                web_results=web_results,
                max_context_tokens=max_context_tokens,
            )

        total_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        latencies["total_retrieval_ms"] = total_ms
        has_evidence = bool(gated_chunks or web_results or (effective_mode == "graph" and graph_entities))

        diagnostics: dict[str, Any] = {
            "request_id": req_id,
            "mode": effective_mode,
            "fusion_strategy": chosen_fusion,
            "rrf_k": k_rrf,
            "reranked": bool(do_rerank and len(fused_candidates) > 0),
            "dense_candidates": len(dense_hits),
            "sparse_candidates": len(sparse_hits),
            "graph_candidates": len(graph_chunks),
            "graph_entities_count": len(graph_entities),
            "fused_unique_candidates": len(fused_candidates),
            "final_selected_chunks": len(gated_chunks),
            "web_results_count": len(web_results),
            "graph_status": graph_status,
            "graph_error": graph_error,
            "web_status": web_status,
            "expanded_acronyms": analysis.expanded_acronyms,
            "extracted_entities": analysis.extracted_entities,
            "sub_queries": analysis.sub_queries,
            "latency_ms": latencies,
        }

        get_observability_tracker().record(
            TraceTelemetryRecord(
                request_id=req_id,
                operation="hybrid_retrieval",
                status="ok",
                query=analysis.normalized_query[:240],
                retrieval_mode=effective_mode,
                fusion_strategy=chosen_fusion,
                reranked=bool(do_rerank),
                dense_count=len(dense_hits),
                sparse_count=len(sparse_hits),
                graph_count=len(graph_chunks) + len(graph_entities),
                web_count=len(web_results),
                final_count=len(gated_chunks),
                graph_status=graph_status,
                latency_ms=latencies,
                total_ms=total_ms,
            )
        )

        return HybridRetrievalResult(
            request_id=req_id,
            query=query,
            analysis=analysis,
            mode=effective_mode,
            fusion_strategy=chosen_fusion,
            reranked=bool(do_rerank),
            chunks=gated_chunks,
            citations=citations,
            web_results=web_results,
            graph_entities=graph_entities,
            graph_status=graph_status,
            web_status=web_status,
            assembled_context=assembled_context,
            has_sufficient_evidence=has_evidence,
            diagnostics=diagnostics,
        )

    async def _embed_query_async(self, text: str) -> list[float]:
        if hasattr(self._embedder, "aembed_query"):
            return await getattr(self._embedder, "aembed_query")(text)
        return await asyncio.to_thread(self._embedder.embed_query, text)

    @traced_step(name="dense_retrieval", run_type="retriever")
    async def _retrieve_dense(
        self,
        semantic_query: str,
        limit: int,
        filter_metadata: dict[str, Any] | None,
        latencies: dict[str, float],
    ) -> list[dict[str, Any]]:
        start = time.perf_counter()
        try:
            q_vec = await self._embed_query_async(semantic_query)
            hits = await self._vector_store.search(
                query_vector=q_vec,
                limit=limit,
                filter_metadata=filter_metadata,
            )
            ranked: list[dict[str, Any]] = []
            for rank, h in enumerate(hits, start=1):
                item = dict(h)
                item["dense_rank"] = rank
                item["dense_score"] = round(float(h.get("score") or 0.0), 6)
                ranked.append(item)
            return ranked
        except Exception as exc:
            logger.warning("Dense vector retrieval degraded: %s", exc)
            return []
        finally:
            latencies["dense_ms"] = round((time.perf_counter() - start) * 1000.0, 2)

    @traced_step(name="sparse_retrieval", run_type="retriever")
    async def _retrieve_sparse(
        self,
        analysis: QueryAnalysis,
        limit: int,
        filter_metadata: dict[str, Any] | None,
        latencies: dict[str, float],
    ) -> list[dict[str, Any]]:
        start = time.perf_counter()
        try:
            if not hasattr(self._vector_store, "search_sparse"):
                return []
            sparse_vec = build_lexical_sparse_vector(analysis.expanded_lexical_query)
            hits = await getattr(self._vector_store, "search_sparse")(
                sparse_vector=sparse_vec,
                limit=limit,
                filter_metadata=filter_metadata,
            )
            ranked: list[dict[str, Any]] = []
            for rank, h in enumerate(hits, start=1):
                item = dict(h)
                item["sparse_rank"] = rank
                item["sparse_score"] = round(float(h.get("score") or 0.0), 6)
                ranked.append(item)
            return ranked
        except Exception as exc:
            logger.warning("Sparse lexical retrieval degraded: %s", exc)
            return []
        finally:
            latencies["sparse_ms"] = round((time.perf_counter() - start) * 1000.0, 2)

    def _enrich_sparse_with_bm25(
        self,
        analysis: QueryAnalysis,
        sparse_hits: list[dict[str, Any]],
        dense_hits: list[dict[str, Any]],
        limit: int,
    ) -> list[dict[str, Any]]:
        """Combine Qdrant sparse vector hits with Okapi BM25 scoring across candidate pool."""
        by_id: dict[str, dict[str, Any]] = {}
        for h in sparse_hits:
            cid = str(h.get("chunk_id") or "")
            if cid:
                by_id[cid] = dict(h)
        for h in dense_hits:
            cid = str(h.get("chunk_id") or "")
            if cid and cid not in by_id:
                by_id[cid] = dict(h)

        pool = list(by_id.values())
        if not pool:
            return []

        bm25_scores = self._bm25.score_documents(
            analysis.original_query,
            analysis.query_tokens,
            pool,
        )
        scored_pool: list[dict[str, Any]] = []
        for item, bm25_sc in zip(pool, bm25_scores):
            raw_sparse = float(item.get("sparse_score") or 0.0)
            combined_lex = raw_sparse + bm25_sc
            if combined_lex <= 0.0:
                continue
            updated = dict(item)
            updated["bm25_score"] = round(bm25_sc, 6)
            updated["sparse_score"] = round(combined_lex, 6)
            scored_pool.append(updated)

        scored_pool.sort(key=lambda x: float(x.get("sparse_score") or 0.0), reverse=True)
        top_sparse = scored_pool[:limit]
        for rank, item in enumerate(top_sparse, start=1):
            item["sparse_rank"] = rank
        return top_sparse

    @staticmethod
    def _extract_scoped_doc_ids(filter_metadata: dict[str, Any] | None) -> list[str]:
        if not filter_metadata:
            return []
        doc_ids: list[str] = []
        single = filter_metadata.get("document_id") or filter_metadata.get("source_id")
        if single:
            doc_ids.append(str(single))
        multi = filter_metadata.get("document_ids") or filter_metadata.get("source_ids")
        if isinstance(multi, (list, tuple, set)):
            for item in multi:
                if item and str(item) not in doc_ids:
                    doc_ids.append(str(item))
        return doc_ids

    @traced_step(name="neo4j_graph_retrieval", run_type="retriever")
    async def _retrieve_graph(
        self,
        entities: list[str],
        limit: int,
        latencies: dict[str, float],
        filter_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        start = time.perf_counter()
        try:
            if self._graph_store is None:
                return {"status": "unavailable", "chunks": [], "entities": [], "error": "Graph store not configured"}
            if not entities:
                return {"status": "empty", "chunks": [], "entities": [], "error": None}
            if not hasattr(self._graph_store, "search_graph_context"):
                return {"status": "empty", "chunks": [], "entities": [], "error": None}

            scoped_doc_ids = self._extract_scoped_doc_ids(filter_metadata)
            search_fn = getattr(self._graph_store, "search_graph_context")
            if scoped_doc_ids:
                try:
                    ctx = await asyncio.wait_for(
                        search_fn(entities, limit=limit, document_ids=scoped_doc_ids),
                        timeout=8.0,
                    )
                except TypeError:
                    ctx = await asyncio.wait_for(
                        search_fn(entities, limit=limit),
                        timeout=8.0,
                    )
            else:
                ctx = await asyncio.wait_for(
                    search_fn(entities, limit=limit),
                    timeout=8.0,
                )
            raw_chunks = list(ctx.get("chunks") or [])
            if scoped_doc_ids:
                allowed = set(scoped_doc_ids)
                raw_chunks = [c for c in raw_chunks if str(c.get("document_id") or "") in allowed]
            raw_entities = list(ctx.get("entities") or [])
            ranked_chunks: list[dict[str, Any]] = []
            for rank, ch in enumerate(raw_chunks, start=1):
                item = dict(ch)
                item["graph_rank"] = rank
                item["graph_score"] = round(float(ch.get("score") or 0.75), 6)
                ranked_chunks.append(item)

            status = "ok" if (ranked_chunks or raw_entities) else "empty"
            return {
                "status": status,
                "chunks": ranked_chunks,
                "entities": raw_entities,
                "error": None,
            }
        except Exception as exc:
            logger.warning("Neo4j graph retrieval unavailable (falling back to vector+sparse): %s", exc)
            return {
                "status": "unavailable",
                "chunks": [],
                "entities": [],
                "error": str(exc),
            }
        finally:
            latencies["graph_ms"] = round((time.perf_counter() - start) * 1000.0, 2)

    async def _retrieve_web(
        self,
        query: str,
        latencies: dict[str, float],
    ) -> dict[str, Any]:
        start = time.perf_counter()
        try:
            return await self._tavily.search(query=query, max_results=5)
        finally:
            latencies["web_ms"] = round((time.perf_counter() - start) * 1000.0, 2)

    @staticmethod
    async def _noop_list() -> list[dict[str, Any]]:
        return []

    @staticmethod
    async def _noop_graph(status: str) -> dict[str, Any]:
        return {"status": status, "chunks": [], "entities": [], "error": None}

    @staticmethod
    async def _noop_web(status: str) -> dict[str, Any]:
        return {"status": status, "results": []}

    def _fuse_candidates(
        self,
        dense_hits: list[dict[str, Any]],
        sparse_hits: list[dict[str, Any]],
        graph_hits: list[dict[str, Any]],
        native_hits: list[dict[str, Any]],
        strategy: FusionStrategy,
        rrf_k: int,
    ) -> list[dict[str, Any]]:
        """Deduplicate by `chunk_id` and compute RRF or min-max normalized fusion scores."""
        merged: dict[str, dict[str, Any]] = {}

        def _upsert_candidate(hit: dict[str, Any], channel_label: str) -> str:
            cid = str(hit.get("chunk_id") or hit.get("content_hash") or id(hit))
            if cid not in merged:
                merged[cid] = {
                    "chunk_id": cid,
                    "document_id": hit.get("document_id"),
                    "chunk_index": hit.get("chunk_index", 0),
                    "text": hit.get("text", ""),
                    "page_number": hit.get("page_number"),
                    "section_title": hit.get("section_title"),
                    "filename": hit.get("filename", ""),
                    "source_type": hit.get("source_type", ""),
                    "content_hash": hit.get("content_hash", ""),
                    "entity_ids": hit.get("entity_ids", []),
                    "metadata": dict(hit.get("metadata") or {}),
                    "dense_rank": None,
                    "sparse_rank": None,
                    "graph_rank": None,
                    "dense_score": None,
                    "sparse_score": None,
                    "bm25_score": None,
                    "graph_score": None,
                    "native_rrf_score": None,
                    "channels": [],
                }
            entry = merged[cid]
            if channel_label not in entry["channels"]:
                entry["channels"].append(channel_label)
            # Fill missing metadata if one channel (e.g., graph) had partial fields
            for fld in ("document_id", "page_number", "section_title", "filename", "source_type", "content_hash"):
                if not entry.get(fld) and hit.get(fld):
                    entry[fld] = hit[fld]
            if not entry.get("text") and hit.get("text"):
                entry["text"] = hit["text"]
            if hit.get("metadata"):
                for mk, mv in hit["metadata"].items():
                    entry["metadata"].setdefault(mk, mv)
            return cid

        for h in dense_hits:
            cid = _upsert_candidate(h, "dense")
            merged[cid]["dense_rank"] = h.get("dense_rank")
            merged[cid]["dense_score"] = h.get("dense_score", h.get("score"))

        for h in sparse_hits:
            cid = _upsert_candidate(h, "sparse")
            merged[cid]["sparse_rank"] = h.get("sparse_rank")
            merged[cid]["sparse_score"] = h.get("sparse_score", h.get("score"))
            if h.get("bm25_score") is not None:
                merged[cid]["bm25_score"] = h.get("bm25_score")

        for h in graph_hits:
            ch_label = str(h.get("retrieval_channel") or "graph")
            cid = _upsert_candidate(h, ch_label)
            merged[cid]["graph_rank"] = h.get("graph_rank")
            merged[cid]["graph_score"] = h.get("graph_score", h.get("score"))

        for rank, h in enumerate(native_hits, start=1):
            cid = _upsert_candidate(h, "qdrant_native_rrf")
            merged[cid]["native_rrf_score"] = float(h.get("score") or (1.0 / (rrf_k + rank)))

        if not merged:
            return []

        items = list(merged.values())

        if strategy == "normalized_linear":
            d_vals = [float(x["dense_score"]) for x in items if x["dense_score"] is not None]
            s_vals = [float(x["sparse_score"]) for x in items if x["sparse_score"] is not None]
            g_vals = [float(x["graph_score"]) for x in items if x["graph_score"] is not None]

            def _norm(val: float | None, vals: list[float]) -> float:
                if val is None or not vals:
                    return 0.0
                lo, hi = min(vals), max(vals)
                if math.isclose(hi, lo):
                    return 1.0 if val > 0 else 0.0
                return max(0.0, min(1.0, (val - lo) / (hi - lo)))

            for x in items:
                dn = _norm(x["dense_score"], d_vals)
                sn = _norm(x["sparse_score"], s_vals)
                gn = _norm(x["graph_score"], g_vals)
                x["fusion_score"] = round(0.55 * dn + 0.35 * sn + 0.10 * gn, 6)
                x["fusion_strategy"] = "normalized_linear"

        elif strategy == "qdrant_native_rrf" and native_hits:
            for x in items:
                base = float(x["native_rrf_score"] or 0.0)
                if base == 0.0:
                    if x["dense_rank"] is not None:
                        base += 1.0 / (rrf_k + int(x["dense_rank"]))
                    if x["sparse_rank"] is not None:
                        base += 0.9 / (rrf_k + int(x["sparse_rank"]))
                if x["graph_rank"] is not None:
                    base += 0.75 / (rrf_k + int(x["graph_rank"]))
                x["fusion_score"] = round(base, 6)
                x["fusion_strategy"] = "qdrant_native_rrf"

        else:
            # Default Reciprocal Rank Fusion (RRF): scale-invariant across dense cosine, sparse BM25, & Neo4j graph
            for x in items:
                rrf_score = 0.0
                if x["dense_rank"] is not None:
                    rrf_score += 1.0 / (rrf_k + int(x["dense_rank"]))
                if x["sparse_rank"] is not None:
                    rrf_score += 0.95 / (rrf_k + int(x["sparse_rank"]))
                if x["graph_rank"] is not None:
                    rrf_score += 0.80 / (rrf_k + int(x["graph_rank"]))
                x["fusion_score"] = round(rrf_score, 6)
                x["fusion_strategy"] = "rrf"

        for x in items:
            x["retrieval_channel"] = "+".join(x["channels"])
            x["score"] = x["fusion_score"]

        items.sort(
            key=lambda x: (
                float(x.get("fusion_score") or 0.0),
                float(x.get("dense_score") or 0.0),
                float(x.get("sparse_score") or 0.0),
            ),
            reverse=True,
        )
        return items

    @staticmethod
    def _filter_irrelevant_noise(
        analysis: QueryAnalysis,
        chunks: list[dict[str, Any]],
        min_threshold: float,
        embedder_model: str = "",
        effective_mode: str = "hybrid",
        has_doc_scope: bool = False,
    ) -> list[dict[str, Any]]:
        """Drop low-similarity / cross-document noise so out-of-domain queries trigger honest abstention."""
        if not chunks:
            return []

        raw_q_tokens = set(tokenize_lexical(analysis.original_query))
        specific_q_tokens = {t for t in raw_q_tokens if t not in _GENERIC_ENTITY_STOPWORDS}
        active_q_tokens = specific_q_tokens or raw_q_tokens
        is_overview = bool(_OVERVIEW_QUERY_RE.search(analysis.original_query))

        # BAAI/bge-m3 has an anisotropic baseline cosine similarity (~0.38-0.49) even on unrelated pairs,
        # whereas MiniLM / isotropic models hover around ~0.05-0.25 on unrelated pairs.
        is_bge = "bge-m3" in embedder_model
        pure_dense_floor = 0.56 if is_bge else 0.38
        lexical_dense_floor = (0.38 if is_bge else min_threshold)
        if has_doc_scope:
            pure_dense_floor -= 0.05

        filtered: list[dict[str, Any]] = []
        for ch in chunks:
            if ch.get("is_overview_lead") or (is_overview and has_doc_scope):
                filtered.append(ch)
                continue

            d_score = float(ch.get("dense_score") or 0.0)
            bm25_sc = float(ch.get("bm25_score") or ch.get("sparse_score") or 0.0)
            has_graph = ch.get("graph_rank") is not None

            ch_model = str(ch.get("embedding_model") or (ch.get("metadata") or {}).get("embedding_model") or "").lower()
            eff_pure_floor = 0.56 if ("bge-m3" in ch_model or is_bge) else pure_dense_floor
            if has_doc_scope:
                eff_pure_floor -= 0.05

            doc_tokens = set(
                tokenize_lexical(
                    f"{ch.get('filename', '')} {ch.get('section_title', '')} {ch.get('text', '')}"
                )
            )
            overlap_tokens = active_q_tokens & doc_tokens
            lexical_overlap = len(overlap_tokens) >= 1

            # Graph-only candidates MUST still share at least one specific query token (or run in explicit graph mode)
            # to prevent unrelated CSV/KG nodes (e.g. suppliers.csv) from polluting document chat.
            if effective_mode == "graph" and has_graph:
                filtered.append(ch)
            elif d_score >= eff_pure_floor:
                filtered.append(ch)
            elif lexical_overlap and (d_score >= lexical_dense_floor or bm25_sc > 0.0 or has_graph):
                filtered.append(ch)
        return filtered

    @staticmethod
    def assemble_context_and_citations(
        kb_chunks: list[dict[str, Any]],
        graph_entities: list[dict[str, Any]],
        web_results: list[dict[str, Any]],
        max_context_tokens: int = 3600,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Construct deduplicated, token-budgeted context and rich `[KB-n]` / `[WEB-n]` citations."""
        citations: list[dict[str, Any]] = []
        kb_blocks: list[str] = []
        used_tokens = 0
        seen_parents: set[str] = set()
        seen_token_sets: list[set[str]] = []

        kb_idx = 0
        for ch in kb_chunks:
            raw_text = str(ch.get("text") or "").strip()
            if not raw_text:
                continue

            meta = dict(ch.get("metadata") or {})
            parent_id = meta.get("parent_chunk_id")
            parent_text = str(meta.get("parent_text") or "").strip()

            if parent_id and parent_text:
                if str(parent_id) in seen_parents:
                    display_text = raw_text
                else:
                    seen_parents.add(str(parent_id))
                    display_text = parent_text
            else:
                display_text = raw_text

            # Near-duplicate excerpt suppression (>88% Jaccard overlap)
            tok_set = set(tokenize_lexical(display_text))
            if tok_set and any(
                (len(tok_set & prev) / max(1, len(tok_set | prev))) > 0.88
                for prev in seen_token_sets
            ):
                continue

            est_toks = estimate_token_count(display_text) + 32
            if kb_blocks and (used_tokens + est_toks) > max_context_tokens:
                break

            seen_token_sets.append(tok_set)
            used_tokens += est_toks
            kb_idx += 1
            cite_label = f"KB-{kb_idx}"

            filename = ch.get("filename") or meta.get("filename") or "Unknown Source"
            page_no = ch.get("page_number") if ch.get("page_number") is not None else meta.get("page_number")
            sec_title = ch.get("section_title") or meta.get("section_title")
            safe_excerpt = sanitize_untrusted_text(display_text, max_chars=2400)

            header_parts = [f"[{cite_label}] File: {filename}"]
            if page_no is not None:
                header_parts.append(f"Page: {page_no}")
            if sec_title:
                header_parts.append(f"Section: {sec_title}")
            header_parts.append(f"ChunkID: {ch.get('chunk_id')}")

            kb_blocks.append(f"{' | '.join(header_parts)}\n{safe_excerpt}")

            citations.append(
                {
                    "citation_id": cite_label,
                    "source_kind": "knowledge_base",
                    "chunk_id": str(ch.get("chunk_id") or ""),
                    "document_id": str(ch.get("document_id") or "") if ch.get("document_id") else None,
                    "chunk_index": int(ch.get("chunk_index") or 0),
                    "filename": filename,
                    "page_number": page_no,
                    "section_title": sec_title,
                    "source_type": ch.get("source_type") or meta.get("source_type") or "",
                    "content_hash": ch.get("content_hash") or meta.get("content_hash") or "",
                    "text_snippet": raw_text[:280] + ("..." if len(raw_text) > 280 else ""),
                    "score": round(float(ch.get("rerank_score") or ch.get("fusion_score") or ch.get("score") or 0.0), 6),
                    "dense_rank": ch.get("dense_rank"),
                    "sparse_rank": ch.get("sparse_rank"),
                    "graph_rank": ch.get("graph_rank"),
                    "final_rank": ch.get("final_rank", kb_idx),
                    "dense_score": ch.get("dense_score"),
                    "sparse_score": ch.get("sparse_score"),
                    "bm25_score": ch.get("bm25_score"),
                    "fusion_score": ch.get("fusion_score"),
                    "rerank_score": ch.get("rerank_score"),
                    "retrieval_channel": ch.get("retrieval_channel", "hybrid"),
                }
            )

        sections_out: list[str] = []
        if kb_blocks:
            sections_out.append("## Private Knowledge Base Evidence\n\n" + "\n\n---\n\n".join(kb_blocks))

        if graph_entities:
            triples: list[str] = []
            for ge in graph_entities[:20]:
                ename = ge.get("entity_name") or ge.get("name")
                etype = ge.get("entity_type") or ge.get("type") or "Entity"
                rel = ge.get("relationship_type")
                target = ge.get("related_entity_name")
                src_chunk = ge.get("chunk_id")
                trace = f" (chunk_id={src_chunk})" if src_chunk else ""
                if ename and rel and target:
                    triples.append(f"- ({ename}:{etype}) -[{rel}]-> ({target}){trace}")
                elif ename:
                    triples.append(f"- ({ename}:{etype}){trace}")
            if triples:
                sections_out.append("## Knowledge Graph Relationships (Neo4j)\n" + "\n".join(triples))

        if web_results:
            web_blocks: list[str] = []
            for wr in web_results[:5]:
                cid = wr.get("citation_id") or "WEB-1"
                web_blocks.append(
                    f"[{cid}] Title: {wr.get('title')} | URL: {wr.get('url')}\n{wr.get('text')}"
                )
                citations.append(dict(wr))
            sections_out.append(
                "## External Web Research (Untrusted External Sources — Cite as [WEB-n])\n"
                "<untrusted_web_evidence>\n"
                + "\n\n---\n\n".join(web_blocks)
                + "\n</untrusted_web_evidence>"
            )

        return "\n\n====================\n\n".join(sections_out), citations

