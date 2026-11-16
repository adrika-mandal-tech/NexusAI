"""Deterministic + LLM-assisted entity and relationship extraction.

Assigns canonical UUIDv5 `entity_id`s (`deterministic_entity_id(type, name)`)
so entities extracted across separate chunks and documents merge cleanly in
Neo4j and link bi-directionally with Qdrant chunk payloads (`entity_ids`).
"""

from __future__ import annotations

import asyncio
import re
from itertools import combinations
from typing import Any

from nexusai.core.ids import deterministic_entity_id, normalize_entity_name
from nexusai.core.logging import get_logger
from nexusai.domain.ports.llm_port import LLMPort

logger = get_logger(__name__)

# Curated technical / domain vocabulary classification hints (PascalCase Neo4j labels)
_TECH_KEYWORDS: dict[str, str] = {
    "cuda": "Technology",
    "simt": "Architecture",
    "pcie": "Hardware",
    "dma": "Hardware",
    "nvlink": "Hardware",
    "nvidia": "Organization",
    "gpu": "Hardware",
    "vram": "Hardware",
    "qdrant": "Database",
    "neo4j": "Database",
    "postgresql": "Database",
    "postgres": "Database",
    "redis": "Database",
    "fastapi": "Framework",
    "nextjs": "Framework",
    "next.js": "Framework",
    "react": "Framework",
    "langchain": "Framework",
    "langsmith": "Tool",
    "ragas": "Tool",
    "deepeval": "Tool",
    "docling": "Tool",
    "paddleocr": "Tool",
    "pymupdf": "Tool",
    "pypdf": "Tool",
    "tavily": "Service",
    "gemini": "Model",
    "groq": "Service",
    "bge-m3": "Model",
    "minilm": "Model",
    "rrf": "Algorithm",
    "bm25": "Algorithm",
    "hnsw": "Algorithm",
    "cosine similarity": "Metric",
    "cypher": "Language",
    "python": "Language",
    "typescript": "Language",
}

# Capitalized multi-word phrases, CamelCase symbols, and ALL-CAPS acronyms (2-12 chars)
_PROPER_PHRASE_RE = re.compile(
    r"\b(?:[A-Z][a-z0-9]+(?:\s+[A-Z][a-z0-9]+){0,3}|[A-Z]{2,12}(?:-[A-Z0-9]{1,8})?)\b"
)

_STOP_PROPER = frozenset(
    {
        "The", "This", "That", "These", "Those", "When", "Where", "Which", "While",
        "With", "From", "Into", "Upon", "Over", "Under", "Between", "Among", "After",
        "Before", "During", "Section", "Chapter", "Table", "Figure", "Page", "Note",
        "Example", "Summary", "Introduction", "Conclusion", "Overview", "Appendix",
        "True", "False", "None", "Null", "Step", "Part", "Item", "Type", "Name",
    }
)

_RELATION_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\buses?\b|\butilizes?\b|\bleverages?\b|\bpowered\s+by\b", re.IGNORECASE), "USES"),
    (re.compile(r"\bstores?\s+in\b|\bindexed\s+in\b|\bpersisted\s+in\b|\nupserts?\s+into\b", re.IGNORECASE), "STORED_IN"),
    (re.compile(r"\bdepends?\s+on\b|\brequires?\b|\bneeds?\b", re.IGNORECASE), "DEPENDS_ON"),
    (re.compile(r"\bpart\s+of\b|\bbelongs?\s+to\b|\bcomponent\s+of\b", re.IGNORECASE), "PART_OF"),
    (re.compile(r"\bcontains?\b|\bincludes?\b|\bcomprises?\b", re.IGNORECASE), "CONTAINS"),
    (re.compile(r"\bimplements?\b|\bexecutes?\b|\bruns?\b", re.IGNORECASE), "IMPLEMENTS"),
    (re.compile(r"\bmeasured\s+by\b|\bevaluated\s+(?:by|with)\b|\bscored\s+by\b", re.IGNORECASE), "MEASURED_BY"),
    (re.compile(r"\bproduces?\b|\bgenerates?\b|\boutputs?\b|\bextracts?\b", re.IGNORECASE), "PRODUCES"),
    (re.compile(r"\boptimizes?\b|\baccelerates?\b|\bimproves?\b|\bboosts?\b", re.IGNORECASE), "OPTIMIZES"),
    (re.compile(r"\bconnects?\s+to\b|\bcommunicates?\s+with\b|\btransfers?\s+(?:to|across)\b", re.IGNORECASE), "TRANSFERS_TO"),
)


def extract_entities_deterministic(
    text: str,
    max_entities: int = 12,
) -> dict[str, list[dict[str, Any]]]:
    """Fast deterministic entity and relationship extractor requiring zero external API calls."""
    if not text or not text.strip():
        return {"entities": [], "relationships": []}

    entities_by_id: dict[str, dict[str, Any]] = {}
    mention_order: list[str] = []

    # 1. Known technical / domain terms (case-insensitive boundary match)
    lower_text = text.lower()
    for term, ent_type in _TECH_KEYWORDS.items():
        if re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", lower_text):
            canonical_name = term.upper() if len(term) <= 5 and term.isalpha() else term.title()
            if term == "bge-m3":
                canonical_name = "BGE-M3"
            elif term in ("next.js", "nextjs"):
                canonical_name = "Next.js"
            elif term == "fastapi":
                canonical_name = "FastAPI"
            elif term == "langchain":
                canonical_name = "LangChain"
            elif term == "langsmith":
                canonical_name = "LangSmith"
            elif term == "paddleocr":
                canonical_name = "PaddleOCR"
            elif term == "pymupdf":
                canonical_name = "PyMuPDF"
            elif term == "pypdf":
                canonical_name = "PyPDF"
            elif term == "deepeval":
                canonical_name = "DeepEval"
            elif term == "minilm":
                canonical_name = "MiniLM-L6-v2"
            elif term == "pcie":
                canonical_name = "PCIe"
            elif term == "nvlink":
                canonical_name = "NVLink"
            elif term == "neo4j":
                canonical_name = "Neo4j"
            elif term == "postgresql":
                canonical_name = "PostgreSQL"

            eid = deterministic_entity_id(ent_type, canonical_name)
            if eid not in entities_by_id:
                entities_by_id[eid] = {"id": eid, "name": canonical_name, "type": ent_type}
                mention_order.append(eid)

    # 2. Proper noun phrases and domain acronyms
    for match in _PROPER_PHRASE_RE.finditer(text):
        candidate = normalize_entity_name(match.group(0))
        if len(candidate) < 3 or candidate in _STOP_PROPER:
            continue
        first_word = candidate.split()[0]
        if first_word in _STOP_PROPER and " " not in candidate:
            continue
        if candidate.lower() in _TECH_KEYWORDS:
            continue

        ent_type = _classify_surface_entity(candidate)
        eid = deterministic_entity_id(ent_type, candidate)
        if eid not in entities_by_id:
            entities_by_id[eid] = {"id": eid, "name": candidate, "type": ent_type}
            mention_order.append(eid)
        if len(entities_by_id) >= max_entities:
            break

    selected_ids = mention_order[:max_entities]
    entities = [entities_by_id[eid] for eid in selected_ids]

    # 3. Sentence-level typed & co-occurrence relationships
    relationships_map: dict[tuple[str, str, str], dict[str, Any]] = {}
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text)

    for sent in sentences:
        if len(sent) < 12:
            continue
        sent_lower = sent.lower()
        present = [
            ent for ent in entities
            if ent["name"].lower() in sent_lower
        ]
        if len(present) < 2:
            continue

        rel_label = "CO_OCCURS_WITH"
        for pattern, candidate_rel in _RELATION_PATTERNS:
            if pattern.search(sent):
                rel_label = candidate_rel
                break

        for src, dst in combinations(present[:5], 2):
            if src["id"] == dst["id"]:
                continue
            key = (src["id"], dst["id"], rel_label)
            if key not in relationships_map:
                relationships_map[key] = {
                    "source_id": src["id"],
                    "target_id": dst["id"],
                    "type": rel_label,
                }

    return {
        "entities": entities,
        "relationships": list(relationships_map.values())[:20],
    }


def _classify_surface_entity(name: str) -> str:
    upper = name.upper()
    if any(tok in upper for tok in ("INC", "CORP", "LTD", "LLC", "TRADERS", "COMPANY", "GROUP", "BANK", "LABS")):
        return "Organization"
    if any(tok in upper for tok in ("LATENCY", "BANDWIDTH", "THROUGHPUT", "PRECISION", "RECALL", "ACCURACY", "RATE", "SCORE")):
        return "Metric"
    if any(tok in upper for tok in ("ALGORITHM", "FUSION", "SEARCH", "RANK", "SORT", "HASH", "EMBEDDING")):
        return "Algorithm"
    if any(tok in upper for tok in ("CPU", "GPU", "MEMORY", "CACHE", "REGISTER", "BUS", "DEVICE", "CORE", "SM")):
        return "Hardware"
    if any(tok in upper for tok in ("PIPELINE", "ARCHITECTURE", "ENGINE", "ROUTER", "GRAPH", "CLUSTER", "SERVER")):
        return "Architecture"
    if any(tok in upper for tok in ("API", "SDK", "KERNEL", "WARP", "THREAD", "VECTOR", "INDEX", "STREAM", "CHUNK")):
        return "Concept"
    if "()" in name or "_" in name or (any(c.islower() for c in name) and any(c.isupper() for c in name[1:])):
        return "CodeSymbol"
    if name.isupper() and 2 <= len(name) <= 8:
        return "Technology"
    return "Concept"


def canonicalize_llm_extraction(raw: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Normalize arbitrary LLM-returned entity/relationship JSON to deterministic UUIDv5 IDs."""
    raw_entities = raw.get("entities") or []
    raw_rels = raw.get("relationships") or []

    old_id_to_new: dict[str, str] = {}
    name_to_new: dict[str, str] = {}
    entities_by_id: dict[str, dict[str, Any]] = {}

    for item in raw_entities:
        if not isinstance(item, dict):
            continue
        name = normalize_entity_name(str(item.get("name") or ""))
        if len(name) < 2:
            continue
        ent_type = str(item.get("type") or "CONCEPT").strip().upper() or "CONCEPT"
        new_id = deterministic_entity_id(ent_type, name)
        old_id = str(item.get("id") or "")
        if old_id:
            old_id_to_new[old_id] = new_id
        name_to_new[name.lower()] = new_id
        entities_by_id[new_id] = {"id": new_id, "name": name, "type": ent_type}

    relationships: list[dict[str, Any]] = []
    seen_rels: set[tuple[str, str, str]] = set()
    for rel in raw_rels:
        if not isinstance(rel, dict):
            continue
        raw_src = str(rel.get("source_id") or rel.get("source") or "")
        raw_dst = str(rel.get("target_id") or rel.get("target") or "")
        src_id = old_id_to_new.get(raw_src) or name_to_new.get(raw_src.lower())
        dst_id = old_id_to_new.get(raw_dst) or name_to_new.get(raw_dst.lower())
        if not src_id or not dst_id or src_id == dst_id:
            continue
        rel_type = re.sub(r"[^A-Z0-9_]+", "_", str(rel.get("type") or "RELATED").upper()).strip("_") or "RELATED"
        key = (src_id, dst_id, rel_type)
        if key not in seen_rels:
            seen_rels.add(key)
            relationships.append({"source_id": src_id, "target_id": dst_id, "type": rel_type})

    return {
        "entities": list(entities_by_id.values()),
        "relationships": relationships,
    }


class HybridEntityExtractor:
    """Combines deterministic extraction with optional LLM enrichment (capped for latency)."""

    def __init__(self, llm: LLMPort | None = None, max_llm_chunks_per_doc: int = 8) -> None:
        self._llm = llm
        self._max_llm_chunks = max_llm_chunks_per_doc

    def _llm_is_usable(self) -> bool:
        if self._llm is None:
            return False
        is_avail = getattr(self._llm, "is_available", None)
        if callable(is_avail):
            return bool(is_avail())
        # Legacy GeminiLLMClient / GroqLLMClient store `_api_keys`
        keys = getattr(self._llm, "_api_keys", None)
        if isinstance(keys, list):
            return len(keys) > 0
        return True

    async def extract_for_chunks(
        self,
        chunk_texts: list[str],
        use_llm: bool = True,
    ) -> list[dict[str, list[dict[str, Any]]]]:
        """Extract canonicalized entities and relationships for a batch of chunks concurrently."""
        results: list[dict[str, list[dict[str, Any]]]] = []
        llm_ready = use_llm and self._llm_is_usable()
        eligible_indices: list[int] = []

        for idx, text in enumerate(chunk_texts):
            det = extract_entities_deterministic(text)
            merged_entities = {e["id"]: e for e in det["entities"]}
            merged_rels = {(r["source_id"], r["target_id"], r["type"]): r for r in det["relationships"]}
            results.append({"_entities": merged_entities, "_rels": merged_rels})  # type: ignore[dict-item]
            if llm_ready and len(eligible_indices) < self._max_llm_chunks and len(text.strip()) >= 80:
                eligible_indices.append(idx)

        if eligible_indices and self._llm is not None:
            sem = asyncio.Semaphore(4)

            async def _enrich_one(chunk_idx: int) -> tuple[int, dict[str, Any] | None]:
                async with sem:
                    try:
                        raw = await asyncio.wait_for(
                            self._llm.extract_entities(chunk_texts[chunk_idx]),  # type: ignore[union-attr]
                            timeout=12.0,
                        )
                        return chunk_idx, raw
                    except Exception as exc:
                        logger.debug("LLM entity enrichment skipped on chunk %d (%s); using deterministic entities", chunk_idx, exc)
                        return chunk_idx, None

            enriched = await asyncio.gather(*(_enrich_one(i) for i in eligible_indices))
            for chunk_idx, raw_llm in enriched:
                if not isinstance(raw_llm, dict):
                    continue
                canon = canonicalize_llm_extraction(raw_llm)
                target = results[chunk_idx]
                merged_entities = target["_entities"]  # type: ignore[index]
                merged_rels = target["_rels"]  # type: ignore[index]
                for e in canon["entities"]:
                    merged_entities[e["id"]] = e
                for r in canon["relationships"]:
                    merged_rels[(r["source_id"], r["target_id"], r["type"])] = r

        return [
            {
                "entities": list(item["_entities"].values()),  # type: ignore[attr-defined]
                "relationships": list(item["_rels"].values()),  # type: ignore[attr-defined]
            }
            for item in results
        ]

