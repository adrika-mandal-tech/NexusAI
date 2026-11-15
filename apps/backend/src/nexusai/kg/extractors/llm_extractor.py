"""LLM-based entity and relationship extraction from text.

Uses the configured LLM (Gemini or Groq) to extract structured entities
and their relationships from parsed document chunks, in a format suitable
for importing into Neo4j.
"""

from __future__ import annotations

import json
import re
from typing import Any

from nexusai.core.config import get_settings
from nexusai.core.logging import get_logger
from nexusai.domain.ports.llm_port import LLMPort

logger = get_logger(__name__)


class ExtractionError(RuntimeError):
    """Raised when entity/relationship extraction fails."""


class ExtractionConfig:
    """Configuration for the extraction pipeline."""

    def __init__(
        self,
        use_llm: bool = True,
        llm_provider: str | None = None,
        llm_model: str | None = None,
        top_k_entities: int = 20,
        top_k_relationships: int = 15,
    ) -> None:
        self.use_llm = use_llm
        self.llm_provider = llm_provider or get_settings().LLM_PROVIDER
        self.llm_model = llm_model or get_settings().DEFAULT_GEMINI_MODEL
        self.top_k_entities = top_k_entities
        self.top_k_relationships = top_k_relationships

    @classmethod
    def from_datasource(cls, datasource: Any) -> "ExtractionConfig":
        return cls(
            use_llm=getattr(datasource, "use_llm_extraction", True),
            llm_provider=getattr(datasource, "llm_provider", None),
            llm_model=getattr(datasource, "llm_model", None),
            top_k_entities=getattr(datasource, "top_k_entities", 20),
            top_k_relationships=getattr(datasource, "top_k_relationships", 15),
        )


class LLMEntityExtractor:
    """Extract entities and relationships from a chunk of text using an LLM."""

    def __init__(self, llm: LLMPort, config: ExtractionConfig | None = None) -> None:
        self._llm = llm
        self._config = config or ExtractionConfig()

    def extract(self, chunk_text: str, document_id: str, chunk_id: str) -> dict[str, Any]:
        """Extract entities and relationships from a chunk of text."""
        if not self._config.use_llm:
            return self._fallback_extract(chunk_text, document_id, chunk_id)

        try:
            prompt = self._build_prompt(chunk_text, document_id, chunk_id)
            response = self._llm.generate(prompt, temperature=0.2, max_tokens=8192)
            raw = response.strip()
            logger.debug("LLM extraction raw response: %s", raw[:500])

            try:
                parsed = self._parse_llm_response(raw)
            except ExtractionError as exc:
                logger.warning("LLM JSON parse failed: %s", exc)
                parsed = self._fallback_extract(chunk_text, document_id, chunk_id)

            self._validate_and_transform(parsed, document_id, chunk_id)
            parsed.setdefault("entity_count", len(parsed.get("entities", [])))
            parsed.setdefault("relationship_count", len(parsed.get("relationships", [])))
            return parsed
        except Exception as exc:
            logger.warning("Entity extraction failed: %s", exc)
            return self._fallback_extract(chunk_text, document_id, chunk_id)

    def _validate_and_transform(
        self,
        parsed: dict[str, Any],
        document_id: str,
        chunk_id: str,
    ) -> None:
        entities = parsed.get("entities", [])
        relationships = parsed.get("relationships", [])
        if not isinstance(entities, list):
            entities = []
        if not isinstance(relationships, list):
            relationships = []

        norm_entities: list[dict[str, Any]] = []
        seen_names: set[str] = set()
        for rec in entities:
            if not isinstance(rec, dict):
                continue
            name = str(rec.get("name", "")).strip()
            etype = str(rec.get("entity_type") or rec.get("type") or "entity").strip()
            desc = rec.get("description", None)
            if not name or name in seen_names:
                continue
            seen_names.add(name)
            norm_entities.append(
                {
                    "name": name,
                    "entity_type": etype,
                    "description": str(desc) if desc is not None else None,
                }
            )

        norm_relationships: list[dict[str, Any]] = []
        for rec in relationships:
            if not isinstance(rec, dict):
                continue
            source = str(rec.get("source", "")).strip()
            target = str(rec.get("target", "")).strip()
            rel_type = str(rec.get("type", "RELATED")).strip()
            if not source or not target or source == target:
                continue
            if source not in seen_names or target not in seen_names:
                continue
            norm_relationships.append({"source": source, "target": target, "type": rel_type})

        parsed["entities"] = norm_entities
        parsed["relationships"] = norm_relationships

    def _fallback_extract(self, text: str, document_id: str, chunk_id: str) -> dict[str, Any]:
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        entities: dict[str, dict[str, Any]] = {}
        relationships: dict[str, dict[str, Any]] = {}

        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            candidates = re.findall(r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\b", sent)
            for cand in candidates:
                if len(cand.split()) >= 2 and len(cand) < 40 and cand not in entities:
                    entities[cand] = {"name": cand, "entity_type": "concept"}

        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            matches = re.findall(r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\b", sent)
            if len(matches) >= 2:
                src = matches[0]
                tgt = matches[-1]
                if src != tgt:
                    relationships[f"{src}|{tgt}"] = {
                        "source": src,
                        "target": tgt,
                        "type": "INTERACTS_WITH",
                    }

        return {
            "entities": list(entities.values())[: self._config.top_k_entities],
            "relationships": list(relationships.values())[: self._config.top_k_relationships],
            "entity_count": len(entities),
            "relationship_count": len(relationships),
        }

    def _build_prompt(self, text: str, document_id: str, chunk_id: str) -> str:
        return (
            "You are a knowledge-graph extraction engine. Given a piece of text,\n"
            "extract structured entities and the relationships between them.\n"
            "Return ONLY valid JSON. Do not include any markdown, explanations, or code fences.\n\n"
            "Output schema:\n"
            "{\n"
            '  "entities": [\n'
            '    {"name": "<string>", "type": "<string>", "description": "<optional string>"},\n'
            "    ...\n"
            "  ],\n"
            '  "relationships": [\n'
            '    {"source": "<entity name>", "target": "<entity name>", "type": "<relationship type>"},\n'
            "    ...\n"
            "  ]\n"
            "}\n\n"
            "Rules:\n"
            f"- Extract at most {self._config.top_k_entities} distinct entities.\n"
            f"- Extract at most {self._config.top_k_relationships} relationships.\n"
            "- Entity types should be concise (e.g. 'person', 'company', 'location', 'product', 'organization').\n"
            "- Relationship types should be verbs or verb phrases (e.g. 'founded_by', 'located_in', 'purchased').\n"
            "- Relationship sources and targets MUST match entity names exactly.\n"
            "- Entities must have unique names; dedupe by name, keeping the first occurrence.\n"
            "- A relationship must reference existing entity names; ignore orphan references.\n\n"
            f"TEXT ({document_id}/{chunk_id}):\n---\n{text}\n---"
        )

    def _parse_llm_response(self, raw: str) -> dict[str, Any]:
        text = raw.strip()
        if "```" in text:
            match = re.search(r"```(?:json)?\s*(.{1,})```", text, re.DOTALL)
            if match:
                text = match.group(1).strip()

        start = text.find("{")
        if start == -1:
            raise ExtractionError("No JSON object found in LLM response")
        text = text[start:].strip()

        brace_count = 0
        end = len(text)
        for i, ch in enumerate(text):
            if ch == "{":
                brace_count += 1
            elif ch == "}":
                brace_count -= 1
                if brace_count == 0:
                    end = i + 1
                    break
        text = text[:end]

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ExtractionError(f"Invalid JSON in LLM response: {exc}") from exc

        if not isinstance(parsed, dict):
            raise ExtractionError("LLM response is not a JSON object")

        entities = parsed.get("entities", [])
        relationships = parsed.get("relationships", [])
        if not isinstance(entities, list):
            raise ExtractionError("'entities' must be an array")
        if not isinstance(relationships, list):
            raise ExtractionError("'relationships' must be an array")

        return {"entities": entities, "relationships": relationships}
