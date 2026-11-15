"""Neo4j entity/relationship loader — converts domain objects to normalized dicts."""

from __future__ import annotations

import hashlib
from typing import Any

from nexusai.kg.types import Entity, Relationship


def _uuid_from_name(name: str, prefix: str) -> str:
    """Deterministic opaque identifier for a Neo4j node."""
    raw = f"{prefix}:{name}".encode("utf-8")
    digest = hashlib.sha1(raw).hexdigest()[:12]
    return f"{prefix}:{name}:{digest}"


def _normalize_entity(entity: Any) -> dict[str, Any]:
    """Normalize an entity record (Entity model or dict) into a canonical dict."""
    if isinstance(entity, Entity):
        return {"name": entity.name, "entity_type": entity.entity_type, "description": entity.description}
    if isinstance(entity, dict):
        return {
            "name": entity.get("name", ""),
            "entity_type": entity.get("type", entity.get("entity_type", "entity")),
            "description": entity.get("description"),
        }
    return {"name": "", "entity_type": "entity", "description": None}


def _normalize_relationship(rel: Any, entity_name_to_id: dict[str, str]) -> dict[str, Any] | None:
    """Normalize a relationship record, resolving entity names to IDs."""
    if isinstance(rel, Relationship):
        source = rel.source_id
        target = rel.target_id
        rel_type = rel.relationship_type
    elif isinstance(rel, dict):
        source = rel.get("source_id", rel.get("source", ""))
        target = rel.get("target_id", rel.get("target", ""))
        rel_type = rel.get("type", rel.get("relationship_type", "RELATED"))
    else:
        return None

    source = str(source).strip()
    target = str(target).strip()
    rel_type = str(rel_type).strip()
    if not source or not target or source == target:
        return None
    source_id = entity_name_to_id.get(source)
    target_id = entity_name_to_id.get(target)
    if not source_id or not target_id:
        return None
    return {"source_id": source_id, "target_id": target_id, "relationship_type": rel_type}
