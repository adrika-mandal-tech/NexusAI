"""Deterministic identifier generation shared across PostgreSQL, Qdrant, and Neo4j.

Ensures idempotent re-indexing, cross-database lineage (`document_id`,
`chunk_id`, `entity_id`), and deduplication across chunks and documents.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from pathlib import Path

# Stable application-specific UUIDv5 namespace for NexusAI artifacts.
NEXUSAI_NAMESPACE = uuid.UUID("7f9c2d10-4b8e-4e92-9a1d-3c6b5f8e2a41")

_WHITESPACE_RE = re.compile(r"\s+")


def compute_sha256_bytes(data: bytes) -> str:
    """Return hex SHA-256 digest for in-memory bytes."""
    return hashlib.sha256(data).hexdigest()


def compute_sha256_file(file_path: str | Path, chunk_bytes: int = 1024 * 1024) -> str:
    """Stream-compute hex SHA-256 digest of a file on disk."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as fh:
        while block := fh.read(chunk_bytes):
            hasher.update(block)
    return hasher.hexdigest()


def deterministic_document_id(user_id: uuid.UUID | str, content_hash: str) -> uuid.UUID:
    """Generate a deterministic UUIDv5 for a document from owner + SHA-256 content hash."""
    return uuid.uuid5(NEXUSAI_NAMESPACE, f"doc:{user_id}:{content_hash}")


def deterministic_chunk_id(
    document_id: uuid.UUID | str,
    chunk_index: int,
    text: str,
) -> uuid.UUID:
    """Generate a deterministic UUIDv5 for a chunk suitable for Qdrant & Neo4j."""
    norm_text = _WHITESPACE_RE.sub(" ", (text or "").strip())
    digest_prefix = hashlib.sha256(norm_text.encode("utf-8")).hexdigest()[:16]
    return uuid.uuid5(NEXUSAI_NAMESPACE, f"chunk:{document_id}:{chunk_index}:{digest_prefix}")


def deterministic_parent_chunk_id(
    document_id: uuid.UUID | str,
    parent_index: int,
) -> uuid.UUID:
    """Generate a deterministic UUIDv5 for a hierarchical parent chunk."""
    return uuid.uuid5(NEXUSAI_NAMESPACE, f"parent_chunk:{document_id}:{parent_index}")


def normalize_entity_name(name: str) -> str:
    """Canonicalize an entity surface string for cross-chunk deduplication."""
    cleaned = _WHITESPACE_RE.sub(" ", (name or "").strip())
    return cleaned


def deterministic_entity_id(entity_type: str, entity_name: str) -> str:
    """Generate a deterministic UUIDv5 string for a Knowledge Graph entity."""
    norm_type = _WHITESPACE_RE.sub("_", (entity_type or "CONCEPT").strip().upper())
    norm_name = normalize_entity_name(entity_name).lower()
    return str(uuid.uuid5(NEXUSAI_NAMESPACE, f"entity:{norm_type}:{norm_name}"))

