"""Qdrant vector store with model-scoped collections, dimension guards, and dense+sparse retrieval.

Guarantees:
- Never mixes vectors from incompatible embedding models (e.g. 1024d `BAAI/bge-m3`
  vs 384d `sentence-transformers/all-MiniLM-L6-v2`).
- Stores `embedding_model` and `embedding_dimension` in point payloads and
  validates live collection vector configuration before writes and queries.
- Supports both dense cosine similarity search and named sparse (`lexical`) search.
- Indexes `document_id`, `chunk_id`, `filename`, `source_type`, and `embedding_model` payloads.
- Executes synchronous Qdrant SDK calls via `asyncio.to_thread` to avoid blocking FastAPI.
"""

from __future__ import annotations

import asyncio
import re
import uuid
from typing import Any
from uuid import UUID

from qdrant_client import QdrantClient as QdrantSDK
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PointStruct,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from nexusai.core.config import get_settings
from nexusai.core.logging import get_logger
from nexusai.domain.ports.vector_store import VectorStorePort

logger = get_logger(__name__)

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "lexical"


class EmbeddingCollectionMismatchError(RuntimeError):
    """Raised when an embedding model's vector dimension or identity conflicts with a Qdrant collection."""


def generate_point_id(chunk_id: UUID | str, text: str | None = None) -> str:
    """Return a canonical UUID string for Qdrant point IDs matching Neo4j `chunk_id`."""
    try:
        return str(UUID(str(chunk_id)))
    except ValueError:
        stable_name = f"{chunk_id}::{text or ''}"
        return str(uuid.uuid5(uuid.NAMESPACE_OID, stable_name))


def resolve_model_collection_name(
    base_collection: str,
    embedding_model: str,
    vector_size: int,
) -> str:
    """Derive an explicit model- and dimension-scoped Qdrant collection name."""
    slug = re.sub(r"[^a-z0-9]+", "_", embedding_model.lower()).strip("_")
    suffix = f"__{slug}_{vector_size}"
    if base_collection.endswith(suffix):
        return base_collection
    return f"{base_collection}{suffix}"


class QdrantVectorStore(VectorStorePort):
    """Qdrant implementation of VectorStorePort with dense + sparse hybrid support."""

    def __init__(
        self,
        collection_name: str | None = None,
        embedding_model: str | None = None,
        expected_dimension: int | None = None,
        auto_scope_by_model: bool = True,
        client: QdrantSDK | None = None,
    ) -> None:
        settings = get_settings()
        self._local_url = settings.QDRANT_URL or "http://localhost:6333"
        self._local_api_key = settings.QDRANT_API_KEY or None
        self._cloud_url = (settings.QDRANT_CLUSTER_ENDPOINT or "").strip()
        self._cloud_api_key = (settings.QDRANT_CLUSTER_API_KEY or settings.QDRANT_API_KEY) or None
        self._base_collection = collection_name or settings.QDRANT_COLLECTION
        self._embedding_model = embedding_model or settings.EMBEDDING_MODEL or "BAAI/bge-m3"
        self._expected_dimension = expected_dimension
        self._auto_scope_by_model = auto_scope_by_model
        self._backend_mode = "injected_client"
        self._cloud_failover_reason: str | None = None

        if client is not None:
            self._client = client
            self._url = self._cloud_url or self._local_url
            self._api_key = self._cloud_api_key or self._local_api_key
        else:
            self._client, self._url, self._api_key = self._init_resilient_client()

        if self._auto_scope_by_model and self._expected_dimension:
            self._collection = resolve_model_collection_name(
                self._base_collection,
                self._embedding_model,
                self._expected_dimension,
            )
        else:
            self._collection = self._base_collection

        self._uses_named_dense: bool = True
        self._supports_sparse: bool = True
        self._validated_size: int | None = None

    def _init_resilient_client(self) -> tuple[QdrantSDK, str, str | None]:
        """Connect to Qdrant Cloud when reachable; fail over seamlessly to local Qdrant if Cloud resets/times out."""
        if self._cloud_url:
            candidates = [self._cloud_url.rstrip("/")]
            if ":6333" not in candidates[0] and candidates[0].startswith("https://"):
                candidates.append(f"{candidates[0]}:6333")
            last_err = ""
            for cand_url in candidates:
                try:
                    probe = QdrantSDK(
                        url=cand_url,
                        api_key=self._cloud_api_key,
                        timeout=2.5,
                        check_compatibility=False,
                    )
                    probe.get_collections()
                    self._backend_mode = "qdrant_cloud"
                    logger.info("Connected to Qdrant Cloud cluster at %s", cand_url)
                    return (
                        QdrantSDK(
                            url=cand_url,
                            api_key=self._cloud_api_key,
                            timeout=15,
                            check_compatibility=False,
                        ),
                        cand_url,
                        self._cloud_api_key,
                    )
                except Exception as exc:
                    last_err = f"{type(exc).__name__}: {exc}"
            self._cloud_failover_reason = (
                f"Qdrant Cloud unreachable ({last_err[:160]}); automatically failed over to local Qdrant ({self._local_url})"
            )
            self._backend_mode = "local_qdrant_failover"
            logger.warning("%s", self._cloud_failover_reason)
        else:
            self._backend_mode = "local_qdrant"

        return (
            QdrantSDK(
                url=self._local_url,
                api_key=self._local_api_key,
                timeout=15,
                check_compatibility=False,
            ),
            self._local_url,
            self._local_api_key,
        )

    def _switch_to_local_on_cloud_drop(self, exc: Exception) -> bool:
        """If Qdrant Cloud drops mid-session, fail over dynamically to local Qdrant once."""
        if self._backend_mode == "qdrant_cloud" and self._local_url and self._url != self._local_url:
            self._cloud_failover_reason = f"Runtime Qdrant Cloud drop ({exc}); failed over to {self._local_url}"
            logger.warning("%s", self._cloud_failover_reason)
            self._client = QdrantSDK(
                url=self._local_url,
                api_key=self._local_api_key,
                timeout=15,
                check_compatibility=False,
            )
            self._url = self._local_url
            self._api_key = self._local_api_key
            self._backend_mode = "local_qdrant_failover"
            return True
        return False

    @property
    def collection_name(self) -> str:
        return self._collection

    @property
    def embedding_model(self) -> str:
        return self._embedding_model

    async def ensure_collection(self, vector_size: int = 1024) -> None:
        """Create or validate collection compatibility for `vector_size` and active model."""
        if self._expected_dimension and vector_size != self._expected_dimension:
            raise EmbeddingCollectionMismatchError(
                f"Vector size {vector_size} does not match configured embedding model "
                f"'{self._embedding_model}' dimension ({self._expected_dimension})."
            )
        if self._auto_scope_by_model:
            self._collection = resolve_model_collection_name(
                self._base_collection,
                self._embedding_model,
                vector_size,
            )
        await asyncio.to_thread(self._ensure_collection_sync, vector_size)

    def _ensure_collection_sync(self, vector_size: int) -> None:
        collections = self._client.get_collections().collections
        exists = any(c.name == self._collection for c in collections)

        if exists:
            info = self._client.get_collection(self._collection)
            vectors_cfg = info.config.params.vectors
            sparse_cfg = getattr(info.config.params, "sparse_vectors", None)

            if isinstance(vectors_cfg, dict):
                if DENSE_VECTOR_NAME in vectors_cfg:
                    actual_dim = vectors_cfg[DENSE_VECTOR_NAME].size
                    self._uses_named_dense = True
                else:
                    first_vec = next(iter(vectors_cfg.values()))
                    actual_dim = first_vec.size
                    self._uses_named_dense = True
            else:
                actual_dim = vectors_cfg.size  # type: ignore[union-attr]
                self._uses_named_dense = False

            self._supports_sparse = bool(sparse_cfg and SPARSE_VECTOR_NAME in sparse_cfg)

            if actual_dim != vector_size:
                suggested = resolve_model_collection_name(
                    self._base_collection, self._embedding_model, vector_size
                )
                raise EmbeddingCollectionMismatchError(
                    f"Qdrant collection '{self._collection}' has vector dimension {actual_dim}, "
                    f"which is incompatible with embedding model '{self._embedding_model}' ({vector_size}d). "
                    f"Migration instructions: switch to model-scoped collection '{suggested}' or trigger "
                    f"document re-indexing via POST /api/v1/documents/{{document_id}}/reindex."
                )

            # Check existing point payload model identifier if points exist
            if (info.points_count or 0) > 0:
                sample_pts, _ = self._client.scroll(
                    collection_name=self._collection,
                    limit=1,
                    with_payload=True,
                    with_vectors=False,
                )
                if sample_pts:
                    stored_model = (sample_pts[0].payload or {}).get("embedding_model")
                    if stored_model and stored_model != self._embedding_model:
                        suggested = resolve_model_collection_name(
                            self._base_collection, self._embedding_model, vector_size
                        )
                        raise EmbeddingCollectionMismatchError(
                            f"Collection '{self._collection}' contains vectors from model '{stored_model}', "
                            f"cannot mix with '{self._embedding_model}'. Use collection '{suggested}' and re-index."
                        )

            self._ensure_payload_indexes_sync()
            self._validated_size = vector_size
            return

        # Create new hybrid (named dense + named sparse) collection
        self._client.create_collection(
            collection_name=self._collection,
            vectors_config={
                DENSE_VECTOR_NAME: VectorParams(size=vector_size, distance=Distance.COSINE),
            },
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: SparseVectorParams(
                    index=SparseIndexParams(on_disk=False),
                ),
            },
        )
        self._uses_named_dense = True
        self._supports_sparse = True
        self._validated_size = vector_size
        self._ensure_payload_indexes_sync()
        logger.info(
            "Created Qdrant collection '%s' (model=%s, dim=%d, sparse=%s)",
            self._collection,
            self._embedding_model,
            vector_size,
            self._supports_sparse,
        )

    def _ensure_payload_indexes_sync(self) -> None:
        for field_name in (
            "document_id",
            "chunk_id",
            "filename",
            "source_type",
            "content_hash",
            "embedding_model",
        ):
            try:
                self._client.create_payload_index(
                    collection_name=self._collection,
                    field_name=field_name,
                    field_schema="keyword",
                )
            except Exception:
                pass

    async def upsert_chunks(self, chunks: list[dict[str, Any]]) -> None:
        """Validate dimensions and upsert chunks with dense + optional sparse embeddings."""
        if not chunks:
            return
        first_vec = chunks[0].get("vector") or []
        if not first_vec:
            raise ValueError("Cannot upsert chunk with empty dense vector.")
        await self.ensure_collection(vector_size=len(first_vec))
        await asyncio.to_thread(self._upsert_chunks_sync, chunks, len(first_vec))

    def _upsert_chunks_sync(self, chunks: list[dict[str, Any]], expected_dim: int) -> None:
        points: list[PointStruct] = []
        for chunk in chunks:
            dense_vec = chunk["vector"]
            if len(dense_vec) != expected_dim:
                raise EmbeddingCollectionMismatchError(
                    f"Chunk vector length {len(dense_vec)} != collection dimension {expected_dim}."
                )

            chunk_id_str = str(chunk["chunk_id"])
            point_id = generate_point_id(chunk_id_str, chunk.get("text"))
            meta = dict(chunk.get("metadata") or {})

            payload = {
                "chunk_id": chunk_id_str,
                "document_id": str(chunk["document_id"]),
                "chunk_index": chunk.get("chunk_index", 0),
                "text": chunk["text"],
                "page_number": chunk.get("page_number"),
                "section_title": chunk.get("section_title"),
                "filename": chunk.get("filename", meta.get("filename", "")),
                "source_type": chunk.get("source_type", meta.get("source_type", "")),
                "content_hash": chunk.get("content_hash", meta.get("content_hash", "")),
                "embedding_model": self._embedding_model,
                "embedding_dimension": expected_dim,
                "entity_ids": chunk.get("entity_ids", meta.get("entity_ids", [])),
                "metadata": meta,
            }

            if self._uses_named_dense:
                vec_payload: Any = {DENSE_VECTOR_NAME: dense_vec}
                sparse_raw = chunk.get("sparse_vector")
                if self._supports_sparse and sparse_raw and sparse_raw.get("indices"):
                    vec_payload[SPARSE_VECTOR_NAME] = SparseVector(
                        indices=list(sparse_raw["indices"]),
                        values=list(sparse_raw["values"]),
                    )
            else:
                vec_payload = dense_vec

            points.append(
                PointStruct(
                    id=point_id,
                    vector=vec_payload,
                    payload=payload,
                )
            )

        # Upsert in safe batches of 128 points
        for start in range(0, len(points), 128):
            self._client.upsert(
                collection_name=self._collection,
                points=points[start : start + 128],
                wait=True,
            )
        logger.info("Upserted %d chunks to Qdrant collection '%s'", len(points), self._collection)

    def _build_filter(self, filter_metadata: dict[str, Any] | None) -> Filter | None:
        if not filter_metadata:
            return None
        conditions = []
        for key, value in filter_metadata.items():
            if value is None:
                continue
            if key == "document_ids" and isinstance(value, (list, tuple, set)):
                str_ids = [str(v) for v in value if v]
                if str_ids:
                    conditions.append(FieldCondition(key="document_id", match=MatchAny(any=str_ids)))
            elif isinstance(value, (list, tuple, set)):
                vals = [str(v) for v in value if v is not None]
                if vals:
                    conditions.append(FieldCondition(key=key, match=MatchAny(any=vals)))
            else:
                conditions.append(FieldCondition(key=key, match=MatchValue(value=str(value) if isinstance(value, UUID) else value)))
        return Filter(must=conditions) if conditions else None

    async def search(
        self,
        query_vector: list[float],
        limit: int = 10,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Dense cosine vector similarity search."""
        if not query_vector:
            return []
        await self.ensure_collection(vector_size=len(query_vector))
        return await asyncio.to_thread(self._search_dense_sync, query_vector, limit, filter_metadata)

    def _search_dense_sync(
        self,
        query_vector: list[float],
        limit: int,
        filter_metadata: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        query_filter = self._build_filter(filter_metadata)
        kwargs: dict[str, Any] = {
            "collection_name": self._collection,
            "query": query_vector,
            "limit": limit,
            "query_filter": query_filter,
            "with_payload": True,
        }
        if self._uses_named_dense:
            kwargs["using"] = DENSE_VECTOR_NAME

        try:
            results = self._client.query_points(**kwargs)
        except Exception as exc:
            if self._switch_to_local_on_cloud_drop(exc):
                results = self._client.query_points(**kwargs)
            else:
                raise
        return [self._point_to_dict(r, channel="dense") for r in results.points]

    async def search_sparse(
        self,
        sparse_vector: dict[str, Any],
        limit: int = 10,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Lexical sparse vector search in Qdrant when supported by the collection."""
        indices = sparse_vector.get("indices") or []
        values = sparse_vector.get("values") or []
        if not indices or not values:
            return []
        return await asyncio.to_thread(self._search_sparse_sync, indices, values, limit, filter_metadata)

    def _search_sparse_sync(
        self,
        indices: list[int],
        values: list[float],
        limit: int,
        filter_metadata: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        collections = self._client.get_collections().collections
        if not any(c.name == self._collection for c in collections):
            return []
        if not self._supports_sparse:
            return []

        query_filter = self._build_filter(filter_metadata)
        results = self._client.query_points(
            collection_name=self._collection,
            query=SparseVector(indices=indices, values=values),
            using=SPARSE_VECTOR_NAME,
            limit=limit,
            query_filter=query_filter,
            with_payload=True,
        )
        return [self._point_to_dict(r, channel="sparse") for r in results.points]

    async def search_hybrid_native(
        self,
        query_vector: list[float],
        sparse_vector: dict[str, Any],
        limit: int = 10,
        prefetch_limit: int = 24,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Execute Qdrant native server-side Prefetch + RRF hybrid fusion when supported."""
        if not query_vector:
            return []
        await self.ensure_collection(vector_size=len(query_vector))
        indices = sparse_vector.get("indices") or []
        values = sparse_vector.get("values") or []
        if not indices or not values or not self._supports_sparse or not self._uses_named_dense:
            return await self.search(query_vector, limit=limit, filter_metadata=filter_metadata)
        return await asyncio.to_thread(
            self._search_hybrid_native_sync,
            query_vector,
            indices,
            values,
            limit,
            prefetch_limit,
            filter_metadata,
        )

    def _search_hybrid_native_sync(
        self,
        query_vector: list[float],
        indices: list[int],
        values: list[float],
        limit: int,
        prefetch_limit: int,
        filter_metadata: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        from qdrant_client.models import Fusion, FusionQuery, Prefetch

        query_filter = self._build_filter(filter_metadata)
        results = self._client.query_points(
            collection_name=self._collection,
            prefetch=[
                Prefetch(
                    query=query_vector,
                    using=DENSE_VECTOR_NAME,
                    limit=max(limit, prefetch_limit),
                    filter=query_filter,
                ),
                Prefetch(
                    query=SparseVector(indices=indices, values=values),
                    using=SPARSE_VECTOR_NAME,
                    limit=max(limit, prefetch_limit),
                    filter=query_filter,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=limit,
            with_payload=True,
        )
        return [self._point_to_dict(r, channel="qdrant_native_rrf") for r in results.points]

    async def get_document_chunks(
        self,
        document_id: UUID | str,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        """Scroll all indexed chunks for a specific `document_id` ordered by `chunk_index`."""
        return await asyncio.to_thread(self._get_document_chunks_sync, str(document_id), limit)

    def _get_document_chunks_sync(self, doc_id_str: str, limit: int) -> list[dict[str, Any]]:
        collections = self._client.get_collections().collections
        if not any(c.name == self._collection for c in collections):
            return []
        pts, _ = self._client.scroll(
            collection_name=self._collection,
            scroll_filter=Filter(
                must=[FieldCondition(key="document_id", match=MatchValue(value=doc_id_str))]
            ),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        items = [
            {
                "chunk_id": (p.payload or {}).get("chunk_id", str(p.id)),
                "document_id": (p.payload or {}).get("document_id", doc_id_str),
                "chunk_index": (p.payload or {}).get("chunk_index", 0),
                "score": 1.0,
                "text": (p.payload or {}).get("text", ""),
                "page_number": (p.payload or {}).get("page_number"),
                "section_title": (p.payload or {}).get("section_title"),
                "filename": (p.payload or {}).get("filename", ""),
                "entity_ids": (p.payload or {}).get("entity_ids", []),
                "metadata": (p.payload or {}).get("metadata", {}),
            }
            for p in pts
        ]
        items.sort(key=lambda x: int(x.get("chunk_index") or 0))
        return items

    async def delete_document_chunks(self, document_id: UUID) -> None:
        """Delete all chunks for a document from the active collection."""
        await asyncio.to_thread(self._delete_document_chunks_sync, str(document_id))

    def _delete_document_chunks_sync(self, doc_id_str: str) -> None:
        collections = self._client.get_collections().collections
        if not any(c.name == self._collection for c in collections):
            return
        self._client.delete(
            collection_name=self._collection,
            points_selector=Filter(
                must=[FieldCondition(key="document_id", match=MatchValue(value=doc_id_str))]
            ),
            wait=True,
        )
        logger.info("Deleted chunks for document %s from Qdrant '%s'", doc_id_str, self._collection)

    async def health(self) -> dict[str, Any]:
        """Check Qdrant connectivity and active collection statistics."""
        return await asyncio.to_thread(self._health_sync)

    def _health_sync(self) -> dict[str, Any]:
        try:
            cols = [c.name for c in self._client.get_collections().collections]
            points_count = 0
            if self._collection in cols:
                info = self._client.get_collection(self._collection)
                points_count = info.points_count or 0
            return {
                "status": "ok",
                "backend_mode": self._backend_mode,
                "active_endpoint": self._url,
                "cloud_failover_reason": self._cloud_failover_reason,
                "active_collection": self._collection,
                "embedding_model": self._embedding_model,
                "points_count": points_count,
                "collections": cols,
                "supports_sparse": self._supports_sparse,
            }
        except Exception as exc:
            if self._switch_to_local_on_cloud_drop(exc):
                return self._health_sync()
            logger.error("Qdrant health check failed: %s", exc)
            return {"status": "error", "detail": str(exc)}

    @staticmethod
    def _point_to_dict(point: Any, channel: str = "dense") -> dict[str, Any]:
        payload = point.payload or {}
        meta = payload.get("metadata") or {}
        return {
            "chunk_id": payload.get("chunk_id", str(point.id)),
            "document_id": payload.get("document_id"),
            "chunk_index": payload.get("chunk_index", 0),
            "score": float(point.score),
            "retrieval_channel": channel,
            "text": payload.get("text", ""),
            "page_number": payload.get("page_number"),
            "section_title": payload.get("section_title"),
            "filename": payload.get("filename", meta.get("filename", "")),
            "source_type": payload.get("source_type", meta.get("source_type", "")),
            "content_hash": payload.get("content_hash", meta.get("content_hash", "")),
            "embedding_model": payload.get("embedding_model", ""),
            "entity_ids": payload.get("entity_ids", []),
            "metadata": meta,
        }
