"""Phase 5 Verification Tests — Configurable Embeddings (BGE-M3 & MiniLM) and Qdrant Vector Store."""

from __future__ import annotations

import uuid

import pytest

from nexusai.core.ids import deterministic_chunk_id
from nexusai.infrastructure.embedding.provider import (
    ConfigurableEmbeddingService,
    EmbeddingConfigurationError,
)
from nexusai.infrastructure.vector_stores.qdrant_store import (
    EmbeddingCollectionMismatchError,
    QdrantVectorStore,
)


def test_both_bge_m3_and_minilm_models_produce_expected_dimensions_and_sparse() -> None:
    minilm = ConfigurableEmbeddingService(
        provider="sentence-transformers",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    )
    assert minilm.get_dimension() == 384
    vecs_384 = minilm.embed_texts(["CUDA warp divergence reduces GPU throughput.", "Northwind customer orders"])
    assert len(vecs_384) == 2 and len(vecs_384[0]) == 384

    # Query LRU cache check
    q1 = minilm.embed_query("CUDA warp divergence")
    q2 = minilm.embed_query("  CUDA   warp divergence ")
    assert q1 == q2
    assert minilm.describe()["cached_queries"] == 1

    bge = ConfigurableEmbeddingService(
        provider="bge-m3",
        model_name="BAAI/bge-m3",
    )
    assert bge.get_dimension() == 1024
    q_1024 = bge.embed_query("Hybrid retrieval with Qdrant and Neo4j")
    assert len(q_1024) == 1024

    sparse = bge.embed_sparse(["CUDA_ERROR_OUT_OF_MEMORY on device 0"])
    assert len(sparse) == 1
    assert len(sparse[0]["indices"]) > 0
    assert len(sparse[0]["indices"]) == len(sparse[0]["values"])


def test_embedding_service_rejects_mismatched_provider_without_silent_substitution() -> None:
    with pytest.raises(EmbeddingConfigurationError):
        ConfigurableEmbeddingService(
            provider="bge-m3",
            model_name="sentence-transformers/all-MiniLM-L6-v2",
        )

    with pytest.raises(EmbeddingConfigurationError):
        ConfigurableEmbeddingService(
            provider="unknown-provider",
            model_name="BAAI/bge-m3",
        )


@pytest.mark.asyncio
async def test_qdrant_model_scoped_collections_dense_sparse_and_dimension_guard() -> None:
    minilm = ConfigurableEmbeddingService(
        provider="sentence-transformers",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    )
    bge = ConfigurableEmbeddingService(
        provider="bge-m3",
        model_name="BAAI/bge-m3",
    )

    test_base = "nexusai_pytest_vec"
    store_384 = QdrantVectorStore(
        collection_name=test_base,
        embedding_model=minilm.model_name,
        expected_dimension=384,
        auto_scope_by_model=True,
    )
    store_1024 = QdrantVectorStore(
        collection_name=test_base,
        embedding_model=bge.model_name,
        expected_dimension=1024,
        auto_scope_by_model=True,
    )
    assert store_384.collection_name != store_1024.collection_name

    doc_id = uuid.uuid4()
    texts = [
        "Coalesced global memory accesses maximize CUDA memory bandwidth on NVIDIA GPUs.",
        "Northwind Traders shipped 42 boxes of Chai tea via Speedy Express.",
    ]
    dense_384 = await minilm.aembed_texts(texts)
    sparse_vecs = await minilm.aembed_sparse(texts)

    chunks_payload = [
        {
            "chunk_id": deterministic_chunk_id(doc_id, idx, txt),
            "document_id": doc_id,
            "chunk_index": idx,
            "vector": dense_384[idx],
            "sparse_vector": sparse_vecs[idx],
            "text": txt,
            "page_number": idx + 1,
            "section_title": "Memory Optimization" if idx == 0 else "Order Logistics",
            "filename": "test_guide.pdf",
            "source_type": "pdf",
            "entity_ids": [f"ent-{idx}"],
            "metadata": {"parser_used": "pypdf"},
        }
        for idx, txt in enumerate(texts)
    ]

    try:
        await store_384.upsert_chunks(chunks_payload)

        # 1. Dense search
        q_vec = await minilm.aembed_query("How to maximize GPU memory bandwidth in CUDA?")
        dense_hits = await store_384.search(q_vec, limit=2, filter_metadata={"document_id": str(doc_id)})
        assert len(dense_hits) == 2
        assert "CUDA" in dense_hits[0]["text"]
        assert dense_hits[0]["page_number"] == 1
        assert dense_hits[0]["embedding_model"] == minilm.model_name

        # 2. Sparse search
        q_sparse = (await minilm.aembed_sparse(["Northwind Chai Speedy Express"]))[0]
        sparse_hits = await store_384.search_sparse(q_sparse, limit=2, filter_metadata={"document_id": str(doc_id)})
        assert len(sparse_hits) >= 1
        assert "Northwind" in sparse_hits[0]["text"]

        # 3. Dimension mismatch guard: trying to bind 1024d BGE-M3 directly onto the 384d collection MUST fail
        incompatible_store = QdrantVectorStore(
            collection_name=store_384.collection_name,
            embedding_model=bge.model_name,
            expected_dimension=1024,
            auto_scope_by_model=False,
        )
        with pytest.raises(EmbeddingCollectionMismatchError, match="incompatible"):
            await incompatible_store.ensure_collection(vector_size=1024)

        # 4. Separate 1024d collection works cleanly in parallel
        bge_vec = await bge.aembed_texts([texts[0]])
        await store_1024.upsert_chunks(
            [
                {
                    "chunk_id": deterministic_chunk_id(doc_id, 0, texts[0]),
                    "document_id": doc_id,
                    "chunk_index": 0,
                    "vector": bge_vec[0],
                    "sparse_vector": sparse_vecs[0],
                    "text": texts[0],
                    "page_number": 1,
                    "filename": "test_guide.pdf",
                    "source_type": "pdf",
                }
            ]
        )
        hits_1024 = await store_1024.search(bge_vec[0], limit=1)
        assert len(hits_1024) == 1
        assert hits_1024[0]["embedding_model"] == "BAAI/bge-m3"

        # 5. Deletion verification
        await store_384.delete_document_chunks(doc_id)
        remaining = await store_384.get_document_chunks(doc_id)
        assert remaining == []
    finally:
        # Clean up ephemeral pytest collections
        for col in (store_384.collection_name, store_1024.collection_name):
            try:
                store_384._client.delete_collection(col)
            except Exception:
                pass

