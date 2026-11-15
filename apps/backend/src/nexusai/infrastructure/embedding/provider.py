"""Configurable Embedding Provider supporting `BAAI/bge-m3` and `sentence-transformers/all-MiniLM-L6-v2`.

Features:
- Process-wide thread-safe model singleton registry (prevents multi-GB per-request reloads).
- Offline-first loading (`local_files_only=True` when weights are cached locally).
- Strict model validation with zero silent model substitution.
- Batch dense embedding + bounded LRU query embedding cache.
- Async threadpool helpers (`aembed_texts`, `aembed_query`, `aembed_sparse`).
- Deterministic lexical sparse vectors (`indices`, `values`) compatible with Qdrant `SparseVector`.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import os
import re
import threading
from collections import Counter, OrderedDict
from typing import Any

from nexusai.core.config import get_settings
from nexusai.core.logging import get_logger
from nexusai.domain.ports.embedding_port import EmbeddingPort

logger = get_logger(__name__)

KNOWN_EMBEDDING_MODELS: dict[str, dict[str, Any]] = {
    "BAAI/bge-m3": {
        "providers": {"bge-m3", "sentence-transformers", "huggingface", "openlm"},
        "expected_dimension": 1024,
        "slug": "bge_m3_1024",
    },
    "sentence-transformers/all-MiniLM-L6-v2": {
        "providers": {"sentence-transformers", "minilm", "huggingface", "openlm"},
        "expected_dimension": 384,
        "slug": "all_minilm_l6_v2_384",
    },
    "all-MiniLM-L6-v2": {
        "providers": {"sentence-transformers", "minilm", "huggingface", "openlm"},
        "expected_dimension": 384,
        "slug": "all_minilm_l6_v2_384",
        "canonical_name": "sentence-transformers/all-MiniLM-L6-v2",
    },
}

SUPPORTED_EMBEDDING_PROVIDERS = {"bge-m3", "sentence-transformers", "minilm", "huggingface", "openlm"}

_MODEL_REGISTRY_LOCK = threading.Lock()
_LOADED_MODELS: dict[str, tuple[Any, int]] = {}

_SPARSE_VOCAB_BUCKETS = 65_535
_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]{2,}")
_STOPWORDS = frozenset(
    {
        "the", "and", "for", "that", "this", "with", "from", "are", "was", "were",
        "have", "has", "had", "not", "but", "what", "all", "can", "will", "into",
        "than", "then", "them", "they", "their", "there", "about", "which", "when",
        "where", "who", "whom", "whose", "why", "how", "also", "been", "being",
    }
)


class EmbeddingConfigurationError(ValueError):
    """Raised when an invalid or incompatible embedding provider/model is configured."""


def build_lexical_sparse_vector(text: str) -> dict[str, Any]:
    """Compute a deterministic L2-normalized log-TF sparse vector for Qdrant hybrid search."""
    tokens = [t.lower() for t in _TOKEN_RE.findall(text or "")]
    if not tokens:
        return {"indices": [], "values": []}

    counts = Counter(tokens)
    bucket_weights: dict[int, float] = {}

    for tok, tf in counts.items():
        digest = hashlib.blake2b(tok.encode("utf-8"), digest_size=4).digest()
        idx = 1 + (int.from_bytes(digest, "little") % _SPARSE_VOCAB_BUCKETS)
        weight = 1.0 + math.log(tf)
        if tok in _STOPWORDS:
            weight *= 0.25
        elif any(ch.isdigit() for ch in tok) or "_" in tok:
            # Boost exact identifiers, error codes, SKUs, and version numbers
            weight *= 1.25
        bucket_weights[idx] = bucket_weights.get(idx, 0.0) + weight

    norm = math.sqrt(sum(w * w for w in bucket_weights.values())) or 1.0
    sorted_items = sorted(bucket_weights.items(), key=lambda kv: kv[0])
    return {
        "indices": [idx for idx, _ in sorted_items],
        "values": [round(val / norm, 6) for _, val in sorted_items],
    }


def _load_sentence_transformer_once(model_name: str) -> tuple[Any, int]:
    """Load a SentenceTransformer model at most once per process without silent substitution."""
    if model_name in _LOADED_MODELS:
        return _LOADED_MODELS[model_name]

    with _MODEL_REGISTRY_LOCK:
        if model_name in _LOADED_MODELS:
            return _LOADED_MODELS[model_name]

        settings = get_settings()
        if settings.HF_API_KEY:
            os.environ.setdefault("HF_TOKEN", settings.HF_API_KEY)

        from sentence_transformers import SentenceTransformer

        model = None
        try:
            # Prefer local HuggingFace cache first to avoid unnecessary network round-trips
            model = SentenceTransformer(model_name, local_files_only=True)
            logger.info("Loaded embedding model '%s' from local cache", model_name)
        except Exception:
            logger.info("Loading embedding model '%s' via SentenceTransformer...", model_name)
            model = SentenceTransformer(model_name)

        get_dim_fn = getattr(model, "get_embedding_dimension", None) or getattr(
            model, "get_sentence_embedding_dimension"
        )
        dim = int(get_dim_fn())
        spec = KNOWN_EMBEDDING_MODELS.get(model_name)
        if spec and dim != spec["expected_dimension"]:
            raise EmbeddingConfigurationError(
                f"Model '{model_name}' reported dimension {dim}, expected {spec['expected_dimension']}."
            )

        _LOADED_MODELS[model_name] = (model, dim)
        return model, dim


class ConfigurableEmbeddingService(EmbeddingPort):
    """Configurable embedding service supporting `BAAI/bge-m3`, `all-MiniLM-L6-v2`, `openlm`, and `huggingface`."""

    def __init__(
        self,
        provider: str | None = None,
        model_name: str | None = None,
        batch_size: int | None = None,
        cache_size: int | None = None,
        openlm_url: str | None = None,
    ) -> None:
        settings = get_settings()
        self.provider = (provider or settings.EMBEDDING_PROVIDER or "bge-m3").strip().lower()
        resolved_model = (model_name or settings.EMBEDDING_MODEL or "").strip()

        if resolved_model in KNOWN_EMBEDDING_MODELS and "canonical_name" in KNOWN_EMBEDDING_MODELS[resolved_model]:
            resolved_model = str(KNOWN_EMBEDDING_MODELS[resolved_model]["canonical_name"])

        if not resolved_model:
            if self.provider == "bge-m3":
                resolved_model = "BAAI/bge-m3"
            elif self.provider in ("sentence-transformers", "minilm"):
                resolved_model = "sentence-transformers/all-MiniLM-L6-v2"
            elif self.provider == "openlm":
                resolved_model = (settings.OPENLM_EMBEDDING_MODEL or "BAAI/bge-m3").strip()
            elif self.provider == "huggingface":
                resolved_model = "BAAI/bge-m3"
            else:
                raise EmbeddingConfigurationError(f"Unsupported EMBEDDING_PROVIDER '{self.provider}'.")

        self._validate_provider_and_model(self.provider, resolved_model)
        self.model_name = resolved_model
        self.openlm_url = (openlm_url if openlm_url is not None else settings.OPENLM_EMBEDDING_URL).strip()
        self.hf_api_key = (settings.HF_API_KEY or "").strip()
        self.batch_size = max(1, batch_size or settings.EMBEDDING_BATCH_SIZE or 32)
        self.cache_size = max(0, cache_size if cache_size is not None else settings.EMBEDDING_CACHE_SIZE)
        self._query_cache: OrderedDict[str, list[float]] = OrderedDict()
        self._cache_lock = threading.Lock()
        self._dimension: int | None = KNOWN_EMBEDDING_MODELS.get(self.model_name, {}).get("expected_dimension")

    @staticmethod
    def _validate_provider_and_model(provider: str, model_name: str) -> None:
        if provider not in SUPPORTED_EMBEDDING_PROVIDERS:
            raise EmbeddingConfigurationError(
                f"Unsupported EMBEDDING_PROVIDER '{provider}'. "
                f"Supported providers: {sorted(SUPPORTED_EMBEDDING_PROVIDERS)}."
            )
        if provider == "bge-m3" and model_name != "BAAI/bge-m3":
            raise EmbeddingConfigurationError(
                f"EMBEDDING_PROVIDER='bge-m3' requires EMBEDDING_MODEL='BAAI/bge-m3', got '{model_name}'. "
                "Use EMBEDDING_PROVIDER='sentence-transformers', 'huggingface', or 'openlm' for other models."
            )

    @property
    def collection_slug(self) -> str:
        """Stable slug for model-specific Qdrant collection naming."""
        known = KNOWN_EMBEDDING_MODELS.get(self.model_name)
        if known:
            return str(known["slug"])
        dim = self.get_dimension()
        safe = re.sub(r"[^a-z0-9]+", "_", self.model_name.lower()).strip("_")
        return f"{safe}_{dim}"

    def get_dimension(self) -> int:
        """Return validated dense vector dimension."""
        if self._dimension is not None and self.provider == "openlm" and self.openlm_url:
            return self._dimension
        if self.provider == "openlm" and self.openlm_url and self._dimension is None:
            probe = self._embed_via_openlm_endpoint(["dimension probe"])
            if probe:
                self._dimension = len(probe[0])
                return self._dimension
        model, dim = _load_sentence_transformer_once(self.model_name)
        _ = model
        self._dimension = dim
        return dim

    @staticmethod
    def _l2_normalize(vec: list[float]) -> list[float]:
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def _embed_via_openlm_endpoint(self, texts: list[str]) -> list[list[float]] | None:
        """Call OpenAI-compatible OpenLM `/embeddings` endpoint if `OPENLM_EMBEDDING_URL` is configured."""
        if not self.openlm_url:
            return None
        import httpx

        settings = get_settings()
        endpoint = self.openlm_url.rstrip("/")
        if not endpoint.endswith("/embeddings"):
            endpoint = f"{endpoint}/embeddings"
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if settings.OPENLM_API_KEY:
            headers["Authorization"] = f"Bearer {settings.OPENLM_API_KEY}"
        try:
            with httpx.Client(timeout=15.0) as client:
                resp = client.post(
                    endpoint,
                    headers=headers,
                    json={"model": self.model_name, "input": texts},
                )
                resp.raise_for_status()
                payload = resp.json()
                data = sorted(payload.get("data", []), key=lambda item: item.get("index", 0))
                vectors = [self._l2_normalize([float(x) for x in item["embedding"]]) for item in data]
                if len(vectors) == len(texts) and vectors:
                    self._dimension = len(vectors[0])
                    return vectors
        except Exception as exc:
            logger.warning(
                "OpenLM embedding endpoint (%s) unavailable (%s); falling back to local HuggingFace transformer '%s'",
                endpoint,
                exc,
                self.model_name,
            )
        return None

    def _embed_via_hf_inference_api(self, texts: list[str]) -> list[list[float]] | None:
        """Call Hugging Face Inference Router API when local model loading is unavailable."""
        if not self.hf_api_key:
            return None
        import httpx

        url = f"https://router.huggingface.co/hf-inference/models/{self.model_name}/pipeline/feature-extraction"
        headers = {
            "Authorization": f"Bearer {self.hf_api_key}",
            "Content-Type": "application/json",
        }
        try:
            with httpx.Client(timeout=20.0) as client:
                resp = client.post(
                    url,
                    headers=headers,
                    json={"inputs": texts, "options": {"wait_for_model": True}},
                )
                resp.raise_for_status()
                raw = resp.json()
                if isinstance(raw, list) and len(raw) == len(texts) and raw:
                    out: list[list[float]] = []
                    for item in raw:
                        # Handle [seq_len, hidden_dim] token embeddings via mean pooling if returned
                        if item and isinstance(item[0], list):
                            seq_len = max(1, len(item))
                            dim = len(item[0])
                            pooled = [sum(tok[d] for tok in item) / seq_len for d in range(dim)]
                            out.append(self._l2_normalize(pooled))
                        else:
                            out.append(self._l2_normalize([float(v) for v in item]))
                    self._dimension = len(out[0])
                    return out
        except Exception as exc:
            logger.warning("HuggingFace Inference API fallback warning for '%s': %s", self.model_name, exc)
        return None

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Batch-encode a list of texts into L2-normalized dense vectors."""
        if not texts:
            return []
        cleaned = [t if (t and t.strip()) else " " for t in texts]

        if self.provider == "openlm" and self.openlm_url:
            remote_vecs = self._embed_via_openlm_endpoint(cleaned)
            if remote_vecs is not None:
                return remote_vecs

        try:
            model, dim = _load_sentence_transformer_once(self.model_name)
            self._dimension = dim
            vectors = model.encode(
                cleaned,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            result: list[list[float]] = vectors.tolist()
            if result and len(result[0]) != dim:
                raise EmbeddingConfigurationError(
                    f"Embedding output dimension {len(result[0])} != expected {dim} for '{self.model_name}'."
                )
            return result
        except EmbeddingConfigurationError:
            raise
        except Exception as local_exc:
            if self.hf_api_key:
                hf_vecs = self._embed_via_hf_inference_api(cleaned)
                if hf_vecs is not None:
                    return hf_vecs
            raise local_exc

    def embed_query(self, query: str) -> list[float]:
        """Encode a single query with thread-safe bounded LRU caching."""
        norm_key = " ".join((query or "").strip().split())
        if self.cache_size > 0 and norm_key:
            with self._cache_lock:
                cached = self._query_cache.get(norm_key)
                if cached is not None:
                    self._query_cache.move_to_end(norm_key)
                    return list(cached)

        vec = self.embed_texts([norm_key or " "])[0]

        if self.cache_size > 0 and norm_key:
            with self._cache_lock:
                self._query_cache[norm_key] = vec
                self._query_cache.move_to_end(norm_key)
                while len(self._query_cache) > self.cache_size:
                    self._query_cache.popitem(last=False)

        return list(vec)

    def embed_sparse(self, texts: list[str]) -> list[dict[str, Any]]:
        """Generate deterministic lexical sparse vectors (`indices`, `values`) for hybrid retrieval."""
        return [build_lexical_sparse_vector(t) for t in texts]

    async def aembed_texts(self, texts: list[str]) -> list[list[float]]:
        """Non-blocking async batch dense embedding."""
        return await asyncio.to_thread(self.embed_texts, texts)

    async def aembed_query(self, query: str) -> list[float]:
        """Non-blocking async query dense embedding."""
        return await asyncio.to_thread(self.embed_query, query)

    async def aembed_sparse(self, texts: list[str]) -> list[dict[str, Any]]:
        """Non-blocking async sparse vector generation."""
        return await asyncio.to_thread(self.embed_sparse, texts)

    def describe(self) -> dict[str, Any]:
        """Return runtime metadata for Settings, Health, and Qdrant collection checks."""
        dim = self._dimension or self.get_dimension()
        return {
            "provider": self.provider,
            "model_name": self.model_name,
            "dimension": dim,
            "batch_size": self.batch_size,
            "cache_size": self.cache_size,
            "cached_queries": len(self._query_cache),
            "collection_slug": self.collection_slug,
            "hf_authenticated": bool(self.hf_api_key),
            "openlm_url_configured": bool(self.openlm_url),
            "supported_models": ["BAAI/bge-m3", "sentence-transformers/all-MiniLM-L6-v2"],
            "supported_providers": sorted(SUPPORTED_EMBEDDING_PROVIDERS),
        }

