"""Application configuration via pydantic-settings.

This is the ONLY module that reads os.environ / .env files.
Everything else imports the Settings object from here.
"""

from functools import lru_cache

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration — values come from environment variables or .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ──────────────────────────────────────────────
    ENVIRONMENT: str = "development"
    APP_NAME: str = "nexusai-backend"
    API_V1_PREFIX: str = "/api/v1"
    LOG_LEVEL: str = "INFO"

    # ── Authentication / JWT ─────────────────────────────────────
    AUTH_SECRET_KEY: str = "nexusai-dev-secret-change-in-production"
    AUTH_ALGORITHM: str = "HS256"
    AUTH_TOKEN_EXPIRE_MINUTES: int = 10080  # 7 days

    # ── CORS ─────────────────────────────────────────────────────
    BACKEND_CORS_ORIGINS: list[str] = ["http://localhost:3000", "http://localhost:3001"]

    # ── LLM / Embeddings ─────────────────────────────────────────
    GOOGLE_API_KEYS: str = ""
    GEMINI_MODELS: str = "gemini-3.5-flash-lite,gemini-3.5-flash,gemini-3.1-flash-lite"
    DEFAULT_GEMINI_MODEL: str = "gemini-3.5-flash-lite"
    GROQ_API_KEYS: str = ""
    GROQ_MODELS: str = "openai/gpt-oss-120b,openai/gpt-oss-20b,qwen/qwen3.8-27b"
    DEFAULT_GROQ_MODEL: str = "openai/gpt-oss-120b"
    OPENLM_BASE_URL: str = ""
    OPENLM_API_KEY: str = ""
    OPENLM_MODEL: str = ""
    OPENLM_API_STYLE: str = "openai"  # "openai" (/v1/chat/completions) or "openllm_legacy" (/generate)
    OPENLM_EMBEDDING_URL: str = ""
    OPENLM_EMBEDDING_MODEL: str = "BAAI/bge-m3"
    LLM_PROVIDER: str = "auto"  # "auto" | "gemini" | "openlm" | "groq"

    # Configurable Embedding Service (BAAI/bge-m3, sentence-transformers/all-MiniLM-L6-v2, or openlm/huggingface)
    EMBEDDING_PROVIDER: str = "bge-m3"  # "bge-m3" | "sentence-transformers" | "openlm" | "huggingface"
    EMBEDDING_MODEL: str = "BAAI/bge-m3"
    EMBEDDING_BATCH_SIZE: int = 32
    EMBEDDING_CACHE_SIZE: int = 512

    # Document Processing & Chunking (Default 10% overlap: 800 tokens / 80 token overlap)
    DOC_MAX_UPLOAD_BYTES: int = 100 * 1024 * 1024
    CHUNK_STRATEGY: str = "structure_aware"  # "structure_aware" | "table_aware" | "recursive" | "token_aware" | "parent_child"
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 80
    OCR_MIN_CHARS_PER_PAGE: int = 60
    OCR_ENGINE: str = "auto"  # "auto" | "paddleocr" | "rapidocr"

    # Hybrid Retrieval
    RETRIEVAL_TOP_K: int = 6
    RETRIEVAL_RRF_K: int = 60
    ENABLE_RERANKING: bool = True
    ENABLE_SPARSE_RETRIEVAL: bool = True

    # LangSmith Observability (supports both LANGCHAIN_* and legacy LANGSMITH_*)
    LANGCHAIN_TRACING_V2: str = "false"
    LANGCHAIN_API_KEY: str = ""
    LANGCHAIN_PROJECT: str = "nexusai"
    LANGSMITH_API_KEY: str = ""
    LANGSMITH_TRACING: str = "false"
    LANGSMITH_PROJECT: str = "nexusai"

    # ── Search ───────────────────────────────────────────────────
    TAVILY_API_KEY: str = ""

    # ── Hugging Face ─────────────────────────────────────────────
    HF_API_KEY: str = ""

    # ── Vector DB ────────────────────────────────────────────────
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str = ""
    QDRANT_COLLECTION: str = "nexusai_documents"
    QDRANT_CLUSTER_ENDPOINT: str = ""
    QDRANT_CLUSTER_API_KEY: str = ""

    # ── Knowledge Graph ──────────────────────────────────────────
    NEO4J_URL: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "changeme"
    NEO4J_DATABASE: str = "neo4j"
    NEO4J_POOL_SIZE: int = 50
    NEO4J_CONNECTION_LIFETIME: int = 3600

    # Read-only account used exclusively by the natural-language graph query
    # path. When unset the NL->Cypher path is DISABLED (fail-closed) — it
    # must never fall back to the write account.
    NEO4J_READONLY_USER: str = ""
    NEO4J_READONLY_PASSWORD: str = ""

    # ── CSV → Graph ingestion ────────────────────────────────────
    CSV_MAX_UPLOAD_BYTES: int = 100 * 1024 * 1024
    CSV_MAX_ROWS: int = 1_000_000
    CSV_MAX_COLUMNS: int = 1_000
    CSV_MAX_CELL_BYTES: int = 64 * 1024
    CSV_MAX_FILES: int = 20
    CSV_PROFILE_SAMPLE_ROWS: int = 20_000
    CSV_INGEST_BATCH_SIZE: int = 1_000
    CSV_FK_OVERLAP_STRONG: float = 0.95
    CSV_FK_OVERLAP_MIN: float = 0.60
    CSV_KEY_SCORE_ACCEPT: int = 60
    UPLOAD_DIR: str = ".data/uploads"

    # ── Graph query safety ────────────────────────────────────────
    GRAPH_QUERY_MAX_ROWS: int = 200
    GRAPH_QUERY_TIMEOUT_MS: int = 5_000
    GRAPH_QUERY_MAX_RETRIES: int = 1

    # ── Database ─────────────────────────────────────────────────
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/nexusai"
    DATABASE_POOL_SIZE: int = 10
    DATABASE_MAX_OVERFLOW: int = 20

    # ── Celery / Background Workers ──────────────────────────────
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/1"
    USE_CELERY_WORKERS: str = "auto"  # "auto" | "true" | "false"

    # ── NEXUS_* aliases (explicit fields so pydantic reads them) ─
    NEXUS_GEMINI_API_KEY: str = ""
    NEXUS_GROQ_API_KEY: str = ""
    NEXUS_TAVILY_API_KEY: str = ""
    NEXUS_LANGSMITH_API_KEY: str = ""
    NEXUS_HF_API_KEY: str = ""
    NEXUS_CLUSTER_ENDPOINT: str = ""
    NEXUS_QDRANT_URL: str = ""
    NEXUS_QDRANT_API_KEY: str = ""
    NEXUS_QDRANT_ID: str = ""

    @model_validator(mode="after")
    def _apply_nexus_aliases(self) -> "Settings":
        import os

        mapping = {
            "GOOGLE_API_KEYS": "NEXUS_GEMINI_API_KEY",
            "GROQ_API_KEYS": "NEXUS_GROQ_API_KEY",
            "TAVILY_API_KEY": "NEXUS_TAVILY_API_KEY",
            "LANGSMITH_API_KEY": "NEXUS_LANGSMITH_API_KEY",
            "HF_API_KEY": "NEXUS_HF_API_KEY",
            "QDRANT_CLUSTER_ENDPOINT": "NEXUS_QDRANT_URL",
            "QDRANT_CLUSTER_API_KEY": "NEXUS_QDRANT_API_KEY",
        }
        for target, alias in mapping.items():
            current = getattr(self, target, None)
            if not current:
                alias_value = getattr(self, alias, None)
                if alias_value:
                    object.__setattr__(self, target, alias_value)

        if not self.QDRANT_CLUSTER_ENDPOINT and self.NEXUS_CLUSTER_ENDPOINT:
            object.__setattr__(self, "QDRANT_CLUSTER_ENDPOINT", self.NEXUS_CLUSTER_ENDPOINT)

        # Export HF_TOKEN into process env so sentence-transformers & HuggingFace Hub authenticate cleanly
        if self.HF_API_KEY:
            os.environ.setdefault("HF_TOKEN", self.HF_API_KEY)
            os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", self.HF_API_KEY)

        # Auto-activate LangSmith tracing when a valid LangSmith key is supplied unless explicitly disabled
        if not self.LANGCHAIN_API_KEY and self.LANGSMITH_API_KEY:
            object.__setattr__(self, "LANGCHAIN_API_KEY", self.LANGSMITH_API_KEY)
        elif not self.LANGSMITH_API_KEY and self.LANGCHAIN_API_KEY:
            object.__setattr__(self, "LANGSMITH_API_KEY", self.LANGCHAIN_API_KEY)

        if self.LANGCHAIN_TRACING_V2.lower() != "true" and self.LANGSMITH_TRACING.lower() == "true":
            object.__setattr__(self, "LANGCHAIN_TRACING_V2", "true")
        elif self.LANGSMITH_TRACING.lower() != "true" and self.LANGCHAIN_TRACING_V2.lower() == "true":
            object.__setattr__(self, "LANGSMITH_TRACING", "true")

        if self.LANGSMITH_PROJECT and self.LANGCHAIN_PROJECT == "nexusai":
            object.__setattr__(self, "LANGCHAIN_PROJECT", self.LANGSMITH_PROJECT)
        return self

    @field_validator("BACKEND_CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: str | list[str]) -> list[str]:
        """Accept both a JSON array string and a Python list."""
        if isinstance(v, str):
            import json

            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                pass
            # Fallback: comma-separated string
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (created once per process)."""
    return Settings()
