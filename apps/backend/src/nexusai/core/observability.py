"""LangSmith tracing, request correlation IDs, and RAG observability telemetry."""

from __future__ import annotations

import contextvars
import functools
import os
import threading
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any, Callable

from nexusai.core.config import get_settings
from nexusai.core.logging import get_logger

logger = get_logger(__name__)

_request_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar("nexusai_request_id", default="")


def get_request_id() -> str:
    """Return active request correlation ID, generating one if unset."""
    rid = _request_id_ctx.get()
    if not rid:
        rid = str(uuid.uuid4())
        _request_id_ctx.set(rid)
    return rid


def set_request_id(request_id: str | None = None) -> str:
    """Bind a correlation ID to the current async execution context."""
    rid = (request_id or "").strip() or str(uuid.uuid4())
    _request_id_ctx.set(rid)
    return rid


def configure_langsmith_env() -> dict[str, Any]:
    """Synchronize LangSmith environment variables safely without exposing secrets."""
    settings = get_settings()
    api_key = (
        settings.LANGCHAIN_API_KEY
        or settings.LANGSMITH_API_KEY
        or os.environ.get("NEXUS_LANGSMITH_API_KEY", "")
    ).strip()
    project = (settings.LANGCHAIN_PROJECT or settings.LANGSMITH_PROJECT or "nexusai").strip()
    tracing_requested = bool(settings.LANGCHAIN_TRACING_V2 or settings.LANGSMITH_TRACING)

    os.environ["LANGCHAIN_PROJECT"] = project
    if tracing_requested and api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = api_key
        active = True
    else:
        # Prevent noisy background upload errors when LANGSMITH_API_KEY is empty
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        active = False

    return {
        "tracing_requested": tracing_requested,
        "api_key_configured": bool(api_key),
        "tracing_active": active,
        "project": project,
    }


def traced_step(name: str, run_type: str = "chain") -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Wrap sync or async functions in LangSmith `@traceable` when active."""
    try:
        from langsmith import traceable  # type: ignore[import-untyped]

        has_langsmith = True
    except Exception:
        traceable = None  # type: ignore[assignment]
        has_langsmith = False

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        if has_langsmith and traceable is not None:
            return traceable(name=name, run_type=run_type)(func)

        @functools.wraps(func)
        def passthrough(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)

        return passthrough

    return decorator


@dataclass
class TraceTelemetryRecord:
    """Structured telemetry record for a single RAG ingestion, retrieval, or chat execution."""

    request_id: str
    operation: str
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    status: str = "ok"
    query: str = ""
    retrieval_mode: str = "hybrid"
    fusion_strategy: str = "rrf"
    reranked: bool = False
    dense_count: int = 0
    sparse_count: int = 0
    graph_count: int = 0
    web_count: int = 0
    final_count: int = 0
    graph_status: str = "unknown"
    llm_provider: str = ""
    llm_model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: dict[str, float] = field(default_factory=dict)
    total_ms: float = 0.0
    error: str | None = None


class ObservabilityTracker:
    """Thread-safe ring buffer recording per-request RAG metrics and latency breakdowns."""

    def __init__(self, maxlen: int = 500) -> None:
        self._records: deque[TraceTelemetryRecord] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def record(self, entry: TraceTelemetryRecord) -> None:
        with self._lock:
            self._records.appendleft(entry)
        logger.info(
            "telemetry request_id=%s op=%s status=%s mode=%s fusion=%s dense=%d sparse=%d graph=%d(%s) final=%d total_ms=%.1f",
            entry.request_id,
            entry.operation,
            entry.status,
            entry.retrieval_mode,
            entry.fusion_strategy,
            entry.dense_count,
            entry.sparse_count,
            entry.graph_count,
            entry.graph_status,
            entry.final_count,
            entry.total_ms,
        )

    def recent(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            items = list(self._records)[: max(1, min(500, limit))]
        return [asdict(item) for item in items]

    def summary(self) -> dict[str, Any]:
        with self._lock:
            items = list(self._records)
        total = len(items)
        if total == 0:
            return {
                "total_requests": 0,
                "error_rate": 0.0,
                "avg_total_ms": 0.0,
                "p95_total_ms": 0.0,
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0,
                "langsmith": configure_langsmith_env(),
            }
        errors = sum(1 for r in items if r.status != "ok")
        latencies = sorted(r.total_ms for r in items)
        p95_idx = min(len(latencies) - 1, int(len(latencies) * 0.95))
        return {
            "total_requests": total,
            "error_count": errors,
            "error_rate": round(errors / total, 4),
            "avg_total_ms": round(sum(latencies) / total, 2),
            "p95_total_ms": round(latencies[p95_idx], 2),
            "total_prompt_tokens": sum(r.prompt_tokens for r in items),
            "total_completion_tokens": sum(r.completion_tokens for r in items),
            "langsmith": configure_langsmith_env(),
        }


_GLOBAL_TRACKER = ObservabilityTracker()


def get_observability_tracker() -> ObservabilityTracker:
    """Return process-wide telemetry tracker."""
    return _GLOBAL_TRACKER


class StepTimer:
    """Lightweight monotonic timer helper recording millisecond durations into a dict."""

    def __init__(self, target: dict[str, float], key: str) -> None:
        self._target = target
        self._key = key
        self._start = 0.0

    def __enter__(self) -> "StepTimer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self._target[self._key] = round((time.perf_counter() - self._start) * 1000.0, 2)

