"""Tavily external web search service with privacy & prompt-injection guards.

Tavily is strictly an external retrieval channel (`source_kind="web"`), never a
replacement for the user's private Qdrant/Neo4j knowledge base.
"""

from __future__ import annotations

import asyncio
import os
import re
import time
from typing import Any

from nexusai.core.config import get_settings
from nexusai.core.logging import get_logger
from nexusai.core.observability import traced_step

logger = get_logger(__name__)

_INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"ignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions", re.IGNORECASE),
    re.compile(r"system\s*prompt\s*:", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(?:a|an)\s+", re.IGNORECASE),
    re.compile(r"<\s*/?\s*(?:system|instruction|tool_call)\s*>", re.IGNORECASE),
)


def sanitize_untrusted_text(text: str, max_chars: int = 1600) -> str:
    """Neutralize prompt-injection directives inside untrusted web or external snippets."""
    if not text:
        return ""
    cleaned = text.replace("\x00", "").strip()
    for pat in _INJECTION_PATTERNS:
        cleaned = pat.sub("[REDACTED_DIRECTIVE]", cleaned)
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars].rstrip() + "..."
    return cleaned


def sanitize_outbound_web_query(query: str, max_chars: int = 280) -> str:
    """Ensure outbound web search queries stay concise and do not dump raw document blocks."""
    collapsed = re.sub(r"\s+", " ", (query or "")).strip()
    # Strip pasted multi-line code blocks or long document excerpts
    collapsed = re.sub(r"```.*?```", "", collapsed).strip()
    if len(collapsed) > max_chars:
        collapsed = collapsed[:max_chars].rsplit(" ", 1)[0]
    return collapsed


class TavilySearchService:
    """Configurable Tavily web research client with graceful unconfigured status."""

    def __init__(self, api_key: str | None = None, client: Any | None = None) -> None:
        settings = get_settings()
        self._api_key = (
            api_key
            if api_key is not None
            else (settings.TAVILY_API_KEY or os.environ.get("NEXUS_TAVILY_API_KEY", ""))
        ).strip()
        self._injected_client = client

    @property
    def is_configured(self) -> bool:
        return bool(self._api_key or self._injected_client is not None)

    def health(self) -> dict[str, Any]:
        return {
            "status": "configured" if self.is_configured else "unconfigured",
            "provider": "tavily",
            "api_key_present": bool(self._api_key),
        }

    @traced_step(name="tavily_web_search", run_type="retriever")
    async def search(
        self,
        query: str,
        max_results: int = 5,
        search_depth: str = "basic",
        timeout_seconds: float = 12.0,
    ) -> dict[str, Any]:
        """Run external web search and return structured `[WEB-n]` citations."""
        start = time.perf_counter()
        safe_query = sanitize_outbound_web_query(query)
        if not safe_query:
            return {
                "status": "empty_query",
                "query": "",
                "results": [],
                "latency_ms": 0.0,
            }

        if not self.is_configured:
            return {
                "status": "unconfigured",
                "detail": "TAVILY_API_KEY is not configured; external web search was skipped.",
                "query": safe_query,
                "results": [],
                "latency_ms": round((time.perf_counter() - start) * 1000.0, 2),
            }

        try:
            raw_response = await asyncio.wait_for(
                self._execute_client_search(
                    safe_query,
                    max_results=max(1, min(10, max_results)),
                    search_depth=search_depth if search_depth in ("basic", "advanced") else "basic",
                ),
                timeout=timeout_seconds,
            )
            raw_items = raw_response.get("results", []) if isinstance(raw_response, dict) else []
            normalized: list[dict[str, Any]] = []
            for idx, item in enumerate(raw_items, start=1):
                if not isinstance(item, dict):
                    continue
                url = str(item.get("url") or "").strip()
                title = sanitize_untrusted_text(str(item.get("title") or url or f"Web Result {idx}"), max_chars=200)
                snippet = sanitize_untrusted_text(str(item.get("content") or item.get("snippet") or ""), max_chars=1400)
                if not snippet and not title:
                    continue
                score = float(item.get("score") or max(0.1, 1.0 - (idx - 1) * 0.12))
                normalized.append(
                    {
                        "citation_id": f"WEB-{idx}",
                        "source_kind": "web",
                        "retrieval_channel": "web_tavily",
                        "url": url,
                        "title": title,
                        "filename": title,
                        "text": snippet,
                        "text_snippet": snippet[:280] + ("..." if len(snippet) > 280 else ""),
                        "score": round(score, 4),
                        "published_date": item.get("published_date"),
                    }
                )

            return {
                "status": "ok",
                "query": safe_query,
                "results": normalized,
                "latency_ms": round((time.perf_counter() - start) * 1000.0, 2),
            }
        except asyncio.TimeoutError:
            logger.warning("Tavily web search timed out after %.1fs for query=%r", timeout_seconds, safe_query)
            return {
                "status": "timeout",
                "detail": f"Tavily web search timed out after {timeout_seconds}s.",
                "query": safe_query,
                "results": [],
                "latency_ms": round((time.perf_counter() - start) * 1000.0, 2),
            }
        except Exception as exc:
            logger.warning("Tavily web search failed: %s", exc)
            return {
                "status": "error",
                "detail": str(exc),
                "query": safe_query,
                "results": [],
                "latency_ms": round((time.perf_counter() - start) * 1000.0, 2),
            }

    async def _execute_client_search(
        self,
        query: str,
        max_results: int,
        search_depth: str,
    ) -> dict[str, Any]:
        if self._injected_client is not None:
            import inspect

            fn = getattr(self._injected_client, "search")
            if inspect.iscoroutinefunction(fn):
                return await fn(query=query, max_results=max_results, search_depth=search_depth)
            return await asyncio.to_thread(fn, query=query, max_results=max_results, search_depth=search_depth)

        try:
            from tavily import AsyncTavilyClient  # type: ignore[import-untyped]

            async_client = AsyncTavilyClient(api_key=self._api_key)
            return await async_client.search(
                query=query,
                max_results=max_results,
                search_depth=search_depth,
                include_answer=False,
            )
        except ImportError:
            from tavily import TavilyClient  # type: ignore[import-untyped]

            sync_client = TavilyClient(api_key=self._api_key)
            return await asyncio.to_thread(
                sync_client.search,
                query=query,
                max_results=max_results,
                search_depth=search_depth,
            )
