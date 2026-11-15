"""Gemini LLM client implementation (delegates to non-blocking GeminiAsyncClient without deprecated imports)."""

from __future__ import annotations

from nexusai.infrastructure.llm.provider import GeminiAsyncClient


class GeminiLLMClient(GeminiAsyncClient):
    """Backward-compatible GeminiLLMClient backed by async Gemini v1beta REST adapter."""
