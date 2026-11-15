"""LLM port — abstract interface for language model operations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator


class LLMPort(ABC):
    """Abstract interface for LLM operations."""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.1,
        max_tokens: int | None = None,
    ) -> str:
        """Generate a complete response."""

    @abstractmethod
    async def stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.1,
        max_tokens: int | None = None,
    ) -> AsyncGenerator[str, None]:
        """Stream response token by token."""

    @abstractmethod
    async def extract_entities(
        self,
        text: str,
        schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Extract entities and relationships from text.

        Returns: {entities: [...], relationships: [...]}
        """
