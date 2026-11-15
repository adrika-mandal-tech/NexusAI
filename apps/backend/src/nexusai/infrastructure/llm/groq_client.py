"""Groq LLM client implementation (async)."""

from __future__ import annotations

import json
import os
from typing import Any, AsyncGenerator

from groq import AsyncGroq

from nexusai.core.config import get_settings
from nexusai.core.logging import get_logger
from nexusai.domain.ports.llm_port import LLMPort

logger = get_logger(__name__)
settings = get_settings()


class GroqLLMClient(LLMPort):
    """Groq implementation of LLMPort (async)."""

    def __init__(self) -> None:
        self._api_keys = self._get_api_keys()
        self._key_index = 0
        self._model_name = settings.DEFAULT_GROQ_MODEL
        self._client = None

    def _get_api_keys(self) -> list[str]:
        keys = settings.GROQ_API_KEYS or os.environ.get("NEXUS_GROQ_API_KEY", "")
        if isinstance(keys, str):
            return [k.strip() for k in keys.split(",") if k.strip()]
        return []

    def _get_client(self) -> AsyncGroq:
        if not self._api_keys:
            raise RuntimeError("No GROQ_API_KEYS configured")
        key = self._api_keys[self._key_index % len(self._api_keys)]
        self._key_index += 1
        return AsyncGroq(api_key=key)

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.1,
        max_tokens: int | None = None,
    ) -> str:
        """Generate a complete response."""
        try:
            client = self._get_client()
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            logger.debug("Groq generate: model=%s, messages=%d", self._model_name, len(messages))
            logger.info("Groq generate: calling API")
            response = await client.chat.completions.create(
                model=self._model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            logger.info("Groq generate completed")
            return response.choices[0].message.content or "No answer generated."

        except Exception as exc:
            logger.error("Groq generation failed: %s", exc)
            raise

    async def stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.1,
        max_tokens: int | None = None,
    ) -> AsyncGenerator[str, None]:
        """Stream response token by token."""
        try:
            client = self._get_client()
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            logger.info("Groq stream: calling API")
            stream = await client.chat.completions.create(
                model=self._model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )

            async for chunk in stream:
                delta = getattr(chunk.choices[0].delta, "content", None) if chunk.choices else None
                if delta:
                    yield delta

        except Exception as exc:
            logger.error("Groq streaming failed: %s", exc)
            raise

    async def extract_entities(
        self,
        text: str,
        schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Extract entities and relationships from text."""
        if not schema:
            schema = {
                "entities": [
                    {"name": "str", "type": "str", "description": "Entity name and type"}
                ],
                "relationships": [
                    {"source": "str", "target": "str", "type": "str", "description": "Relationship between entities"}
                ],
            }

        prompt = f"""
Extract entities and relationships from the following text. Return as JSON.

Text:
{text}

Schema:
{json.dumps(schema, indent=2)}

Return JSON only with this structure:
{{
  "entities": [{{"id": "uuid", "name": "entity name", "type": "entity type"}}],
  "relationships": [{{"source_id": "uuid", "target_id": "uuid", "type": "relationship type"}}]
}}
"""

        try:
            response = await self.generate(prompt, temperature=0.0)
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                return json.loads(json_str)
        except Exception as exc:
            logger.error("Entity extraction failed: %s", exc)

        return {"entities": [], "relationships": []}
