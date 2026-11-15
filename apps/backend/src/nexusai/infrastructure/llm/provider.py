"""Configurable LLM Provider Abstraction: Async Gemini v1beta REST, OpenLM / OpenAI-compatible, Groq, & Grounded Extractive Fallback."""

from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass
from typing import Any, AsyncGenerator

import httpx

from nexusai.core.config import get_settings
from nexusai.core.logging import get_logger
from nexusai.core.observability import traced_step
from nexusai.domain.ports.llm_port import LLMPort

logger = get_logger(__name__)

_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


class LLMProviderUnconfiguredError(RuntimeError):
    """Raised when an external LLM provider has no configured credentials or endpoint."""


@dataclass
class LLMGenerationUsage:
    """Token usage metadata extracted from provider responses."""

    provider: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class GeminiAsyncClient(LLMPort):
    """Non-blocking async Gemini Generative Language API v1beta client with key rotation & exponential backoff."""

    BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(
        self,
        api_keys: list[str] | None = None,
        model_name: str | None = None,
        timeout_seconds: float = 45.0,
        max_retries: int = 3,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        settings = get_settings()
        if api_keys is not None:
            self._api_keys = [k.strip() for k in api_keys if k.strip()]
        else:
            raw = settings.GOOGLE_API_KEYS or os.environ.get("NEXUS_GEMINI_API_KEY", "")
            self._api_keys = [k.strip() for k in raw.split(",") if k.strip()]
        self._key_index = 0
        self._model_name = (model_name or settings.DEFAULT_GEMINI_MODEL or "gemini-3.5-flash-lite").strip()
        self._fallback_models = [
            m.strip()
            for m in (settings.GEMINI_MODELS or "gemini-3.5-flash-lite,gemini-3.1-flash-lite,gemini-3.5-flash").split(",")
            if m.strip()
        ]
        self._timeout = timeout_seconds
        self._max_retries = max(1, max_retries)
        self._http_client = http_client
        self.last_usage = LLMGenerationUsage(provider="gemini", model=self._model_name)

    @property
    def is_configured(self) -> bool:
        return bool(self._api_keys)

    @property
    def model_name(self) -> str:
        return self._model_name

    def _next_key(self) -> str:
        if not self._api_keys:
            raise LLMProviderUnconfiguredError(
                "Gemini API key is not configured (GOOGLE_API_KEYS is empty)."
            )
        key = self._api_keys[self._key_index % len(self._api_keys)]
        self._key_index += 1
        return key

    def _build_payload(
        self,
        prompt: str,
        system_prompt: str | None,
        temperature: float,
        max_tokens: int | None,
        active_model: str = "",
    ) -> dict[str, Any]:
        gen_cfg: dict[str, Any] = {"temperature": temperature}
        if max_tokens is not None:
            gen_cfg["maxOutputTokens"] = max_tokens
        # Disable internal hidden thought budget on Gemini 2.5/3.x Flash models for sub-second grounded RAG latency
        if any(tag in active_model for tag in ("2.5-flash", "3-flash", "3.1-flash", "3.5-flash")):
            gen_cfg["thinkingConfig"] = {"thinkingBudget": 0}
        payload: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": gen_cfg,
        }
        if system_prompt:
            payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}
        return payload

    def _candidate_models(self, requested_model: str | None) -> list[str]:
        primary = (requested_model or self._model_name).strip()
        ordered = [primary]
        for fm in self._fallback_models + ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-3.5-flash"]:
            if fm and fm not in ordered:
                ordered.append(fm)
        return ordered

    @traced_step(name="gemini_generate", run_type="llm")
    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.1,
        max_tokens: int | None = None,
        model: str | None = None,
    ) -> str:
        models_to_try = self._candidate_models(model) if self._http_client is None else [(model or self._model_name).strip()]
        last_exc: Exception | None = None

        for active_model in models_to_try:
            payload = self._build_payload(prompt, system_prompt, temperature, max_tokens, active_model=active_model)
            for attempt in range(self._max_retries):
                api_key = self._next_key()
                url = f"{self.BASE_URL}/{active_model}:generateContent"
                try:
                    if self._http_client is not None:
                        resp = await self._http_client.post(
                            url,
                            params={"key": api_key},
                            json=payload,
                            timeout=self._timeout,
                        )
                    else:
                        async with httpx.AsyncClient(timeout=self._timeout) as client:
                            resp = await client.post(url, params={"key": api_key}, json=payload)

                    if resp.status_code == 404 and len(models_to_try) > 1:
                        logger.warning("Gemini model '%s' returned 404; rotating to next active Gemini model", active_model)
                        last_exc = RuntimeError(f"Gemini model {active_model} returned 404")
                        break

                    if resp.status_code in _RETRYABLE_STATUS_CODES and attempt + 1 < self._max_retries:
                        backoff = min(4.0, 0.35 * (2**attempt))
                        logger.warning("Gemini HTTP %d on attempt %d; rotating key & retrying in %.2fs", resp.status_code, attempt + 1, backoff)
                        await asyncio.sleep(backoff)
                        continue

                    resp.raise_for_status()
                    data = resp.json()
                    usage = data.get("usageMetadata") or {}
                    self.last_usage = LLMGenerationUsage(
                        provider="gemini",
                        model=active_model,
                        prompt_tokens=int(usage.get("promptTokenCount") or 0),
                        completion_tokens=int(usage.get("candidatesTokenCount") or 0),
                        total_tokens=int(usage.get("totalTokenCount") or 0),
                    )
                    text = self._extract_candidate_text(data)
                    if text:
                        return text
                except LLMProviderUnconfiguredError:
                    raise
                except Exception as exc:
                    last_exc = exc
                    if attempt + 1 < self._max_retries:
                        await asyncio.sleep(min(2.5, 0.3 * (2**attempt)))
                        continue
                    if self._http_client is not None:
                        raise

        if last_exc is not None:
            logger.error("Gemini generation failed across candidate models: %s", last_exc)
            raise last_exc
        return "The model returned an empty response."

    @traced_step(name="gemini_stream", run_type="llm")
    async def stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.1,
        max_tokens: int | None = None,
        model: str | None = None,
    ) -> AsyncGenerator[str, None]:
        models_to_try = self._candidate_models(model) if self._http_client is None else [(model or self._model_name).strip()]
        last_exc: Exception | None = None

        for active_model in models_to_try:
            api_key = self._next_key()
            payload = self._build_payload(prompt, system_prompt, temperature, max_tokens, active_model=active_model)
            url = f"{self.BASE_URL}/{active_model}:streamGenerateContent"

            if self._http_client is not None:
                async with self._http_client.stream(
                    "POST",
                    url,
                    params={"alt": "sse", "key": api_key},
                    json=payload,
                    timeout=self._timeout,
                ) as resp:
                    resp.raise_for_status()
                    async for delta in self._iter_sse_lines(resp, active_model):
                        yield delta
                return

            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    async with client.stream(
                        "POST",
                        url,
                        params={"alt": "sse", "key": api_key},
                        json=payload,
                    ) as resp:
                        if resp.status_code == 404 and len(models_to_try) > 1:
                            logger.warning("Gemini stream model '%s' returned 404; rotating to next model", active_model)
                            last_exc = RuntimeError(f"Gemini stream model {active_model} returned 404")
                            continue
                        resp.raise_for_status()
                        async for delta in self._iter_sse_lines(resp, active_model):
                            yield delta
                        return
            except Exception as exc:
                last_exc = exc
                logger.warning("Gemini stream on '%s' failed (%s); trying next fallback model", active_model, exc)
                continue

        if last_exc is not None:
            raise last_exc

    async def _iter_sse_lines(self, resp: httpx.Response, active_model: str) -> AsyncGenerator[str, None]:
        async for line in resp.aiter_lines():
            trimmed = line.strip()
            if not trimmed.startswith("data:"):
                continue
            raw_json = trimmed[5:].strip()
            if not raw_json or raw_json == "[DONE]":
                continue
            try:
                data = json.loads(raw_json)
            except json.JSONDecodeError:
                continue
            usage = data.get("usageMetadata")
            if isinstance(usage, dict):
                self.last_usage = LLMGenerationUsage(
                    provider="gemini",
                    model=active_model,
                    prompt_tokens=int(usage.get("promptTokenCount") or 0),
                    completion_tokens=int(usage.get("candidatesTokenCount") or 0),
                    total_tokens=int(usage.get("totalTokenCount") or 0),
                )
            piece = self._extract_candidate_text(data)
            if piece:
                yield piece

    @staticmethod
    def _extract_candidate_text(data: dict[str, Any]) -> str:
        candidates = data.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            return ""
        content = (candidates[0] or {}).get("content") or {}
        parts = content.get("parts") or []
        texts = [
            str(p.get("text"))
            for p in parts
            if isinstance(p, dict) and p.get("text") and not p.get("thought")
        ]
        return "".join(texts)

    async def extract_entities(
        self,
        text: str,
        schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self.is_configured:
            return {"entities": [], "relationships": []}
        schema_json = json.dumps(
            schema
            or {
                "entities": [{"name": "str", "type": "str"}],
                "relationships": [{"source": "str", "target": "str", "type": "str"}],
            }
        )
        prompt = (
            "Extract entities and relationships from the text below as valid JSON with keys "
            "`entities` ([{\"name\": \"...\", \"type\": \"...\"}]) and `relationships` "
            "([{\"source\": \"...\", \"target\": \"...\", \"type\": \"...\"}]).\n\n"
            f"Schema: {schema_json}\n\nText:\n{text[:4000]}"
        )
        try:
            raw = await self.generate(prompt, temperature=0.0)
            start, end = raw.find("{"), raw.rfind("}") + 1
            if start >= 0 and end > start:
                parsed = json.loads(raw[start:end])
                if isinstance(parsed, dict):
                    return parsed
        except Exception as exc:
            logger.warning("Gemini entity extraction skipped/failed: %s", exc)
        return {"entities": [], "relationships": []}


class OpenLMClient(LLMPort):
    """Configurable adapter for OpenLM / BentoML OpenLLM, Ollama, vLLM, LM Studio, & Groq OpenAI-compatible endpoints.

    Supports both:
    - `openai_chat`: `POST {base_url}/chat/completions` (OpenLLM >= 0.6, vLLM, Ollama `/v1`, Groq)
    - `openllm_legacy`: `POST {base_url}/generate` (BentoML OpenLLM 0.4/0.5 legacy protocol)
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model_name: str | None = None,
        api_style: str | None = None,
        provider_label: str = "openlm",
        timeout_seconds: float = 45.0,
        max_retries: int = 2,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        settings = get_settings()
        self._provider_label = provider_label
        self._base_url = (base_url if base_url is not None else settings.OPENLM_BASE_URL).strip().rstrip("/")
        self._api_key = (api_key if api_key is not None else settings.OPENLM_API_KEY).strip()
        self._model_name = (model_name if model_name is not None else settings.OPENLM_MODEL).strip() or "default"
        self._api_style = (api_style or settings.OPENLM_API_STYLE or "openai_chat").strip().lower()
        self._timeout = timeout_seconds
        self._max_retries = max(1, max_retries)
        self._http_client = http_client
        self.last_usage = LLMGenerationUsage(provider=provider_label, model=self._model_name)

    @property
    def is_configured(self) -> bool:
        if self._provider_label == "groq":
            return bool(self._base_url and self._api_key)
        return bool(self._base_url)

    @property
    def model_name(self) -> str:
        return self._model_name

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    @traced_step(name="openlm_generate", run_type="llm")
    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.1,
        max_tokens: int | None = None,
        model: str | None = None,
    ) -> str:
        if not self.is_configured:
            raise LLMProviderUnconfiguredError(
                f"{self._provider_label.upper()} endpoint/credentials are not configured."
            )
        active_model = (model or self._model_name).strip()

        if self._api_style == "openllm_legacy":
            full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
            url = f"{self._base_url}/generate"
            payload: dict[str, Any] = {
                "prompt": full_prompt,
                "llm_config": {"temperature": temperature, "max_new_tokens": max_tokens or 1024},
            }
        else:
            url = f"{self._base_url}/chat/completions"
            messages: list[dict[str, str]] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            payload = {
                "model": active_model,
                "messages": messages,
                "temperature": temperature,
            }
            if max_tokens is not None:
                payload["max_tokens"] = max_tokens

        for attempt in range(self._max_retries):
            try:
                if self._http_client is not None:
                    resp = await self._http_client.post(url, headers=self._headers(), json=payload, timeout=self._timeout)
                else:
                    async with httpx.AsyncClient(timeout=self._timeout) as client:
                        resp = await client.post(url, headers=self._headers(), json=payload)

                if resp.status_code in _RETRYABLE_STATUS_CODES and attempt + 1 < self._max_retries:
                    await asyncio.sleep(0.5 * (2**attempt))
                    continue

                resp.raise_for_status()
                data = resp.json()
                if self._api_style == "openllm_legacy":
                    outputs = data.get("outputs") or []
                    if outputs and isinstance(outputs[0], dict):
                        return str(outputs[0].get("text") or "").strip()
                    return str(data.get("text") or "").strip()

                usage = data.get("usage") or {}
                self.last_usage = LLMGenerationUsage(
                    provider=self._provider_label,
                    model=active_model,
                    prompt_tokens=int(usage.get("prompt_tokens") or 0),
                    completion_tokens=int(usage.get("completion_tokens") or 0),
                    total_tokens=int(usage.get("total_tokens") or 0),
                )
                choices = data.get("choices") or []
                if choices:
                    msg = (choices[0] or {}).get("message") or {}
                    return str(msg.get("content") or "").strip()
                return ""
            except Exception as exc:
                if attempt + 1 < self._max_retries:
                    await asyncio.sleep(0.4 * (2**attempt))
                    continue
                logger.error("%s generation failed: %s", self._provider_label, exc)
                raise

        return ""

    @traced_step(name="openlm_stream", run_type="llm")
    async def stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.1,
        max_tokens: int | None = None,
        model: str | None = None,
    ) -> AsyncGenerator[str, None]:
        if self._api_style == "openllm_legacy":
            text = await self.generate(prompt, system_prompt, temperature, max_tokens, model=model)
            if text:
                yield text
            return

        if not self.is_configured:
            raise LLMProviderUnconfiguredError(
                f"{self._provider_label.upper()} endpoint/credentials are not configured."
            )
        active_model = (model or self._model_name).strip()
        url = f"{self._base_url}/chat/completions"
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        payload: dict[str, Any] = {
            "model": active_model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        if self._http_client is not None:
            async with self._http_client.stream("POST", url, headers=self._headers(), json=payload, timeout=self._timeout) as resp:
                resp.raise_for_status()
                async for delta in self._parse_openai_sse(resp):
                    yield delta
            return

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            async with client.stream("POST", url, headers=self._headers(), json=payload) as resp:
                resp.raise_for_status()
                async for delta in self._parse_openai_sse(resp):
                    yield delta

    @staticmethod
    async def _parse_openai_sse(resp: httpx.Response) -> AsyncGenerator[str, None]:
        async for line in resp.aiter_lines():
            trimmed = line.strip()
            if not trimmed.startswith("data:"):
                continue
            raw = trimmed[5:].strip()
            if not raw or raw == "[DONE]":
                break
            try:
                data = json.loads(raw)
                choices = data.get("choices") or []
                if choices:
                    delta = ((choices[0] or {}).get("delta") or {}).get("content")
                    if delta:
                        yield str(delta)
            except json.JSONDecodeError:
                continue

    async def extract_entities(
        self,
        text: str,
        schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self.is_configured:
            return {"entities": [], "relationships": []}
        try:
            raw = await self.generate(
                f"Extract JSON with `entities` and `relationships` from:\n{text[:3000]}",
                temperature=0.0,
            )
            start, end = raw.find("{"), raw.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(raw[start:end])
        except Exception:
            pass
        return {"entities": [], "relationships": []}


def synthesize_extractive_grounded_answer(
    query: str,
    citations: list[dict[str, Any]],
    graph_entities: list[dict[str, Any]] | None = None,
    provider_notice: str | None = None,
) -> str:
    """Deterministic source-grounded extractive synthesis used ONLY when no external LLM key is configured.

    Never fabricates facts or citations: extracts the highest-scoring sentences directly
    from retrieved `[KB-n]` / `[WEB-n]` evidence and explicitly discloses extractive mode.
    """
    if not citations and not graph_entities:
        return (
            "No relevant supporting evidence was found in the indexed knowledge base for this question. "
            "Please upload relevant documents or broaden your search query."
        )

    q_words = {
        w.lower()
        for w in re.findall(r"[A-Za-z0-9_+#.-]{3,}", query or "")
    }
    bullet_lines: list[str] = []

    for cite in citations[:5]:
        cid = cite.get("citation_id") or "KB-1"
        fname = cite.get("filename") or cite.get("title") or "Source"
        page = cite.get("page_number")
        loc = f" ({fname}, p. {page})" if page is not None else f" ({fname})"
        snippet = str(cite.get("text_snippet") or cite.get("text") or "").strip()
        if not snippet:
            continue
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", snippet) if len(s.strip()) > 15]
        if sentences and q_words:
            best = max(
                sentences,
                key=lambda s: sum(1 for w in q_words if w in s.lower()),
            )
        else:
            best = (sentences[0] if sentences else snippet[:240]).strip()
        bullet_lines.append(f"- {best}{loc} [{cid}]")

    if graph_entities:
        rel_items = []
        for ge in (graph_entities or [])[:5]:
            en = ge.get("entity_name")
            rel = ge.get("relationship_type")
            tgt = ge.get("related_entity_name")
            if en and rel and tgt:
                rel_items.append(f"`{en}` -[{rel}]-> `{tgt}`")
        if rel_items:
            bullet_lines.append(f"- **Knowledge Graph Relationships**: {', '.join(rel_items)}")

    header = "**Retrieved Source-Grounded Evidence**"
    if provider_notice:
        header += f" *(Note: {provider_notice})*"
    return header + "\n\n" + "\n".join(bullet_lines)


class LLMProviderRegistry:
    """Unified registry resolving Gemini, OpenLM, and Groq providers with explicit health introspection."""

    def __init__(
        self,
        gemini: GeminiAsyncClient | None = None,
        openlm: OpenLMClient | None = None,
        groq: OpenLMClient | None = None,
    ) -> None:
        settings = get_settings()
        self.gemini = gemini or GeminiAsyncClient()
        self.openlm = openlm or OpenLMClient(provider_label="openlm")
        groq_keys = [
            k.strip()
            for k in (settings.GROQ_API_KEYS or os.environ.get("NEXUS_GROQ_API_KEY", "")).split(",")
            if k.strip()
        ]
        self.groq = groq or OpenLMClient(
            base_url="https://api.groq.com/openai/v1",
            api_key=groq_keys[0] if groq_keys else "",
            model_name=settings.DEFAULT_GROQ_MODEL,
            api_style="openai_chat",
            provider_label="groq",
        )

    def resolve_candidates(self, requested_provider: str | None = None) -> list[tuple[str, LLMPort]]:
        """Return ordered list of all configured `(provider_name, client)` candidates for resilient failover."""
        settings = get_settings()
        pref = (requested_provider or settings.LLM_PROVIDER or "gemini").strip().lower()

        if pref == "openlm":
            order: list[tuple[str, Any]] = [("openlm", self.openlm), ("gemini", self.gemini), ("groq", self.groq)]
        elif pref == "groq":
            order = [("groq", self.groq), ("gemini", self.gemini), ("openlm", self.openlm)]
        else:
            order = [("gemini", self.gemini), ("groq", self.groq), ("openlm", self.openlm)]

        return [(name, client) for name, client in order if getattr(client, "is_configured", False)]

    def resolve(self, requested_provider: str | None = None) -> tuple[str, LLMPort | None]:
        """Resolve requested or default configured provider; returns `(provider_name, client_or_None)`."""
        settings = get_settings()
        pref = (requested_provider or settings.LLM_PROVIDER or "gemini").strip().lower()
        candidates = self.resolve_candidates(requested_provider)
        if candidates:
            return candidates[0]
        return pref, None

    def status(self) -> dict[str, Any]:
        settings = get_settings()
        active_name, active_client = self.resolve()
        return {
            "default_provider": settings.LLM_PROVIDER,
            "active_provider": active_name if active_client is not None else "extractive_fallback",
            "any_llm_configured": active_client is not None,
            "providers": {
                "gemini": {
                    "configured": self.gemini.is_configured,
                    "default_model": self.gemini.model_name,
                    "models": [m.strip() for m in settings.GEMINI_MODELS.split(",") if m.strip()],
                },
                "openlm": {
                    "configured": self.openlm.is_configured,
                    "base_url": settings.OPENLM_BASE_URL or None,
                    "default_model": self.openlm.model_name,
                    "api_style": settings.OPENLM_API_STYLE,
                },
                "groq": {
                    "configured": self.groq.is_configured,
                    "default_model": self.groq.model_name,
                    "models": [m.strip() for m in settings.GROQ_MODELS.split(",") if m.strip()],
                },
            },
        }

