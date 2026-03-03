# llm_router/providers/groq.py
"""
Groq provider adapter.

Wraps the groq AsyncGroq client. Groq's API is OpenAI-compatible, so
the adapter is essentially the same as OpenAIProvider but uses the Groq SDK.
Supports BYOC.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx

from .base import BaseProvider
from ..constants import DEFAULT_GROQ_MODEL


class GroqProvider(BaseProvider):
    """Adapter wrapping groq.AsyncGroq."""

    def __init__(
        self,
        name: str = "groq",
        model: str = DEFAULT_GROQ_MODEL,
        api_key: str | None = None,
        rpm_limit: int = 30,
        tpm_limit: int = 100_000,
        weight: float = 1.0,
        enabled: bool = True,
        client: Any = None,  # BYOC
    ) -> None:
        super().__init__(
            name=name,
            model=model,
            rpm_limit=rpm_limit,
            tpm_limit=tpm_limit,
            weight=weight,
            enabled=enabled,
        )

        if client is not None:
            self._client = client
        else:
            try:
                from groq import AsyncGroq  # type: ignore[import]
            except ImportError as exc:
                raise ImportError(
                    "groq package is required for GroqProvider. "
                    "Install it with: pip install groq"
                ) from exc
            self._client = AsyncGroq(
                api_key=api_key,
                http_client=httpx.AsyncClient(
                    limits=httpx.Limits(max_connections=100, max_keepalive_connections=20)
                ),
            )

    async def chat(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> tuple[str, int, int]:
        response = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )
        content = response.choices[0].message.content or ""
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        return content, input_tokens, output_tokens

    async def stream(  # type: ignore
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        stream = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
            **kwargs,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
