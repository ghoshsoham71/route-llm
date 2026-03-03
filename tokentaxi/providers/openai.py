# llm_router/providers/openai.py
"""
OpenAI provider adapter.

Wraps an AsyncOpenAI client. The router registers this adapter when the
developer either:
  a) Provides api_key in ProviderConfig (adapter creates its own client), or
  b) Calls router.register("openai", client=openai_client, ...) — BYOC mode
     (adapter uses the supplied client directly).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, cast

import httpx

from .base import BaseProvider
from ..constants import DEFAULT_OPENAI_MODEL


class OpenAIProvider(BaseProvider):
    """Adapter wrapping openai.AsyncOpenAI."""

    def __init__(
        self,
        name: str = "openai",
        model: str = DEFAULT_OPENAI_MODEL,
        api_key: str | None = None,
        rpm_limit: int = 500,
        tpm_limit: int = 200_000,
        weight: float = 1.0,
        enabled: bool = True,
        client: Any = None,  # pre-configured AsyncOpenAI — BYOC
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
                import openai  # type: ignore[import]
            except ImportError as exc:
                raise ImportError(
                    "openai package is required for OpenAIProvider. "
                    "Install it with: pip install openai"
                ) from exc
            self._client = openai.AsyncOpenAI(
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
            messages=cast(list, messages),
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
            messages=cast(list, messages),
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
            **kwargs,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
