# llm_router/providers/anthropic.py
"""
Anthropic provider adapter.

Wraps an AsyncAnthropic client. Supports BYOC (pass an existing client)
or creates its own client from api_key.

Notes on message format
-----------------------
Anthropic's API separates the system message from the messages list.
This adapter transparently handles the conversion so the router can use
the uniform OpenAI-style messages format throughout.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx

from .base import BaseProvider
from ..constants import DEFAULT_ANTHROPIC_MODEL


class AnthropicProvider(BaseProvider):
    """Adapter wrapping anthropic.AsyncAnthropic."""

    def __init__(
        self,
        name: str = "anthropic",
        model: str = DEFAULT_ANTHROPIC_MODEL,
        api_key: str | None = None,
        rpm_limit: int = 50,
        tpm_limit: int = 200_000,
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
                import anthropic  # type: ignore[import]
            except ImportError as exc:
                raise ImportError(
                    "anthropic package is required for AnthropicProvider. "
                    "Install it with: pip install anthropic"
                ) from exc
            self._client = anthropic.AsyncAnthropic(
                api_key=api_key,
                http_client=httpx.AsyncClient(
                    limits=httpx.Limits(max_connections=100, max_keepalive_connections=20)
                ),
            )

    def _split_messages(
        self, messages: list[dict[str, Any]]
    ) -> tuple[str | None, list[dict[str, Any]]]:
        """Extract optional system message; return (system, user_messages)."""
        system: str | None = None
        filtered: list[dict[str, Any]] = []
        for msg in messages:
            if msg.get("role") == "system":
                system = msg.get("content", "")
            else:
                filtered.append(msg)
        return system, filtered

    async def chat(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> tuple[str, int, int]:
        system, filtered = self._split_messages(messages)
        call_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": filtered,
            "max_tokens": max_tokens,
            **kwargs,
        }
        if system:
            call_kwargs["system"] = system

        response = await self._client.messages.create(**call_kwargs)
        content = response.content[0].text if response.content else ""
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        return content, input_tokens, output_tokens

    async def stream(  # type: ignore
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        system, filtered = self._split_messages(messages)
        call_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": filtered,
            "max_tokens": max_tokens,
            **kwargs,
        }
        if system:
            call_kwargs["system"] = system

        async with self._client.messages.stream(**call_kwargs) as stream:
            async for text in stream.text_stream:
                yield text
