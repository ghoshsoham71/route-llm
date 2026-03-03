# llm_router/providers/gemini.py
"""
Google Gemini provider adapter.

Wraps google-generativeai's async client. Supports BYOC.

Message format conversion
-------------------------
Converts the standard OpenAI-style messages list to Gemini's
ContentsType format (role + parts).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from google.generativeai.types import Content  # type: ignore[import]

from .base import BaseProvider
from ..constants import DEFAULT_GEMINI_MODEL


class GeminiProvider(BaseProvider):
    """Adapter wrapping google.generativeai.GenerativeModel."""

    def __init__(
        self,
        name: str = "gemini",
        model: str = DEFAULT_GEMINI_MODEL,
        api_key: str | None = None,
        rpm_limit: int = 60,
        tpm_limit: int = 100_000,
        weight: float = 1.0,
        enabled: bool = True,
        client: Any = None,  # BYOC — pass a configured GenerativeModel
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
            self._model = client
        else:
            try:
                from google.generativeai import GenerativeModel  # type: ignore[import]
            except ImportError as exc:
                raise ImportError(
                    "google-generativeai is required for GeminiProvider. "
                    "Install it with: pip install google-generativeai"
                ) from exc
            self._model = GenerativeModel(model)

    @staticmethod
    def _to_gemini_messages(messages: list[dict[str, Any]]) -> tuple[str | None, list[dict[str, Any]]]:
        """Convert OpenAI-style messages to Gemini format."""
        system_parts: list[str] = []
        history: list[dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                system_parts.append(content)
            elif role == "assistant":
                history.append({"role": "model", "parts": [content]})
            else:
                history.append({"role": "user", "parts": [content]})

        system_instruction = "\n".join(system_parts) if system_parts else None
        return system_instruction, history

    async def chat(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> tuple[str, int, int]:
        from google.generativeai.types import GenerationConfig  # type: ignore[import]

        _, history = self._to_gemini_messages(messages)
        # Separate the last user message as the prompt
        prompt_parts = history[-1]["parts"] if history else [""]
        chat_history = history[:-1]

        generation_config = GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
        )
        chat_history_content = [Content(role=h["role"], parts=h["parts"]) for h in chat_history]
        chat = self._model.start_chat(history=chat_history_content)
        response = await chat.send_message_async(
            prompt_parts,
            generation_config=generation_config,
        )
        content = response.text or ""
        # Gemini usage metadata
        input_tokens = response.usage_metadata.prompt_token_count if hasattr(response, "usage_metadata") else 0
        output_tokens = response.usage_metadata.candidates_token_count if hasattr(response, "usage_metadata") else 0
        return content, input_tokens, output_tokens

    async def stream(  # type: ignore
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        from google.generativeai.types import GenerationConfig  # type: ignore[import]

        _, history = self._to_gemini_messages(messages)
        prompt_parts = history[-1]["parts"] if history else [""]
        chat_history = history[:-1]

        generation_config = GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
        )
        chat_history_content = [Content(role=h["role"], parts=h["parts"]) for h in chat_history]
        chat = self._model.start_chat(history=chat_history_content)
        async for chunk in await chat.send_message_async(
            prompt_parts,
            generation_config=generation_config,
            stream=True,
        ):
            if chunk.text:
                yield chunk.text
