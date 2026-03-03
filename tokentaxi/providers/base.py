# llm_router/providers/base.py
"""
BaseProvider — abstract contract every provider adapter must implement.

An adapter wraps a pre-configured provider SDK client and exposes a
uniform interface to the router. The router never calls provider SDKs
directly; it always goes through an adapter.

This design means:
  - Provider-specific error handling is contained inside each adapter.
  - The router doesn't need to know about 429 vs ConnectionError vs
    provider-specific status codes.
  - Adding a new provider requires only implementing this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any


class BaseProvider(ABC):
    """
    Abstract base class for all LLM provider adapters.

    Attributes
    ----------
    name:
        Unique identifier, e.g. "openai", "anthropic".
    model:
        Model string, e.g. "gpt-4o", "claude-sonnet-4-5".
    rpm_limit:
        Max requests per minute for this provider key.
    tpm_limit:
        Max tokens per minute for this provider key.
    weight:
        Static preference weight (0.0–1.0).
    enabled:
        Whether this provider is currently active.
    """

    def __init__(
        self,
        name: str,
        model: str,
        rpm_limit: int,
        tpm_limit: int,
        weight: float = 1.0,
        enabled: bool = True,
    ) -> None:
        self.name = name
        self.model = model
        self.rpm_limit = rpm_limit
        self.tpm_limit = tpm_limit
        self.weight = weight
        self.enabled = enabled

    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float,
        **kwargs: Any,
    ) -> tuple[str, int, int]:
        """
        Send a non-streaming chat request.

        Parameters
        ----------
        messages:
            List of message dicts in the standard OpenAI format.
        max_tokens:
            Maximum completion tokens.
        temperature:
            Sampling temperature.
        **kwargs:
            Passed through to the underlying SDK call.

        Returns
        -------
        (content, input_tokens, output_tokens)
            content: completion text.
            input_tokens: prompt tokens consumed.
            output_tokens: completion tokens produced.

        Raises
        ------
        Any exception from the underlying SDK. The router handles all
        exceptions from this method uniformly (records failure, tries next).
        """

    async def stream(  # type: ignore
        self,
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Send a streaming chat request.

        Yields individual text chunks as they arrive.

        The same exception semantics as chat() apply.
        """
        raise NotImplementedError("Subclasses must implement stream()")

    async def close(self) -> None:
        """Release any resources held by this adapter (HTTP clients, etc.)."""

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"{self.__class__.__name__}("
            f"name={self.name!r}, model={self.model!r}, "
            f"rpm_limit={self.rpm_limit}, tpm_limit={self.tpm_limit}, "
            f"weight={self.weight}, enabled={self.enabled})"
        )
