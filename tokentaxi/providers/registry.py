# llm_router/providers/registry.py
"""
ProviderRegistry — thread-safe container for all registered providers.

The registry is the single source of truth for which providers are
available. The router queries it on every routing decision.

Thread-safety is achieved via asyncio.Lock so concurrent coroutines
don't race when registering or querying providers.
"""

from __future__ import annotations

import asyncio
from typing import Any

from .base import BaseProvider
from .openai import OpenAIProvider
from .anthropic import AnthropicProvider
from .gemini import GeminiProvider
from .groq import GroqProvider
from ..models import ProviderConfig


# Map provider name → adapter class
_ADAPTER_MAP: dict[str, type[BaseProvider]] = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "gemini": GeminiProvider,
    "groq": GroqProvider,
}


class ProviderRegistry:
    """Holds all registered provider adapters."""

    def __init__(self) -> None:
        self._providers: dict[str, BaseProvider] = {}
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    async def register_from_config(self, config: ProviderConfig) -> None:
        """Create and register a provider adapter from a ProviderConfig."""
        if not config.enabled:
            return

        adapter_cls = _ADAPTER_MAP.get(config.name)
        if adapter_cls is None:
            raise ValueError(
                f"Unknown provider '{config.name}'. "
                f"Supported built-in providers: {list(_ADAPTER_MAP)}. "
                "For custom providers, use register_adapter() directly."
            )

        adapter = adapter_cls(
            name=config.name,
            model=config.model,
            rpm_limit=config.rpm_limit,
            tpm_limit=config.tpm_limit,
            weight=config.weight,
            enabled=config.enabled,
        )
        await self.register_adapter(adapter)

    async def register_adapter(self, adapter: BaseProvider) -> None:
        """Register a pre-built provider adapter directly."""
        async with self._lock:
            self._providers[adapter.name] = adapter

    async def register_byoc(
        self,
        name: str,
        client: Any,
        model: str,
        rpm: int,
        tpm: int,
        weight: float = 1.0,
    ) -> None:
        """
        Register a BYOC (Bring Your Own Client) provider.

        The adapter class is inferred from *name*. If *name* is not a known
        built-in, raises ValueError — the developer should use
        register_adapter() with a custom BaseProvider subclass instead.
        """
        adapter_cls = _ADAPTER_MAP.get(name)
        if adapter_cls is None:
            raise ValueError(
                f"Unknown provider '{name}' for BYOC registration. "
                "Implement a BaseProvider subclass and use register_adapter()."
            )
        adapter = adapter_cls(
            name=name,
            model=model,
            rpm_limit=rpm,
            tpm_limit=tpm,
            weight=weight,
        )
        set_client = getattr(adapter, 'set_client', None)
        if set_client is not None:
            set_client(client)
        await self.register_adapter(adapter)

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    async def get_all(self) -> list[BaseProvider]:
        """Return all enabled providers."""
        async with self._lock:
            return [p for p in self._providers.values() if p.enabled]

    async def get(self, name: str) -> BaseProvider | None:
        """Return provider by name, or None if not found."""
        async with self._lock:
            return self._providers.get(name)

    async def names(self) -> list[str]:
        """Return names of all registered providers."""
        async with self._lock:
            return list(self._providers.keys())

    async def close_all(self) -> None:
        """Call close() on every provider (releases HTTP connections, etc.)."""
        async with self._lock:
            for provider in self._providers.values():
                await provider.close()
