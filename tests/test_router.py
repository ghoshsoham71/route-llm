# tests/test_router.py
"""
Integration tests for LLMRouter.

Uses unittest.mock to simulate provider adapters so no real API calls
are made. Tests cover:
  - Successful routing.
  - Automatic fallback on provider failure.
  - Circuit breaker integration.
  - Session affinity (sticky routing).
  - Provider pinning (force_provider).
  - Priority routing.
  - AllProvidersFailed when all providers fail.
  - on_route callback.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_router import (
    AllProvidersFailed,
    LLMRouter,
    NoProvidersConfigured,
    RouterRequest,
    RouterResponse,
)
from llm_router.config import CircuitBreakerConfig, RouterConfig, RoutingWeights
from llm_router.models import ProviderConfig
from llm_router.providers.base import BaseProvider


class MockProvider(BaseProvider):
    """In-memory mock provider for testing."""

    def __init__(
        self,
        name: str,
        should_fail: bool = False,
        fail_times: int = 0,
        rpm_limit: int = 100,
        tpm_limit: int = 50_000,
        weight: float = 1.0,
    ) -> None:
        super().__init__(
            name=name,
            model=f"mock-model-{name}",
            rpm_limit=rpm_limit,
            tpm_limit=tpm_limit,
            weight=weight,
        )
        self.should_fail = should_fail
        self.fail_times = fail_times
        self._call_count = 0

    async def chat(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> tuple[str, int, int]:
        self._call_count += 1
        if self.should_fail or self._call_count <= self.fail_times:
            raise RuntimeError(f"Provider {self.name} failed (call #{self._call_count})")
        return f"Response from {self.name}", 10, 20

    async def stream(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs: Any,
    ):
        async def _stream_impl():
            if self.should_fail:
                raise RuntimeError(f"Provider {self.name} stream failed")
            yield f"Chunk from {self.name}"
        return _stream_impl()


def make_router(providers: list[MockProvider], **config_kwargs) -> LLMRouter:
    """Create a router with mock providers pre-registered."""
    config = RouterConfig(
        providers=[],
        circuit_breaker=CircuitBreakerConfig(failure_threshold=3, cooldown_seconds=5),
        **config_kwargs,
    )
    router = LLMRouter(config)
    # Directly inject mock providers into the registry
    for provider in providers:
        router._registry._providers[provider.name] = provider
    router._initialized = True
    return router


def make_request(
    messages: list[dict[str, Any]] | None = None,
    **kwargs: Any,
) -> RouterRequest:
    """Create a RouterRequest with default messages, allowing overrides via kwargs.

    Args:
        messages: Optional custom messages list. Defaults to [{"role": "user", "content": "Hello"}].
        **kwargs: Additional parameters passed to RouterRequest (max_tokens, temperature, etc.).

    Returns:
        A RouterRequest instance.
    """
    if messages is None:
        messages = [{"role": "user", "content": "Hello"}]
    return RouterRequest(messages=messages, **kwargs)


@pytest.mark.asyncio
class TestBasicRouting:
    async def test_successful_chat(self):
        router = make_router([MockProvider("openai")])
        response = await router.chat(make_request())
        assert isinstance(response, RouterResponse)
        assert response.provider == "openai"
        assert response.attempts == 1
        assert "openai" in response.content

    async def test_no_providers_raises(self):
        router = make_router([])
        with pytest.raises(NoProvidersConfigured):
            await router.chat(make_request())


@pytest.mark.asyncio
class TestFallback:
    async def test_falls_back_when_first_provider_fails(self):
        failing = MockProvider("openai", should_fail=True)
        working = MockProvider("anthropic", weight=0.5)  # lower score → second rank
        router = make_router([failing, working])
        response = await router.chat(make_request())
        assert response.provider == "anthropic"
        assert response.attempts == 2

    async def test_all_fail_raises(self):
        providers = [
            MockProvider("openai", should_fail=True),
            MockProvider("anthropic", should_fail=True),
        ]
        router = make_router(providers)
        with pytest.raises(AllProvidersFailed) as exc_info:
            await router.chat(make_request())
        assert exc_info.value.attempts == 2

    async def test_fallback_attempts_tracked(self):
        providers = [
            MockProvider("a", should_fail=True),
            MockProvider("b", should_fail=True),
            MockProvider("c"),
        ]
        router = make_router(providers)
        response = await router.chat(make_request())
        assert response.attempts == 3


@pytest.mark.asyncio
class TestCircuitBreaker:
    async def test_circuit_trips_and_skips_provider(self):
        """After failure_threshold failures, the provider circuit trips."""
        failing = MockProvider("bad", should_fail=True)
        working = MockProvider("good", weight=0.5)
        router = make_router([failing, working], circuit_breaker=CircuitBreakerConfig(failure_threshold=2, cooldown_seconds=60))

        # Trip the circuit manually by recording failures
        await router._breaker.record_failure("bad")
        await router._breaker.record_failure("bad")

        # Now bad should be skipped
        response = await router.chat(make_request())
        assert response.provider == "good"
        assert response.attempts == 1  # bad was skipped by circuit check, not by attempt


@pytest.mark.asyncio
class TestSessionAffinity:
    async def test_same_session_routes_to_same_provider(self):
        providers = [
            MockProvider("openai", weight=1.0),
            MockProvider("anthropic", weight=0.8),
        ]
        router = make_router(providers)
        req = make_request(session_id="user-abc")

        resp1 = await router.chat(req)
        resp2 = await router.chat(req)
        assert resp1.provider == resp2.provider


@pytest.mark.asyncio
class TestProviderPinning:
    async def test_force_provider_is_used_first(self):
        providers = [
            MockProvider("openai", weight=1.0),
            MockProvider("anthropic", weight=0.5),
        ]
        router = make_router(providers)
        response = await router.chat(make_request(force_provider="anthropic"))
        assert response.provider == "anthropic"
        assert response.attempts == 1

    async def test_fallback_applies_when_pinned_fails(self):
        providers = [
            MockProvider("openai", weight=1.0),
            MockProvider("anthropic", should_fail=True, weight=0.5),
        ]
        router = make_router(providers)
        response = await router.chat(make_request(force_provider="anthropic"))
        # anthropic was tried first (pinned) but failed; fallback to openai
        assert response.provider == "openai"


@pytest.mark.asyncio
class TestStreaming:
    async def test_stream_yields_chunks(self):
        router = make_router([MockProvider("openai")])
        chunks = []
        async for chunk in router.stream(make_request()):
            chunks.append(chunk)
        assert len(chunks) > 0
        assert all(isinstance(c, str) for c in chunks)

    async def test_stream_falls_back_on_failure(self):
        providers = [
            MockProvider("openai", should_fail=True),
            MockProvider("anthropic"),
        ]
        router = make_router(providers)
        chunks = []
        async for chunk in router.stream(make_request()):
            chunks.append(chunk)
        assert "anthropic" in chunks[0]


@pytest.mark.asyncio
class TestStatusMethod:
    async def test_status_returns_all_providers(self):
        providers = [MockProvider("openai"), MockProvider("anthropic")]
        router = make_router(providers)
        status = await router.status()
        assert "openai" in status
        assert "anthropic" in status

    async def test_status_keys(self):
        router = make_router([MockProvider("openai")])
        status = await router.status()
        info = status["openai"]
        assert "rpm_used" in info
        assert "rpm_limit" in info
        assert "tpm_used" in info
        assert "tpm_limit" in info
        assert "headroom_pct" in info
        assert "circuit_open" in info
        assert "avg_latency_ms" in info


@pytest.mark.asyncio
class TestOnRouteCallback:
    async def test_callback_is_fired(self):
        fired_events = []

        async def on_route(event):
            fired_events.append(event)

        config = RouterConfig(providers=[], on_route=on_route)
        router = LLMRouter(config)
        router._registry._providers["mock"] = MockProvider("mock")
        router._initialized = True

        await router.chat(make_request())
        assert len(fired_events) == 1
        assert fired_events[0].provider == "mock"

    async def test_callback_error_does_not_propagate(self):
        async def bad_callback(event):
            raise RuntimeError("callback failed")

        config = RouterConfig(providers=[], on_route=bad_callback)
        router = LLMRouter(config)
        router._registry._providers["mock"] = MockProvider("mock")
        router._initialized = True

        # Should not raise
        response = await router.chat(make_request())
        assert response.provider == "mock"


@pytest.mark.asyncio
class TestConcurrency:
    async def test_100_concurrent_requests_no_race(self):
        """100 concurrent chat() calls should all succeed with correct tracking."""
        router = make_router([MockProvider("openai", rpm_limit=200, tpm_limit=500_000)])
        requests = [router.chat(make_request()) for _ in range(100)]
        responses = await asyncio.gather(*requests)
        assert all(r.provider == "openai" for r in responses)
        # Check usage was tracked
        rpm, tpm = await router._state.get_usage("openai", router._config.window_seconds)
        assert rpm == 100
