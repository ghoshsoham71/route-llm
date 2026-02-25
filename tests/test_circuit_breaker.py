# tests/test_circuit_breaker.py
"""
Unit tests for CircuitBreaker.

Verifies:
  - Circuit trips after failure_threshold consecutive failures.
  - Tripped circuit blocks all requests (is_open → True).
  - guard() raises CircuitOpenError when circuit is open.
  - Provider is re-admitted after cooldown.
  - Successful request after cooldown resets failure count.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from llm_router.breaker.circuit import CircuitBreaker
from llm_router.exceptions import CircuitOpenError


@pytest.mark.asyncio
class TestCircuitBreaker:
    async def test_initially_closed(self):
        cb = CircuitBreaker(failure_threshold=3, cooldown_seconds=5)
        assert not await cb.is_open("provider_a")

    async def test_single_failure_does_not_trip(self):
        cb = CircuitBreaker(failure_threshold=3, cooldown_seconds=5)
        await cb.record_failure("provider_a")
        assert not await cb.is_open("provider_a")

    async def test_trips_after_threshold(self):
        cb = CircuitBreaker(failure_threshold=3, cooldown_seconds=5)
        for _ in range(3):
            await cb.record_failure("provider_a")
        assert await cb.is_open("provider_a")

    async def test_guard_raises_when_open(self):
        cb = CircuitBreaker(failure_threshold=2, cooldown_seconds=5)
        await cb.record_failure("p")
        await cb.record_failure("p")
        with pytest.raises(CircuitOpenError):
            await cb.guard("p")

    async def test_guard_does_not_raise_when_closed(self):
        cb = CircuitBreaker(failure_threshold=3, cooldown_seconds=5)
        await cb.guard("provider_a")  # Should not raise

    async def test_success_resets_failure_count(self):
        cb = CircuitBreaker(failure_threshold=5, cooldown_seconds=5)
        for _ in range(4):
            await cb.record_failure("p")
        await cb.record_success("p")
        # Should need 5 more failures to trip now
        for _ in range(4):
            await cb.record_failure("p")
        assert not await cb.is_open("p")

    async def test_re_admitted_after_cooldown(self):
        cb = CircuitBreaker(failure_threshold=1, cooldown_seconds=1)
        await cb.record_failure("p")
        assert await cb.is_open("p")
        # Wait for cooldown
        await asyncio.sleep(1.1)
        assert not await cb.is_open("p")

    async def test_isolation_between_providers(self):
        cb = CircuitBreaker(failure_threshold=2, cooldown_seconds=5)
        await cb.record_failure("a")
        await cb.record_failure("a")
        # Provider b should be unaffected
        assert await cb.is_open("a")
        assert not await cb.is_open("b")

    async def test_get_status_returns_dict(self):
        cb = CircuitBreaker(failure_threshold=2, cooldown_seconds=5)
        status = await cb.get_status("p")
        assert "circuit_open" in status
        assert "failure_count" in status
        assert status["circuit_open"] is False
        assert status["failure_count"] == 0
