# llm_router/breaker/circuit.py
"""
Per-provider circuit breaker.

State machine
-------------
CLOSED  → normal operation. Failures increment a counter.
OPEN    → provider is blocked for cooldown_seconds.
HALF-OPEN is implicit: after cooldown the provider is re-admitted and, if the
next request succeeds, the failure counter is reset (CLOSED). If it fails, the
circuit trips again immediately.

Multi-instance support
----------------------
When a RedisStateBackend is used, circuit state is stored as a Redis key with
TTL equal to cooldown_seconds. No background job is needed — when the key
expires the provider is automatically re-admitted and all instances see the
same state.

For the in-memory backend, state is per-process. This is acceptable for
single-instance deployments.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..constants import CIRCUIT_COOLDOWN_SECONDS, CIRCUIT_FAILURE_THRESHOLD, REDIS_CIRCUIT_KEY_TMPL
from ..exceptions import CircuitOpenError


@dataclass
class _InMemoryCircuitState:
    failures: int = 0
    open_until: float = 0.0  # epoch timestamp; 0 means CLOSED


class CircuitBreaker:
    """
    Thread/coroutine-safe circuit breaker.

    When redis_client is provided, circuit open state is backed by Redis
    keys with TTL. Otherwise pure in-memory state is used.

    Parameters
    ----------
    failure_threshold:
        Consecutive failures required to trip the circuit.
    cooldown_seconds:
        Seconds the circuit stays OPEN before the provider is re-admitted.
    redis_client:
        Optional async Redis client (from redis.asyncio). When provided,
        circuit OPEN state is stored as a Redis key with TTL.
    """

    def __init__(
        self,
        failure_threshold: int = CIRCUIT_FAILURE_THRESHOLD,
        cooldown_seconds: int = CIRCUIT_COOLDOWN_SECONDS,
        redis_client: object | None = None,
    ) -> None:
        self._threshold = failure_threshold
        self._cooldown = cooldown_seconds
        self._redis = redis_client
        # provider → local state (used even with Redis for failure count)
        self._state: dict[str, _InMemoryCircuitState] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_state(self, provider: str) -> _InMemoryCircuitState:
        if provider not in self._state:
            self._state[provider] = _InMemoryCircuitState()
        return self._state[provider]

    async def _redis_is_open(self, provider: str) -> bool:
        if self._redis is None:
            return False
        key = REDIS_CIRCUIT_KEY_TMPL.format(provider=provider)
        return await self._redis.exists(key)  # type: ignore[union-attr]

    async def _redis_set_open(self, provider: str) -> None:
        if self._redis is None:
            return
        key = REDIS_CIRCUIT_KEY_TMPL.format(provider=provider)
        await self._redis.set(key, "1", ex=self._cooldown)  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def is_open(self, provider: str) -> bool:
        """Return True if the circuit is currently OPEN for *provider*."""
        # Check Redis first (authoritative for multi-instance)
        if await self._redis_is_open(provider):
            return True

        # Check local state (for in-memory deployments)
        state = self._get_state(provider)
        if state.open_until > 0 and time.time() < state.open_until:
            return True

        # Cooldown elapsed — reset local open_until
        if state.open_until > 0 and time.time() >= state.open_until:
            state.open_until = 0.0
            state.failures = 0

        return False

    async def guard(self, provider: str) -> None:
        """
        Raise CircuitOpenError if the circuit for *provider* is OPEN.
        Call this before attempting a provider request.
        """
        if await self.is_open(provider):
            raise CircuitOpenError(provider)

    async def record_success(self, provider: str) -> None:
        """Record a successful request — resets the failure counter (CLOSED)."""
        state = self._get_state(provider)
        state.failures = 0
        state.open_until = 0.0

    async def record_failure(self, provider: str) -> None:
        """
        Record a failed request.
        Trips the circuit if consecutive failures reach the threshold.
        """
        state = self._get_state(provider)
        state.failures += 1

        if state.failures >= self._threshold:
            state.open_until = time.time() + self._cooldown
            await self._redis_set_open(provider)

    async def get_status(self, provider: str) -> dict:
        """Return a dict with current circuit state for *provider*."""
        open_ = await self.is_open(provider)
        state = self._get_state(provider)
        return {
            "circuit_open": open_,
            "failure_count": state.failures,
            "open_until": state.open_until if open_ else None,
        }
