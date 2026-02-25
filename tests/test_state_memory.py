# tests/test_state_memory.py
"""
Tests for InMemoryStateBackend.

Verifies:
  - Sliding window accuracy.
  - RPM and TPM tracking.
  - Session affinity storage and retrieval.
  - Concurrent access safety.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from llm_router.state.memory import InMemoryStateBackend


@pytest.mark.asyncio
class TestInMemoryStateBackend:
    async def test_initial_usage_is_zero(self):
        state = InMemoryStateBackend()
        rpm, tpm = await state.get_usage("openai", window_seconds=60)
        assert rpm == 0
        assert tpm == 0

    async def test_record_increments_rpm(self):
        state = InMemoryStateBackend()
        await state.record_request("openai", tokens=100, window_seconds=60)
        rpm, _ = await state.get_usage("openai", window_seconds=60)
        assert rpm == 1

    async def test_record_accumulates_tpm(self):
        state = InMemoryStateBackend()
        await state.record_request("openai", tokens=300, window_seconds=60)
        await state.record_request("openai", tokens=200, window_seconds=60)
        _, tpm = await state.get_usage("openai", window_seconds=60)
        assert tpm == 500

    async def test_isolation_between_providers(self):
        state = InMemoryStateBackend()
        await state.record_request("openai", tokens=100, window_seconds=60)
        rpm_ant, tpm_ant = await state.get_usage("anthropic", window_seconds=60)
        assert rpm_ant == 0
        assert tpm_ant == 0

    async def test_old_entries_are_purged(self):
        """Entries older than window_seconds should not be counted."""
        state = InMemoryStateBackend()
        # Manually inject an old entry
        old_timestamp = time.time() - 120
        state._windows["p"].append((old_timestamp, 1000))
        # Record a fresh entry
        await state.record_request("p", tokens=50, window_seconds=60)
        rpm, tpm = await state.get_usage("p", window_seconds=60)
        # Old entry should be purged
        assert rpm == 1
        assert tpm == 50

    async def test_session_affinity_set_and_get(self):
        state = InMemoryStateBackend()
        await state.set_session_provider("sess-1", "openai", ttl_seconds=60)
        result = await state.get_session_provider("sess-1")
        assert result == "openai"

    async def test_session_not_found_returns_none(self):
        state = InMemoryStateBackend()
        result = await state.get_session_provider("no-such-session")
        assert result is None

    async def test_session_expires_after_ttl(self):
        state = InMemoryStateBackend()
        await state.set_session_provider("sess-exp", "openai", ttl_seconds=1)
        await asyncio.sleep(1.1)
        result = await state.get_session_provider("sess-exp")
        assert result is None

    async def test_concurrent_access_no_race(self):
        """100 concurrent record_request calls should produce correct counts."""
        state = InMemoryStateBackend()
        n = 100
        await asyncio.gather(
            *[state.record_request("p", tokens=1, window_seconds=60) for _ in range(n)]
        )
        rpm, tpm = await state.get_usage("p", window_seconds=60)
        assert rpm == n
        assert tpm == n
