# tests/conftest.py
"""
Shared pytest fixtures for route-llm tests.
"""

from __future__ import annotations

import pytest
import pytest_asyncio

from llm_router.config import CircuitBreakerConfig, RouterConfig, RoutingWeights
from llm_router.models import ProviderConfig
from llm_router.state.memory import InMemoryStateBackend


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def basic_provider_config():
    return ProviderConfig(
        name="openai",
        model="gpt-4o",
        api_key="sk-test",
        rpm_limit=100,
        tpm_limit=50_000,
        weight=1.0,
    )


@pytest.fixture
def multi_provider_config():
    return [
        ProviderConfig(
            name="openai",
            model="gpt-4o",
            api_key="sk-test-openai",
            rpm_limit=100,
            tpm_limit=50_000,
            weight=0.8,
        ),
        ProviderConfig(
            name="anthropic",
            model="claude-sonnet-4-5",
            api_key="sk-ant-test",
            rpm_limit=50,
            tpm_limit=50_000,
            weight=1.0,
        ),
    ]


@pytest.fixture
def router_config(multi_provider_config):
    return RouterConfig(
        providers=multi_provider_config,
        weights=RoutingWeights(),
        circuit_breaker=CircuitBreakerConfig(failure_threshold=3, cooldown_seconds=5),
    )


@pytest_asyncio.fixture
async def memory_state():
    return InMemoryStateBackend()
