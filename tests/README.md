# `tests/` — Test Suite

All tests use `pytest` with `pytest-asyncio` for async support. No real API calls are made — provider adapters are replaced with `MockProvider` instances.

## Running Tests

```bash
# Install dev dependencies
pip install "route-llm[dev]"

# Run all tests
pytest

# Run with verbose output
pytest -v

# Run a specific file
pytest tests/test_scorer.py

# Run with coverage
pytest --cov=llm_router --cov-report=term-missing
```

## Files

### `conftest.py`

Shared pytest fixtures:
- `basic_provider_config` — a single `ProviderConfig` for OpenAI.
- `multi_provider_config` — OpenAI + Anthropic configs.
- `router_config` — full `RouterConfig` with the above.
- `memory_state` — a fresh `InMemoryStateBackend` instance.

### `test_scorer.py`

Unit tests for `Scorer` and `LatencyTracker`:
- Ranking order under various usage states.
- Zero-headroom returns `None`.
- At-risk providers skipped for non-high priority.
- Priority weight adjustments.
- High-priority reserve blocking low/normal requests.
- EMA convergence.

### `test_circuit_breaker.py`

Unit tests for `CircuitBreaker`:
- Initially CLOSED.
- Trips after `failure_threshold` consecutive failures.
- `guard()` raises `CircuitOpenError` when OPEN.
- Provider re-admitted after cooldown.
- Success resets failure counter.
- Isolation between providers.

### `test_state_memory.py`

Unit tests for `InMemoryStateBackend`:
- Initial usage is zero.
- RPM and TPM tracking.
- Isolation between providers.
- Stale entry purging.
- Session affinity set, get, expiry.
- Concurrent access safety (100 concurrent coroutines).

### `test_predictor.py`

Unit tests for `ExhaustionPredictor`:
- No history → not at risk.
- Low consumption → not at risk.
- High consumption near capacity → at risk.
- Isolation between providers.

### `test_router.py`

Integration tests for `LLMRouter` using `MockProvider`:
- Successful routing.
- Automatic fallback on provider failure.
- `AllProvidersFailed` raised when all fail.
- Circuit breaker integration.
- Session affinity (sticky routing).
- Provider pinning (`force_provider`).
- Streaming with fallback.
- `status()` method.
- `on_route` callback fires and errors don't propagate.
- 100 concurrent requests produce correct usage tracking.

## Test Architecture

The key pattern is `MockProvider` — a `BaseProvider` subclass that returns a fixed response (or raises on demand) without any HTTP calls. The `make_router()` helper injects mock providers directly into the registry, bypassing the initialisation path.

This means tests run in milliseconds with no external dependencies.
