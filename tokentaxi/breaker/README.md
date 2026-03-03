# `llm_router/breaker/` — Circuit Breaker

The circuit breaker prevents the router from sending requests to a provider that is repeatedly failing, giving it time to recover before being re-admitted.

## File

### `circuit.py` — `CircuitBreaker`

**State machine:**
- **CLOSED** (normal) — requests flow through. Failures increment a per-provider counter.
- **OPEN** (blocked) — provider is excluded from routing for `cooldown_seconds`. No requests are sent.
- **HALF-OPEN** (implicit) — after `cooldown_seconds` elapses, the provider is re-admitted. If the next request succeeds, the counter resets (back to CLOSED). If it fails, the circuit trips again immediately.

**API:**
```python
cb = CircuitBreaker(failure_threshold=5, cooldown_seconds=30)

await cb.guard("openai")           # raises CircuitOpenError if OPEN
await cb.record_success("openai")  # resets failure counter
await cb.record_failure("openai")  # increments counter; trips if >= threshold
await cb.is_open("openai")         # bool: is the circuit OPEN?
await cb.get_status("openai")      # dict with circuit_open, failure_count, open_until
```

**Multi-instance support (Redis):**

When a `redis_client` is passed to `CircuitBreaker.__init__`, the OPEN state is stored as a Redis key with TTL equal to `cooldown_seconds`:

```
Key:   llm_router:circuit:{provider}
Value: "1"
TTL:   cooldown_seconds
```

When the TTL expires, the key disappears and the provider is automatically re-admitted. All router instances see the same circuit state immediately. No background job or polling is needed.

Failure *count* is always tracked in-process (not Redis) — this avoids a Redis write on every request and slight cross-instance inconsistency in failure counting is acceptable.

**CircuitOpenError:**

`CircuitOpenError` is an internal exception. It is raised by `guard()` and caught immediately by the router's fallback loop. It is never surfaced to the developer's application.

## Tuning

| Parameter | Default | Description |
|---|---|---|
| `failure_threshold` | 5 | Consecutive failures before tripping. |
| `cooldown_seconds` | 30 | Seconds the circuit stays OPEN. |

Set via `CircuitBreakerConfig` in `RouterConfig`:

```python
RouterConfig(
    circuit_breaker=CircuitBreakerConfig(
        failure_threshold=3,
        cooldown_seconds=60,
    ),
    ...
)
```
