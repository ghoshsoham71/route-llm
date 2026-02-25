# `llm_router/state/` — State Backends

State backends track per-provider rolling usage (RPM/TPM) and session affinity (sticky routing). The router uses the state backend on every routing decision and after every completed request.

## Files

### `base.py` — `AbstractStateBackend`

Interface contract all state backends must implement:

```python
async def record_request(provider, tokens, window_seconds) → None
async def get_usage(provider, window_seconds) → (rpm, tpm)
async def get_session_provider(session_id) → str | None
async def set_session_provider(session_id, provider, ttl_seconds) → None
async def close() → None
```

### `memory.py` — `InMemoryStateBackend`

Default backend. Zero dependencies. Per-process state.

**Sliding window implementation:**
- Each provider has a `deque` of `(timestamp, token_count)` tuples.
- On `record_request`: append entry, then purge entries older than `window_seconds`.
- On `get_usage`: purge stale entries, return `(len(deque), sum(token_counts))`.
- Protected by `asyncio.Lock` for safe concurrent coroutine access.

**Session affinity:**
- `dict[session_id → (provider, expiry_timestamp)]`
- Entries are lazily expired on read.

**Suitable for:** single-instance deployments, development, testing.

### `redis.py` — `RedisStateBackend`

Redis-backed backend for multi-instance deployments. Requires `redis[asyncio]>=5.0`.

**Sliding window implementation (atomic):**
- RPM: Redis sorted set (`ZADD` + `ZREMRANGEBYSCORE` in a pipeline) keyed by `llm_router:rpm:{provider}`.
- TPM: Redis sorted set with members `"{timestamp}:{token_count}"`.
- All reads and writes use atomic `PIPELINE` + `TRANSACTION=True` to prevent race conditions across instances.
- Keys have TTL = `window_seconds * 2` to auto-expire.

**Session affinity:**
- Redis string keys with `EX` (expiry in seconds).

**Circuit breaker:**
- Circuit state is stored by `CircuitBreaker` as Redis keys with TTL. No background job needed — when the TTL expires, the provider is automatically re-admitted. See `breaker/circuit.py`.

## Selecting a Backend

The router selects the backend at initialisation time:

```python
# In-memory (default) — no config needed
router = LLMRouter.from_dict({"providers": [...]})

# Redis — set redis_url in config
router = LLMRouter.from_dict({
    "providers": [...],
    "redis_url": "redis://localhost:6379",
})
```

Or in YAML:
```yaml
redis_url: "redis://localhost:6379"
```

## Sliding Window Accuracy

**In-memory:** accurate within a single process. Multiple instances each maintain their own window — they collectively underestimate shared usage. Acceptable for single-instance or low-concurrency deployments.

**Redis:** accurate across all instances. All instances share the same sorted sets. The atomic pipeline ensures no entry is double-counted or missed even under concurrent writes.
