# `llm_router/engine/` — Routing Engine

The engine contains pure-logic components with no I/O. Every function and class here is stateless or holds only in-process state.

## Files

### `estimator.py`

Pre-flight token count estimation using `tiktoken`.

- **`estimate_tokens(messages)`** — Counts tokens in a messages list using the `cl100k_base` encoding (GPT-4-compatible). The estimate is intentionally conservative — it's better to slightly over-count and route away from a nearly-full provider than to under-count and hit a 429.
- The tiktoken encoding object is cached via `functools.lru_cache` — loading it is expensive and only needs to happen once.
- Called before every routing decision so the scorer can factor in the estimated token cost when calculating TPM headroom.

### `scorer.py`

Ranks available providers using a weighted scoring formula.

**`Scorer`** — stateless scoring engine. All inputs come from the caller (usage data from state backend, latency from LatencyTracker). This makes the scorer 100% unit-testable without any I/O.

**`LatencyTracker`** — per-provider exponential moving average (EMA) of response latency. Alpha = 0.2 by default. Intentionally per-instance (not shared via Redis) to avoid writing to Redis on every completed request.

**Scoring formula:**
```
score = (capacity_score × w_capacity)
      + (latency_score  × w_latency)
      + (static_score   × w_static)

capacity_score = min(rpm_headroom, tpm_headroom)
latency_score  = max(0, 1 - latency_ema_ms / 3000)
static_score   = provider.weight
```

**Priority adjustments:**
- `"high"` → `w_capacity=0.5, w_latency=0.4, w_static=0.1`
- `"normal"` → `w_capacity=0.5, w_latency=0.3, w_static=0.2`
- `"low"` → `w_capacity=0.3, w_latency=0.1, w_static=0.6`

### `predictor.py`

Quota exhaustion prediction using consumption rate analysis.

**`ExhaustionPredictor`** — records per-provider usage history and predicts whether a provider is at risk of exhausting its quota within the look-ahead window (default: 2 minutes).

Algorithm:
1. Record `(timestamp, tokens)` tuples in a rolling history deque.
2. Calculate observed RPM and TPM rates over the last N seconds.
3. If consumption rate exceeds `multiplier × average` AND projected exhaustion < look-ahead, mark as at risk.
4. At-risk providers are skipped for `"low"` and `"normal"` priority requests. High-priority requests still route to them.

This prevents reactive 429 errors for predictable traffic patterns.

## Testing

All three components are tested in isolation in `tests/test_scorer.py` and `tests/test_predictor.py`. No mocking of I/O is required because the engine has no I/O.
