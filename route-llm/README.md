# `llm_router/` — Core Package

This directory contains the entire `route-llm` library. It is the installed Python package.

## File Index

| File | Purpose |
|---|---|
| `__init__.py` | Public API surface. Exports everything developers need. Never expose internals through here. |
| `router.py` | `LLMRouter` — the main class. Orchestrates the full routing pipeline: estimate → score → call → fallback → record. |
| `config.py` | `RouterConfig`, `RoutingWeights`, `CircuitBreakerConfig`. Supports `from_dict`, `from_yaml`, `from_env`. |
| `models.py` | Pydantic v2 models: `RouterRequest`, `RouterResponse`, `ProviderConfig`, `RouteEvent`. |
| `exceptions.py` | `AllProvidersFailed`, `NoProvidersConfigured`, `TokenLimitExceeded`, `CircuitOpenError`. |
| `constants.py` | All default tuning values: scoring weights, EMA alpha, window size, circuit breaker defaults, Redis key templates. |
| `cli.py` | `typer`-based CLI. Commands: `status` (with `--watch`), `dashboard`. |
| `_dashboard.py` | Streamlit dashboard. Launched via `route-llm dashboard`. Auto-refreshes every 3 seconds. |

## Subdirectories

- **`engine/`** — Scoring, token estimation, quota prediction. Pure logic, no I/O.
- **`providers/`** — Provider adapters (OpenAI, Anthropic, Gemini, Groq) + registry.
- **`state/`** — Sliding window state backends (in-memory, Redis).
- **`breaker/`** — Per-provider circuit breaker.

## Design Principles

1. **No I/O in scoring.** The `Scorer` receives all state as arguments and never touches the state backend. This makes it trivially testable.
2. **No SDK normalization.** Adapters wrap existing SDK clients. When OpenAI ships a new parameter, the developer updates their `openai` package — `route-llm` doesn't care.
3. **Async-first.** Every I/O-touching method is `async`. The router is safe to use in any asyncio event loop.
4. **BYOC.** `router.register()` accepts a pre-configured SDK client. No adapter creation needed.
