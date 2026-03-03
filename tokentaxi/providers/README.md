# `llm_router/providers/` — Provider Adapters

Each adapter wraps a provider's async SDK client and exposes a uniform interface to the router. The router never calls provider SDKs directly — it always goes through an adapter.

## Files

### `base.py` — `BaseProvider`

Abstract base class all adapters must implement:
- `async chat(messages, max_tokens, temperature, **kwargs) → (content, input_tokens, output_tokens)`
- `async stream(messages, max_tokens, temperature, **kwargs) → AsyncIterator[str]`
- `async close()` — release resources (HTTP client connections, etc.)

Attributes every adapter exposes: `name`, `model`, `rpm_limit`, `tpm_limit`, `weight`, `enabled`.

### `registry.py` — `ProviderRegistry`

Thread-safe container for all registered providers. Uses `asyncio.Lock` for concurrent access.

Key methods:
- `register_from_config(config: ProviderConfig)` — creates an adapter from config and registers it.
- `register_adapter(adapter: BaseProvider)` — registers a pre-built adapter.
- `register_byoc(name, client, model, rpm, tpm, weight)` — BYOC registration path.
- `get_all()` → all enabled providers.
- `get(name)` → single provider by name, or None.
- `close_all()` → closes all adapters.

### `openai.py` — `OpenAIProvider`

Wraps `openai.AsyncOpenAI`. Supports both config-based and BYOC modes.

### `anthropic.py` — `AnthropicProvider`

Wraps `anthropic.AsyncAnthropic`. Handles system message extraction (Anthropic separates system from the messages list). Supports BYOC.

### `gemini.py` — `GeminiProvider`

Wraps `google.generativeai.GenerativeModel`. Converts OpenAI-style messages to Gemini's format. Supports BYOC.

### `groq.py` — `GroqProvider`

Wraps `groq.AsyncGroq`. Groq's API is OpenAI-compatible, so the adapter is nearly identical to `OpenAIProvider`. Supports BYOC.

## Adding a Custom Provider

Subclass `BaseProvider`, implement `chat()` and `stream()`, then register it:

```python
from llm_router.providers.base import BaseProvider

class MyCustomProvider(BaseProvider):
    async def chat(self, messages, max_tokens, temperature, **kwargs):
        # Call your custom endpoint
        ...
        return content, input_tokens, output_tokens

    async def stream(self, messages, max_tokens, temperature, **kwargs):
        # Yield chunks
        yield chunk

adapter = MyCustomProvider(name="my-provider", model="my-model", rpm_limit=100, tpm_limit=50_000)
await router._registry.register_adapter(adapter)
```

## BYOC Pattern

For built-in providers, use `router.register()`:

```python
router.register("openai", client=openai.AsyncOpenAI(...), model="gpt-4o", rpm=500, tpm=200_000)
```

The registry infers the correct adapter class from the `name` argument and injects the supplied client. The developer's SDK configuration — authentication, timeouts, retry settings — is preserved exactly as-is.
