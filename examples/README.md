# `examples/` — Usage Examples

Runnable examples demonstrating the main usage patterns for `route-llm`.

## Prerequisites

```bash
pip install route-llm

# Set at least one provider API key
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GROQ_API_KEY="gsk-..."
```

## Files

### `quickstart.py`

The simplest way to get started. Configures the router from a Python dict with multiple providers, runs a single chat request, and prints the status of all providers.

```bash
python examples/quickstart.py
```

### `byoc.py`

Demonstrates the **Bring Your Own Client** pattern. The developer creates their own fully configured `AsyncOpenAI` and `AsyncAnthropic` clients (with custom timeouts, retries, etc.) and registers them with the router via `router.register()`. The router wraps them — it does not replace them.

```bash
python examples/byoc.py
```

### `streaming.py`

Demonstrates streaming chat completions. The router routes the request and then streams chunks from the selected provider back to the caller.

```bash
python examples/streaming.py
```

### `router.yaml`

A complete YAML configuration file covering all available options:
- Multi-provider setup (OpenAI, Anthropic, Groq)
- Scoring weight tuning
- Circuit breaker configuration
- Redis URL for multi-instance state sharing
- Window size and high-priority reserve

Use it with any `from_yaml` call:

```python
router = LLMRouter.from_yaml("examples/router.yaml")
```

Or with the CLI:

```bash
route-llm status --config examples/router.yaml
```

Environment variable interpolation is supported in YAML files:
```yaml
api_key: "${OPENAI_API_KEY}"  # replaced at load time
```
