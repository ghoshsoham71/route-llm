# examples/quickstart.py
"""
Quickstart — LLM Router via dict config.

Run with:
  python examples/quickstart.py
"""

import asyncio
import os

from llm_router import LLMRouter, RouterRequest


async def main():
    router = LLMRouter.from_dict({
        "providers": [
            {
                "name": "openai",
                "api_key": os.environ["OPENAI_API_KEY"],
                "model": "gpt-4o",
                "rpm_limit": 500,
                "tpm_limit": 200_000,
                "weight": 0.8,
            },
            {
                "name": "anthropic",
                "api_key": os.environ["ANTHROPIC_API_KEY"],
                "model": "claude-sonnet-4-5",
                "rpm_limit": 50,
                "tpm_limit": 200_000,
                "weight": 1.0,
            },
        ]
    })

    async with router:
        response = await router.chat(RouterRequest(
            messages=[{"role": "user", "content": "Summarise the benefits of functional programming."}],
            priority="normal",
        ))

        print(f"Content:    {response.content[:200]}...")
        print(f"Provider:   {response.provider}")
        print(f"Model:      {response.model}")
        print(f"Latency:    {response.latency_ms:.1f}ms")
        print(f"Attempts:   {response.attempts}")
        print(f"Tokens in:  {response.input_tokens}")
        print(f"Tokens out: {response.output_tokens}")

        # Status overview
        status = await router.status()
        for name, info in status.items():
            print(
                f"\n{name}: RPM {info['rpm_used']}/{info['rpm_limit']}, "
                f"headroom {info['headroom_pct']}%, "
                f"circuit {'OPEN' if info['circuit_open'] else 'CLOSED'}, "
                f"latency {info['avg_latency_ms']}ms"
            )


if __name__ == "__main__":
    asyncio.run(main())
