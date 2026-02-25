# examples/byoc.py
"""
BYOC — Bring Your Own Client.

The developer keeps their existing, fully configured SDK clients.
The router wraps them and adds routing intelligence on top.

Run with:
  python examples/byoc.py
"""

import asyncio
import os

import openai
import anthropic

from llm_router import LLMRouter, RouterRequest


async def main():
    # Developer's existing, fully configured clients — unchanged.
    openai_client = openai.AsyncOpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        timeout=30,
        max_retries=0,  # Router handles retries via fallback
    )
    anthropic_client = anthropic.AsyncAnthropic(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        timeout=30,
    )

    # Router starts with no providers
    router = LLMRouter.from_dict({"providers": []})

    # Register existing clients — BYOC
    router.register(
        "openai",
        client=openai_client,
        model="gpt-4o",
        rpm=500,
        tpm=200_000,
    )
    router.register(
        "anthropic",
        client=anthropic_client,
        model="claude-sonnet-4-5",
        rpm=50,
        tpm=200_000,
    )

    async with router:
        response = await router.chat(RouterRequest(
            messages=[{"role": "user", "content": "Hello, which provider am I talking to?"}],
        ))
        print(f"Provider: {response.provider}")
        print(f"Response: {response.content}")


if __name__ == "__main__":
    asyncio.run(main())
