# examples/streaming.py
"""
Streaming chat completion.

Run with:
  python examples/streaming.py
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
            },
        ]
    })

    async with router:
        print("Streaming response:\n")
        async for chunk in router.stream(RouterRequest(
            messages=[{"role": "user", "content": "Tell me a short story about a robot."}],
        )):
            print(chunk, end="", flush=True)
        print("\n\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
