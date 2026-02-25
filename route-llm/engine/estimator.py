# llm_router/engine/estimator.py
"""
Pre-flight token count estimation.

Uses tiktoken to count tokens before routing so the scorer can calculate
TPM headroom accurately, and so TokenLimitExceeded can be raised early
rather than discovering the limit mid-request.

Provider-specific tokenisers differ, but cl100k_base (used by GPT-4 and
Claude via compatibility) is a close-enough approximation for routing
decisions. The estimate is intentionally conservative — it's better to
slightly over-count and route away from a nearly-full provider than to
under-count and hit a 429.
"""

from __future__ import annotations

import functools
from typing import Any

_ENCODING_NAME = "cl100k_base"
_OVERHEAD_PER_MESSAGE = 4  # role + separators in chat format


@functools.lru_cache(maxsize=1)
def _get_encoding():  # type: ignore[return]
    """Cache the tiktoken encoding object — loading it is expensive."""
    try:
        import tiktoken  # type: ignore[import]

        return tiktoken.get_encoding(_ENCODING_NAME)
    except ImportError as exc:
        raise ImportError(
            "tiktoken is required for token estimation. "
            "Install it with: pip install tiktoken"
        ) from exc


def estimate_tokens(messages: list[dict[str, Any]]) -> int:
    """
    Estimate the total number of tokens for a list of chat messages.

    This is a best-effort estimate. The actual token count may differ
    slightly depending on the provider and model.

    Parameters
    ----------
    messages:
        Chat messages in the standard format:
        [{"role": "user", "content": "Hello"}, ...]

    Returns
    -------
    int
        Estimated token count for the entire messages list.
    """
    enc = _get_encoding()
    total = 0
    for message in messages:
        total += _OVERHEAD_PER_MESSAGE
        for key, value in message.items():
            if isinstance(value, str):
                total += len(enc.encode(value))
    # Add 2 for the reply primer
    total += 2
    return total
