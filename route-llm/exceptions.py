# llm_router/exceptions.py
"""
Custom exceptions for route-llm.

All public exceptions inherit from LLMRouterError so callers can catch
the whole family with a single except clause if preferred.
"""

from __future__ import annotations


class LLMRouterError(Exception):
    """Base exception for all router errors."""


class NoProvidersConfigured(LLMRouterError):
    """Raised when router.chat() is called with no providers registered."""


class AllProvidersFailed(LLMRouterError):
    """
    Raised when every provider in the fallback chain has been tried and failed.

    Attributes
    ----------
    attempts:
        Number of providers that were attempted.
    errors:
        List of exceptions raised by each provider attempt, in order.
    """

    def __init__(self, message: str, attempts: int, errors: list[Exception]) -> None:
        self.attempts = attempts
        self.errors = errors
        super().__init__(message)

    def __str__(self) -> str:  # pragma: no cover
        base = super().__str__()
        details = "; ".join(f"[{i+1}] {type(e).__name__}: {e}" for i, e in enumerate(self.errors))
        return f"{base} | Errors: {details}"


class TokenLimitExceeded(LLMRouterError):
    """
    Raised when the estimated token count for a request exceeds the TPM
    limit of every available provider.
    """


class CircuitOpenError(LLMRouterError):
    """
    Raised *internally* when a provider's circuit breaker is open.
    This exception is never surfaced to the developer — the router catches
    it and skips to the next provider.
    """

    def __init__(self, provider: str) -> None:
        self.provider = provider
        super().__init__(f"Circuit breaker is open for provider '{provider}'")
