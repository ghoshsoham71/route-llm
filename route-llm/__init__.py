# llm_router/__init__.py
"""
route-llm — Adaptive rate-limit-aware LLM routing. Bring your own clients.

Public API surface:
  LLMRouter          — main class; register providers, call chat() / stream()
  RouterConfig       — top-level configuration model
  RoutingWeights     — weight coefficients for the scoring formula
  CircuitBreakerConfig — circuit breaker tuning
  RouterRequest      — request model passed to chat() / stream()
  RouterResponse     — response model returned by chat()
  ProviderConfig     — per-provider config used in RouterConfig
  RouteEvent         — event fired by the on_route callback
  AllProvidersFailed — raised when every provider in the chain fails
  NoProvidersConfigured — raised when no providers are registered
  TokenLimitExceeded — raised when estimated tokens exceed all TPM limits
"""

from .router import LLMRouter
from .config import RouterConfig, RoutingWeights, CircuitBreakerConfig
from .models import RouterRequest, RouterResponse, ProviderConfig, RouteEvent
from .exceptions import AllProvidersFailed, NoProvidersConfigured, TokenLimitExceeded

__all__ = [
    "LLMRouter",
    "RouterConfig",
    "RoutingWeights",
    "CircuitBreakerConfig",
    "RouterRequest",
    "RouterResponse",
    "ProviderConfig",
    "RouteEvent",
    "AllProvidersFailed",
    "NoProvidersConfigured",
    "TokenLimitExceeded",
]

__version__ = "0.1.0"
