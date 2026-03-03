# llm_router/engine/predictor.py
"""
Quota exhaustion prediction.

Monitors the rate at which each provider's quota is being consumed.
If a provider is consuming quota at 3× the average rate *and* is projected
to exhaust quota within the look-ahead window, the predictor signals that
load should be shifted away — before the hard limit is hit.

This avoids reactive 429 errors for predictable traffic patterns.

Algorithm
---------
1. Record (timestamp, tokens) tuples in a short history deque.
2. Calculate the observed consumption rate over the last N seconds.
3. Project how long until the remaining quota is exhausted.
4. If projected exhaustion < PREDICTION_WINDOW_SECONDS, mark as "at risk".
"""

from __future__ import annotations

import time
from collections import defaultdict, deque

from ..constants import (
    PREDICTION_CONSUMPTION_MULTIPLIER,
    PREDICTION_WINDOW_SECONDS,
    WINDOW_SECONDS,
)


class ExhaustionPredictor:
    """
    Tracks per-provider consumption velocity and predicts quota exhaustion.
    """

    def __init__(
        self,
        window_seconds: int = WINDOW_SECONDS,
        look_ahead_seconds: int = PREDICTION_WINDOW_SECONDS,
        multiplier: float = PREDICTION_CONSUMPTION_MULTIPLIER,
    ) -> None:
        self._window = window_seconds
        self._look_ahead = look_ahead_seconds
        self._multiplier = multiplier
        # provider → deque of (timestamp, token_count)
        self._history: dict[str, deque[tuple[float, int]]] = defaultdict(deque)

    def record(self, provider: str, tokens: int) -> None:
        """Record a completed request for *provider*."""
        now = time.time()
        dq = self._history[provider]
        dq.append((now, tokens))
        cutoff = now - self._window
        while dq and dq[0][0] < cutoff:
            dq.popleft()

    def is_at_risk(self, provider: str, rpm_used: int, rpm_limit: int, tpm_used: int, tpm_limit: int) -> bool:
        """
        Return True if *provider* is projected to exhaust quota within
        the look-ahead window under the current consumption rate.

        Parameters
        ----------
        provider:
            Provider identifier.
        rpm_used, rpm_limit:
            Current RPM usage and limit.
        tpm_used, tpm_limit:
            Current TPM usage and limit.
        """
        now = time.time()
        dq = self._history.get(provider)
        if not dq:
            return False

        # Purge old history
        cutoff = now - self._window
        while dq and dq[0][0] < cutoff:
            dq.popleft()

        if not dq:
            return False

        elapsed = max(now - dq[0][0], 1.0)  # avoid div-by-zero
        observed_rpm = len(dq) / elapsed * 60  # requests per minute
        observed_tpm = sum(t for _, t in dq) / elapsed * 60  # tokens per minute

        avg_rpm = rpm_limit * 0.5  # assume 50% average utilisation
        avg_tpm = tpm_limit * 0.5

        # Check if consumption rate is elevated
        rpm_elevated = observed_rpm > avg_rpm * self._multiplier
        tpm_elevated = observed_tpm > avg_tpm * self._multiplier

        if not (rpm_elevated or tpm_elevated):
            return False

        # Project time to exhaustion
        rpm_remaining = rpm_limit - rpm_used
        tpm_remaining = tpm_limit - tpm_used

        if observed_rpm > 0:
            seconds_to_rpm_exhaustion = (rpm_remaining / observed_rpm) * 60
        else:
            seconds_to_rpm_exhaustion = float("inf")

        if observed_tpm > 0:
            seconds_to_tpm_exhaustion = (tpm_remaining / observed_tpm) * 60
        else:
            seconds_to_tpm_exhaustion = float("inf")

        projected = min(seconds_to_rpm_exhaustion, seconds_to_tpm_exhaustion)
        return projected < self._look_ahead
