# tests/test_predictor.py
"""
Tests for ExhaustionPredictor.
"""

from __future__ import annotations

import pytest

from llm_router.engine.predictor import ExhaustionPredictor


class TestExhaustionPredictor:
    def test_no_history_not_at_risk(self):
        predictor = ExhaustionPredictor()
        assert not predictor.is_at_risk(
            "p", rpm_used=50, rpm_limit=100, tpm_used=10_000, tpm_limit=50_000
        )

    def test_low_consumption_not_at_risk(self):
        predictor = ExhaustionPredictor(multiplier=3.0)
        for _ in range(5):
            predictor.record("p", tokens=100)
        assert not predictor.is_at_risk(
            "p", rpm_used=5, rpm_limit=100, tpm_used=500, tpm_limit=50_000
        )

    def test_high_consumption_at_nearly_full_is_at_risk(self):
        """Simulate a spike: many requests, near capacity, fast consumption."""
        predictor = ExhaustionPredictor(
            window_seconds=60,
            look_ahead_seconds=120,
            multiplier=1.0,  # any elevated rate triggers it
        )
        # Record many large requests to drive up consumption rate
        for _ in range(50):
            predictor.record("p", tokens=1_000)
        # Nearly at capacity
        at_risk = predictor.is_at_risk(
            "p",
            rpm_used=95,
            rpm_limit=100,
            tpm_used=48_000,
            tpm_limit=50_000,
        )
        assert at_risk

    def test_isolation_between_providers(self):
        predictor = ExhaustionPredictor(multiplier=1.0)
        for _ in range(50):
            predictor.record("a", tokens=1_000)
        # Provider b should not be at risk
        assert not predictor.is_at_risk(
            "b", rpm_used=95, rpm_limit=100, tpm_used=48_000, tpm_limit=50_000
        )
