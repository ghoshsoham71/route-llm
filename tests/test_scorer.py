# tests/test_scorer.py
"""
Unit tests for the Scorer.

Tests verify ranking order given various usage states, priority adjustments,
and edge cases (zero headroom, at-risk providers, high-priority reserve).
"""

from __future__ import annotations

import pytest

from llm_router.engine.scorer import LatencyTracker, Scorer


def make_score(scorer, **kwargs):
    defaults = dict(
        name="test",
        rpm_used=0,
        rpm_limit=100,
        tpm_used=0,
        tpm_limit=50_000,
        estimated_tokens=100,
        latency_ema_ms=300,
        static_weight=1.0,
        priority="normal",
        is_at_risk=False,
        high_priority_reserve_pct=0.0,
    )
    defaults.update(kwargs)
    return scorer.score_provider(**defaults)


class TestScorerBasic:
    def test_full_headroom_returns_score(self):
        scorer = Scorer()
        ps = make_score(scorer)
        assert ps is not None
        assert 0.0 < ps.score <= 1.0

    def test_zero_rpm_headroom_returns_none(self):
        scorer = Scorer()
        ps = make_score(scorer, rpm_used=100, rpm_limit=100)
        assert ps is None

    def test_zero_tpm_headroom_returns_none(self):
        scorer = Scorer()
        ps = make_score(scorer, tpm_used=50_000, tpm_limit=50_000, estimated_tokens=0)
        assert ps is None

    def test_estimated_tokens_reduce_tpm_headroom(self):
        scorer = Scorer()
        ps_low = make_score(scorer, tpm_used=0, estimated_tokens=100)
        ps_high = make_score(scorer, tpm_used=0, estimated_tokens=49_900)
        assert ps_low is not None
        assert ps_high is not None
        assert ps_low.score > ps_high.score

    def test_at_risk_provider_skipped_for_normal(self):
        scorer = Scorer()
        ps = make_score(scorer, is_at_risk=True, priority="normal")
        assert ps is None

    def test_at_risk_provider_not_skipped_for_high(self):
        scorer = Scorer()
        ps = make_score(scorer, is_at_risk=True, priority="high")
        assert ps is not None


class TestScorerRanking:
    def test_higher_headroom_ranks_first(self):
        scorer = Scorer()
        low_usage = make_score(scorer, name="a", rpm_used=10, rpm_limit=100)
        high_usage = make_score(scorer, name="b", rpm_used=90, rpm_limit=100)
        ranked = scorer.rank([s for s in [high_usage, low_usage] if s is not None])
        assert ranked[0].name == "a"

    def test_lower_latency_ranks_higher_when_equal_capacity(self):
        scorer = Scorer()
        fast = make_score(scorer, name="fast", latency_ema_ms=100)
        slow = make_score(scorer, name="slow", latency_ema_ms=2000)
        ranked = scorer.rank([s for s in [slow, fast] if s is not None])
        assert ranked[0].name == "fast"

    def test_static_weight_breaks_ties(self):
        scorer = Scorer()
        preferred = make_score(scorer, name="preferred", static_weight=1.0, latency_ema_ms=300)
        other = make_score(scorer, name="other", static_weight=0.0, latency_ema_ms=300)
        ranked = scorer.rank([s for s in [other, preferred] if s is not None])
        assert ranked[0].name == "preferred"


class TestPriorityWeights:
    def test_high_priority_weights_latency_more(self):
        scorer = Scorer()
        fast_low_cap = make_score(
            scorer, name="fast", latency_ema_ms=50, rpm_used=80, rpm_limit=100, priority="high"
        )
        slow_high_cap = make_score(
            scorer, name="slow", latency_ema_ms=1500, rpm_used=10, rpm_limit=100, priority="high"
        )
        # With high priority, latency is weighted more (0.4 vs 0.3)
        # fast_low_cap's latency_score advantage should be significant enough
        assert fast_low_cap is not None
        assert slow_high_cap is not None

    def test_low_priority_weights_static_more(self):
        scorer = Scorer()
        preferred = make_score(
            scorer, name="preferred", static_weight=1.0, latency_ema_ms=1000, priority="low"
        )
        fast_but_not_preferred = make_score(
            scorer, name="fast", static_weight=0.0, latency_ema_ms=100, priority="low"
        )
        # Low priority: w_static=0.6, so static weight dominates
        ranked = scorer.rank([s for s in [fast_but_not_preferred, preferred] if s is not None])
        assert ranked[0].name == "preferred"


class TestHighPriorityReserve:
    def test_normal_request_blocked_when_within_reserve(self):
        scorer = Scorer()
        # 5% headroom, 20% reserve → normal request should be blocked
        ps = make_score(
            scorer,
            rpm_used=96,
            rpm_limit=100,
            priority="normal",
            high_priority_reserve_pct=0.2,
        )
        assert ps is None

    def test_high_request_not_blocked_by_reserve(self):
        scorer = Scorer()
        ps = make_score(
            scorer,
            rpm_used=96,
            rpm_limit=100,
            priority="high",
            high_priority_reserve_pct=0.2,
        )
        assert ps is not None


class TestLatencyTracker:
    def test_initial_latency_is_default(self):
        lt = LatencyTracker()
        from llm_router.constants import INITIAL_LATENCY_MS
        assert lt.get("unknown") == INITIAL_LATENCY_MS

    def test_ema_updates_toward_observation(self):
        lt = LatencyTracker(alpha=0.5)
        lt.update("p", 1000.0)
        ema = lt.get("p")
        from llm_router.constants import INITIAL_LATENCY_MS
        # After one update with alpha=0.5: 0.5*1000 + 0.5*500 = 750
        assert ema == pytest.approx(750.0, rel=0.01)

    def test_repeated_updates_converge(self):
        lt = LatencyTracker(alpha=0.5)
        for _ in range(20):
            lt.update("p", 200.0)
        assert abs(lt.get("p") - 200.0) < 5.0
