"""Unit tests for the ReturnGuard Agent decision policy.

Run with: pytest tests/test_policy.py -v
"""

from datetime import datetime, timedelta

import pytest

from src.policy import ALLOW, FLAG_FOR_REVIEW, RESTRICT_COD, DecisionPolicy


def test_low_risk_is_allowed():
    policy = DecisionPolicy()
    decision = policy.decide(risk_score=0.10, payment_method="UPI")
    assert decision.action == ALLOW
    assert not decision.capped


def test_high_risk_cod_is_restricted():
    policy = DecisionPolicy()
    decision = policy.decide(risk_score=0.80, payment_method="COD")
    assert decision.action == RESTRICT_COD


def test_high_risk_non_cod_is_flagged_not_restricted():
    """restrict_cod is meaningless on a non-COD order - it should fall
    back to flag_for_review instead of doing nothing or erroring."""
    policy = DecisionPolicy()
    decision = policy.decide(risk_score=0.80, payment_method="card")
    assert decision.action == FLAG_FOR_REVIEW


def test_mid_risk_is_flagged():
    policy = DecisionPolicy()
    decision = policy.decide(risk_score=0.50, payment_method="COD")
    assert decision.action == FLAG_FOR_REVIEW


def test_hourly_cap_downgrades_to_flag_for_review():
    policy = DecisionPolicy(max_restrict_cod_per_hour=2)
    ts = datetime(2026, 1, 1, 10, 0)

    d1 = policy.decide(0.9, "COD", timestamp=ts)
    d2 = policy.decide(0.9, "COD", timestamp=ts + timedelta(minutes=5))
    d3 = policy.decide(0.9, "COD", timestamp=ts + timedelta(minutes=10))

    assert d1.action == RESTRICT_COD and not d1.capped
    assert d2.action == RESTRICT_COD and not d2.capped
    assert d3.action == FLAG_FOR_REVIEW and d3.capped  # cap hit on the 3rd call


def test_cap_resets_in_a_new_hour_bucket():
    policy = DecisionPolicy(max_restrict_cod_per_hour=1)
    d1 = policy.decide(0.9, "COD", timestamp=datetime(2026, 1, 1, 10, 30))
    d2 = policy.decide(0.9, "COD", timestamp=datetime(2026, 1, 1, 11, 5))  # new hour bucket
    assert d1.action == RESTRICT_COD
    assert d2.action == RESTRICT_COD  # cap is per hour bucket, so this is allowed


def test_reset_clears_the_cap_within_the_same_hour():
    policy = DecisionPolicy(max_restrict_cod_per_hour=1)
    ts = datetime(2026, 1, 1, 10, 0)
    policy.decide(0.9, "COD", timestamp=ts)
    policy.reset()
    d2 = policy.decide(0.9, "COD", timestamp=ts + timedelta(minutes=1))
    assert d2.action == RESTRICT_COD and not d2.capped


def test_invalid_risk_score_raises():
    policy = DecisionPolicy()
    with pytest.raises(ValueError):
        policy.decide(risk_score=1.5, payment_method="UPI")


def test_thresholds_must_be_ordered():
    with pytest.raises(ValueError):
        DecisionPolicy(allow_threshold=0.7, restrict_threshold=0.3)
