"""
ReturnGuard Agent - end-to-end pipeline smoke test.

TODO (Day 12, per razorpay_buildathon_plan.md Section 5):
  Once src/features.py, src/model.py, and src/explain.py are implemented
  (Days 5-7), add a smoke test here that:
    1. Loads a handful of rows from data/test.csv
    2. Runs them through the feature pipeline + model + policy end to end
    3. Asserts the output shape/types are correct (risk_score in [0, 1],
       action in policy.VALID_ACTIONS, reasons is a non-empty list)

  This is deliberately NOT a test of model accuracy (that's
  evaluation/evaluate.py's job) - it's a test that the pieces are wired
  together correctly.
"""

import pytest

from src.policy import VALID_ACTIONS, DecisionPolicy


def test_policy_output_is_always_a_valid_action():
    """Cheap sanity check that doesn't depend on the (not yet built)
    model - confirms the policy module alone is already wired correctly."""
    policy = DecisionPolicy()
    for score in [0.0, 0.1, 0.35, 0.5, 0.65, 0.8, 1.0]:
        for method in ["UPI", "card", "netbanking", "COD"]:
            decision = policy.decide(risk_score=score, payment_method=method)
            assert decision.action in VALID_ACTIONS


@pytest.mark.skip(reason="Enable once src/model.py and src/features.py are implemented (Day 5-7).")
def test_end_to_end_scoring_pipeline():
    pass
