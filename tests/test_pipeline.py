"""
ReturnGuard Agent - end-to-end pipeline smoke test.

Loads a handful of rows from data/<dataset>/test.csv, runs them through
the feature pipeline + model + policy end to end, and asserts the output
shape/types are correct (risk_score in [0, 1], action in
policy.VALID_ACTIONS, reasons a 3-element list). Parametrized over both
dataset schemas so the synthetic AND real column layouts are covered.

This is deliberately NOT a test of model accuracy (that's
evaluation/evaluate.py's job) - it's a test that the pieces are wired
together correctly.
"""

import os

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


@pytest.mark.parametrize(
    "dataset, model_path",
    [
        ("synthetic", "models/xgboost_model_synthetic.joblib"),
        ("real", "models/xgboost_model_real.joblib"),
    ],
)
def test_end_to_end_scoring_pipeline(dataset, model_path):
    """Wiring test (not an accuracy test): a few rows of the matching
    test.csv go through feature pipeline + model + policy end to end, and
    the outputs have the right shape/types for BOTH dataset schemas."""
    if not os.path.exists(model_path):
        pytest.skip(f"Run `python -m scripts.train --dataset {dataset}` first.")

    from src.explain import top_reasons
    from src.features import load_split
    from src.model import ReturnRiskModel

    model = ReturnRiskModel.load(model_path)
    assert model.dataset == dataset

    df = load_split(f"data/{dataset}/test.csv").head(5).reset_index(drop=True)
    scores = model.predict_proba(df)
    assert len(scores) == len(df)
    assert all(0.0 <= float(s) <= 1.0 for s in scores)

    policy = DecisionPolicy(
        allow_threshold=model.allow_threshold, restrict_threshold=model.restrict_threshold
    )
    for s, pm in zip(scores, df["payment_method"]):
        assert policy.decide(risk_score=float(s), payment_method=pm).action in VALID_ACTIONS

    reasons = top_reasons(model, df.iloc[[0]], n=3)
    assert isinstance(reasons, list) and len(reasons) == 3
