"""Tests for src/explain.py. Skips if no trained model exists yet."""

import os
import pandas as pd
import pytest

MODEL_PATH = "models/xgboost_model.joblib"


@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="Run scripts/train.py first.")
def test_top_reasons_returns_three_readable_strings():
    from src.explain import top_reasons
    from src.model import ReturnRiskModel

    model = ReturnRiskModel.load(MODEL_PATH)
    val = pd.read_csv("data/val.csv", parse_dates=["order_date"])
    one_order = val.iloc[[0]]

    reasons = top_reasons(model, one_order, n=3)

    assert isinstance(reasons, list)
    assert len(reasons) == 3
    assert all(isinstance(r, str) for r in reasons)
    assert all(("up" in r or "down" in r) for r in reasons)