"""Tests for src/explain.py. Skips per-dataset if that model doesn't exist yet."""

import os

import pandas as pd
import pytest

SYNTHETIC_MODEL_PATH = "models/xgboost_model_synthetic.joblib"
REAL_MODEL_PATH = "models/xgboost_model_real.joblib"


@pytest.mark.skipif(
    not os.path.exists(SYNTHETIC_MODEL_PATH),
    reason="Run `python -m scripts.train --dataset synthetic` first.",
)
def test_top_reasons_synthetic():
    from src.explain import top_reasons
    from src.model import ReturnRiskModel

    model = ReturnRiskModel.load(SYNTHETIC_MODEL_PATH)
    val = pd.read_csv("data/synthetic/val.csv", parse_dates=["order_date"])
    reasons = top_reasons(model, val.iloc[[0]], n=3)

    assert isinstance(reasons, list)
    assert len(reasons) == 3
    assert all(isinstance(r, str) for r in reasons)
    assert all(("up" in r or "down" in r) for r in reasons)


@pytest.mark.skipif(
    not os.path.exists(REAL_MODEL_PATH),
    reason="Run `python -m scripts.train --dataset real` first.",
)
def test_top_reasons_real():
    from src.explain import top_reasons
    from src.model import ReturnRiskModel

    model = ReturnRiskModel.load(REAL_MODEL_PATH)
    assert model.dataset == "real"
    val = pd.read_csv("data/real/val.csv", parse_dates=["order_date"])
    reasons = top_reasons(model, val.iloc[[0]], n=3)

    assert len(reasons) == 3
    assert all(("up" in r or "down" in r) for r in reasons)
