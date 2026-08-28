"""
Tests for backend/main.py (the FastAPI app).

Uses FastAPI's TestClient - this calls the app directly in-process, so
it exercises the exact same code a real HTTP request would, without
needing a live uvicorn server running. Requires a trained model on disk
(run `python -m scripts.train` first) since backend/main.py loads the
model at import time.
"""

import importlib
import os

import pytest

SYNTHETIC_MODEL_PATH = "models/xgboost_model_synthetic.joblib"
REAL_MODEL_PATH = "models/xgboost_model_real.joblib"

pytestmark = pytest.mark.skipif(
    not os.path.exists(SYNTHETIC_MODEL_PATH),
    reason="Run `python -m scripts.train --dataset synthetic` first to produce a model file.",
)

# Synthetic-schema order: discount_pct present, category "apparel", COD.
SAMPLE_ORDER = {
    "order_id": "test-order-1",
    "order_amount": 2500.0,
    "category": "apparel",
    "item_count": 2,
    "discount_pct": 48.0,
    "payment_method": "COD",
    "is_new_customer": True,
    "customer_prior_return_rate": None,
    "days_since_signup": 2,
    "delivery_pincode_risk_tier": "high",
    "size_flag": True,
    "review_score_avg": 3.0,
}

# Real (Olist) schema order: discount_pct omitted entirely, Olist-style
# category slug, payment_method "card", state-derived risk tier.
SAMPLE_ORDER_REAL = {
    "order_id": "test-order-real-1",
    "order_amount": 189.9,
    "category": "bed_bath_table",
    "item_count": 1,
    "payment_method": "card",
    "is_new_customer": True,
    "customer_prior_return_rate": None,
    "days_since_signup": 0,
    "delivery_pincode_risk_tier": "medium",
    "size_flag": False,
    "review_score_avg": None,
}


def _client_for(dataset: str):
    """Build a TestClient with the backend loaded for the given dataset.
    backend.main reads RETURNGUARD_DATASET at import time, so set it and
    reload the module."""
    os.environ["RETURNGUARD_DATASET"] = dataset
    import backend.main as backend_main

    backend_main = importlib.reload(backend_main)
    from fastapi.testclient import TestClient

    return TestClient(backend_main.app)


@pytest.fixture
def client():
    yield _client_for("synthetic")
    os.environ.pop("RETURNGUARD_DATASET", None)


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["dataset"] == "synthetic"


@pytest.mark.skipif(
    not os.path.exists(REAL_MODEL_PATH),
    reason="Run `python -m scripts.train --dataset real` first.",
)
def test_real_dataset_backend_scores_real_schema_order():
    """The real-schema order has no discount_pct at all - the backend
    must still score it (discount_pct is imputed away for real data)."""
    try:
        client = _client_for("real")
        health = client.get("/health").json()
        assert health["dataset"] == "real"

        response = client.post("/score", json=SAMPLE_ORDER_REAL)
        assert response.status_code == 200, response.text
        body = response.json()
        assert 0.0 <= body["risk_score"] <= 1.0
        assert body["action"] in {"allow", "restrict_cod", "flag_for_review"}
        assert len(body["reasons"]) == 3
    finally:
        os.environ.pop("RETURNGUARD_DATASET", None)


def test_score_returns_valid_action_and_reasons(client):
    response = client.post("/score", json=SAMPLE_ORDER)
    assert response.status_code == 200
    body = response.json()
    assert body["order_id"] == SAMPLE_ORDER["order_id"]
    assert 0.0 <= body["risk_score"] <= 1.0
    assert body["action"] in {"allow", "restrict_cod", "flag_for_review"}
    assert len(body["reasons"]) == 3


def test_override_updates_action_and_links_to_original(client):
    client.post("/score", json=SAMPLE_ORDER)
    response = client.post(
        f"/override/{SAMPLE_ORDER['order_id']}",
        json={"new_action": "allow", "reviewer": "test_reviewer", "reason": "test override"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["new_action"] == "allow"
    assert body["overrides_log_id"] is not None


def test_override_on_unknown_order_returns_404(client):
    response = client.post(
        "/override/does-not-exist",
        json={"new_action": "allow", "reviewer": "test_reviewer"},
    )
    assert response.status_code == 404


def test_override_with_invalid_action_returns_400(client):
    client.post("/score", json=SAMPLE_ORDER)
    response = client.post(
        f"/override/{SAMPLE_ORDER['order_id']}",
        json={"new_action": "not_a_real_action", "reviewer": "test_reviewer"},
    )
    assert response.status_code == 400


def test_metrics_reflects_scored_orders(client):
    client.post("/score", json=SAMPLE_ORDER)
    response = client.get("/metrics")
    assert response.status_code == 200
    body = response.json()
    assert body["total_scored_orders"] >= 1
    assert "action_distribution" in body
    assert "override_rate_by_action" in body