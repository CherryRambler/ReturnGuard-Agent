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
    # Agent-operational block, added for the dashboard.
    assert body["operational"]["orders_evaluated"] >= 1
    assert "cap" in body


def test_score_response_carries_agent_fields(client):
    body = client.post("/score", json=SAMPLE_ORDER).json()
    assert body["risk_level"] in {"low", "medium", "high"}
    assert body["status"] in {"executed", "pending_review", "blocked_by_cap"}
    assert body["scored_at"]
    assert body["audit_log_id"]


def test_cap_endpoint_reports_budget(client):
    body = client.get("/cap").json()
    assert body["capped_action"] == "restrict_cod"
    assert body["limit"] >= 1
    assert body["used"] <= body["limit"] or body["cap_reached"]
    assert body["downgrade_action"] == "flag_for_review"


def test_decisions_and_detail_endpoints(client):
    client.post("/score", json=SAMPLE_ORDER)
    decisions = client.get("/decisions").json()["decisions"]
    assert any(d["order_id"] == SAMPLE_ORDER["order_id"] for d in decisions)

    detail = client.get(f"/decisions/{SAMPLE_ORDER['order_id']}").json()
    assert detail["order"]["order_id"] == SAMPLE_ORDER["order_id"]
    assert detail["decision"]["risk_level"] in {"low", "medium", "high"}
    assert any(e["event_type"] == "auto_action" for e in detail["audit_events"])

    assert client.get("/decisions/no-such-order").status_code == 404


def test_review_queue_holds_flagged_orders_only(client):
    # Force a flag_for_review: non-COD, mid/high risk order.
    flagged = {**SAMPLE_ORDER, "order_id": "rq-1", "payment_method": "card"}
    body = client.post("/score", json=flagged).json()
    queue = client.get("/review-queue").json()["queue"]
    in_queue = any(q["order_id"] == "rq-1" for q in queue)
    # It's in the queue iff the agent actually flagged it.
    assert in_queue == (body["action"] == "flag_for_review")


def test_model_metrics_reads_eval_report(client):
    body = client.get("/model-metrics").json()
    assert body["dataset"] == "synthetic"
    # Repo ships evaluation/eval_report_synthetic.md, so this should be
    # real parsed numbers, not the "not found" fallback.
    if body["available"]:
        assert 0.0 <= body["precision"] <= 1.0
        assert 0.0 <= body["recall"] <= 1.0
        assert 0.0 <= body["f1"] <= 1.0
        assert 0.0 <= body["roc_auc"] <= 1.0


def test_operational_restrict_cod_count_matches_override_rate_auto_count(client):
    """Regression test for the two COD-restriction numbers disagreeing:
    both must be computed from the same scored_orders query, so this
    must hold for ANY state of the (possibly non-empty, shared) DB -
    not just right after one fresh score."""
    metrics = client.get("/metrics").json()
    op = metrics["operational"]
    rates = metrics["override_rate_by_action"]
    for action in ("allow", "restrict_cod", "flag_for_review"):
        rate_row = rates.get(action)
        auto_count = rate_row["auto_count"] if rate_row else 0
        assert op[action] == auto_count, (
            f"operational[{action}]={op[action]} != "
            f"override_rate_by_action[{action}].auto_count={auto_count}"
        )


def test_audit_log_endpoint_is_append_only_and_links_overrides(client):
    client.post("/score", json=SAMPLE_ORDER)
    client.post(
        f"/override/{SAMPLE_ORDER['order_id']}",
        json={"new_action": "allow", "reviewer": "ops", "reason": "verified"},
    )
    body = client.get("/audit-log").json()
    assert body["append_only"] is True
    events = body["events"]
    override = next(e for e in events if e["event_type"] == "human_override")
    assert override["override_of_log_id"] is not None
    auto = next(e for e in events if e["event_type"] == "auto_action")
    assert auto["was_overridden"] is True