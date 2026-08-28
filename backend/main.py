"""
ReturnGuard Agent - FastAPI backend.

Wires together everything built on Days 5-8 into one running service:
  - src/model.py     -> scores an order (risk_score)
  - src/explain.py   -> explains the score (top_reasons)
  - src/policy.py    -> turns the score into a bounded action
  - backend/db.py    -> logs every decision, permanently

The model, the policy, and the database connection are all loaded ONCE
when this file is imported (i.e. once when the server starts) - not on
every request. That's what makes /score fast, and it's also what makes
the DecisionPolicy's hourly cap actually mean something: if we made a
fresh DecisionPolicy on every request, the cap counter would always
reset to zero and never actually cap anything.
"""

from __future__ import annotations

import os
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from backend import db
from src.explain import top_reasons
from src.model import ReturnRiskModel
from src.policy import VALID_ACTIONS, DecisionPolicy

# Which dataset's model to serve. Set RETURNGUARD_DATASET=real to serve
# the Olist-trained model; anything else (or unset) serves synthetic.
DATASET = os.environ.get("RETURNGUARD_DATASET", "synthetic").strip().lower()
if DATASET not in ("synthetic", "real"):
    raise RuntimeError(
        f"RETURNGUARD_DATASET must be 'synthetic' or 'real', got {DATASET!r}"
    )
MODEL_PATH = f"models/xgboost_model_{DATASET}.joblib"

app = FastAPI(title="ReturnGuard Agent")

model = ReturnRiskModel.load(MODEL_PATH)
policy = DecisionPolicy(allow_threshold=model.allow_threshold, restrict_threshold=model.restrict_threshold)
db.init_db()
conn = db.get_connection()


class OrderRequest(BaseModel):
    """Everything /score needs about one order. Field names match the
    columns src/features.py expects - see ALL_FEATURE_COLUMNS there."""

    order_id: str
    order_amount: float
    category: str
    item_count: int
    # discount_pct does not exist in the Olist real dataset - optional so
    # a real-schema request can omit it (it is imputed away server-side).
    discount_pct: Optional[float] = None
    payment_method: str
    is_new_customer: bool
    customer_prior_return_rate: Optional[float] = None
    days_since_signup: int
    delivery_pincode_risk_tier: str
    size_flag: bool
    review_score_avg: Optional[float] = None


class ScoreResponse(BaseModel):
    order_id: str
    risk_score: float
    action: str
    capped: bool
    reasons: list[str]


class OverrideRequest(BaseModel):
    new_action: str
    reviewer: str
    reason: Optional[str] = None


@app.get("/health")
def health():
    """Trivial liveness check - hit this first to confirm the server is up.

    Also reports which dataset's model this backend is serving, so the
    dashboard can show it."""
    return {"status": "ok", "dataset": DATASET, "model_path": MODEL_PATH}


@app.post("/score", response_model=ScoreResponse)
def score_order(order: OrderRequest):
    order_df = pd.DataFrame([order.model_dump()])

    risk_score = float(model.predict_proba(order_df)[0])
    reasons = top_reasons(model, order_df, n=3)
    decision = policy.decide(risk_score=risk_score, payment_method=order.payment_method)

    db.insert_scored_order(conn, order.model_dump(), risk_score=risk_score, action_taken=decision.action)
    db.insert_audit_log(
        conn,
        order_id=order.order_id,
        event_type="auto_action",
        action=decision.action,
        risk_score=risk_score,
        top_reasons=reasons,
        actor="agent",
    )

    return ScoreResponse(
        order_id=order.order_id,
        risk_score=risk_score,
        action=decision.action,
        capped=decision.capped,
        reasons=reasons,
    )


@app.post("/override/{order_id}")
def override_action(order_id: str, override: OverrideRequest):
    if override.new_action not in VALID_ACTIONS:
        raise HTTPException(status_code=400, detail=f"new_action must be one of {sorted(VALID_ACTIONS)}")

    original = db.get_latest_auto_action_log(conn, order_id)
    if original is None:
        raise HTTPException(status_code=404, detail=f"No auto_action found for order_id={order_id}")

    log_id = db.insert_audit_log(
        conn,
        order_id=order_id,
        event_type="human_override",
        action=override.new_action,
        risk_score=original["risk_score"],
        top_reasons=[override.reason] if override.reason else [],
        actor=override.reviewer,
        override_of_log_id=original["log_id"],
    )
    conn.execute("UPDATE scored_orders SET action_taken = ? WHERE order_id = ?", (override.new_action, order_id))
    conn.commit()

    return {
        "log_id": log_id,
        "order_id": order_id,
        "new_action": override.new_action,
        "overrides_log_id": original["log_id"],
    }


@app.get("/metrics")
def get_metrics():
    recent = db.get_recent_scored_orders(conn, limit=1000)
    action_counts: dict[str, int] = {}
    for row in recent:
        action_counts[row["action_taken"]] = action_counts.get(row["action_taken"], 0) + 1

    return {
        "total_scored_orders": len(recent),
        "action_distribution": action_counts,
        "override_rate_by_action": db.get_override_rate(conn),
    }