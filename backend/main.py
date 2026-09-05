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

import json
import os
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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

# The dashboard is served from a different origin in production (Streamlit
# Community Cloud vs. this API on Render), so the browser needs CORS
# headers to call it. Restrict to the configured frontend origin(s); allow
# "*" only as a local-dev fallback when nothing is set.
_allowed_origins = [
    o.strip()
    for o in os.environ.get("RETURNGUARD_ALLOWED_ORIGINS", "*").split(",")
    if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    # Added for the agent dashboard. Derived server-side from risk_score
    # and the same thresholds the policy uses, so the UI never has to
    # re-derive risk level or action status itself.
    risk_level: str
    status: str
    scored_at: str
    audit_log_id: str
    latency_ms: float


class OverrideRequest(BaseModel):
    new_action: str
    reviewer: str
    reason: Optional[str] = None


RISK_LEVEL_LOW = "low"
RISK_LEVEL_MEDIUM = "medium"
RISK_LEVEL_HIGH = "high"


def risk_level(score: float) -> str:
    """Bucket a risk_score into low/medium/high using the SAME thresholds
    the policy uses to decide - so the label the UI shows always lines up
    with the action the agent took."""
    if score < policy.allow_threshold:
        return RISK_LEVEL_LOW
    if score >= policy.restrict_threshold:
        return RISK_LEVEL_HIGH
    return RISK_LEVEL_MEDIUM


ACTION_STATUS_EXECUTED = "executed"
ACTION_STATUS_PENDING_REVIEW = "pending_review"
ACTION_STATUS_BLOCKED_BY_CAP = "blocked_by_cap"


def action_status(action: str, capped: bool) -> str:
    if capped:
        return ACTION_STATUS_BLOCKED_BY_CAP
    if action == "flag_for_review":
        return ACTION_STATUS_PENDING_REVIEW
    return ACTION_STATUS_EXECUTED


def _cap_state() -> dict:
    """Read the live hourly-cap counter off the shared DecisionPolicy.
    restrict_cod is the only capped action (see src/policy.py)."""
    now = datetime.now(timezone.utc)
    bucket = now.strftime("%Y-%m-%dT%H")
    used = int(policy._hourly_counts.get(bucket, 0))
    limit = int(policy.max_restrict_cod_per_hour)
    next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    return {
        "capped_action": "restrict_cod",
        "used": used,
        "limit": limit,
        "remaining": max(limit - used, 0),
        "cap_reached": used >= limit,
        "hour_bucket": bucket,
        "resets_at": next_hour.isoformat(),
        "seconds_until_reset": int((next_hour - now).total_seconds()),
        "downgrade_action": "flag_for_review",
    }


@app.get("/health")
def health():
    """Trivial liveness check - hit this first to confirm the server is up.

    Also reports which dataset's model this backend is serving, so the
    dashboard can show it."""
    return {
        "status": "ok",
        "dataset": DATASET,
        "model_path": MODEL_PATH,
        "agent_active": True,
        "allow_threshold": policy.allow_threshold,
        "restrict_threshold": policy.restrict_threshold,
        "cap": _cap_state(),
    }


@app.post("/score", response_model=ScoreResponse)
def score_order(order: OrderRequest):
    started = time.perf_counter()

    order_df = pd.DataFrame([order.model_dump()])

    risk_score = float(model.predict_proba(order_df)[0])
    reasons = top_reasons(model, order_df, n=3)
    decision = policy.decide(risk_score=risk_score, payment_method=order.payment_method)

    # Measured up to the point the decision is made - the two DB writes
    # below are logging the decision, not part of computing it, so they
    # aren't counted in "decision latency".
    latency_ms = (time.perf_counter() - started) * 1000

    db.insert_scored_order(
        conn,
        order.model_dump(),
        risk_score=risk_score,
        action_taken=decision.action,
        capped=decision.capped,
        top_reasons=reasons,
        latency_ms=latency_ms,
    )
    scored_at = datetime.now(timezone.utc).isoformat()
    audit_log_id = db.insert_audit_log(
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
        risk_level=risk_level(risk_score),
        status=action_status(decision.action, decision.capped),
        scored_at=scored_at,
        audit_log_id=audit_log_id,
        latency_ms=latency_ms,
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
        "operational": db.get_operational_counts(conn),
        "cap": _cap_state(),
    }


def _decision_view(order: dict) -> dict:
    """Shape one scored_orders row into the decision-queue record the
    dashboard renders (risk level + action status derived the same way
    /score derives them)."""
    score = order.get("risk_score") or 0.0
    capped = bool(order.get("capped"))
    return {
        "order_id": order["order_id"],
        "risk_score": score,
        "risk_level": risk_level(score),
        "action": order["action_taken"],
        "status": action_status(order["action_taken"], capped),
        "capped": capped,
        "scored_at": order.get("scored_at"),
        "top_reasons": order.get("top_reasons") or [],
        "payment_method": order.get("payment_method"),
        "order_amount": order.get("order_amount"),
        "category": order.get("category"),
    }


@app.get("/cap")
def get_cap():
    """Live hourly-action-budget state. The frontend must treat this as
    read-only truth - it can't grant itself more budget."""
    return _cap_state()


@app.get("/decisions")
def get_decisions(limit: int = 100):
    """Recent agent decisions, newest first - one row per order."""
    rows = db.get_recent_scored_orders(conn, limit=limit)
    for r in rows:
        try:
            r["top_reasons"] = json.loads(r["top_reasons"]) if r.get("top_reasons") else []
        except (TypeError, ValueError):
            r["top_reasons"] = []
    return {"decisions": [_decision_view(r) for r in rows]}


@app.get("/decisions/{order_id}")
def get_decision_detail(order_id: str):
    """Full record for one order: the scored order, its derived decision
    view, and the chronological audit-event timeline."""
    order = db.get_scored_order(conn, order_id)
    if order is None:
        raise HTTPException(status_code=404, detail=f"No scored order {order_id!r}")
    events = db.get_audit_events_for_order(conn, order_id)
    return {
        "order": order,
        "decision": _decision_view(order),
        "audit_events": events,
        "cap": _cap_state(),
    }


@app.get("/review-queue")
def get_review_queue(limit: int = 100):
    """Orders whose current action is flag_for_review - the human queue."""
    rows = db.get_review_queue(conn, limit=limit)
    return {"queue": [_decision_view(r) for r in rows]}


@app.get("/audit-log")
def get_audit_log_endpoint(limit: int = 500):
    """The append-only audit_log, newest first. Rows are never updated or
    deleted by the app - an override is a NEW row that points back at the
    auto_action it supersedes (override_of_log_id)."""
    return {"events": db.get_audit_log(conn, limit=limit), "append_only": True}


EVAL_REPORT_PATH = Path(f"evaluation/eval_report_{DATASET}.md")

# Matches lines like "- Precision: 0.489" in evaluation/eval_report_*.md
# (see evaluation/evaluate.py, which writes that file). Kept as simple,
# explicit key->regex pairs rather than a generic markdown parser, since
# the report format is small and hand-written.
_METRIC_PATTERNS = {
    "precision": r"-\s*Precision:\s*([0-9.]+)",
    "recall": r"-\s*Recall:\s*([0-9.]+)",
    "f1": r"-\s*F1:\s*([0-9.]+)",
    "roc_auc": r"-\s*ROC-AUC:\s*([0-9.]+)",
    "pr_auc": r"-\s*PR-AUC:\s*([0-9.]+)",
}


@app.get("/model-metrics")
def get_model_metrics():
    """Held-out test metrics for the model currently being served, read
    live from evaluation/eval_report_{dataset}.md (written by
    evaluation/evaluate.py - never re-derived or guessed here).

    Returns available=False (no fabricated numbers) if that file doesn't
    exist yet, e.g. `python -m evaluation.evaluate` hasn't been run."""
    if not EVAL_REPORT_PATH.exists():
        return {
            "available": False,
            "dataset": DATASET,
            "source_file": str(EVAL_REPORT_PATH),
            "detail": (
                f"{EVAL_REPORT_PATH} not found. Run "
                f"`python -m evaluation.evaluate --dataset {DATASET}` to generate it."
            ),
        }

    text = EVAL_REPORT_PATH.read_text(encoding="utf-8")
    metrics: dict[str, Optional[float]] = {}
    for key, pattern in _METRIC_PATTERNS.items():
        match = re.search(pattern, text)
        metrics[key] = float(match.group(1)) if match else None

    return {
        "available": any(v is not None for v in metrics.values()),
        "dataset": DATASET,
        "source_file": str(EVAL_REPORT_PATH),
        **metrics,
    }