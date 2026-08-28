"""
Tests for backend/db.py.

Uses a temporary throwaway database file per test run - never touches
your real returnguard.db.
"""

import os
import tempfile

import pytest

from backend import db


@pytest.fixture
def conn():
    """A fresh, temporary database for each test - so tests can't
    interfere with each other or with your real demo data."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db.init_db(path)
    connection = db.get_connection(path)
    yield connection
    connection.close()
    os.remove(path)


SAMPLE_ORDER = {
    "order_id": "order-001",
    "order_amount": 2500.0,
    "category": "apparel",
    "item_count": 2,
    "discount_pct": 45.0,
    "payment_method": "COD",
    "is_new_customer": True,
    "customer_prior_return_rate": None,
    "days_since_signup": 3,
    "delivery_pincode_risk_tier": "high",
    "size_flag": True,
    "review_score_avg": 3.2,
}


def test_insert_and_read_back_scored_order(conn):
    db.insert_scored_order(conn, SAMPLE_ORDER, risk_score=0.78, action_taken="restrict_cod")
    rows = db.get_recent_scored_orders(conn)
    assert len(rows) == 1
    assert rows[0]["order_id"] == "order-001"
    assert rows[0]["action_taken"] == "restrict_cod"


def test_override_rate_reflects_actual_overrides(conn):
    db.insert_scored_order(conn, SAMPLE_ORDER, risk_score=0.78, action_taken="restrict_cod")
    auto_log_id = db.insert_audit_log(
        conn, order_id="order-001", event_type="auto_action", action="restrict_cod",
        risk_score=0.78, top_reasons=["high discount"], actor="agent",
    )
    db.insert_audit_log(
        conn, order_id="order-001", event_type="human_override", action="allow",
        risk_score=0.78, top_reasons=["reviewer judgment"], actor="reviewer_priya",
        override_of_log_id=auto_log_id,
    )

    rates = db.get_override_rate(conn)
    assert rates["restrict_cod"]["auto_count"] == 1
    assert rates["restrict_cod"]["override_count"] == 1
    assert rates["restrict_cod"]["rate"] == 1.0


def test_action_with_no_override_shows_zero_rate(conn):
    db.insert_scored_order(conn, SAMPLE_ORDER, risk_score=0.10, action_taken="allow")
    db.insert_audit_log(
        conn, order_id="order-001", event_type="auto_action", action="allow",
        risk_score=0.10, top_reasons=["low discount"], actor="agent",
    )
    rates = db.get_override_rate(conn)
    assert rates["allow"]["override_count"] == 0
    assert rates["allow"]["rate"] == 0.0
