"""
ReturnGuard Agent - SQLite storage layer.

Two tables:

scored_orders
    One row per order we've scored. Overwritten if the same order is
    scored again (e.g. re-run in a demo).

audit_log
    One row per DECISION - an auto_action from the agent, or a
    human_override of one. This table is append-only: we only ever
    INSERT here, never UPDATE or DELETE a row. That's what makes it a
    real audit trail instead of just a log that can be quietly edited.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Optional

DB_PATH = "returnguard.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS scored_orders (
    order_id TEXT PRIMARY KEY,
    order_amount REAL,
    category TEXT,
    item_count INTEGER,
    discount_pct REAL,
    payment_method TEXT,
    is_new_customer INTEGER,
    customer_prior_return_rate REAL,
    days_since_signup INTEGER,
    delivery_pincode_risk_tier TEXT,
    size_flag INTEGER,
    review_score_avg REAL,
    risk_score REAL,
    action_taken TEXT,
    scored_at TEXT
);

CREATE TABLE IF NOT EXISTS audit_log (
    log_id TEXT PRIMARY KEY,
    order_id TEXT REFERENCES scored_orders(order_id),
    event_type TEXT CHECK (event_type IN ('auto_action', 'human_override')),
    action TEXT,
    risk_score REAL,
    top_reasons TEXT,
    actor TEXT,
    timestamp TEXT,
    override_of_log_id TEXT REFERENCES audit_log(log_id)
);
"""


def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: str = DB_PATH) -> None:
    conn = get_connection(db_path)
    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()


def insert_scored_order(conn: sqlite3.Connection, order: dict, risk_score: float, action_taken: str) -> None:
    """order is a plain dict with the raw order fields (e.g. one row of a
    DataFrame turned into a dict via .to_dict())."""
    conn.execute(
        """
        INSERT OR REPLACE INTO scored_orders (
            order_id, order_amount, category, item_count, discount_pct,
            payment_method, is_new_customer, customer_prior_return_rate,
            days_since_signup, delivery_pincode_risk_tier, size_flag,
            review_score_avg, risk_score, action_taken, scored_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            order["order_id"],
            order["order_amount"],
            order["category"],
            order["item_count"],
            order["discount_pct"],
            order["payment_method"],
            int(order["is_new_customer"]),
            order.get("customer_prior_return_rate"),
            order["days_since_signup"],
            order["delivery_pincode_risk_tier"],
            int(order["size_flag"]),
            order.get("review_score_avg"),
            risk_score,
            action_taken,
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()


def insert_audit_log(
    conn: sqlite3.Connection,
    order_id: str,
    event_type: str,
    action: str,
    risk_score: float,
    top_reasons: list[str],
    actor: str,
    override_of_log_id: Optional[str] = None,
) -> str:
    """Returns the new row's log_id, so an override row can later point
    back at the auto_action row it's overriding."""
    log_id = str(uuid.uuid4())
    conn.execute(
        """
        INSERT INTO audit_log (
            log_id, order_id, event_type, action, risk_score,
            top_reasons, actor, timestamp, override_of_log_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            log_id,
            order_id,
            event_type,
            action,
            risk_score,
            json.dumps(top_reasons),
            actor,
            datetime.now(timezone.utc).isoformat(),
            override_of_log_id,
        ),
    )
    conn.commit()
    return log_id


def get_recent_scored_orders(conn: sqlite3.Connection, limit: int = 50) -> list[dict]:
    """For the dashboard's flagged-orders queue."""
    cur = conn.execute("SELECT * FROM scored_orders ORDER BY scored_at DESC LIMIT ?", (limit,))
    columns = [d[0] for d in cur.description]
    return [dict(zip(columns, row)) for row in cur.fetchall()]


def get_latest_auto_action_log(conn: sqlite3.Connection, order_id: str) -> Optional[dict]:
    """Finds the auto_action row for an order, so /override can link a
    human_override row back to it."""
    cur = conn.execute(
        """
        SELECT * FROM audit_log
        WHERE order_id = ? AND event_type = 'auto_action'
        ORDER BY timestamp DESC LIMIT 1
        """,
        (order_id,),
    )
    row = cur.fetchone()
    if row is None:
        return None
    columns = [d[0] for d in cur.description]
    return dict(zip(columns, row))


def get_override_rate(conn: sqlite3.Connection) -> dict:
    """For each action type, what fraction of auto_actions were later
    overridden by a human? This is a LIVE, automatically-tracked version
    of the Day 11 trust metric - useful for the dashboard, though it's
    not a substitute for the manual review sample described in the plan
    (a real override only happens if someone actually clicks the button;
    an unreviewed bad action won't show up here)."""
    cur = conn.execute(
        """
        SELECT a.action, COUNT(DISTINCT a.log_id) AS auto_count,
               COUNT(DISTINCT o.log_id) AS override_count
        FROM audit_log a
        LEFT JOIN audit_log o
          ON o.override_of_log_id = a.log_id AND o.event_type = 'human_override'
        WHERE a.event_type = 'auto_action'
        GROUP BY a.action
        """
    )
    result = {}
    for action, auto_count, override_count in cur.fetchall():
        rate = override_count / auto_count if auto_count else 0.0
        result[action] = {"auto_count": auto_count, "override_count": override_count, "rate": rate}
    return result


if __name__ == "__main__":
    init_db()
    print(f"Initialized schema at {DB_PATH}")