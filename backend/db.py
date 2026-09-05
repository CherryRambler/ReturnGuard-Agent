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
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Optional

# Runtime SQLite file. Override with RETURNGUARD_DB_PATH; defaults to
# returnguard.db in the current working directory (gitignored).
DB_PATH = os.environ.get("RETURNGUARD_DB_PATH", "returnguard.db")

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
    scored_at TEXT,
    -- Set when the agent's uncapped decision would have been restrict_cod
    -- but the hourly cap downgraded it to flag_for_review. Lets the
    -- dashboard show "BLOCKED BY HOURLY CAP" without re-deriving it.
    capped INTEGER DEFAULT 0,
    -- JSON list of the top SHAP reason strings at score time, so the
    -- decision queue / timeline views have the explanation without a
    -- second round-trip to audit_log.
    top_reasons TEXT,
    -- Wall-clock time /score took to run (model + policy + 2 writes),
    -- in milliseconds. Powers the "average decision latency" metric.
    latency_ms REAL
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
    _migrate(conn)
    conn.commit()
    conn.close()


def _migrate(conn: sqlite3.Connection) -> None:
    """Add columns introduced after a DB was first created. SQLite's
    CREATE TABLE IF NOT EXISTS won't add a column to a table that already
    exists, so bring older returnguard.db files up to date in place -
    append-only audit_log is untouched, this only ever ADDs nullable
    columns to scored_orders."""
    existing = {row[1] for row in conn.execute("PRAGMA table_info(scored_orders)")}
    if "capped" not in existing:
        conn.execute("ALTER TABLE scored_orders ADD COLUMN capped INTEGER DEFAULT 0")
    if "top_reasons" not in existing:
        conn.execute("ALTER TABLE scored_orders ADD COLUMN top_reasons TEXT")
    if "latency_ms" not in existing:
        conn.execute("ALTER TABLE scored_orders ADD COLUMN latency_ms REAL")


def insert_scored_order(
    conn: sqlite3.Connection,
    order: dict,
    risk_score: float,
    action_taken: str,
    capped: bool = False,
    top_reasons: Optional[list[str]] = None,
    latency_ms: Optional[float] = None,
) -> None:
    """order is a plain dict with the raw order fields (e.g. one row of a
    DataFrame turned into a dict via .to_dict())."""
    conn.execute(
        """
        INSERT OR REPLACE INTO scored_orders (
            order_id, order_amount, category, item_count, discount_pct,
            payment_method, is_new_customer, customer_prior_return_rate,
            days_since_signup, delivery_pincode_risk_tier, size_flag,
            review_score_avg, risk_score, action_taken, scored_at,
            capped, top_reasons, latency_ms
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            order["order_id"],
            order["order_amount"],
            order["category"],
            order["item_count"],
            order.get("discount_pct"),
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
            int(capped),
            json.dumps(top_reasons) if top_reasons is not None else None,
            latency_ms,
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
    """For each action type, what fraction of orders CURRENTLY showing
    that action were ever touched by a human override?

    Deliberately scoped to `scored_orders` - the same one-row-per-order,
    current-state table get_operational_counts() uses - so that e.g. the
    "COD restrictions" count on the Metrics page and the "restrict_cod"
    row here always agree. (An earlier version counted raw audit_log
    auto_action events instead, which could drift from the operational
    count whenever an order was rescored or overridden - the two numbers
    were answering different questions without saying so.)

    This is a LIVE, automatically-tracked version of the Day 11 trust
    metric - useful for the dashboard, though it's not a substitute for
    the manual review sample described in the plan (a real override only
    happens if someone actually clicks the button; an unreviewed bad
    action won't show up here)."""
    cur = conn.execute(
        """
        SELECT s.action_taken AS action,
               COUNT(*) AS auto_count,
               COUNT(*) FILTER (
                 WHERE EXISTS (
                   SELECT 1 FROM audit_log o
                   WHERE o.order_id = s.order_id AND o.event_type = 'human_override'
                 )
               ) AS override_count
        FROM scored_orders s
        GROUP BY s.action_taken
        """
    )
    result = {}
    for action, auto_count, override_count in cur.fetchall():
        rate = override_count / auto_count if auto_count else 0.0
        result[action] = {"auto_count": auto_count, "override_count": override_count, "rate": rate}
    return result


def _rows(cur) -> list[dict]:
    columns = [d[0] for d in cur.description]
    return [dict(zip(columns, row)) for row in cur.fetchall()]


def get_audit_log(conn: sqlite3.Connection, limit: int = 500) -> list[dict]:
    """Every audit_log row, newest first, with the reasons JSON already
    decoded and a flag for whether an auto_action was later overridden."""
    cur = conn.execute(
        """
        SELECT a.*,
               EXISTS (
                 SELECT 1 FROM audit_log o
                 WHERE o.override_of_log_id = a.log_id
               ) AS was_overridden
        FROM audit_log a
        ORDER BY a.timestamp DESC
        LIMIT ?
        """,
        (limit,),
    )
    out = []
    for row in _rows(cur):
        try:
            row["top_reasons"] = json.loads(row["top_reasons"]) if row["top_reasons"] else []
        except (TypeError, json.JSONDecodeError):
            row["top_reasons"] = []
        row["was_overridden"] = bool(row["was_overridden"])
        out.append(row)
    return out


def get_audit_events_for_order(conn: sqlite3.Connection, order_id: str) -> list[dict]:
    """All audit_log rows for one order, OLDEST first (chronological) -
    for the per-order timeline / detail view."""
    cur = conn.execute(
        "SELECT * FROM audit_log WHERE order_id = ? ORDER BY timestamp ASC",
        (order_id,),
    )
    out = []
    for row in _rows(cur):
        try:
            row["top_reasons"] = json.loads(row["top_reasons"]) if row["top_reasons"] else []
        except (TypeError, json.JSONDecodeError):
            row["top_reasons"] = []
        out.append(row)
    return out


def get_scored_order(conn: sqlite3.Connection, order_id: str) -> Optional[dict]:
    cur = conn.execute("SELECT * FROM scored_orders WHERE order_id = ?", (order_id,))
    rows = _rows(cur)
    if not rows:
        return None
    row = rows[0]
    try:
        row["top_reasons"] = json.loads(row["top_reasons"]) if row.get("top_reasons") else []
    except (TypeError, json.JSONDecodeError):
        row["top_reasons"] = []
    return row


def get_review_queue(conn: sqlite3.Connection, limit: int = 200) -> list[dict]:
    """Scored orders whose CURRENT action is flag_for_review - i.e. the
    agent asked for a human, and no override has resolved it yet."""
    cur = conn.execute(
        """
        SELECT * FROM scored_orders
        WHERE action_taken = 'flag_for_review'
        ORDER BY scored_at DESC
        LIMIT ?
        """,
        (limit,),
    )
    out = []
    for row in _rows(cur):
        try:
            row["top_reasons"] = json.loads(row["top_reasons"]) if row.get("top_reasons") else []
        except (TypeError, json.JSONDecodeError):
            row["top_reasons"] = []
        out.append(row)
    return out


def get_operational_counts(conn: sqlite3.Connection) -> dict:
    """Agent-operational tallies (distinct from ML model metrics).

    allow / restrict_cod / flag_for_review counts here are the SAME
    query basis as get_override_rate()'s auto_count per action (both
    read current-state scored_orders, grouped by action_taken), so the
    two are guaranteed to agree wherever they're shown together."""
    scored = _rows(conn.execute("SELECT action_taken, capped, latency_ms FROM scored_orders"))
    counts = {"allow": 0, "restrict_cod": 0, "flag_for_review": 0}
    capped_count = 0
    latencies = []
    for row in scored:
        counts[row["action_taken"]] = counts.get(row["action_taken"], 0) + 1
        if row["capped"]:
            capped_count += 1
        if row["latency_ms"] is not None:
            latencies.append(row["latency_ms"])

    auto_actions = conn.execute(
        "SELECT COUNT(*) FROM audit_log WHERE event_type = 'auto_action'"
    ).fetchone()[0]
    overrides = conn.execute(
        "SELECT COUNT(*) FROM audit_log WHERE event_type = 'human_override'"
    ).fetchone()[0]

    # "Executed" = an action that actually took effect on the order and
    # did not need a human: allow and restrict_cod auto-actions.
    executed = counts["allow"] + counts["restrict_cod"]

    return {
        "orders_evaluated": len(scored),
        "actions_executed": executed,
        "allow": counts["allow"],
        "restrict_cod": counts["restrict_cod"],
        "flag_for_review": counts["flag_for_review"],
        "capped_downgrades": capped_count,
        "auto_actions_logged": auto_actions,
        "human_overrides": overrides,
        "avg_decision_latency_ms": (sum(latencies) / len(latencies)) if latencies else None,
    }


if __name__ == "__main__":
    init_db()
    print(f"Initialized schema at {DB_PATH}")