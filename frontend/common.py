"""
ReturnGuard Agent - shared frontend code.

Everything the individual pages in frontend/views/ have in common lives
here: the API client, display helpers, the theme CSS, and the reusable
components (cap card, decision timeline, "why this decision?", the
decisions table). Pages import from this module; frontend/app.py is the
entry point that wires them into Streamlit's multipage navigation.

Streamlit puts the entry script's folder (frontend/) on sys.path, which
is what lets every page do a plain `from common import ...`.
"""

from __future__ import annotations

import os
import re
from datetime import datetime

import pandas as pd
import requests
import streamlit as st

API_BASE_URL = os.environ.get("RETURNGUARD_API_URL", "http://127.0.0.1:8000")
REQUEST_TIMEOUT = 10

# --------------------------------------------------------------------------
# Theme - "Private Bank". Colours, fonts and the ink sidebar come from
# .streamlit/config.toml; this CSS adds what the theme options can't
# express: card treatment for metrics and bordered containers, the brass
# active-nav marker, and the pill / badge / stage primitives.
# --------------------------------------------------------------------------

INK = "#121a24"
TEAL = "#0e5e6f"
BRASS = "#b08a3e"

THEME_CSS = """
<style>
  .block-container, [data-testid="stMainBlockContainer"] {
    padding-top: 4.75rem !important; max-width: 1280px;
  }

  /* Sidebar: brass marker on the active page, quieter section labels. */
  [data-testid="stSidebarNav"] a[aria-current="page"],
  [data-testid="stSidebarNavLink"][aria-current="page"] {
    box-shadow: inset 3px 0 0 #b08a3e;
  }
  [data-testid="stNavSectionHeader"] {
    font-size: 0.68rem; letter-spacing: 0.14em; text-transform: uppercase;
    color: #9aa5b3;
  }

  /* Metric tiles read as ledger cards: white, hairline border, serif figure. */
  [data-testid="stMetric"] {
    background: #ffffff; border: 1px solid #d8dde4; border-radius: 4px;
    padding: 14px 16px;
  }
  [data-testid="stMetricLabel"] p {
    font-size: 0.72rem; letter-spacing: 0.08em; text-transform: uppercase;
    color: #5d6774;
  }
  [data-testid="stMetricValue"] {
    font-family: "Source Serif 4", Georgia, serif; font-weight: 600;
    font-variant-numeric: tabular-nums;
  }

  /* Bordered containers created with key="card-..." get the same white
     card treatment (Streamlit adds an "st-key-<key>" class to them). */
  [class*="st-key-card-"] { background: #ffffff; }

  .rg-pill {
    display:inline-block; padding:2px 9px; border-radius:3px;
    font-size:0.74rem; font-weight:600; letter-spacing:0.03em;
    border:1px solid transparent;
  }
  .rg-pill.ok    { background:#e3f0e7; color:#2b7a4b; border-color:#c5e0cf; }
  .rg-pill.warn  { background:#f6ecd6; color:#9a6b12; border-color:#e8d3a6; }
  .rg-pill.bad   { background:#f6e2e3; color:#a8323a; border-color:#e7c1c4; }
  .rg-pill.muted { background:#eef1f4; color:#5d6774; border-color:#d8dde4; }
  .rg-badge {
    display:inline-block; padding:3px 11px; border-radius:3px;
    font-weight:700; font-size:0.76rem; letter-spacing:0.06em;
  }
  .rg-badge.low    { background:#e3f0e7; color:#2b7a4b; }
  .rg-badge.medium { background:#f6ecd6; color:#9a6b12; }
  .rg-badge.high   { background:#f6e2e3; color:#a8323a; }
  .rg-kicker { font-size:0.7rem; letter-spacing:0.14em; color:#5d6774;
               text-transform:uppercase; margin-bottom:6px; font-weight:600; }
  .rg-eyebrow { font-size:0.72rem; letter-spacing:0.16em; color:#b08a3e;
                text-transform:uppercase; font-weight:700;
                line-height:1.8; padding-top:2px; }
  .rg-big { font-family:"Source Serif 4", Georgia, serif; font-size:2.3rem;
            font-weight:600; line-height:1.1; color:#121a24;
            font-variant-numeric: tabular-nums; }
  .rg-sub { color:#5d6774; font-size:0.85rem; }
  .rg-stage { padding:9px 0; border-bottom:1px solid #e9ecf0; }
  .rg-stage:last-child { border-bottom:none; }
  .rg-side-status { font-size:0.8rem; color:#9aa5b3; line-height:1.9; }
  .rg-side-status b { color:#f1f3f6; font-weight:600; }
  .rg-dot { display:inline-block; width:7px; height:7px; border-radius:50%;
            margin-right:6px; vertical-align:1px; }
</style>
"""


def apply_theme():
    st.markdown(THEME_CSS, unsafe_allow_html=True)


def page_header(eyebrow: str, title: str, caption: str):
    """Consistent page heading: brass eyebrow, serif title, one-line caption."""
    st.markdown(f'<div class="rg-eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
    st.title(title)
    st.caption(caption)


# --------------------------------------------------------------------------
# API client - every backend call goes through here so error / loading
# handling is consistent and the UI never silently fails.
# --------------------------------------------------------------------------


class ApiError(Exception):
    pass


def _get(path: str, **params):
    r = requests.get(f"{API_BASE_URL}{path}", params=params or None, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.json()


def _post(path: str, payload: dict):
    r = requests.post(f"{API_BASE_URL}{path}", json=payload, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=5, show_spinner=False)
def api_health():
    try:
        return _get("/health"), None
    except requests.exceptions.RequestException as e:
        return None, str(e)


@st.cache_data(ttl=5, show_spinner=False)
def api_cap():
    try:
        return _get("/cap"), None
    except requests.exceptions.RequestException as e:
        return None, str(e)


@st.cache_data(ttl=5, show_spinner=False)
def api_decisions(limit: int = 100):
    try:
        return _get("/decisions", limit=limit).get("decisions", []), None
    except requests.exceptions.RequestException as e:
        return None, str(e)


@st.cache_data(ttl=5, show_spinner=False)
def api_decision_detail(order_id: str):
    try:
        return _get(f"/decisions/{order_id}"), None
    except requests.exceptions.RequestException as e:
        return None, str(e)


@st.cache_data(ttl=5, show_spinner=False)
def api_review_queue(limit: int = 100):
    try:
        return _get("/review-queue", limit=limit).get("queue", []), None
    except requests.exceptions.RequestException as e:
        return None, str(e)


@st.cache_data(ttl=5, show_spinner=False)
def api_audit_log(limit: int = 500):
    try:
        data = _get("/audit-log", limit=limit)
        return data.get("events", []), data.get("append_only", False), None
    except requests.exceptions.RequestException as e:
        return None, False, str(e)


@st.cache_data(ttl=5, show_spinner=False)
def api_metrics():
    try:
        return _get("/metrics"), None
    except requests.exceptions.RequestException as e:
        return None, str(e)


@st.cache_data(ttl=30, show_spinner=False)
def api_model_metrics():
    try:
        return _get("/model-metrics"), None
    except requests.exceptions.RequestException as e:
        return None, str(e)


def refresh_all():
    """Drop every cached API response - call after any write (score /
    override) so the queues, cap and metrics reflect it immediately."""
    api_health.clear()
    api_cap.clear()
    api_decisions.clear()
    api_decision_detail.clear()
    api_review_queue.clear()
    api_audit_log.clear()
    api_metrics.clear()
    api_model_metrics.clear()


# --------------------------------------------------------------------------
# Presentation helpers
# --------------------------------------------------------------------------

ACTION_LABEL = {
    "allow": "Allow",
    "restrict_cod": "Restrict COD",
    "flag_for_review": "Flag for Review",
}
ACTION_DISPLAY = {
    "allow": "ALLOW",
    "restrict_cod": "RESTRICT COD",
    "flag_for_review": "FLAG FOR REVIEW",
}
STATUS_LABEL = {
    "executed": "Executed",
    "pending_review": "Pending Review",
    "blocked_by_cap": "Blocked by Hourly Cap",
}
STATUS_PILL = {
    "executed": "ok",
    "pending_review": "warn",
    "blocked_by_cap": "bad",
}
RISK_LABEL = {"low": "LOW RISK", "medium": "MEDIUM RISK", "high": "HIGH RISK"}


def pill(text: str, kind: str = "muted") -> str:
    return f'<span class="rg-pill {kind}">{text}</span>'


def risk_badge(level: str) -> str:
    return f'<span class="rg-badge {level}">{RISK_LABEL.get(level, level.upper())}</span>'


def status_pill(status: str) -> str:
    return pill(STATUS_LABEL.get(status, status), STATUS_PILL.get(status, "muted"))


def action_pill(action: str) -> str:
    kind = {"allow": "ok", "flag_for_review": "warn", "restrict_cod": "bad"}.get(action, "muted")
    return pill(ACTION_DISPLAY.get(action, action.upper()), kind)


def fmt_ts(raw: str | None) -> str:
    if not raw:
        return "-"
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return dt.astimezone().strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return raw


def short_id(order_id: str | None) -> str:
    """Display label for an order ID. Order IDs come straight from the
    backend (the primary key of scored_orders) and are shown in full -
    truncating to a fixed-width suffix (e.g. splitting on '-' and taking
    the last 8 chars) can make two DIFFERENT real orders render with the
    identical label, which is exactly the kind of thing that undermines
    trust in an audit dashboard. If a display ever needs to be short,
    render only a handful of the longest, and never in a table where two
    different orders would otherwise appear side by side."""
    if not order_id:
        return "-"
    return f"#{order_id}"


def fmt_num(value, decimals: int = 2) -> str:
    """Consistent numeric display: round to `decimals` places instead of
    showing raw floating-point noise (e.g. 0.35000000000000003 -> 0.35).
    Trailing zeros are trimmed so 0.50 still reads as 0.5."""
    if value is None:
        return "-"
    try:
        rounded = round(float(value), decimals)
    except (TypeError, ValueError):
        return str(value)
    text = f"{rounded:.{decimals}f}".rstrip("0").rstrip(".")
    return text if text else "0"


def fmt_pct(value, decimals: int = 1) -> str:
    """A [0, 1] fraction as a percentage string, e.g. 0.489 -> '48.9%'."""
    if value is None:
        return "-"
    try:
        return f"{float(value) * 100:.{decimals}f}%"
    except (TypeError, ValueError):
        return str(value)


def fmt_ms(value) -> str:
    if value is None:
        return "-"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{value:.0f} ms" if value >= 1 else f"{value:.2f} ms"


def cap_state_name(cap: dict) -> str:
    if cap["cap_reached"]:
        return "CAP REACHED"
    if cap["remaining"] <= max(1, int(cap["limit"] * 0.1)):
        return "NEAR LIMIT"
    return "NORMAL"



# --------------------------------------------------------------------------
# Shared: sidebar status block (rendered once by app.py, on every page)
# --------------------------------------------------------------------------


def render_sidebar_status():
    health, _ = api_health()
    cap, _ = api_cap()

    if health and health.get("agent_active"):
        agent = ('<span class="rg-dot" style="background:#4fbf82"></span>'
                 "<b>Agent active</b>")
    elif health:
        agent = ('<span class="rg-dot" style="background:#d9b56a"></span>'
                 "<b>Agent reachable</b>")
    else:
        agent = ('<span class="rg-dot" style="background:#e0646d"></span>'
                 "<b>Backend offline</b>")
    lines = [agent]
    if health:
        lines.append(f"Model · {health.get('dataset', '?')}")
    if cap:
        lines.append(f"Actions this hour · <b>{cap['used']} / {cap['limit']}</b>")
    st.sidebar.markdown(
        f'<div class="rg-side-status">{"<br>".join(lines)}</div>',
        unsafe_allow_html=True,
    )
    label = "Refresh data" if health else "Retry connection"
    if st.sidebar.button(label, width="stretch", key="sidebar_refresh"):
        refresh_all()
        st.rerun()


# --------------------------------------------------------------------------
# Shared: compact connection banner (never a giant permanent error)
# --------------------------------------------------------------------------


def connection_banner(key: str) -> bool:
    """Returns True if the backend is reachable. Shows a compact bar +
    Retry when it isn't, instead of a wall-of-text error."""
    health, _ = api_health()
    if health:
        return True
    c1, c2 = st.columns([5, 1])
    with c1:
        st.warning(
            f"Backend unavailable at `{API_BASE_URL}`. "
            "Start it with `uvicorn backend.main:app --reload`.",
            icon="⚠️",
        )
    with c2:
        if st.button("Retry", key=f"retry_{key}", width="stretch"):
            refresh_all()
            st.rerun()
    return False



# ==========================================================================
# Reusable: hourly action budget card
# ==========================================================================


def render_cap_card(cap: dict | None, err: str | None):
    st.markdown('<div class="rg-kicker">Hourly Action Budget</div>', unsafe_allow_html=True)
    if not cap:
        st.info("Cap status unavailable." + (f" ({err})" if err else ""))
        return

    used, limit = cap["used"], cap["limit"]
    state = cap_state_name(cap)
    kind = {"NORMAL": "muted", "NEAR LIMIT": "warn", "CAP REACHED": "bad"}[state]
    st.markdown(
        f'<span class="rg-big">{used} / {limit}</span> &nbsp; {pill(state, kind)}',
        unsafe_allow_html=True,
    )
    st.progress(min(used / limit, 1.0) if limit else 0.0)
    mins = max(cap["seconds_until_reset"] // 60, 0)
    st.markdown(
        f'<span class="rg-sub">{cap["remaining"]} actions remaining · '
        f'resets in {mins} min</span>',
        unsafe_allow_html=True,
    )
    st.caption(
        f"If the hourly limit is reached, new {ACTION_LABEL.get(cap['capped_action'], cap['capped_action']).lower()} "
        f"actions are automatically downgraded to a review flag instead of "
        "being blocked outright."
    )
    if cap["cap_reached"]:
        st.error(
            "Hourly action cap reached. Further automated COD restrictions "
            "are downgraded to human review until the cap resets.",
            icon="🚫",
        )


# ==========================================================================
# Reusable: order decision timeline
# ==========================================================================


def render_timeline(detail: dict):
    order = detail["order"]
    decision = detail["decision"]
    events = detail.get("audit_events", [])
    cap = detail.get("cap", {})

    auto = next((e for e in events if e["event_type"] == "auto_action"), None)
    override = next((e for e in events if e["event_type"] == "human_override"), None)

    capped = decision["capped"]
    status = decision["status"]

    stages = [
        ("Order Received", "completed",
         f"{short_id(order['order_id'])} · {order.get('category', '-')} · "
         f"₹{order.get('order_amount', 0):,.0f} · {order.get('payment_method', '-')}"),
        ("Risk Scored", "completed",
         f"Return probability {decision['risk_score']:.0%}"),
        ("Risk Level", "completed", RISK_LABEL.get(decision["risk_level"], "-")),
        ("Policy Evaluated", "completed",
         f"Agent decision: {ACTION_DISPLAY.get(decision['action'], decision['action'])}"),
        ("Hourly Cap Checked", "blocked" if capped else "completed",
         "Cap reached - action downgraded to human review" if capped
         else f"{cap.get('used', '?')} / {cap.get('limit', '?')} used this hour"),
        ("Action Executed",
         "blocked" if status == "blocked_by_cap"
         else "pending" if status == "pending_review"
         else "completed",
         STATUS_LABEL.get(status, status)),
        ("Audit Log Written", "completed" if auto else "pending",
         f"auto_action · id {auto['log_id'][:8]}" if auto else "not yet written"),
    ]
    if override:
        stages.append((
            "Human Override", "completed",
            f"{ACTION_DISPLAY.get(override['action'], override['action'])} by "
            f"{override['actor']}"
            + (f" - {override['top_reasons'][0]}" if override.get("top_reasons") else ""),
        ))

    icon = {"completed": "✅", "pending": "🕓", "blocked": "🚫", "failed": "❌"}
    for name, state, detail_text in stages:
        st.markdown(
            f'<div class="rg-stage">{icon.get(state, "•")} <b>{name}</b> '
            f'&nbsp;{pill(state.upper(), {"completed":"ok","pending":"warn","blocked":"bad","failed":"bad"}.get(state,"muted"))}'
            f'<br><span class="rg-sub">{detail_text}</span></div>',
            unsafe_allow_html=True,
        )


# ==========================================================================
# Reusable: "Why this decision?" explainability
# ==========================================================================

# Raw snake_case feature names (as they come back from src/explain.py's
# SHAP reason strings, e.g. "size_flag (pushes risk up, impact 0.346)")
# mapped to a human-readable label. Categorical features are one-hot
# encoded server-side, so their raw names carry a "<column>_<value>"
# suffix (e.g. "delivery_pincode_risk_tier_high") - those are handled
# separately in humanize_feature_name() below rather than listed here
# one-by-one.
FEATURE_LABELS = {
    "order_amount": "Order amount",
    "item_count": "Item count",
    "discount_pct": "Discount %",
    "days_since_signup": "Days since customer signed up",
    "customer_prior_return_rate": "Customer's prior return rate",
    "review_score_avg": "Average review score",
    "is_new_customer": "New customer",
    "size_flag": "Size-related category (apparel/footwear)",
    "has_return_history": "Has any prior-return history on file",
    "category": "Category",
    "payment_method": "Payment method",
    "delivery_pincode_risk_tier": "Delivery pincode risk",
}

# The categorical columns whose one-hot-encoded feature names carry a
# "<column>_<value>" suffix (e.g. "category_electronics",
# "delivery_pincode_risk_tier_high") - checked longest-prefix-first so
# "delivery_pincode_risk_tier" doesn't get shadowed by a shorter match.
_CATEGORICAL_PREFIXES = sorted(
    ["category", "payment_method", "delivery_pincode_risk_tier"], key=len, reverse=True
)


def humanize_feature_name(raw_name: str) -> str:
    """'size_flag' -> 'Size-related category (apparel/footwear)',
    'category_electronics' -> 'Category: Electronics'. Falls back to a
    lightly cleaned-up version of the raw name (underscores -> spaces,
    capitalized) for anything not in the known feature set, rather than
    showing raw snake_case."""
    if raw_name in FEATURE_LABELS:
        return FEATURE_LABELS[raw_name]

    for prefix in _CATEGORICAL_PREFIXES:
        if raw_name.startswith(prefix + "_"):
            value = raw_name[len(prefix) + 1:].replace("_", " ")
            return f"{FEATURE_LABELS.get(prefix, prefix)}: {value.title()}"

    return raw_name.replace("_", " ").capitalize()


_REASON_PATTERN = re.compile(
    r"^(?P<feature>[A-Za-z0-9_]+)\s*\((?P<direction>pushes risk up|pushes risk down),\s*impact\s*(?P<impact>[0-9.]+)\)$"
)


def humanize_reason(reason: str) -> str:
    """Turn one raw SHAP reason string from the backend into a plain
    sentence with the feature name humanized, e.g.:
    'size_flag (pushes risk up, impact 0.346)' ->
    'Size-related category (apparel/footwear) - pushes risk up (impact 0.35)'.
    Falls back to the raw string unchanged if it doesn't match the
    expected "<feature> (<direction>, impact <n>)" shape, so an
    unexpected format is shown rather than silently dropped."""
    match = _REASON_PATTERN.match(reason.strip())
    if not match:
        return reason
    label = humanize_feature_name(match.group("feature"))
    direction = match.group("direction")
    impact = fmt_num(match.group("impact"), 2)
    return f"{label} — {direction} (impact {impact})"


def render_explainability(reasons: list[str]):
    st.markdown('<div class="rg-kicker">Why this decision?</div>', unsafe_allow_html=True)
    st.caption(
        "These are the factors that pushed this order's risk score up or "
        "down, and by how much."
    )
    if not reasons:
        st.info(
            "Risk-factor explanation unavailable for this order. "
            "The backend returns these reasons on `/score`; older audit rows "
            "may predate that.",
            icon="ℹ️",
        )
        return
    for r in reasons:
        dot = "🔴" if "pushes risk up" in r else "🟢" if "pushes risk down" in r else "🟠"
        st.markdown(f"{dot} {humanize_reason(r)}")



# ==========================================================================
# Decisions table (shared by Dashboard + Decision Queue)
# ==========================================================================


def decisions_dataframe(decisions: list[dict]) -> pd.DataFrame:
    rows = []
    for d in decisions:
        rows.append({
            "Order ID": short_id(d["order_id"]),
            "Return Risk": f"{d['risk_score']:.0%}",
            "Risk Level": d["risk_level"].upper(),
            "Agent Decision": ACTION_LABEL.get(d["action"], d["action"]),
            "Action": {
                "restrict_cod": "COD Restricted",
                "flag_for_review": "Human Review",
                "allow": "Order Allowed",
            }.get(d["action"], d["action"]),
            "Status": STATUS_LABEL.get(d["status"], d["status"]),
            "Timestamp": fmt_ts(d.get("scored_at")),
        })
    return pd.DataFrame(rows)

