"""
ReturnGuard Agent - Streamlit operations dashboard.

Talks ONLY to the FastAPI backend (never touches the model or database
directly) - so what you see here is exactly what the API returns.

Run with (two terminals, backend must already be running):
    uvicorn backend.main:app --reload
    streamlit run frontend/app.py

This is an AGENT dashboard, not a scoring calculator. It visualises the
whole bounded-action pipeline:

    ORDER -> RETURN PROBABILITY -> RISK LEVEL -> AGENT DECISION
          -> HOURLY CAP CHECK -> ACTION EXECUTION -> AUDIT LOG

The permanent record of every decision lives in the backend's audit_log
table (backend/db.py). "Recent Agent Decisions" here is a live read of
that store via the API - not browser-only session state.
"""

from __future__ import annotations

import os
import re
import uuid
from datetime import datetime

import pandas as pd
import requests
import streamlit as st

API_BASE_URL = os.environ.get("RETURNGUARD_API_URL", "http://127.0.0.1:8000")
REQUEST_TIMEOUT = 10

st.set_page_config(
    page_title="ReturnGuard Agent",
    page_icon="🛡️",
    layout="wide",
)

# --------------------------------------------------------------------------
# Styling - extends the dark theme in .streamlit/config.toml with a few
# control-room primitives: status pills, risk badges, stage rows.
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
      .block-container { padding-top: 2rem; max-width: 1400px; }
      .rg-pill {
        display:inline-block; padding:2px 10px; border-radius:999px;
        font-size:0.75rem; font-weight:600; letter-spacing:0.03em;
        border:1px solid transparent;
      }
      .rg-pill.ok    { background:#10331f; color:#5ee08a; border-color:#1c5a37; }
      .rg-pill.warn  { background:#3a2f14; color:#f2c24b; border-color:#6b5420; }
      .rg-pill.bad   { background:#3a1a1c; color:#ff7b82; border-color:#6b2c30; }
      .rg-pill.muted { background:#1c2230; color:#9fb0c7; border-color:#2c3444; }
      .rg-badge {
        display:inline-block; padding:3px 12px; border-radius:6px;
        font-weight:700; font-size:0.8rem; letter-spacing:0.05em;
      }
      .rg-badge.low    { background:#10331f; color:#5ee08a; }
      .rg-badge.medium { background:#3a2f14; color:#f2c24b; }
      .rg-badge.high   { background:#3a1a1c; color:#ff7b82; }
      .rg-card {
        background:#161b22; border:1px solid #262d38; border-radius:12px;
        padding:18px 20px; height:100%;
      }
      .rg-kicker { font-size:0.72rem; letter-spacing:0.14em; color:#8b98a9;
                   text-transform:uppercase; margin-bottom:6px; }
      .rg-big { font-size:2.4rem; font-weight:800; line-height:1.1; }
      .rg-sub { color:#9fb0c7; font-size:0.85rem; }
      .rg-flow {
        display:flex; flex-wrap:wrap; gap:8px; align-items:center;
        color:#9fb0c7; font-size:0.78rem; letter-spacing:0.04em;
        margin:4px 0 18px 0;
      }
      .rg-flow b { color:#e6edf3; }
      .rg-flow .arrow { color:#4c8dff; }
      .rg-stage { padding:8px 0; border-bottom:1px solid #222a35; }
      .rg-stage:last-child { border-bottom:none; }
    </style>
    """,
    unsafe_allow_html=True,
)

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
# Sidebar - navigation + agent status
# --------------------------------------------------------------------------

health, health_err = api_health()
cap, cap_err = api_cap()

st.sidebar.markdown("## 🛡️ ReturnGuard")
NAV_PAGES = ["Dashboard", "Decision Queue", "Review Queue", "Audit Log", "Metrics", "Settings"]
# Decision Queue and Review Queue are easy to confuse at a glance -
# Review Queue is Decision Queue filtered to just the orders still
# waiting on a human (flag_for_review with no override yet). Spelling
# that out here means a first-time visitor doesn't have to click into
# both to find out.
NAV_CAPTIONS = [
    "Run + browse every decision",
    "All orders, every outcome",
    "Only orders awaiting a human",
    "Permanent decision + override log",
    "Agent + model performance",
    "Read-only config",
]
page = st.sidebar.radio(
    "Navigate",
    NAV_PAGES,
    captions=NAV_CAPTIONS,
    label_visibility="collapsed",
)

st.sidebar.divider()
st.sidebar.markdown("**Agent**")
if health and health.get("agent_active"):
    st.sidebar.markdown(pill("🟢 ACTIVE", "ok"), unsafe_allow_html=True)
elif health:
    st.sidebar.markdown(pill("🟡 REACHABLE", "warn"), unsafe_allow_html=True)
else:
    st.sidebar.markdown(pill("🔴 OFFLINE", "bad"), unsafe_allow_html=True)

st.sidebar.markdown("**Backend / API**")
if health:
    st.sidebar.markdown(
        pill(f"connected · {health.get('dataset', '?')}", "ok"), unsafe_allow_html=True
    )
    if st.sidebar.button("Refresh", width='stretch'):
        refresh_all()
        st.rerun()
else:
    st.sidebar.markdown(pill("unreachable", "bad"), unsafe_allow_html=True)
    st.sidebar.caption(f"{API_BASE_URL}")
    if st.sidebar.button("Retry connection", type="primary", width='stretch'):
        refresh_all()
        st.rerun()

st.sidebar.markdown("**Actions**")
if cap:
    st.sidebar.markdown(
        pill(f"{cap['used']} / {cap['limit']} this hour",
             "bad" if cap["cap_reached"] else "warn" if cap["remaining"] <= 3 else "muted"),
        unsafe_allow_html=True,
    )
else:
    st.sidebar.markdown(pill("cap unknown", "muted"), unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Shared: dismissible first-time onboarding banner (Dashboard only)
# --------------------------------------------------------------------------


def render_onboarding_banner():
    if st.session_state.get("onboarding_dismissed"):
        return
    c1, c2 = st.columns([8, 1])
    with c1:
        st.info(
            "**New here?** 1. Score an order below → 2. See why it was "
            "flagged, in plain language → 3. Check the permanent audit trail.",
            icon="👋",
        )
    with c2:
        st.write("")
        if st.button("Dismiss", key="dismiss_onboarding"):
            st.session_state["onboarding_dismissed"] = True
            st.rerun()


# --------------------------------------------------------------------------
# Shared: compact connection banner (never a giant permanent error)
# --------------------------------------------------------------------------


def connection_banner() -> bool:
    """Returns True if the backend is reachable. Shows a compact bar +
    Retry when it isn't, instead of a wall-of-text error."""
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
        if st.button("Retry", key=f"retry_{page}", width='stretch'):
            refresh_all()
            st.rerun()
    return False


# The static, non-interactive pipeline breadcrumb was previously shown on
# the Dashboard and Decision Queue pages, where it had no visual link to
# the content below it and read as decoration. The real pipeline is now
# only shown per-order, as the interactive checklist in render_timeline()
# (Dashboard's post-submit result, and Decision Queue's Order detail) -
# there each stage is a genuine status, not a static label.


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
# Order evaluation form  (shared by Dashboard)
# ==========================================================================

CATEGORIES = ["apparel", "electronics", "home", "beauty", "footwear"]
PAYMENT_METHODS = ["UPI", "card", "netbanking", "COD"]
RISK_TIERS = ["low", "medium", "high"]


def _new_order_id() -> str:
    return f"order-{uuid.uuid4().hex[:8]}"


def render_order_form():
    st.subheader("Run ReturnGuard Decision")
    st.caption(
        "Evaluate the order, determine return risk, and automatically execute "
        "a bounded action when permitted."
    )

    # The Order ID field's suggested default must be generated ONCE per
    # order, not re-rolled on every script rerun. Streamlit reruns this
    # whole script on any interaction, and an unkeyed widget's `value=`
    # is re-evaluated on each rerun - so a plain
    # `value=f"order-{uuid.uuid4()...}"` silently swaps in a NEW random ID
    # every time the user touches any other field, before they ever
    # submit. That's how two different real orders could end up sharing a
    # displayed ID, or a submitted ID could silently change out from
    # under the user.
    #
    # Binding the widget directly to a fixed session_state key doesn't
    # work either: Streamlit forbids writing to a widget-bound key once
    # that widget has rendered THIS run, and it also DROPS the key
    # entirely from session_state on any run where the widget doesn't
    # render (e.g. the user is on a different page) - so a fixed key can
    # vanish and crash on the next navigation back to this page.
    #
    # Instead: a generation counter picks the widget's `key` each render.
    # Bumping the counter after a submit gives the NEXT order a
    # brand-new widget identity (and therefore a freshly-generated
    # default), without ever mutating a widget's own key after the fact.
    if "order_id_gen" not in st.session_state:
        st.session_state["order_id_gen"] = 0
    gen = st.session_state["order_id_gen"]
    widget_key = f"order_id_input_{gen}"
    if widget_key not in st.session_state:
        st.session_state[widget_key] = _new_order_id()

    with st.form("decision_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            order_id = st.text_input("Order ID", key=widget_key)
            order_amount = st.number_input("Order amount (INR)", min_value=0.0, value=2500.0)
            category = st.selectbox("Category", CATEGORIES)
            item_count = st.number_input("Item count", min_value=1, max_value=10, value=2)
        with col2:
            discount_pct = st.slider("Discount %", 0.0, 60.0, 45.0)
            payment_method = st.selectbox("Payment method", PAYMENT_METHODS, index=3)
            is_new_customer = st.checkbox("New customer?", value=True)
            days_since_signup = st.number_input("Days since signup", min_value=0, value=2)
        with col3:
            delivery_pincode_risk_tier = st.selectbox("Delivery pincode risk", RISK_TIERS, index=2)
            size_flag = st.checkbox("Size-related category (apparel/footwear)?", value=True)
            review_score_avg = st.number_input(
                "Avg review score (0 = unknown)", min_value=0.0, max_value=5.0, value=3.0
            )
            customer_prior_return_rate = st.number_input(
                "Customer's prior return rate (-1 = new customer)",
                min_value=-1.0, max_value=1.0, value=-1.0,
            )
        submitted = st.form_submit_button(
            "Run ReturnGuard Decision", type="primary", width='stretch'
        )

    if not submitted:
        return

    order = {
        "order_id": order_id,
        "order_amount": order_amount,
        "category": category,
        "item_count": int(item_count),
        "discount_pct": discount_pct,
        "payment_method": payment_method,
        "is_new_customer": is_new_customer,
        "customer_prior_return_rate": None if customer_prior_return_rate < 0 else customer_prior_return_rate,
        "days_since_signup": int(days_since_signup),
        "delivery_pincode_risk_tier": delivery_pincode_risk_tier,
        "size_flag": size_flag,
        "review_score_avg": None if review_score_avg == 0 else review_score_avg,
    }
    with st.spinner("Running agent pipeline: score → decide → cap check → execute…"):
        try:
            result = _post("/score", order)
        except requests.exceptions.RequestException as e:
            st.error(f"Decision failed - backend error. ({e})", icon="❌")
            return
    refresh_all()
    st.session_state["last_result"] = result
    # Move to a new Order ID widget generation so the NEXT order gets a
    # fresh suggested ID - see the generation-counter note above
    # render_order_form()'s form for why this can't just reassign the
    # current widget's key directly.
    st.session_state["order_id_gen"] = st.session_state.get("order_id_gen", 0) + 1
    st.toast(f"Decision recorded: {ACTION_DISPLAY.get(result['action'], result['action'])}", icon="✅")
    st.rerun()


def render_last_result():
    result = st.session_state.get("last_result")
    if not result:
        return

    st.divider()
    # ---- Prediction -----------------------------------------------------
    rc1, rc2, rc3 = st.columns(3)
    with rc1:
        st.markdown('<div class="rg-card">', unsafe_allow_html=True)
        st.markdown('<div class="rg-kicker">Return Risk · Prediction</div>', unsafe_allow_html=True)
        st.markdown(f'<span class="rg-big">{result["risk_score"]:.0%}</span>', unsafe_allow_html=True)
        st.markdown(risk_badge(result["risk_level"]), unsafe_allow_html=True)
        st.markdown(
            f'<div class="rg-sub">Order {short_id(result["order_id"])} · '
            f'{fmt_ts(result.get("scored_at"))}</div>',
            unsafe_allow_html=True,
        )
        # Confidence: only shown if the backend supplies it. It does not
        # today, so this stays hidden rather than inventing a number.
        if result.get("confidence") is not None:
            st.markdown(
                f'<div class="rg-sub">Model confidence: {result["confidence"]:.0%}</div>',
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    with rc2:
        st.markdown('<div class="rg-card">', unsafe_allow_html=True)
        st.markdown('<div class="rg-kicker">Agent Decision</div>', unsafe_allow_html=True)
        st.markdown(
            f'<span class="rg-big" style="font-size:1.8rem">'
            f'{ACTION_DISPLAY.get(result["action"], result["action"])}</span>',
            unsafe_allow_html=True,
        )
        st.markdown(action_pill(result["action"]), unsafe_allow_html=True)
        if result["reasons"]:
            st.markdown(
                f'<div class="rg-sub">{humanize_reason(result["reasons"][0])}</div>',
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    with rc3:
        st.markdown('<div class="rg-card">', unsafe_allow_html=True)
        st.markdown('<div class="rg-kicker">Action</div>', unsafe_allow_html=True)
        action_name = {
            "restrict_cod": "COD RESTRICTION",
            "flag_for_review": "HUMAN REVIEW",
            "allow": "ORDER ALLOWED",
        }.get(result["action"], result["action"].upper())
        st.markdown(
            f'<span class="rg-big" style="font-size:1.6rem">{action_name}</span>',
            unsafe_allow_html=True,
        )
        st.markdown(status_pill(result["status"]), unsafe_allow_html=True)
        st.markdown(
            f'<div class="rg-sub">Order {short_id(result["order_id"])} · '
            f'audit id <code>{result.get("audit_log_id", "-")[:8]}</code></div>',
            unsafe_allow_html=True,
        )
        if result["status"] == "blocked_by_cap":
            st.markdown(
                f'<div class="rg-sub">Blocked by hourly cap - routed to human review.</div>',
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("")
    ec1, ec2 = st.columns([1, 1])
    with ec1:
        render_explainability(result.get("reasons", []))
    with ec2:
        detail, derr = api_decision_detail(result["order_id"])
        st.markdown('<div class="rg-kicker">Order Decision Timeline</div>', unsafe_allow_html=True)
        if detail:
            render_timeline(detail)
        else:
            st.info("Timeline unavailable." + (f" ({derr})" if derr else ""))


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


# ==========================================================================
# PAGE: Dashboard
# ==========================================================================

if page == "Dashboard":
    st.title("ReturnGuard Agent")
    st.caption(
        "Autonomous return-risk decisions with bounded actions and a permanent audit trail."
    )
    # Global status (agent active, backend connection, dataset, hourly
    # action count) lives ONLY in the sidebar - it's visible on every
    # page there, so repeating it in the main content area here was pure
    # duplication. This page shows page-specific content only.
    render_onboarding_banner()

    ok = connection_banner()

    left, right = st.columns([2, 1])
    with left:
        if ok:
            render_order_form()
        else:
            st.info("Connect to the backend to run decisions.")
    with right:
        render_cap_card(cap, cap_err)

    render_last_result()

    st.divider()
    st.subheader("Recent Agent Decisions")
    decisions, derr = api_decisions(limit=25)
    if decisions is None:
        st.warning("Could not load recent decisions." + (f" ({derr})" if derr else ""))
    elif not decisions:
        st.info("No agent decisions yet. Run a decision above to populate this queue.")
    else:
        st.dataframe(decisions_dataframe(decisions), width='stretch', hide_index=True)


# ==========================================================================
# PAGE: Decision Queue
# ==========================================================================

elif page == "Decision Queue":
    st.title("Decision Queue")
    st.caption(
        "Every order the agent has evaluated, newest first - including ones "
        "later resolved by a human. Select an order below to see its full "
        "pipeline, stage by stage."
    )

    if not connection_banner():
        st.stop()

    decisions, derr = api_decisions(limit=200)
    if decisions is None:
        st.error(f"Could not load decisions. ({derr})")
        st.stop()
    if not decisions:
        st.info("No agent decisions yet.")
        st.stop()

    fcol1, fcol2 = st.columns(2)
    with fcol1:
        lvl = st.multiselect("Risk level", ["low", "medium", "high"], default=[])
    with fcol2:
        act = st.multiselect(
            "Agent decision", list(ACTION_LABEL), default=[],
            format_func=lambda a: ACTION_LABEL[a],
        )
    filtered = [
        d for d in decisions
        if (not lvl or d["risk_level"] in lvl) and (not act or d["action"] in act)
    ]
    st.dataframe(decisions_dataframe(filtered), width='stretch', hide_index=True)

    st.divider()
    st.subheader("Order detail")
    options = {short_id(d["order_id"]): d["order_id"] for d in filtered}
    if options:
        picked = st.selectbox("Select an order", list(options))
        order_id = options[picked]
        detail, err = api_decision_detail(order_id)
        if detail:
            dv = detail["decision"]
            c1, c2, c3 = st.columns(3)
            c1.markdown(
                f'<div class="rg-kicker">Return Risk</div><span class="rg-big">'
                f'{dv["risk_score"]:.0%}</span><br>{risk_badge(dv["risk_level"])}',
                unsafe_allow_html=True,
            )
            c2.markdown(
                f'<div class="rg-kicker">Agent Decision</div>{action_pill(dv["action"])}',
                unsafe_allow_html=True,
            )
            c3.markdown(
                f'<div class="rg-kicker">Action Status</div>{status_pill(dv["status"])}',
                unsafe_allow_html=True,
            )
            st.markdown("")
            tl, ex = st.columns([1, 1])
            with tl:
                st.markdown('<div class="rg-kicker">Pipeline Timeline</div>', unsafe_allow_html=True)
                render_timeline(detail)
            with ex:
                render_explainability(detail["order"].get("top_reasons", []))
        else:
            st.info(f"No detail available. ({err})")


# ==========================================================================
# PAGE: Review Queue
# ==========================================================================

elif page == "Review Queue":
    st.title("Review Queue")
    st.caption("Orders the agent flagged for a human decision (FLAG FOR REVIEW).")

    if not connection_banner():
        st.stop()

    queue, qerr = api_review_queue(limit=100)
    if queue is None:
        st.error(f"Could not load the review queue. ({qerr})")
        st.stop()
    if not queue:
        st.info("No orders currently require human review.")
        st.stop()

    st.markdown(f"**{len(queue)}** order(s) awaiting review.")
    for item in queue:
        with st.expander(
            f'{short_id(item["order_id"])} · {item["risk_score"]:.0%} · '
            f'{item["risk_level"].upper()} · flagged {fmt_ts(item.get("scored_at"))}'
        ):
            detail, err = api_decision_detail(item["order_id"])
            reason = ""
            auto = None
            if detail:
                auto = next(
                    (e for e in detail["audit_events"] if e["event_type"] == "auto_action"), None
                )
            m1, m2, m3 = st.columns(3)
            m1.markdown(
                f'<div class="rg-kicker">Risk</div><span class="rg-big">'
                f'{item["risk_score"]:.0%}</span><br>{risk_badge(item["risk_level"])}',
                unsafe_allow_html=True,
            )
            m2.markdown(
                f'<div class="rg-kicker">Agent Decision</div>{action_pill("flag_for_review")}',
                unsafe_allow_html=True,
            )
            m3.markdown(
                f'<div class="rg-kicker">Created</div>'
                f'<span class="rg-sub">{fmt_ts(item.get("scored_at"))}</span>',
                unsafe_allow_html=True,
            )
            st.markdown("**Reason**")
            st.caption(
                "These are the factors that pushed this order's risk score up "
                "or down, and by how much."
            )
            for r in (item.get("top_reasons") or ["No stored explanation for this order."]):
                st.markdown(f"- {humanize_reason(r)}")

            st.markdown("**Resolve** — records a human_override in the audit log")
            rc1, rc2 = st.columns([1, 2])
            with rc1:
                # Only actions the backend accepts (VALID_ACTIONS).
                choice = st.selectbox(
                    "Human decision",
                    ["allow", "restrict_cod", "flag_for_review"],
                    format_func=lambda a: {
                        "allow": "Override → Allow",
                        "restrict_cod": "Override → Restrict COD",
                        "flag_for_review": "Approve Agent Action (keep review)",
                    }[a],
                    key=f"rq_choice_{item['order_id']}",
                )
                reviewer = st.text_input(
                    "Reviewer", value="ops_analyst", key=f"rq_rev_{item['order_id']}"
                )
            with rc2:
                # /override treats reason as optional; we require it for
                # any decision that changes the agent's action.
                reason_txt = st.text_area(
                    "Override reason", key=f"rq_reason_{item['order_id']}",
                    placeholder="e.g. Customer verified shipping details.",
                )
            if st.button("Submit decision", key=f"rq_submit_{item['order_id']}", type="primary"):
                changes_action = choice != "flag_for_review"
                if changes_action and not reason_txt.strip():
                    st.warning("A reason is required when overriding the agent's action.")
                else:
                    try:
                        with st.spinner("Recording human decision…"):
                            _post(
                                f"/override/{item['order_id']}",
                                {
                                    "new_action": choice,
                                    "reviewer": reviewer.strip() or "ops_analyst",
                                    "reason": reason_txt.strip() or None,
                                },
                            )
                        refresh_all()
                        st.toast("Human decision recorded in audit log.", icon="✅")
                        st.rerun()
                    except requests.exceptions.RequestException as e:
                        st.error(f"Override failed. ({e})", icon="❌")

            if auto and detail:
                override = next(
                    (e for e in detail["audit_events"] if e["event_type"] == "human_override"), None
                )
                if override:
                    st.markdown("---")
                    st.markdown(
                        f'{pill("AGENT DECISION", "muted")} {ACTION_DISPLAY[auto["action"]]} &nbsp; '
                        f'{pill("HUMAN DECISION", "warn")} {ACTION_DISPLAY[override["action"]]}',
                        unsafe_allow_html=True,
                    )
                    if override.get("top_reasons"):
                        st.caption(f"Override reason: {override['top_reasons'][0]}")
                    st.caption(f"Actor: {override['actor']} · {fmt_ts(override['timestamp'])}")


# ==========================================================================
# PAGE: Audit Log
# ==========================================================================

elif page == "Audit Log":
    st.title("Audit Log")
    st.caption(
        "Append-only record of every agent action and every human override. "
        "The app only ever INSERTs here - an override is a new row linked to "
        "the action it supersedes."
    )

    if not connection_banner():
        st.stop()

    events, append_only, aerr = api_audit_log(limit=1000)
    if events is None:
        st.error(f"Could not load the audit log. ({aerr})")
        st.stop()

    st.markdown(
        pill("append-only (INSERT-only table)" if append_only else "storage model unknown",
             "ok" if append_only else "muted"),
        unsafe_allow_html=True,
    )
    st.caption(
        "Append-only means the app never updates or deletes rows. It is not "
        "cryptographically immutable - it's a SQLite table."
    )

    if not events:
        st.info("No audit events available.")
        st.stop()

    f1, f2, f3, f4 = st.columns(4)
    with f1:
        order_filter = st.text_input("Order ID contains")
    with f2:
        actor_filter = st.multiselect(
            "Actor", sorted({e["actor"] for e in events})
        )
    with f3:
        action_filter = st.multiselect(
            "Action", sorted({e["action"] for e in events if e["action"]})
        )
    with f4:
        type_filter = st.multiselect(
            "Event type", ["auto_action", "human_override"]
        )

    rows = []
    for e in events:
        if order_filter and order_filter.lower() not in (e["order_id"] or "").lower():
            continue
        if actor_filter and e["actor"] not in actor_filter:
            continue
        if action_filter and e["action"] not in action_filter:
            continue
        if type_filter and e["event_type"] not in type_filter:
            continue
        rows.append({
            "Timestamp": fmt_ts(e["timestamp"]),
            "Order ID": short_id(e["order_id"]),
            "Risk": f"{e['risk_score']:.0%}" if e.get("risk_score") is not None else "-",
            "Event": e["event_type"],
            "Action": ACTION_LABEL.get(e["action"], e["action"]),
            "Actor": e["actor"],
            "Status": "Overridden" if e["was_overridden"] else (
                "Override" if e["event_type"] == "human_override" else "Recorded"
            ),
            "Audit ID": e["log_id"][:8],
        })

    st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)
    st.caption(f"{len(rows)} of {len(events)} events shown.")

    st.divider()
    st.subheader("Audit event detail")
    by_id = {e["log_id"][:8]: e for e in events}
    if by_id:
        picked = st.selectbox("Audit ID", list(by_id))
        e = by_id[picked]
        st.json({
            "log_id": e["log_id"],
            "order_id": e["order_id"],
            "event_type": e["event_type"],
            "action": e["action"],
            "risk_score": e["risk_score"],
            "actor": e["actor"],
            "timestamp": e["timestamp"],
            "override_of_log_id": e.get("override_of_log_id"),
            "top_reasons": e.get("top_reasons", []),
            "later_overridden": e["was_overridden"],
        })


# ==========================================================================
# PAGE: Metrics
# ==========================================================================

elif page == "Metrics":
    st.title("Metrics")
    st.caption("Agent operations first. Model performance is reported separately below.")

    if not connection_banner():
        st.stop()

    metrics, merr = api_metrics()
    if metrics is None:
        st.error(f"Could not load metrics. ({merr})")
        st.stop()

    op = metrics.get("operational", {})
    capm = metrics.get("cap", {})

    st.subheader("Operational Metrics")
    if not op or op.get("orders_evaluated", 0) == 0:
        st.info("No operational data available yet.")
    else:
        r1 = st.columns(4)
        r1[0].metric("Orders evaluated", op["orders_evaluated"])
        r1[1].metric("Actions executed", op["actions_executed"])
        r1[2].metric("Allow decisions", op["allow"])
        r1[3].metric("COD restrictions", op["restrict_cod"])
        r2 = st.columns(4)
        r2[0].metric("Review cases", op["flag_for_review"])
        r2[1].metric("Human overrides", op["human_overrides"])
        r2[2].metric("Cap downgrades", op["capped_downgrades"])
        r2[3].metric(
            "Cap utilisation",
            f"{(capm.get('used', 0) / capm['limit'] * 100):.0f}%" if capm.get("limit") else "-",
        )
        r3 = st.columns(4)
        r3[0].metric("Avg decision latency", fmt_ms(op.get("avg_decision_latency_ms")))

        if metrics.get("action_distribution"):
            st.markdown("**Action distribution**")
            st.bar_chart(pd.Series(metrics["action_distribution"]))

        st.markdown("**Human-override rate by action**")
        rate_rows = [
            {
                "action": ACTION_LABEL.get(a, a),
                "auto_actions": s["auto_count"],
                "overridden": s["override_count"],
                "override_rate": f"{s['rate']:.0%}",
            }
            for a, s in metrics.get("override_rate_by_action", {}).items()
        ]
        if rate_rows:
            st.table(pd.DataFrame(rate_rows))
        else:
            st.info("No auto-actions logged yet.")

    st.divider()
    st.subheader("Model Performance")
    st.caption(
        f"Held-out test metrics for the **{health.get('dataset', '?')}** model "
        "currently being served - not agent operations, kept separate above."
    )

    model_metrics, mmerr = api_model_metrics()
    if model_metrics is None:
        st.warning("Could not load model performance." + (f" ({mmerr})" if mmerr else ""))
    elif not model_metrics.get("available"):
        st.info(
            model_metrics.get("detail")
            or "No held-out evaluation report available for this dataset yet."
        )
    else:
        mc = st.columns(4)
        mc[0].metric("Precision", fmt_pct(model_metrics.get("precision")))
        mc[1].metric("Recall", fmt_pct(model_metrics.get("recall")))
        mc[2].metric("F1", fmt_num(model_metrics.get("f1"), 3))
        mc[3].metric("ROC-AUC", fmt_num(model_metrics.get("roc_auc"), 3))
        st.caption(f"Source: `{model_metrics.get('source_file')}`")

    st.markdown(
        f"- Allow threshold: `{fmt_num(health.get('allow_threshold'))}`  \n"
        f"- Restrict threshold: `{fmt_num(health.get('restrict_threshold'))}`"
    )


# ==========================================================================
# PAGE: Settings
# ==========================================================================

elif page == "Settings":
    st.title("Settings")
    st.caption("Runtime configuration. Policy thresholds and the cap are set on the backend.")

    st.subheader("Connection")
    st.code(f"RETURNGUARD_API_URL = {API_BASE_URL}", language="text")
    if st.button("Test connection"):
        refresh_all()
        h, e = api_health()
        if h:
            st.success(f"Reachable. Serving dataset: {h.get('dataset')}", icon="✅")
        else:
            st.error(f"Unreachable. ({e})", icon="❌")

    st.subheader("Agent policy (read-only)")
    if health:
        st.markdown(
            f"- Dataset: **{health.get('dataset', '-')}**  \n"
            f"- Allow threshold: `{fmt_num(health.get('allow_threshold'))}` "
            "(risk scores below this → allow)  \n"
            f"- Restrict threshold: `{fmt_num(health.get('restrict_threshold'))}` "
            "(risk scores at/above this + COD → restrict COD)"
        )
        with st.expander("View raw config"):
            st.json({
                "agent_active": health.get("agent_active"),
                "dataset": health.get("dataset"),
                "model_path": health.get("model_path"),
                "allow_threshold": health.get("allow_threshold"),
                "restrict_threshold": health.get("restrict_threshold"),
            })
    else:
        st.info("Backend unavailable - cannot read policy configuration.")

    st.subheader("Hourly action cap (read-only)")
    if cap:
        st.markdown(
            f"- Caps: **{cap['capped_action']}** at **{cap['limit']}** per hour  \n"
            f"- Used this hour: **{cap['used']} / {cap['limit']}**  \n"
            f"- When reached: downgrades to **{cap['downgrade_action']}** "
            "instead of being blocked outright"
        )
        st.caption(
            "The cap is enforced server-side in `src/policy.py`. This dashboard "
            "cannot raise or bypass it."
        )
        with st.expander("View raw JSON"):
            st.json(cap)
    else:
        st.info("Cap status unavailable.")

    st.subheader("Data integrity")
    st.markdown(
        "- The audit log is an **append-only** SQLite table (`audit_log`) - "
        "the app only INSERTs. It is not cryptographically immutable.\n"
        "- All figures on this dashboard come from the backend API. Nothing "
        "here is mocked or fabricated."
    )
