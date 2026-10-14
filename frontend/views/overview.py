"""Overview - the landing page: headline numbers, decisions by hour, the
hourly action budget and the latest decisions. Read-only; scoring an
order happens on its own page."""

import pandas as pd
import streamlit as st

from common import (
    ACTION_LABEL,
    api_cap,
    api_decisions,
    api_metrics,
    connection_banner,
    fmt_pct,
    fmt_ts,
    page_header,
    render_cap_card,
    short_id,
)

# Fixed colour per action so the chart reads the same as the pills:
# allow = green, review = brass, COD restriction = red.
ACTION_COLORS = {
    "Allow": "#2b7a4b",
    "Flag for Review": "#b08a3e",
    "Restrict COD": "#a8323a",
}


def render_onboarding_banner():
    if st.session_state.get("onboarding_dismissed"):
        return
    c1, c2 = st.columns([8, 1], vertical_alignment="center")
    with c1:
        st.info(
            "**New here?** 1. Score an order → 2. See why it was flagged, in "
            "plain language → 3. Check the permanent audit trail.",
            icon="👋",
        )
    with c2:
        if st.button("Dismiss", key="dismiss_onboarding"):
            st.session_state["onboarding_dismissed"] = True
            st.rerun()


def decisions_by_hour(decisions: list[dict]) -> pd.DataFrame:
    """Count decisions per local hour, one column per action."""
    df = pd.DataFrame(
        {
            "scored_at": pd.to_datetime(
                [d.get("scored_at") for d in decisions], utc=True, errors="coerce"
            ),
            "action": [ACTION_LABEL.get(d["action"], d["action"]) for d in decisions],
        }
    ).dropna(subset=["scored_at"])
    if df.empty:
        return df
    df["hour"] = df["scored_at"].dt.tz_convert(None).dt.floor("h")
    table = df.pivot_table(index="hour", columns="action", aggfunc="size", fill_value=0)
    return table[[a for a in ACTION_COLORS if a in table.columns]]


page_header(
    "Return-risk agent",
    "Overview",
    "Autonomous return-risk decisions at checkout, with bounded actions and "
    "a permanent audit trail.",
)
render_onboarding_banner()

if not connection_banner("overview"):
    st.stop()

metrics, merr = api_metrics()
op = (metrics or {}).get("operational", {})
evaluated = op.get("orders_evaluated", 0)

k = st.columns(4)
k[0].metric("Orders scored", evaluated)
k[1].metric(
    "Flagged for review",
    op.get("flag_for_review", 0),
    fmt_pct(op["flag_for_review"] / evaluated) + " of orders" if evaluated else None,
    delta_color="off",
    delta_arrow="off",
)
k[2].metric(
    "COD restricted",
    op.get("restrict_cod", 0),
    fmt_pct(op["restrict_cod"] / evaluated) + " of orders" if evaluated else None,
    delta_color="off",
    delta_arrow="off",
)
k[3].metric("Human overrides", op.get("human_overrides", 0))
if metrics is None:
    st.caption(f"Could not load metrics. ({merr})")

decisions, derr = api_decisions(limit=200)

left, right = st.columns([2, 1])
with left.container(border=True, height="stretch", key="card-chart"):
    st.markdown('<div class="rg-kicker">Decisions per hour</div>', unsafe_allow_html=True)
    hourly = decisions_by_hour(decisions or [])
    if hourly.empty:
        st.info("No decisions yet. Score an order to start the chart.")
    else:
        st.bar_chart(
            hourly,
            color=[ACTION_COLORS[c] for c in hourly.columns],
            height=240,
            x_label="",
            y_label="Orders",
        )
with right.container(border=True, height="stretch", key="card-cap"):
    cap, cap_err = api_cap()
    render_cap_card(cap, cap_err)

st.markdown("")
head, link1, link2 = st.columns([4, 1, 1], vertical_alignment="bottom")
head.subheader("Latest decisions")
link1.page_link("views/score.py", label="Score an order", icon="➕")
link2.page_link("views/decisions.py", label="All decisions", icon="📋")

if decisions is None:
    st.warning("Could not load recent decisions." + (f" ({derr})" if derr else ""))
elif not decisions:
    st.info("No agent decisions yet. Score an order to populate this list.")
else:
    st.dataframe(
        pd.DataFrame(
            {
                "Order": short_id(d["order_id"]),
                "Category": (d.get("category") or "-").title(),
                "Amount (₹)": d.get("order_amount"),
                "Payment": d.get("payment_method") or "-",
                "Return risk": d["risk_score"],
                "Decision": ACTION_LABEL.get(d["action"], d["action"]),
                "Scored at": fmt_ts(d.get("scored_at")),
            }
            for d in decisions[:10]
        ),
        hide_index=True,
        width="stretch",
        column_config={
            "Amount (₹)": st.column_config.NumberColumn(format="%.0f"),
            "Return risk": st.column_config.ProgressColumn(
                min_value=0.0, max_value=1.0, format="percent"
            ),
        },
    )
