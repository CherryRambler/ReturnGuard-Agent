"""Decisions - every order the agent has scored, filterable, with a
per-order detail panel (pipeline timeline + reasons)."""

import streamlit as st

from common import (
    ACTION_LABEL,
    action_pill,
    api_decision_detail,
    api_decisions,
    connection_banner,
    decisions_dataframe,
    page_header,
    render_explainability,
    render_timeline,
    risk_badge,
    short_id,
    status_pill,
)

page_header(
    "Every order the agent has scored",
    "Decisions",
    "Every order the agent has evaluated, newest first - including ones "
    "later resolved by a human. Select an order below to see its full "
    "pipeline, stage by stage."
)

if not connection_banner("decisions"):
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
