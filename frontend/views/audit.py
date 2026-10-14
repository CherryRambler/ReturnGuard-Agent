"""Audit Log - the append-only record of every agent action and every
human override, with filters and a raw event view."""

import pandas as pd
import streamlit as st

from common import (
    ACTION_LABEL,
    api_audit_log,
    connection_banner,
    fmt_ts,
    page_header,
    pill,
    short_id,
)

page_header(
    "Permanent record",
    "Audit Log",
    "Append-only record of every agent action and every human override. "
    "The app only ever INSERTs here - an override is a new row linked to "
    "the action it supersedes."
)

if not connection_banner("audit"):
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
