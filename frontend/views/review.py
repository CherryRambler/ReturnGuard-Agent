"""Review Queue - orders the agent flagged for a person, with the form
that approves the agent's call or records an override."""

import requests
import streamlit as st

from common import (
    ACTION_DISPLAY,
    _post,
    action_pill,
    api_decision_detail,
    api_review_queue,
    connection_banner,
    fmt_ts,
    humanize_reason,
    page_header,
    pill,
    refresh_all,
    risk_badge,
    short_id,
)

page_header(
    "Needs a person",
    "Review Queue",
    "Orders the agent flagged for a human decision (FLAG FOR REVIEW).")

if not connection_banner("review"):
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
