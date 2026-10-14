"""Settings - connection check and the read-only policy and cap config."""

import streamlit as st

from common import (
    API_BASE_URL,
    api_cap,
    api_health,
    fmt_num,
    page_header,
    refresh_all,
)

health, _ = api_health()
cap, _ = api_cap()

page_header(
    "Configuration",
    "Settings",
    "Runtime configuration. Policy thresholds and the cap are set on the backend.")

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
