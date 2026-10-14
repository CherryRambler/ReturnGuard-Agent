"""Analytics - agent operations metrics, then held-out model performance."""

import pandas as pd
import streamlit as st

from common import (
    ACTION_LABEL,
    api_health,
    api_metrics,
    api_model_metrics,
    connection_banner,
    fmt_ms,
    fmt_num,
    fmt_pct,
    page_header,
)

health, _ = api_health()

page_header(
    "Performance",
    "Analytics",
    "Agent operations first. Model performance is reported separately below.")

if not connection_banner("analytics"):
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
