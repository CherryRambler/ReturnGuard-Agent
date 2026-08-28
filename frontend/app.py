"""
ReturnGuard Agent - Streamlit dashboard.

Talks ONLY to the FastAPI backend (never touches the model or database
directly) - so what you see here is exactly what the API returns.

Run with (two terminals, backend must already be running):
    uvicorn backend.main:app --reload
    streamlit run frontend/app.py

Note: the "recent scored orders" list below lives in THIS BROWSER
SESSION ONLY (Streamlit's session_state) - it resets if you refresh the
page. The permanent record of every decision lives in the audit_log
table via backend/db.py, not here - this page is just a live view of
what you've scored during this session, for demo purposes.
"""

from __future__ import annotations

import os
import uuid

import pandas as pd
import requests
import streamlit as st

API_BASE_URL = os.environ.get("RETURNGUARD_API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="ReturnGuard Agent", layout="wide")
st.title("ReturnGuard Agent")
st.caption("Return-risk scoring with a bounded, auditable decision layer.")

if "scored_orders" not in st.session_state:
    st.session_state.scored_orders = []  # list of dicts, most recent first

def backend_health():
    try:
        r = requests.get(f"{API_BASE_URL}/health", timeout=5)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException:
        return None


_health = backend_health()
if _health and _health.get("dataset"):
    st.sidebar.success(f"Backend dataset: **{_health['dataset']}**")
elif _health:
    st.sidebar.info("Backend connected (dataset unknown - older backend)")
else:
    st.sidebar.error(f"Backend unreachable at {API_BASE_URL}")

page = st.sidebar.radio("View", ["Score & Queue", "Metrics"])

CATEGORIES = ["apparel", "electronics", "home", "beauty", "footwear"]
PAYMENT_METHODS = ["UPI", "card", "netbanking", "COD"]
RISK_TIERS = ["low", "medium", "high"]

ACTION_LABELS = {"allow": "✅ ALLOW", "flag_for_review": "🟡 REVIEW", "restrict_cod": "🔴 RESTRICTED"}

def score_order(order: dict):
    try:
        response = requests.post(f"{API_BASE_URL}/score", json=order, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Could not reach the backend at {API_BASE_URL}. Is uvicorn running? ({e})")
        return None


def override_order(order_id: str, new_action: str, reviewer: str, reason: str):
    try:
        response = requests.post(
            f"{API_BASE_URL}/override/{order_id}",
            json={"new_action": new_action, "reviewer": reviewer, "reason": reason},
            timeout=10,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Override failed: {e}")
        return None


if page == "Score & Queue":
    st.subheader("Score a new order")

    with st.form("score_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            order_id = st.text_input("Order ID", value=f"order-{uuid.uuid4().hex[:8]}")
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
            review_score_avg = st.number_input("Avg review score (0 = unknown)", min_value=0.0, max_value=5.0, value=3.0)
            customer_prior_return_rate = st.number_input(
                "Customer's prior return rate (-1 = new customer)", min_value=-1.0, max_value=1.0, value=-1.0
            )

        submitted = st.form_submit_button("Score this order")

    if submitted:
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
        result = score_order(order)
        if result:
            st.session_state.scored_orders.insert(0, result)
            label = ACTION_LABELS.get(result["action"], result["action"])
            status = st.success if result["action"] == "allow" else st.warning if result["action"] == "flag_for_review" else st.error
            status(f"{label} — risk_score={result['risk_score']:.3f}")
            for reason in result["reasons"]:
                st.write(f"- {reason}")

    st.divider()
    st.subheader("Recent scored orders (this session)")

    if not st.session_state.scored_orders:
        st.info("No orders scored yet this session - use the form above.")
    else:
        reviewer_name = st.text_input("Reviewer name (used when overriding a decision)", value="dashboard_user")

        for order in st.session_state.scored_orders:
            label = ACTION_LABELS.get(order["action"], order["action"])
            overridden_note = " (overridden)" if order.get("_overridden") else ""
            with st.expander(f"{label} — {order['order_id']}{overridden_note} (risk={order['risk_score']:.3f})"):
                st.write("**Reasons:**")
                for reason in order["reasons"]:
                    st.write(f"- {reason}")

                if order["action"] != "allow":
                    st.write("**Override this decision:**")
                    oc1, oc2, oc3 = st.columns([2, 2, 1])
                    with oc1:
                        new_action = st.selectbox(
                            "New action", ["allow", "flag_for_review", "restrict_cod"],
                            key=f"override_action_{order['order_id']}",
                        )
                    with oc2:
                        reason_text = st.text_input("Reason (required)", key=f"override_reason_{order['order_id']}")
                    with oc3:
                        st.write("")
                        st.write("")
                        if st.button("Override", key=f"override_btn_{order['order_id']}"):
                            if not reason_text.strip():
                                st.warning("Please enter a reason before overriding.")
                            else:
                                override_result = override_order(
                                    order["order_id"], new_action, reviewer_name.strip() or "dashboard_user", reason_text
                                )
                                if override_result:
                                    order["action"] = new_action
                                    order["_overridden"] = True
                                    st.success(f"Overridden to: {new_action}")
                                    st.rerun()

else:  # Metrics page
    st.subheader("Live metrics (from the backend's audit log)")

    try:
        response = requests.get(f"{API_BASE_URL}/metrics", timeout=10)
        response.raise_for_status()
        metrics = response.json()

        st.metric("Total scored orders", metrics["total_scored_orders"])

        col1, col2 = st.columns(2)
        with col1:
            st.write("**Action distribution**")
            if metrics["action_distribution"]:
                st.bar_chart(pd.Series(metrics["action_distribution"]))
            else:
                st.info("No orders scored yet.")

        with col2:
            st.write("**Human-override rate by action**")
            rate_rows = []
            for action, stats in metrics["override_rate_by_action"].items():
                rate_rows.append({
                    "action": action,
                    "auto_actions": stats["auto_count"],
                    "overridden": stats["override_count"],
                    "override_rate": f"{stats['rate']:.0%}",
                })
            if rate_rows:
                st.table(pd.DataFrame(rate_rows))
            else:
                st.info("No auto-actions logged yet.")

    except requests.exceptions.RequestException as e:
        st.error(f"Could not reach the backend at {API_BASE_URL}. Is uvicorn running? ({e})")
