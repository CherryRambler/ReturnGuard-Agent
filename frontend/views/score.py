"""Score an Order - run one order through the agent pipeline and show
the risk score, the decision, why it was made, and the stage timeline."""

import uuid

import requests
import streamlit as st

from common import (
    ACTION_DISPLAY,
    _post,
    action_pill,
    api_cap,
    api_decision_detail,
    connection_banner,
    fmt_ts,
    humanize_reason,
    page_header,
    refresh_all,
    render_cap_card,
    render_explainability,
    render_timeline,
    risk_badge,
    short_id,
    status_pill,
)

CATEGORIES = ["apparel", "electronics", "home", "beauty", "footwear"]
PAYMENT_METHODS = ["UPI", "card", "netbanking", "COD"]
RISK_TIERS = ["low", "medium", "high"]


def _new_order_id() -> str:
    return f"order-{uuid.uuid4().hex[:8]}"


def render_order_form():

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

    with st.form("decision_form", border=False):
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
            "Score order and decide", type="primary", width='stretch'
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
    with rc1.container(border=True, height="stretch", key="card-risk"):
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

    with rc2.container(border=True, height="stretch", key="card-decision"):
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

    with rc3.container(border=True, height="stretch", key="card-action"):
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

    st.markdown("")
    ec1, ec2 = st.columns([1, 1])
    with ec1.container(border=True, key="card-why"):
        render_explainability(result.get("reasons", []))
    with ec2.container(border=True, key="card-timeline"):
        detail, derr = api_decision_detail(result["order_id"])
        st.markdown('<div class="rg-kicker">Order Decision Timeline</div>', unsafe_allow_html=True)
        if detail:
            render_timeline(detail)
        else:
            st.info("Timeline unavailable." + (f" ({derr})" if derr else ""))


page_header(
    "Checkout decision",
    "Score an Order",
    "Enter the order as it looks at checkout. The agent scores its return "
    "risk, picks a bounded action, checks the hourly cap and logs it.",
)

if connection_banner("score"):
    left, right = st.columns([2, 1])
    with left.container(border=True, key="card-form"):
        render_order_form()
    with right.container(border=True, key="card-cap"):
        cap, cap_err = api_cap()
        render_cap_card(cap, cap_err)

    render_last_result()
