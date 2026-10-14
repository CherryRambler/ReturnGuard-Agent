"""
ReturnGuard Agent - Streamlit operations dashboard (entry point).

Talks ONLY to the FastAPI backend (never touches the model or database
directly) - so what you see here is exactly what the API returns.

Run with (two terminals, backend must already be running):
    uvicorn backend.main:app --reload
    streamlit run frontend/app.py

This file only sets up the page, the theme and the navigation. Each
page lives in its own file under frontend/views/ and gets its own URL;
shared API calls, helpers and components live in frontend/common.py.

The permanent record of every decision lives in the backend's audit_log
table (backend/db.py). Every page reads it live through the API - not
browser-only session state.
"""

import streamlit as st

from common import apply_theme, render_sidebar_status

st.set_page_config(
    page_title="ReturnGuard Agent",
    page_icon="🛡️",
    layout="wide",
)
apply_theme()

pages = {
    "Monitor": [
        st.Page("views/overview.py", title="Overview", icon=":material/dashboard:",
                default=True),
        st.Page("views/score.py", title="Score an Order", icon=":material/add_card:",
                url_path="score"),
        st.Page("views/decisions.py", title="Decisions", icon=":material/receipt_long:",
                url_path="decisions"),
        st.Page("views/review.py", title="Review Queue", icon=":material/fact_check:",
                url_path="review"),
    ],
    "Records": [
        st.Page("views/audit.py", title="Audit Log", icon=":material/history_edu:",
                url_path="audit"),
        st.Page("views/analytics.py", title="Analytics", icon=":material/monitoring:",
                url_path="analytics"),
        st.Page("views/settings.py", title="Settings", icon=":material/tune:",
                url_path="settings"),
    ],
}

st.logo(
    "data:image/svg+xml;utf8,"
    "<svg xmlns='http://www.w3.org/2000/svg' width='180' height='32' viewBox='0 0 180 32'>"
    "<rect x='0' y='3' width='26' height='26' rx='4' fill='%23b08a3e'/>"
    "<text x='13' y='21.5' font-family='Georgia,serif' font-size='16' font-weight='700' "
    "fill='%23121a24' text-anchor='middle'>R</text>"
    "<text x='36' y='22' font-family='Georgia,serif' font-size='18' font-weight='600' "
    "fill='%23f1f3f6'>ReturnGuard</text></svg>",
    size="large",
)
nav = st.navigation(pages)
render_sidebar_status()
nav.run()
