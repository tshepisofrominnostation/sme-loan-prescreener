"""Streamlit reference UI: SME Loan Pre-Screener for SA credit officers.

Run:  streamlit run app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from src.prescreen import prescreen  # noqa: E402

st.set_page_config(page_title="SME Loan Pre-Screener", page_icon="🧾",
                   layout="wide", initial_sidebar_state="expanded")

BAND_STYLE = {
    "Likely to qualify": ("🟢", "#009639"),
    "Needs review": ("🟡", "#e08a00"),
    "Unlikely to qualify": ("🔴", "#c62828"),
}

st.title("SME Loan Pre-Screener")
st.caption("Explainable, fair pre-screening for South African SME working-capital "
           "applications. Indicative only — human judgment is final.")

st.error("⚠️ **This is a decision-support prototype.** It does not approve or "
         "decline any loan. All indicative outputs must be reviewed by a "
         "qualified credit officer. Production deployment would require "
         "compliance with the National Credit Act (NCA) and POPIA — out of "
         "scope for this prototype.", icon="⚖️")

with st.sidebar:
    st.header("Application inputs")
    revenue = st.number_input("Monthly revenue (ZAR)", 10000, 5000000, 250000, 10000)
    costs = st.number_input("Monthly operating costs (ZAR)", 5000, 5000000, 180000, 5000)
    debt = st.number_input("Existing monthly debt repayments (ZAR)", 0, 2000000, 15000, 1000)
    years = st.slider("Years trading", 0.5, 15.0, 4.0, 0.5)
    sector = st.selectbox("Sector", ["retail", "services", "construction",
                                     "hospitality", "manufacturing"])
    loan = st.number_input("Requested loan amount (ZAR)", 10000, 20000000, 500000, 10000)
    history = st.radio("Prior loan history?", ["No", "Yes"], horizontal=True)

application = {
    "Monthly_Revenue": float(revenue),
    "Monthly_Costs": float(costs),
    "Monthly_Debt_Repayment": float(debt),
    "Years_Trading": float(years),
    "Sector": sector,                    # recorded for auditing; excluded from model inputs
    "Requested_Loan": float(loan),
    "Prior_Loan_History": 1.0 if history == "Yes" else 0.0,
}

result = prescreen(application)
emoji, color = BAND_STYLE[result["band"]]

col1, col2, col3 = st.columns(3)
col1.metric("Indicative assessment", f"{emoji} {result['band']}")
col2.metric("Confidence", f"{result['confidence']}%")
col3.metric("Model approval probability", f"{result['approval_probability']:.0%}")

low, high = result["thresholds"]
st.caption(f"Band thresholds: Unlikely < {low:.0%} · Needs review {low:.0%}–{high:.0%} · "
           f"Likely > {high:.0%} (calibrated so borderline fundable cases stay in review).")

left, right = st.columns([3, 2])
with left:
    st.subheader("Why this assessment")
    for factor in result["factors"]:
        icon = {"up": "▲", "down": "▼", "neutral": "•"}[factor["direction"]]
        tint = {"up": "#009639", "down": "#c62828", "neutral": "#8aa095"}[factor["direction"]]
        st.markdown(
            f"<div style='border-left:3px solid {tint};padding-left:.6rem;margin:.4rem 0;"
            f"color:#cbd5e1;font-size:.95rem'>{icon} {factor['text']}</div>",
            unsafe_allow_html=True)

with right:
    st.subheader("Model transparency")
    st.info("Deployed model: logistic regression over four auditable ratios "
            "(instalment coverage, operating margin, debt burden, loan-to-revenue) "
            "plus trading years and prior history. Every factor above maps 1:1 to a "
            "model input — no black-box scores.", icon="🔍")
    ratios = pd.Series({
        "Instalment coverage (x)": None, "Operating margin": None,
        "Debt burden": None, "Loan / monthly revenue": None,
    })
    st.dataframe(ratios.rename("value"), use_container_width=True)
