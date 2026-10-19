"""Explainable pre-screening: the single entry point used by app.py and the web port.

For every application it returns:
  - indicative band (Likely to qualify / Needs review / Unlikely to qualify)
  - confidence score (0-100, distance from band thresholds)
  - plain-language factors that RAISED or LOWERED the assessment

The factor explanations are generated from the SAME engineered ratios the
model consumes, each tied to a credit convention (35% debt-burden ceiling,
2x instalment coverage, 15% healthy margin). They are auditable by
construction: no factor is ever shown that the model did not see.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np

from src.features import (BAND_THRESHOLDS, FEATURE_ORDER, band_confidence,
                          engineer_features, probability_to_band)

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "logreg_model.joblib"
_MODEL = None


def load_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = joblib.load(MODEL_PATH)
    return _MODEL


def _factors(raw: Dict[str, float]) -> List[Dict[str, str]]:
    """Plain-language factors with direction and standard-formatted amounts."""
    revenue = float(raw["Monthly_Revenue"])
    costs = float(raw["Monthly_Costs"])
    debt = float(raw["Monthly_Debt_Repayment"])
    loan = float(raw["Requested_Loan"])
    years = float(raw["Years_Trading"])
    history = float(raw["Prior_Loan_History"])

    free_cash = max(revenue - costs - debt, 0.0)
    repayment = max(loan / 24.0, 1.0)
    coverage = free_cash / repayment
    debt_burden = (debt / revenue) if revenue else 0.0
    margin = (revenue - costs) / revenue
    loan_months = (loan / revenue) if revenue else 0.0

    def zar(v: float) -> str:
        return f"R{v:,.0f}"

    factors: List[Dict[str, str]] = []

    # affordability / coverage
    if coverage >= 2:
        factors.append({"direction": "up",
            "text": f"After costs and existing debt, free cash of {zar(free_cash)} per month "
                    f"covers the estimated new instalment of {zar(repayment)} about "
                    f"{coverage:.1f} times over — a healthy affordability cushion."})
    elif coverage >= 1:
        factors.append({"direction": "down",
            "text": f"Free cash of {zar(free_cash)} covers the estimated instalment of "
                    f"{zar(repayment)} only {coverage:.1f} times. Under 2x coverage the "
                    "committee treats affordability as tight, lowering the score."})
    else:
        factors.append({"direction": "down",
            "text": f"Free cash of {zar(free_cash)} per month does not cover the estimated "
                    f"instalment of {zar(repayment)} (coverage {coverage:.1f}x). As modeled, "
                    "this loan is not affordable on current cash flow."})

    # debt burden
    if debt_burden >= 0.35:
        factors.append({"direction": "down",
            "text": f"Existing debt repayments consume {debt_burden:.0%} of monthly revenue, "
                    f"above the 30% affordability ceiling — this significantly lowers the "
                    f"qualification score."})
    elif debt > 0:
        factors.append({"direction": "up",
            "text": f"Existing debt repayments consume only {debt_burden:.0%} of monthly "
                    f"revenue, comfortably inside the 30% ceiling."})
    else:
        factors.append({"direction": "up",
            "text": "No existing monthly debt repayments are recorded — the balance sheet "
                    "has capacity for new credit."})

    # margin
    if margin >= 0.15:
        factors.append({"direction": "up",
            "text": f"Operating costs consume {1 - margin:.0%} of revenue, leaving a "
                    f"{margin:.0%} operating margin — above the 15% healthy-business bar."})
    elif margin >= 0.05:
        factors.append({"direction": "down",
            "text": f"Operating costs consume {1 - margin:.0%} of revenue, leaving only a "
                    f"{margin:.0%} margin — thin for absorbing a new instalment."})
    else:
        factors.append({"direction": "down",
            "text": f"Operating costs consume {1 - margin:.0%} of revenue; a {margin:.0%} "
                    f"margin leaves almost no room to service new debt."})

    # loan size
    if loan_months <= 3:
        factors.append({"direction": "up",
            "text": f"The requested {zar(loan)} is about {loan_months:.1f} months of revenue — "
                    f"a conservative ask relative to turnover."})
    elif loan_months <= 6:
        factors.append({"direction": "neutral",
            "text": f"The requested {zar(loan)} is about {loan_months:.1f} months of revenue, "
                    f"a sizeable but not unusual working-capital request."})
    else:
        factors.append({"direction": "down",
            "text": f"The requested {zar(loan)} is about {loan_months:.1f} months of revenue — "
                    f"a large ask relative to turnover, which raises repayment risk."})

    # track record
    if years >= 5:
        factors.append({"direction": "up",
            "text": f"{years:.0f} years of trading history provides an established track record."})
    elif years >= 2:
        factors.append({"direction": "neutral",
            "text": f"{years:.1f} years of trading — a moderate track record."})
    else:
        factors.append({"direction": "down",
            "text": f"Only {years:.1f} years of trading; young businesses carry higher "
                    f"failure risk in the committee's view."})

    # history
    if history == 1:
        factors.append({"direction": "up",
            "text": "The owner has prior loan history, which supports a repayment track record."})
    else:
        factors.append({"direction": "down",
            "text": "No prior loan history is recorded, so repayment behaviour is unproven."})

    return factors


def prescreen(application: Dict[str, float]) -> Dict[str, object]:
    """Assess one application and return band, confidence and factors."""
    import pandas as pd
    raw = pd.DataFrame([application])
    X = engineer_features(raw)[FEATURE_ORDER]
    model = load_model()
    probability = float(model.predict_proba(X)[0, 1])

    return {
        "band": probability_to_band(probability),
        "confidence": band_confidence(probability),
        "approval_probability": round(probability, 4),
        "thresholds": list(BAND_THRESHOLDS),
        "factors": _factors(application),
        "disclaimer": ("Indicative pre-screen only. This tool does not approve or "
                       "decline loans; a qualified credit officer makes the final "
                       "decision. Production use requires NCA and POPIA compliance."),
    }
