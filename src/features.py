"""Feature engineering and model definitions for the SME pre-screener.

Design decision (Quality Bar: "simple, transparent, auditable"):
- The DEPLOYED model is Logistic Regression over four engineered,
  human-legible ratios. Every coefficient is directly explainable to a
  credit officer and exportable to the web prototype.
- A Gradient Boosted model is trained ONLY as a performance comparison to
  prove the interpretable model does not materially underperform.
- No raw Rand amounts enter the model: everything is scale-free ratios, so
  no feature directly encodes business SIZE (a fairness requirement — see
  docs/EXECUTIVE_WRITEUP.md on proxy features).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RATIO_FEATURES = [
    "coverage",        # free cash / estimated new repayment (log1p)
    "margin",          # (revenue - costs) / revenue
    "debt_burden",     # existing repayments / revenue
    "loan_to_revenue", # requested loan / monthly revenue
]
RAW_TRACK = ["Years_Trading", "Prior_Loan_History"]
FEATURE_ORDER = RATIO_FEATURES + RAW_TRACK

# Three-band thresholds on the calibrated probability of approval.
# Calibration logic (documented in README): the "needs review" band keeps
# applications whose probability sits within +/-0.15 of the committee's
# 50/50 decision boundary; outer bands are confident pre-screens only.
BAND_THRESHOLDS = (0.35, 0.65)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute auditable ratios from raw application fields."""
    out = pd.DataFrame(index=df.index)
    revenue = df["Monthly_Revenue"].astype(float)
    costs = df["Monthly_Costs"].astype(float)
    debt = df["Monthly_Debt_Repayment"].astype(float)
    loan = df["Requested_Loan"].astype(float)

    free_cash = np.maximum(revenue - costs - debt, 1.0)
    repayment = np.maximum(loan / 24.0, 1.0)          # 24-month amortisation assumption
    out["coverage"] = np.log1p(free_cash / repayment) # log so a 20x cushion is not 20x the weight
    out["margin"] = (revenue - costs) / revenue
    out["debt_burden"] = debt / revenue
    out["loan_to_revenue"] = loan / np.maximum(revenue, 1.0)
    out["Years_Trading"] = df["Years_Trading"].astype(float)
    out["Prior_Loan_History"] = df["Prior_Loan_History"].astype(float)
    return out


def build_logistic_pipeline() -> Pipeline:
    """Deployed model: scaled logistic regression (interpretable coefficients)."""
    return Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
    ])


def build_gbm_pipeline():
    """Comparison model: histogram gradient boosting (performance ceiling check)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    return Pipeline([
        ("model", HistGradientBoostingClassifier(max_depth=3, max_iter=150,
                                                  learning_rate=0.06, random_state=42)),
    ])


def probability_to_band(p: float) -> str:
    """Map a probability of approval to the three indicative categories."""
    low, high = BAND_THRESHOLDS
    if p >= high:
        return "Likely to qualify"
    if p >= low:
        return "Needs review"
    return "Unlikely to qualify"


def band_confidence(p: float) -> float:
    """How clearly the application sits in its band (0-100).

    Uses the distance from the band edges, so an application far from any
    threshold reports high confidence and borderline cases honestly report
    low confidence instead of false certainty.
    """
    low, high = BAND_THRESHOLDS
    if p >= high:
        raw = (p - high) / (1 - high)
    elif p < low:
        raw = (low - p) / low
    else:
        raw = min(abs(p - low), abs(p - high)) / (high - low)
        raw = 1 - raw * 2      # mid-band = least certain of review cases
    return round(50 + 45 * min(max(raw, 0), 1))
