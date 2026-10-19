"""Interpretable rule-based baseline (project brief, section B).

A credit officer with no ML at all could implement this by hand — that is the
point: any model we deploy must beat this transparent baseline to justify
its complexity. Returns one of the three indicative bands per application.

Rules (derived from standard SME affordability practice):
  LIKELY      free cash covers the estimated new instalment >= 2x,
              debt burden < 30% of revenue, margin >= 15%,
              and (trading >= 2 years or prior loan history)
  NEEDS REVIEW coverage 1x-2x, or borderline on exactly one other rule
  UNLIKELY     coverage < 1x, or debt burden >= 30%, or margin < 5%
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def rule_based_band(row: pd.Series) -> str:
    """Return the baseline band for a single application row."""
    revenue = float(row["Monthly_Revenue"])
    costs = float(row["Monthly_Costs"])
    debt = float(row["Monthly_Debt_Repayment"])
    loan = float(row["Requested_Loan"])
    years = float(row["Years_Trading"])
    history = float(row["Prior_Loan_History"])

    free_cash = max(revenue - costs - debt, 0.0)
    repayment = max(loan / 24.0, 1.0)
    coverage = free_cash / repayment
    debt_burden = debt / revenue
    margin = (revenue - costs) / revenue

    hard_fails = (coverage < 1.0) or (debt_burden >= 0.30) or (margin < 0.05)
    if hard_fails:
        return "Unlikely to qualify"

    strong = (coverage >= 2.0) and (debt_burden < 0.30) and (margin >= 0.15) \
             and ((years >= 2) or history == 1)
    if strong:
        return "Likely to qualify"
    return "Needs review"


def score_baseline(df: pd.DataFrame) -> pd.Series:
    """Apply the rule baseline to a dataframe of applications."""
    return df.apply(rule_based_band, axis=1)
