"""Generate a synthetic South African SME loan-application dataset (800 records).

SYNTHETIC DATA — no real customers, banks or applications are represented.

Source & assumptions (documented per project brief, section A):
- All values are drawn from distributions calibrated to plausible SA SME
  ranges (monthly revenue R30k-R900k; sectors: retail, services,
  construction, hospitality, manufacturing).
- Ground-truth "committee decision" is ENGINEERED from affordability ratios
  so the target label is transparent by construction:
    free_cash  = revenue - costs - debt
    repayment  = requested loan amortised over 24 months (prime-agnostic)
    coverage   = free_cash / repayment          (how many times the business
                                                 could cover the new instalment)
    debt_burden = debt / revenue
    margin     = (revenue - costs) / revenue
- A latent creditworthiness score is a weighted sum of coverage, margin,
  years trading, prior loan history and a SECTOR RISK PREMIUM. The premium
  is deliberately included so the fairness audit (src/train.py) has a real
  sector disparity to surface and discuss — this is disclosed in the README
  and docs/EXECUTIVE_WRITEUP.md.
- The binary target `Approved` = committee would pre-approve (1) or not (0),
  drawn Bernoulli(sigmoid(score)) with Gaussian noise, i.e. realistic label
  uncertainty rather than a deterministic step function.

Missing values (5% of debt repayment fields) are imputed as 0 = "no existing
monthly repayment", a decision chosen because the majority of zero-debt SMEs
in the generator genuinely have no facilities; it is logged transparently.

Run:  python data/generate_dataset.py  ->  data/sme_applications.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SEED = 42
N = 800
SECTORS = ["retail", "services", "construction", "hospitality", "manufacturing"]
# Engineered sector risk premium (disclosed): construction/hospitality carry
# payment-cycle and seasonality risk in the committee's view of the data.
SECTOR_PREMIUM = {"retail": 0.0, "services": 0.0, "construction": -0.8,
                  "hospitality": -0.6, "manufacturing": 0.2}


def generate(n: int = N, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    sector = rng.choice(SECTORS, n, p=[0.28, 0.30, 0.16, 0.14, 0.12])
    revenue = np.round(rng.lognormal(mean=11.2, sigma=0.55, size=n), -2)   # R30k-R900k/mo
    cost_ratio = np.clip(rng.beta(6, 3, n) * 0.5 + 0.45, 0.45, 0.95)      # 45-95% of revenue
    costs = np.round(revenue * cost_ratio, -2)
    has_debt = rng.binomial(1, 0.55, n)
    debt = np.round(revenue * rng.uniform(0.05, 0.45, n) * has_debt, -2)   # 0 if debt-free
    years_trading = np.round(rng.uniform(0.5, 15, n), 1)
    loan_multiple = rng.uniform(0.5, 6, n)                                  # months of revenue
    loan = np.round(revenue * loan_multiple, -3)
    prior_history = rng.binomial(1, 0.45, n)

    # ---- engineered ground truth (the transparent "committee model") ----
    free_cash = np.maximum(revenue - costs - debt, 1.0)
    repayment = np.maximum(loan / 24.0, 1.0)                               # 24-month amortisation
    coverage = free_cash / repayment
    debt_burden = debt / revenue
    margin = (revenue - costs) / revenue

    score = (
        -3.5                                     # intercept, tuned for ~40% approval
        + 1.1 * np.log1p(coverage)               # affordability is the dominant factor
        + 4.2 * margin                           # sustainable margins
        + 0.16 * years_trading                   # track record
        + 0.9 * prior_history                    # prior repayment history
        - 3.5 * np.maximum(0, debt_burden - 0.30)    # continuous over-indebtedness penalty
                                                  # above the 30% affordability ceiling
        - 0.09 * loan_multiple                      # committee discounts large asks
                                                  # relative to turnover
        + np.array([SECTOR_PREMIUM[s] for s in sector])
        + rng.normal(0, 0.28, n)                 # committee disagreement / noise
    )
    approved = rng.binomial(1, 1 / (1 + np.exp(-score)))

    df = pd.DataFrame({
        "SME_ID": [f"SME{i:04d}" for i in range(1, n + 1)],
        "Monthly_Revenue": revenue.astype(int),
        "Monthly_Costs": costs.astype(int),
        "Monthly_Debt_Repayment": debt.astype(int),
        "Years_Trading": years_trading,
        "Sector": sector,
        "Requested_Loan": loan.astype(int),
        "Prior_Loan_History": prior_history,
        "Approved": approved,                    # engineered target label
        # audit columns (not model inputs, used for fairness/analysis only)
        "_coverage": np.round(coverage, 2),
        "_debt_burden": np.round(debt_burden, 3),
        "_margin": np.round(margin, 3),
    })

    # transparent missing-value handling: 5% of debt fields -> 0 (= no facility)
    mask = rng.binomial(1, 0.05, n).astype(bool)
    df.loc[mask, "Monthly_Debt_Repayment"] = 0
    df.attrs["missing_debt_imputed"] = int(mask.sum())
    return df


if __name__ == "__main__":
    df = generate()
    df.to_csv("data/sme_applications.csv", index=False)
    print(f"Wrote {len(df)} rows -> data/sme_applications.csv")
    print(f"Approval rate: {df['Approved'].mean():.1%}")
    print(f"Debt fields imputed to 0: {df.attrs['missing_debt_imputed']} rows")
    print(df[["Monthly_Revenue", "Monthly_Costs", "Requested_Loan", "Years_Trading"]]
          .describe().round(0).to_string())
