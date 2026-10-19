# Executive Write-Up — SME Loan Pre-Screener

**Author:** Freddy Thosago · **Status:** prototype · **Data:** synthetic (no real customers)

## The problem

SA SMEs face slow, opaque working-capital decisions; credit officers drown in
manual first-pass screening. The prototype returns an instant indicative band
(Likely / Needs review / Unlikely) with plain-language reasons, so officers
spend their time on cases that need judgment.

## Technical approach

1. **Synthetic data with an engineered, documented target** — 800 applications;
   the label is a transparent "committee score" of affordability ratios, so the
   ground truth itself is auditable.
2. **Interpretable-first modelling** — a rule-based baseline written the way a
   credit officer would (coverage ≥ 2x, debt burden < 30%, margin ≥ 15%), then
   logistic regression (deployed) and gradient boosting (comparison), all on
   four human-legible ratios.
3. **Calibrated three-band output** — 35%/65% probability thresholds keep
   borderline-fundable businesses in the human-review band.

## Evaluation results (held-out 200 applications)

| Model | ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|
| Rule-based baseline | — | 0.625 | 0.379 | 0.472 |
| **Logistic regression (deployed)** | **0.722** | 0.494 | **0.636** | **0.556** |
| Gradient boosting (comparison) | 0.675 | 0.523 | 0.348 | 0.418 |

Deployed-model confusion matrix: TN 91 · FP 43 ·
FN 24 · TP 42.

**Error economics.** False positives (43 cases) cost the lender
wasted underwriting. False negatives (24 cases) cost the *applicant*:
a creditworthy SA SME is discouraged from growth. The model deliberately trades
precision for recall (recall 0.636 vs the baseline's 0.379) and the wide
🟡 review band ensures borderline cases always reach a human.

## Fairness audit (synthetic, disclosed)

The data generator includes a sector risk premium on purpose, so the audit has
a real disparity to catch. False-negative rate among approvable SMEs by sector:

- construction: 0.667
- manufacturing: 0.5
- retail: 0.421
- hospitality: 0.25
- services: 0.1

**Finding:** the engineered penalty against construction/hospitality applications
produces a visible FN-rate gap — demonstrating exactly the failure mode a
production fairness dashboard must catch. Mitigations in the prototype: sector
is excluded from model inputs; raw Rand amounts (business-size proxies) are
excluded in favour of scale-free ratios; FN rates are reported per group.

Business size: smaller-half FN rate 0.263 vs
larger-half 0.5 — flagged for monitoring
(smaller businesses must never be systematically screened out).

## Limitations

- Synthetic data; percentages illustrate the method, not the SA market.
- 800 records is a prototype scale; production needs years of labelled history,
  quarterly retraining and drift monitoring.
- No counterfactual explanations yet; the factors are transparent but not
  prescriptive about *what would change the band* (a planned enhancement).

## Ethical considerations

- **Human judgment is final** — the tool never approves or declines; the
  NCA/POPIA disclaimer is on every interface.
- **POPIA:** only business metrics are processed; no PII is collected.
- **Bias:** FN rates are audited by sector and size; proxy features are
  excluded by design.
- **Transparency:** the deployed model is a published logistic regression —
  intercept, coefficients and scaler are exported to the web prototype, so any
  engineer can verify every score by hand.
