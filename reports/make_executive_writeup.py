"""Executive write-up generator: assembles the executive overview from REAL
run artifacts (evaluation.json, fairness.json) so numbers can never drift
from the code. Run after `python -m src.train`.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ev = json.loads((ROOT / "reports" / "evaluation.json").read_text())
fair = json.loads((ROOT / "reports" / "fairness.json").read_text())
lr, gbm, base = ev["logistic_regression"], ev["gradient_boosting"], ev["baseline_rules"]
sectors = sorted(fair["by_sector"].items(), key=lambda x: -x[1]["fn_rate_among_approvable"])
size = fair["by_size"]

doc = f"""# Executive Write-Up — SME Loan Pre-Screener

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
| Rule-based baseline | — | {base['precision']} | {base['recall']} | {base['f1']} |
| **Logistic regression (deployed)** | **{lr['roc_auc']}** | {lr['precision']} | **{lr['recall']}** | **{lr['f1']}** |
| Gradient boosting (comparison) | {gbm['roc_auc']} | {gbm['precision']} | {gbm['recall']} | {gbm['f1']} |

Deployed-model confusion matrix: TN {lr['confusion']['tn']} · FP {lr['confusion']['fp']} ·
FN {lr['confusion']['fn']} · TP {lr['confusion']['tp']}.

**Error economics.** False positives ({lr['confusion']['fp']} cases) cost the lender
wasted underwriting. False negatives ({lr['confusion']['fn']} cases) cost the *applicant*:
a creditworthy SA SME is discouraged from growth. The model deliberately trades
precision for recall (recall {lr['recall']} vs the baseline's {base['recall']}) and the wide
🟡 review band ensures borderline cases always reach a human.

## Fairness audit (synthetic, disclosed)

The data generator includes a sector risk premium on purpose, so the audit has
a real disparity to catch. False-negative rate among approvable SMEs by sector:

{chr(10).join(f"- {s}: {v['fn_rate_among_approvable']}" for s, v in sectors)}

**Finding:** the engineered penalty against construction/hospitality applications
produces a visible FN-rate gap — demonstrating exactly the failure mode a
production fairness dashboard must catch. Mitigations in the prototype: sector
is excluded from model inputs; raw Rand amounts (business-size proxies) are
excluded in favour of scale-free ratios; FN rates are reported per group.

Business size: smaller-half FN rate {size['smaller half']['fn_rate_among_approvable']} vs
larger-half {size['larger half']['fn_rate_among_approvable']} — flagged for monitoring
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
"""

out = ROOT / "docs" / "EXECUTIVE_WRITEUP.md"
out.write_text(doc)
print(f"wrote {out} ({len(doc)} chars)")
