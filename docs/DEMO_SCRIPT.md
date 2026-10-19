# 5-Minute Demo Script — SME Loan Pre-Screener

**Live demo:** https://tshepisofrominnostation.github.io/sme-loan-prescreener/
(keep the repo and the workflow diagram open in other tabs)

| Time | Beat | What to say / show |
|---|---|---|
| 0:00–0:30 | **Frame the problem** | "SA SMEs wait weeks for opaque credit decisions. This prototype is a co-pilot for credit officers — it never approves or declines anything, it accelerates first-pass screening." Point at the disclaimer banner. |
| 0:30–1:15 | **Healthy case** | Enter the defaults (R250k revenue / R175k costs / R8k debt / 6 years / R400k loan / prior history). Run. Result: **🟢 Likely to qualify**, ~86% model probability. Walk two factors: instalment coverage ~3.2x, debt burden 3% — "every factor maps to a model input; nothing is black-box." |
| 1:15–2:00 | **Over-indebted case** | Change debt to R80,000 (40% of revenue) and loan to R500,000. Run. Result flips to **🔴 Unlikely** — the two killer factors: free cash no longer covers the instalment, and debt burden breaches the 30% ceiling. "Notice the tool explains *why*, in language a business owner could read." |
| 2:00–2:45 | **Borderline case** | Revenue R120k, costs R95k, debt R12k, 2.5 years, R250k loan, no history. Result lands near the 35% boundary — show the **confidence score drops** for borderline cases: "the tool is honest about its own uncertainty; borderline fundable businesses stay in the 🟡 human-review band, never auto-declined." |
| 2:45–3:30 | **How it works** | Open `docs/workflow_diagram.png`. "Seven inputs → four auditable ratios → a published logistic regression → calibrated bands. The gradient-boosted comparison model didn't beat it, so we kept the interpretable one — in financial services, auditable beats clever." Quote test metrics: ROC-AUC 0.72, recall 0.64 vs the hand-rule baseline's 0.38. |
| 3:30–4:15 | **Fairness & responsibility** | "FN rates are audited by sector and business size — the synthetic data deliberately contains a construction/hospitality penalty, and the audit catches it: construction FN rate 0.67 vs services 0.10. Raw Rand amounts never enter the model, so business size can't act as a proxy. Sector is captured for audit but excluded from scoring." |
| 4:15–5:00 | **Close** | "Limitations are documented — synthetic data, prototype scale. Next steps: counterfactual explanations ('what would move this SME into 🟢'), calibration curves, and a live fairness dashboard. Full evaluation and executive write-up are in the repo." Point at GitHub link. |

**Presenter Q&A prep**
- *Why logistic regression, not XGBoost?* — Comparable AUC (0.72 vs 0.68), full auditability, exportable to a browser. The quality bar: transparent beats black-box.
- *Why does recall matter more than precision?* — A false negative costs a creditworthy SME its growth; a false positive costs an underwriter an hour. The wide review band mitigates both.
- *Is it POPIA-compliant?* — The prototype processes business metrics only and stores nothing; production would need a full NCA/POPIA compliance process, stated up front.
