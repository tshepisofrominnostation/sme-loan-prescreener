# Explainable AI Loan Pre-Screener for SA SMEs

**Live prototype:** https://tshepisofrominnostation.github.io/sme-loan-prescreener/
**Author:** Freddy Thosago · CAPACITI / Clickatell AI & Software Development Bootcamp

> **Indicative pre-screen only — human judgment is final.** This tool does not
> approve or decline loans. Production deployment would require compliance with
> the **National Credit Act (NCA)** and **POPIA** (out of scope for this prototype).

## Problem

South African SMEs seeking working capital face slow, opaque credit processes.
This prototype is an **intelligent co-pilot for human credit officers**: it
accelerates evaluation, surfaces critical insights, and keeps the assessment
fair and transparent. Given seven lightweight business inputs it returns:

- 🟢 **Likely to qualify** (model probability > 65%)
- 🟡 **Needs review** (35–65%)
- 🔴 **Unlikely to qualify** (< 35%)

…each with a confidence score and **plain-language factors** explaining exactly
what raised or lowered the assessment.

## Architecture

```
Intake (7 fields) → Feature engineering (4 auditable ratios)
→ Logistic model (calibrated 3-band probability)
→ Plain-language factors → Human credit officer decides
        ↳ governance at every step: NCA/POPIA disclaimer,
          fairness audit, proxy-feature review, published coefficients
```

One-page workflow diagram: [`docs/workflow_diagram.png`](docs/workflow_diagram.png)

**Why the ML layer is trustworthy here:** the deployed model is *logistic
regression over four human-legible ratios* — instalment coverage, operating
margin, debt burden, loan-to-revenue — plus trading years and prior history.
Every displayed factor maps 1:1 to a model input. A gradient-boosted model was
trained as a performance ceiling check and did **not** beat the interpretable
model, so the black box was rejected on purpose (see evaluation below).

## Data foundation (synthetic, documented)

- 800 synthetic SME applications generated from calibrated SA-plausible
  distributions (monthly revenue R10k–R420k, 5 sectors). **No real customers.**
- Target label engineered transparently: a "committee score" built from
  affordability ratios (coverage, margin, debt burden above a 30% ceiling,
  large-ask discount, trading years, prior history, sector risk premium) passed
  through a sigmoid with label noise — full derivation in
  [`data/generate_dataset.py`](data/generate_dataset.py).
- **Missing values:** 5% of debt fields imputed to 0 (= no existing facility),
  a decision logged in the generator output; nothing was dropped.
- **Disclosure:** a sector risk premium (construction −0.8, hospitality −0.6)
  is deliberately engineered so the fairness audit has a real disparity to
  surface — see the audit results below.

## Evaluation (held-out test set, 200 of 800 applications)

| Model | ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|
| Rule-based baseline (hand-written affordability rules) | — | 0.62 | 0.38 | 0.47 |
| **Logistic regression (deployed)** | **0.72** | 0.49 | **0.64** | **0.56** |
| Gradient boosting (comparison) | 0.68 | 0.52 | 0.35 | 0.42 |

**Baseline insight:** hand-written rules are reasonably precise but miss ~62%
of fundable SMEs. The model recovers most of them.

**FP vs FN business cost (lending context):**
- **False positives** (predicted fundable, committee would decline) cost the
  *lender* wasted underwriting effort.
- **False negatives** (fundable SME screened out) harm the *applicant* — lost
  growth for a creditworthy business. The model is therefore tuned for recall
  (0.64 vs the baseline's 0.38) and the middle 🟡 band is deliberately wide so
  borderline-fundable businesses are **never auto-discouraged**.

**Threshold calibration:** bands at 35%/65% keep any application within ±15%
of the committee's 50/50 boundary in the human-review band — borderline cases
go to a person, not a verdict.

## Fairness audit (synthetic, disclosed)

False-negative rates among approvable SMEs, by sector — the engineered
construction/hospitality premium shows up as expected, which is exactly what
the audit exists to catch:

| Sector | FN rate among approvable |
|---|---|
| construction | 0.67 |
| manufacturing | 0.50 |
| retail | 0.42 |
| hospitality | 0.25 |
| services | 0.10 |

**Proxy features:** raw Rand amounts never enter the model — only scale-free
ratios — so business *size* cannot act as a proxy for anything. Sector is
recorded for auditing but excluded from model inputs. Full audit:
[`reports/fairness.json`](reports/fairness.json).

## Responsible AI guardrails

1. **Bias & disparity** — FN/FP rates monitored by sector and business size;
   size-based proxy features excluded by design; human review is mandatory.
2. **PII / POPIA** — the tool only ever sees business metrics; no personal
   identifying data is collected anywhere in the prototype.
3. **No black-box verdicts** — the deployed model is a published, auditable
   logistic regression; every output is explained by factors tied 1:1 to inputs,
   and the NCA/POPIA disclaimer is displayed on every screen.

## Repository layout

```
data/generate_dataset.py      synthetic data generator (documented)
data/sme_applications.csv    the generated dataset (800 rows)
src/features.py               engineered ratios + model definitions
src/rules_baseline.py         interpretable rule-based baseline
src/train.py                  training, evaluation, FP/FN cost, fairness audit
src/prescreen.py              explainable assessment entry point
app.py                        Streamlit reference UI
docs/index.html               live web prototype (client-side port)
docs/workflow_diagram.png     one-page workflow map
docs/EXECUTIVE_WRITEUP.md     executive overview
docs/DEMO_SCRIPT.md           5-minute demo walkthrough
reports/                      evaluation + fairness JSON outputs
models/                       trained model + published coefficient export
```

## Run it

```bash
git clone https://github.com/tshepisofrominnostation/sme-loan-prescreener.git
cd sme-loan-prescreener
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python data/generate_dataset.py     # regenerate the synthetic dataset
python -m src.train                 # train + evaluate + fairness audit
streamlit run app.py                # reference UI at http://localhost:8501
```

The live web prototype (GitHub Pages) runs the same published model entirely
in your browser — no install needed.

## Limitations & future enhancements

- Synthetic data: percentages are illustrative, not market estimates.
- 800 records: a production system would train on years of labelled
  applications, retrain quarterly, and add drift monitoring.
- Future: calibration curves, PR-AUC threshold optimisation on a validation
  split, counterfactual explanations ("what would move this SME into 🟢"),
  and a fairness dashboard over live decision logs.
