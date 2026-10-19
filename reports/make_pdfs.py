"""Generate PROJECT_DOCUMENTATION.pdf and PRESENTATION.pdf from real artifacts.

Sources of truth:
  reports/evaluation.json  - model metrics, confusion matrix, FP/FN cost analysis
  reports/fairness.json    - sector and size disparity audit
  docs/workflow_diagram.png - architecture visual

Run:  python reports/make_pdfs.py
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

from weasyprint import HTML

ROOT = Path(__file__).resolve().parent.parent
GREEN, DARK, LIGHT = "#009639", "#0b4d24", "#f2f6f3"
INK, MUTED = "#15321f", "#51685c"
AMBER, RED = "#b25e00", "#c62828"

ev = json.loads((ROOT / "reports/evaluation.json").read_text())
fa = json.loads((ROOT / "reports/fairness.json").read_text())
diagram_b64 = base64.b64encode((ROOT / "docs/workflow_diagram.png").read_bytes()).decode()
lr, gb, base_ = ev["logistic_regression"], ev["gradient_boosting"], ev["baseline_rules"]
cm = lr["confusion"]
sectors = sorted(fa["by_sector"].items(), key=lambda kv: -kv[1]["fn_rate_among_approvable"])
size = fa["by_size"]

DOC_CSS = f"""
@page {{ size: A4; margin: 18mm 16mm 20mm;
  @bottom-center {{ content: "SME Loan Pre-Screener - Project Documentation - page " counter(page);
                   font-size: 8pt; color: {MUTED}; font-family: sans-serif; }} }}
body {{ font-family: sans-serif; color: {INK}; font-size: 10pt; line-height: 1.45; }}
h1 {{ color: {DARK}; font-size: 21pt; margin: 0 0 2mm; }}
h2 {{ color: {GREEN}; font-size: 13pt; margin: 7mm 0 2mm; border-bottom: 1.2pt solid {GREEN}; padding-bottom: 1mm; }}
h3 {{ color: {DARK}; font-size: 11pt; margin: 4mm 0 1mm; }}
table {{ border-collapse: collapse; width: 100%; margin: 2mm 0 3mm; font-size: 9pt; }}
th {{ background: {DARK}; color: #fff; text-align: left; padding: 2mm 3mm; }}
td {{ border: 0.6pt solid #c9d8ce; padding: 1.8mm 3mm; vertical-align: top; }}
.cover {{ background: {DARK}; color: #fff; padding: 14mm 10mm; border-radius: 4mm; margin-bottom: 6mm; }}
.cover h1 {{ color: #fff; }}
.kicker {{ color: #9fd6b2; font-size: 9pt; letter-spacing: 2pt; text-transform: uppercase; }}
.band {{ display: inline-block; padding: 1mm 3mm; border-radius: 999pt; font-size: 8.5pt; font-weight: bold; }}
.b-green {{ background: {GREEN}; color: #fff; }} .b-amber {{ background: {AMBER}; color: #fff; }}
.b-red {{ background: {RED}; color: #fff; }}
.callout {{ background: {LIGHT}; border-left: 3pt solid {GREEN}; padding: 2.5mm 4mm; margin: 2mm 0 3mm; }}
.warn {{ background: #fff8e8; border-left: 3pt solid #e0b64f; padding: 2.5mm 4mm; margin: 2mm 0; }}
code {{ background: {LIGHT}; padding: 0.4mm 1.5mm; border-radius: 2pt; font-size: 8.5pt; }}
img.diagram {{ width: 100%; margin: 3mm 0; }}
"""

def _kv_rows(label, d):
    return f"<tr><td><b>{label}</b></td><td>{d['precision']:.3f}</td><td>{d['recall']:.3f}</td><td>{d['f1']:.3f}</td><td>{d.get('roc_auc', 0):.3f}</td><td>{d['confusion']['tn']}/{d['confusion']['fp']}/{d['confusion']['fn']}/{d['confusion']['tp']}</td></tr>"

DOC_HTML = f"""<html><head><meta charset="utf-8"><style>{DOC_CSS}</style></head><body>
<div class="cover">
  <p class="kicker">Portfolio Project Documentation</p>
  <h1>SME Loan Pre-Screener</h1>
  <p style="font-size:11pt; margin:2mm 0 0;">Explainable, Fair AI Loan Pre-Screening for South African SMEs</p>
  <p style="font-size:9pt; color:#9fd6b2; margin-top:3mm;">Tshepiso Freddy Thosago &middot; October 2026 &middot; Synthetic data only &middot; Indicative tool, human judgment final</p>
</div>

<h2>1. Problem statement</h2>
<p>South African SMEs seeking working capital face slow, opaque credit processes. This prototype is an
<b>AI co-pilot for credit officers</b>: it accelerates evaluation with a three-band indicative assessment
and plain-language reasons, and it never approves or declines a loan itself.</p>
<p><span class="band b-green">LIKELY TO QUALIFY</span>
<span class="band b-amber">NEEDS REVIEW</span>
<span class="band b-red">UNLIKELY TO QUALIFY</span></p>

<h2>2. Architecture</h2>
<img class="diagram" src="data:image/png;base64,{diagram_b64}">
<p class="callout">Seven lightweight inputs are engineered into affordability ratios, scored by an interpretable
logistic model, mapped to calibrated probability bands, and explained with plain-language factors. A human credit
officer makes the final call at every outcome.</p>

<h2>3. Data foundation</h2>
<ul>
<li><b>Source:</b> fully synthetic, seeded generator (<code>data/generate_dataset.py</code>), 800 applications; no real customer data.</li>
<li><b>Target engineering:</b> a simulated credit committee approves an application when an affordability
score (instalment coverage, margin, debt ceiling at 30%, trading years, prior history, sector premium) clears a
calibrated logistic threshold plus noise. Approval rate: <b>{ev['dataset']['approval_rate']:.1%}</b>.</li>
<li><b>Missing values:</b> debt-free applicants (no existing repayments) are explicitly imputed to 0 and documented; no rows dropped.</li>
</ul>

<h2>4. Model selection and evaluation</h2>
<table>
<tr><th>Model</th><th>Precision</th><th>Recall</th><th>F1</th><th>ROC-AUC</th><th>Confusion (TN/FP/FN/TP)</th></tr>
{_kv_rows("Rule-based baseline", base_)}
{_kv_rows("Logistic regression (deployed)", lr)}
{_kv_rows("Gradient boosting (comparison)", gb)}
</table>
<p>Held-out test set: {ev['dataset']['test']} applications (25% stratified split, seed 42). The logistic model is
deployed because it matches the gradient booster's discrimination while remaining fully auditable. <code>class_weight='balanced'</code>
deliberately trades precision for recall: <b>false negatives harm the applicant</b> (a fundable SME is discouraged),
while false positives cost the lender underwriting time and are caught in detailed review.</p>
<h3>False-positive vs false-negative cost</h3>
<table>
<tr><th></th><th>Meaning in this prototype</th><th>Who bears the cost</th></tr>
<tr><td><b>False positives ({cm['fp']})</b></td><td>{ev['fp_fn_cost_analysis']['false_positives']['meaning']}</td><td>The lender</td></tr>
<tr><td><b>False negatives ({cm['fn']})</b></td><td>{ev['fp_fn_cost_analysis']['false_negatives']['meaning']}</td><td>The applicant</td></tr>
</table>

<h2>5. Explainability (non-negotiable)</h2>
<p>Every assessment lists the top factors that raised or lowered the score, in credit-officer language.
Factors are computed from the same audited ratios the model sees, for example:</p>
<ul>
<li>Free cash of R67,000 per month covers the estimated new instalment of R16,667 about 4.0 times.</li>
<li>Existing debt repayments consume 40% of monthly revenue, above the 30% affordability ceiling.</li>
<li>The requested R600,000 is about 6.7 months of revenue - a large ask relative to turnover.</li>
</ul>

<h2>6. Fairness audit</h2>
<p>False-negative rate among approvable applications, by sector (test set):</p>
<table><tr><th>Sector</th><th>N</th><th>False negatives</th><th>FN rate</th></tr>
{''.join(f"<tr><td>{k.title()}</td><td>{v['n']}</td><td>{v['false_negatives']}</td><td>{v['fn_rate_among_approvable']:.2f}</td></tr>" for k, v in sectors)}
</table>
<p>By business size (revenue median split): smaller half <b>{size['smaller half']['fn_rate_among_approvable']:.2f}</b>,
larger half <b>{size['larger half']['fn_rate_among_approvable']:.2f}</b>.</p>
<div class="warn"><b>Disclosure:</b> the sector gaps are engineered into the synthetic generator (construction and
hospitality carry a committee risk premium) so the audit has a real disparity to surface. In production the
monitoring pipeline is the deliverable: FN-rate gaps by sector and size would be tracked monthly, and persistent
gaps trigger model review. No protected characteristics are inputs; the small/large revenue split is monitored as a proxy-risk feature.</div>

<h2>7. Responsible AI and regulatory framing</h2>
<ul>
<li><b>Indicative only:</b> prominent, persistent disclaimer - the tool never approves or declines.</li>
<li><b>NCA and POPIA:</b> production deployment would require National Credit Act registration and POPIA
compliance; out of scope for this prototype and stated on the interface.</li>
<li><b>Proxy features:</b> sector and size are retained for explainability but flagged and audited; protected
characteristics are excluded by design.</li>
</ul>

<h2>8. Reproduce</h2>
<code>git clone https://github.com/tshepisofrominnostation/sme-loan-prescreener</code><br>
<code>pip install -r requirements.txt</code><br>
<code>python data/generate_dataset.py &amp;&amp; python -m src.train</code><br>
<code>streamlit run app.py</code><br>
<p>Web prototype: <b>https://tshepisofrominnostation.github.io/sme-loan-prescreener/</b> - the same
model runs client-side with exact probability parity against Python.</p>

<h2>9. Limitations and future work</h2>
<ul>
<li>Synthetic labels approximate, not replace, real credit committee behaviour.</li>
<li>Binary committee target collapses a spectrum; three bands recover the middle ground for human review.</li>
<li>Future: calibration validation curves, PR-AUC threshold optimisation, SHAP factor attribution as a secondary explainer, and lender-side A/B evaluation.</li>
</ul>
</body></html>"""

SLIDE_CSS = f"""
@page {{ size: A4 landscape; margin: 0; }}
body {{ font-family: sans-serif; color: {INK}; margin: 0; }}
.slide {{ width: 297mm; height: 210mm; box-sizing: border-box; padding: 12mm 16mm; page-break-after: always; position: relative; }}
.slide:last-child {{ page-break-after: auto; }}
.title-slide {{ background: {DARK}; color: #fff; }}
.title-slide h1 {{ color: #fff; }}
.kicker {{ color: #9fd6b2; font-size: 10pt; letter-spacing: 2.5pt; text-transform: uppercase; }}
h1 {{ color: {DARK}; font-size: 24pt; margin: 0 0 4mm; }}
h2 {{ color: {GREEN}; font-size: 15pt; margin: 0 0 3mm; }}
p, li {{ font-size: 11pt; line-height: 1.45; }}
ul {{ margin: 2mm 0; padding-left: 6mm; }}
table {{ border-collapse: collapse; width: 100%; margin: 3mm 0; font-size: 10.5pt; }}
th {{ background: {DARK}; color: #fff; text-align: left; padding: 2.2mm 3mm; }}
td {{ border: 0.6pt solid #c9d8ce; padding: 2mm 3mm; }}
.footer {{ position: absolute; bottom: 7mm; left: 16mm; right: 16mm; font-size: 8pt; color: {MUTED};
          border-top: 0.8pt solid #c9d8ce; padding-top: 2mm; }}
.pill {{ display: inline-block; padding: 1.5mm 4mm; border-radius: 999pt; font-weight: bold; font-size: 10.5pt; color: #fff; margin-right: 2mm; }}
.grid2 {{ display: table; width: 100%; }} .col {{ display: table-cell; width: 50%; vertical-align: top; padding-right: 6mm; }}
.callout {{ background: {LIGHT}; border-left: 3pt solid {GREEN}; padding: 3mm 5mm; margin: 3mm 0; }}
.big {{ font-size: 26pt; font-weight: bold; color: {GREEN}; }}
img.diagram {{ width: 78%; display: block; margin: 3mm auto; }}
"""

def footer(n):
    return f'<div class="footer">SME Loan Pre-Screener &middot; Explainable AI for SA SMEs &middot; slide {n} / 11 &middot; tshepisofrominnostation.github.io/sme-loan-prescreener</div>'

SLIDES_HTML = f"""<html><head><meta charset="utf-8"><style>{SLIDE_CSS}</style></head><body>

<div class="slide title-slide">
  <p class="kicker">Portfolio &middot; Fintech AI</p>
  <h1 style="font-size:34pt; margin-top:8mm;">SME Loan Pre-Screener</h1>
  <p style="font-size:15pt; color:#cfeeda;">An explainable, fair AI co-pilot for SME working-capital assessment</p>
  <p style="position:absolute; bottom:12mm; font-size:10.5pt; color:#9fd6b2;">Tshepiso Freddy Thosago &middot; October 2026 &middot;
  Synthetic data &middot; Indicative only - human judgment is final</p>
</div>

<div class="slide"><h1>1. The problem</h1>
<ul style="font-size:13pt;">
<li>SMEs are the heartbeat of the SA economy - and hit a brick wall seeking working capital.</li>
<li>Credit processes are slow and opaque; good businesses wait weeks for an answer they cannot reason about.</li>
<li>Credit officers drown in manual first-pass screening of clearly strong or clearly weak applications.</li>
</ul>
<div class="callout" style="margin-top:8mm;"><b>Vision:</b> not an automated decision-maker - an intelligent co-pilot that
accelerates evaluation, surfaces critical insights, and keeps a human in charge of every decision.</div>
{footer(2)}</div>

<div class="slide"><h1>2. The solution in one screen</h1>
<div class="grid2"><div class="col">
<p>Seven lightweight inputs (revenue, costs, debt, years trading, sector, loan amount, prior history) produce:</p>
<p><span class="pill" style="background:{GREEN};">LIKELY TO QUALIFY</span>
<span class="pill" style="background:{AMBER};">NEEDS REVIEW</span>
<span class="pill" style="background:{RED};">UNLIKELY TO QUALIFY</span></p>
<ul><li>A calibrated probability and confidence score</li>
<li>Plain-language factors - why the score moved, up or down</li>
<li>A prominent disclaimer: indicative only, human judgment final</li></ul>
</div><div class="col">
<p class="big">3 bands.</p><p class="big">7 inputs.</p><p class="big">0 black boxes.</p>
<p style="color:{MUTED};">Deployed model: logistic regression over affordability ratios - every coefficient is a readable lending rule.</p>
</div></div>
{footer(3)}</div>

<div class="slide"><h1>3. Architecture - intake to outcome</h1>
<img class="diagram" src="data:image/png;base64,{diagram_b64}">
{footer(4)}</div>

<div class="slide"><h1>4. Data foundation (synthetic, transparent)</h1>
<ul>
<li><b>800 seeded synthetic applications</b> - no real customer data anywhere.</li>
<li>Labels engineered from a <b>simulated credit committee</b>: an affordability score (coverage, margin, 30% debt ceiling,
trading years, history, sector premium) plus noise decides approval. Approval rate {ev['dataset']['approval_rate']:.0%}.</li>
<li>Debt-free applicants imputed to 0 explicitly - every data decision documented in code.</li>
<li>Engineered ratios: instalment coverage, operating margin, debt burden, loan-to-revenue.</li>
</ul>
{footer(5)}</div>

<div class="slide"><h1>5. Models compared (held-out test set, n={ev['dataset']['test']})</h1>
<table><tr><th>Model</th><th>Precision</th><th>Recall</th><th>F1</th><th>ROC-AUC</th></tr>
<tr><td>Rule-based baseline (30% ceiling, 2x coverage)</td><td>{base_['precision']:.2f}</td><td>{base_['recall']:.2f}</td><td>{base_['f1']:.2f}</td><td>-</td></tr>
<tr style="background:{LIGHT};"><td><b>Logistic regression - deployed</b></td><td><b>{lr['precision']:.2f}</b></td><td><b>{lr['recall']:.2f}</b></td><td><b>{lr['f1']:.2f}</b></td><td><b>{lr['roc_auc']:.2f}</b></td></tr>
<tr><td>Gradient boosting (comparison)</td><td>{gb['precision']:.2f}</td><td>{gb['recall']:.2f}</td><td>{gb['f1']:.2f}</td><td>{gb['roc_auc']:.2f}</td></tr>
</table>
<p><b>Why LR wins here:</b> same-class discrimination as the booster, fully auditable, and the committee can challenge any factor.
Recall is deliberately protected (balanced weights) - a false negative discourages a fundable SME; a false positive costs underwriting time.</p>
{footer(6)}</div>

<div class="slide"><h1>6. Explainability - factors in credit-officer language</h1>
<p style="font-size:12pt;">Example output for a risky application:</p>
<ul style="font-size:12pt;">
<li style="color:{RED};">&#9660; Existing debt repayments consume 40% of monthly revenue, above the 30% affordability ceiling.</li>
<li style="color:{RED};">&#9660; Free cash of R0 does not cover the estimated new instalment of R20,833.</li>
<li style="color:{RED};">&#9660; The requested R600,000 is about 6.7 months of revenue - a large ask relative to turnover.</li>
<li style="color:{GREEN};">&#9650; Operating margin of 25% is above the 15% healthy-business bar.</li>
</ul>
<div class="callout">No unexplained scores: every displayed factor maps to a ratio the model actually consumed,
and a human officer can challenge any of them in review.</div>
{footer(7)}</div>

<div class="slide"><h1>7. Fairness audit - FN rate among approvable, by sector</h1>
<table><tr><th>Sector</th><th>N</th><th>FN rate</th></tr>
{''.join(f"<tr><td>{k.title()}</td><td>{v['n']}</td><td>{v['fn_rate_among_approvable']:.2f}</td></tr>" for k, v in sectors)}
</table>
<p><b>Disclosure:</b> the construction/hospitality gap is engineered into the synthetic committee so the audit
has a real disparity to surface. The production deliverable is the <b>monitoring pipeline</b>: monthly FN-rate gaps
by sector and size, with persistent gaps triggering model review. Protected characteristics are excluded by design;
revenue size is audited as a proxy-risk feature.</p>
{footer(8)}</div>

<div class="slide"><h1>8. Responsible AI framing</h1>
<ul>
<li><b>Indicative only</b> - persistent banner on every interface; the tool never approves or declines.</li>
<li><b>NCA and POPIA</b> - production would require National Credit Act registration and POPIA compliance; declared out of scope.</li>
<li><b>Proxy features</b> - sector and business size retained for explainability but flagged and audited.</li>
<li><b>Simplicity bar</b> - a transparent, auditable model beats an uninterpretable one in financial services.</li>
</ul>
{footer(9)}</div>

<div class="slide"><h1>9. Live demo flow (5 minutes)</h1>
<ul style="font-size:12pt;">
<li>Open <b>tshepisofrominnostation.github.io/sme-loan-prescreener</b> - enter a healthy retailer: <b>LIKELY</b> with coverage and margin factors.</li>
<li>Raise debt repayments to 40% of revenue - watch the band flip and the debt-ceiling factor appear.</li>
<li>Push years trading to 1 and loan to 6.7 months of revenue - <b>UNLIKELY</b>, factors explain exactly why.</li>
<li>Find the borderline case that lands in <b>NEEDS REVIEW</b> - that is the human-judgment band working as designed.</li>
<li>Full walkthrough script in docs/DEMO_SCRIPT.md; Python/JS probability parity verified to 4 decimals.</li>
</ul>
{footer(10)}</div>

<div class="slide"><h1>10. Limitations and future work</h1>
<div class="grid2"><div class="col"><h2>Known limits</h2>
<ul><li>Synthetic committee labels approximate real credit behaviour</li>
<li>Binary target, three bands recover the middle ground</li>
<li>Prototype compliance posture, not a registered credit provider</li></ul>
</div><div class="col"><h2>Next</h2>
<ul><li>Calibration curves and PR-AUC threshold tuning</li>
<li>SHAP as a secondary explainer alongside ratio factors</li>
<li>Lender-side A/B evaluation and drift monitoring</li></ul>
</div></div>
<p style="margin-top:6mm;"><b>Repo:</b> github.com/tshepisofrominnostation/sme-loan-prescreener &middot;
<b>Live:</b> tshepisofrominnostation.github.io/sme-loan-prescreener</p>
{footer(11)}</div>
</body></html>"""


(ROOT / "docs/PROJECT_DOCUMENTATION.pdf").write_bytes(HTML(string=DOC_HTML).write_pdf())
(ROOT / "docs/PRESENTATION.pdf").write_bytes(HTML(string=SLIDES_HTML).write_pdf())
print("wrote docs/PROJECT_DOCUMENTATION.pdf and docs/PRESENTATION.pdf")
