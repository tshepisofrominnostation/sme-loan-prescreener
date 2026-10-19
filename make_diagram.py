"""Generate the one-page workflow diagram (docs/workflow_diagram.png)."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

NAVY, PANEL, CYAN, ORANGE, MUTED = "#f2f6f3", "#ffffff", "#009639", "#e08a00", "#51685c"

fig, ax = plt.subplots(figsize=(11, 7), dpi=160)
fig.patch.set_facecolor(NAVY); ax.set_facecolor(NAVY)
ax.set_xlim(0, 100); ax.set_ylim(0, 70); ax.axis("off")

def box(x, y, w, h, title, lines, edge=CYAN, fill=PANEL):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6",
                                fc=fill, ec=edge, lw=1.6))
    ax.text(x + w/2, y + h - 2.4, title, ha="center", va="top",
            fontsize=10.5, fontweight="bold", color="#0b4d24")
    for i, line in enumerate(lines):
        ax.text(x + w/2, y + h - 5.6 - i*3.1, line, ha="center", va="top",
                fontsize=8.2, color="#2f4237")

def arrow(x1, y1, x2, y2, color=CYAN):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=16, lw=1.6, color=color))

ax.text(50, 67, "SME Loan Pre-Screening — Application Workflow",
        ha="center", fontsize=15, fontweight="bold", color="#0b4d24")
ax.text(50, 63.2, "Indicative pre-screen only — a human credit officer makes every final decision",
        ha="center", fontsize=9, color=ORANGE)

box(2, 44, 22, 14, "1 · INTAKE", ["Credit officer enters 7 fields:",
                                  "revenue, costs, debt, years,",
                                  "sector, loan amount, history"])
box(30, 44, 22, 14, "2 · FEATURE ENGINEERING", ["Four auditable ratios:",
                                                "coverage, margin,",
                                                "debt burden, loan-to-revenue"])
box(58, 44, 22, 14, "3 · LOGISTIC MODEL", ["Scaled logistic regression;",
                                          "probability of committee",
                                          "approval (calibrated bands)"])
box(58, 24, 22, 14, "4 · THREE-BAND OUTPUT", ["LIKELY to qualify   (p > 65%)",
                                              "NEEDS REVIEW   (35-65%)",
                                              "UNLIKELY   (p < 35%)",
                                              "+ confidence score"])
box(30, 24, 22, 14, "5 · EXPLANATIONS", ["Plain-language factors that",
                                         "raised or lowered the score,",
                                         "each mapped to a model input"])
box(2, 24, 22, 14, "6 · HUMAN DECISION", ["Credit officer reviews the",
                                          "factors and decides.",
                                          "The tool never approves",
                                          "or declines loans"])
box(2, 4, 96, 12, "GOVERNANCE LAYER (applies at every step)", [
    "NCA / POPIA disclaimer · fairness audit (FN rate by sector & size) · proxy-feature review",
    "intercept and coefficients published for audit · no PII stored — inputs are business metrics only"],
    edge=ORANGE)

arrow(24.6, 51, 29.4, 51)           # intake -> features
arrow(52.6, 51, 57.4, 51)           # features -> model
arrow(69, 43.4, 69, 38.6)           # model -> bands
arrow(57.4, 31, 52.6, 31)           # bands -> explanations
arrow(29.4, 31, 24.6, 31)           # explanations -> human decision
plt.savefig("docs/workflow_diagram.png", facecolor=NAVY, bbox_inches="tight")
print("diagram written")
