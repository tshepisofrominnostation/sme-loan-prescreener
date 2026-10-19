"""Train, evaluate and compare models; produce the fairness audit.

Outputs:
  models/logreg_model.joblib      deployed interpretable pipeline
  models/gbm_model.joblib         comparison model
  models/model_export.json        scaler + coefficients for the web prototype
  reports/evaluation.json         full metrics, confusion matrices, FP/FN costs
  reports/fairness.json           false-negative/positive disparity audit

Run:  python -m src.train
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

from src.features import (FEATURE_ORDER, build_gbm_pipeline,
                          build_logistic_pipeline, engineer_features)

ROOT = Path(__file__).resolve().parent.parent
MODELS, REPORTS = ROOT / "models", ROOT / "reports"
BANDS = ["Likely to qualify", "Needs review", "Unlikely to qualify"]
BINARY = {"Likely to qualify": 1, "Needs review": 0, "Unlikely to qualify": 0}


def _binary_metrics(y_true, y_prob, threshold: float = 0.5):
    """Precision/recall/F1/AUC + confusion matrix at a decision threshold."""
    pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 3),
        "precision": round(float(precision_score(y_true, pred)), 3),
        "recall": round(float(recall_score(y_true, pred)), 3),
        "f1": round(float(f1_score(y_true, pred)), 3),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def _false_positive_false_negative_cost(cm: dict) -> dict:
    """Business-cost analysis of FP vs FN in a lending context (brief 3.B)."""
    return {
        "false_positives": {
            "count": cm["fp"],
            "meaning": "SME pre-screened as LIKELY to qualify but the committee would decline.",
            "business_cost": "Wasted underwriting effort on applications that fail detailed "
                             "review; risk of adverse selection if the pre-screen is trusted "
                             "too much. Cost falls on the LENDER.",
        },
        "false_negatives": {
            "count": cm["fn"],
            "meaning": "Fundable SME pre-screened as unlikely/negative but the committee would approve.",
            "business_cost": "Lost lending revenue and lost growth for a creditworthy SA SME; "
                              "the harm falls on the APPLICANT, so this error carries both a "
                              "commercial AND a fairness cost. Mitigation: keep the middle band "
                              "wide (Needs review) so borderline fundable businesses are never "
                              "auto-discouraged, and always route a human decision.",
        },
    }


def _fairness_audit(df_test: pd.DataFrame, y_true, y_prob) -> dict:
    """Disparity check across sectors and business sizes (brief 3.D)."""
    pred = (y_prob >= 0.5).astype(int)
    sizes = pd.qcut(df_test["Monthly_Revenue"], 2, labels=["smaller half", "larger half"])

    def _group_stats(group_values, y_true_arr, pred_arr):
        fn = int(((pred_arr == 0) & (y_true_arr == 1)).sum())
        fp = int(((pred_arr == 1) & (y_true_arr == 0)).sum())
        pos = int((y_true_arr == 1).sum())
        return {
            "n": int(len(y_true_arr)),
            "approval_rate_truth": round(float(y_true_arr.mean()), 3),
            "false_negatives": fn,
            "fn_rate_among_approvable": round(fn / pos, 3) if pos else None,
            "false_positives": fp,
        }

    audit = {"note": "Synthetic data: the sector gap below is engineered into the "
                     "generator (construction/hospitality risk premium) so the audit "
                     "has a real disparity to surface; see README disclosure.",
             "by_sector": {}, "by_size": {}}
    for sector, group in df_test.groupby("Sector"):
        p = df_test.index.get_indexer(group.index)
        audit["by_sector"][str(sector)] = _group_stats(sector, y_true[p], pred[p])
    for size, group in df_test.groupby(sizes):
        p = df_test.index.get_indexer(group.index)
        audit["by_size"][str(size)] = _group_stats(size, y_true[p], pred[p])
    return audit


def train() -> None:
    MODELS.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)

    df = pd.read_csv(ROOT / "data" / "sme_applications.csv")
    X = engineer_features(df)[FEATURE_ORDER]
    y = df["Approved"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42)
    test_rows = df.loc[X_test.index]

    # ---- baseline ----
    from src.rules_baseline import score_baseline
    base_bands = score_baseline(test_rows)
    base_binary = np.array([BINARY[b] for b in base_bands])
    tn, fp, fn, tp = confusion_matrix(y_test, base_binary).ravel()
    baseline = {
        "description": "Hand-written affordability rules (no ML)",
        "precision": round(float(precision_score(y_test, base_binary)), 3),
        "recall": round(float(recall_score(y_test, base_binary)), 3),
        "f1": round(float(f1_score(y_test, base_binary)), 3),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }

    # ---- model A: logistic regression (deployed) ----
    lr = build_logistic_pipeline().fit(X_train, y_train)
    lr_prob = lr.predict_proba(X_test)[:, 1]

    # ---- model B: gradient boosting (comparison) ----
    gbm = build_gbm_pipeline().fit(X_train, y_train)
    gbm_prob = gbm.predict_proba(X_test)[:, 1]

    lr_metrics = _binary_metrics(y_test, lr_prob)
    gbm_metrics = _binary_metrics(y_test, gbm_prob)

    report = {
        "dataset": {"rows": len(df), "train": len(X_train), "test": len(X_test),
                     "approval_rate": round(float(y.mean()), 3)},
        "baseline_rules": baseline,
        "logistic_regression": lr_metrics,
        "gradient_boosting": gbm_metrics,
        "fp_fn_cost_analysis": _false_positive_false_negative_cost(lr_metrics["confusion"]),
        "selected_model": "logistic_regression",
        "selection_rationale": (
            f"GBM AUC {gbm_metrics['roc_auc']} vs LR AUC {lr_metrics['roc_auc']}: "
            "comparable statistical performance, but LR offers auditable, "
            "exportable coefficients and monotone reasoning a credit officer can "
            "verify by hand — the brief's quality bar favours it."),
    }
    (REPORTS / "evaluation.json").write_text(json.dumps(report, indent=2))

    # ---- fairness audit on the deployed model ----
    audit = _fairness_audit(test_rows, y_test.values, lr_prob)
    (REPORTS / "fairness.json").write_text(json.dumps(audit, indent=2))

    # ---- persist models + web export ----
    joblib.dump(lr, MODELS / "logreg_model.joblib")
    joblib.dump(gbm, MODELS / "gbm_model.joblib")
    scaler = lr.named_steps["scale"]
    export = {
        "features": FEATURE_ORDER,
        "scaler_mean": [round(float(m), 4) for m in scaler.mean_],
        "scaler_scale": [round(float(s), 4) for s in scaler.scale_],
        "intercept": round(float(lr.named_steps["model"].intercept_[0]), 4),
        "coefficients": {f: round(float(c), 4)
                         for f, c in zip(FEATURE_ORDER, lr.named_steps["model"].coef_[0])},
    }
    (MODELS / "model_export.json").write_text(json.dumps(export, indent=2))

    # ---- console summary ----
    print(json.dumps(report, indent=2)[:1400])
    print("\nFairness audit (FN rate among approvable, by sector):")
    for s, v in audit["by_sector"].items():
        print(f"  {s:15s} n={v['n']:3d} fn={v['false_negatives']:2d} "
              f"({v['fn_rate_among_approvable']})")
    print("\nWeb export:", export["coefficients"])


if __name__ == "__main__":
    train()
