"""
Statistical validation for churn-model research claims.

This module is intentionally additive: it consumes artifacts produced by the
existing pipeline and writes research summaries under outputs/. It does not
change train/test splits, feature engineering, tuning, or model training.
"""

import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)


def _read_csv(name):
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def _load_model(name):
    path = os.path.join(MODEL_DIR, name)
    if not os.path.exists(path):
        return None
    return joblib.load(path)


def _predict_proba(model, X):
    if model is None or X is None:
        return None
    try:
        return np.asarray(model.predict_proba(X)[:, 1], dtype=float)
    except Exception:
        return None


def _bootstrap_auc(y_true, p_ref, p_cmp, n_boot=500, seed=42):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true, dtype=int)
    p_ref = np.asarray(p_ref, dtype=float)
    p_cmp = np.asarray(p_cmp, dtype=float)

    ref_scores, cmp_scores, diffs = [], [], []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        ref_auc = roc_auc_score(y_true[idx], p_ref[idx])
        cmp_auc = roc_auc_score(y_true[idx], p_cmp[idx])
        ref_scores.append(ref_auc)
        cmp_scores.append(cmp_auc)
        diffs.append(cmp_auc - ref_auc)

    if not diffs:
        return None

    return {
        "ref_mean": float(np.mean(ref_scores)),
        "cmp_mean": float(np.mean(cmp_scores)),
        "diff": float(np.mean(diffs)),
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
    }


def _practical_label(diff, ci_low, ci_high, threshold=0.005):
    if ci_low <= 0 <= ci_high:
        return "Not statistically distinguishable"
    if abs(diff) < threshold:
        return "Statistically detectable but practically negligible"
    return "Practically meaningful"


def _occlude_columns(X, columns):
    X_occ = X.copy()
    for col in columns:
        if col in X_occ.columns:
            X_occ[col] = 0
    return X_occ


def run_statistical_validation(n_boot=500):
    """Generate outputs/statistical_validation.csv and .md."""
    X_test = _read_csv("X_test.csv")
    X_test_clust = _read_csv("X_test_clust.csv")
    X_test_dist = _read_csv("X_test_dist.csv")
    y_test_df = _read_csv("y_test.csv")

    rows = []
    if y_test_df is None:
        _write_empty_outputs("Missing y_test.csv; statistical validation skipped.")
        return pd.DataFrame()

    y_test = y_test_df.squeeze().astype(int).values

    xgb_base = _load_model("xgb_model.pkl")
    xgb_clust = _load_model("xgb_clust_model.pkl")
    lr_base = _load_model("lr_base_model.pkl")
    lr_dist = _load_model("lr_dist_model.pkl")

    base_prob = _predict_proba(xgb_base, X_test)
    clust_prob = _predict_proba(xgb_clust, X_test_clust)

    if base_prob is not None and X_test is not None:
        rfm_cols = [c for c in X_test.columns if c.startswith("rfm_")]
        if rfm_cols:
            without_rfm_prob = _predict_proba(xgb_base, _occlude_columns(X_test, rfm_cols))
            rows.append(_build_row(
                "With vs Without RFM",
                "XGBoost baseline compared with an RFM-occluded test matrix",
                y_test,
                without_rfm_prob,
                base_prob,
                n_boot,
            ))

    if base_prob is not None and clust_prob is not None:
        rows.append(_build_row(
            "With vs Without Cluster Features",
            "XGBoost with k-means cluster feature compared with base XGBoost",
            y_test,
            base_prob,
            clust_prob,
            n_boot,
        ))

    lr_base_prob = _predict_proba(lr_base, X_test)
    lr_dist_prob = _predict_proba(lr_dist, X_test_dist)
    if lr_base_prob is not None and lr_dist_prob is not None:
        rows.append(_build_row(
            "Leakage Removal Experiment",
            "Distance-feature logistic model compared with base logistic model",
            y_test,
            lr_dist_prob,
            lr_base_prob,
            n_boot,
        ))

    if not rows:
        _write_empty_outputs("No compatible model artifacts were available.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "statistical_validation.csv"), index=False)
    _write_summary(df)
    print("[Research] Saved: statistical_validation.csv, statistical_validation_summary.md")
    return df


def _build_row(experiment, method, y_test, p_ref, p_cmp, n_boot):
    if p_ref is None or p_cmp is None:
        return {
            "Experiment": experiment,
            "Method": method,
            "Mean ROC-AUC": np.nan,
            "95% CI": "Unavailable",
            "Difference": np.nan,
            "Practical significance": "Unavailable",
        }

    stats = _bootstrap_auc(y_test, p_ref, p_cmp, n_boot=n_boot)
    if stats is None:
        return {
            "Experiment": experiment,
            "Method": method,
            "Mean ROC-AUC": np.nan,
            "95% CI": "Unavailable",
            "Difference": np.nan,
            "Practical significance": "Insufficient class variation in resamples",
        }

    return {
        "Experiment": experiment,
        "Method": method,
        "Mean ROC-AUC": round(stats["cmp_mean"], 4),
        "95% CI": f"[{stats['ci_low']:.4f}, {stats['ci_high']:.4f}]",
        "Difference": round(stats["diff"], 4),
        "Practical significance": _practical_label(
            stats["diff"], stats["ci_low"], stats["ci_high"]
        ),
    }


def _write_empty_outputs(reason):
    df = pd.DataFrame([{
        "Experiment": "Unavailable",
        "Method": reason,
        "Mean ROC-AUC": np.nan,
        "95% CI": "Unavailable",
        "Difference": np.nan,
        "Practical significance": "Unavailable",
    }])
    df.to_csv(os.path.join(OUT_DIR, "statistical_validation.csv"), index=False)
    with open(os.path.join(OUT_DIR, "statistical_validation_summary.md"), "w", encoding="utf-8") as f:
        f.write("# Statistical Validation Summary\n\n")
        f.write(reason + "\n")


def _write_summary(df):
    lines = [
        "# Statistical Validation Summary",
        "",
        "Bootstrap resampling was used to estimate whether observed ROC-AUC differences are stable enough to support research claims. Positive differences favor the enhanced condition named in the experiment.",
        "",
    ]
    for _, row in df.iterrows():
        lines.extend([
            f"## {row['Experiment']}",
            f"- Method: {row['Method']}",
            f"- Mean ROC-AUC: {row['Mean ROC-AUC']}",
            f"- Difference: {row['Difference']} with 95% CI {row['95% CI']}",
            f"- Interpretation: {row['Practical significance']}",
            "",
        ])
    lines.append("Small ROC-AUC changes should be interpreted alongside segmentation value, fairness, cost reduction, and operational usefulness.")
    with open(os.path.join(OUT_DIR, "statistical_validation_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    run_statistical_validation()
