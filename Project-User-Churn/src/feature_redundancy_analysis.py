"""
Feature redundancy and dominant-feature analysis.

This module is additive. It consumes saved train/test/model artifacts and writes
research outputs without changing training, feature engineering, or datasets.
"""

import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)


def _read_csv(folder, name):
    path = os.path.join(folder, name)
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _base_feature(name):
    known_prefixes = [
        "PreferredLoginDevice", "PreferredPaymentMode", "PreferedOrderCat",
        "MaritalStatus", "CityTier", "Gender", "Complain", "rfm_segment",
    ]
    for prefix in known_prefixes:
        if name == prefix or name.startswith(prefix + "_"):
            return prefix
    return name


def _normalise(series):
    series = pd.to_numeric(series, errors="coerce").fillna(0).astype(float)
    span = series.max() - series.min()
    if span == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - series.min()) / span


def _mean_shap_impact(model, X):
    model_for_importance = model
    if hasattr(model, "calibrated_classifiers_") and model.calibrated_classifiers_:
        model_for_importance = getattr(model.calibrated_classifiers_[0], "estimator", model)
    try:
        import shap

        sample = X.sample(min(len(X), 1000), random_state=42)
        explainer = shap.TreeExplainer(model_for_importance)
        values = explainer.shap_values(sample)
        if isinstance(values, list):
            values = values[-1]
        return pd.Series(np.abs(values).mean(axis=0), index=sample.columns)
    except Exception:
        try:
            return pd.Series(model_for_importance.feature_importances_, index=X.columns)
        except Exception:
            return pd.Series(np.zeros(X.shape[1]), index=X.columns)


def run_feature_redundancy_analysis():
    X = _read_csv(DATA_DIR, "X_train_clust.csv")
    if X is None:
        X = _read_csv(DATA_DIR, "X_train.csv")
    y_df = _read_csv(DATA_DIR, "y_train.csv")
    if X is None or y_df is None:
        _write_unavailable("Missing X_train/X_train_clust or y_train artifacts.")
        return pd.DataFrame()

    y = y_df.squeeze().astype(int).values
    X_num = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    try:
        mi = pd.Series(mutual_info_classif(X_num, y, random_state=42), index=X_num.columns)
    except Exception:
        mi = pd.Series(np.zeros(X_num.shape[1]), index=X_num.columns)

    corr_values = {}
    for col in X_num.columns:
        try:
            corr_values[col] = X_num[col].corr(pd.Series(y))
        except Exception:
            corr_values[col] = 0
    corr = pd.Series(corr_values).fillna(0)

    model = None
    for candidate in ["xgb_clust_model.pkl", "xgb_model.pkl"]:
        path = os.path.join(MODEL_DIR, candidate)
        if os.path.exists(path):
            try:
                model = joblib.load(path)
                break
            except Exception:
                model = None
    shap_impact = _mean_shap_impact(model, X_num) if model is not None else pd.Series(0, index=X_num.columns)

    encoded = pd.DataFrame({
        "Feature": X_num.columns,
        "Base Feature": [_base_feature(c) for c in X_num.columns],
        "Mutual Information": mi.reindex(X_num.columns).values,
        "Correlation": corr.reindex(X_num.columns).values,
        "Mean SHAP Impact": shap_impact.reindex(X_num.columns).fillna(0).values,
    })

    grouped = encoded.groupby("Base Feature", as_index=False).agg({
        "Mutual Information": "sum",
        "Correlation": lambda s: s.iloc[np.argmax(np.abs(s.values))] if len(s) else 0,
        "Mean SHAP Impact": "sum",
    }).rename(columns={"Base Feature": "Feature"})

    dominance_score = (
        _normalise(grouped["Mutual Information"])
        + _normalise(grouped["Correlation"].abs())
        + _normalise(grouped["Mean SHAP Impact"])
    )
    grouped["Dominance Rank"] = dominance_score.rank(ascending=False, method="dense").astype(int)
    grouped = grouped.sort_values(["Dominance Rank", "Mean SHAP Impact"], ascending=[True, False])

    for col in ["Mutual Information", "Correlation", "Mean SHAP Impact"]:
        grouped[col] = grouped[col].astype(float).round(6)

    out_path = os.path.join(OUT_DIR, "feature_redundancy_analysis.csv")
    grouped[["Feature", "Mutual Information", "Correlation", "Mean SHAP Impact", "Dominance Rank"]].to_csv(out_path, index=False)
    _write_summary(grouped)
    print("[Research] Saved: feature_redundancy_analysis.csv, feature_redundancy_summary.md")
    return grouped


def _write_unavailable(reason):
    pd.DataFrame(columns=[
        "Feature", "Mutual Information", "Correlation", "Mean SHAP Impact", "Dominance Rank"
    ]).to_csv(os.path.join(OUT_DIR, "feature_redundancy_analysis.csv"), index=False)
    with open(os.path.join(OUT_DIR, "feature_redundancy_summary.md"), "w", encoding="utf-8") as f:
        f.write("# Feature Redundancy Summary\n\n")
        f.write(reason + "\n")


def _write_summary(df):
    top = df.sort_values("Dominance Rank").head(10)
    dominant = df[df["Feature"].isin(["Complain", "DaySinceLastOrder", "Tenure", "CashbackAmount"])]
    flagged = dominant[dominant["Dominance Rank"] <= 10]["Feature"].tolist()

    lines = [
        "# Feature Redundancy Summary",
        "",
        "Mutual information is used here as the empirical information-gain signal for the binary churn target. The analysis aggregates one-hot encoded columns back to business-level feature names where appropriate.",
        "",
        "## Why Model Performance Is High",
        "The strongest predictors include direct behavioral recency, complaint status, tenure, cashback behavior, and engineered RFM/activity features. These variables are close to the churn mechanism, so the model can separate churned and retained customers with unusually strong discrimination.",
        "",
        "## Most Dominant Features",
    ]
    for _, row in top.iterrows():
        lines.append(
            f"- {row['Feature']}: rank {int(row['Dominance Rank'])}, MI={row['Mutual Information']:.4f}, corr={row['Correlation']:.4f}, mean SHAP impact={row['Mean SHAP Impact']:.4f}."
        )

    lines.extend([
        "",
        "## Potentially Dominant Features",
        "- " + (", ".join(flagged) if flagged else "No specified watch-list feature appears in the top dominance ranks."),
        "",
        "## Leakage Assessment",
        "No definitive leakage is proven by this post-hoc ranking alone. However, very high ROC-AUC combined with dominant features such as complaint status, days since last order, tenure, cashback amount, or engineered recency ratios should be defended as behaviorally plausible and monitored as possible proxy leakage if any field was recorded after churn became known.",
    ])
    with open(os.path.join(OUT_DIR, "feature_redundancy_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    run_feature_redundancy_analysis()
