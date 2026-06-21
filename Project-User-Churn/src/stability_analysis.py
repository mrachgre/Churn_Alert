"""
Repeated stratified K-fold stability analysis for existing feature artifacts.
"""

import os
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)


def _read_csv(name):
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _models():
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=42, class_weight="balanced", n_jobs=-1
        ),
    }
    try:
        from xgboost import XGBClassifier

        models["XGBoost"] = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
    except Exception:
        pass
    return models


def run_stability_analysis():
    X = _read_csv("X_train_clust.csv")
    if X is None:
        X = _read_csv("X_train.csv")
    y_df = _read_csv("y_train.csv")
    if X is None or y_df is None:
        return _write_unavailable("Missing X_train/X_train_clust or y_train artifacts.")

    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
    y = y_df.squeeze().astype(int).values
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)

    rows = []
    for name, model in _models().items():
        accs, f1s, aucs = [], [], []
        for train_idx, test_idx in cv.split(X, y):
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]
            model.fit(X_tr, y_tr)
            pred = model.predict(X_te)
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_te)[:, 1]
            else:
                proba = pred
            accs.append(accuracy_score(y_te, pred))
            f1s.append(f1_score(y_te, pred, zero_division=0))
            aucs.append(roc_auc_score(y_te, proba))

        rows.append({
            "Model": name,
            "Mean Accuracy": round(float(np.mean(accs)), 4),
            "Std Accuracy": round(float(np.std(accs)), 4),
            "Mean F1": round(float(np.mean(f1s)), 4),
            "Std F1": round(float(np.std(f1s)), 4),
            "Mean ROC-AUC": round(float(np.mean(aucs)), 4),
            "Std ROC-AUC": round(float(np.std(aucs)), 4),
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "model_stability.csv"), index=False)
    _write_summary(df)
    print("[Research] Saved: model_stability.csv, model_stability_summary.md")
    return df


def _write_unavailable(reason):
    df = pd.DataFrame(columns=[
        "Model", "Mean Accuracy", "Std Accuracy", "Mean F1", "Std F1", "Mean ROC-AUC", "Std ROC-AUC"
    ])
    df.to_csv(os.path.join(OUT_DIR, "model_stability.csv"), index=False)
    with open(os.path.join(OUT_DIR, "model_stability_summary.md"), "w", encoding="utf-8") as f:
        f.write("# Model Stability Summary\n\n")
        f.write(reason + "\n")
    return df


def _write_summary(df):
    best = df.sort_values("Mean ROC-AUC", ascending=False).iloc[0] if not df.empty else None
    lines = [
        "# Model Stability Summary",
        "",
        "Repeated Stratified K-Fold validation used 5 folds and 10 repetitions. This does not replace the original train/test split; it provides a robustness audit across resampled training folds.",
        "",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"- {row['Model']}: ROC-AUC = {row['Mean ROC-AUC']:.3f} +/- {row['Std ROC-AUC']:.3f}, F1 = {row['Mean F1']:.3f} +/- {row['Std F1']:.3f}."
        )
    if best is not None:
        lines.extend([
            "",
            f"Best mean discrimination is observed for {best['Model']}. Low standard deviation supports generalization confidence; larger variance would indicate dependence on a favorable split.",
        ])
    with open(os.path.join(OUT_DIR, "model_stability_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    run_stability_analysis()
