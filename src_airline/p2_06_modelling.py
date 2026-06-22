"""
p2_06_modelling.py  —  Step 6: 6-Model Comparison
===================================================
Models (mirror Churn Alert):
  XGB-Base : Optuna 20 trials + CalibratedClassifierCV + Youden threshold
  XGB-Clust: XGB-Base params + cluster_id feature
  RF-Clust : GridSearch(n_estimators, max_depth) + cluster_id
  DT-Clust : GridSearch(max_depth, min_samples_leaf) + cluster_id
  LR-Base  : GridSearch(C) + chi2 Base features
  LR-Dist  : GridSearch(C) + chi2 Dist features (+ KMeans distances)
"""
import pandas as pd
import numpy as np
import os, joblib, warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             average_precision_score, confusion_matrix,
                             roc_curve)
import xgboost as xgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ── Helpers ───────────────────────────────────────────────────────────────────
def youden_threshold(model, X_val, y_val):
    proba = model.predict_proba(X_val)[:, 1]
    fpr, tpr, thr = roc_curve(y_val, proba)
    j = tpr - fpr
    return float(thr[np.argmax(j)])


def evaluate(name, model, X_te, y_te, threshold=0.5):
    proba = model.predict_proba(X_te)[:, 1]
    y_pred = (proba >= threshold).astype(int)
    return {
        "Model":     name,
        "Threshold": round(threshold, 3),
        "Accuracy":  round(accuracy_score(y_te, y_pred), 4),
        "Precision": round(f1_score(y_te, y_pred, zero_division=0, pos_label=1,
                                    average=None)[0] if True else 0, 4),
        "Recall":    round(f1_score(y_te, y_pred, zero_division=0, pos_label=1,
                                    average=None)[0] if True else 0, 4),
        "F1":        round(f1_score(y_te, y_pred, average="binary",
                                    zero_division=0), 4),
        "ROC_AUC":   round(roc_auc_score(y_te, proba), 4),
        "PR_AUC":    round(average_precision_score(y_te, proba), 4),
    }


def _precision_recall_from_model(model, X_te, y_te, threshold):
    proba = model.predict_proba(X_te)[:, 1]
    y_pred = (proba >= threshold).astype(int)
    tp = ((y_pred == 1) & (y_te == 1)).sum()
    fp = ((y_pred == 1) & (y_te == 0)).sum()
    fn = ((y_pred == 0) & (y_te == 1)).sum()
    p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return round(p, 4), round(r, 4)


def run_step6(cfg, step5_result: dict, step5b_result: dict) -> dict:
    sep = "=" * 70
    print(f"\n{sep}\nSTEP 6: MODELLING — 6 MODELS\n{sep}")

    X_tr_base  = step5_result["X_train_base"]
    X_te_base  = step5_result["X_test_base"]
    X_tr_clust = step5_result["X_train_clust"]
    X_te_clust = step5_result["X_test_clust"]
    y_train    = step5_result["y_train"]
    y_test     = step5_result["y_test"]

    X_tr_lr_base = step5b_result["X_tr_lr_base"]
    X_te_lr_base = step5b_result["X_te_lr_base"]
    X_tr_lr_dist = step5b_result["X_tr_lr_dist"]
    X_te_lr_dist = step5b_result["X_te_lr_dist"]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.RANDOM_STATE)
    results = []
    models  = {}

    # ── XGB-Base ──────────────────────────────────────────────────────────────
    print(f"\n[6.1] XGB-Base — Optuna ({cfg.OPTUNA_TRIALS} trials, CV=5)...")

    def xgb_objective(trial):
        params = {
            "n_estimators":       trial.suggest_int("n_estimators", 100, 600),
            "max_depth":          trial.suggest_int("max_depth", 3, 9),
            "learning_rate":      trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":          trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":   trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight":   trial.suggest_int("min_child_weight", 1, 10),
            "random_state": cfg.RANDOM_STATE,
            "eval_metric": "logloss",
            "use_label_encoder": False,
            "verbosity": 0,
        }
        scores = []
        for tr_idx, va_idx in cv.split(X_tr_base, y_train):
            m = xgb.XGBClassifier(**params)
            m.fit(X_tr_base.iloc[tr_idx], y_train[tr_idx],
                  eval_set=[(X_tr_base.iloc[va_idx], y_train[va_idx])],
                  verbose=False)
            scores.append(roc_auc_score(y_train[va_idx],
                                        m.predict_proba(X_tr_base.iloc[va_idx])[:, 1]))
        return np.mean(scores)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=cfg.RANDOM_STATE))
    study.optimize(xgb_objective, n_trials=cfg.OPTUNA_TRIALS, show_progress_bar=False)
    best_params = study.best_params
    best_params.update({"random_state": cfg.RANDOM_STATE, "verbosity": 0,
                        "eval_metric": "logloss", "use_label_encoder": False})
    print(f"  Best params: {best_params}")
    print(f"  Best CV ROC-AUC: {study.best_value:.4f}")

    xgb_base_raw = xgb.XGBClassifier(**best_params)
    xgb_base_raw.fit(X_tr_base, y_train, verbose=False)
    xgb_base_cal = CalibratedClassifierCV(xgb_base_raw, cv=5, method="isotonic")
    xgb_base_cal.fit(X_tr_base, y_train)
    thr_base = youden_threshold(xgb_base_cal, X_tr_base, y_train)
    p, r = _precision_recall_from_model(xgb_base_cal, X_te_base, y_test, thr_base)
    row = evaluate("XGB-Base", xgb_base_cal, X_te_base, y_test, thr_base)
    row["Precision"] = p; row["Recall"] = r
    results.append(row); models["xgb_base"] = xgb_base_cal
    joblib.dump(xgb_base_cal, os.path.join(cfg.MODEL_DIR, cfg.MODEL_NAMES["xgb_base"]))
    print(f"  XGB-Base → Accuracy={row['Accuracy']}  F1={row['F1']}  ROC-AUC={row['ROC_AUC']}")

    # ── XGB-Clust ─────────────────────────────────────────────────────────────
    print(f"\n[6.2] XGB-Clust — reuse Optuna params + cluster_id...")
    xgb_clust_raw = xgb.XGBClassifier(**best_params)
    xgb_clust_raw.fit(X_tr_clust, y_train, verbose=False)
    xgb_clust_cal = CalibratedClassifierCV(xgb_clust_raw, cv=5, method="isotonic")
    xgb_clust_cal.fit(X_tr_clust, y_train)
    thr_clust = youden_threshold(xgb_clust_cal, X_tr_clust, y_train)
    p, r = _precision_recall_from_model(xgb_clust_cal, X_te_clust, y_test, thr_clust)
    row = evaluate("XGB-Clust", xgb_clust_cal, X_te_clust, y_test, thr_clust)
    row["Precision"] = p; row["Recall"] = r
    results.append(row); models["xgb_clust"] = xgb_clust_cal
    joblib.dump(xgb_clust_cal, os.path.join(cfg.MODEL_DIR, cfg.MODEL_NAMES["xgb_clust"]))
    print(f"  XGB-Clust → Accuracy={row['Accuracy']}  F1={row['F1']}  ROC-AUC={row['ROC_AUC']}")

    # ── RF-Clust ──────────────────────────────────────────────────────────────
    print(f"\n[6.3] RF-Clust — GridSearchCV...")
    rf_params = {"n_estimators": [100, 200], "max_depth": [6, 9, None],
                 "random_state": [cfg.RANDOM_STATE]}
    rf = GridSearchCV(RandomForestClassifier(), rf_params, cv=cv,
                      scoring="roc_auc", n_jobs=-1)
    rf.fit(X_tr_clust, y_train)
    thr_rf = youden_threshold(rf.best_estimator_, X_tr_clust, y_train)
    p, r = _precision_recall_from_model(rf.best_estimator_, X_te_clust, y_test, thr_rf)
    row = evaluate("RF-Clust", rf.best_estimator_, X_te_clust, y_test, thr_rf)
    row["Precision"] = p; row["Recall"] = r
    results.append(row); models["rf_clust"] = rf.best_estimator_
    joblib.dump(rf.best_estimator_, os.path.join(cfg.MODEL_DIR, cfg.MODEL_NAMES["rf_clust"]))
    print(f"  RF-Clust → Accuracy={row['Accuracy']}  F1={row['F1']}  ROC-AUC={row['ROC_AUC']}")

    # ── DT-Clust ──────────────────────────────────────────────────────────────
    print(f"\n[6.4] DT-Clust — GridSearchCV...")
    dt_params = {"max_depth": [4, 6, 8, 10], "min_samples_leaf": [10, 30, 50]}
    dt = GridSearchCV(DecisionTreeClassifier(random_state=cfg.RANDOM_STATE),
                      dt_params, cv=cv, scoring="roc_auc", n_jobs=-1)
    dt.fit(X_tr_clust, y_train)
    thr_dt = youden_threshold(dt.best_estimator_, X_tr_clust, y_train)
    p, r = _precision_recall_from_model(dt.best_estimator_, X_te_clust, y_test, thr_dt)
    row = evaluate("DT-Clust", dt.best_estimator_, X_te_clust, y_test, thr_dt)
    row["Precision"] = p; row["Recall"] = r
    results.append(row); models["dt_clust"] = dt.best_estimator_
    joblib.dump(dt.best_estimator_, os.path.join(cfg.MODEL_DIR, cfg.MODEL_NAMES["dt_clust"]))
    print(f"  DT-Clust → Accuracy={row['Accuracy']}  F1={row['F1']}  ROC-AUC={row['ROC_AUC']}")

    # ── LR-Base ───────────────────────────────────────────────────────────────
    print(f"\n[6.5] LR-Base — GridSearchCV(C)...")
    lr_params = {"C": [0.001, 0.01, 0.1, 1, 10, 100]}
    lr_b = GridSearchCV(LogisticRegression(max_iter=1000, random_state=cfg.RANDOM_STATE),
                        lr_params, cv=cv, scoring="roc_auc", n_jobs=-1)
    lr_b.fit(X_tr_lr_base, y_train)
    thr_lrb = youden_threshold(lr_b.best_estimator_, X_tr_lr_base, y_train)
    p, r = _precision_recall_from_model(lr_b.best_estimator_, X_te_lr_base, y_test, thr_lrb)
    row = evaluate("LR-Base", lr_b.best_estimator_, X_te_lr_base, y_test, thr_lrb)
    row["Precision"] = p; row["Recall"] = r
    results.append(row); models["lr_base"] = lr_b.best_estimator_
    joblib.dump(lr_b.best_estimator_, os.path.join(cfg.MODEL_DIR, cfg.MODEL_NAMES["lr_base"]))
    print(f"  LR-Base  → Accuracy={row['Accuracy']}  F1={row['F1']}  ROC-AUC={row['ROC_AUC']}")

    # ── LR-Dist ───────────────────────────────────────────────────────────────
    print(f"\n[6.6] LR-Dist — GridSearchCV(C)...")
    lr_d = GridSearchCV(LogisticRegression(max_iter=1000, random_state=cfg.RANDOM_STATE),
                        lr_params, cv=cv, scoring="roc_auc", n_jobs=-1)
    lr_d.fit(X_tr_lr_dist, y_train)
    thr_lrd = youden_threshold(lr_d.best_estimator_, X_tr_lr_dist, y_train)
    p, r = _precision_recall_from_model(lr_d.best_estimator_, X_te_lr_dist, y_test, thr_lrd)
    row = evaluate("LR-Dist", lr_d.best_estimator_, X_te_lr_dist, y_test, thr_lrd)
    row["Precision"] = p; row["Recall"] = r
    results.append(row); models["lr_dist"] = lr_d.best_estimator_
    joblib.dump(lr_d.best_estimator_, os.path.join(cfg.MODEL_DIR, cfg.MODEL_NAMES["lr_dist"]))
    print(f"  LR-Dist  → Accuracy={row['Accuracy']}  F1={row['F1']}  ROC-AUC={row['ROC_AUC']}")

    # ── Comparison table ──────────────────────────────────────────────────────
    mc = pd.DataFrame(results)[["Model","Threshold","Accuracy","Precision",
                                 "Recall","F1","ROC_AUC","PR_AUC"]]
    mc.to_csv(os.path.join(cfg.DATA_DIR, "model_comparison.csv"), index=False)
    print(f"\n[6.7] Model Comparison Table:")
    print(mc.to_string(index=False))

    # Comparison bar chart
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    for ax, metric in zip(axes, ["Accuracy", "F1", "ROC_AUC", "PR_AUC"]):
        colors = ["#3b82f6" if "XGB" in m else "#22c55e" if "RF" in m
                  else "#f59e0b" if "DT" in m else "#a855f7" for m in mc["Model"]]
        ax.bar(mc["Model"], mc[metric], color=colors, edgecolor="white")
        ax.set_title(metric, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.3)
        for i, (_, row) in enumerate(mc.iterrows()):
            ax.text(i, row[metric] + 0.01, f"{row[metric]:.3f}",
                    ha="center", fontsize=8, fontweight="bold")
    plt.suptitle("Step 6: Model Comparison — Airline Satisfaction", fontweight="bold")
    plt.tight_layout()
    out = os.path.join(cfg.OUT_DIR, "p2_06_model_comparison.png")
    plt.savefig(out, dpi=130, bbox_inches="tight"); plt.close()
    print(f"  Chart saved → {out}")

    print(f"\n{'─'*70}\nSTEP 6 DONE\n{'─'*70}")
    return {"comparison": mc, "models": models,
            "xgb_clust": xgb_clust_cal, "thr_clust": thr_clust}


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import src_airline.airline_config as cfg
    print("Run via run_airline_ml.py")
