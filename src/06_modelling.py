"""
Step 6: Modelling — XGBoost + Optuna + CalibratedClassifier + SHAP
         + Logistic Regression (outlier-cleaned linear features từ Step 5b)
         + Decision Tree & Random Forest (K-Means cluster-ID feature)
         + XGBoost retrained with cluster-ID feature (Strategy B)
         + Unified model comparison chart

Data sources per model
  LR-Base   → X_train_linear.csv          (Step 5b: outlier-removed, SMOTE)
  LR-Dist   → X_train_linear_dist.csv     (Step 5b: base + K-Means dist, SMOTE)
  DT-Clust  → X_train_clust.csv           (Step 5:  SMOTE, K-Means cluster_id)
  RF-Clust  → X_train_clust.csv
  XGB-Base  → X_train.csv                 (Step 5:  SMOTE, full feature set)
  XGB-Clust → X_train_clust.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import joblib
import shap
import optuna
import os
import warnings
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, average_precision_score,
                              roc_curve, confusion_matrix, ConfusionMatrixDisplay)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

from pipeline_config import DATA_DIR, MODEL_DIR, OUT_DIR   # noqa: E402

N_TRIALS = 40


# ─────────────────────────────────────────────────────────────────────────────
# Data loaders
# ─────────────────────────────────────────────────────────────────────────────
def load_data():
    X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
    X_test  = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).squeeze()
    y_test  = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).squeeze()
    return X_train, X_test, y_train, y_test


def load_feature_variants():
    """Load K-Means cluster-ID splits từ Step 05 (cho Tree models)."""
    X_train_clust = pd.read_csv(os.path.join(DATA_DIR, "X_train_clust.csv"))
    X_test_clust  = pd.read_csv(os.path.join(DATA_DIR, "X_test_clust.csv"))
    return X_train_clust, X_test_clust


def load_linear_data():
    """
    Load outlier-cleaned linear data từ Step 05b.
    LR-Base  — base features only.
    LR-Dist  — base + K-Means distance features.
    Mỗi bộ có y_train riêng (SMOTE thực hiện trong Step 05b).
    """
    X_train_lin      = pd.read_csv(os.path.join(DATA_DIR, "X_train_linear.csv"))
    X_test_lin       = pd.read_csv(os.path.join(DATA_DIR, "X_test_linear.csv"))
    X_train_lin_dist = pd.read_csv(os.path.join(DATA_DIR, "X_train_linear_dist.csv"))
    X_test_lin_dist  = pd.read_csv(os.path.join(DATA_DIR, "X_test_linear_dist.csv"))
    y_train_lin      = pd.read_csv(os.path.join(DATA_DIR, "y_train_linear.csv")).squeeze()
    y_train_lin_dist = pd.read_csv(os.path.join(DATA_DIR, "y_train_linear_dist.csv")).squeeze()
    y_test           = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).squeeze()
    return (X_train_lin, X_test_lin, y_train_lin,
            X_train_lin_dist, X_test_lin_dist, y_train_lin_dist,
            y_test)


# ─────────────────────────────────────────────────────────────────────────────
# XGBoost + Optuna
# ─────────────────────────────────────────────────────────────────────────────
def get_objective(X_train, y_train):
    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 600),
            "max_depth":        trial.suggest_int("max_depth", 3, 8),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma":            trial.suggest_float("gamma", 0, 5),
            "use_label_encoder": False,
            "eval_metric":      "logloss",
            "random_state":     42,
            "n_jobs":           -1,
        }
        pipeline = ImbPipeline([
            ("smote",      SMOTE(random_state=42, k_neighbors=5)),
            ("classifier", XGBClassifier(**params)),
        ])
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(pipeline, X_train, y_train,
                                 cv=cv, scoring="roc_auc", n_jobs=-1)
        return scores.mean()
    return objective


def tune_hyperparameters(X_train, y_train, n_trials=N_TRIALS):
    print(f"[06] Starting Optuna ({n_trials} trials)…")
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(get_objective(X_train, y_train), n_trials=n_trials,
                   show_progress_bar=True)
    print(f"[06] Best ROC-AUC (CV): {study.best_value:.4f}")
    print(f"[06] Best params: {study.best_params}")
    return study.best_params


def train_calibrated_model(X_train, y_train, best_params):
    """Train XGBoost trong ImbPipeline, rồi calibrate với isotonic regression."""
    inner_pipeline = ImbPipeline([
        ("smote",      SMOTE(random_state=42, k_neighbors=5)),
        ("classifier", XGBClassifier(**best_params,
                                     use_label_encoder=False,
                                     eval_metric="logloss",
                                     random_state=42,
                                     n_jobs=-1)),
    ])
    inner_pipeline.fit(X_train, y_train)
    calibrated = CalibratedClassifierCV(inner_pipeline, method="isotonic", cv="prefit")
    calibrated.fit(X_train, y_train)
    print("[06] Calibrated XGBoost trained.")
    return calibrated, inner_pipeline


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────
def _youden_threshold(probs, y_test):
    fpr, tpr, thresholds = roc_curve(y_test, probs)
    return float(thresholds[np.argmax(tpr - fpr)])


def _metrics_dict(name, probs, y_test):
    thr   = _youden_threshold(probs, y_test)
    preds = (probs >= thr).astype(int)
    return {
        "Model":     name,
        "Threshold": round(thr, 3),
        "Accuracy":  round(accuracy_score(y_test, preds), 4),
        "Precision": round(precision_score(y_test, preds, zero_division=0), 4),
        "Recall":    round(recall_score(y_test, preds, zero_division=0), 4),
        "F1":        round(f1_score(y_test, preds, zero_division=0), 4),
        "ROC_AUC":   round(roc_auc_score(y_test, probs), 4),
        "PR_AUC":    round(average_precision_score(y_test, probs), 4),
    }


def find_optimal_threshold(model, X_test, y_test):
    probs = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, probs)
    best_idx = np.argmax(tpr - fpr)
    best_thr = thresholds[best_idx]
    print(f"[06] Optimal threshold (Youden's J): {best_thr:.3f}")
    return best_thr, probs, fpr, tpr


def evaluate(model, X_test, y_test, threshold, probs, fpr, tpr):
    preds   = (probs >= threshold).astype(int)
    metrics = {
        "Threshold": round(threshold, 3),
        "Accuracy":  round(accuracy_score(y_test, preds), 4),
        "Precision": round(precision_score(y_test, preds), 4),
        "Recall":    round(recall_score(y_test, preds), 4),
        "F1":        round(f1_score(y_test, preds), 4),
        "ROC_AUC":   round(roc_auc_score(y_test, probs), 4),
        "PR_AUC":    round(average_precision_score(y_test, probs), 4),
    }
    print(f"\n[06] ── XGB-Base Performance ────────────────────────────────")
    for k, v in metrics.items():
        print(f"       {k:<12}: {v}")
    return metrics, preds


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────
def plot_roc_curve(fpr, tpr, auc_score):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#3b82f6", lw=2, label=f"ROC-AUC = {auc_score:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.fill_between(fpr, tpr, alpha=0.1, color="#3b82f6")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — XGBoost (Calibrated)", fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "06_roc_curve.png"), dpi=150)
    plt.close()
    print("[06] Saved: 06_roc_curve.png")


def plot_confusion_matrix(y_test, preds):
    cm   = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Retained", "Churned"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix — XGBoost (Calibrated)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "06_confusion_matrix.png"), dpi=150)
    plt.close()
    print("[06] Saved: 06_confusion_matrix.png")


def plot_calibration_curve(model, X_test, y_test):
    probs = model.predict_proba(X_test)[:, 1]
    prob_true, prob_pred = calibration_curve(y_test, probs, n_bins=10)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].plot(prob_pred, prob_true, "o-", color="#3b82f6",
                 linewidth=2, markersize=7, label="XGBoost (Calibrated)")
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Perfect calibration")
    axes[0].fill_between(prob_pred, prob_pred, prob_true, alpha=0.15, color="#3b82f6")
    axes[0].set_xlabel("Mean Predicted Probability")
    axes[0].set_ylabel("Fraction of Positives (Actual)")
    axes[0].set_title("Reliability Diagram\n(Calibration Curve)", fontweight="bold")
    axes[0].legend(loc="upper left")
    axes[0].grid(axis="both", linestyle="--", alpha=0.4)
    axes[0].set_xlim(0, 1); axes[0].set_ylim(0, 1)
    mean_error = np.mean(np.abs(prob_true - prob_pred))
    axes[0].text(0.05, 0.90, f"Mean calibration error: {mean_error:.4f}",
                 transform=axes[0].transAxes, fontsize=10, color="#1e3a8a",
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="#eff6ff", alpha=0.8))

    axes[1].hist(probs[y_test == 0], bins=30, alpha=0.65,
                 color="#22c55e", label="Retained (actual)", edgecolor="white")
    axes[1].hist(probs[y_test == 1], bins=30, alpha=0.65,
                 color="#ef4444", label="Churned (actual)", edgecolor="white")
    axes[1].axvline(0.5, color="gray", linestyle="--", linewidth=1.2, label="Threshold 0.5")
    axes[1].set_xlabel("Predicted Churn Probability")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Probability Distribution by True Label", fontweight="bold")
    axes[1].legend()
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)

    plt.suptitle("Model Calibration Validation", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "06_calibration_curve.png"), dpi=150)
    plt.close()
    print("[06] Saved: 06_calibration_curve.png")


def compute_shap(inner_pipeline, X_test, feature_names):
    xgb_core    = inner_pipeline.named_steps["classifier"]
    explainer   = shap.TreeExplainer(xgb_core)
    X_test_df   = pd.DataFrame(X_test, columns=feature_names)
    shap_values = explainer.shap_values(X_test_df)

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test_df, show=False, max_display=15)
    plt.title("SHAP Summary Plot (Top 15 Features)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "06_shap_summary.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("[06] Saved: 06_shap_summary.png")

    mean_shap = np.abs(shap_values).mean(axis=0)
    shap_df   = pd.DataFrame({"Feature": feature_names, "MeanSHAP": mean_shap})
    shap_df   = shap_df.nlargest(15, "MeanSHAP").sort_values("MeanSHAP")
    fig, ax   = plt.subplots(figsize=(10, 7))
    ax.barh(shap_df["Feature"], shap_df["MeanSHAP"], color="#3b82f6", edgecolor="white")
    ax.set_title("Mean |SHAP| Value — Feature Importance", fontweight="bold")
    ax.set_xlabel("Mean |SHAP|")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "06_shap_bar.png"), dpi=150)
    plt.close()
    print("[06] Saved: 06_shap_bar.png")

    return shap_values, shap_df


# ─────────────────────────────────────────────────────────────────────────────
# Additional models
# ─────────────────────────────────────────────────────────────────────────────
def train_logistic_regression(X_train, X_test, y_train, y_test, name="LR-Base"):
    """
    Logistic Regression với GridSearch over C.
    Training data đã SMOTE-balanced từ Step 05b.
    """
    print(f"\n[06] Training {name}…")
    cv   = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    lr   = LogisticRegression(max_iter=2000, random_state=42, n_jobs=-1,
                               class_weight="balanced", solver="lbfgs")
    grid = GridSearchCV(lr, {"C": [0.01, 0.1, 1, 10]},
                        cv=cv, scoring="roc_auc", n_jobs=-1)
    grid.fit(X_train, y_train)
    best_lr = grid.best_estimator_
    probs   = best_lr.predict_proba(X_test)[:, 1]
    metrics = _metrics_dict(name, probs, y_test)
    print(f"[06] {name}  best C={grid.best_params_['C']}  "
          f"ROC-AUC={metrics['ROC_AUC']}  F1={metrics['F1']}")
    joblib.dump(best_lr,
                os.path.join(MODEL_DIR, f"{name.lower().replace('-','_')}_model.pkl"))
    return metrics, best_lr


def train_decision_tree(X_train, X_test, y_train, y_test, name="DT-Clust"):
    """Decision Tree (Strategy B — với kmeans_cluster_id feature)."""
    print(f"\n[06] Training {name}…")
    cv   = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    dt   = DecisionTreeClassifier(random_state=42, class_weight="balanced")
    grid = GridSearchCV(
        dt,
        {"max_depth": [3, 5, 7, 10, None], "min_samples_leaf": [1, 5, 10]},
        cv=cv, scoring="roc_auc", n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    best_dt = grid.best_estimator_
    probs   = best_dt.predict_proba(X_test)[:, 1]
    metrics = _metrics_dict(name, probs, y_test)
    print(f"[06] {name}  best={grid.best_params_}  "
          f"ROC-AUC={metrics['ROC_AUC']}  F1={metrics['F1']}")
    joblib.dump(best_dt, os.path.join(MODEL_DIR, "dt_clust_model.pkl"))
    return metrics, best_dt


def train_random_forest(X_train, X_test, y_train, y_test, name="RF-Clust"):
    """Random Forest (Strategy B — với kmeans_cluster_id feature)."""
    print(f"\n[06] Training {name}…")
    cv   = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rf   = RandomForestClassifier(random_state=42, class_weight="balanced", n_jobs=-1)
    grid = GridSearchCV(
        rf,
        {"n_estimators": [100, 200], "max_depth": [5, 10, None]},
        cv=cv, scoring="roc_auc", n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    best_rf = grid.best_estimator_
    probs   = best_rf.predict_proba(X_test)[:, 1]
    metrics = _metrics_dict(name, probs, y_test)
    print(f"[06] {name}  best={grid.best_params_}  "
          f"ROC-AUC={metrics['ROC_AUC']}  F1={metrics['F1']}")
    joblib.dump(best_rf, os.path.join(MODEL_DIR, "rf_clust_model.pkl"))
    return metrics, best_rf


def train_xgboost_clust(X_train, X_test, y_train, y_test, best_params, name="XGB-Clust"):
    """XGBoost retrained với cluster_id feature (reuse Optuna params từ XGB-Base)."""
    print(f"\n[06] Training {name} (reusing Optuna params + cluster_id feature)…")
    xgb = XGBClassifier(
        **best_params,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    calibrated = CalibratedClassifierCV(xgb, method="isotonic", cv=5)
    calibrated.fit(X_train, y_train)
    probs   = calibrated.predict_proba(X_test)[:, 1]
    metrics = _metrics_dict(name, probs, y_test)
    print(f"[06] {name}  ROC-AUC={metrics['ROC_AUC']}  F1={metrics['F1']}")
    joblib.dump(calibrated, os.path.join(MODEL_DIR, "xgb_clust_model.pkl"))
    return metrics, calibrated


# ─────────────────────────────────────────────────────────────────────────────
# Comparison chart
# ─────────────────────────────────────────────────────────────────────────────
def plot_model_comparison(all_metrics: list):
    df   = pd.DataFrame(all_metrics)
    mets = ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "PR_AUC"]

    palette = ["#94a3b8", "#3b82f6", "#6366f1", "#8b5cf6", "#f97316", "#ec4899"]
    colours = palette[:len(df)]

    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    axes = axes.flatten()

    for i, met in enumerate(mets):
        ax   = axes[i]
        bars = ax.bar(df["Model"], df[met],
                      color=colours, edgecolor="white", width=0.6)
        for bar, val in zip(bars, df[met]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.4f}", ha="center", fontsize=9, fontweight="bold")
        best_idx = int(df[met].idxmax())
        bars[best_idx].set_edgecolor("#ef4444")
        bars[best_idx].set_linewidth(2.5)
        ax.set_title(met, fontweight="bold", fontsize=12)
        ax.set_ylim(0, min(1.15, df[met].max() * 1.20))
        ax.tick_params(axis="x", rotation=25, labelsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.suptitle(
        "Model Comparison — Impact of K-Means Feature Engineering\n"
        "(red border = best per metric)",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "06b_model_comparison.png"), dpi=150)
    plt.close()
    print("[06] Saved: 06b_model_comparison.png")

    df.to_csv(os.path.join(DATA_DIR, "model_comparison.csv"), index=False)
    print("[06] Saved: model_comparison.csv")
    print("\n[06] ── Model Comparison Table ──────────────────────────────────────")
    print(df[["Model", "Accuracy", "F1", "ROC_AUC", "PR_AUC"]].to_string(index=False))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def run_modelling():
    print(f"\n{'='*60}")
    print("STEP 6: MODELLING (XGBoost + Optuna + SHAP + Multi-Model Comparison)")
    print(f"{'='*60}")

    X_train, X_test, y_train, y_test = load_data()
    feature_names = X_train.columns.tolist()

    # ── XGBoost-Base ─────────────────────────────────────────────────────────
    best_params = tune_hyperparameters(X_train, y_train, n_trials=N_TRIALS)
    calibrated, inner_pipeline = train_calibrated_model(X_train, y_train, best_params)
    threshold, probs, fpr, tpr = find_optimal_threshold(calibrated, X_test, y_test)
    metrics, preds = evaluate(calibrated, X_test, y_test, threshold, probs, fpr, tpr)

    plot_roc_curve(fpr, tpr, metrics["ROC_AUC"])
    plot_confusion_matrix(y_test, preds)
    plot_calibration_curve(calibrated, X_test, y_test)
    shap_values, shap_df = compute_shap(inner_pipeline, X_test.values, feature_names)

    joblib.dump(calibrated,     os.path.join(MODEL_DIR, "xgb_model.pkl"))
    joblib.dump(inner_pipeline, os.path.join(MODEL_DIR, "xgb_inner_pipeline.pkl"))
    joblib.dump({
        "threshold":     threshold,
        "best_params":   best_params,
        "metrics":       metrics,
        "feature_names": feature_names,
        "shap_df":       shap_df,
    }, os.path.join(MODEL_DIR, "model_metadata.pkl"))
    print(f"[06] Saved: xgb_model.pkl, xgb_inner_pipeline.pkl, model_metadata.pkl")

    pred_df = pd.DataFrame({
        "y_true":            y_test.values,
        "churn_probability": probs,
        "churn_prediction":  preds,
    })
    pred_df.to_csv(os.path.join(DATA_DIR, "predictions.csv"), index=False)
    print(f"[06] Saved: predictions.csv")

    # ── Linear models (Step-05b) ──────────────────────────────────────────────
    print(f"\n[06] ── Linear Models (outlier-cleaned, SMOTE-balanced) ─────────────")
    (X_tr_lin, X_te_lin, y_tr_lin,
     X_tr_lin_dist, X_te_lin_dist, y_tr_lin_dist,
     y_test_lin) = load_linear_data()
    print(f"[06] LR-Base  features: {X_tr_lin.shape[1]}  | train rows: {len(X_tr_lin):,}")
    print(f"[06] LR-Dist  features: {X_tr_lin_dist.shape[1]}  | train rows: {len(X_tr_lin_dist):,}")

    lr_base_metrics, _ = train_logistic_regression(
        X_tr_lin, X_te_lin, y_tr_lin, y_test_lin, name="LR-Base")
    lr_dist_metrics, _ = train_logistic_regression(
        X_tr_lin_dist, X_te_lin_dist, y_tr_lin_dist, y_test_lin, name="LR-Dist")

    # ── Tree models (Step-05 K-Means cluster-ID features) ────────────────────
    print(f"\n[06] ── Tree Models (K-Means cluster-ID features) ──────────────────")
    X_train_clust, X_test_clust = load_feature_variants()

    dt_metrics,        _ = train_decision_tree(X_train_clust, X_test_clust, y_train, y_test)
    rf_metrics,        _ = train_random_forest(X_train_clust, X_test_clust, y_train, y_test)
    xgb_clust_metrics, _ = train_xgboost_clust(
        X_train_clust, X_test_clust, y_train, y_test, best_params)

    xgb_base_metrics = {"Model": "XGB-Base", **metrics}

    # ── Unified comparison ────────────────────────────────────────────────────
    all_results = [
        lr_base_metrics,
        lr_dist_metrics,
        dt_metrics,
        rf_metrics,
        xgb_base_metrics,
        xgb_clust_metrics,
    ]
    plot_model_comparison(all_results)

    print(f"\n{'='*60}\n[06] ✓ Xong.\n{'='*60}\n")
    return calibrated, metrics, shap_df


if __name__ == "__main__":
    run_modelling()