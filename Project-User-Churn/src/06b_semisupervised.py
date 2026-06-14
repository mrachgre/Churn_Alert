"""
Step 6b: Semi-supervised Learning — Strategy C
============================================================
Theory: instead of labeling thousands of random records, we
  1) run K-Means on the full training set
  2) pick the ONE most representative point per cluster
     (the point closest to its centroid)
  3) "label" those K points (ground-truth labels exist, we
     just pretend everything else is unlabeled)
  4) optionally propagate the representative label to the
     nearest P% of each cluster's members

Four variants are compared:
  ① Random-K      — K randomly chosen labeled points    (baseline)
  ② Rep-only      — K representative points only
  ③ Propagated-20% — representative + nearest 20% of cluster
  ④ Propagated-30% — representative + nearest 30% of cluster

All four train a Logistic Regression and are evaluated on
the held-out test set.

Output:
  outputs/06b_semisupervised_comparison.png
  data/semisupervised_results.csv
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                              average_precision_score, precision_score,
                              recall_score)

DATA_DIR  = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
OUT_DIR   = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load():
    """
    Load standard (non-SMOTE) training data to keep the
    semi-supervised simulation realistic: we start from real,
    class-imbalanced data and pretend labels are unavailable.
    """
    X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
    X_test  = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).squeeze()
    y_test  = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).squeeze()
    km      = joblib.load(os.path.join(MODEL_DIR, "kmeans_fe_model.pkl"))
    return X_train, X_test, y_train, y_test, km


# ─────────────────────────────────────────────────────────────────────────────
# Core helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_representative_indices(X_arr: np.ndarray, km) -> np.ndarray:
    """
    For each cluster, find the index of the sample closest to the centroid.
    Returns an array of shape (K,).
    """
    dists = km.transform(X_arr)           # (n_samples, K)
    return dists.argmin(axis=0)           # one index per cluster


def propagate_labels(
    X_arr: np.ndarray,
    km,
    rep_indices: np.ndarray,
    propagation_ratio: float,
) -> np.ndarray:
    """
    Expand the labeled set by adding the nearest `propagation_ratio` fraction
    of each cluster's members (sorted by distance to centroid).

    Parameters
    ----------
    X_arr            : encoded training feature matrix
    km               : fitted KMeans model
    rep_indices      : K representative indices (one per cluster)
    propagation_ratio: e.g. 0.20 → nearest 20% of each cluster

    Returns
    -------
    np.ndarray of unique sample indices forming the expanded labeled set
    """
    dists          = km.transform(X_arr)      # (n, K)
    cluster_labels = km.predict(X_arr)        # (n,)
    K              = km.n_clusters

    labeled = set(rep_indices.tolist())

    for c in range(K):
        members  = np.where(cluster_labels == c)[0]           # cluster member indices
        c_dists  = dists[members, c]                          # their distances to centroid c
        sorted_m = members[np.argsort(c_dists)]               # sort by proximity
        n_prop   = max(1, int(len(sorted_m) * propagation_ratio))
        labeled.update(sorted_m[:n_prop].tolist())

    return np.array(sorted(labeled))


def evaluate_lr(
    X_labeled: np.ndarray,
    y_labeled: np.ndarray,
    X_test:    np.ndarray,
    y_test:    np.ndarray,
    name:      str,
) -> dict | None:
    """
    Train a Logistic Regression on the labeled subset and evaluate on test set.
    Returns None if the labeled set contains only one class (edge case).
    """
    classes = np.unique(y_labeled)
    if len(classes) < 2:
        print(f"[06b] ⚠  {name}: only class(es) {classes} in labeled set — skipped.")
        return None

    lr = LogisticRegression(
        max_iter=2000, random_state=42,
        class_weight="balanced",   # critical: labeled set is tiny & imbalanced
        solver="lbfgs",
    )
    lr.fit(X_labeled, y_labeled)
    probs = lr.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    return {
        "Method":    name,
        "n_labeled": int(len(y_labeled)),
        "Accuracy":  round(accuracy_score(y_test, preds), 4),
        "Precision": round(precision_score(y_test, preds, zero_division=0), 4),
        "Recall":    round(recall_score(y_test, preds, zero_division=0), 4),
        "F1":        round(f1_score(y_test, preds, zero_division=0), 4),
        "ROC_AUC":   round(roc_auc_score(y_test, probs), 4),
        "PR_AUC":    round(average_precision_score(y_test, probs), 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def plot_semisupervised_comparison(results: list):
    """
    Three panels (Accuracy, F1, ROC-AUC) as grouped bar charts.
    Each bar is annotated with the metric value and sample count.
    The best bar per panel gets a red border.
    """
    df      = pd.DataFrame(results)
    metrics = ["Accuracy", "F1", "ROC_AUC"]
    labels  = {
        "Accuracy": "Accuracy",
        "F1":       "F1-Score",
        "ROC_AUC":  "ROC-AUC",
    }

    # Palette: grey (random baseline) → blue → green → amber
    palette = ["#94a3b8", "#3b82f6", "#22c55e", "#f59e0b"]
    colours = palette[:len(df)]

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))

    for ax, met in zip(axes, metrics):
        bars = ax.bar(
            df["Method"], df[met],
            color=colours, edgecolor="white", width=0.6,
        )
        # Annotate value + sample count
        for bar, val, n in zip(bars, df[met], df["n_labeled"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.008,
                f"{val:.4f}\n(n={n:,})",
                ha="center", va="bottom", fontsize=9.5, fontweight="bold",
            )
        # Highlight best with red border
        best_idx = int(df[met].idxmax())
        bars[best_idx].set_edgecolor("#ef4444")
        bars[best_idx].set_linewidth(2.5)

        ax.set_title(labels[met], fontweight="bold", fontsize=13)
        ax.set_ylim(0, min(1.2, df[met].max() * 1.30))
        ax.tick_params(axis="x", rotation=15, labelsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_xlabel("Labeling Strategy")

    # Legend patch for propagation ratios
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=palette[0], label="Random-K (baseline)"),
        Patch(facecolor=palette[1], label="Representative only"),
        Patch(facecolor=palette[2], label="Propagated 20%"),
        Patch(facecolor=palette[3], label="Propagated 30%"),
    ]
    fig.legend(handles=legend_handles, loc="upper center",
               ncol=4, fontsize=10, framealpha=0.9,
               bbox_to_anchor=(0.5, 1.02))

    plt.suptitle(
        "Semi-supervised LR — Label Propagation Impact (Strategy C)\n"
        "K-Means representative points → propagating cluster labels",
        fontsize=13, fontweight="bold", y=1.06,
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUT_DIR, "06b_semisupervised_comparison.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close()
    print("[06b] Saved: 06b_semisupervised_comparison.png")


# ─────────────────────────────────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────────────────────────────────

def run_semisupervised():
    print(f"\n{'='*60}")
    print("STEP 6b: SEMI-SUPERVISED LEARNING (Strategy C)")
    print(f"{'='*60}")

    X_train, X_test, y_train, y_test, km = load()
    K           = km.n_clusters
    X_train_arr = X_train.values
    X_test_arr  = X_test.values
    y_train_arr = y_train.values
    y_test_arr  = y_test.values

    print(f"[06b] KMeans K={K} | Training samples: {len(y_train_arr):,} "
          f"| Test samples: {len(y_test_arr):,}")

    # ── ① Find representative indices (one per cluster) ──────────────────────
    rep_indices = get_representative_indices(X_train_arr, km)
    X_rep = X_train_arr[rep_indices]
    y_rep = y_train_arr[rep_indices]
    rep_label_dist = dict(zip(*np.unique(y_rep, return_counts=True)))
    print(f"[06b] Representative points selected: {K}  |  label dist: {rep_label_dist}")

    # ── ② Random-K baseline (K random labeled points) ────────────────────────
    rng        = np.random.default_rng(42)
    rand_idx   = rng.choice(len(y_train_arr), size=K, replace=False)
    X_rand     = X_train_arr[rand_idx]
    y_rand     = y_train_arr[rand_idx]

    # ── ③ Propagated sets (20 % and 30 %) ────────────────────────────────────
    prop20_idx = propagate_labels(X_train_arr, km, rep_indices, propagation_ratio=0.20)
    prop30_idx = propagate_labels(X_train_arr, km, rep_indices, propagation_ratio=0.30)
    X_prop20   = X_train_arr[prop20_idx]
    y_prop20   = y_train_arr[prop20_idx]
    X_prop30   = X_train_arr[prop30_idx]
    y_prop30   = y_train_arr[prop30_idx]

    print(f"[06b] Labeled set sizes:")
    print(f"      Random-K        : {K:>6,} samples")
    print(f"      Rep-only        : {len(y_rep):>6,} samples")
    print(f"      Propagated-20%  : {len(y_prop20):>6,} samples")
    print(f"      Propagated-30%  : {len(y_prop30):>6,} samples")

    # ── ④ Train & evaluate all variants ──────────────────────────────────────
    variants = [
        (X_rand,   y_rand,   f"Random-{K}pts"),
        (X_rep,    y_rep,    f"Rep-only ({K}pts)"),
        (X_prop20, y_prop20, "Propagated-20%"),
        (X_prop30, y_prop30, "Propagated-30%"),
    ]
    results = []
    print(f"\n[06b] {'Method':<22} {'n_labeled':>10} "
          f"{'Accuracy':>10} {'F1':>8} {'ROC-AUC':>10}")
    print(f"[06b] {'-'*65}")

    for X_l, y_l, name in variants:
        r = evaluate_lr(X_l, y_l, X_test_arr, y_test_arr, name)
        if r is not None:
            results.append(r)
            print(f"[06b] {r['Method']:<22} {r['n_labeled']:>10,} "
                  f"{r['Accuracy']:>10.4f} {r['F1']:>8.4f} {r['ROC_AUC']:>10.4f}")

    if not results:
        print("[06b] No valid results — check that kmeans_fe_model.pkl exists "
              "and Step 5 has been run.")
        return []

    # ── ⑤ Plot ───────────────────────────────────────────────────────────────
    plot_semisupervised_comparison(results)

    # ── ⑥ Save CSV ───────────────────────────────────────────────────────────
    pd.DataFrame(results).to_csv(
        os.path.join(DATA_DIR, "semisupervised_results.csv"), index=False)
    print("[06b] Saved: semisupervised_results.csv")

    return results


if __name__ == "__main__":
    run_semisupervised()
