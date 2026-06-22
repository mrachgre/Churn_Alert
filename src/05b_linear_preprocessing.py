"""
Step 5b: Linear Model-Specific Preprocessing
============================================================
Tách riêng preprocessing cho Logistic Regression (LR-Base, LR-Dist)
khỏi pipeline chung dành cho Tree models.

Lý do:
  - Tree models không nhạy cảm với outlier / scale → dùng output Step 5
  - Linear models nhạy với outlier và hưởng lợi từ feature selection

Pipeline:
  1. Outlier removal: IQR clip (1.5 × IQR) trên numeric cols, chỉ training
  2. OHE: reuse encoder.pkl từ Step 5 (cùng category mapping)
  3. MinMaxScaler MỚI (fit on cleaned train) + KNNImputer MỚI
  4. Chi-square (χ²): SelectKBest(chi2), giữ feature có p-value < 0.05
  5. SMOTE (training set only)

Variant _dist (cho LR-Dist):
  Sau bước 3, thêm K-Means cluster distances từ kmeans_fe_model.pkl
  → chi-square trên combined (base + dist)

Outputs
-------
  data/X_train_linear.csv          base features, chi2-selected, SMOTE-balanced
  data/X_test_linear.csv           base features, chi2-selected
  data/X_train_linear_dist.csv     base+dist features, chi2-selected, SMOTE-balanced
  data/X_test_linear_dist.csv      base+dist features, chi2-selected
  models/linear_scaler.pkl         MinMaxScaler (fitted on cleaned train)
  models/linear_imputer.pkl        KNNImputer   (fitted on cleaned train)
  models/chi2_selector.pkl         SelectKBest for base features
  models/chi2_dist_selector.pkl    SelectKBest for base+dist features
  outputs/05b_chi2_scores.png      chi2 score visualisation
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

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.feature_selection import SelectKBest, chi2
from imblearn.over_sampling import SMOTE

# Reuse helpers from Step 5
from importlib.util import spec_from_file_location, module_from_spec

from pipeline_config import (   # noqa: E402
    DATA_DIR  as DATA,
    MODEL_DIR as MODEL,
    OUT_DIR   as OUT,
)
RFM_PATH = os.path.join(DATA, "ecommerce_churn_rfm.csv")

# Location of this file — needed to locate sibling step scripts
_THIS = os.path.dirname(os.path.abspath(__file__))

MIN_FEATURES  = 10      # Fallback: if p<0.05 selects fewer than this, keep top-N
IQR_MULT      = 1.5     # Standard IQR multiplier for outlier clip
CHI2_ALPHA    = 0.05    # Significance threshold for chi-square


# ── Load Step-5 helpers ───────────────────────────────────────────────────────

def _load_step5():
    """Import load() + feature_engineering() + prepare_features() from Step 5."""
    path = os.path.join(_THIS, "05_preprocessing.py")
    spec = spec_from_file_location("step5", path)
    mod  = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── Outlier removal ───────────────────────────────────────────────────────────

def remove_outliers_iqr(X: pd.DataFrame,
                         num_cols: list,
                         multiplier: float = IQR_MULT) -> pd.DataFrame:
    """
    IQR clip on numeric columns — preserves all rows (clip, not drop).
    Only fitted/applied to training data; test data is NOT clipped here.
    Returns clipped copy + per-column clip report.
    """
    X_clean = X.copy()
    report  = {}
    for col in num_cols:
        if col not in X_clean.columns:
            continue
        q1  = X_clean[col].quantile(0.25)
        q3  = X_clean[col].quantile(0.75)
        iqr = q3 - q1
        lo  = q1 - multiplier * iqr
        hi  = q3 + multiplier * iqr
        n_out = ((X_clean[col] < lo) | (X_clean[col] > hi)).sum()
        X_clean[col] = X_clean[col].clip(lo, hi)
        if n_out > 0:
            report[col] = {"lower": round(lo, 4), "upper": round(hi, 4), "clipped": n_out}
    return X_clean, report


# ── Chi-square selection ──────────────────────────────────────────────────────

def select_chi2_features(X_train: np.ndarray,
                          y_train: pd.Series,
                          feature_names: list,
                          alpha: float = CHI2_ALPHA,
                          min_k: int = MIN_FEATURES) -> tuple:
    """
    Fit SelectKBest(chi2) on all features, then keep those with p-value < alpha.
    Falls back to top min_k if fewer than min_k survive the threshold.

    Returns
    -------
    selector     : fitted SelectKBest object
    selected     : list of selected feature names
    scores_df    : DataFrame with [feature, chi2_score, p_value, selected]
    """
    # Run on all features to get scores
    sel_all = SelectKBest(chi2, k="all")
    sel_all.fit(X_train, y_train)

    scores_df = pd.DataFrame({
        "feature":    feature_names,
        "chi2_score": sel_all.scores_,
        "p_value":    sel_all.pvalues_,
    }).sort_values("chi2_score", ascending=False)

    # Select by p-value threshold
    mask_sig = scores_df["p_value"] < alpha
    n_sig    = mask_sig.sum()

    # Fallback: keep top min_k if not enough significant features
    k_best   = max(n_sig, min_k)
    scores_df["selected"] = (
        scores_df["p_value"] < alpha
        if n_sig >= min_k
        else scores_df.index.isin(scores_df.head(min_k).index)
    )

    selected = scores_df.loc[scores_df["selected"], "feature"].tolist()

    # Refit with exact k so transform() works
    selector = SelectKBest(chi2, k=len(selected))
    selector.fit(X_train, y_train)

    # Ensure selector actually picks the right features (sort by score internally)
    # Use boolean mask from scores_df
    selector._scores = sel_all.scores_
    selector._pvalues = sel_all.pvalues_

    return selector, selected, scores_df


def apply_selector(X: np.ndarray, selected_features: list,
                   all_features: list) -> pd.DataFrame:
    """Select columns by name (avoids sklearn's positional transform quirk)."""
    df = pd.DataFrame(X, columns=all_features)
    return df[selected_features]


# ── Visualisation ─────────────────────────────────────────────────────────────

def plot_chi2_scores(base_df: pd.DataFrame,
                     dist_df: pd.DataFrame | None = None):
    """
    Horizontal bar chart of chi2 scores.
    Green = selected (p < 0.05), Grey = not selected.
    """
    n_panels = 2 if dist_df is not None else 1
    fig, axes = plt.subplots(1, n_panels,
                             figsize=(10 * n_panels, max(6, len(base_df) * 0.35)))

    def _draw(ax, df, title):
        df_sorted = df.sort_values("chi2_score", ascending=True)
        colors = ["#22c55e" if s else "#94a3b8" for s in df_sorted["selected"]]
        bars = ax.barh(df_sorted["feature"], df_sorted["chi2_score"],
                       color=colors, edgecolor="white", height=0.7)
        ax.set_title(title, fontweight="bold", fontsize=12)
        ax.set_xlabel("χ² Score")
        ax.axvline(0, color="#374151", linewidth=0.8)
        # Annotate p-value on bars
        for bar, (_, row) in zip(bars, df_sorted.iterrows()):
            ax.text(bar.get_width() + bar.get_width() * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"p={row['p_value']:.3f}", va="center",
                    fontsize=7, color="#6b7280")
        # Legend patch
        from matplotlib.patches import Patch
        legend = [Patch(facecolor="#22c55e", label=f"Selected (p<0.05)  n={df['selected'].sum()}"),
                  Patch(facecolor="#94a3b8", label=f"Dropped             n={(~df['selected']).sum()}")]
        ax.legend(handles=legend, loc="lower right", fontsize=9)
        ax.grid(axis="x", linestyle="--", alpha=0.4)

    _draw(axes[0] if n_panels > 1 else axes,
          base_df, "Chi-Square Feature Selection — Base Features")
    if dist_df is not None:
        _draw(axes[1], dist_df,
              "Chi-Square Feature Selection — Base + K-Means Dist Features")

    plt.suptitle("Step 5b: Chi-Square (χ²) Feature Importance for Linear Models",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "05b_chi2_scores.png"), dpi=150,
                bbox_inches="tight")
    plt.close()
    print("[05b] Saved: 05b_chi2_scores.png")


# ── Main runner ───────────────────────────────────────────────────────────────

def run_linear_preprocessing(random_state: int = 42):
    print(f"\n{'='*60}")
    print("STEP 5b: LINEAR MODEL PREPROCESSING")
    print("         (Outlier Removal + Chi-Square Feature Selection)")
    print(f"{'='*60}")

    # ── Load Step-5 utilities ─────────────────────────────────────────────────
    s5 = _load_step5()

    # ── Reload raw data with SAME split (random_state=42 → identical fold) ───
    df = s5.load(RFM_PATH)
    df = s5.feature_engineering(df)
    X, y = s5.prepare_features(df)

    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state)
    print(f"[05b] Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # ── 1. Outlier removal (training set only) ────────────────────────────────
    X_train_clean, report = remove_outliers_iqr(X_train, num_cols)
    y_train_clean = y_train.loc[X_train_clean.index]
    if report:
        print(f"[05b] Outlier clip (IQR×{IQR_MULT}) — columns affected: {len(report)}")
        for col, info in report.items():
            print(f"       {col:<35} clipped {info['clipped']:>4} values  "
                  f"[{info['lower']:.2f}, {info['upper']:.2f}]")
    else:
        print("[05b] No outliers clipped (IQR check passed for all numeric cols)")

    # ── 2. OHE: reuse encoder.pkl from Step 5 ────────────────────────────────
    encoder    = joblib.load(os.path.join(MODEL, "encoder.pkl"))
    ohe_names  = encoder.get_feature_names_out(cat_cols).tolist()

    def encode(X_raw):
        cat_enc = pd.DataFrame(encoder.transform(X_raw[cat_cols]),
                               columns=ohe_names, index=X_raw.index)
        num_p   = X_raw[num_cols].reset_index(drop=True)
        cat_p   = cat_enc.reset_index(drop=True)
        return pd.concat([num_p, cat_p], axis=1)

    X_train_enc = encode(X_train_clean)
    X_test_enc  = encode(X_test)
    all_cols    = X_train_enc.columns.tolist()
    print(f"[05b] After OHE: {len(all_cols)} features")

    # ── 3. NEW MinMaxScaler + KNNImputer (fit on cleaned train) ──────────────
    num_enc_cols = X_train_enc.select_dtypes(include=[np.number]).columns.tolist()

    linear_scaler = MinMaxScaler()
    X_train_enc[num_enc_cols] = linear_scaler.fit_transform(X_train_enc[num_enc_cols])
    X_test_enc[num_enc_cols]  = linear_scaler.transform(X_test_enc[num_enc_cols])
    X_test_enc[num_enc_cols]  = X_test_enc[num_enc_cols].clip(0, 1)  # guard against test outliers

    linear_imputer = KNNImputer(n_neighbors=5)
    X_train_arr = linear_imputer.fit_transform(X_train_enc)
    X_test_arr  = linear_imputer.transform(X_test_enc)

    X_train_scaled = pd.DataFrame(X_train_arr, columns=all_cols)
    X_test_scaled  = pd.DataFrame(X_test_arr,  columns=all_cols)
    print(f"[05b] MinMaxScaler + KNNImputer fitted on {len(X_train_scaled):,} cleaned samples")

    # ── 4a. Chi-square — Base features ───────────────────────────────────────
    print(f"\n[05b] ── Chi-Square Selection (base features) ──────────────────")
    _, selected_base, scores_base = select_chi2_features(
        X_train_scaled.values, y_train_clean, all_cols)

    print(f"[05b] Features: {len(all_cols)} total → "
          f"{len(selected_base)} selected (p < {CHI2_ALPHA})")
    print(f"[05b] Selected: {selected_base}")

    X_train_sel = X_train_scaled[selected_base]
    X_test_sel  = X_test_scaled[selected_base]

    # ── 4b. Chi-square — Base + K-Means distances ────────────────────────────
    km = joblib.load(os.path.join(MODEL, "kmeans_fe_model.pkl"))
    k  = km.n_clusters
    dist_cols = [f"dist_to_cluster_{i}" for i in range(k)]

    train_dists = km.transform(X_train_scaled.values)
    test_dists  = km.transform(X_test_scaled.values)

    X_train_comb = X_train_scaled.copy()
    X_test_comb  = X_test_scaled.copy()
    X_train_comb[dist_cols] = train_dists
    X_test_comb[dist_cols]  = test_dists
    all_dist_cols = all_cols + dist_cols

    print(f"\n[05b] ── Chi-Square Selection (base + {k} dist features) ──────")
    _, selected_dist, scores_dist = select_chi2_features(
        X_train_comb.values, y_train_clean, all_dist_cols)

    print(f"[05b] Features: {len(all_dist_cols)} total → "
          f"{len(selected_dist)} selected (p < {CHI2_ALPHA})")
    print(f"[05b] Selected: {selected_dist}")

    X_train_sel_dist = X_train_comb[selected_dist]
    X_test_sel_dist  = X_test_comb[selected_dist]

    # ── 5. SMOTE (training sets only) ────────────────────────────────────────
    smote = SMOTE(random_state=random_state, k_neighbors=5)

    X_tr_lin, y_tr_lin = smote.fit_resample(X_train_sel, y_train_clean)
    X_tr_lin = pd.DataFrame(X_tr_lin, columns=selected_base)

    X_tr_lin_dist, y_tr_lin_dist = smote.fit_resample(X_train_sel_dist, y_train_clean)
    X_tr_lin_dist = pd.DataFrame(X_tr_lin_dist, columns=selected_dist)

    print(f"\n[05b] After SMOTE (base):")
    print(f"       Class 0: {(y_tr_lin==0).sum()}  |  Class 1: {(y_tr_lin==1).sum()}")
    print(f"[05b] After SMOTE (dist):")
    print(f"       Class 0: {(y_tr_lin_dist==0).sum()}  |  Class 1: {(y_tr_lin_dist==1).sum()}")

    # ── Save data ─────────────────────────────────────────────────────────────
    X_tr_lin.to_csv(os.path.join(DATA, "X_train_linear.csv"), index=False)
    X_test_sel.to_csv(os.path.join(DATA, "X_test_linear.csv"), index=False)
    X_tr_lin_dist.to_csv(os.path.join(DATA, "X_train_linear_dist.csv"), index=False)
    X_test_sel_dist.to_csv(os.path.join(DATA, "X_test_linear_dist.csv"), index=False)
    y_tr_lin.to_csv(os.path.join(DATA, "y_train_linear.csv"), index=False)
    y_tr_lin_dist.to_csv(os.path.join(DATA, "y_train_linear_dist.csv"), index=False)
    print(f"\n[05b] Saved: X_train_linear.csv  ({X_tr_lin.shape[1]} features, "
          f"{len(X_tr_lin):,} rows)")
    print(f"[05b] Saved: X_test_linear.csv   ({X_test_sel.shape[1]} features, "
          f"{len(X_test_sel):,} rows)")
    print(f"[05b] Saved: X_train_linear_dist.csv  ({X_tr_lin_dist.shape[1]} features)")
    print(f"[05b] Saved: X_test_linear_dist.csv")

    # ── Save models ───────────────────────────────────────────────────────────
    joblib.dump(linear_scaler,  os.path.join(MODEL, "linear_scaler.pkl"))
    joblib.dump(linear_imputer, os.path.join(MODEL, "linear_imputer.pkl"))
    joblib.dump({"selected_base": selected_base,
                 "selected_dist": selected_dist,
                 "scores_base":   scores_base,
                 "scores_dist":   scores_dist},
                os.path.join(MODEL, "chi2_meta.pkl"))
    print(f"[05b] Saved: linear_scaler.pkl, linear_imputer.pkl, chi2_meta.pkl")

    # ── Visualisation ─────────────────────────────────────────────────────────
    plot_chi2_scores(scores_base, scores_dist)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n[05b] ── Summary ───────────────────────────────────────────────")
    print(f"       Outlier clip applied to {len(report)} numeric columns")
    print(f"       LR-Base  : {len(all_cols)} → {len(selected_base)} features "
          f"(reduction: {100*(1 - len(selected_base)/len(all_cols)):.0f}%)")
    print(f"       LR-Dist  : {len(all_dist_cols)} → {len(selected_dist)} features "
          f"(reduction: {100*(1 - len(selected_dist)/len(all_dist_cols)):.0f}%)")

    return {
        "X_train_linear":      X_tr_lin,
        "X_test_linear":       X_test_sel,
        "X_train_linear_dist": X_tr_lin_dist,
        "X_test_linear_dist":  X_test_sel_dist,
        "y_train_linear":      y_tr_lin,
        "selected_base":       selected_base,
        "selected_dist":       selected_dist,
        "scores_base":         scores_base,
        "scores_dist":         scores_dist,
    }


if __name__ == "__main__":
    run_linear_preprocessing()
