"""
p2_05b_linear_preprocessing.py  —  Step 5b: Linear Path Preprocessing
=======================================================================
| Step       | Technique                                              |
|------------|-------------------------------------------------------|
| Clip       | Departure Delay ≤ p99 of train                        |
| Impute     | 14 rating cols → median (fit train) + keep was_na_*  |
| MinMax     | All numeric to [0,1] (fit train)                      |
| Chi²       | SelectKBest(chi2), p < 0.05 — Base & Dist variants   |
| SMOTE      | According to Step 5 decision                          |
"""
import pandas as pd
import numpy as np
import os
import joblib
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2


def run_step5b(cfg, step5_result: dict) -> dict:
    sep = "=" * 70
    print(f"\n{sep}\nSTEP 5b: PREPROCESSING — LINEAR PATH\n{sep}")

    X_train_dist = step5_result["X_train_dist"].copy()
    X_test_dist  = step5_result["X_test_dist"].copy()
    y_train      = step5_result["y_train"]
    y_test       = step5_result["y_test"]
    dist_cols    = step5_result["dist_cols"]

    # ── ① Outlier clip — Departure Delay at p99 of train ─────────────────────
    dep_col = "Departure Delay in Minutes"
    if dep_col in X_train_dist.columns:
        p99 = float(np.percentile(X_train_dist[dep_col].dropna(), 99))
        n_clipped_tr = (X_train_dist[dep_col] > p99).sum()
        n_clipped_te = (X_test_dist[dep_col] > p99).sum()
        X_train_dist[dep_col] = X_train_dist[dep_col].clip(upper=p99)
        X_test_dist[dep_col]  = X_test_dist[dep_col].clip(upper=p99)
        print(f"\n[5b.1] Clipped '{dep_col}' at p99={p99:.1f} min:")
        print(f"  Train clipped: {n_clipped_tr:,}  Test clipped: {n_clipped_te:,}")
    else:
        print(f"\n[5b.1] '{dep_col}' not found — skip clip.")

    # ── ② Impute 14 rating cols ───────────────────────────────────────────────
    rating_cols = [c for c in cfg.SERVICE_COLS if c in X_train_dist.columns]
    print(f"\n[5b.2] Impute {len(rating_cols)} rating cols (median, fit train):")
    imp_rating = SimpleImputer(strategy="median")
    X_train_dist[rating_cols] = imp_rating.fit_transform(X_train_dist[rating_cols])
    X_test_dist[rating_cols]  = imp_rating.transform(X_test_dist[rating_cols])
    joblib.dump(imp_rating, os.path.join(cfg.MODEL_DIR, "imputer.pkl"))
    print(f"  Medians: { {c: round(float(imp_rating.statistics_[i]),1) for i,c in enumerate(rating_cols[:4])} } ...")

    # ── ③ MinMaxScaler ────────────────────────────────────────────────────────
    was_na_cols = [c for c in X_train_dist.columns if c.startswith("was_na_")]
    non_scale   = was_na_cols  # binary indicators don't need scaling

    num_cols = [c for c in X_train_dist.columns
                if c not in non_scale and X_train_dist[c].dtype != object]

    scaler = MinMaxScaler()
    X_train_scaled = X_train_dist.copy()
    X_test_scaled  = X_test_dist.copy()
    X_train_scaled[num_cols] = scaler.fit_transform(X_train_dist[num_cols])
    X_test_scaled[num_cols]  = scaler.transform(X_test_dist[num_cols])
    joblib.dump(scaler, os.path.join(cfg.MODEL_DIR, "scaler.pkl"))
    print(f"\n[5b.3] MinMaxScaler fit on {len(num_cols)} numeric cols (binary was_na_* excluded).")

    # ── ④ Chi² Feature Selection ──────────────────────────────────────────────
    # Base: no KMeans dist cols
    # Dist: include dist_cols
    print(f"\n[5b.4] Chi-Square Feature Selection (SelectKBest, p < 0.05):")

    def chi2_select(X_tr, X_te, y_tr, label):
        selector = SelectKBest(chi2, k="all")
        selector.fit(X_tr, y_tr)
        pvals  = selector.pvalues_
        keep   = [col for col, p in zip(X_tr.columns, pvals) if p < 0.05]
        fallback = False
        if len(keep) < 10:
            top10 = sorted(zip(X_tr.columns, pvals), key=lambda x: x[1])[:10]
            keep  = [c for c, _ in top10]
            fallback = True
        print(f"  [{label}] {len(keep)}/{len(X_tr.columns)} features kept "
              f"(p<0.05){'  ← fallback top-10' if fallback else ''}")
        scores_df = pd.DataFrame({
            "feature": X_tr.columns,
            "chi2_score": selector.scores_,
            "p_value": pvals,
        }).sort_values("chi2_score", ascending=False)
        print(f"    Top 8: {list(scores_df['feature'][:8])}")
        return X_tr[keep], X_te[keep], keep, scores_df

    all_cols = list(X_train_scaled.columns)
    base_cols = [c for c in all_cols if c not in dist_cols]

    X_tr_base, X_te_base, keep_base, scores_base = chi2_select(
        X_train_scaled[base_cols], X_test_scaled[base_cols], y_train, "Base"
    )
    X_tr_dist, X_te_dist, keep_dist, scores_dist = chi2_select(
        X_train_scaled, X_test_scaled, y_train, "Dist"
    )

    chi2_meta = {
        "selected_base": keep_base,
        "selected_dist": keep_dist,
        "scores_base": scores_base,
        "scores_dist": scores_dist,
    }
    joblib.dump(chi2_meta, os.path.join(cfg.MODEL_DIR, "chi2_meta.pkl"))

    # Chi2 bar chart (top 20 by score)
    fig, ax = plt.subplots(figsize=(10, 7))
    top20 = scores_base.head(20)
    colors = ["#22c55e" if p < 0.05 else "#ef4444" for p in top20["p_value"]]
    ax.barh(top20["feature"][::-1], top20["chi2_score"][::-1],
            color=colors[::-1], edgecolor="white")
    ax.set_title("Chi² Feature Selection — Base Variant (Top 20)", fontweight="bold")
    ax.set_xlabel("Chi² Score")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    chi2_path = os.path.join(cfg.OUT_DIR, "p2_05b_chi2_scores.png")
    plt.savefig(chi2_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Chi² chart saved → {chi2_path}")

    # Save
    X_tr_base.to_parquet(os.path.join(cfg.DATA_INT_DIR, "X_train_lr_base.parquet"), index=False)
    X_te_base.to_parquet(os.path.join(cfg.DATA_INT_DIR, "X_test_lr_base.parquet"),  index=False)
    X_tr_dist.to_parquet(os.path.join(cfg.DATA_INT_DIR, "X_train_lr_dist.parquet"), index=False)
    X_te_dist.to_parquet(os.path.join(cfg.DATA_INT_DIR, "X_test_lr_dist.parquet"),  index=False)

    result = {
        "X_tr_lr_base": X_tr_base, "X_te_lr_base": X_te_base,
        "X_tr_lr_dist": X_tr_dist, "X_te_lr_dist": X_te_dist,
        "chi2_meta": chi2_meta,
    }

    print(f"\n{'─'*70}\nSTEP 5b DONE\n{'─'*70}")
    return result


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import src_airline.airline_config as cfg
    import joblib as jl
    s5 = {
        "X_train_dist": pd.read_parquet(os.path.join(cfg.DATA_INT_DIR, "X_train_dist.parquet")),
        "X_test_dist":  pd.read_parquet(os.path.join(cfg.DATA_INT_DIR, "X_test_dist.parquet")),
        "y_train": np.load(os.path.join(cfg.DATA_INT_DIR, "y_train.npy")),
        "y_test":  np.load(os.path.join(cfg.DATA_INT_DIR, "y_test.npy")),
        "dist_cols": [c for c in pd.read_parquet(
            os.path.join(cfg.DATA_INT_DIR, "X_train_dist.parquet")).columns
            if c.startswith("dist_centroid_")],
    }
    run_step5b(cfg, s5)
