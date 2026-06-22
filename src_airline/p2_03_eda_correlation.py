"""
p2_03_eda_correlation.py  —  Step 3: EDA Bổ Sung
==================================================
Mirror Churn Alert Step 3:
  - Cramér's V: satisfaction vs categoricals
  - Point-Biserial: satisfaction vs numerics
  - Pearson Heatmap: 14 rating cols
"""
import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pointbiserialr, chi2_contingency


def cramers_v(col_a: pd.Series, col_b: pd.Series) -> float:
    ct     = pd.crosstab(col_a, col_b)
    chi2   = chi2_contingency(ct, correction=False)[0]
    n      = len(col_a)
    r, c   = ct.shape
    return float(np.sqrt(chi2 / (n * (min(r, c) - 1)))) if min(r, c) > 1 else 0.0


def run_step3(cfg, train_df: pd.DataFrame):
    sep = "=" * 70
    print(f"\n{sep}\nSTEP 3: EDA — CORRELATION ANALYSIS\n{sep}")

    target = train_df["target"]

    # ── 3.1  Cramér's V ───────────────────────────────────────────────────────
    print(f"\n[3.1] Cramér's V — satisfaction vs categorical variables:")
    cat_checks = {
        "loyal":      "Customer Type (Loyal=1)",
        "biz_travel": "Type of Travel (Business=1)",
        "Class_ord":  "Class (Eco=0, EcoPlus=1, Business=2)",
    }
    cv_results = []
    for col, label in cat_checks.items():
        if col not in train_df.columns:
            continue
        v = cramers_v(train_df[col].astype(str), target.astype(str))
        cv_results.append((label, v))
    cv_results.sort(key=lambda x: -x[1])
    hdr = f"  {'Variable':<40}  {'Cramers V':>11}  {'Association':>12}"
    print(hdr)
    print("  " + "─" * 67)
    for label, v in cv_results:
        strength = ("Strong" if v > 0.3 else "Moderate" if v > 0.1 else "Weak")
        bar = "█" * int(v * 30)
        print(f"  {label:<40}  {v:>11.4f}  {strength:>12}  {bar}")

    # ── 3.2  Point-Biserial correlation ───────────────────────────────────────
    print(f"\n[3.2] Point-Biserial correlation — satisfaction (0/1) vs numeric columns:")
    numeric_cols = (
        cfg.SERVICE_COLS +
        ["Age", "Flight Distance", "Departure Delay in Minutes",
         "ground_experience_score", "inflight_experience_score", "delay_severity"]
    )
    pb_results = []
    for col in numeric_cols:
        if col not in train_df.columns:
            continue
        valid = train_df[[col, "target"]].dropna()
        if len(valid) < 100:
            continue
        r, p = pointbiserialr(valid["target"], valid[col])
        pb_results.append((col, r, p))
    pb_results.sort(key=lambda x: -abs(x[1]))

    print(f"  {'Column':<45}  {'r':>8}  {'p-value':>12}  Strength")
    print("  " + "─" * 75)
    for col, r, p in pb_results[:15]:
        sig   = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        dir_  = "↑ satisfied" if r > 0 else "↓ dissatisfied"
        print(f"  {col:<45}  {r:>8.4f}  {p:>12.2e}  {sig:>3}  {dir_}")

    # ── 3.3  Pearson Heatmap (14 rating cols) ─────────────────────────────────
    print(f"\n[3.3] Pearson Heatmap among 14 rating columns (train set):")
    avail_ratings = [c for c in cfg.SERVICE_COLS if c in train_df.columns]
    corr_mat = train_df[avail_ratings].corr(method="pearson")

    # Find highly correlated pairs
    high_pairs = []
    for i in range(len(avail_ratings)):
        for j in range(i + 1, len(avail_ratings)):
            r = corr_mat.iloc[i, j]
            if abs(r) > 0.4:
                high_pairs.append((avail_ratings[i], avail_ratings[j], r))
    high_pairs.sort(key=lambda x: -abs(x[2]))
    if high_pairs:
        print(f"  Pairs with |r| > 0.4:")
        for a, b, r in high_pairs[:10]:
            print(f"    {a} ↔ {b}: {r:.3f}")

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    short = [c.replace("Departure/", "Dep/").replace(" service", " svc")
               .replace(" booking", " bkg").replace("On-board", "On-brd")
             for c in avail_ratings]
    sns.heatmap(
        corr_mat, annot=True, fmt=".2f", cmap="coolwarm",
        center=0, vmin=-1, vmax=1,
        linewidths=0.5, linecolor="#1e293b",
        xticklabels=short, yticklabels=short, ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Pearson Correlation — 14 Service Rating Columns (Train Set)",
                 fontsize=12, fontweight="bold", pad=15)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    out = os.path.join(cfg.OUT_DIR, "p2_03_pearson_heatmap.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Heatmap saved → {out}")

    # Point-biserial bar chart
    fig2, ax2 = plt.subplots(figsize=(10, 7))
    top = pb_results[:12]
    cols_plot  = [x[0] for x in top]
    r_vals     = [x[1] for x in top]
    colors     = ["#22c55e" if r > 0 else "#ef4444" for r in r_vals]
    short_lbl  = [c.replace("experience_score", "score") for c in cols_plot]
    ax2.barh(short_lbl[::-1], r_vals[::-1], color=colors[::-1], edgecolor="white")
    ax2.axvline(0, color="white", linewidth=1)
    ax2.set_xlabel("Point-Biserial r  (positive = more satisfied)")
    ax2.set_title("Top 12 Features — Point-Biserial Correlation with Satisfaction",
                  fontweight="bold")
    ax2.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    out2 = os.path.join(cfg.OUT_DIR, "p2_03_point_biserial.png")
    plt.savefig(out2, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Point-biserial chart saved → {out2}")

    print(f"\n{'─'*70}\nSTEP 3 DONE\n{'─'*70}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import src_airline.airline_config as cfg
    train = pd.read_parquet(os.path.join(cfg.DATA_INT_DIR, "train_prepared.parquet"))
    run_step3(cfg, train)
