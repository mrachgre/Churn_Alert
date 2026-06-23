"""
Step 3: Deep Exploratory Data Analysis
- Class imbalance chart
- Churn rate by Tenure & SatisfactionScore
- Pearson correlation heatmap (numeric)
- Cramér's V heatmap (categorical)
- Point-biserial correlation (numeric vs Churn)
- Customer profiling: Gender, CityTier, MaritalStatus, Complain
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
import os
import warnings
warnings.filterwarnings("ignore")

from pipeline_config import DATA_DIR, OUT_DIR   # noqa: E402
CLEAN_PATH = os.path.join(DATA_DIR, "ecommerce_churn_train.csv")

PALETTE = {"0": "#22c55e", "1": "#ef4444", 0: "#22c55e", 1: "#ef4444"}


def load(path=CLEAN_PATH):
    df = pd.read_csv(path, dtype={"CityTier": str, "Complain": str})
    df["Churn"] = df["Churn"].astype(int)
    return df


# ── 3.1  Class distribution ──────────────────────────────────────────────────
def plot_class_distribution(df: pd.DataFrame):
    counts = df["Churn"].value_counts().sort_index()
    labels = ["Not Churned (0)", "Churned (1)"]
    colors = ["#22c55e", "#ef4444"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Bar
    bars = axes[0].bar(labels, counts.values, color=colors, edgecolor="white", width=0.5)
    for bar, val in zip(bars, counts.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
                     f"{val:,}\n({val/len(df)*100:.1f}%)", ha="center", fontsize=11)
    axes[0].set_title("Class Distribution (Count)", fontweight="bold")
    axes[0].set_ylabel("Number of Customers")

    # Pie
    axes[1].pie(counts.values, labels=labels, colors=colors,
                autopct="%1.1f%%", startangle=90, pctdistance=0.75,
                wedgeprops={"edgecolor": "white", "linewidth": 2})
    axes[1].set_title("Class Distribution (Proportion)", fontweight="bold")

    plt.suptitle("Target Variable: Churn Distribution", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "03_class_distribution.png"), dpi=150)
    plt.close()
    print("[03] Saved: 03_class_distribution.png")


# ── 3.2  Tenure & SatisfactionScore ─────────────────────────────────────────
def plot_tenure_satisfaction(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Tenure bins
    df2 = df.copy()
    df2["Tenure_Bin"] = pd.cut(df2["Tenure"], bins=[-1, 3, 6, 12, 24, 100],
                                labels=["0-3m", "3-6m", "6-12m", "12-24m", "24m+"])
    tenure_churn = df2.groupby("Tenure_Bin", observed=True)["Churn"].mean() * 100
    bars = axes[0].bar(tenure_churn.index.astype(str), tenure_churn.values,
                       color=["#ef4444" if v > 25 else "#f97316" if v > 15 else "#22c55e"
                              for v in tenure_churn.values], edgecolor="white")
    for bar, val in zip(bars, tenure_churn.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f"{val:.1f}%", ha="center", fontsize=10, fontweight="bold")
    axes[0].set_title("Churn Rate by Tenure (months)", fontweight="bold")
    axes[0].set_xlabel("Tenure Group")
    axes[0].set_ylabel("Churn Rate (%)")
    axes[0].axhline(df["Churn"].mean() * 100, color="gray", linestyle="--", alpha=0.6, label="Overall avg")
    axes[0].legend()

    # SatisfactionScore
    sat_churn = df.groupby("SatisfactionScore")["Churn"].mean() * 100
    bars2 = axes[1].bar(sat_churn.index, sat_churn.values,
                        color=["#ef4444" if v > 20 else "#f97316" if v > 15 else "#22c55e"
                               for v in sat_churn.values], edgecolor="white")
    for bar, val in zip(bars2, sat_churn.values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                     f"{val:.1f}%", ha="center", fontsize=10, fontweight="bold")
    axes[1].set_title("Churn Rate by Satisfaction Score", fontweight="bold")
    axes[1].set_xlabel("Satisfaction Score (1–5)")
    axes[1].set_ylabel("Churn Rate (%)")
    axes[1].axhline(df["Churn"].mean() * 100, color="gray", linestyle="--", alpha=0.6, label="Overall avg")
    axes[1].legend()

    plt.suptitle("Churn Rate by Tenure & Satisfaction Score", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "03_tenure_satisfaction.png"), dpi=150)
    plt.close()
    print("[03] Saved: 03_tenure_satisfaction.png")


# ── 3.3  Pearson correlation heatmap ─────────────────────────────────────────
def plot_pearson(df: pd.DataFrame):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr = df[num_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
                center=0, linewidths=0.5, ax=ax, annot_kws={"size": 8})
    ax.set_title("Pearson Correlation Matrix (Numeric Features)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "03_pearson_heatmap.png"), dpi=150)
    plt.close()
    print("[03] Saved: 03_pearson_heatmap.png")


# ── 3.4  Cramér's V heatmap ──────────────────────────────────────────────────
def cramers_v(x, y):
    ct = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(ct)[0]
    n = ct.sum().sum()
    r, k = ct.shape
    v = np.sqrt(chi2 / (n * (min(r, k) - 1))) if min(r, k) > 1 else 0
    return min(v, 1.0)


def plot_cramers_v(df: pd.DataFrame):
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    # exclude CustomerID if present
    cat_cols = [c for c in cat_cols if c.lower() != "customerid"]
    n = len(cat_cols)
    mat = np.zeros((n, n))
    for i, c1 in enumerate(cat_cols):
        for j, c2 in enumerate(cat_cols):
            mat[i, j] = cramers_v(df[c1].fillna("_"), df[c2].fillna("_"))
    cramer_df = pd.DataFrame(mat, index=cat_cols, columns=cat_cols)
    mask = np.triu(np.ones_like(cramer_df, dtype=bool))
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cramer_df, mask=mask, annot=True, fmt=".2f", cmap="YlOrRd",
                vmin=0, vmax=1, linewidths=0.5, ax=ax, annot_kws={"size": 9})
    ax.set_title("Cramér's V — Categorical Feature Associations", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "03_cramers_v.png"), dpi=150)
    plt.close()
    print("[03] Saved: 03_cramers_v.png")


# ── 3.5  Point-biserial correlation ──────────────────────────────────────────
def plot_point_biserial(df: pd.DataFrame):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in ["Churn", "CustomerID"]]
    pb_results = []
    for col in num_cols:
        valid = df[[col, "Churn"]].dropna()
        r, p = stats.pointbiserialr(valid[col], valid["Churn"])
        pb_results.append({"Feature": col, "Correlation": r, "p_value": p})
    pb_df = pd.DataFrame(pb_results).sort_values("Correlation")

    colors = ["#22c55e" if c < 0 else "#ef4444" for c in pb_df["Correlation"]]
    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(pb_df["Feature"], pb_df["Correlation"], color=colors, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Point-Biserial Correlation: Numeric Features vs Churn",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Correlation Coefficient")
    for bar, val in zip(bars, pb_df["Correlation"]):
        ax.text(val + (0.005 if val >= 0 else -0.005), bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", ha="left" if val >= 0 else "right", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "03_point_biserial.png"), dpi=150)
    plt.close()
    print("[03] Saved: 03_point_biserial.png")
    return pb_df


# ── 3.6  Customer profiling ───────────────────────────────────────────────────
def plot_customer_profile(df: pd.DataFrame):
    cat_feats = ["Gender", "CityTier", "MaritalStatus", "Complain"]
    cat_feats = [c for c in cat_feats if c in df.columns]
    fig, axes = plt.subplots(1, len(cat_feats), figsize=(18, 6))
    for ax, feat in zip(axes, cat_feats):
        churn_rate = df.groupby(feat)["Churn"].mean() * 100
        churn_rate = churn_rate.sort_values(ascending=False)
        bars = ax.bar(churn_rate.index.astype(str), churn_rate.values,
                      color=["#ef4444" if v > 25 else "#f97316" if v > 15 else "#22c55e"
                             for v in churn_rate.values], edgecolor="white")
        for bar, val in zip(bars, churn_rate.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{val:.1f}%", ha="center", fontsize=9, fontweight="bold")
        ax.axhline(df["Churn"].mean() * 100, color="gray", linestyle="--",
                   alpha=0.7, linewidth=1)
        ax.set_title(feat, fontweight="bold", fontsize=12)
        ax.set_ylabel("Churn Rate (%)")
        ax.set_ylim(0, churn_rate.max() * 1.25)
        ax.tick_params(axis="x", rotation=15)
    plt.suptitle("Churn Rate by Customer Demographics", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "03_customer_profile.png"), dpi=150)
    plt.close()
    print("[03] Saved: 03_customer_profile.png")


# ── 3.7  Cashback vs Churn ───────────────────────────────────────────────
def plot_cashback_dist(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8,5))

    sns.violinplot(
        data=df,
        x="Churn",
        y="CashbackAmount",
        inner="quartile"
    )

    ax.set_xticklabels(["Retained", "Churned"])
    ax.set_title("CashbackAmount vs Churn")
    plt.savefig(os.path.join(OUT_DIR, "03_cashback_churn.png"), dpi=150)
    plt.close()
    print("[03] Saved: 03_cashback_churn.png")


def run_eda(path=CLEAN_PATH):
    df = load(path)
    print(f"\n{'='*60}")
    print("STEP 3: EXPLORATORY DATA ANALYSIS")
    print(f"{'='*60}")
    print(f"  Shape: {df.shape}")
    print(f"  Churn rate: {df['Churn'].mean()*100:.1f}%")
    plot_class_distribution(df)
    plot_tenure_satisfaction(df)
    plot_pearson(df)
    plot_cramers_v(df)
    pb_df = plot_point_biserial(df)
    plot_customer_profile(df)
    plot_cashback_dist(df)
    print("\n[03] EDA complete.")
    return df, pb_df


if __name__ == "__main__":
    run_eda()




