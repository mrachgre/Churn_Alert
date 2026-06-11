"""
Step 2: Missing Data Analysis
- Missingness matrix (missingno)
- Missing pattern heatmap
- MCAR / MAR / MNAR classification per column
- Skewness analysis on missing columns
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import os
import warnings
warnings.filterwarnings("ignore")

CLEAN_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "ecommerce_churn_clean.csv")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUT_DIR, exist_ok=True)


def load(path=CLEAN_PATH):
    return pd.read_csv(path, dtype={"CityTier": str, "Complain": str})

def plot_missingness_bar(df: pd.DataFrame):
    """Bar chart: % missing per column."""
    missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    missing_pct = missing_pct[missing_pct > 0]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(missing_pct.index, missing_pct.values, color="#ef4444")
    ax.set_xlabel("Missing (%)", fontsize=12)
    ax.set_title("Missing Data Rate by Column", fontsize=14, fontweight="bold")
    for bar, val in zip(bars, missing_pct.values):
        ax.text(val + 0.1, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "02_missing_bar.png"), dpi=150)
    plt.close()
    print(f"[02] Saved: 02_missing_bar.png")
    return missing_pct


def plot_missingness_matrix(df: pd.DataFrame):
    """missingno matrix: visual pattern of missing data."""
    fig, ax = plt.subplots(figsize=(14, 6))
    msno.matrix(df, ax=ax, sparkline=False, color=(0.93, 0.27, 0.27))
    ax.set_title("Missingness Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "02_missing_matrix.png"), dpi=150)
    plt.close()
    print(f"[02] Saved: 02_missing_matrix.png")


def mcar_mar_mnar_test(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each column with missing values, compare churn rate:
      - In rows WHERE the column is missing
      - In rows where it is NOT missing
    If churn_rate_missing >> churn_rate_present → likely MAR/MNAR
    """
    missing_cols = df.columns[df.isnull().any()].tolist()
    results = []
    for col in missing_cols:
        mask_missing = df[col].isnull()
        churn_missing = df.loc[mask_missing, "Churn"].astype(int).mean() if mask_missing.sum() > 0 else None
        churn_present = df.loc[~mask_missing, "Churn"].astype(int).mean()
        n_missing = mask_missing.sum()
        pct_missing = n_missing / len(df) * 100
        if churn_missing is not None:
            diff = abs(churn_missing - churn_present)
            classification = "MAR/MNAR" if diff > 0.05 else "MCAR"
        else:
            classification = "—"
        results.append({
            "Column": col,
            "N_Missing": n_missing,
            "Pct_Missing": round(pct_missing, 2),
            "Churn_Rate_Missing": round(churn_missing, 3) if churn_missing is not None else None,
            "Churn_Rate_Present": round(churn_present, 3),
            "Classification": classification,
        })
    result_df = pd.DataFrame(results).sort_values("N_Missing", ascending=False)
    print("\n[02] MCAR / MAR / MNAR Classification:")
    print(result_df.to_string(index=False))
    return result_df


def plot_distribution_missing_cols(df: pd.DataFrame, missing_cols: list):
    """Distribution plots for columns with missing data."""
    n = len(missing_cols)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(16, 8))
    axes = axes.flatten()
    for i, col in enumerate(missing_cols):
        data = df[col].dropna()
        skewness = data.skew()
        axes[i].hist(data, bins=30, color="#3b82f6", alpha=0.8, edgecolor="white")
        axes[i].set_title(f"{col}\n(skew={skewness:.2f})", fontsize=10)
        axes[i].set_xlabel("Value")
        axes[i].set_ylabel("Count")
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle("Distribution of Columns with Missing Values", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "02_missing_distributions.png"), dpi=150)
    plt.close()
    print(f"[02] Saved: 02_missing_distributions.png")


def analyze_missing(path=CLEAN_PATH):
    df = load(path)
    print(f"\n{'='*60}")
    print("STEP 2: MISSING DATA ANALYSIS")
    print(f"{'='*60}")
    missing_pct = plot_missingness_bar(df)
    plot_missingness_matrix(df)
    mcar_df = mcar_mar_mnar_test(df)
    missing_cols = missing_pct.index.tolist()
    num_missing_cols = [c for c in missing_cols if pd.api.types.is_numeric_dtype(df[c])]
    if num_missing_cols:
        plot_distribution_missing_cols(df, num_missing_cols)
    return df, mcar_df


if __name__ == "__main__":
    analyze_missing()
