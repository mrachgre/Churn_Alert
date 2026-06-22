"""
Step 4: RFM Segmentation using SQLite
- Map R → DaySinceLastOrder, F → OrderCount, M → CashbackAmount
- NTILE(5) window function scoring in SQLite
- CASE WHEN segment labelling
- Churn rate per segment
- Add rfm_total + rfm_segment back to DataFrame
"""

import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
import os
import warnings
warnings.filterwarnings("ignore")

from pipeline_config import DATA_DIR, OUT_DIR   # noqa: E402
CLEAN_PATH = os.path.join(DATA_DIR, "ecommerce_churn_clean.csv")
RFM_PATH   = os.path.join(DATA_DIR, "ecommerce_churn_rfm.csv")


def load(path=CLEAN_PATH):
    df = pd.read_csv(path, dtype={"CityTier": str, "Complain": str})
    df["Churn"] = df["Churn"].astype(int)
    return df


def score_rfm_sql(df: pd.DataFrame) -> pd.DataFrame:
    """
    Use SQLite + NTILE(5) to score each customer on R, F, M.
    R: DaySinceLastOrder (lower days = more recent = higher score → ORDER DESC)
    F: OrderCount        (higher = better → ORDER ASC)
    M: CashbackAmount    (higher = better → ORDER ASC)
    """
    # Work on rows where key columns are non-null
    df_sql = df.copy()

    conn = sqlite3.connect(":memory:")
    df_sql.to_sql("customers", conn, if_exists="replace", index=False)

    query = """
    SELECT
        CustomerID,
        NTILE(5) OVER (ORDER BY DaySinceLastOrder DESC NULLS LAST) AS R_score,
        NTILE(5) OVER (ORDER BY OrderCount         ASC  NULLS LAST) AS F_score,
        NTILE(5) OVER (ORDER BY CashbackAmount     ASC  NULLS LAST) AS M_score
    FROM customers
    """
    scores = pd.read_sql(query, conn)
    conn.close()

    df_sql = df_sql.merge(scores, on="CustomerID", how="left")
    df_sql["rfm_total"] = df_sql["R_score"] + df_sql["F_score"] + df_sql["M_score"]
    print(f"[04] RFM scores computed. rfm_total range: "
          f"{df_sql['rfm_total'].min():.0f} – {df_sql['rfm_total'].max():.0f}")
    return df_sql


def assign_segments(df: pd.DataFrame) -> pd.DataFrame:
    """Assign business segment labels via rule-based CASE WHEN logic."""
    def segment_label(row):
        r, f, m = row["R_score"], row["F_score"], row["M_score"]
        if pd.isna(r) or pd.isna(f) or pd.isna(m):
            return "Unknown"
        if f >= 4 and m >= 4:
            return "Champions"
        elif f >= 3 and m >= 3:
            return "Loyal"
        elif r >= 4 and f < 3:
            return "Recent Customers"
        elif r <= 2 and f >= 3:
            return "At Risk"
        else:
            return "Hibernating"

    df = df.copy()
    df["rfm_segment"] = df.apply(segment_label, axis=1)
    print(f"[04] Segment distribution:\n{df['rfm_segment'].value_counts().to_string()}\n")
    return df


def plot_rfm_churn(df: pd.DataFrame):
    """Bar chart: churn rate per RFM segment."""
    seg_stats = (df.groupby("rfm_segment")
                   .agg(count=("Churn", "count"),
                        churn_rate=("Churn", "mean"))
                   .reset_index()
                   .sort_values("churn_rate", ascending=False))
    seg_stats["churn_pct"] = seg_stats["churn_rate"] * 100

    colors = ["#ef4444" if v > 20 else "#f97316" if v > 12 else "#22c55e"
              for v in seg_stats["churn_pct"]]
    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(seg_stats["rfm_segment"], seg_stats["churn_pct"],
                  color=colors, edgecolor="white", width=0.6)
    for bar, val, cnt in zip(bars, seg_stats["churn_pct"], seg_stats["count"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                f"{val:.1f}%\n(n={cnt:,})", ha="center", fontsize=10, fontweight="bold")
    ax.axhline(df["Churn"].mean() * 100, color="gray", linestyle="--",
               linewidth=1, label=f"Overall avg {df['Churn'].mean()*100:.1f}%")
    ax.set_title("Churn Rate by RFM Segment", fontsize=14, fontweight="bold")
    ax.set_xlabel("RFM Segment")
    ax.set_ylabel("Churn Rate (%)")
    ax.legend()
    ax.set_ylim(0, seg_stats["churn_pct"].max() * 1.30)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "04_rfm_churn_rate.png"), dpi=150)
    plt.close()
    print("[04] Saved: 04_rfm_churn_rate.png")


def plot_rfm_distribution(df: pd.DataFrame):
    """Pie chart: RFM segment distribution."""
    counts = df["rfm_segment"].value_counts()
    colors = ["#3b82f6", "#22c55e", "#f97316", "#ef4444", "#a855f7"]
    fig, ax = plt.subplots(figsize=(8, 7))
    wedges, texts, autotexts = ax.pie(
        counts.values, labels=counts.index, colors=colors[:len(counts)],
        autopct="%1.1f%%", startangle=140, pctdistance=0.75,
        wedgeprops={"edgecolor": "white", "linewidth": 2})
    for t in autotexts:
        t.set_fontsize(10)
    ax.set_title("Customer Distribution by RFM Segment", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "04_rfm_segments_pie.png"), dpi=150)
    plt.close()
    print("[04] Saved: 04_rfm_segments_pie.png")


def plot_rfm_total_vs_churn(df: pd.DataFrame):
    """Scatter: rfm_total vs churn probability (binned)."""
    bins = list(range(3, 17))
    df2 = df.copy()
    rfm_churn = df2.groupby("rfm_total")["Churn"].mean() * 100
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(rfm_churn.index, rfm_churn.values, "o-", color="#3b82f6",
            linewidth=2, markersize=7)
    ax.fill_between(rfm_churn.index, rfm_churn.values, alpha=0.15, color="#3b82f6")
    ax.set_title("Churn Rate vs RFM Total Score", fontsize=14, fontweight="bold")
    ax.set_xlabel("RFM Total Score (3–15)")
    ax.set_ylabel("Churn Rate (%)")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "04_rfm_total_vs_churn.png"), dpi=150)
    plt.close()
    print("[04] Saved: 04_rfm_total_vs_churn.png")


def evaluate_rfm_as_predictor(df: pd.DataFrame):
    """Use rfm_total as standalone churn predictor; compute ROC-AUC."""
    valid = df[["rfm_total", "Churn"]].dropna()
    # Invert: higher rfm_total → lower churn → use negative as churn score
    score = -valid["rfm_total"]
    auc = roc_auc_score(valid["Churn"], score)
    print(f"[04] RFM standalone ROC-AUC: {auc:.4f}  "
          f"(reference: XGBoost ≈ 0.989)")
    return auc


def run_rfm(path=CLEAN_PATH, save=True):
    df = load(path)
    print(f"\n{'='*60}")
    print("STEP 4: RFM SEGMENTATION")
    print(f"{'='*60}")
    df = score_rfm_sql(df)
    df = assign_segments(df)
    plot_rfm_churn(df)
    plot_rfm_distribution(df)
    plot_rfm_total_vs_churn(df)
    evaluate_rfm_as_predictor(df)
    if save:
        df.to_csv(RFM_PATH, index=False)
        print(f"[04] Saved with RFM features → {RFM_PATH}")
    return df


if __name__ == "__main__":
    run_rfm()
