"""
Step 7: Clustering — K-Means on test set
- Elbow Method (WCSS) to find optimal K
- Silhouette Score validation
- K-Means cluster profiling: count, churn_rate, avg_churn_prob, avg_tenure
- Strategic recommendations per cluster
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score

DATA_DIR  = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
OUT_DIR   = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUT_DIR, exist_ok=True)


def load():
    X_test_raw = pd.read_csv(os.path.join(DATA_DIR, "X_test_raw.csv"))
    y_test     = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).squeeze()
    preds_df   = pd.read_csv(os.path.join(DATA_DIR, "predictions.csv"))
    return X_test_raw, y_test, preds_df


def scale_for_clustering(X_test_raw: pd.DataFrame) -> np.ndarray:
    """Scale numeric features only for K-Means."""
    num_cols = X_test_raw.select_dtypes(include=[np.number]).columns.tolist()
    scaler   = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_test_raw[num_cols].fillna(X_test_raw[num_cols].median()))
    return X_scaled, num_cols


def find_optimal_k(X_scaled: np.ndarray, k_range=range(2, 11)):
    """Elbow + Silhouette to select K."""
    wcss_list  = []
    sil_list   = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        wcss_list.append(km.inertia_)
        sil_list.append(silhouette_score(X_scaled, labels))

    # Plot Elbow + Silhouette side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(list(k_range), wcss_list, "o-", color="#3b82f6", linewidth=2, markersize=7)
    ax1.set_title("Elbow Method (WCSS)", fontweight="bold")
    ax1.set_xlabel("Number of Clusters (K)")
    ax1.set_ylabel("WCSS / Inertia")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    ax2.plot(list(k_range), sil_list, "s-", color="#22c55e", linewidth=2, markersize=7)
    ax2.set_title("Silhouette Score", fontweight="bold")
    ax2.set_xlabel("Number of Clusters (K)")
    ax2.set_ylabel("Silhouette Score")
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    best_k = list(k_range)[np.argmax(sil_list)]
    ax2.axvline(best_k, color="#ef4444", linestyle="--", label=f"Best K={best_k}")
    ax2.legend()

    plt.suptitle("Optimal K Selection for K-Means Clustering", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "07_elbow_silhouette.png"), dpi=150)
    plt.close()
    print(f"[07] Saved: 07_elbow_silhouette.png")
    print(f"[07] Best K by silhouette: {best_k} (score={max(sil_list):.4f})")
    return best_k, sil_list, wcss_list
def auto_name_cluster(row: pd.Series, top_diff: list) -> dict:
    """
    Tự động đặt tên persona dựa trên top features phân biệt cluster.
    Trả về {'persona': str, 'strategy': str, 'action_high': str, 'action_medium': str}
    """
    churn_pct = row.get("churn_rate_pct", 0)

    # Đọc giá trị các feature phân biệt
    complain_col   = next((c for c in row.index if "complain" in c.lower()), None)
    tenure_col     = next((c for c in row.index if "tenure" in c.lower() and "mean" in c.lower()), None)
    cashback_col   = next((c for c in row.index if "cashback" in c.lower()), None)
    citytier_col   = next((c for c in row.index if "citytier" in c.lower()), None)

    is_complainer  = (row[complain_col]  > 0.4) if complain_col  else False
    is_long_tenure = (row[tenure_col]   > 12)   if tenure_col   else False
    high_cashback  = (row[cashback_col] > 160)   if cashback_col else False
    is_urban       = (str(row.get(citytier_col, "2")) in ["1", "2"]) if citytier_col else False

    # ── Phân loại 4 persona chuẩn ────────────────────────────────────────────
    if churn_pct > 35 and is_complainer and not is_long_tenure:
        return {
            "persona":        "Săn Sale & Phàn nàn",
            "persona_emoji":  "🔴",
            "strategy":       "🔴 Critical: Gửi ngay Cashback Voucher + Gọi xử lý khiếu nại",
            "action_high":    "💸 Auto-Refund: Gửi ngay High-Cashback Voucher qua SMS",
            "action_medium":  "🎯 Gửi thông báo Flash Sale ưu đãi sớm",
        }
    elif churn_pct > 20 and is_complainer and is_urban:
        return {
            "persona":        "Đô thị & Phàn nàn",
            "persona_emoji":  "🟠",
            "strategy":       "🟠 High-Risk Urban: Gọi điện xử lý khiếu nại ngay",
            "action_high":    "🚨 CSKH Khẩn: Gọi điện xử lý khiếu nại (Đô thị)",
            "action_medium":  "📧 Gửi email hướng dẫn sử dụng / Hỗ trợ kỹ thuật",
        }
    elif churn_pct < 15 and is_long_tenure and high_cashback:
        return {
            "persona":        "VIP Ổn định",
            "persona_emoji":  "🟢",
            "strategy":       "🟢 Loyal: VIP perks + Cross-sell / Up-sell",
            "action_high":    "🎁 VIP Care: Liên hệ cá nhân, tặng quà tri ân",
            "action_medium":  "⭐ Đề xuất nâng cấp gói hội viên (Up-sell)",
        }
    else:
        return {
            "persona":        "Đô thị Trung thành",
            "persona_emoji":  "🔵",
            "strategy":       "🔵 Moderate: Freeship định kỳ + Flash sale",
            "action_high":    "⚠️ Risk Alert: Gửi email thăm dò mức độ hài lòng",
            "action_medium":  "💳 Gửi mã Freeship định kỳ",
        }


def profile_clusters(X_test_raw, y_test, preds_df, cluster_labels, K):
    # ... (giữ nguyên phần profile gốc) ...
    profile = X_test_raw.copy()
    profile["cluster"]           = cluster_labels
    profile["y_true"]            = y_test.values
    profile["churn_probability"] = preds_df["churn_probability"].values

    agg_dict = {"y_true": ["count", "mean"], "churn_probability": ["mean"]}
    num_cols = X_test_raw.select_dtypes(include=[np.number]).columns.tolist()
    for col in num_cols:
        agg_dict[col] = "mean"

    summary_raw = profile.groupby("cluster").agg(agg_dict).round(2)
    summary_raw.columns = ["_".join(c).strip("_") for c in summary_raw.columns]
    summary_raw = summary_raw.reset_index()

    mean_cols = [c for c in summary_raw.columns
                 if c not in ["cluster", "y_true_count", "y_true_mean", "churn_probability_mean"]]
    variance  = summary_raw[mean_cols].var()
    top_diff  = variance.nlargest(5).index.tolist()

    summary = summary_raw.rename(columns={
        "y_true_count":           "count",
        "y_true_mean":            "churn_rate",
        "churn_probability_mean": "avg_churn_prob",
    })
    summary["churn_rate_pct"] = (summary["churn_rate"] * 100).round(1)

    # ── THAY strategy cũ bằng auto_name_cluster ──────────────────────────────
    persona_rows = summary.apply(lambda row: auto_name_cluster(row, top_diff), axis=1)
    persona_df   = pd.DataFrame(persona_rows.tolist())
    summary      = pd.concat([summary.reset_index(drop=True),
                               persona_df.reset_index(drop=True)], axis=1)

    print("\n[07] ── Cluster Personas (auto-detected) ──────────────────")
    print(summary[["cluster", "count", "churn_rate_pct",
                   "persona_emoji", "persona", "strategy"]].to_string(index=False))

    return profile, summary, top_diff


def plot_cluster_churn(summary, top_diff):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    colors = ["#ef4444" if v > 40 else "#f97316" if v > 15 else "#22c55e"
              for v in summary["churn_rate_pct"]]

    # Panel 1: Churn rate (giữ nguyên)
    bars = axes[0].bar(summary["cluster"].astype(str),
                       summary["churn_rate_pct"],
                       color=colors, edgecolor="white", width=0.5)
    for bar, val, cnt in zip(bars, summary["churn_rate_pct"], summary["count"]):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                     f"{val:.1f}%\n(n={cnt})", ha="center", fontsize=10, fontweight="bold")
    axes[0].set_title("Churn Rate per Cluster", fontweight="bold")
    axes[0].set_ylabel("Churn Rate (%)")
    axes[0].set_xlabel("Cluster")

    # Panel 2-6: Top 5 features phân biệt clusters
    palette = ["#3b82f6", "#6366f1", "#8b5cf6", "#a855f7"]
    for i, feat in enumerate(top_diff[:5]):
        ax = axes[i + 1]
        col_name = f"{feat}_mean" if f"{feat}_mean" in summary.columns else feat
        if col_name not in summary.columns:
            continue
        ax.bar(summary["cluster"].astype(str), summary[col_name],
               color=palette[:len(summary)], edgecolor="white", width=0.5)
        for bar, val in zip(ax.patches, summary[col_name]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
                    f"{val:.2f}", ha="center", fontsize=10, fontweight="bold")
        ax.set_title(f"Avg {feat}", fontweight="bold")
        ax.set_xlabel("Cluster")

    plt.suptitle("K-Means Cluster Profiling — Key Differentiators",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "07_cluster_profiling.png"), dpi=150)
    plt.close()
    print("[07] Saved: 07_cluster_profiling.png")


def run_clustering():
    print(f"\n{'='*60}")
    print("STEP 7: CLUSTERING (K-MEANS)")
    print(f"{'='*60}")

    X_test_raw, y_test, preds_df = load()
    X_scaled, _ = scale_for_clustering(X_test_raw)

    # Find optimal K
    K, sil_list, wcss_list = find_optimal_k(X_scaled, k_range=range(2, 8))
    K = max(K, 3)  # ensure at least 3 clusters for business logic

    # Fit final K-Means
    km = KMeans(n_clusters=K, random_state=42, n_init=10)
    cluster_labels = km.fit_predict(X_scaled)
    print(f"[07] Final K-Means: K={K} | Cluster sizes: "
          f"{pd.Series(cluster_labels).value_counts().sort_index().tolist()}")

    # Profile clusters
    profile, summary, top_diff = profile_clusters(X_test_raw, y_test, preds_df, cluster_labels, K)
    plot_cluster_churn(summary, top_diff)

    # Save
    profile.to_csv(os.path.join(DATA_DIR, "cluster_profiles.csv"), index=False)
    summary.to_csv(os.path.join(DATA_DIR, "cluster_summary.csv"), index=False)
    joblib.dump(km, os.path.join(MODEL_DIR, "kmeans_model.pkl"))
    print(f"[07] Saved: cluster_profiles.csv, cluster_summary.csv, kmeans_model.pkl")

    return profile, summary, km


if __name__ == "__main__":
    run_clustering()
