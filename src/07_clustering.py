"""
Step 7: Cluster Profiling (Business Reporting Layer)
============================================================
KMeans KHÔNG fit ở đây. Model (kmeans_fe_model.pkl) đã fit ở Step 5 trên
TRAIN → ở bước này chỉ predict cụm cho test rồi THỐNG KÊ.

Phạm vi (cố ý giữ ĐƠN GIẢN — chỉ thống kê, không "AI tự đặt persona"):
  1. Load KMeans đã fit + test set
  2. Gán nhãn cụm cho test (predict, cùng feature space lúc train → no leakage)
  3. Tính churn rate + trung bình mỗi feature theo cụm
  4. Vẽ biểu đồ mô tả (overview + numeric + categorical, cả mean và phân phối)
  5. Lưu cluster_profiles.csv, cluster_summary.csv

LƯU Ý: cụm để tên trung tính "Cluster 0..K-1". Việc DIỄN GIẢI (đặt tên nhóm,
chọn chiến lược) dành cho người đọc biểu đồ — đáng tin hơn một hàm if-else đoán.
Step 8 đọc cluster_summary.csv vẫn chạy: thiếu cột action_* thì nó tự dùng
giá trị mặc định.

Outputs (outputs/{dataset}/):
  07_cluster_profiling.png            — churn rate + top-5 feature phân biệt
  07_cluster_numeric_features.png     — mean mỗi numeric feature / cụm
  07_cluster_categorical_features.png — % share mỗi categorical feature / cụm
  07_cluster_numeric_dist.png         — violin phân phối numeric / cụm
  07_cluster_categorical_dist.png     — stacked count categorical / cụm
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import joblib

from pipeline_config import DATA_DIR, MODEL_DIR, OUT_DIR   # noqa: E402

# Bảng màu cụm (đủ tới 8 cụm, tự lặp nếu nhiều hơn)
CLUSTER_PAL = ["#3b82f6", "#6366f1", "#8b5cf6", "#a855f7",
               "#ec4899", "#f97316", "#22c55e", "#14b8a6"]
# Bảng màu cho giá trị category (tới 10 mức)
CAT_PAL = ["#3b82f6", "#22c55e", "#f59e0b", "#ef4444", "#8b5cf6",
           "#ec4899", "#14b8a6", "#f97316", "#6366f1", "#84cc16"]

N_COLS = 3  # số cột lưới subplot


# ─────────────────────────────────────────────────────────────────────────────
# Load
# ─────────────────────────────────────────────────────────────────────────────
def load():
    """
    X_test_raw : test features dạng người-đọc-được (sau FE + was_missing flags)
    X_test_enc : test features đã encode — đúng feature space KMeans train ở Step 5
    y_test     : nhãn thật
    preds_df   : churn_probability từ Step 6
    km         : KMeans fit sẵn ở Step 5 (KHÔNG refit)
    """
    X_test_raw = pd.read_csv(os.path.join(DATA_DIR, "X_test_raw.csv"))
    X_test_enc = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
    y_test     = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).squeeze()
    preds_df   = pd.read_csv(os.path.join(DATA_DIR, "predictions.csv"))
    km         = joblib.load(os.path.join(MODEL_DIR, "kmeans_fe_model.pkl"))
    return X_test_raw, X_test_enc, y_test, preds_df, km


# ─────────────────────────────────────────────────────────────────────────────
# Thống kê theo cụm  (KHÔNG đặt persona tự động)
# ─────────────────────────────────────────────────────────────────────────────
def profile_clusters(X_test_raw, y_test, preds_df, cluster_labels, K):
    """
    Trả về:
      profile  : từng dòng test + cột cluster / y_true / churn_probability
      summary  : một dòng / cụm — count, churn_rate(_pct), avg_churn_prob,
                 và trung bình mỗi numeric feature ({col}_mean)
      top_diff : 5 feature phân biệt cụm nhất (theo phương sai của mean giữa cụm)
    """
    profile = X_test_raw.copy()
    profile["cluster"]           = cluster_labels
    profile["y_true"]            = y_test.values
    profile["churn_probability"] = preds_df["churn_probability"].values

    # Tổng hợp theo cụm
    agg = {"y_true": ["count", "mean"], "churn_probability": ["mean"]}
    num_cols = X_test_raw.select_dtypes(include=[np.number]).columns.tolist()
    for c in num_cols:
        agg[c] = "mean"

    summary = profile.groupby("cluster").agg(agg).round(2)
    summary.columns = ["_".join(c).strip("_") for c in summary.columns]
    summary = summary.reset_index().rename(columns={
        "y_true_count":           "count",
        "y_true_mean":            "churn_rate",
        "churn_probability_mean": "avg_churn_prob",
    })
    summary["churn_rate_pct"] = (summary["churn_rate"] * 100).round(1)
    # Nhãn trung tính — KHÔNG suy diễn persona ở đây
    summary["persona"] = "Cluster " + summary["cluster"].astype(str)

    # Top-5 feature phân biệt cụm nhất (phương sai của mean giữa các cụm)
    mean_cols = [c for c in summary.columns if c.endswith("_mean")]
    top_diff  = [c.replace("_mean", "")
                 for c in summary[mean_cols].var().nlargest(5).index.tolist()]

    # In bảng gọn
    print("\n[07] ── Cluster summary (test set) ───────────────────────────────")
    print(summary[["cluster", "count", "churn_rate_pct", "avg_churn_prob"]]
          .to_string(index=False))
    print(f"[07] Top-5 feature phân biệt cụm: {top_diff}")
    return profile, summary, top_diff


# ─────────────────────────────────────────────────────────────────────────────
# Chart 1 — Overview: churn rate + top-5 differentiators
# ─────────────────────────────────────────────────────────────────────────────
def plot_cluster_overview(summary, top_diff, K):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    x_labels = summary["cluster"].astype(str).tolist()

    # Panel 0 — churn rate / cụm (xanh thấp, cam vừa, đỏ cao — theo NGƯỠNG TƯƠNG ĐỐI)
    churn = summary["churn_rate_pct"]
    hi, lo = churn.quantile(0.66), churn.quantile(0.34)
    colors = ["#ef4444" if v >= hi else "#22c55e" if v <= lo else "#f97316"
              for v in churn]
    bars = axes[0].bar(x_labels, churn, color=colors, edgecolor="white", width=0.5)
    for bar, val, cnt in zip(bars, churn, summary["count"]):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                     f"{val:.1f}%\n(n={cnt})", ha="center",
                     fontsize=10, fontweight="bold")
    axes[0].set_title("Churn Rate per Cluster", fontweight="bold", pad=12)
    axes[0].set_ylabel("Churn Rate (%)"); axes[0].set_xlabel("Cluster")
    ymax = churn.max()
    axes[0].set_ylim(0, ymax * 1.25 if ymax > 0 else 1)

    # Panel 1-5 — top-5 feature phân biệt
    palette = ["#3b82f6", "#6366f1", "#8b5cf6", "#a855f7", "#ec4899"]
    for i, feat in enumerate(top_diff[:5]):
        ax = axes[i + 1]
        col = f"{feat}_mean"
        if col not in summary.columns:
            ax.set_visible(False); continue
        ax.bar(x_labels, summary[col],
               color=palette[:len(summary)], edgecolor="white", width=0.5)
        for bar, val in zip(ax.patches, summary[col]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                    f"{val:.2f}", ha="center", fontsize=10, fontweight="bold")
        ax.set_title(f"Avg {feat}", fontweight="bold", pad=12)
        ax.set_xlabel("Cluster")
        cmax = summary[col].max()
        ax.set_ylim(0, cmax * 1.25 if cmax > 0 else 1)

    plt.suptitle(f"K-Means Cluster Profiling — Key Differentiators\n"
                 f"(K={K}, model trained in Step 5 on training data)",
                 fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(os.path.join(OUT_DIR, "07_cluster_profiling.png"), dpi=150)
    plt.close()
    print("[07] Saved: 07_cluster_profiling.png")


# ─────────────────────────────────────────────────────────────────────────────
# Chart 2&3 — mean mỗi feature / cụm
# ─────────────────────────────────────────────────────────────────────────────
def plot_all_features(summary, profile, K):
    bar_colors = CLUSTER_PAL[:K]
    x_labels   = summary["cluster"].astype(str).tolist()
    skip = {"cluster", "count", "churn_rate", "churn_rate_pct",
            "avg_churn_prob", "persona"}

    # --- Numeric: mean / cụm ---
    feat_cols = sorted([c for c in summary.columns
                        if c.endswith("_mean") and c not in skip],
                       key=str.lower)
    n_feat = len(feat_cols)
    n_rows = max(1, -(-n_feat // N_COLS))
    fig1, axes1 = plt.subplots(n_rows, N_COLS, figsize=(18, n_rows * 4.2),
                               squeeze=False)
    axes1 = axes1.flatten()
    for idx, col in enumerate(feat_cols):
        ax   = axes1[idx]
        vals = summary[col].fillna(0).tolist()
        vmax = max(vals) if max(vals) != 0 else 1
        bars = ax.bar(x_labels, vals, color=bar_colors, edgecolor="white", width=0.55)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + vmax * 0.02,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_title(col.replace("_mean", ""), fontweight="bold", fontsize=11, pad=12)
        ax.set_xlabel("Cluster", fontsize=9); ax.set_ylabel("Mean Value", fontsize=9)
        ax.set_ylim(0, vmax * 1.30); ax.grid(axis="y", linestyle="--", alpha=0.35)
    for j in range(n_feat, len(axes1)):
        axes1[j].set_visible(False)
    fig1.legend(handles=[Patch(facecolor=bar_colors[c], label=f"Cluster {c}")
                         for c in range(K)],
                loc="lower center", ncol=K, fontsize=11,
                framealpha=0.9, bbox_to_anchor=(0.5, -0.01))
    fig1.suptitle(f"Cluster Profiling — All Numeric Features (Mean per Cluster)\n"
                  f"K={K}  |  {n_feat} features",
                  fontsize=14, fontweight="bold", y=0.995)
    fig1.tight_layout(rect=[0, 0.04, 1, 0.93])
    fig1.savefig(os.path.join(OUT_DIR, "07_cluster_numeric_features.png"),
                 dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print("[07] Saved: 07_cluster_numeric_features.png")

    # --- Categorical: % share / cụm ---
    cat_cols = [c for c in profile.select_dtypes(include=["object", "category"]).columns
                if c != "cluster"]
    if not cat_cols:
        print("[07] No categorical features — skipping categorical mean chart.")
        return
    n_cat   = len(cat_cols)
    n_rows2 = max(1, -(-n_cat // N_COLS))
    fig2, axes2 = plt.subplots(n_rows2, N_COLS, figsize=(18, n_rows2 * 5.5),
                               squeeze=False)
    axes2 = axes2.flatten()
    for idx, col in enumerate(cat_cols):
        ax     = axes2[idx]
        ct     = profile.groupby(["cluster", col]).size().unstack(fill_value=0)
        ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
        bottoms = np.zeros(len(ct_pct))
        clust_x = ct_pct.index.astype(str).tolist()
        for cat_val, color in zip(ct_pct.columns, CAT_PAL):
            vals = ct_pct[cat_val].values
            ax.bar(clust_x, vals, bottom=bottoms, color=color,
                   edgecolor="white", width=0.55, label=str(cat_val))
            for x_i, (val, bot) in enumerate(zip(vals, bottoms)):
                if val >= 8:
                    ax.text(x_i, bot + val / 2, f"{val:.0f}%", ha="center",
                            va="center", fontsize=8.5, fontweight="bold", color="white")
            bottoms += vals
        ax.set_title(col, fontweight="bold", fontsize=11, pad=12)
        ax.set_xlabel("Cluster", fontsize=9)
        ax.set_ylabel("Share per Cluster (%)", fontsize=9)
        ax.set_ylim(0, 115); ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.85,
                  title=col, title_fontsize=8)
    for j in range(n_cat, len(axes2)):
        axes2[j].set_visible(False)
    fig2.suptitle(f"Cluster Profiling — All Categorical Features (% share per Cluster)\n"
                  f"K={K}  |  {n_cat} features",
                  fontsize=14, fontweight="bold", y=0.995)
    fig2.tight_layout(rect=[0, 0, 1, 0.93])
    fig2.savefig(os.path.join(OUT_DIR, "07_cluster_categorical_features.png"),
                 dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print("[07] Saved: 07_cluster_categorical_features.png")


# ─────────────────────────────────────────────────────────────────────────────
# Chart 4&5 — phân phối theo cụm (violin numeric + stacked count categorical)
# ─────────────────────────────────────────────────────────────────────────────
def plot_feature_distributions(profile, K):
    bar_colors  = CLUSTER_PAL[:K]
    cluster_ids = sorted(profile["cluster"].unique().tolist())
    meta = {"cluster", "y_true", "churn_probability"}

    # --- Numeric: violin ---
    num_cols = sorted(c for c in profile.select_dtypes(include=[np.number]).columns
                      if c not in meta)
    n_num  = len(num_cols)
    n_rows = max(1, -(-n_num // N_COLS))
    fig1, axes1 = plt.subplots(n_rows, N_COLS, figsize=(18, n_rows * 4.8),
                               squeeze=False)
    axes1 = axes1.flatten()
    for idx, col in enumerate(num_cols):
        ax     = axes1[idx]
        data   = [profile.loc[profile["cluster"] == c, col].dropna().values
                  for c in cluster_ids]
        counts = [len(d) for d in data]
        # mỗi violin cần ≥2 điểm
        drawable = [d if len(d) >= 2 else
                    (np.array([d[0], d[0]]) if len(d) == 1 else np.array([0.0, 0.0]))
                    for d in data]
        vp = ax.violinplot(drawable, positions=cluster_ids,
                           showmedians=True, showmeans=False, widths=0.65)
        for body, color in zip(vp["bodies"], bar_colors):
            body.set_facecolor(color); body.set_alpha(0.72)
            body.set_edgecolor("white"); body.set_linewidth(1.2)
        vp["cmedians"].set_color("#0f172a"); vp["cmedians"].set_linewidth(2.2)
        for key in ("cbars", "cmins", "cmaxes"):
            if key in vp:
                vp[key].set_color("#475569"); vp[key].set_linewidth(1.0)
        y_top    = max((d.max() if len(d) else 0) for d in data)
        y_bottom = min((d.min() if len(d) else 0) for d in data)
        y_range  = (y_top - y_bottom) if (y_top - y_bottom) != 0 else (abs(y_top) or 1)
        for c_id, cnt in zip(cluster_ids, counts):
            ax.text(c_id, y_top + y_range * 0.06, f"n={cnt:,}", ha="center",
                    va="bottom", fontsize=8.5, fontweight="bold", color="#1e293b")
        ax.set_ylim(y_bottom - y_range * 0.05, y_top + y_range * 0.22)
        ax.set_title(col, fontweight="bold", fontsize=11, pad=12)
        ax.set_xlabel("Cluster", fontsize=9); ax.set_ylabel("Feature Value", fontsize=9)
        ax.set_xticks(cluster_ids)
        ax.set_xticklabels([f"Cluster {c}" for c in cluster_ids], fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
    for j in range(n_num, len(axes1)):
        axes1[j].set_visible(False)
    fig1.legend(handles=[Patch(facecolor=bar_colors[i], label=f"Cluster {i}")
                         for i in range(K)],
                loc="lower center", ncol=K, fontsize=11,
                framealpha=0.9, bbox_to_anchor=(0.5, -0.01))
    fig1.suptitle(f"Feature Distribution per Cluster — Numeric Features\n"
                  f"K={K}  |  x = cluster  |  violin width ∝ sample density  |  {n_num} features",
                  fontsize=13, fontweight="bold", y=0.995)
    fig1.tight_layout(rect=[0, 0.04, 1, 0.93])
    fig1.savefig(os.path.join(OUT_DIR, "07_cluster_numeric_dist.png"),
                 dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print("[07] Saved: 07_cluster_numeric_dist.png")

    # --- Categorical: stacked count ---
    cat_cols = [c for c in profile.select_dtypes(include=["object", "category"]).columns
                if c not in meta]
    if not cat_cols:
        print("[07] No categorical features — skipping categorical dist chart.")
        return
    n_cat   = len(cat_cols)
    n_rows2 = max(1, -(-n_cat // N_COLS))
    fig2, axes2 = plt.subplots(n_rows2, N_COLS, figsize=(18, n_rows2 * 5.8),
                               squeeze=False)
    axes2 = axes2.flatten()
    for idx, col in enumerate(cat_cols):
        ax      = axes2[idx]
        ct      = profile.groupby(["cluster", col]).size().unstack(fill_value=0)
        bottoms = np.zeros(len(ct))
        clust_x = ct.index.astype(str).tolist()
        totals  = ct.sum(axis=1).values
        for cat_val, color in zip(ct.columns, CAT_PAL):
            vals = ct[cat_val].values.astype(float)
            ax.bar(clust_x, vals, bottom=bottoms, color=color,
                   edgecolor="white", width=0.55, label=str(cat_val))
            for x_i, (val, tot, bot) in enumerate(zip(vals, totals, bottoms)):
                if tot > 0 and val / tot >= 0.05:
                    ax.text(x_i, bot + val / 2, f"{int(val):,}", ha="center",
                            va="center", fontsize=8.5, fontweight="bold", color="white")
            bottoms += vals
        for x_i, tot in enumerate(totals):
            ax.text(x_i, tot + max(totals) * 0.015, f"n={int(tot):,}", ha="center",
                    va="bottom", fontsize=9.5, fontweight="bold", color="#0f172a")
        ax.set_title(col, fontweight="bold", fontsize=11, pad=12)
        ax.set_xlabel("Cluster", fontsize=9)
        ax.set_ylabel("Number of Samples", fontsize=9)
        ax.set_ylim(0, max(totals) * 1.25); ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.85,
                  title=col, title_fontsize=8)
    for j in range(n_cat, len(axes2)):
        axes2[j].set_visible(False)
    fig2.suptitle(f"Feature Distribution per Cluster — Categorical Features\n"
                  f"K={K}  |  x = cluster  |  y = number of samples  |  {n_cat} features",
                  fontsize=13, fontweight="bold", y=0.995)
    fig2.tight_layout(rect=[0, 0, 1, 0.93])
    fig2.savefig(os.path.join(OUT_DIR, "07_cluster_categorical_dist.png"),
                 dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print("[07] Saved: 07_cluster_categorical_dist.png")


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────
def run_clustering():
    print(f"\n{'='*60}\nSTEP 7: CLUSTER PROFILING (statistics only)\n{'='*60}")

    X_test_raw, X_test_enc, y_test, preds_df, km = load()
    K = km.n_clusters
    print(f"[07] Loaded kmeans_fe_model.pkl (K={K}, fitted in Step 5)")

    # Gán cụm — cùng feature space lúc train, không refit, không leakage
    cluster_labels = km.predict(X_test_enc.values)
    sizes = pd.Series(cluster_labels).value_counts().sort_index().tolist()
    print(f"[07] Cluster sizes (test): {sizes}")

    profile, summary, top_diff = profile_clusters(
        X_test_raw, y_test, preds_df, cluster_labels, K)

    plot_cluster_overview(summary, top_diff, K)
    plot_all_features(summary, profile, K)
    plot_feature_distributions(profile, K)

    profile.to_csv(os.path.join(DATA_DIR, "cluster_profiles.csv"), index=False)
    summary.to_csv(os.path.join(DATA_DIR, "cluster_summary.csv"),  index=False)
    print(f"[07] Saved: cluster_profiles.csv, cluster_summary.csv")
    print(f"\n{'='*60}\n[07] ✓ Xong (thống kê thuần, persona để người đọc tự diễn giải).\n{'='*60}\n")
    return profile, summary, km


if __name__ == "__main__":
    run_clustering()