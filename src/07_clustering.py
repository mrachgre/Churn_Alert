"""
Step 7: Cluster Profiling (Business Reporting Layer)
============================================================
K-Means is NO LONGER fitted here.
The model (kmeans_fe_model.pkl) was fitted in Step 5 on training
data only, ensuring no test-set leakage.

This step:
  1. Loads the pre-fitted KMeans model from Step 5
  2. Assigns cluster labels to the test set (using X_test.csv,
     the same encoded feature space the model was trained on)
  3. Profiles each cluster with X_test_raw.csv (human-readable)
  4. Generates persona labels and strategic recommendations
  5. Saves cluster_profiles.csv, cluster_summary.csv and THREE charts:
       07_cluster_profiling.png          — churn rate + top-5 differentiators
       07_cluster_numeric_features.png   — ALL numeric features (mean/cluster)
       07_cluster_categorical_features.png — ALL categorical features (%/cluster)
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

from pipeline_config import DATA_DIR, MODEL_DIR, OUT_DIR   # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────────────

def load():
    """
    Returns
    -------
    X_test_raw : pd.DataFrame   raw (unencoded) test features for display
    X_test_enc : pd.DataFrame   encoded test features — must match the feature
                                space KMeans was trained on in Step 5
    y_test     : pd.Series
    preds_df   : pd.DataFrame   churn probabilities from Step 6
    km         : KMeans         model fitted in Step 5 (no re-fitting here)
    """
    X_test_raw = pd.read_csv(os.path.join(DATA_DIR, "X_test_raw.csv"))
    X_test_enc = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
    y_test     = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).squeeze()
    preds_df   = pd.read_csv(os.path.join(DATA_DIR, "predictions.csv"))
    km         = joblib.load(os.path.join(MODEL_DIR, "kmeans_fe_model.pkl"))
    return X_test_raw, X_test_enc, y_test, preds_df, km


# ─────────────────────────────────────────────────────────────────────────────
# Auto-naming personas
# ─────────────────────────────────────────────────────────────────────────────

def auto_name_cluster(row: pd.Series, top_diff: list) -> dict:
    """
    Automatically assign a business persona based on top differentiating
    features.  Returns {'persona', 'persona_emoji', 'strategy',
    'action_high', 'action_medium'}.
    """
    churn_pct = row.get("churn_rate_pct", 0)

    complain_col  = next((c for c in row.index if "complain" in c.lower()), None)
    tenure_col    = next((c for c in row.index if "tenure"   in c.lower()
                          and "mean" in c.lower()), None)
    cashback_col  = next((c for c in row.index if "cashback" in c.lower()), None)
    citytier_col  = next((c for c in row.index if "citytier" in c.lower()), None)

    is_complainer  = (row[complain_col]  > 0.4) if complain_col  else False
    is_long_tenure = (row[tenure_col]   > 12)   if tenure_col   else False
    high_cashback  = (row[cashback_col] > 160)   if cashback_col else False
    is_urban       = (str(row.get(citytier_col, "2")) in ["1", "2"]) \
                     if citytier_col else False

    if churn_pct > 35 and is_complainer and not is_long_tenure:
        return {
            "persona":       "Săn Sale & Phàn nàn",
            "persona_emoji": "🔴",
            "strategy":      "🔴 Critical: Gửi ngay Cashback Voucher + Gọi xử lý khiếu nại",
            "action_high":   "💸 Auto-Refund: Gửi ngay High-Cashback Voucher qua SMS",
            "action_medium": "🎯 Gửi thông báo Flash Sale ưu đãi sớm",
        }
    elif churn_pct > 20 and is_complainer and is_urban:
        return {
            "persona":       "Đô thị & Phàn nàn",
            "persona_emoji": "🟠",
            "strategy":      "🟠 High-Risk Urban: Gọi điện xử lý khiếu nại ngay",
            "action_high":   "🚨 CSKH Khẩn: Gọi điện xử lý khiếu nại (Đô thị)",
            "action_medium": "📧 Gửi email hướng dẫn sử dụng / Hỗ trợ kỹ thuật",
        }
    elif churn_pct < 15 and is_long_tenure and high_cashback:
        return {
            "persona":       "VIP Ổn định",
            "persona_emoji": "🟢",
            "strategy":      "🟢 Loyal: VIP perks + Cross-sell / Up-sell",
            "action_high":   "🎁 VIP Care: Liên hệ cá nhân, tặng quà tri ân",
            "action_medium": "⭐ Đề xuất nâng cấp gói hội viên (Up-sell)",
        }
    else:
        return {
            "persona":       "Đô thị Trung thành",
            "persona_emoji": "🔵",
            "strategy":      "🔵 Moderate: Freeship định kỳ + Flash sale",
            "action_high":   "⚠️ Risk Alert: Gửi email thăm dò mức độ hài lòng",
            "action_medium": "💳 Gửi mã Freeship định kỳ",
        }


# ─────────────────────────────────────────────────────────────────────────────
# Profiling
# ─────────────────────────────────────────────────────────────────────────────

def profile_clusters(X_test_raw, y_test, preds_df, cluster_labels, K):
    """Build per-cluster summary with churn rates and auto-detected personas."""
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
                 if c not in ["cluster", "y_true_count", "y_true_mean",
                               "churn_probability_mean"]]
    variance = summary_raw[mean_cols].var()
    top_diff = variance.nlargest(5).index.tolist()

    summary = summary_raw.rename(columns={
        "y_true_count":           "count",
        "y_true_mean":            "churn_rate",
        "churn_probability_mean": "avg_churn_prob",
    })
    summary["churn_rate_pct"] = (summary["churn_rate"] * 100).round(1)

    persona_rows = summary.apply(lambda row: auto_name_cluster(row, top_diff), axis=1)
    persona_df   = pd.DataFrame(persona_rows.tolist())
    summary      = pd.concat([summary.reset_index(drop=True),
                               persona_df.reset_index(drop=True)], axis=1)

    print("\n[07] ── Cluster Personas (K-Means model from Step 5) ─────────────")
    print(summary[["cluster", "count", "churn_rate_pct",
                   "persona_emoji", "persona", "strategy"]].to_string(index=False))
    return profile, summary, top_diff


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def plot_cluster_churn(summary, top_diff, K):
    n_panels = min(6, 1 + len(top_diff))
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    colors = ["#ef4444" if v > 40 else "#f97316" if v > 15 else "#22c55e"
              for v in summary["churn_rate_pct"]]

    # Panel 0 — Churn rate per cluster
    bars = axes[0].bar(
        summary["cluster"].astype(str),
        summary["churn_rate_pct"],
        color=colors, edgecolor="white", width=0.5,
    )
    for bar, val, cnt in zip(bars, summary["churn_rate_pct"], summary["count"]):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
            f"{val:.1f}%\n(n={cnt})", ha="center", fontsize=10, fontweight="bold",
        )
    axes[0].set_title("Churn Rate per Cluster", fontweight="bold", pad=12)
    axes[0].set_ylabel("Churn Rate (%)")
    axes[0].set_xlabel("Cluster")
    # Extra headroom so the value/count annotation doesn't collide with the title
    y_max = summary["churn_rate_pct"].max()
    axes[0].set_ylim(0, y_max * 1.25 if y_max > 0 else 1)

    # Panels 1-5 — Top 5 differentiating features
    palette = ["#3b82f6", "#6366f1", "#8b5cf6", "#a855f7", "#ec4899"]
    for i, feat in enumerate(top_diff[:5]):
        ax      = axes[i + 1]
        col_name = f"{feat}_mean" if f"{feat}_mean" in summary.columns else feat
        if col_name not in summary.columns:
            ax.set_visible(False)
            continue
        ax.bar(summary["cluster"].astype(str), summary[col_name],
               color=palette[:len(summary)], edgecolor="white", width=0.5)
        for bar, val in zip(ax.patches, summary[col_name]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                    f"{val:.2f}", ha="center", fontsize=10, fontweight="bold")
        ax.set_title(f"Avg {feat}", fontweight="bold", pad=12)
        ax.set_xlabel("Cluster")
        # Extra headroom for the value annotation above each bar
        col_max = summary[col_name].max()
        ax.set_ylim(0, col_max * 1.25 if col_max > 0 else 1)

    plt.suptitle(
        f"K-Means Cluster Profiling — Key Differentiators\n"
        f"(K={K}, model trained in Step 5 on training data)",
        fontsize=14, fontweight="bold", y=0.99,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(os.path.join(OUT_DIR, "07_cluster_profiling.png"), dpi=150)
    plt.close()
    print("[07] Saved: 07_cluster_profiling.png")


def plot_all_features(summary: pd.DataFrame, profile: pd.DataFrame, K: int):
    """
    Two comprehensive figures showing ALL features across clusters.

    Figure 1 — 07_cluster_numeric_features.png
      Mean value per cluster for every numeric feature (17 features).
      Layout: ceil(n/3) rows × 3 columns.

    Figure 2 — 07_cluster_categorical_features.png
      Stacked proportion bar charts (%) for every categorical feature (6 features).
      Layout: ceil(n/3) rows × 3 columns.
    """
    from matplotlib.patches import Patch

    # Cluster colour palette (supports up to 8 clusters)
    CLUSTER_PAL = ["#3b82f6", "#6366f1", "#8b5cf6", "#a855f7",
                   "#ec4899", "#f97316", "#22c55e", "#14b8a6"]
    bar_colors  = CLUSTER_PAL[:K]
    x_labels    = summary["cluster"].astype(str).tolist()

    # Columns to skip — already plotted or non-feature
    _SKIP = {"cluster", "count", "churn_rate", "churn_rate_pct",
             "avg_churn_prob", "churn_probability_mean",
             "persona", "persona_emoji", "strategy",
             "action_high", "action_medium"}

    # ──────────────────────────────────────────────────────────────────────────────
    # Figure 1: ALL NUMERIC FEATURES
    # ──────────────────────────────────────────────────────────────────────────────
    feat_cols = sorted(
        [c for c in summary.columns
         if c.endswith("_mean") and c not in _SKIP],
        key=lambda c: c.lower(),
    )

    N_COLS = 3
    n_feat = len(feat_cols)
    n_rows = max(1, -(-n_feat // N_COLS))   # ceiling division

    fig1, axes1 = plt.subplots(
        n_rows, N_COLS,
        figsize=(18, n_rows * 4.2),
        squeeze=False,
    )
    axes1_flat = axes1.flatten()

    for idx, col in enumerate(feat_cols):
        ax        = axes1_flat[idx]
        feat_name = col.replace("_mean", "")
        vals      = summary[col].fillna(0).tolist()
        max_val   = max(vals) if max(vals) != 0 else 1

        bars = ax.bar(
            x_labels, vals,
            color=bar_colors, edgecolor="white", width=0.55,
        )
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max_val * 0.02,
                f"{val:.2f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold",
            )

        ax.set_title(feat_name, fontweight="bold", fontsize=11, pad=12)
        ax.set_xlabel("Cluster", fontsize=9)
        ax.set_ylabel("Mean Value", fontsize=9)
        # Extra headroom above the bars so value labels don't collide with the title
        ax.set_ylim(0, max_val * 1.30)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.tick_params(labelsize=9)

    # Hide unused panels
    for j in range(n_feat, len(axes1_flat)):
        axes1_flat[j].set_visible(False)

    # Shared cluster legend at bottom
    legend_handles = [Patch(facecolor=bar_colors[c], label=f"Cluster {c}")
                      for c in range(K)]
    fig1.legend(
        handles=legend_handles, loc="lower center",
        ncol=K, fontsize=11, framealpha=0.9,
        bbox_to_anchor=(0.5, -0.01),
    )

    fig1.suptitle(
        f"Cluster Profiling — All Numeric Features (Mean per Cluster)\n"
        f"K={K}  |  {n_feat} features",
        fontsize=14, fontweight="bold", y=0.995,
    )
    # Reserve ~7% of figure height at the top for the 2-line suptitle so it
    # doesn't overlap the title of the first row of subplots.
    fig1.tight_layout(rect=[0, 0.04, 1, 0.93])
    fig1.savefig(
        os.path.join(OUT_DIR, "07_cluster_numeric_features.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close(fig1)
    print("[07] Saved: 07_cluster_numeric_features.png")

    # ──────────────────────────────────────────────────────────────────────────────
    # Figure 2: ALL CATEGORICAL FEATURES
    # ──────────────────────────────────────────────────────────────────────────────
    cat_cols = [
        c for c in profile.select_dtypes(include=["object", "category"]).columns
        if c != "cluster"
    ]

    if not cat_cols:
        print("[07] No categorical features — skipping categorical chart.")
        return

    # Colour palette for category values (up to 10 distinct values)
    CAT_PAL = ["#3b82f6", "#22c55e", "#f59e0b", "#ef4444",
               "#8b5cf6", "#ec4899", "#14b8a6", "#f97316",
               "#6366f1", "#84cc16"]

    n_cat   = len(cat_cols)
    n_rows2 = max(1, -(-n_cat // N_COLS))

    fig2, axes2 = plt.subplots(
        n_rows2, N_COLS,
        figsize=(18, n_rows2 * 5.5),
        squeeze=False,
    )
    axes2_flat = axes2.flatten()

    for idx, col in enumerate(cat_cols):
        ax = axes2_flat[idx]

        # Cross-tab: cluster (rows) × category value (cols), normalised to %
        ct     = profile.groupby(["cluster", col]).size().unstack(fill_value=0)
        ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100

        category_values = ct_pct.columns.tolist()
        bottoms         = np.zeros(len(ct_pct))
        clust_x         = ct_pct.index.astype(str).tolist()

        for j, (cat_val, color) in enumerate(
            zip(category_values, CAT_PAL[:len(category_values)])
        ):
            vals = ct_pct[cat_val].values
            bars = ax.bar(
                clust_x, vals,
                bottom=bottoms,
                color=color, edgecolor="white", width=0.55,
                label=str(cat_val),
            )
            # Annotate segments wider than 8 %
            for bar, val, bot in zip(bars, vals, bottoms):
                if val >= 8:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bot + val / 2,
                        f"{val:.0f}%",
                        ha="center", va="center",
                        fontsize=8.5, fontweight="bold", color="white",
                    )
            bottoms += vals

        ax.set_title(col, fontweight="bold", fontsize=11, pad=12)
        ax.set_xlabel("Cluster", fontsize=9)
        ax.set_ylabel("Share per Cluster (%)", fontsize=9)
        ax.set_ylim(0, 115)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.legend(
            loc="upper right", fontsize=8,
            framealpha=0.85, title=col, title_fontsize=8,
        )
        ax.tick_params(labelsize=9)

    for j in range(n_cat, len(axes2_flat)):
        axes2_flat[j].set_visible(False)

    fig2.suptitle(
        f"Cluster Profiling — All Categorical Features (% share per Cluster)\n"
        f"K={K}  |  {n_cat} features",
        fontsize=14, fontweight="bold", y=0.995,
    )
    # Reserve top space for the 2-line suptitle.
    fig2.tight_layout(rect=[0, 0, 1, 0.93])
    fig2.savefig(
        os.path.join(OUT_DIR, "07_cluster_categorical_features.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close(fig2)
    print("[07] Saved: 07_cluster_categorical_features.png")


def plot_feature_distributions(profile: pd.DataFrame, K: int):
    """
    Count-based distribution charts — one plot per feature type.

    Numeric features  →  07_cluster_numeric_dist.png
      Violin plots: x = cluster (0 … K-1), violin width ∝ sample count
      Each violin shows the full empirical distribution of that feature
      within the cluster.  Median line + per-cluster sample count annotated.

    Categorical features  →  07_cluster_categorical_dist.png
      Stacked count bars: x = cluster, y = number of samples, stacked by
      category value.  Raw counts annotated inside each segment.
    """
    import matplotlib.patches as mpatches

    CLUSTER_PAL = ["#3b82f6", "#6366f1", "#8b5cf6", "#a855f7",
                   "#ec4899", "#f97316", "#22c55e", "#14b8a6"]
    bar_colors  = CLUSTER_PAL[:K]
    cluster_ids = sorted(profile["cluster"].unique().tolist())
    N_COLS      = 3

    # Columns that are meta / not raw features
    _META = {"cluster", "y_true", "churn_probability"}

    # ──────────────────────────────────────────────────────────────────────────
    # Figure 1 — NUMERIC FEATURES  (violin: x=cluster, width ∝ count)
    # ──────────────────────────────────────────────────────────────────────────
    num_cols = sorted(
        [c for c in profile.select_dtypes(include=[np.number]).columns
         if c not in _META],
    )
    n_num  = len(num_cols)
    n_rows = max(1, -(-n_num // N_COLS))

    fig1, axes1 = plt.subplots(
        n_rows, N_COLS,
        figsize=(18, n_rows * 4.8),
        squeeze=False,
    )
    axes1_flat = axes1.flatten()

    for idx, col in enumerate(num_cols):
        ax = axes1_flat[idx]

        # Per-cluster data arrays (drop NaN)
        data   = [profile.loc[profile["cluster"] == c, col].dropna().values
                  for c in cluster_ids]
        counts = [len(d) for d in data]

        # Require at least 2 points per cluster for a violin
        drawable = [d if len(d) >= 2 else np.array([d[0], d[0]]) if len(d) == 1
                    else np.array([0.0, 0.0])
                    for d in data]

        vp = ax.violinplot(
            drawable,
            positions=cluster_ids,
            showmedians=True,
            showmeans=False,
            widths=0.65,
        )
        # Colour each violin body
        for body, color in zip(vp["bodies"], bar_colors):
            body.set_facecolor(color)
            body.set_alpha(0.72)
            body.set_edgecolor("white")
            body.set_linewidth(1.2)
        vp["cmedians"].set_color("#0f172a")
        vp["cmedians"].set_linewidth(2.2)
        for key in ("cbars", "cmins", "cmaxes"):
            if key in vp:
                vp[key].set_color("#475569")
                vp[key].set_linewidth(1.0)

        # Annotate sample count above each violin.
        # y_pad is scaled by the data RANGE (not just y_top) so it works
        # correctly even when y_top is small, zero, or negative.
        y_top    = max((d.max() if len(d) > 0 else 0) for d in data)
        y_bottom = min((d.min() if len(d) > 0 else 0) for d in data)
        y_range  = (y_top - y_bottom) if (y_top - y_bottom) != 0 else (abs(y_top) or 1)
        y_pad    = y_range * 0.06

        for c_id, cnt in zip(cluster_ids, counts):
            ax.text(
                c_id, y_top + y_pad,
                f"n={cnt:,}",
                ha="center", va="bottom",
                fontsize=8.5, fontweight="bold", color="#1e293b",
            )

        # Reserve extra headroom above the annotation so it doesn't collide
        # with the subplot title.
        ax.set_ylim(y_bottom - y_range * 0.05, y_top + y_range * 0.22)

        ax.set_title(col, fontweight="bold", fontsize=11, pad=12)
        ax.set_xlabel("Cluster", fontsize=9)
        ax.set_ylabel("Feature Value", fontsize=9)
        ax.set_xticks(cluster_ids)
        ax.set_xticklabels([f"Cluster {c}" for c in cluster_ids], fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.35)

    # Hide empty panels
    for j in range(n_num, len(axes1_flat)):
        axes1_flat[j].set_visible(False)

    # Shared cluster legend
    legend_h = [mpatches.Patch(facecolor=bar_colors[i], label=f"Cluster {i}")
                for i in range(K)]
    fig1.legend(handles=legend_h, loc="lower center", ncol=K,
                fontsize=11, framealpha=0.9, bbox_to_anchor=(0.5, -0.01))

    fig1.suptitle(
        f"Feature Distribution per Cluster — Numeric Features\n"
        f"K={K}  |  x = cluster  |  violin width ∝ sample density  |  {n_num} features",
        fontsize=13, fontweight="bold", y=0.995,
    )
    # Reserve top space for the 2-line suptitle and bottom space for the legend.
    fig1.tight_layout(rect=[0, 0.04, 1, 0.93])
    fig1.savefig(
        os.path.join(OUT_DIR, "07_cluster_numeric_dist.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close(fig1)
    print("[07] Saved: 07_cluster_numeric_dist.png")

    # ──────────────────────────────────────────────────────────────────────────
    # Figure 2 — CATEGORICAL FEATURES  (stacked count bars: x=cluster, y=count)
    # ──────────────────────────────────────────────────────────────────────────
    cat_cols = [
        c for c in profile.select_dtypes(include=["object", "category"]).columns
        if c not in _META
    ]

    if not cat_cols:
        print("[07] No categorical features — skipping categorical dist chart.")
        return

    CAT_PAL = ["#3b82f6", "#22c55e", "#f59e0b", "#ef4444",
               "#8b5cf6", "#ec4899", "#14b8a6", "#f97316",
               "#6366f1", "#84cc16"]

    n_cat   = len(cat_cols)
    n_rows2 = max(1, -(-n_cat // N_COLS))

    fig2, axes2 = plt.subplots(
        n_rows2, N_COLS,
        figsize=(18, n_rows2 * 5.8),
        squeeze=False,
    )
    axes2_flat = axes2.flatten()

    for idx, col in enumerate(cat_cols):
        ax = axes2_flat[idx]

        # Raw count table: rows=cluster, cols=category value
        ct      = profile.groupby(["cluster", col]).size().unstack(fill_value=0)
        cat_vals = ct.columns.tolist()
        bottoms  = np.zeros(len(ct))
        clust_x  = ct.index.astype(str).tolist()
        totals   = ct.sum(axis=1).values

        for j, (cat_val, color) in enumerate(zip(cat_vals, CAT_PAL)):
            vals = ct[cat_val].values.astype(float)
            bars = ax.bar(
                clust_x, vals,
                bottom=bottoms,
                color=color, edgecolor="white", width=0.55,
                label=str(cat_val),
            )
            # Annotate count inside segment if large enough (> 5% of total)
            for bar, val, tot, bot in zip(bars, vals, totals, bottoms):
                if tot > 0 and val / tot >= 0.05:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bot + val / 2,
                        f"{int(val):,}",
                        ha="center", va="center",
                        fontsize=8.5, fontweight="bold", color="white",
                    )
            bottoms += vals

        # Total count on top of each bar stack
        for i, (x, tot) in enumerate(zip(clust_x, totals)):
            ax.text(
                i, tot + max(totals) * 0.015,
                f"n={int(tot):,}",
                ha="center", va="bottom",
                fontsize=9.5, fontweight="bold", color="#0f172a",
            )

        ax.set_title(col, fontweight="bold", fontsize=11, pad=12)
        ax.set_xlabel("Cluster", fontsize=9)
        ax.set_ylabel("Number of Samples", fontsize=9)
        # Extra headroom so the "n=" total annotation doesn't collide with the title
        ax.set_ylim(0, max(totals) * 1.25)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.legend(
            loc="upper right", fontsize=8,
            framealpha=0.85, title=col, title_fontsize=8,
        )
        ax.tick_params(labelsize=9)

    for j in range(n_cat, len(axes2_flat)):
        axes2_flat[j].set_visible(False)

    fig2.suptitle(
        f"Feature Distribution per Cluster — Categorical Features\n"
        f"K={K}  |  x = cluster  |  y = number of samples  |  {n_cat} features",
        fontsize=13, fontweight="bold", y=0.995,
    )
    # Reserve top space for the 2-line suptitle.
    fig2.tight_layout(rect=[0, 0, 1, 0.93])
    fig2.savefig(
        os.path.join(OUT_DIR, "07_cluster_categorical_dist.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close(fig2)
    print("[07] Saved: 07_cluster_categorical_dist.png")

# ─────────────────────────────────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────────────────────────────────

def run_clustering():
    print(f"\n{'='*60}")
    print("STEP 7: CLUSTER PROFILING (Business Reporting)")
    print(f"{'='*60}")

    X_test_raw, X_test_enc, y_test, preds_df, km = load()
    K = km.n_clusters
    print(f"[07] Loaded kmeans_fe_model.pkl  (K={K}, fitted in Step 5)")

    # ── Assign cluster labels using the pre-fitted model ─────────────────────
    # X_test_enc shares the same feature space as the training data used to
    # fit the KMeans — no re-fitting, no leakage.
    cluster_labels = km.predict(X_test_enc.values)
    cluster_counts = pd.Series(cluster_labels).value_counts().sort_index().tolist()
    print(f"[07] Cluster sizes (test set): {cluster_counts}")

    # ── Profile & visualise ───────────────────────────────────────────────────
    profile, summary, top_diff = profile_clusters(
        X_test_raw, y_test, preds_df, cluster_labels, K)

    # Chart 1: churn rate + top-5 differentiators (overview)
    plot_cluster_churn(summary, top_diff, K)

    # Chart 2 & 3: mean value per feature, per cluster
    plot_all_features(summary, profile, K)

    # Chart 4 & 5: sample-COUNT distributions (violin + stacked bars)
    plot_feature_distributions(profile, K)

    # ── Save ─────────────────────────────────────────────────────────────────
    profile.to_csv(os.path.join(DATA_DIR, "cluster_profiles.csv"), index=False)
    summary.to_csv(os.path.join(DATA_DIR, "cluster_summary.csv"),  index=False)
    print(f"[07] Saved: cluster_profiles.csv, cluster_summary.csv")

    return profile, summary, km


if __name__ == "__main__":
    run_clustering()