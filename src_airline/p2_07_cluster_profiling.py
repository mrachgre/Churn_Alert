"""
p2_07_cluster_profiling.py  —  Step 7: Passenger Personas
==========================================================
Load KMeans (no refit), assign cluster to test set via predict().
Profile each cluster and assign persona name based on real data.
"""
import pandas as pd
import numpy as np
import os, joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def _auto_persona(row: dict, k: int) -> tuple[str, str, str]:
    """
    Rule-based persona naming from cluster profile values.
    Returns (emoji, persona_name, strategy)
    """
    g = row.get("ground_experience_score", 3)
    inf = row.get("inflight_experience_score", 3)
    dsv = row.get("delay_severity", 0)
    disat = row.get("dissatisfied_rate_pct", 50)
    biz_pct = row.get("biz_travel_pct", 0)
    biz_class_pct = row.get("class_business_pct", 0)

    if disat < 30:
        return "🟢", "Hài Lòng Toàn Diện", "Upsell + Loyalty"
    if disat > 70:
        if dsv > np.percentile([dsv], 50):
            return "🔴", "Nhạy Cảm Với Trễ Giờ", "Ưu tiên đền bù delay + xin lỗi"
        return "🔴", "Trải Nghiệm Kém", "Khẩn cấp: service recovery toàn diện"
    if g < 2.5:
        return "🟡", "Trải Nghiệm Mặt Đất Kém", "Cải thiện check-in & boarding"
    if inf < 2.5:
        return "🟡", "Trải Nghiệm Trên Chuyến Bay Kém", "Cải thiện cabin service & wifi"
    if biz_pct > 70:
        return "🔵", "Khách Công Vụ Không Hài Lòng", "VIP recovery: nâng hạng, phòng chờ"
    return "🟠", "Rủi Ro Trung Bình", "Theo dõi + ưu đãi chuyến sau"


def run_step7(cfg, step5_result: dict, step6_result: dict) -> pd.DataFrame:
    sep = "=" * 70
    print(f"\n{sep}\nSTEP 7: CLUSTER PROFILING — PASSENGER PERSONAS\n{sep}")

    X_te_clust = step5_result["X_test_clust"].copy()
    y_test     = step5_result["y_test"]
    km         = joblib.load(os.path.join(cfg.MODEL_DIR, "kmeans_model.pkl"))
    km_scaler  = joblib.load(os.path.join(cfg.MODEL_DIR, "kmeans_scaler.pkl"))
    best_k     = step5_result["best_k"]

    # Load test_fe to get original column values for profiling
    test_fe = pd.read_parquet(os.path.join(cfg.DATA_INT_DIR, "test_fe.parquet"))
    test_fe["cluster"] = X_te_clust["kmeans_cluster_id"].values
    test_fe["target"]  = y_test
    test_fe["dissatisfied"] = 1 - y_test

    profile_rows = []
    for c in range(best_k):
        mask = test_fe["cluster"] == c
        grp  = test_fe[mask]
        n    = len(grp)

        def mn(col):
            return round(grp[col].mean(), 3) if col in grp.columns else None

        def pct_val(col, val):
            if col not in grp.columns:
                return 0
            return round((grp[col] == val).mean() * 100, 1)

        row = {
            "cluster": c,
            "count":   n,
            "dissatisfied_rate_pct":        round(grp["dissatisfied"].mean() * 100, 1),
            "ground_experience_score":      mn("ground_experience_score"),
            "inflight_experience_score":    mn("inflight_experience_score"),
            "delay_severity":               mn("delay_severity"),
            "age_mean":                     mn("Age"),
            "flight_dist_mean":             mn("Flight Distance"),
            "biz_travel_pct":               pct_val("biz_travel", 1),
            "loyal_pct":                    pct_val("loyal", 1),
            "class_business_pct":           pct_val("Class_ord", 2),
            "class_eco_pct":                pct_val("Class_ord", 0),
        }
        emoji, persona, strategy = _auto_persona(row, best_k)
        row["persona_emoji"] = emoji
        row["persona"]       = persona
        row["strategy"]      = strategy
        profile_rows.append(row)

    cluster_df = pd.DataFrame(profile_rows).sort_values("dissatisfied_rate_pct", ascending=False)
    cluster_df.to_csv(os.path.join(cfg.DATA_DIR, "cluster_summary.csv"), index=False)

    # ── Print profile ─────────────────────────────────────────────────────────
    print(f"\n  KMeans K={best_k}, Test set: {len(test_fe):,} rows")
    print(f"\n  {'CL':>3}  {'N':>6}  {'Disat%':>7}  {'Ground':>7}  "
          f"{'Inflight':>9}  {'DelaySev':>9}  {'BizTravel%':>11}  Persona")
    print("  " + "─" * 90)
    for _, r in cluster_df.iterrows():
        print(f"  {int(r['cluster']):>3}  {int(r['count']):>6,}  "
              f"{r['dissatisfied_rate_pct']:>6.1f}%  "
              f"{r['ground_experience_score']:>7.3f}  "
              f"{r['inflight_experience_score']:>9.3f}  "
              f"{r['delay_severity']:>9.5f}  "
              f"{r['biz_travel_pct']:>10.1f}%  "
              f"{r['persona_emoji']} {r['persona']}")

    # ── Cluster profiling chart ───────────────────────────────────────────────
    radar_cols = ["ground_experience_score", "inflight_experience_score",
                  "delay_severity", "age_mean", "flight_dist_mean"]
    radar_avail = [c for c in radar_cols if c in cluster_df.columns]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    palette = ["#22c55e", "#3b82f6", "#f59e0b", "#ef4444",
               "#a855f7", "#ec4899", "#06b6d4"]

    metrics = [
        ("dissatisfied_rate_pct",     "Dissatisfied Rate (%)"),
        ("ground_experience_score",   "Ground Experience Score"),
        ("inflight_experience_score", "Inflight Experience Score"),
        ("biz_travel_pct",            "Business Travel (%)"),
    ]
    for ax, (col, title) in zip(axes, metrics):
        if col not in cluster_df.columns:
            continue
        colors = [palette[int(c)] for c in cluster_df["cluster"]]
        labels = [f"C{int(c)}: {p}" for c, p in zip(
            cluster_df["cluster"], cluster_df["persona"])]
        ax.bar(labels, cluster_df[col], color=colors, edgecolor="white")
        ax.set_title(title, fontweight="bold", fontsize=10)
        ax.tick_params(axis="x", rotation=25, labelsize=7)
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle(f"Step 7: Passenger Cluster Profiling (K={best_k})",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(cfg.OUT_DIR, "p2_07_cluster_profiling.png")
    plt.savefig(out, dpi=130, bbox_inches="tight"); plt.close()
    print(f"\n  Chart saved → {out}")
    print(f"  cluster_summary.csv saved → {cfg.DATA_DIR}")

    print(f"\n{'─'*70}\nSTEP 7 DONE\n{'─'*70}")
    return cluster_df


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import src_airline.airline_config as cfg
    print("Run via run_airline_ml.py")
