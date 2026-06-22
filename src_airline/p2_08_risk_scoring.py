"""
p2_08_risk_scoring.py  —  Step 8: Dissatisfaction Risk Score + Action Matrix
=============================================================================
Formula (mirror Churn Alert, cashback_pct → passenger_value_proxy):

  passenger_value_proxy = (Class_ord/2 + biz_travel) / 2
  RawScore  = 0.7 * P(dissatisfied) + 0.3 * passenger_value_proxy
  RiskScore = min-max normalize(RawScore)   # by test set population

Quintile thresholds (p20, p40, p60, p80) → 5 risk categories.
Action Matrix: Risk Category × Cluster → priority + action.
"""
import pandas as pd
import numpy as np
import os, joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def run_step8(cfg, step5_result: dict, step6_result: dict,
              cluster_df: pd.DataFrame) -> pd.DataFrame:
    sep = "=" * 70
    print(f"\n{sep}\nSTEP 8: DISSATISFACTION RISK SCORING + ACTION MATRIX\n{sep}")

    X_te_clust = step5_result["X_test_clust"].copy()
    y_test     = step5_result["y_test"]
    model      = step6_result["xgb_clust"]

    # ── 8.1  P(dissatisfied) from XGB-Clust ──────────────────────────────────
    print(f"\n[8.1] Computing P(dissatisfied) with XGB-Clust...")
    proba_sat    = model.predict_proba(X_te_clust)[:, 1]   # P(satisfied=1)
    p_dissatisfied = 1 - proba_sat

    # ── 8.2  passenger_value_proxy ────────────────────────────────────────────
    class_ord  = X_te_clust["Class_ord"].values   if "Class_ord"  in X_te_clust.columns else np.zeros(len(X_te_clust))
    biz_travel = X_te_clust["biz_travel"].values  if "biz_travel" in X_te_clust.columns else np.zeros(len(X_te_clust))
    pv_proxy   = (class_ord / 2 + biz_travel) / 2

    print(f"  passenger_value_proxy stats:")
    print(f"    min={pv_proxy.min():.3f}  mean={pv_proxy.mean():.3f}  max={pv_proxy.max():.3f}")

    # ── 8.3  RawScore & RiskScore ─────────────────────────────────────────────
    raw_score = 0.7 * p_dissatisfied + 0.3 * pv_proxy
    rs_min, rs_max = raw_score.min(), raw_score.max()
    risk_score = (raw_score - rs_min) / (rs_max - rs_min)

    p20, p40, p60, p80 = np.percentile(risk_score, [20, 40, 60, 80])
    print(f"\n  RiskScore distribution:")
    print(f"    min={rs_min:.4f}  max={rs_max:.4f}")
    print(f"    p20={p20:.4f}  p40={p40:.4f}  p60={p60:.4f}  p80={p80:.4f}")

    def cat(rs):
        if rs <= p20: return "Very Low"
        if rs <= p40: return "Low"
        if rs <= p60: return "Medium"
        if rs <= p80: return "High"
        return "Critical"

    risk_cats = np.array([cat(r) for r in risk_score])

    print(f"\n  Risk category distribution:")
    for rc in ["Very Low", "Low", "Medium", "High", "Critical"]:
        n   = (risk_cats == rc).sum()
        pct = n / len(risk_cats) * 100
        bar = "█" * int(pct / 2)
        print(f"  {rc:<10}: {n:>6,}  ({pct:.1f}%)  {bar}")

    # ── 8.4  Save meta + action list ─────────────────────────────────────────
    risk_meta = {
        "rs_min": float(rs_min), "rs_max": float(rs_max),
        "p20": float(p20), "p40": float(p40),
        "p60": float(p60), "p80": float(p80),
    }
    joblib.dump(risk_meta, os.path.join(cfg.MODEL_DIR, "risk_score_meta.pkl"))

    test_fe = pd.read_parquet(os.path.join(cfg.DATA_INT_DIR, "test_fe.parquet"))
    action_df = test_fe.reset_index(drop=True).copy()
    action_df["cluster"]        = X_te_clust["kmeans_cluster_id"].values
    action_df["y_true"]         = y_test
    action_df["p_dissatisfied"] = p_dissatisfied
    action_df["risk_score"]     = risk_score
    action_df["risk_category"]  = risk_cats

    # Action lookup from cluster summary
    cluster_map = dict(zip(cluster_df["cluster"], cluster_df["persona"]))
    action_df["persona"] = action_df["cluster"].map(cluster_map)

    # ── 8.5  Action Matrix ────────────────────────────────────────────────────
    print(f"\n[8.5] Action Matrix (Risk Category × Cluster Persona):")
    ACTION_MATRIX = {
        ("Critical",  True):  ("P1 — Ngay lập tức",   "📞 Gọi điện xin lỗi + Hoàn tiền 1 phần + Ưu đãi chuyến sau"),
        ("Critical",  False): ("P1 — Ngay lập tức",   "⚠️ Email xin lỗi + Voucher nâng hạng + Flash sale"),
        ("High",      True):  ("P2 — Trong 24h",       "🎯 Email cá nhân hóa + Đền bù delay nếu áp dụng"),
        ("High",      False): ("P2 — Trong 24h",       "💳 Discount chuyến sau + Khảo sát ngắn 3 câu"),
        ("Medium",    True):  ("P3 — Tuần tới",        "📧 Newsletter VIP + Survey mức độ hài lòng"),
        ("Medium",    False): ("P3 — Tuần tới",        "📧 Newsletter + Đề xuất dịch vụ phù hợp"),
        ("Low",       True):  ("P4 — Tháng tới",       "📊 Monitor + Ưu tiên check-in"),
        ("Low",       False): ("P4 — Tháng tới",       "📊 Monitor + Newsletter"),
        ("Very Low",  True):  ("P5 — Duy trì",         "✅ Chăm sóc tiêu chuẩn + Loyalty points"),
        ("Very Low",  False): ("P5 — Duy trì",         "✅ Chăm sóc tiêu chuẩn"),
    }

    header = f"  {'Risk':<12}  {'High-Value?':>12}  {'Priority':<24}  Action"
    print(header)
    print("  " + "─" * 90)
    for (rc, hv), (priority, action) in ACTION_MATRIX.items():
        hv_str = "Business/BizClass" if hv else "Eco/Personal"
        print(f"  {rc:<12}  {hv_str:>17}  {priority:<24}  {action}")

    # Assign action per row
    def get_action(row):
        is_high_value = (row.get("biz_travel", 0) == 1 or row.get("Class_ord", 0) == 2)
        rc = row["risk_category"]
        key = (rc, bool(is_high_value))
        return ACTION_MATRIX.get(key, ("P3", "Monitor"))[1]

    action_df["action"] = action_df.apply(get_action, axis=1)
    action_df.to_csv(os.path.join(cfg.DATA_DIR, "targeted_action_list.csv"), index=False)
    print(f"\n  targeted_action_list.csv saved → {cfg.DATA_DIR}")

    # ── 8.6  Risk score distribution chart ───────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # Histogram
    axes[0].hist(risk_score, bins=50, color="#3b82f6", edgecolor="none", alpha=0.85)
    for thr, label, color in [
        (p20, "p20", "#22c55e"), (p40, "p40", "#f59e0b"),
        (p60, "p60", "#f97316"), (p80, "p80", "#ef4444"),
    ]:
        axes[0].axvline(thr, color=color, linestyle="--", linewidth=1.5, label=f"{label}={thr:.3f}")
    axes[0].set_title("Risk Score Distribution (Test Set)")
    axes[0].set_xlabel("Risk Score"); axes[0].set_ylabel("Count")
    axes[0].legend(fontsize=8); axes[0].grid(axis="y", alpha=0.3)
    # Bar by cluster
    cluster_risk = action_df.groupby("cluster")["risk_score"].mean().reset_index()
    persona_map  = dict(zip(cluster_df["cluster"], cluster_df["persona"]))
    cluster_risk["persona"] = cluster_risk["cluster"].map(persona_map)
    palette = ["#22c55e", "#3b82f6", "#f59e0b", "#ef4444", "#a855f7"]
    colors = [palette[int(c) % len(palette)] for c in cluster_risk["cluster"]]
    axes[1].bar(cluster_risk["persona"], cluster_risk["risk_score"],
                color=colors, edgecolor="white")
    axes[1].set_title("Mean Risk Score by Cluster")
    axes[1].set_ylabel("Mean Risk Score")
    axes[1].tick_params(axis="x", rotation=30, labelsize=8)
    axes[1].grid(axis="y", alpha=0.3)
    plt.suptitle("Step 8: Dissatisfaction Risk Scoring", fontweight="bold")
    plt.tight_layout()
    out = os.path.join(cfg.OUT_DIR, "p2_08_risk_score.png")
    plt.savefig(out, dpi=130, bbox_inches="tight"); plt.close()
    print(f"  Chart saved → {out}")

    print(f"\n{'─'*70}\nSTEP 8 DONE\n{'─'*70}")
    return action_df


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import src_airline.airline_config as cfg
    print("Run via run_airline_ml.py")
