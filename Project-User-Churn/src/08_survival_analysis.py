"""
Step 8: Actionable Risk Matrix
- Combine XGBoost Probability with K-Means Clusters
- Generate highly targeted intervention strategies
- Export actionable lists for CRM targeting
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import os
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUT_DIR  = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

HIGH_THRESH   = 0.70
MEDIUM_THRESH = 0.40

def load_data_step8():
    preds_df = pd.read_csv(os.path.join(DATA_DIR, "predictions.csv"))
    X_raw    = pd.read_csv(os.path.join(DATA_DIR, "X_test_raw.csv"))
    clusters = pd.read_csv(os.path.join(DATA_DIR, "cluster_profiles.csv"))[["cluster"]]
    return preds_df, X_raw, clusters

def build_action_matrix(preds_df, X_raw, clusters):
    """Load persona→action mapping from Step 7's cluster_summary.csv
    instead of hardcoding cluster IDs."""

    cluster_summary_path = os.path.join(DATA_DIR, "cluster_summary.csv")
    cluster_strategy_map = {}
    if os.path.exists(cluster_summary_path):
        cs = pd.read_csv(cluster_summary_path)
        for _, row in cs.iterrows():
            cid = int(row["cluster"])
            cluster_strategy_map[cid] = {
                "action_high":    row.get("action_high",    "⚠️ Liên hệ khách hàng ngay"),
                "action_medium":  row.get("action_medium",  "📧 Gửi email chăm sóc"),
                "persona":        row.get("persona",        f"Cluster {cid}"),
                "persona_emoji":  row.get("persona_emoji",  "⚪"),
            }
    else:
        print("[08] WARNING: cluster_summary.csv not found — actions will be generic.")

    df = preds_df.copy()
    df = pd.concat([
        df.reset_index(drop=True),
        clusters.reset_index(drop=True),
        X_raw.reset_index(drop=True),
    ], axis=1)

    df["risk_tier"] = np.where(
        df["churn_probability"] >= HIGH_THRESH,   "High Risk",
        np.where(df["churn_probability"] >= MEDIUM_THRESH, "Medium Risk", "Low Risk")
    )

    def assign_action(row):
        prob    = row["churn_probability"]
        cluster = int(row["cluster"]) if not pd.isna(row["cluster"]) else -1
        meta    = cluster_strategy_map.get(cluster, {})
        if prob >= HIGH_THRESH:
            return meta.get("action_high",   "🚨 Liên hệ ngay")
        elif prob >= MEDIUM_THRESH:
            return meta.get("action_medium", "📧 Gửi email")
        return "✅ Duy trì chăm sóc tiêu chuẩn"

    def get_persona_label(c):
        if pd.isna(c):
            return "Unknown"
        meta = cluster_strategy_map.get(int(c), {})
        emoji = meta.get("persona_emoji", "⚪")
        name  = meta.get("persona", f"Cluster {int(c)}")
        return f"{emoji} {name}"

    df["persona"]            = df["cluster"].apply(get_persona_label)
    df["recommended_action"] = df.apply(assign_action, axis=1)
    return df


def run_step8():
    print(f"\n{'='*60}")
    print("STEP 8: ACTIONABLE RISK MATRIX (XGBoost x K-Means)")
    print(f"{'='*60}")

    preds_df, X_raw, clusters = load_data_step8()
    df = build_action_matrix(preds_df, X_raw, clusters)
    
    # Thống kê chiến dịch
    action_stats = df.groupby(["risk_tier", "cluster", "recommended_action"]).size().reset_index(name='count')
    action_stats = action_stats[action_stats['risk_tier'] == 'High Risk'].sort_values("count", ascending=False)
    
    print("\n[08] ── Danh sách ưu tiên can thiệp (High Risk) ──")
    for _, row in action_stats.iterrows():
        print(f"Cụm {row['cluster']} | {row['count']:>4} users | Hành động: {row['recommended_action']}")

    df.to_csv(os.path.join(DATA_DIR, "targeted_action_list.csv"), index=False)
    print(f"\n[08] Saved: targeted_action_list.csv")
    return df

"""
Step 9: Persona-Based Survival Analysis
- Kaplan-Meier curves specifically grouped by K-Means Clusters
- Identify the exact 'Drop-off month' for each persona
"""
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test

def plot_km_by_cluster(df):
    """Vẽ đường sinh tồn dựa trên Cụm Persona để tìm điểm rơi rụng"""
    
    # --- SỬA LỖI TẠI ĐÂY: Lọc bỏ các dòng bị khuyết (NaN) dữ liệu Tenure hoặc y_true ---
    df_clean = df.dropna(subset=["Tenure", "y_true"]).copy()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    T, E = df_clean["Tenure"], df_clean["y_true"]
    
    # Định nghĩa màu theo 4 cụm đã phân tích
    cluster_meta = {
        0: {"name": "Cụm 0 (Đô thị phàn nàn)", "color": "#f97316"},
        1: {"name": "Cụm 1 (VIP Tỉnh ổn định)", "color": "#3b82f6"},
        2: {"name": "Cụm 2 (Đô thị trung thành)", "color": "#22c55e"},
        3: {"name": "Cụm 3 (Săn Sale phàn nàn)", "color": "#ef4444"}
    }
    
    kmf = KaplanMeierFitter()
    
    for cluster_id in sorted(df_clean["cluster"].dropna().unique()):
        mask = df_clean["cluster"] == cluster_id
        meta = cluster_meta.get(cluster_id, {"name": f"Cluster {cluster_id}", "color": "gray"})
        
        kmf.fit(T[mask], E[mask], label=meta["name"])
        kmf.plot_survival_function(ax=ax, color=meta["color"], linewidth=2.5, ci_show=False)

    # Tính P-Value để chứng minh các cụm có hành vi rời bỏ khác biệt thực sự
    result = multivariate_logrank_test(T, df_clean["cluster"].fillna(-1), E)
    p_val  = result.p_value

    ax.set_title(f"Kaplan-Meier Survival Curves by Customer Persona\n(log-rank p-value: {p_val:.4e})", fontweight="bold")
    ax.set_xlabel("Tháng sử dụng (Tenure)")
    ax.set_ylabel("Tỉ lệ trụ lại (Survival Probability)")
    ax.axhline(0.70, color="gray", linestyle="--", linewidth=1.2, label="Ngưỡng báo động (70%)")
    
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(axis="both", linestyle="--", alpha=0.4)
    ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "08_km_by_cluster.png"), dpi=150)
    plt.close()
    print("[08] Saved: 08_km_by_cluster.png")


def run_step9(df_with_clusters):
    print(f"\n{'='*60}")
    print("STEP 9: PERSONA-BASED SURVIVAL ANALYSIS")
    print(f"{'='*60}")
    plot_km_by_cluster(df_with_clusters)


def run_survival_analysis():
    """Entry point called by run_pipeline.py.
    Runs Step 8 (Risk Action Matrix) then Step 9 (KM Survival by Cluster).
    """
    df_action = run_step8()
    run_step9(df_action)
    return df_action


if __name__ == "__main__":
    run_survival_analysis()