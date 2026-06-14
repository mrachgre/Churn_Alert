"""
Step 8: XGB-Clust Risk Scoring + Actionable Matrix
Step 9: Persona-Based Survival Analysis (Kaplan-Meier)
============================================================
Changes from original:
  - Uses xgb_clust_model.pkl (K-Means cluster features) as the main model.
  - cashback_pct = CashbackAmount / OrderCount  (per-order cashback value)
  - RiskScore = 0.7 × churn_prob + 0.3 × cashback_pct, min-max normalised
  - Risk categories based on population percentiles:
      Bottom 20 % → Very Low | 20-40 % → Low | 40-60 % → Medium
      60-80 %     → High     | Top 20 % → Critical
  - Saves models/risk_score_meta.pkl for dashboard live-prediction
  - KM plot uses plain "Cluster 0 / 1 / 2 / 3" labels
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

from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test

DATA_DIR  = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
OUT_DIR   = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

RISK_LABELS = ["Very Low", "Low", "Medium", "High", "Critical"]

# ─────────────────────────────────────────────────────────────────────────────
# Step 8 — Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_data_step8():
    """
    Returns
    -------
    preds_df : DataFrame  — churn_probability (XGB-Clust), y_true, cashback_pct
    X_raw    : DataFrame  — human-readable test features
    clusters : DataFrame  — cluster column from cluster_profiles.csv
    feat_names_clust : list — column names expected by xgb_clust_model
    """
    X_test_clust = pd.read_csv(os.path.join(DATA_DIR, "X_test_clust.csv"))
    X_raw        = pd.read_csv(os.path.join(DATA_DIR, "X_test_raw.csv"))
    y_test       = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).squeeze()
    clusters     = pd.read_csv(os.path.join(DATA_DIR, "cluster_profiles.csv"))[["cluster"]]

    # ── Load XGB-Clust; fall back to base model if artefact is missing ────────
    clust_path = os.path.join(MODEL_DIR, "xgb_clust_model.pkl")
    base_path  = os.path.join(MODEL_DIR, "xgb_model.pkl")
    model_path = clust_path if os.path.exists(clust_path) else base_path
    model      = joblib.load(model_path)
    print(f"[08] Loaded model: {os.path.basename(model_path)}")

    churn_prob = model.predict_proba(X_test_clust)[:, 1]

    # cashback_pct = CashbackAmount / OrderCount
    cashback   = X_raw["CashbackAmount"].fillna(0).values
    orders     = np.maximum(X_raw["OrderCount"].fillna(1).values, 1)
    cashback_pct = cashback / orders

    preds_df = pd.DataFrame({
        "y_true":           y_test.values,
        "churn_probability": churn_prob,
        "churn_prediction": (churn_prob >= 0.5).astype(int),
        "cashback_pct":     cashback_pct,
    })

    return preds_df, X_raw, clusters, X_test_clust.columns.tolist()


# ─────────────────────────────────────────────────────────────────────────────
# Step 8 — RiskScore computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_risk_scores(preds_df: pd.DataFrame) -> tuple:
    """
    RiskScore (raw)    = 0.7 × churn_probability + 0.3 × cashback_pct
    RiskScore (normed) = min-max normalised to [0, 1]
    Risk category      = percentile-based quintile labels

    Returns
    -------
    preds_df : DataFrame with 'risk_score' and 'risk_category' columns
    meta     : dict    saved as risk_score_meta.pkl for dashboard normalisation
    """
    raw = 0.7 * preds_df["churn_probability"] + 0.3 * preds_df["cashback_pct"]

    rs_min = float(raw.min())
    rs_max = float(raw.max())
    risk_score = (raw - rs_min) / (rs_max - rs_min + 1e-10)

    # Percentile thresholds on the normalised [0,1] scale
    p20 = float(np.percentile(risk_score, 20))
    p40 = float(np.percentile(risk_score, 40))
    p60 = float(np.percentile(risk_score, 60))
    p80 = float(np.percentile(risk_score, 80))

    def _cat(s):
        if s <= p20: return "Very Low"
        if s <= p40: return "Low"
        if s <= p60: return "Medium"
        if s <= p80: return "High"
        return "Critical"

    df = preds_df.copy()
    df["risk_score"]    = risk_score.round(4)
    df["risk_category"] = risk_score.apply(_cat)

    # Keep a legacy 3-tier column so the existing Risk Dashboard page still works
    _tier_map = {"Very Low": "Low Risk", "Low": "Low Risk",
                 "Medium": "Medium Risk",
                 "High": "High Risk", "Critical": "High Risk"}
    df["risk_tier"] = df["risk_category"].map(_tier_map)

    meta = {
        "rs_min":             rs_min,
        "rs_max":             rs_max,
        "p20":                p20,
        "p40":                p40,
        "p60":                p60,
        "p80":                p80,
        # For dashboard: cashback_pct clipping bounds
        "cashback_pct_p01": float(np.percentile(preds_df["cashback_pct"], 1)),
        "cashback_pct_p99": float(np.percentile(preds_df["cashback_pct"], 99)),
    }
    return df, meta


# ─────────────────────────────────────────────────────────────────────────────
# Step 8 — Action matrix
# ─────────────────────────────────────────────────────────────────────────────

def build_action_matrix(preds_df: pd.DataFrame,
                        X_raw: pd.DataFrame,
                        clusters: pd.DataFrame) -> pd.DataFrame:
    """Merge predictions + raw features + cluster labels, assign actions."""
    cs_path = os.path.join(DATA_DIR, "cluster_summary.csv")
    strategy_map = {}
    if os.path.exists(cs_path):
        cs = pd.read_csv(cs_path)
        for _, row in cs.iterrows():
            cid = int(row["cluster"])
            strategy_map[cid] = {
                "action_high":   row.get("action_high",   "⚠️ Liên hệ khách hàng ngay"),
                "action_medium": row.get("action_medium", "📧 Gửi email chăm sóc"),
                "persona":       row.get("persona",       f"Cluster {cid}"),
                "persona_emoji": row.get("persona_emoji", "⚪"),
            }

    df = pd.concat([
        preds_df.reset_index(drop=True),
        clusters.reset_index(drop=True),
        X_raw.reset_index(drop=True),
    ], axis=1)

    def _action(row):
        cid  = int(row["cluster"]) if not pd.isna(row.get("cluster")) else -1
        meta = strategy_map.get(cid, {})
        rc   = row.get("risk_category", "")
        if rc in ("Critical", "High"):
            return meta.get("action_high",   "🚨 Liên hệ ngay")
        if rc == "Medium":
            return meta.get("action_medium", "📧 Gửi email")
        return "✅ Duy trì chăm sóc tiêu chuẩn"

    df["persona"]            = df["cluster"].apply(
        lambda c: strategy_map.get(int(c), {}).get("persona", f"Cluster {int(c)}")
        if not pd.isna(c) else "Unknown"
    )
    df["recommended_action"] = df.apply(_action, axis=1)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 8 — Main runner
# ─────────────────────────────────────────────────────────────────────────────

def run_step8():
    print(f"\n{'='*60}")
    print("STEP 8: XGB-CLUST RISK SCORING + ACTIONABLE MATRIX")
    print(f"{'='*60}")

    preds_df, X_raw, clusters, feat_names_clust = load_data_step8()
    preds_df, meta = compute_risk_scores(preds_df)

    # Print category distribution
    cat_counts = preds_df["risk_category"].value_counts()
    print("\n[08] ── Risk Category Distribution ────────────────────────")
    for cat in RISK_LABELS:
        n = cat_counts.get(cat, 0)
        pct = n / len(preds_df) * 100
        print(f"  {cat:<10}: {n:>4} customers  ({pct:.1f}%)")

    df = build_action_matrix(preds_df, X_raw, clusters)

    # Intervention summary
    action_stats = (df[df["risk_category"].isin(["Critical", "High"])]
                    .groupby(["risk_category", "cluster", "recommended_action"])
                    .size().reset_index(name="count")
                    .sort_values("count", ascending=False))
    print("\n[08] ── Priority Interventions (Critical + High) ───────────")
    for _, row in action_stats.iterrows():
        print(f"  Cluster {row['cluster']} | {row['risk_category']:<8} "
              f"| {row['count']:>4} users | {row['recommended_action']}")

    # Save outputs
    meta["feature_names_clust"] = feat_names_clust
    joblib.dump(meta, os.path.join(MODEL_DIR, "risk_score_meta.pkl"))
    print(f"\n[08] Saved: risk_score_meta.pkl")

    df.to_csv(os.path.join(DATA_DIR, "targeted_action_list.csv"), index=False)
    print(f"[08] Saved: targeted_action_list.csv")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 9 — Kaplan-Meier Survival Analysis
# ─────────────────────────────────────────────────────────────────────────────

def plot_km_by_cluster(df: pd.DataFrame):
    """Kaplan-Meier curves per cluster — labels are 'Cluster 0 / 1 / 2 / 3'."""
    df_clean = df.dropna(subset=["Tenure", "y_true"]).copy()

    # Colour by cluster risk profile (matches user descriptions)
    CLUSTER_STYLE = {
        0: {"name": "Cluster 0", "color": "#22c55e"},   # stable, low risk
        1: {"name": "Cluster 1", "color": "#3b82f6"},   # stable, low risk
        2: {"name": "Cluster 2", "color": "#f59e0b"},   # needs monitoring
        3: {"name": "Cluster 3", "color": "#ef4444"},   # priority intervention
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    T, E = df_clean["Tenure"], df_clean["y_true"]

    kmf = KaplanMeierFitter()
    for cid in sorted(df_clean["cluster"].dropna().unique()):
        mask  = df_clean["cluster"] == cid
        style = CLUSTER_STYLE.get(int(cid),
                                  {"name": f"Cluster {int(cid)}", "color": "gray"})
        kmf.fit(T[mask], E[mask], label=style["name"])
        kmf.plot_survival_function(ax=ax, color=style["color"],
                                   linewidth=2.5, ci_show=False)

    try:
        result = multivariate_logrank_test(T, df_clean["cluster"].fillna(-1), E)
        p_val  = result.p_value
        p_txt  = f"log-rank p = {p_val:.4e}"
    except Exception:
        p_txt = ""

    ax.set_title(f"Kaplan-Meier Survival Curves by Cluster\n({p_txt})",
                 fontweight="bold")
    ax.set_xlabel("Tenure (months)")
    ax.set_ylabel("Survival Probability")
    ax.axhline(0.70, color="gray", linestyle="--", linewidth=1.2,
               label="Alert threshold (70%)")
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(axis="both", linestyle="--", alpha=0.4)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "08_km_by_cluster.png"), dpi=150)
    plt.close()
    print("[09] Saved: 08_km_by_cluster.png")


def run_step9(df_with_clusters: pd.DataFrame):
    print(f"\n{'='*60}")
    print("STEP 9: PERSONA-BASED SURVIVAL ANALYSIS")
    print(f"{'='*60}")
    plot_km_by_cluster(df_with_clusters)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point (called by run_pipeline.py)
# ─────────────────────────────────────────────────────────────────────────────

def run_survival_analysis():
    df_action = run_step8()
    run_step9(df_action)
    return df_action


if __name__ == "__main__":
    run_survival_analysis()