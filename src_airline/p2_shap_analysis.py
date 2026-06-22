"""
p2_shap_analysis.py  —  SHAP Analysis + Personalized Action Matrix
===================================================================
Steps A, B, C: Global importance, per-cluster, Critical-risk group.
Pipeline PAUSES after Step C for user to fill Action Dictionary.

Steps D, E: Run via run_airline_shap.py after user fills action_dict.csv.
"""
import pandas as pd
import numpy as np
import os, joblib, warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap


# ── Known Point-Biserial rankings from Step 3 ─────────────────────────────────
STEP3_PB = {
    "Inflight entertainment":              {"r": 0.5892, "rank": 1},
    "inflight_experience_score":           {"r": 0.5699, "rank": 2},
    "ground_experience_score":             {"r": 0.4413, "rank": 3},
    "Ease of Online booking":              {"r": 0.4322, "rank": 4},
    "Online support":                      {"r": 0.3903, "rank": 5},
    "On-board service":                    {"r": 0.3499, "rank": 6},
    "Seat comfort":                        {"r": 0.3479, "rank": 7},
    "Online boarding":                     {"r": 0.3385, "rank": 8},
    "Leg room service":                    {"r": 0.3110, "rank": 9},
    "Checkin service":                     {"r": 0.2653, "rank": 10},
    "Baggage handling":                    {"r": 0.2606, "rank": 11},
    "Cleanliness":                         {"r": 0.2587, "rank": 12},
    "Inflight wifi service":               {"r": 0.2278, "rank": 13},
    "Food and drink":                      {"r": 0.1869, "rank": 14},
    "Age":                                 {"r": 0.1192, "rank": 15},
    "Flight Distance":                     {"r": None,   "rank": None},
    "Departure Delay in Minutes":          {"r": None,   "rank": None},
    "Class_ord":                           {"r": None,   "rank": None},
    "loyal":                               {"r": None,   "rank": None},
    "biz_travel":                          {"r": None,   "rank": None},
}


def _get_base_xgb(cal_model):
    """Extract base XGBoost estimator from CalibratedClassifierCV."""
    cc = cal_model.calibrated_classifiers_[0]
    for attr in ["estimator", "base_estimator"]:
        if hasattr(cc, attr):
            return getattr(cc, attr)
    raise AttributeError("Cannot extract base estimator from CalibratedClassifierCV")


def run_step_a(cfg, X_test_base: pd.DataFrame, cal_model) -> tuple:
    """Step A: Global SHAP importance."""
    sep = "=" * 70
    print(f"\n{sep}\nSTEP A: SHAP GLOBAL IMPORTANCE\n{sep}")

    base_xgb = _get_base_xgb(cal_model)
    print(f"  Base model: {type(base_xgb).__name__}")

    # Compute SHAP values
    print(f"  Computing SHAP values for {len(X_test_base):,} test rows...")
    explainer   = shap.TreeExplainer(base_xgb)
    shap_values = explainer.shap_values(X_test_base)   # shape: (n, p)
    feature_names = list(X_test_base.columns)
    print(f"  SHAP values shape: {shap_values.shape}")

    # Mean |SHAP| per feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_rank_df = pd.DataFrame({
        "feature":          feature_names,
        "mean_abs_shap":    mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    shap_rank_df["shap_rank"] = shap_rank_df.index + 1

    # ── Bar chart (top 15) ────────────────────────────────────────────────────
    top15 = shap_rank_df.head(15)
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["#3b82f6" if not c.startswith("was_na") else "#94a3b8"
              for c in top15["feature"][::-1]]
    ax.barh(top15["feature"][::-1], top15["mean_abs_shap"][::-1],
            color=colors, edgecolor="white")
    ax.set_title("SHAP Global Importance — XGB-Base (Top 15, Test Set)",
                 fontweight="bold")
    ax.set_xlabel("Mean |SHAP value|")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    out_bar = os.path.join(cfg.OUT_DIR, "p2_shap_bar.png")
    plt.savefig(out_bar, dpi=130, bbox_inches="tight"); plt.close()
    print(f"  Bar chart → {out_bar}")

    # ── Beeswarm plot (top 15) ────────────────────────────────────────────────
    top15_idx = [shap_rank_df.iloc[i]["feature"] for i in range(min(15, len(shap_rank_df)))]
    top15_col_idx = [feature_names.index(f) for f in top15_idx if f in feature_names]
    shap_top = shap.Explanation(
        values      = shap_values[:, top15_col_idx],
        base_values = np.full(len(shap_values), explainer.expected_value),
        data        = X_test_base.iloc[:, top15_col_idx].values,
        feature_names = [feature_names[i] for i in top15_col_idx],
    )
    # beeswarm() renders to current matplotlib figure — save with plt.savefig
    shap.plots.beeswarm(shap_top, max_display=15, show=False)
    plt.title("SHAP Beeswarm — XGB-Base (Top 15)", fontweight="bold", pad=15)
    out_bee = os.path.join(cfg.OUT_DIR, "p2_shap_beeswarm.png")
    plt.savefig(out_bee, dpi=130, bbox_inches="tight"); plt.close("all")
    print(f"  Beeswarm chart → {out_bee}")

    # ── Print top 15 ──────────────────────────────────────────────────────────
    print(f"\n[A.1] Top 15 features by mean(|SHAP value|):")
    print(f"  {'SHAP Rank':>10}  {'Feature':<45}  {'Mean|SHAP|':>10}")
    print("  " + "─" * 70)
    for _, row in shap_rank_df.head(15).iterrows():
        print(f"  {int(row['shap_rank']):>10}  {row['feature']:<45}  {row['mean_abs_shap']:>10.4f}")

    # ── Compare with Step 3 Point-Biserial rankings ───────────────────────────
    print(f"\n[A.2] SHAP vs Point-Biserial ranking comparison:")
    print(f"  {'Feature':<45}  {'SHAP Rank':>10}  {'PtBis Rank':>11}  {'Δ Rank':>8}  Note")
    print("  " + "─" * 85)
    for _, row in shap_rank_df.head(20).iterrows():
        feat = row["feature"]
        sr   = int(row["shap_rank"])
        pb   = STEP3_PB.get(feat, {})
        pr   = pb.get("rank")
        if pr is not None:
            delta = sr - pr
            note  = ""
            if abs(delta) >= 3:
                note = f"⚠ shifted {abs(delta)} ranks {'down' if delta > 0 else 'up'}"
            print(f"  {feat:<45}  {sr:>10}  {pr:>11}  {delta:>+8}  {note}")
        else:
            print(f"  {feat:<45}  {sr:>10}  {'—':>11}  {'—':>8}  (no PtBis — categorical/composite)")

    # Save SHAP rank table
    shap_rank_df.to_csv(os.path.join(cfg.DATA_DIR, "shap_global_importance.csv"), index=False)

    return explainer, shap_values, shap_rank_df


def run_step_b(cfg, X_test_base: pd.DataFrame, shap_values: np.ndarray,
               cluster_series: pd.Series) -> None:
    """Step B: SHAP by cluster (C0 vs C1)."""
    sep = "=" * 70
    print(f"\n{sep}\nSTEP B: SHAP BY CLUSTER (C0 vs C1)\n{sep}")

    feature_names = list(X_test_base.columns)
    clusters = cluster_series.values

    cluster_shap = {}
    for c in sorted(cluster_series.unique()):
        mask       = clusters == c
        sv_c       = shap_values[mask]
        mean_abs_c = np.abs(sv_c).mean(axis=0)
        top_df = pd.DataFrame({
            "feature":       feature_names,
            "mean_abs_shap": mean_abs_c,
        }).sort_values("mean_abs_shap", ascending=False).head(10).reset_index(drop=True)
        top_df["rank"] = top_df.index + 1
        cluster_shap[c] = top_df
        print(f"\n  ── Cluster C{c} (n={mask.sum():,}) — Top 10 features ──")
        for _, row in top_df.iterrows():
            bar = "█" * int(row["mean_abs_shap"] * 80)
            print(f"    {int(row['rank']):>2}. {row['feature']:<45}  {row['mean_abs_shap']:.4f}  {bar}")

    # Side-by-side comparison
    print(f"\n[B.2] Side-by-side: Top 10 of C0 vs C1")
    print(f"  {'Rank':>5}  {'C0 — Hài Lòng Toàn Diện':<45}  {'Mean|SHAP|':>10}  "
          f"  {'C1 — Trải Nghiệm Kém':<45}  {'Mean|SHAP|':>10}")
    print("  " + "─" * 115)
    for i in range(10):
        r0 = cluster_shap[0].iloc[i] if 0 in cluster_shap else None
        r1 = cluster_shap[1].iloc[i] if 1 in cluster_shap else None
        c0_str = f"{r0['feature']:<45}  {r0['mean_abs_shap']:>10.4f}" if r0 is not None else f"{'—':<55}"
        c1_str = f"{r1['feature']:<45}  {r1['mean_abs_shap']:>10.4f}" if r1 is not None else f"{'—':<55}"
        print(f"  {i+1:>5}  {c0_str}    {c1_str}")

    # Chart: side-by-side bar
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, (c, label, color) in zip(axes, [
        (0, "C0 — Hài Lòng Toàn Diện", "#22c55e"),
        (1, "C1 — Trải Nghiệm Kém",    "#ef4444"),
    ]):
        if c not in cluster_shap: continue
        top = cluster_shap[c]
        ax.barh(top["feature"][::-1], top["mean_abs_shap"][::-1],
                color=color, edgecolor="white", alpha=0.85)
        ax.set_title(f"Cluster {label}\nTop 10 SHAP features", fontweight="bold")
        ax.set_xlabel("Mean |SHAP value|")
        ax.grid(axis="x", alpha=0.3)
    plt.suptitle("Step B: SHAP by Cluster — C0 vs C1", fontweight="bold")
    plt.tight_layout()
    out = os.path.join(cfg.OUT_DIR, "p2_shap_by_cluster.png")
    plt.savefig(out, dpi=130, bbox_inches="tight"); plt.close()
    print(f"\n  Chart → {out}")

    print(f"\n{'─'*70}\nSTEP B DONE\n{'─'*70}")


def run_step_c(cfg, X_test_base: pd.DataFrame, shap_values: np.ndarray,
               action_df: pd.DataFrame) -> pd.DataFrame:
    """Step C: SHAP for Critical risk group + top-3 negative drivers per customer."""
    sep = "=" * 70
    print(f"\n{sep}\nSTEP C: SHAP — CRITICAL RISK GROUP\n{sep}")

    feature_names = list(X_test_base.columns)
    critical_mask = (action_df["risk_category"] == "Critical").values
    n_critical    = critical_mask.sum()
    print(f"  Critical risk rows: {n_critical:,}")

    sv_crit = shap_values[critical_mask]     # shape: (n_critical, p)

    # Top 10 by mean |SHAP| within Critical
    mean_abs_crit = np.abs(sv_crit).mean(axis=0)
    top_crit = pd.DataFrame({
        "feature":       feature_names,
        "mean_abs_shap": mean_abs_crit,
    }).sort_values("mean_abs_shap", ascending=False).head(10).reset_index(drop=True)

    print(f"\n[C.1] Top 10 SHAP features — CRITICAL group only:")
    print(f"  {'Rank':>5}  {'Feature':<45}  {'Mean|SHAP|':>10}  Bar")
    print("  " + "─" * 75)
    for i, row in top_crit.iterrows():
        bar = "█" * int(row["mean_abs_shap"] * 80)
        print(f"  {i+1:>5}  {row['feature']:<45}  {row['mean_abs_shap']:>10.4f}  {bar}")

    # Top-3 NEGATIVE drivers per critical customer (most negative SHAP = pushing toward dissatisfied)
    # For a model predicting P(satisfied=1): negative SHAP = push toward dissatisfied
    print(f"\n[C.2] Computing top-3 negative SHAP drivers per Critical customer...")
    driver_records = []
    for i, (shap_row) in enumerate(sv_crit):
        # Sort by SHAP value ascending (most negative first)
        sorted_idx = np.argsort(shap_row)
        top3 = []
        for idx in sorted_idx[:3]:
            top3.append({
                "feature":    feature_names[idx],
                "shap_value": float(shap_row[idx]),
            })
        driver_records.append(top3)

    # Add to action_df (Critical rows only)
    crit_indices = action_df[action_df["risk_category"] == "Critical"].index
    for j, orig_idx in enumerate(crit_indices):
        for k in range(3):
            col_f = f"top_driver_{k+1}"
            col_s = f"top_driver_{k+1}_shap"
            action_df.at[orig_idx, col_f] = driver_records[j][k]["feature"]
            action_df.at[orig_idx, col_s] = round(driver_records[j][k]["shap_value"], 4)

    # Fill NaN for non-Critical rows
    for k in range(3):
        action_df[f"top_driver_{k+1}"]      = action_df.get(f"top_driver_{k+1}", None)
        action_df[f"top_driver_{k+1}_shap"] = action_df.get(f"top_driver_{k+1}_shap", None)

    # ── Frequency of top-3 drivers in Critical group ───────────────────────────
    print(f"\n[C.3] Frequency of features in top-3 negative drivers (Critical group):")
    freq = {}
    for rec in driver_records:
        for d in rec:
            freq[d["feature"]] = freq.get(d["feature"], 0) + 1
    freq_df = pd.DataFrame([{"feature": k, "count": v, "pct": v/n_critical*100}
                              for k, v in freq.items()]).sort_values("count", ascending=False)
    print(f"  {'Feature':<45}  {'Count':>7}  {'% of Critical':>14}  Bar")
    print("  " + "─" * 80)
    for _, row in freq_df.head(15).iterrows():
        bar = "█" * int(row["pct"] / 3)
        print(f"  {row['feature']:<45}  {int(row['count']):>7,}  {row['pct']:>13.1f}%  {bar}")

    # Save frequency
    freq_df.to_csv(os.path.join(cfg.DATA_DIR, "critical_driver_frequency.csv"), index=False)

    # Chart Critical top 10
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].barh(top_crit["feature"][::-1], top_crit["mean_abs_shap"][::-1],
                 color="#ef4444", edgecolor="white", alpha=0.85)
    axes[0].set_title("SHAP — Critical Group (Top 10)", fontweight="bold")
    axes[0].set_xlabel("Mean |SHAP value|"); axes[0].grid(axis="x", alpha=0.3)

    axes[1].barh(freq_df["feature"][:10][::-1], freq_df["pct"][:10][::-1],
                 color="#f97316", edgecolor="white", alpha=0.85)
    axes[1].set_title("Top Negative Drivers — Frequency in Critical (%)", fontweight="bold")
    axes[1].set_xlabel("% of Critical customers"); axes[1].grid(axis="x", alpha=0.3)

    plt.suptitle("Step C: SHAP Analysis — Critical Risk Group", fontweight="bold")
    plt.tight_layout()
    out = os.path.join(cfg.OUT_DIR, "p2_shap_critical.png")
    plt.savefig(out, dpi=130, bbox_inches="tight"); plt.close()
    print(f"\n  Chart → {out}")
    print(f"  critical_driver_frequency.csv saved")

    print(f"\n{'─'*70}\nSTEP C DONE\n{'─'*70}")
    return action_df, freq_df


def run_step_d(cfg, shap_rank_df: pd.DataFrame, freq_df: pd.DataFrame) -> pd.DataFrame:
    """Step D: Build Action Dictionary template (user fills content)."""
    sep = "=" * 70
    print(f"\n{sep}\nSTEP D: ACTION DICTIONARY TEMPLATE\n{sep}")

    # Collect all candidate features
    candidates = list(shap_rank_df["feature"].head(20))
    extra = [
        "Departure Delay in Minutes", "delay_severity",
        "Class_ord", "loyal", "biz_travel", "Age", "Flight Distance",
    ]
    for f in extra:
        if f not in candidates:
            candidates.append(f)

    template_rows = []
    for feat in candidates:
        template_rows.append({
            "feature_name":      feat,
            "hanh_dong_cu_the":  "",          # USER FILLS THIS
            "muc_do_chi_phi":    "",          # thấp / trung / cao — USER FILLS
            "ghi_chu":           "",
        })

    template_df = pd.DataFrame(template_rows)
    out_path = os.path.join(cfg.DATA_DIR, "action_dictionary_template.csv")
    template_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"\n  Template created with {len(template_df)} features:")
    print(f"  {'Feature':<45}  (hanh_dong_cu_the — USER FILL)")
    print("  " + "─" * 60)
    for _, row in template_df.iterrows():
        print(f"  {row['feature_name']:<45}  [EMPTY — waiting for user input]")

    print(f"\n  Saved → {out_path}")
    print(f"\n  ⏸  NEXT: Open action_dictionary_template.csv, fill in:")
    print(f"    - hanh_dong_cu_the : the specific action for this feature driver")
    print(f"    - muc_do_chi_phi   : thấp / trung / cao")
    print(f"    Then run: python run_airline_shap.py --step E")

    print(f"\n{'─'*70}\nSTEP D DONE\n{'─'*70}")
    return template_df


def run_step_e(cfg, action_df: pd.DataFrame, action_dict_path: str) -> pd.DataFrame:
    """Step E: Personalized targeted_action_list_v2.csv."""
    sep = "=" * 70
    print(f"\n{sep}\nSTEP E: PERSONALIZED ACTION LIST v2\n{sep}")

    # Load action dictionary
    if not os.path.exists(action_dict_path):
        print(f"  ⚠ Action dictionary not found: {action_dict_path}")
        print(f"  Run Step D first, fill in the template, then re-run Step E.")
        return action_df

    act_dict = pd.read_csv(action_dict_path)
    filled   = act_dict[act_dict["hanh_dong_cu_the"].notna() &
                        (act_dict["hanh_dong_cu_the"] != "")]
    print(f"  Action dictionary: {len(act_dict)} features, {len(filled)} filled")

    feat_to_action = dict(zip(act_dict["feature_name"], act_dict["hanh_dong_cu_the"]))

    def recommend(row):
        rc = row.get("risk_category", "")
        if rc not in ("High", "Critical"):
            return ""   # low-risk: use general action from Step 8
        d1 = row.get("top_driver_1", "")
        d2 = row.get("top_driver_2", "")
        d3 = row.get("top_driver_3", "")
        actions = []
        for d in [d1, d2, d3]:
            if pd.notna(d) and d in feat_to_action and feat_to_action[d]:
                actions.append(feat_to_action[d])
        return " | ".join(actions) if actions else ""

    action_df["recommended_action"] = action_df.apply(recommend, axis=1)

    # Select output columns
    out_cols = [c for c in [
        "cluster", "risk_category", "risk_score", "p_dissatisfied",
        "top_driver_1", "top_driver_1_shap",
        "top_driver_2", "top_driver_2_shap",
        "top_driver_3", "top_driver_3_shap",
        "recommended_action",
        "persona", "action",   # from v1
    ] if c in action_df.columns]

    out_df = action_df[out_cols].copy()
    out_path = os.path.join(cfg.DATA_DIR, "targeted_action_list_v2.csv")
    out_df.to_csv(out_path, index=True, index_label="customer_index",
                  encoding="utf-8-sig")

    print(f"\n  targeted_action_list_v2.csv saved → {out_path}")
    print(f"  Rows: {len(out_df):,}")
    print(f"  Rows with personalized action: {(out_df['recommended_action'] != '').sum():,}")

    # Summary stats
    print(f"\n  Rows with personalized action by risk category:")
    for rc in ["Critical", "High", "Medium", "Low", "Very Low"]:
        n = (action_df["risk_category"] == rc).sum()
        n_act = ((action_df["risk_category"] == rc) &
                 (action_df["recommended_action"] != "")).sum()
        print(f"  {rc:<10}: {n:>6,}  personalized={n_act:,}")

    print(f"\n{'─'*70}\nSTEP E DONE\n{'─'*70}")
    return out_df


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import src_airline.airline_config as cfg
    print("Run via run_airline_shap.py")
