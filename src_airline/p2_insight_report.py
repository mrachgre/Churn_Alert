"""
p2_insight_report.py  —  Steps D-new & E-new
=============================================
Purely descriptive outputs — NO action recommendations, NO cost levels.

Step D-new: customer_risk_insight_report.csv (per-customer)
Step E-new: system_level_insight_report.md  (aggregate patterns)
"""
import pandas as pd
import numpy as np
import os, joblib, warnings, textwrap
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap


# ── Helpers ───────────────────────────────────────────────────────────────────
def _get_base_xgb(cal_model):
    cc = cal_model.calibrated_classifiers_[0]
    for attr in ["estimator", "base_estimator"]:
        if hasattr(cc, attr):
            return getattr(cc, attr)
    raise AttributeError("Cannot extract base XGB from CalibratedClassifierCV")


def _compute_shap(cal_model, X_test_base: pd.DataFrame) -> np.ndarray:
    """Compute SHAP values for all test rows (reuse TreeExplainer)."""
    print(f"  Computing SHAP values for {len(X_test_base):,} rows...")
    explainer   = shap.TreeExplainer(_get_base_xgb(cal_model))
    shap_values = explainer.shap_values(X_test_base)
    print(f"  SHAP values shape: {shap_values.shape}")
    return shap_values


def _top3_neg_drivers(shap_row: np.ndarray, feature_names: list, X_row: pd.Series
                      ) -> dict:
    """Return top-3 most negative SHAP drivers + actual rating for a single row."""
    sorted_idx = np.argsort(shap_row)          # ascending → most negative first
    result = {}
    for k, idx in enumerate(sorted_idx[:3], 1):
        feat      = feature_names[idx]
        shap_val  = float(shap_row[idx])
        raw_val   = X_row[feat] if feat in X_row.index else None
        # Round raw value: keep 1dp if float, int if integer-like
        if raw_val is not None and not pd.isna(raw_val):
            raw_val = round(float(raw_val), 2)
        result[f"top_driver_{k}"]            = feat
        result[f"top_driver_{k}_shap_value"] = round(shap_val, 4)
        result[f"top_driver_{k}_actual_rating"] = raw_val
    return result


# ══════════════════════════════════════════════════════════════════════════════
# STEP D-NEW: Per-Customer Risk Insight CSV
# ══════════════════════════════════════════════════════════════════════════════
def run_step_d_new(cfg, X_test_base: pd.DataFrame, X_test_clust: pd.DataFrame,
                   action_df: pd.DataFrame, cal_model) -> pd.DataFrame:
    sep = "=" * 70
    print(f"\n{sep}\nSTEP D-NEW: CUSTOMER RISK INSIGHT REPORT\n{sep}")

    shap_values   = _compute_shap(cal_model, X_test_base)
    feature_names = list(X_test_base.columns)

    # Build per-row records
    print(f"  Extracting top-3 negative drivers for each of {len(X_test_base):,} customers...")
    records = []
    cluster_map = {0: "Hài Lòng Toàn Diện", 1: "Trải Nghiệm Kém"}

    for i in range(len(X_test_base)):
        shap_row = shap_values[i]
        X_row    = X_test_base.iloc[i]
        a_row    = action_df.iloc[i]

        drivers = _top3_neg_drivers(shap_row, feature_names, X_row)

        cid     = int(X_test_clust["kmeans_cluster_id"].iloc[i])
        record  = {
            "customer_index":  i,
            "risk_category":   a_row.get("risk_category", ""),
            "risk_score":      round(float(a_row.get("risk_score", 0)), 4),
            "cluster_id":      f"C{cid}",
            "persona_label":   cluster_map.get(cid, ""),
            **drivers,
        }
        records.append(record)

    report_df = pd.DataFrame(records)

    # Print sample
    print(f"\n  Sample (first 5 Critical rows):")
    sample = report_df[report_df["risk_category"] == "Critical"].head(5)
    cols_show = ["customer_index", "risk_score", "cluster_id",
                 "top_driver_1", "top_driver_1_actual_rating",
                 "top_driver_2", "top_driver_2_actual_rating"]
    print(sample[cols_show].to_string(index=False))

    # Save
    out_path = os.path.join(cfg.DATA_DIR, "customer_risk_insight_report.csv")
    report_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n  Saved → {out_path}  ({len(report_df):,} rows × {len(report_df.columns)} cols)")

    print(f"\n{'─'*70}\nSTEP D-NEW DONE\n{'─'*70}")
    return report_df, shap_values


# ══════════════════════════════════════════════════════════════════════════════
# STEP E-NEW: System-Level Insight Report (Markdown)
# ══════════════════════════════════════════════════════════════════════════════
def run_step_e_new(cfg, report_df: pd.DataFrame, X_test_base: pd.DataFrame,
                   action_df: pd.DataFrame, shap_values: np.ndarray) -> str:
    sep = "=" * 70
    print(f"\n{sep}\nSTEP E-NEW: SYSTEM-LEVEL INSIGHT REPORT\n{sep}")

    feature_names = list(X_test_base.columns)
    lines = []

    def h(text, level=2):
        lines.append(f"\n{'#' * level} {text}\n")

    def p(text):
        lines.append(text)

    # ── Header ────────────────────────────────────────────────────────────────
    lines.append("# Airline Passenger Satisfaction — System-Level Risk Insight Report")
    lines.append(f"\n_Generated from SHAP analysis on test set (n={len(report_df):,} passengers)_\n")
    lines.append("---")
    lines.append("> **Scope:** Descriptive analysis only. No action recommendations included.")

    # ── 1. Risk tier distribution ─────────────────────────────────────────────
    h("1. Risk Tier Distribution")
    order = ["Very Low", "Low", "Medium", "High", "Critical"]
    vc    = report_df["risk_category"].value_counts().reindex(order, fill_value=0)
    lines.append("| Risk Category | Count | % of Total |")
    lines.append("|:-------------|------:|-----------:|")
    for cat in order:
        n   = vc[cat]
        pct = n / len(report_df) * 100
        lines.append(f"| {cat} | {n:,} | {pct:.1f}% |")
    total_high = vc["High"] + vc["Critical"]
    lines.append(f"\n{total_high:,} passengers ({total_high/len(report_df)*100:.1f}%) are in the **High or Critical** tier.")

    # ── 2. Critical group driver analysis ─────────────────────────────────────
    h("2. Critical Group — Driver Frequency & Actual Ratings")
    crit_mask = report_df["risk_category"] == "Critical"
    n_crit    = crit_mask.sum()
    p(f"Critical group: **{n_crit:,} passengers** ({n_crit/len(report_df)*100:.1f}% of test set)\n")

    # Frequency of top-3 drivers
    freq = {}
    crit_rows = report_df[crit_mask]
    for col in ["top_driver_1", "top_driver_2", "top_driver_3"]:
        for v in crit_rows[col].dropna():
            freq[v] = freq.get(v, 0) + 1

    freq_df = (pd.DataFrame([{"feature": k, "count": v}
                              for k, v in freq.items()])
               .sort_values("count", ascending=False))

    # Mean actual rating per driver within Critical (vs full dataset)
    crit_indices = report_df[crit_mask]["customer_index"].values
    X_crit       = X_test_base.iloc[crit_indices]

    rating_comparison = []
    for feat in freq_df["feature"].head(10).values:
        if feat not in X_test_base.columns:
            continue
        mean_crit    = X_crit[feat].mean()
        mean_full    = X_test_base[feat].mean()
        pct_of_crit  = freq[feat] / n_crit * 100
        rating_comparison.append({
            "feature":       feat,
            "pct_critical":  round(pct_of_crit, 1),
            "mean_critical": round(mean_crit, 3),
            "mean_full":     round(mean_full, 3),
            "delta":         round(mean_crit - mean_full, 3),
        })

    rc_df = pd.DataFrame(rating_comparison)
    lines.append("| Driver Feature | % Critical Customers | Mean Rating (Critical) | Mean Rating (All) | Δ |")
    lines.append("|:--------------|---------------------:|----------------------:|------------------:|--:|")
    for _, row in rc_df.iterrows():
        lines.append(f"| {row['feature']} | {row['pct_critical']:.1f}% | "
                     f"{row['mean_critical']:.2f} | {row['mean_full']:.2f} | "
                     f"{row['delta']:+.2f} |")

    top3_drivers = rc_df.head(3)
    p("\n**Top 3 drivers in the Critical group:**")
    for _, row in top3_drivers.iterrows():
        p(f"- **{row['feature']}**: present in {row['pct_critical']:.1f}% of Critical customers. "
          f"Mean rating in Critical group: {row['mean_critical']:.2f} "
          f"(full-dataset mean: {row['mean_full']:.2f}, Δ = {row['delta']:+.2f}).")

    # ── 3. Persona comparison C0 vs C1 ────────────────────────────────────────
    h("3. Cluster Persona Comparison — C0 vs C1")

    c0_mask = report_df["cluster_id"] == "C0"
    c1_mask = report_df["cluster_id"] == "C1"
    n_c0, n_c1 = c0_mask.sum(), c1_mask.sum()

    p(f"- **C0 (Hài Lòng Toàn Diện)**: {n_c0:,} passengers ({n_c0/len(report_df)*100:.1f}%)")
    p(f"- **C1 (Trải Nghiệm Kém)**: {n_c1:,} passengers ({n_c1/len(report_df)*100:.1f}%)\n")

    # Dissatisfied rate per cluster
    dissatisfied_c0 = (report_df.loc[c0_mask, "risk_category"].isin(["High","Critical"])).mean()
    dissatisfied_c1 = (report_df.loc[c1_mask, "risk_category"].isin(["High","Critical"])).mean()
    rs_c0 = report_df.loc[c0_mask, "risk_score"].mean()
    rs_c1 = report_df.loc[c1_mask, "risk_score"].mean()

    lines.append("| Metric | C0 — Hài Lòng Toàn Diện | C1 — Trải Nghiệm Kém |")
    lines.append("|:-------|:-----------------------:|:--------------------:|")
    lines.append(f"| Mean risk_score | {rs_c0:.3f} | {rs_c1:.3f} |")
    lines.append(f"| High/Critical rate | {dissatisfied_c0*100:.1f}% | {dissatisfied_c1*100:.1f}% |")

    # Service rating comparison C0 vs C1
    service_cols = [c for c in ["Seat comfort", "Inflight entertainment",
                                 "Ease of Online booking", "On-board service",
                                 "Leg room service", "Cleanliness", "Food and drink",
                                 "inflight_experience_score", "ground_experience_score"]
                    if c in X_test_base.columns]
    p("\n**Mean service ratings by cluster:**\n")
    lines.append("| Feature | C0 mean | C1 mean | C0−C1 |")
    lines.append("|:--------|--------:|--------:|------:|")
    c0_idx = report_df[c0_mask]["customer_index"].values
    c1_idx = report_df[c1_mask]["customer_index"].values
    for col in service_cols:
        m0 = X_test_base.iloc[c0_idx][col].mean()
        m1 = X_test_base.iloc[c1_idx][col].mean()
        lines.append(f"| {col} | {m0:.2f} | {m1:.2f} | {m0-m1:+.2f} |")

    # ── 3b. C1 breakdown: loyal vs disloyal ───────────────────────────────────
    h("3b. C1 — Loyal vs Disloyal Sub-group Analysis", level=3)
    p("This section examines whether the SHAP prominence of `loyal` in C1 reflects "
      "genuine risk differences or a rating bias pattern.\n")

    c1_df = X_test_base.iloc[c1_idx].copy().reset_index(drop=True)
    c1_report = report_df[c1_mask].reset_index(drop=True)

    if "loyal" in c1_df.columns:
        for lval, lname in [(1, "Loyal Customer"), (0, "Disloyal Customer")]:
            lmask     = c1_df["loyal"] == lval
            n_l       = lmask.sum()
            pct_l     = n_l / len(c1_df) * 100
            rs_l      = c1_report.loc[lmask, "risk_score"].mean()
            hcrit_l   = c1_report.loc[lmask, "risk_category"].isin(["High","Critical"]).mean()
            p(f"**{lname}** (n={n_l:,}, {pct_l:.1f}% of C1):")
            p(f"  - Mean risk_score: {rs_l:.3f}")
            p(f"  - High/Critical rate: {hcrit_l*100:.1f}%\n")

            # Mean service ratings
            svc_avail = [c for c in service_cols if c in c1_df.columns]
            rating_means = c1_df.loc[lmask, svc_avail].mean().round(2)
            p(f"  Service rating means (within C1, {lname}):")
            for feat, val in rating_means.items():
                p(f"    - {feat}: {val:.2f}")
            p("")

        # Comparison table
        loy   = c1_df["loyal"] == 1
        dis_l = c1_df["loyal"] == 0
        lines.append("\n| Feature | Loyal (C1) | Disloyal (C1) | Difference |")
        lines.append("|:--------|----------:|-------------:|-----------:|")
        for col in service_cols:
            if col not in c1_df.columns:
                continue
            ml = c1_df.loc[loy, col].mean()
            md = c1_df.loc[dis_l, col].mean()
            lines.append(f"| {col} | {ml:.2f} | {md:.2f} | {ml-md:+.2f} |")

        # Interpretation note
        rs_loy  = c1_report.loc[c1_df["loyal"].values == 1, "risk_score"].mean()
        rs_disy = c1_report.loc[c1_df["loyal"].values == 0, "risk_score"].mean()
        rt_loy  = c1_df.loc[loy, service_cols[0]].mean() if service_cols else None
        rt_dis  = c1_df.loc[dis_l, service_cols[0]].mean() if service_cols else None
        p(f"\n**Observation:** Mean risk_score — Loyal={rs_loy:.3f}, Disloyal={rs_disy:.3f} "
          f"(difference: {rs_loy-rs_disy:+.3f}).")
        if rt_loy and rt_dis:
            gap = abs(rs_loy - rs_disy)
            rt_gap = abs(rt_loy - rt_dis)
            if gap > 0.05 and rt_gap < 0.3:
                p("Risk scores differ noticeably between Loyal and Disloyal sub-groups within C1, "
                  "yet service ratings are similar — this pattern is consistent with a loyalty-related "
                  "rating behavior difference rather than a large difference in actual service experience.")
            elif gap < 0.03:
                p("Risk scores are similar between Loyal and Disloyal sub-groups within C1, "
                  "suggesting the `loyal` SHAP prominence reflects a broad cluster-level pattern "
                  "rather than a sub-group-specific risk signal.")

    # ── 4. SHAP Driver top-3 driver comparison C0 vs C1 ──────────────────────
    h("4. Top SHAP Drivers — C0 vs C1 Comparison (from Step B)")
    p("_Reference: SHAP mean|value| computed on test set per cluster._\n")
    lines.append("| Rank | C0 Driver | C1 Driver |")
    lines.append("|:----:|:----------|:----------|")
    C0_DRIVERS = ["Seat comfort","Inflight entertainment","biz_travel",
                  "loyal","Departure/Arrival time convenient",
                  "inflight_experience_score","Ease of Online booking",
                  "Leg room service","Baggage handling","Gate location"]
    C1_DRIVERS = ["Seat comfort","Inflight entertainment","loyal",
                  "inflight_experience_score","Departure/Arrival time convenient",
                  "Gate location","Ease of Online booking","biz_travel",
                  "Baggage handling","Cleanliness"]
    for i, (d0, d1) in enumerate(zip(C0_DRIVERS, C1_DRIVERS), 1):
        flag = " ← **differs**" if d0 != d1 else ""
        lines.append(f"| {i} | {d0} | {d1}{flag} |")

    p("\nKey differences: `biz_travel` ranks 3rd in C0 but 8th in C1; "
      "`loyal` ranks 4th in C0 but 3rd in C1; "
      "`Gate location` ranks 10th in C0 but 6th in C1.")

    # ── Footer ────────────────────────────────────────────────────────────────
    lines.append("\n---")
    lines.append(f"_Report based on XGB-Base model, test set n={len(report_df):,}. "
                 f"SHAP values reflect model-derived feature contributions (log-odds scale)._")

    report_text = "\n".join(lines)

    # Save markdown
    md_path = os.path.join(cfg.DATA_DIR, "system_level_insight_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\n  Saved → {md_path}")

    # Print summary to console
    print(f"\n{'─'*70}")
    print("STEP E-NEW SUMMARY (key numbers):")
    print(f"{'─'*70}")
    for cat in order:
        n   = vc[cat]
        pct = n / len(report_df) * 100
        bar = "█" * int(pct / 2)
        print(f"  {cat:<12}: {n:>6,}  ({pct:.1f}%)  {bar}")
    print(f"\n  Critical top-3 drivers by frequency:")
    for _, row in freq_df.head(3).iterrows():
        print(f"    {row['feature']:<45}  {row['count']/n_crit*100:.1f}% of Critical")
    print(f"\n  C0 mean risk_score: {rs_c0:.3f}  |  C1 mean risk_score: {rs_c1:.3f}")
    print(f"\n{'─'*70}\nSTEP E-NEW DONE\n{'─'*70}")

    return report_text


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import src_airline.airline_config as cfg
    print("Run via run_airline_insight.py")
