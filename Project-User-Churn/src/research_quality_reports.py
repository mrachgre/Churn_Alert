"""
Additive research-quality reports for churn analysis.

The functions here consume existing data/model artifacts and write new outputs.
They do not modify the train/test split, feature engineering, tuning, model
training, SHAP generation, existing outputs, or dataset-related modules.
"""

import os
import sys
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)


def _csv(folder, name):
    path = os.path.join(folder, name)
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _write_md(name, lines):
    with open(os.path.join(OUT_DIR, name), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def run_research_quality_reports():
    """Run all additive reporting layers and fail gracefully by section."""
    tasks = [
        _run_feature_redundancy,
        _run_stability,
        _run_error_analysis,
        run_survival_audit,
        run_cluster_persona_reports,
        run_business_roi_analysis,
        run_fairness_extended_report,
        run_shap_business_interpretation,
        run_research_findings,
        run_project_contributions,
    ]
    for task in tasks:
        try:
            task()
        except Exception as exc:
            print(f"[Research] {task.__name__} skipped: {exc}")


def _run_feature_redundancy():
    from feature_redundancy_analysis import run_feature_redundancy_analysis
    run_feature_redundancy_analysis()


def _run_stability():
    from stability_analysis import run_stability_analysis
    run_stability_analysis()


def _run_error_analysis():
    from error_analysis import run_error_analysis
    run_error_analysis()


def run_survival_audit():
    raw = _csv(DATA_DIR, "ecommerce_churn_clean.csv")
    cox_existing = _csv(DATA_DIR, "cox_summary.csv")
    if raw is None:
        _write_survival_unavailable("Missing ecommerce_churn_clean.csv.")
        return

    duration_col = "Tenure"
    event_col = "Churn"
    candidate_cols = [
        "Tenure", "Churn", "SatisfactionScore", "DaySinceLastOrder", "Complain",
        "OrderCount", "CashbackAmount", "NumberOfDeviceRegistered",
        "WarehouseToHome", "HourSpendOnApp", "NumberOfAddress",
        "OrderAmountHikeFromlastYear", "CouponUsed", "CityTier",
    ]
    df = raw[[c for c in candidate_cols if c in raw.columns]].copy()
    df = df.dropna(subset=[duration_col, event_col])
    df[duration_col] = pd.to_numeric(df[duration_col], errors="coerce")
    df[event_col] = pd.to_numeric(df[event_col], errors="coerce").fillna(0).astype(int)
    df = df.dropna(subset=[duration_col])
    df = df[df[duration_col] >= 0].copy()
    df.loc[df[duration_col] == 0, duration_col] = 0.5

    covariates = [c for c in df.columns if c not in [duration_col, event_col]]
    for col in covariates:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna()

    assumption_df = pd.DataFrame()
    diagnostics = None
    fitted_summary = None
    try:
        from lifelines import CoxPHFitter
        from lifelines.statistics import proportional_hazard_test

        cph = CoxPHFitter()
        cph.fit(df[[duration_col, event_col] + covariates], duration_col=duration_col, event_col=event_col)
        fitted_summary = cph.summary.reset_index().rename(columns={"covariate": "Feature"})

        # Schoenfeld residual test. Counterintuitive HRs can occur when a feature's
        # effect changes over tenure, when churned customers are observed earlier,
        # or when correlated covariates suppress each other's marginal direction.
        ph = proportional_hazard_test(cph, df[[duration_col, event_col] + covariates], time_transform="rank")
        assumption_df = ph.summary.reset_index().rename(columns={
            "index": "Feature",
            "test_statistic": "test_statistic",
            "p": "p-value",
        })
        assumption_df["PH Assumption Pass/Fail"] = np.where(
            assumption_df["p-value"] >= 0.05, "Pass", "Fail"
        )
    except Exception:
        fitted_summary = None

    summary_source = fitted_summary
    if summary_source is None and cox_existing is not None:
        summary_source = cox_existing.rename(columns={"covariate": "Feature", "exp(coef)": "Hazard Ratio", "p": "p-value"})
    if summary_source is None:
        _write_survival_unavailable("CoxPH fitting failed and cox_summary.csv is unavailable.")
        return

    rows = []
    for _, row in summary_source.iterrows():
        feat = row.get("Feature", row.get("covariate", "Unknown"))
        hr = row.get("Hazard Ratio", row.get("exp(coef)", np.nan))
        p_val = row.get("p-value", row.get("p", np.nan))
        ph_match = assumption_df[assumption_df["Feature"] == feat] if not assumption_df.empty else pd.DataFrame()
        ph_value = ph_match["PH Assumption Pass/Fail"].iloc[0] if not ph_match.empty else "Not tested"
        ph_pass = True if ph_value == "Pass" else False if ph_value == "Fail" else "Not tested"
        confidence = _interpretation_confidence(p_val, ph_pass)
        rows.append({
            "Feature": feat,
            "Hazard Ratio": round(float(hr), 4) if pd.notna(hr) else np.nan,
            "p-value": round(float(p_val), 6) if pd.notna(p_val) else np.nan,
            "PH Assumption Pass/Fail": ph_value,
            "Interpretation confidence": confidence,
        })
    diagnostics = pd.DataFrame(rows)
    diagnostics.to_csv(os.path.join(OUT_DIR, "survival_diagnostics.csv"), index=False)

    if assumption_df.empty:
        assumption_df = pd.DataFrame([{
            "Feature": "Unavailable",
            "test_statistic": np.nan,
            "p-value": np.nan,
            "PH Assumption Pass/Fail": "Not tested",
        }])
    if "PH Assumption Pass/Fail" not in assumption_df.columns and "PH assumption passed" in assumption_df.columns:
        assumption_df["PH Assumption Pass/Fail"] = np.where(assumption_df["PH assumption passed"], "Pass", "Fail")
    assumption_cols = [c for c in ["Feature", "PH Assumption Pass/Fail", "test_statistic", "p-value"] if c in assumption_df.columns]
    assumption_df[assumption_cols].to_csv(os.path.join(OUT_DIR, "cox_assumption_test.csv"), index=False)
    _write_survival_report(raw, diagnostics)
    print("[Research] Saved: survival_diagnostics.csv, cox_assumption_test.csv, survival_validation_report.md, survival_business_interpretation.md")


def _interpretation_confidence(p_val, ph_pass):
    if pd.isna(p_val):
        return "Low"
    if ph_pass is False:
        return "Low"
    if p_val < 0.01 and ph_pass is True:
        return "High"
    if p_val < 0.05:
        return "Medium"
    return "Low"


def _write_survival_unavailable(reason):
    pd.DataFrame(columns=["Feature", "Hazard Ratio", "p-value", "PH assumption passed", "Interpretation confidence"]).to_csv(
        os.path.join(OUT_DIR, "survival_diagnostics.csv"), index=False
    )
    pd.DataFrame([{"Feature": "Unavailable", "PH Assumption Pass/Fail": reason, "test_statistic": np.nan, "p-value": np.nan}]).to_csv(
        os.path.join(OUT_DIR, "cox_assumption_test.csv"), index=False
    )
    _write_md("survival_interpretation_report.md", ["# Survival Interpretation Report", "", reason])
    _write_md("survival_validation_report.md", ["# Survival Validation Report", "", reason])
    _write_md("survival_business_interpretation.md", ["# Survival Business Interpretation", "", reason])


def _write_survival_report(raw, diagnostics):
    event_rate = raw["Churn"].mean() if "Churn" in raw.columns else np.nan
    censored = 1 - event_rate if pd.notna(event_rate) else np.nan
    top = diagnostics.dropna(subset=["Hazard Ratio"]).copy()
    top["distance"] = (top["Hazard Ratio"] - 1).abs()
    top = top.sort_values("distance", ascending=False).head(8)

    validation_lines = [
        "# Survival Validation Report",
        "",
        "## Pipeline Audit",
        "- Event definition: Churn = 1 is treated as the churn event.",
        "- Censoring definition: Churn = 0 is treated as right-censored, meaning the customer had not churned by the observed tenure.",
        "- Time variable: Tenure is used as the duration variable. Zero-tenure records are shifted to 0.5 months for Cox model stability.",
        f"- Observed event rate: {event_rate:.1%}; censored share: {censored:.1%}." if pd.notna(event_rate) else "- Event rate unavailable.",
        "",
        "## Why Hazard Ratios May Look Counterintuitive",
        "Hazard ratios are conditional estimates, not simple correlations. SatisfactionScore HR > 1 or DaySinceLastOrder HR < 1 can appear when correlated variables suppress each other, when dissatisfied customers churn early, when recent buyers include both active and rescue-at-risk customers, or when the proportional hazards assumption is weak for a feature.",
        "",
        "Business caveat: treat low-confidence or PH-violating features as directional signals for further investigation, not as standalone causal claims.",
        "",
    ]
    interpretation_lines = [
        "# Survival Business Interpretation",
        "",
        "## Business-Friendly Findings",
    ]
    for _, row in top.iterrows():
        hr = row["Hazard Ratio"]
        direction = "higher" if hr > 1 else "lower"
        interpretation_lines.append(f"- {row['Feature']}: HR={hr:.2f}. Customers with higher values have approximately {abs(hr):.2f}x {direction} churn hazard, conditional on the other Cox covariates. Confidence: {row['Interpretation confidence']}.")
    _write_md("survival_validation_report.md", validation_lines)
    _write_md("survival_business_interpretation.md", interpretation_lines)
    _write_md("survival_interpretation_report.md", validation_lines + interpretation_lines)


def run_cluster_persona_reports():
    profile = _csv(DATA_DIR, "cluster_profiles.csv")
    if profile is None or "cluster" not in profile.columns:
        return
    required = ["Tenure", "OrderCount", "CashbackAmount", "SatisfactionScore", "NumberOfDeviceRegistered", "y_true"]
    rows = []
    global_means = profile[[c for c in required if c in profile.columns]].mean(numeric_only=True)
    for cid, grp in profile.groupby("cluster"):
        stats = {c: float(grp[c].mean()) for c in required if c in grp.columns}
        churn_rate = stats.get("y_true", np.nan)
        name = _persona_name(stats, global_means)
        risk = "High" if churn_rate >= 0.25 else "Medium" if churn_rate >= 0.15 else "Low"
        characteristics = _cluster_characteristics(stats, global_means)
        rows.append({
            "Cluster": int(cid),
            "Persona Name": name,
            "Behavior Summary": characteristics,
            "Churn Risk": risk,
            "Recommended Strategy": _persona_strategy(name, risk),
        })
    out = pd.DataFrame(rows).sort_values("Cluster")
    out.to_csv(os.path.join(OUT_DIR, "cluster_persona_report.csv"), index=False)
    _write_cluster_summary(out)
    print("[Research] Saved: cluster_persona_report.csv, cluster_persona_summary.md")


def _persona_name(stats, means):
    churn = stats.get("y_true", 0)
    tenure = stats.get("Tenure", 0)
    orders = stats.get("OrderCount", 0)
    cashback = stats.get("CashbackAmount", 0)
    satisfaction = stats.get("SatisfactionScore", 0)
    devices = stats.get("NumberOfDeviceRegistered", 0)
    if churn >= 0.25:
        return "High-Risk Customers"
    if tenure >= means.get("Tenure", 0) and orders >= means.get("OrderCount", 0) and churn < 0.15:
        return "VIP Loyal Customers"
    if satisfaction < means.get("SatisfactionScore", 0) and orders >= means.get("OrderCount", 0):
        return "Service-Sensitive Active Customers"
    if tenure < means.get("Tenure", 0) and orders <= means.get("OrderCount", 0):
        return "New and Exploring Customers"
    if cashback >= means.get("CashbackAmount", 0) and orders >= means.get("OrderCount", 0):
        return "Discount-Driven Buyers"
    if devices >= means.get("NumberOfDeviceRegistered", 0):
        return "Multi-Device Convenience Shoppers"
    return "Steady Value Customers"


def _cluster_characteristics(stats, means):
    labels = []
    mapping = {
        "Tenure": "tenure",
        "OrderCount": "order count",
        "CashbackAmount": "cashback amount",
        "SatisfactionScore": "satisfaction",
        "NumberOfDeviceRegistered": "registered devices",
        "y_true": "churn rate",
    }
    for col, label in mapping.items():
        if col in stats and col in means:
            direction = "high" if stats[col] >= means[col] else "low"
            value = f"{stats[col]:.2f}" if col != "y_true" else f"{stats[col]:.1%}"
            labels.append(f"{direction} {label} ({value})")
    return "; ".join(labels)


def _persona_strategy(name, risk):
    if "VIP" in name:
        return "Protect loyalty with exclusive rewards, early access, and premium support."
    if "Discount" in name:
        return "Use margin-aware coupons, cashback caps, and personalized bundles."
    if "New" in name:
        return "Accelerate onboarding with category education, first-repeat purchase nudges, and low-friction support."
    if "Service-Sensitive" in name:
        return "Improve service experience, ask for feedback, and provide targeted support before risk escalates."
    if "High-Risk" in name:
        return "Prioritize complaint resolution, save offers, and service recovery within 24-48 hours."
    return "Maintain engagement with periodic relevance-based campaigns."


def _write_cluster_summary(out):
    lines = ["# Cluster Persona Summary", ""]
    for _, row in out.iterrows():
        lines.extend([
            f"## Cluster {row['Cluster']}: {row['Persona Name']}",
            f"- Why this cluster exists: {row['Behavior Summary']}.",
            f"- Risk level: {row['Churn Risk']}.",
            f"- Recommended action: {row['Recommended Strategy']}",
            "",
        ])
    _write_md("cluster_persona_summary.md", lines)


def run_business_roi_analysis():
    y_df = _csv(DATA_DIR, "y_test.csv")
    if y_df is None:
        return
    y = y_df.squeeze().astype(int).values
    model_specs = [
        ("Logistic Regression", "lr_base_model.pkl", "X_test.csv"),
        ("Decision Tree", "dt_clust_model.pkl", "X_test_clust.csv"),
        ("Random Forest", "rf_clust_model.pkl", "X_test_clust.csv"),
        ("XGBoost", "xgb_clust_model.pkl", "X_test_clust.csv"),
        ("CatBoost", "catboost_model.pkl", "X_test_clust.csv"),
    ]
    baseline_cost = int(y.sum()) * 500
    rows = []
    for name, model_name, x_name in model_specs:
        model_path = os.path.join(MODEL_DIR, model_name)
        X = _csv(DATA_DIR, x_name)
        if not os.path.exists(model_path) or X is None:
            if name == "CatBoost":
                continue
            rows.append({"Model": name, "Status": "Unavailable"})
            continue
        model = joblib.load(model_path)
        proba = model.predict_proba(X)[:, 1]
        pred = (proba >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        intervention_cost = (tp + fp) * 50
        expected_cost = fn * 500 + intervention_cost
        savings = baseline_cost - expected_cost
        roi = savings / intervention_cost if intervention_cost else np.nan
        retained_customers = int(tp)
        retention_value = retained_customers * 500
        rows.append({
            "Model": name,
            "Status": "Available",
            "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
            "Expected Cost": round(float(expected_cost), 2),
            "Expected Savings": round(float(savings), 2),
            "Cost Reduction %": round((savings / baseline_cost * 100) if baseline_cost else 0, 2),
            "Estimated Retained Customers": retained_customers,
            "Estimated Retention Value": round(float(retention_value), 2),
            "ROI Score": round(float(roi), 3) if pd.notna(roi) else np.nan,
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "business_roi_analysis.csv"), index=False)
    _plot_roi(df)
    _write_business_recommendation(df, baseline_cost)
    _write_business_roi_summary(df, baseline_cost)
    print("[Research] Saved: business_roi_analysis.csv, business_roi_comparison.png, business_recommendation.md, business_roi_summary.md")


def _plot_roi(df):
    avail = df[df.get("Status", "") == "Available"].copy()
    if avail.empty:
        return
    plt.figure(figsize=(10, 5))
    plt.bar(avail["Model"], avail["Cost Reduction %"], color="#2563eb")
    plt.ylabel("Cost Reduction (%)")
    plt.title("Business ROI Comparison by Model")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "business_roi_comparison.png"), dpi=150)
    plt.close()


def _write_business_recommendation(df, baseline_cost):
    avail = df[df.get("Status", "") == "Available"].copy()
    lines = [
        "# Business Recommendation",
        "",
        "Assumptions: false negatives cost 500 units in lost customer value; each targeted retention contact costs 50 units. Baseline cost assumes no retention intervention.",
        f"Baseline expected churn cost: {baseline_cost:.2f}.",
        "",
    ]
    if not avail.empty:
        best = avail.sort_values("Expected Savings", ascending=False).iloc[0]
        roi_score = best.get("ROI Score", best.get("Retention ROI", np.nan))
        lines.append(f"Recommended deployment candidate: {best['Model']} with {best['Cost Reduction %']}% estimated cost reduction and ROI score {roi_score}.")
        lines.append("The recommendation should be revisited after campaign A/B testing with observed retention uplift and real incentive costs.")
    _write_md("business_recommendation.md", lines)


def _write_business_roi_summary(df, baseline_cost):
    avail = df[df.get("Status", "") == "Available"].copy()
    lines = [
        "# Business ROI Summary",
        "",
        "This analysis converts model errors into expected business cost using a retained-customer value assumption of 500 units and a targeting/intervention cost of 50 units.",
        f"Baseline expected churn cost without intervention: {baseline_cost:.2f}.",
        "",
    ]
    if not avail.empty:
        best = avail.sort_values("Expected Savings", ascending=False).iloc[0]
        lines.append(f"Best economic model: {best['Model']} with {best['Cost Reduction %']}% cost reduction, {best['Estimated Retained Customers']} estimated retained customers, and ROI score {best['ROI Score']}.")
    lines.append("ROI should be recalibrated with observed campaign uplift, real incentive cost, and customer lifetime value.")
    _write_md("business_roi_summary.md", lines)


def run_shap_business_interpretation():
    feature_df = _csv(OUT_DIR, "feature_redundancy_analysis.csv")
    if feature_df is None or feature_df.empty:
        feature_df = _csv(DATA_DIR, "X_train_clust.csv")
        if feature_df is None:
            _write_md("shap_business_interpretation.md", ["# SHAP Business Interpretation", "", "Feature importance artifacts are unavailable."])
            return
        features = list(feature_df.columns[:15])
    else:
        features = feature_df.sort_values("Dominance Rank")["Feature"].head(15).tolist()

    lines = ["# SHAP Business Interpretation", ""]
    for feature in features:
        meaning, action = _business_feature_mapping(feature)
        lines.extend([
            f"## {feature}",
            f"- Technical Meaning: {meaning}",
            f"- Business Meaning: {meaning}",
            f"- Retention Recommendation: {action}",
            "",
        ])
    _write_md("shap_business_interpretation.md", lines)
    print("[Research] Saved: shap_business_interpretation.md")


def _business_feature_mapping(feature):
    mapping = {
        "Complain": ("Customer complaint indicator.", "Prioritize support intervention and confirm issue resolution."),
        "DaySinceLastOrder": ("Recency since the last purchase.", "Trigger reactivation outreach when inactivity rises."),
        "Tenure": ("Length of customer relationship.", "Tailor retention offers by lifecycle stage."),
        "CashbackAmount": ("Cashback value received by the customer.", "Use margin-aware cashback or reward optimization."),
        "OrderCount": ("Total order frequency.", "Protect frequent buyers with loyalty benefits and cross-sell offers."),
        "SatisfactionScore": ("Customer satisfaction rating.", "Escalate low-satisfaction customers to service recovery."),
        "NumberOfDeviceRegistered": ("Number of devices linked to the account.", "Monitor friction or account complexity for multi-device users."),
        "rfm_total": ("Combined recency, frequency, and monetary score.", "Use RFM tiering for campaign prioritization."),
        "inactivity_ratio": ("Inactive days relative to tenure.", "Act before inactivity becomes sustained churn risk."),
        "cashback_per_order": ("Reward intensity per order.", "Tune promotion dependency and reduce unnecessary subsidy."),
    }
    return mapping.get(feature, ("Encoded behavioral or profile feature used by the churn model.", "Review subgroup behavior and design targeted retention messaging."))


def run_fairness_extended_report():
    raw = _csv(DATA_DIR, "X_test_raw.csv")
    y_df = _csv(DATA_DIR, "y_test.csv")
    preds = _csv(DATA_DIR, "targeted_action_list.csv")
    if preds is None:
        preds = _csv(DATA_DIR, "predictions.csv")
    if raw is None or y_df is None or preds is None:
        return
    y = y_df.squeeze().astype(int).values
    if "churn_prediction" in preds.columns:
        pred = preds["churn_prediction"].astype(int).values
    else:
        pred = (preds["churn_probability"].values >= 0.5).astype(int)
    rows = []
    summary_rows = []
    for attr in ["Gender", "MaritalStatus"]:
        if attr not in raw.columns:
            continue
        group_metrics = []
        for group, idx in raw.groupby(attr).groups.items():
            idx = np.asarray(list(idx))
            yt, yp = y[idx], pred[idx]
            tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
            precision = precision_score(yt, yp, zero_division=0)
            recall = recall_score(yt, yp, zero_division=0)
            fpr = fp / (fp + tn) if (fp + tn) else 0
            fnr = fn / (fn + tp) if (fn + tp) else 0
            selection = yp.mean()
            group_metrics.append({"group": group, "recall": recall, "selection": selection})
            rows.append({
                "Protected Attribute": attr,
                "Group": group,
                "Support": int(len(idx)),
                "Precision": round(precision, 4),
                "Recall": round(recall, 4),
                "FPR": round(fpr, 4),
                "FNR": round(fnr, 4),
                "Selection Rate": round(selection, 4),
            })
        if group_metrics:
            recalls = [g["recall"] for g in group_metrics]
            selections = [g["selection"] for g in group_metrics]
            summary_rows.append({
                "Protected Attribute": attr,
                "Group": "Overall disparity",
                "Support": int(len(raw)),
                "Precision": np.nan,
                "Recall": np.nan,
                "FPR": np.nan,
                "FNR": np.nan,
                "Selection Rate": np.nan,
                "Demographic Parity Difference": round(max(selections) - min(selections), 4),
                "Equal Opportunity Difference": round(max(recalls) - min(recalls), 4),
            })
    df = pd.DataFrame(rows + summary_rows)
    df.to_csv(os.path.join(OUT_DIR, "fairness_extended_report.csv"), index=False)
    _write_fairness_summary(df)
    print("[Research] Saved: fairness_extended_report.csv, fairness_summary.md")


def _write_fairness_summary(df):
    lines = ["# Fairness Summary", ""]
    disparities = df[df["Group"] == "Overall disparity"]
    for _, row in disparities.iterrows():
        dpd = row.get("Demographic Parity Difference", np.nan)
        eod = row.get("Equal Opportunity Difference", np.nan)
        severity = "small" if max(dpd, eod) < 0.03 else "moderate" if max(dpd, eod) < 0.10 else "material"
        lines.append(f"- {row['Protected Attribute']}: demographic parity difference={dpd:.2%}, equal opportunity difference={eod:.2%}; disparity is {severity}.")
    lines.append("")
    lines.append("Fairness metrics are descriptive on the held-out sample and should be paired with domain review before policy decisions.")
    _write_md("fairness_summary.md", lines)


def run_research_findings():
    stat = _csv(OUT_DIR, "statistical_validation.csv")
    roi = _csv(OUT_DIR, "business_roi_analysis.csv")
    fairness = _csv(OUT_DIR, "fairness_extended_report.csv")
    cluster = _csv(OUT_DIR, "cluster_persona_report.csv")
    survival = _csv(OUT_DIR, "survival_diagnostics.csv")
    rows = []
    rows.append(_finding_rfm(stat))
    rows.append(_finding_cluster(stat, cluster))
    rows.append(_finding_survival(survival))
    rows.append(_finding_roi(roi))
    rows.append(_finding_fairness(fairness))
    df = pd.DataFrame(rows)
    expected_cols = ["Research Question", "Hypothesis", "Methodology", "Result", "Conclusion", "Business Impact"]
    df = df[[c for c in expected_cols if c in df.columns]]
    df.to_csv(os.path.join(OUT_DIR, "research_findings.csv"), index=False)
    _write_research_findings_md(df)
    print("[Research] Saved: research_findings.csv, research_findings.md")


def _write_research_findings_md(df):
    lines = [
        "# Research Findings",
        "",
        "This section summarizes the main empirical contributions of the churn project in an academic reporting style. The findings combine ablation-style validation, leakage-aware diagnostics, clustering interpretation, survival modeling, fairness, stability, and business cost analysis.",
        "",
    ]
    for i, row in df.iterrows():
        lines.extend([
            f"## Finding {i + 1}: {row['Research Question']}",
            f"- Hypothesis: {row['Hypothesis']}",
            f"- Methodology: {row['Methodology']}",
            f"- Result: {row['Result']}",
            f"- Conclusion: {row['Conclusion']}",
            f"- Business impact: {row['Business Impact']}",
            "",
        ])
    lines.append("Overall, the project moves beyond point-estimate ROC-AUC by adding uncertainty, interpretability, operational cost, fairness, time-to-event reasoning, and segment-level actionability.")
    _write_md("research_findings.md", lines)


def _finding_rfm(stat):
    row = _stat_row(stat, "With vs Without RFM")
    diff = row.get("Difference", np.nan) if row is not None else np.nan
    return {
        "Research Question": "Does RFM improve churn prediction?",
        "Hypothesis": "RFM features provide incremental predictive signal beyond behavioral variables.",
            "Methodology": "Bootstrap ROC-AUC comparison using saved model predictions and RFM feature occlusion.",
        "Result": f"ROC-AUC difference = {diff}" if pd.notna(diff) else "Unavailable",
        "Conclusion": _diff_conclusion(diff),
        "Business Impact": "Even if prediction lift is small, RFM remains useful for segmentation and campaign design.",
    }


def _finding_cluster(stat, cluster):
    row = _stat_row(stat, "With vs Without Cluster Features")
    diff = row.get("Difference", np.nan) if row is not None else np.nan
    n_personas = len(cluster) if cluster is not None else 0
    return {
        "Research Question": "Do cluster features improve churn prediction and interpretation?",
        "Hypothesis": "K-means membership captures latent behavioral structure.",
            "Methodology": "Bootstrap ROC-AUC comparison plus persona profiling by churn and engagement features.",
        "Result": f"ROC-AUC difference = {diff}; personas generated = {n_personas}.",
        "Conclusion": _diff_conclusion(diff),
        "Business Impact": "Cluster personas turn model outputs into targeted retention strategies.",
    }


def _finding_survival(survival):
    if survival is None or survival.empty:
        result = "Unavailable"
        conclusion = "Survival diagnostics should be regenerated after Cox fitting."
    else:
        high = survival[survival["Interpretation confidence"].isin(["High", "Medium"])]
        result = f"{len(high)} Cox features have medium/high interpretation confidence."
        conclusion = "Survival analysis is defensible when PH diagnostics and confidence flags are reported."
    return {
        "Research Question": "Which customer factors affect churn timing?",
        "Hypothesis": "Complaint and engagement variables change churn hazard over tenure.",
        "Methodology": "Cox proportional hazards model with PH assumption diagnostics.",
        "Result": result,
        "Conclusion": conclusion,
        "Business Impact": "Hazard ratios support timing-aware retention prioritization.",
    }


def _finding_roi(roi):
    if roi is None or roi.empty or "Expected Savings" not in roi.columns:
        result = "Unavailable"
        conclusion = "ROI evaluation requires model predictions."
        impact = "Cost assumptions should be calibrated with finance data."
    else:
        avail = roi[roi.get("Status", "") == "Available"]
        best = avail.sort_values("Expected Savings", ascending=False).iloc[0] if not avail.empty else None
        result = f"Best model = {best['Model']}, cost reduction = {best['Cost Reduction %']}%." if best is not None else "Unavailable"
        conclusion = "Business value should be judged by expected savings, not ROC-AUC alone."
        impact = "Provides a deployment-oriented criterion for retention model selection."
    return {
        "Research Question": "Which model produces the best economic retention value?",
        "Hypothesis": "The highest discrimination model also reduces expected churn cost.",
        "Methodology": "Confusion-matrix cost model with intervention cost and avoided churn loss.",
        "Result": result,
        "Conclusion": conclusion,
        "Business Impact": impact,
    }


def _finding_fairness(fairness):
    if fairness is None or fairness.empty:
        result = "Unavailable"
        conclusion = "Fairness cannot be assessed without protected attributes."
    else:
        disp = fairness[fairness["Group"] == "Overall disparity"]
        max_disp = 0 if disp.empty else max(
            disp["Demographic Parity Difference"].max(),
            disp["Equal Opportunity Difference"].max(),
        )
        result = f"Maximum observed disparity = {max_disp:.2%}."
        conclusion = "Disparity appears small" if max_disp < 0.03 else "Disparity requires monitoring"
    return {
        "Research Question": "Does the churn model show disparity across protected groups?",
        "Hypothesis": "Model error and selection rates are similar across Gender and MaritalStatus.",
        "Methodology": "Group metrics, demographic parity difference, and equal opportunity difference.",
        "Result": result,
        "Conclusion": conclusion,
        "Business Impact": "Supports responsible retention targeting and identifies monitoring needs.",
    }


def _stat_row(stat, experiment):
    if stat is None or stat.empty or "Experiment" not in stat.columns:
        return None
    rows = stat[stat["Experiment"] == experiment]
    return rows.iloc[0] if not rows.empty else None


def _diff_conclusion(diff):
    if pd.isna(diff):
        return "Evidence unavailable from current artifacts."
    if abs(float(diff)) < 0.005:
        return "Predictive improvement is negligible."
    return "Predictive improvement is potentially meaningful and should be validated on future cohorts."


def run_project_contributions():
    lines = [
        "# Project Contributions",
        "",
        "This project contributes an end-to-end, research-oriented churn analytics framework that connects predictive performance, explainability, segmentation, survival modeling, fairness, and business decision quality.",
        "",
        "## 1. Explainable AI",
        "The project uses model interpretation outputs to support transparent churn-risk reasoning and to connect individual predictions with business-facing drivers.",
        "",
        "## 2. Leakage Detection",
        "The workflow distinguishes training-time feature construction from post-hoc reporting layers and includes leakage-oriented comparison artifacts for defensible experimentation.",
        "",
        "## 3. Statistical Validation",
        "Bootstrap validation quantifies uncertainty in ROC-AUC differences, allowing claims about RFM, clustering, and leakage-sensitive features to be stated with confidence intervals rather than point estimates alone.",
        "",
        "## 4. Customer Segmentation",
        "K-means profiles are translated into actionable customer personas using tenure, order volume, cashback, satisfaction, device behavior, and churn rate.",
        "",
        "## 5. Survival Analysis",
        "Cox proportional hazards diagnostics add time-to-event reasoning, documenting event definition, censoring, tenure duration, PH assumptions, and interpretation confidence.",
        "",
        "## 6. Fairness Assessment",
        "Extended group metrics evaluate precision, recall, false-positive rates, false-negative rates, selection rates, demographic parity, and equal opportunity across available protected attributes.",
        "",
        "## 7. Business Cost Optimization",
        "Confusion-matrix cost analysis estimates expected cost, expected savings, cost reduction, and retention ROI for each available model.",
        "",
        "## 8. Actionable Retention Strategies",
        "The final reporting layer connects research findings to operational decisions, including service recovery, targeted coupons, VIP retention, onboarding, and risk-based prioritization.",
    ]
    _write_md("project_contributions.md", lines)
    print("[Research] Saved: project_contributions.md")


if __name__ == "__main__":
    run_research_quality_reports()
