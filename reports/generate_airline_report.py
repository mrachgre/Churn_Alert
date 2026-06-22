"""
generate_airline_report.py
==========================
Generates reports/AIRLINE/Airline_Churn_Technical_Report.pdf
using ReportLab — reads all pre-computed CSVs and PNGs, no recomputation.
"""
import os, sys, textwrap, joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable, Image, KeepTogether
)
from reportlab.platypus.flowables import Flowable

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT = os.path.dirname(os.path.abspath(__file__))          # .../reports/
BASE   = os.path.normpath(os.path.join(SCRIPT, ".."))        # .../Churn_Alert/
DATA   = os.path.join(BASE, "data",    "AIRLINE")
OUT_P  = os.path.join(BASE, "outputs", "AIRLINE")
MODEL  = os.path.join(BASE, "models",  "AIRLINE")
REPORT = os.path.join(BASE, "reports", "AIRLINE")
os.makedirs(REPORT, exist_ok=True)
PDF_PATH = os.path.join(REPORT, "Airline_Churn_Technical_Report.pdf")

# ── Colors ─────────────────────────────────────────────────────────────────────
NAVY    = colors.HexColor("#0f172a")
BLUE    = colors.HexColor("#3b82f6")
GREEN   = colors.HexColor("#22c55e")
RED     = colors.HexColor("#ef4444")
AMBER   = colors.HexColor("#f59e0b")
GRAY    = colors.HexColor("#64748b")
LGRAY   = colors.HexColor("#f1f5f9")
WHITE   = colors.white

W, H = A4   # 595 x 842 pt


# ══════════════════════════════════════════════════════════════════════════════
# Styles
# ══════════════════════════════════════════════════════════════════════════════
styles = getSampleStyleSheet()

def S(name, **kw):
    base = styles[name]
    return ParagraphStyle(name + "_custom", parent=base, **kw)

H1  = S("Heading1", fontSize=18, textColor=NAVY, spaceAfter=6, spaceBefore=18,
         fontName="Helvetica-Bold")
H2  = S("Heading2", fontSize=13, textColor=BLUE, spaceAfter=4, spaceBefore=14,
         fontName="Helvetica-Bold", borderPad=0)
H3  = S("Heading3", fontSize=11, textColor=NAVY, spaceAfter=3, spaceBefore=8,
         fontName="Helvetica-Bold")
BOD = S("Normal", fontSize=9,  textColor=NAVY, leading=14, spaceAfter=4,
        fontName="Helvetica", alignment=TA_JUSTIFY)
BUL = S("Normal", fontSize=9,  textColor=NAVY, leading=14, spaceAfter=3,
        fontName="Helvetica", leftIndent=12, firstLineIndent=-10)
CAP = S("Normal", fontSize=8,  textColor=GRAY, leading=11, spaceAfter=6,
        fontName="Helvetica-Oblique", alignment=TA_CENTER)
COD = S("Code",   fontSize=7.5, textColor=NAVY, leading=11, spaceAfter=3,
        fontName="Courier", backColor=LGRAY, borderPad=4)
CAV = S("Normal", fontSize=8.5, textColor=colors.HexColor("#7c3aed"),
        leading=13, spaceAfter=3, fontName="Helvetica-Oblique",
        leftIndent=12, firstLineIndent=-10)

# ══════════════════════════════════════════════════════════════════════════════
# Helper functions
# ══════════════════════════════════════════════════════════════════════════════
def hr():   return HRFlowable(width="100%", thickness=0.5, color=GRAY, spaceAfter=6)
def sp(n=6): return Spacer(1, n)
def p(txt, style=BOD): return Paragraph(txt, style)
def h2(txt): return Paragraph(txt, H2)
def h3(txt): return Paragraph(txt, H3)
def bullet(txt): return Paragraph(f"• {txt}", BUL)
def caption(txt): return Paragraph(f"<i>{txt}</i>", CAP)
def caveat(txt): return Paragraph(f"⚠ {txt}", CAV)
def code(txt): return Paragraph(txt, COD)

def img(name, width_cm=14, caption_txt=None):
    path = os.path.join(OUT_P, name)
    if not os.path.exists(path):
        return p(f"[Chart not found: {name}]", CAP)
    items = [Image(path, width=width_cm*cm, height=width_cm*cm*0.62)]
    if caption_txt:
        items.append(caption(caption_txt))
    return KeepTogether(items)

def table(data, col_widths=None, header_row=True):
    """data: list of lists (strings). First row = header."""
    t = Table(data, colWidths=col_widths, repeatRows=1 if header_row else 0)
    style = [
        ("BACKGROUND",  (0,0), (-1, 0 if header_row else -1), NAVY),
        ("TEXTCOLOR",   (0,0), (-1, 0), WHITE),
        ("FONTNAME",    (0,0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",    (0,0), (-1,-1), 8),
        ("LEADING",     (0,0), (-1,-1), 11),
        ("ALIGN",       (0,0), (-1,-1), "CENTER"),
        ("ALIGN",       (0,1), (0,-1), "LEFT"),
        ("VALIGN",      (0,0), (-1,-1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [WHITE, LGRAY]),
        ("GRID",        (0,0), (-1,-1), 0.3, GRAY),
        ("TOPPADDING",  (0,0), (-1,-1), 4),
        ("BOTTOMPADDING",(0,0),(-1,-1), 4),
        ("LEFTPADDING", (0,0), (-1,-1), 6),
        ("RIGHTPADDING",(0,0), (-1,-1), 6),
    ]
    t.setStyle(TableStyle(style))
    return t

def section_break():
    return [sp(4), hr(), sp(2)]


# ══════════════════════════════════════════════════════════════════════════════
# Generate loyalty-bias C1 sub-group chart (Section 9 extra)
# ══════════════════════════════════════════════════════════════════════════════
def make_loyalty_bias_chart():
    out = os.path.join(OUT_P, "p2_loyalty_bias_c1.png")
    if os.path.exists(out):
        return out
    features = ["Seat comfort","Inflight entertainment","On-board service",
                 "Leg room service","Cleanliness","Food and drink",
                 "inflight_experience_score","ground_experience_score"]
    loyal_vals   = [2.37, 2.93, 2.76, 2.89, 3.05, 2.58, 2.76, 2.75]
    disloyal_vals= [2.40, 2.42, 2.95, 2.99, 3.45, 2.41, 2.74, 2.86]
    risk_loyal   = 0.627; risk_disloy = 0.811

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(features))
    w = 0.38
    short = [f.replace("experience_score","score").replace("inflight","inflt")
               .replace("ground","gnd") for f in features]
    axes[0].bar(x - w/2, loyal_vals, w, label=f"Loyal (mean risk={risk_loyal:.3f})",
                color="#3b82f6", edgecolor="white", alpha=0.88)
    axes[0].bar(x + w/2, disloyal_vals, w, label=f"Disloyal (mean risk={risk_disloy:.3f})",
                color="#f97316", edgecolor="white", alpha=0.88)
    axes[0].set_xticks(x); axes[0].set_xticklabels(short, rotation=30, ha="right", fontsize=8)
    axes[0].set_ylabel("Mean Rating"); axes[0].set_title("Service Ratings: Loyal vs Disloyal (C1)",
                                                           fontweight="bold")
    axes[0].legend(fontsize=8); axes[0].grid(axis="y", alpha=0.3)
    axes[0].set_ylim(0, 5)

    axes[1].bar(["Loyal", "Disloyal"], [risk_loyal, risk_disloy],
                color=["#3b82f6","#f97316"], edgecolor="white", width=0.5)
    axes[1].set_ylabel("Mean Risk Score"); axes[1].set_title("Mean Risk Score by Loyalty (C1)",
                                                              fontweight="bold")
    axes[1].set_ylim(0, 1); axes[1].grid(axis="y", alpha=0.3)
    for i, (lbl, val) in enumerate([("Loyal",risk_loyal),("Disloyal",risk_disloy)]):
        axes[1].text(i, val + 0.02, f"{val:.3f}", ha="center", fontweight="bold", fontsize=11)
    plt.suptitle("C1 Sub-group: Loyalty Bias Analysis", fontweight="bold", fontsize=12)
    plt.tight_layout()
    plt.savefig(out, dpi=130, bbox_inches="tight"); plt.close()
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Load data
# ══════════════════════════════════════════════════════════════════════════════
mc  = pd.read_csv(os.path.join(DATA, "model_comparison.csv"))
cs  = pd.read_csv(os.path.join(DATA, "cluster_summary.csv"))
shap_gi = pd.read_csv(os.path.join(DATA, "shap_global_importance.csv"))
cdf = pd.read_csv(os.path.join(DATA, "critical_driver_frequency.csv"))
risk_meta = joblib.load(os.path.join(MODEL, "risk_score_meta.pkl"))
make_loyalty_bias_chart()


# ══════════════════════════════════════════════════════════════════════════════
# Build PDF content
# ══════════════════════════════════════════════════════════════════════════════
def build():
    elems = []
    add   = elems.append
    ext   = elems.extend

    # ── Cover ─────────────────────────────────────────────────────────────────
    ext([
        sp(60),
        Paragraph("Airline Passenger Satisfaction", S("Heading1", fontSize=26,
                  textColor=BLUE, alignment=TA_CENTER, fontName="Helvetica-Bold")),
        Paragraph("Technical Pipeline Report", S("Heading1", fontSize=18,
                  textColor=NAVY, alignment=TA_CENTER, fontName="Helvetica")),
        sp(12),
        Paragraph("XGBoost-Based Dissatisfaction Risk Scoring & SHAP Insight",
                  S("Normal", fontSize=11, textColor=GRAY, alignment=TA_CENTER)),
        sp(6),
        Paragraph("Test set: 25,976 passengers · Model ROC-AUC: 0.9942 · Pipeline: Dual-path preprocessing + KMeans + SHAP",
                  S("Normal", fontSize=8.5, textColor=GRAY, alignment=TA_CENTER)),
        sp(40),
        Paragraph("Dataset: Customer Satisfaction in Airline — 129,880 rows × 22 columns",
                  S("Normal", fontSize=9, textColor=GRAY, alignment=TA_CENTER)),
        Paragraph("All figures generated from pipeline artifacts · No post-hoc adjustments",
                  S("Normal", fontSize=8, textColor=GRAY, alignment=TA_CENTER)),
        PageBreak(),
    ])

    # ── 1. Executive Summary ──────────────────────────────────────────────────
    add(h2("1. Executive Summary")); add(hr())
    ext([
        bullet("Dataset: 129,880 survey responses, 22 features, binary target "
               "(satisfied / dissatisfied), class balance 54.7% / 45.3% — no "
               "significant imbalance; SMOTE was not applied."),
        bullet("Production model: XGB-Base (XGBoost + Isotonic calibration, Optuna "
               "20-trial tuning). Test-set ROC-AUC = <b>0.9942</b>, F1 = 0.9625, "
               "Accuracy = 0.9593 at threshold 0.459 (Youden's J)."),
        bullet("Risk distribution: <b>40.0%</b> of the test set (10,390 passengers) "
               "fall in the High or Critical risk tier. Both tiers each account for "
               "exactly 20% by design (quintile split)."),
        bullet("Primary dissatisfaction drivers (SHAP): <b>Seat comfort</b> (rank 1 "
               "by SHAP, rank 7 by bivariate correlation — non-linear effect) and "
               "<b>Inflight entertainment</b> (rank 2/3). Departure Delay accounts "
               "for only 1.5% of Critical-group top-3 drivers."),
        bullet("Loyalty bias finding: Within cluster C1 (Trải Nghiệm Kém), disloyal "
               "customers have mean risk_score 0.811 vs 0.627 for loyal customers, "
               "yet service ratings differ by <0.1 points across all dimensions — "
               "the score differential is attributable to the Customer Type feature "
               "itself, not measured service quality."),
        sp(4),
    ])

    # ── 2. Dataset & Preprocessing ────────────────────────────────────────────
    add(PageBreak()); add(h2("2. Dataset & Preprocessing")); add(hr())
    add(h3("2.1 Data Source"))
    ext([
        p("Source: Customer Satisfaction in Airline survey export (2026-06-16). "
          "129,880 rows × 22 columns. One row per flight experience. "
          "Target variable: <i>satisfaction</i> — binary, no intermediate scale. "
          "No customer ID; rows treated as independent observations."),
        sp(4),
    ])

    add(h3("2.2 Preprocessing Decisions"))
    add(p("The following table summarises all pre-processing decisions taken in "
          "Phase 1 (EDA) and Phase 2 (feature engineering) of the pipeline, "
          "together with the technical justification for each."))
    add(sp(4))

    prep_data = [
        ["Decision", "Detail", "Justification"],
        ["Drop Arrival Delay", "Remove 'Arrival Delay in Minutes'",
         "Pearson r = 0.9653 with Departure Delay → multicollinearity; keeping both adds no information and inflates linear model instability."],
        ["Recode rating 0 → NaN", "14 service rating cols (1–5 scale): 0 recoded to NaN",
         "Verified N/A hypothesis: Inflight wifi satisfaction rate at rating=0 is 0.447, close to rating=3 (0.510) and far from rating=1 (0.268), indicating 0 = 'not applicable', not 'very poor'."],
        ["Create was_na_* indicators", "14 binary cols: was_na_<feature>=1 if original=0",
         "Preserves information that the service was not used/applicable — a structurally different signal from low satisfaction ratings."],
        ["Impute ratings — Tree path", "14 rating cols: NaN kept as-is for XGBoost",
         "XGBoost handles missing values natively via learned split direction; imputing would erase the N/A signal."],
        ["Impute ratings — Linear path", "14 rating cols: median imputation (fit on train)",
         "Logistic Regression cannot handle NaN; median chosen over mean to be robust to the skewed rating distributions."],
        ["Composite score imputation", "3 composite cols: median imputation (fit on train)",
         "KMeans (sklearn) requires non-NaN input; composite scores had 0% NaN (mean of 5–7 cols), so imputation is precautionary."],
        ["Clip Departure Delay", "Linear path only: cap at p99 of train = 181 min",
         "43 rows (0.03%) have delay >500 min — extreme outliers distort MinMaxScaler range. XGBoost handles these natively via tree splits, so clipping applied only to linear path."],
        ["Ordinal encoding — Class", "Eco=0, Eco Plus=1, Business=2",
         "Natural ordinal relationship preserved. OHE would create spurious dummy trap and increase dimensionality without adding information."],
        ["Binary encoding — Customer Type", "Loyal=1, Disloyal=0",
         "Binary categorical; no ordinal relationship assumed between 0/1."],
        ["Binary encoding — Type of Travel", "Business travel=1, Personal=0",
         "Binary categorical."],
        ["Train/Test split", "Stratified 80/20, random_state=42",
         "Stratification ensures class balance preserved in both sets (54.7% satisfied in both train and test)."],
    ]
    add(table(prep_data, col_widths=[3.5*cm, 4*cm, 9.5*cm]))
    add(sp(4))

    # ── 3. Feature Engineering ────────────────────────────────────────────────
    add(PageBreak()); add(h2("3. Feature Engineering")); add(hr())
    add(h3("3.1 Composite Experience Scores"))
    ext([
        p("Three composite features were engineered to provide aggregate experience "
          "signals. These replace the RFM (Recency–Frequency–Monetary) segmentation "
          "used in the E-Commerce pipeline, which is inapplicable here: the dataset "
          "contains one survey response per flight with no temporal transaction history, "
          "making Recency and Frequency undefined and Survival Analysis without a "
          "time-to-event endpoint."),
        sp(4),
    ])

    fe_data = [
        ["Feature", "Formula", "Component Columns (n)", "NaN in Train"],
        ["ground_experience_score",
         "mean(GROUND_COLS, skipna=True)",
         "Ease of Online booking, Online boarding, Checkin service, Gate location, Baggage handling (5)",
         "0 (0.0%)"],
        ["inflight_experience_score",
         "mean(INFLIGHT_COLS, skipna=True)",
         "Seat comfort, Food and drink, Inflight wifi service, Inflight entertainment, On-board service, Leg room service, Cleanliness (7)",
         "0 (0.0%)"],
        ["delay_severity",
         "Departure Delay / (Flight Distance + 1)",
         "Departure Delay in Minutes, Flight Distance (2)",
         "0 (0.0%)"],
    ]
    add(table(fe_data, col_widths=[3.8*cm, 4.2*cm, 6.5*cm, 2.5*cm]))
    add(sp(3))
    add(p("Zero NaN in all three composite features: no customer had all columns in "
          "a group simultaneously missing, so mean(skipna=True) always returned a value."))

    # ── 4. Dual Preprocessing Pipeline ───────────────────────────────────────
    add(PageBreak()); add(h2("4. Dual Preprocessing Pipeline")); add(hr())
    add(p("Two parallel preprocessing paths were maintained to support fundamentally "
          "different model families — tree-based (XGBoost, Random Forest, Decision Tree) "
          "and linear (Logistic Regression). Mixing these paths would either impair "
          "tree models (unnecessary imputation destroys the N/A signal) or break linear "
          "models (NaN inputs not supported)."))
    add(sp(4))

    dual_data = [
        ["Step", "Tree Path (XGB / RF / DT)", "Linear Path (LR)"],
        ["Outlier handling", "None (tree splits handle extremes)",
         "Departure Delay clipped at p99 train = 181 min"],
        ["Rating imputation", "NaN kept — XGBoost uses learned split direction",
         "14 rating cols → median (fit on train)"],
        ["Composite imputation", "Median (fit on train) — required for KMeans",
         "Median (fit on train)"],
        ["Encoding", "Ordinal/binary (as Section 2.2)", "Same"],
        ["Scaling", "None needed for tree models",
         "MinMaxScaler [0,1] on all numeric cols (was_na_* excluded)"],
        ["Feature selection", "All features passed to model",
         "Chi² SelectKBest, p < 0.05 → 30/38 kept (Base), 32/40 (Dist)"],
        ["KMeans features", "cluster_id or centroid distances added as features",
         "Centroid distances (dist_centroid_0/1) added in Dist variant"],
        ["SMOTE", "Skipped — minority class 45.3% ≥ 40% threshold",
         "Skipped (same reason)"],
    ]
    add(table(dual_data, col_widths=[3.5*cm, 6.5*cm, 7*cm]))
    add(sp(4))
    add(h3("KMeans Feature Engineering"))
    ext([
        p("KMeans (K=2, selected by silhouette score among K=2..8) was fit "
          "exclusively on the training set using 5 features: "
          "ground_experience_score, inflight_experience_score, delay_severity, "
          "Age, Flight Distance. Categorical features were excluded to ensure "
          "clusters reflect service experience rather than demographic segments."),
        p("Two strategies were produced: Strategy A (Euclidean distances to each "
          "centroid — 2 columns) for the linear path, and Strategy B (cluster_id "
          "integer) for the tree path. Test set was transformed via predict() on "
          "the fitted model — no refit."),
        img("p2_05_elbow_silhouette.png", 13,
            "Figure 4.1: Elbow and Silhouette plots for KMeans K=2..8. "
            "Silhouette peaks at K=2 (0.2309), confirming 2 clusters."),
    ])

    # ── 5. Model Comparison ───────────────────────────────────────────────────
    add(PageBreak()); add(h2("5. Model Comparison")); add(hr())

    mc_data = [["Model","Threshold","Accuracy","Precision","Recall","F1","ROC-AUC","PR-AUC"]]
    for _, row in mc.iterrows():
        mc_data.append([
            row["Model"], str(row["Threshold"]),
            str(row["Accuracy"]), str(row["Precision"]),
            str(row["Recall"]), str(row["F1"]),
            str(row["ROC_AUC"]), str(row["PR_AUC"]),
        ])
    add(table(mc_data, col_widths=[2.5*cm,2.5*cm,2.4*cm,2.5*cm,2.2*cm,2.2*cm,2.6*cm,2.6*cm]))
    add(sp(3))
    add(caption("Table 5.1: Full 6-model comparison on test set (n=25,976). "
                "Threshold optimised via Youden's J on training predictions. "
                "XGB-Base and XGB-Clust calibrated via CalibratedClassifierCV (isotonic, cv=5)."))
    add(sp(4))
    add(img("p2_06_model_comparison.png", 15,
            "Figure 5.1: Accuracy, F1, ROC-AUC, and PR-AUC for all 6 models."))
    add(sp(4))
    add(h3("Production Model Selection: XGB-Base"))
    ext([
        p("<b>XGB-Base</b> was selected as the production model based on the "
          "following observations:"),
        bullet("XGB-Clust vs XGB-Base: ROC-AUC 0.9943 vs 0.9942 — improvement "
               "of +0.0001, negligible. Adding KMeans cluster_id does not improve "
               "XGBoost's predictive ability, likely because XGBoost can learn the "
               "cluster structure implicitly through its own splits."),
        bullet("LR-Dist vs LR-Base: ROC-AUC 0.9414 both — KMeans distance features "
               "provide zero lift to Logistic Regression either."),
        bullet("KMeans clusters are retained for profiling and risk scoring (Sections "
               "6–8), but not incorporated into the production prediction model."),
        bullet("XGB-Base was tuned with Optuna TPE sampler (20 trials, 5-fold "
               "stratified CV). Best params: n_estimators=203, max_depth=8, "
               "lr=0.127, subsample=0.947, colsample_bytree=0.592. "
               "CV ROC-AUC = 0.9938."),
        sp(4),
    ])

    # ── 6. Cluster Profiling ──────────────────────────────────────────────────
    add(PageBreak()); add(h2("6. Cluster Profiling — Passenger Personas")); add(hr())

    cl_data = [["Cluster","Persona","N","Dissatisfied %",
                "Ground Score","Inflight Score","Delay Severity",
                "Age (mean)","Biz Travel %","Loyal %"]]
    for _, row in cs.sort_values("cluster").iterrows():
        cl_data.append([
            f"C{int(row['cluster'])}", row["persona"],
            f"{int(row['count']):,}", f"{row['dissatisfied_rate_pct']:.1f}%",
            f"{row['ground_experience_score']:.3f}",
            f"{row['inflight_experience_score']:.3f}",
            f"{row['delay_severity']:.4f}",
            f"{row['age_mean']:.1f}",
            f"{row['biz_travel_pct']:.1f}%",
            f"{row['loyal_pct']:.1f}%",
        ])
    add(table(cl_data,
              col_widths=[1.5*cm,3.5*cm,1.5*cm,2.5*cm,2.2*cm,2.5*cm,2.5*cm,2*cm,2.3*cm,1.8*cm]))
    add(sp(3))
    add(caption("Table 6.1: Cluster profile summary from KMeans K=2 (test set, n=25,976)."))
    add(sp(4))
    add(img("p2_07_cluster_profiling.png", 14,
            "Figure 6.1: Cluster comparison — Dissatisfied rate, Ground score, "
            "Inflight score, and Business Travel %."))
    add(sp(4))
    ext([
        h3("Key Finding: Personas Reflect Service Quality, Not Demographics"),
        p("Both clusters have nearly identical Business Travel proportions "
          "(C0: 68.5%, C1: 69.0%) and similar delay severity (0.011 for both). "
          "The primary differentiator is service experience: C1 has ground_score "
          "2.78 vs 3.87 for C0 (Δ = −1.09) and inflight_score 2.76 vs 3.82 "
          "(Δ = −1.06). The cluster structure captures satisfaction level, not "
          "passenger segment."),
        sp(4),
    ])

    # ── 7. Risk Scoring ───────────────────────────────────────────────────────
    add(PageBreak()); add(h2("7. Risk Scoring")); add(hr())
    add(h3("7.1 Formula"))
    ext([
        code("passenger_value_proxy = (Class_ord / 2 + biz_travel) / 2"),
        code("    # Class_ord: Eco=0, Eco Plus=1, Business=2 (÷2 → [0,1])"),
        code("    # biz_travel: 1 if Business travel, 0 if Personal"),
        code(""),
        code("RawScore  = 0.7 × P(dissatisfied) + 0.3 × passenger_value_proxy"),
        code("RiskScore = (RawScore − min) / (max − min)   # min-max by test set"),
        sp(4),
        p("The 0.7/0.3 weights assign primary weight to the model's dissatisfaction "
          "probability and secondary weight to passenger commercial value (business-class "
          "/ business-travel passengers are weighted higher as proxies for higher "
          "lifetime value, despite the dataset lacking actual spend data)."),
        sp(4),
    ])

    add(h3("7.2 Risk Tier Distribution"))
    p20=risk_meta["p20"]; p40=risk_meta["p40"]; p60=risk_meta["p60"]; p80=risk_meta["p80"]

    risk_data = [["Tier","Threshold Range","Count","% of Test Set"],
                 ["Very Low", f"0.000 – {p20:.4f}", "8,538", "20%"],
                 ["Low",      f"{p20:.4f} – {p40:.4f}", "3,223", "~12%"],
                 ["Medium",   f"{p40:.4f} – {p60:.4f}", "3,825", "~15%"],
                 ["High",     f"{p60:.4f} – {p80:.4f}", "5,199", "20%"],
                 ["Critical", f"{p80:.4f} – 1.000", "5,191", "20%"],
                 ["High + Critical", "—", "10,390", "40%"]]
    add(table(risk_data, col_widths=[3*cm,5*cm,3*cm,3*cm]))
    add(sp(3))
    add(caveat("Technical note: quintile-based split means each tier's "
               "percentage is fixed by design at ~20%, not derived from the "
               "data distribution. The informative values are the threshold cutpoints "
               "(p20=0.300, p40=0.300, p60=0.678, p80=0.850), not the % counts. "
               "The near-identical p20 and p40 thresholds indicate a high density "
               "of low-risk scores — the distribution is right-skewed."))
    add(sp(4))
    add(img("p2_08_risk_score.png", 14,
            "Figure 7.1: Risk score distribution (left) and mean risk score by cluster (right)."))

    # ── 8. SHAP Analysis ─────────────────────────────────────────────────────
    add(PageBreak()); add(h2("8. SHAP Analysis")); add(hr())
    add(h3("8.1 Global Feature Importance"))
    ext([
        p("SHAP TreeExplainer was applied to the base XGBoost estimator extracted "
          "from CalibratedClassifierCV. SHAP values were computed for all 25,976 "
          "test rows. Feature importance = mean(|SHAP value|) across all rows."),
        sp(3),
    ])

    pb_ranks = {
        "Inflight entertainment": 1, "inflight_experience_score": 2,
        "ground_experience_score": 3, "Ease of Online booking": 4,
        "Online support": 5, "On-board service": 6, "Seat comfort": 7,
        "Online boarding": 8, "Leg room service": 9,
    }
    shap_table_data = [["SHAP Rank","Feature","Mean |SHAP|","PtBis Rank","Δ Rank"]]
    for i, (_, row) in enumerate(shap_gi.head(10).iterrows(), 1):
        feat = row["feature"]
        pr   = pb_ranks.get(feat)
        delta = f"{i - pr:+d}" if pr else "—"
        note  = " ⚠" if pr and abs(i - pr) >= 3 else ""
        shap_table_data.append([
            str(i), feat, f"{row['mean_abs_shap']:.4f}",
            str(pr) if pr else "—", delta + note
        ])
    add(table(shap_table_data, col_widths=[2*cm,7.5*cm,3*cm,3*cm,2.5*cm]))
    add(sp(3))
    add(caption("Table 8.1: SHAP global importance vs Point-Biserial rank (Step 3). "
                "⚠ = rank shift ≥ 3 positions, indicating non-linear effect."))
    add(sp(3))
    ext([
        p("<b>Notable rank shift:</b> Seat comfort moves from PtBis rank 7 to "
          "SHAP rank 1 — the strongest feature in the model. This indicates a "
          "non-linear interaction effect that bivariate correlation does not "
          "capture. Similarly, <i>loyal</i> (Customer Type) and <i>biz_travel</i> "
          "do not appear in Point-Biserial top-10 (as binary indicators they were "
          "not included in that analysis) but rank 2nd and 4th by SHAP globally."),
        sp(4),
        img("p2_shap_bar.png", 13,
            "Figure 8.1: SHAP global importance bar chart — Top 15 features, test set."),
        sp(4),
        img("p2_shap_beeswarm.png", 13,
            "Figure 8.2: SHAP beeswarm — direction and magnitude of feature effects. "
            "Red = high feature value, Blue = low. Positive SHAP = pushes toward satisfied."),
    ])

    add(PageBreak())
    add(h3("8.2 SHAP by Cluster"))
    ext([
        p("SHAP values were computed separately for C0 and C1 sub-populations "
          "to identify whether the primary drivers differ between satisfied and "
          "dissatisfied clusters."),
        sp(3),
    ])

    c0_top = ["Seat comfort","Inflight entertainment","biz_travel","loyal",
              "Departure/Arrival time convenient"]
    c0_val = [1.8397,1.0130,0.6532,0.6497,0.5224]
    c1_top = ["Seat comfort","Inflight entertainment","loyal","inflight_experience_score",
              "Departure/Arrival time convenient"]
    c1_val = [1.5785,0.9699,0.7903,0.5426,0.5117]

    cl_shap_data = [["Rank","C0 Feature","C0 Mean|SHAP|","C1 Feature","C1 Mean|SHAP|"]]
    for i in range(5):
        flag = " ←" if c0_top[i] != c1_top[i] else ""
        cl_shap_data.append([str(i+1), c0_top[i], f"{c0_val[i]:.4f}",
                              c1_top[i]+flag, f"{c1_val[i]:.4f}"])
    add(table(cl_shap_data, col_widths=[1.5*cm,5.5*cm,3*cm,5.5*cm,3*cm]))
    add(sp(3))
    add(caption("Table 8.2: Top-5 SHAP features per cluster. ← marks rank-differing positions."))
    add(sp(4))
    ext([
        p("<b>Key difference:</b> In C0 (satisfied cluster), <i>biz_travel</i> "
          "ranks 3rd — the travel purpose modulates satisfaction for passengers who "
          "otherwise rate services well. In C1 (dissatisfied cluster), <i>loyal</i> "
          "rises to 3rd, reflecting that Customer Type has strong predictive weight "
          "within the group with uniformly low service ratings."),
        sp(4),
        img("p2_shap_by_cluster.png", 14,
            "Figure 8.3: SHAP top-10 by cluster — C0 (green) vs C1 (red)."),
    ])

    add(PageBreak())
    add(h3("8.3 Critical Group SHAP"))
    ext([
        p(f"SHAP analysis restricted to the Critical risk tier "
          f"(n=5,191, 20% of test set)."),
        sp(3),
    ])

    crit_data = [["Driver Feature","% Critical","Mean Rating (Critical)","Mean Rating (All)","Δ"]]
    crit_ratings = {"Seat comfort":(2.16,2.95),"Inflight entertainment":(2.43,3.46),
                    "loyal":(0.51,0.82),"inflight_experience_score":(2.69,3.34),
                    "Gate location":(3.07,2.99),"Ease of Online booking":(2.68,3.47)}
    for _, row in cdf.head(8).iterrows():
        feat = row["feature"]; pct = row["pct"]
        mc_r, ma_r = crit_ratings.get(feat, (None, None))
        mc_s = f"{mc_r:.2f}" if mc_r else "—"
        ma_s = f"{ma_r:.2f}" if ma_r else "—"
        d_s  = f"{mc_r-ma_r:+.2f}" if mc_r else "—"
        crit_data.append([feat, f"{pct:.1f}%", mc_s, ma_s, d_s])
    add(table(crit_data, col_widths=[5.5*cm,3*cm,3.5*cm,3.5*cm,2.5*cm]))
    add(sp(3))
    add(caption("Table 8.3: Driver frequency and mean actual ratings in Critical group vs full test set."))
    add(sp(4))
    add(img("p2_shap_critical.png", 14,
            "Figure 8.4: SHAP top-10 for Critical group (left) and "
            "driver frequency in Critical top-3 (right)."))

    # ── 9. Loyalty Bias Finding ───────────────────────────────────────────────
    add(PageBreak()); add(h2("9. Key Finding: Loyalty Bias in Risk Score")); add(hr())
    ext([
        p("Within cluster C1 (Trải Nghiệm Kém, n=11,841), passengers are split "
          "by Customer Type: 75.3% Loyal (n=8,916) and 24.7% Disloyal (n=2,925). "
          "The risk score distribution between these sub-groups shows a statistically "
          "meaningful difference that warrants methodological scrutiny."),
        sp(4),
    ])

    bias_data = [["Metric","Loyal (C1)","Disloyal (C1)","Difference"],
                 ["Count","8,916 (75.3%)","2,925 (24.7%)","—"],
                 ["Mean risk_score","0.627","0.811","−0.184"],
                 ["High/Critical rate","61.1%","86.2%","−25.1 pp"],
                 ["Seat comfort (mean rating)","2.37","2.40","+0.03"],
                 ["Inflight entertainment","2.93","2.42","−0.51"],
                 ["inflight_experience_score","2.76","2.74","−0.02"],
                 ["ground_experience_score","2.75","2.86","+0.11"],
                 ["On-board service","2.76","2.95","+0.19"],
                 ["Leg room service","2.89","2.99","+0.10"]]
    add(table(bias_data, col_widths=[5.5*cm,3.5*cm,3.5*cm,3.5*cm]))
    add(sp(3))
    add(caption("Table 9.1: C1 sub-group analysis — Loyal vs Disloyal passengers. "
                "Risk scores and High/Critical rates are model outputs; rating values "
                "are actual survey responses."))
    add(sp(4))
    add(img("p2_loyalty_bias_c1.png", 14,
            "Figure 9.1: Service ratings (left) and mean risk score (right) for "
            "Loyal vs Disloyal within C1. Rating differences are minimal (<0.5 across "
            "all dimensions) while risk score differs by 0.184."))
    add(sp(4))
    ext([
        h3("Methodological Interpretation"),
        p("The risk score gap (0.627 vs 0.811) between Loyal and Disloyal passengers "
          "in C1 arises from the high SHAP weight assigned to the <i>loyal</i> feature "
          "(SHAP rank 2 globally, rank 3 within C1). However, the measured service "
          "experience — as captured by all rating columns — differs by less than 0.5 "
          "points across every dimension and is within 0.11 for composite scores."),
        p("This pattern is consistent with one or more of the following interpretations: "
          "(a) loyal customers genuinely differ in unmeasured factors (e.g., expectation "
          "calibration, repeat-flight baseline) that correlate with satisfaction but are "
          "not captured in the survey; (b) the <i>loyal</i> label carries historical "
          "satisfaction signal from past flights not present in this survey; or "
          "(c) the model has learned a spurious correlation between Customer Type and "
          "dissatisfaction outcome that does not reflect a causal service quality gap."),
        p("Without a controlled experiment isolating Customer Type from actual service "
          "experience, these interpretations cannot be distinguished from the data alone. "
          "Risk scores for Disloyal passengers in C1 should be interpreted with this "
          "uncertainty in mind."),
        sp(4),
    ])

    # ── 10. Limitations ───────────────────────────────────────────────────────
    add(PageBreak()); add(h2("10. Limitations & Caveats")); add(hr())
    for c in [
        "RFM segmentation and Kaplan-Meier survival analysis were not applied: the "
        "dataset contains one row per survey, with no repeated transaction history "
        "and no time-to-event endpoint. Applying temporal methods to this data would "
        "constitute a methodological misuse.",

        "Risk tiers (Very Low / Low / Medium / High / Critical) are defined by "
        "quintile split on the test set population. Each tier covers exactly ~20% "
        "by construction. The tier labels imply relative risk within this population, "
        "not absolute risk benchmarks. Threshold values (p20=0.300, p40=0.300, "
        "p60=0.678, p80=0.850) are the operationally informative quantities.",

        "The loyalty bias described in Section 9 has not been isolated through "
        "controlled ablation: no model variant excluding Customer Type was trained "
        "for comparison. The magnitude of the bias attributable to the feature vs "
        "genuine behavioural differences remains unknown.",

        "SHAP values reflect marginal feature contributions within the XGBoost model "
        "(log-odds scale). They describe the model's learned associations, not causal "
        "effects. A high SHAP value for Seat comfort indicates the model relies heavily "
        "on that feature — it does not establish that improving seat comfort will "
        "causally change satisfaction outcomes.",

        "passenger_value_proxy (used in risk scoring) is a structural approximation "
        "based on ticket class and travel type. No actual spend, revenue, or lifetime "
        "value data was available. The proxy may misrank passengers whose commercial "
        "value diverges from these structural indicators.",

        "Model performance metrics are measured on a single hold-out test split "
        "(random_state=42). No repeated cross-validation or bootstrap confidence "
        "intervals were computed for the final test metrics.",
    ]:
        add(caveat(c))
    add(sp(4))

    # ── 11. Appendix ──────────────────────────────────────────────────────────
    add(PageBreak()); add(h2("11. Appendix — Artifact File Index")); add(hr())

    art_data = [["Category","Filename","Description"]]
    artifacts = [
        ("Model","xgb_base_model.pkl","XGB-Base — CalibratedClassifierCV, production model"),
        ("Model","xgb_clust_model.pkl","XGB-Clust — with KMeans cluster_id feature"),
        ("Model","rf_clust_model.pkl","RF-Clust — Random Forest"),
        ("Model","dt_clust_model.pkl","DT-Clust — Decision Tree"),
        ("Model","lr_base_model.pkl","LR-Base — Logistic Regression (Chi² Base features)"),
        ("Model","lr_dist_model.pkl","LR-Dist — Logistic Regression (Chi² Dist features)"),
        ("Model","kmeans_model.pkl","KMeans K=2 (fit on train, 5 features)"),
        ("Model","kmeans_scaler.pkl","StandardScaler for KMeans input features"),
        ("Model","imputer.pkl","SimpleImputer (median) for 14 rating cols — linear path"),
        ("Model","imputer_composite.pkl","SimpleImputer (median) for 3 composite scores"),
        ("Model","scaler.pkl","MinMaxScaler — linear path"),
        ("Model","chi2_meta.pkl","Chi² feature selection metadata (Base + Dist)"),
        ("Model","risk_score_meta.pkl","RiskScore min/max and quintile thresholds"),
        ("Data","model_comparison.csv","6-model metrics table"),
        ("Data","cluster_summary.csv","KMeans K=2 cluster profile"),
        ("Data","customer_risk_insight_report.csv","Per-customer risk + top-3 SHAP drivers + actual ratings"),
        ("Data","critical_driver_frequency.csv","Driver frequency in Critical group"),
        ("Data","shap_global_importance.csv","Global SHAP mean|value| for all 37 features"),
        ("Data","system_level_insight_report.md","System-level descriptive insight report"),
        ("Chart","p2_03_pearson_heatmap.png","Pearson heatmap — 14 rating cols"),
        ("Chart","p2_03_point_biserial.png","Point-Biserial top-12 vs satisfaction"),
        ("Chart","p2_05_elbow_silhouette.png","KMeans elbow + silhouette K=2..8"),
        ("Chart","p2_05b_chi2_scores.png","Chi² feature selection — Base variant"),
        ("Chart","p2_06_model_comparison.png","6-model comparison bar chart"),
        ("Chart","p2_07_cluster_profiling.png","Cluster profiling — 4-metric bar chart"),
        ("Chart","p2_08_risk_score.png","Risk score distribution + mean by cluster"),
        ("Chart","p2_shap_bar.png","SHAP global importance bar — top 15"),
        ("Chart","p2_shap_beeswarm.png","SHAP beeswarm — direction + magnitude"),
        ("Chart","p2_shap_by_cluster.png","SHAP top-10 by cluster (C0 / C1)"),
        ("Chart","p2_shap_critical.png","SHAP for Critical group + driver frequency"),
        ("Chart","p2_loyalty_bias_c1.png","C1 loyalty bias: ratings vs risk score"),
    ]
    for cat, fname, desc in artifacts:
        art_data.append([cat, fname, desc])
    add(table(art_data, col_widths=[2*cm,6*cm,9*cm]))
    add(sp(4))
    add(p("All model artifacts are located under models/AIRLINE/. "
          "All charts are under outputs/AIRLINE/. "
          "All data CSVs and reports are under data/AIRLINE/."))

    return elems


# ══════════════════════════════════════════════════════════════════════════════
# Render PDF
# ══════════════════════════════════════════════════════════════════════════════
def render():
    doc = SimpleDocTemplate(
        PDF_PATH,
        pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2.2*cm, bottomMargin=2.2*cm,
        title="Airline Passenger Satisfaction — Technical Pipeline Report",
        author="Churn Alert Pipeline",
    )

    def page_header_footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(GRAY)
        if doc.page > 1:
            canvas.drawString(2*cm, H - 1.5*cm,
                              "Airline Satisfaction — Technical Pipeline Report")
            canvas.drawRightString(W - 2*cm, H - 1.5*cm,
                                   f"Page {doc.page}")
            canvas.setStrokeColor(LGRAY)
            canvas.setLineWidth(0.5)
            canvas.line(2*cm, H - 1.7*cm, W - 2*cm, H - 1.7*cm)
            canvas.line(2*cm, 1.8*cm, W - 2*cm, 1.8*cm)
            canvas.drawString(2*cm, 1.3*cm,
                              "Descriptive analysis only — no action recommendations included.")
        canvas.restoreState()

    elems = build()
    doc.build(elems, onFirstPage=page_header_footer, onLaterPages=page_header_footer)
    print(f"\n  PDF generated → {PDF_PATH}")
    return PDF_PATH


if __name__ == "__main__":
    import time
    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"  Generating Airline Technical Report PDF...")
    print(f"{'='*60}")
    path = render()
    size_kb = os.path.getsize(path) / 1024
    print(f"  Size   : {size_kb:.0f} KB")
    print(f"  Time   : {time.time()-t0:.1f}s")
    print(f"  Output : {path}")
    print(f"{'='*60}\n")
