"""
Churn Alert — Streamlit Dashboard
7-page interactive app:
  1. Overview
  2. EDA
  3. RFM Segmentation
  4. Model Results
  5. Risk Dashboard
  6. Survival Analysis
  7. Customer Lookup
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
import warnings
from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Alert Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE   = os.path.dirname(os.path.abspath(__file__))
DATA   = os.path.join(BASE, "..", "data")
MODEL  = os.path.join(BASE, "..", "models")
OUTPUT = os.path.join(BASE, "..", "outputs")

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0f172a 0%, #1e293b 100%);
    border-right: 1px solid #334155;
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #f1f5f9 !important; }

/* Main header */
.main-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0f172a 100%);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    border: 1px solid #1e3a8a;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}
.main-header h1 {
    font-size: 2.2rem;
    font-weight: 800;
    color: #f8fafc;
    margin: 0;
    letter-spacing: -0.5px;
}
.main-header p {
    color: #94a3b8;
    margin: 0.5rem 0 0;
    font-size: 1rem;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    text-align: center;
    box-shadow: 0 4px 16px rgba(0,0,0,0.2);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.3);
}
.metric-card .value {
    font-size: 2rem;
    font-weight: 800;
    color: #38bdf8;
    display: block;
}
.metric-card .label {
    font-size: 0.8rem;
    color: #94a3b8;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 0.25rem;
    display: block;
}
.metric-card .delta {
    font-size: 0.75rem;
    color: #4ade80;
    margin-top: 0.25rem;
    display: block;
}

/* Section headers */
.section-header {
    font-size: 1.4rem;
    font-weight: 700;
    color: #1e293b;
    border-left: 4px solid #3b82f6;
    padding-left: 0.75rem;
    margin: 1.5rem 0 1rem;
}

/* Risk badges */
.badge-high   { background: #fee2e2; color: #b91c1c; padding: 3px 10px; border-radius: 20px; font-weight: 600; font-size: 0.8rem; }
.badge-medium { background: #fef3c7; color: #b45309; padding: 3px 10px; border-radius: 20px; font-weight: 600; font-size: 0.8rem; }
.badge-low    { background: #d1fae5; color: #065f46; padding: 3px 10px; border-radius: 20px; font-weight: 600; font-size: 0.8rem; }

/* Insight box */
.insight-box {
    background: linear-gradient(135deg, #eff6ff, #e0f2fe);
    border: 1px solid #93c5fd;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    margin: 0.75rem 0;
    color: #1e3a8a;
    font-size: 0.9rem;
}

/* Footer */
.footer {
    text-align: center;
    color: #94a3b8;
    font-size: 0.8rem;
    padding: 1.5rem 0 0.5rem;
    border-top: 1px solid #e2e8f0;
    margin-top: 2rem;
}

/* Streamlit overrides */
div[data-testid="stMetricValue"] { font-size: 1.8rem !important; font-weight: 700 !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# Data Loaders (cached)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_raw():
    p = os.path.join(DATA, "ecommerce_churn_rfm.csv")
    if not os.path.exists(p):
        p = os.path.join(DATA, "ecommerce_churn_clean.csv")
    return pd.read_csv(p, dtype={"CityTier": str, "Complain": str})

@st.cache_data
def load_predictions():
    return pd.read_csv(os.path.join(DATA, "predictions.csv"))

@st.cache_data
def load_risk_table():
    # Step 8 now saves targeted_action_list.csv (replaces old risk_table.csv)
    action_path = os.path.join(DATA, "targeted_action_list.csv")
    risk_path   = os.path.join(DATA, "risk_table.csv")
    return pd.read_csv(action_path if os.path.exists(action_path) else risk_path)

@st.cache_data
def load_cluster_summary():
    return pd.read_csv(os.path.join(DATA, "cluster_summary.csv"))

@st.cache_data
def load_persona_map():
    """Build cluster_id → persona metadata dict from cluster_summary.csv."""
    cs_path = os.path.join(DATA, "cluster_summary.csv")
    if not os.path.exists(cs_path):
        return {}
    cs = pd.read_csv(cs_path)
    result = {}
    for _, row in cs.iterrows():
        cid = int(row["cluster"])
        result[cid] = {
            "persona":   row.get("persona",       f"Cluster {cid}"),
            "emoji":     row.get("persona_emoji",  "⚪"),
            "strategy":  row.get("strategy",       ""),
            "churn_pct": float(row.get("churn_rate_pct", 0)),
        }
    return result

@st.cache_resource
def load_kmeans():
    km_path = os.path.join(MODEL, "kmeans_model.pkl")
    if not os.path.exists(km_path):
        return None
    return joblib.load(km_path)

@st.cache_data
def load_xtest_raw_for_scaler():
    """Load X_test_raw to re-fit the MinMaxScaler used during clustering."""
    return pd.read_csv(os.path.join(DATA, "X_test_raw.csv"))

@st.cache_resource
def load_model():
    # xgb_inner_pipeline.pkl = ImbPipeline(SMOTE, XGBClassifier) — gives correct probabilities
    # xgb_model.pkl = CalibratedClassifierCV — isotonic calibration produced degenerate 0/1 outputs
    inner_path = os.path.join(MODEL, "xgb_inner_pipeline.pkl")
    model_path = os.path.join(MODEL, "xgb_model.pkl")
    return joblib.load(inner_path if os.path.exists(inner_path) else model_path)

@st.cache_resource
def load_metadata():
    return joblib.load(os.path.join(MODEL, "model_metadata.pkl"))

@st.cache_resource
def load_transformers():
    enc    = joblib.load(os.path.join(MODEL, "encoder.pkl"))
    scaler = joblib.load(os.path.join(MODEL, "scaler.pkl"))
    imp    = joblib.load(os.path.join(MODEL, "imputer.pkl"))
    # Derive cat/num cols from the encoder and the saved X_train header
    x_train_path = os.path.join(DATA, "X_train.csv")
    if os.path.exists(x_train_path):
        sample = pd.read_csv(x_train_path, nrows=1)
        ohe_names = enc.get_feature_names_out().tolist()
        all_cols  = sample.columns.tolist()
        num_cols  = [c for c in all_cols if c not in ohe_names]
        cat_cols  = enc.feature_names_in_.tolist()
    else:
        ohe_names = enc.get_feature_names_out().tolist()
        cat_cols  = enc.feature_names_in_.tolist()
        num_cols  = []
    return enc, scaler, imp, cat_cols, num_cols, ohe_names

def data_ready():
    return os.path.exists(os.path.join(DATA, "predictions.csv"))

def model_ready():
    return os.path.exists(os.path.join(MODEL, "xgb_model.pkl"))

# ══════════════════════════════════════════════════════════════════════════════
# Sidebar Navigation
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("##  Churn Alert")
    st.markdown("*E-commerce Customer Churn Prediction*")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["📊 Overview",
         "🔍 EDA",
         "💎 RFM Segmentation",
         "🤖 Model Results",
         "⚠️ Risk Dashboard",
         "📈 Survival Analysis",
         "🔎 Customer Lookup"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("**Dataset:** E-commerce Churn (Kaggle)")
    st.markdown("**Model:** XGBoost + Optuna + SHAP")
    st.markdown("**Version:** 1.0.0")

# ══════════════════════════════════════════════════════════════════════════════
# Page 1: Overview
# ══════════════════════════════════════════════════════════════════════════════


def assign_cluster_to_customer(input_raw_df: pd.DataFrame) -> int:
    """
    Assigns a K-Means cluster to a new customer.
    Re-fits MinMaxScaler on X_test_raw.csv (same columns as training) then predicts.
    Returns cluster ID (int) or -1 if K-Means model unavailable.
    """
    km = load_kmeans()
    if km is None:
        return -1
    try:
        X_ref = load_xtest_raw_for_scaler()
        num_cols = X_ref.select_dtypes(include=[np.number]).columns.tolist()
        scaler_km = MinMaxScaler()
        scaler_km.fit(X_ref[num_cols].fillna(X_ref[num_cols].median()))
        input_num = input_raw_df.copy()
        for col in num_cols:
            if col not in input_num.columns:
                input_num[col] = 0
        input_num = input_num[num_cols].fillna(0)
        X_scaled = scaler_km.transform(input_num)
        return int(km.predict(X_scaled)[0])
    except Exception:
        return -1


if page == "📊 Overview":
    st.markdown("""
    <div class="main-header">
        <h1>🚨 Churn Alert Dashboard</h1>
        <p>End-to-end e-commerce customer churn prediction & retention intelligence platform</p>
    </div>
    """, unsafe_allow_html=True)

    if not os.path.exists(os.path.join(DATA, "ecommerce_churn_clean.csv")):
        st.error("⚠️ Dataset not found. Please follow the setup instructions below.")
        st.info("""
        **Setup Instructions:**
        1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction)
        2. Place the CSV file in the `data/` folder as `ecommerce_churn.csv`
        3. Run the pipeline: `python src/10_pipeline.py`
        4. Refresh this page
        """)
        st.stop()

    df = load_raw()
    df["Churn"] = df["Churn"].astype(int)

    # KPI Cards
    total = len(df)
    churned = df["Churn"].sum()
    churn_rate = churned / total * 100
    avg_tenure = df["Tenure"].dropna().mean()
    avg_cashback = df["CashbackAmount"].dropna().mean()

    cols = st.columns(5)
    kpis = [
        ("Total Customers", f"{total:,}", "Full dataset"),
        ("Churned", f"{churned:,}", f"{churn_rate:.1f}% rate"),
        ("Retained", f"{total-churned:,}", f"{100-churn_rate:.1f}% rate"),
        ("Avg Tenure", f"{avg_tenure:.1f}m", "months on platform"),
        ("Avg Cashback", f"${avg_cashback:.0f}", "per transaction"),
    ]
    for col, (label, value, delta) in zip(cols, kpis):
        col.markdown(f"""
        <div class="metric-card">
            <span class="value">{value}</span>
            <span class="label">{label}</span>
            <span class="delta">{delta}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown('<p class="section-header">Churn Distribution</p>', unsafe_allow_html=True)
        churn_counts = df["Churn"].value_counts().reset_index()
        churn_counts.columns = ["Churn", "Count"]
        churn_counts["Label"] = churn_counts["Churn"].map({0: "Retained", 1: "Churned"})
        fig = px.pie(churn_counts, values="Count", names="Label",
                     color="Label",
                     color_discrete_map={"Retained": "#22c55e", "Churned": "#ef4444"},
                     hole=0.45)
        fig.update_traces(textinfo="percent+label", textfont_size=13)
        fig.update_layout(showlegend=True, height=360,
                          margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<p class="section-header">Churn Rate by City Tier</p>', unsafe_allow_html=True)
        tier_churn = df.groupby("CityTier")["Churn"].mean().reset_index()
        tier_churn.columns = ["CityTier", "ChurnRate"]
        tier_churn["ChurnRate"] *= 100
        fig2 = px.bar(tier_churn, x="CityTier", y="ChurnRate",
                      color="ChurnRate",
                      color_continuous_scale=["#22c55e", "#f97316", "#ef4444"],
                      text=tier_churn["ChurnRate"].apply(lambda x: f"{x:.1f}%"),
                      labels={"ChurnRate": "Churn Rate (%)", "CityTier": "City Tier"})
        fig2.update_traces(textposition="outside")
        fig2.update_layout(height=360, coloraxis_showscale=False,
                           margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig2, use_container_width=True)

    # Pipeline overview
    st.markdown('<p class="section-header">Project Pipeline</p>', unsafe_allow_html=True)
    steps = [
        ("1️⃣", "Data Cleaning", "Type fixes & typo correction"),
        ("2️⃣", "Missing Analysis", "MCAR/MAR/MNAR classification"),
        ("3️⃣", "Deep EDA", "Correlations & customer profiling"),
        ("4️⃣", "RFM Segmentation", "SQLite NTILE scoring"),
        ("5️⃣", "Preprocessing", "OHE + KNN Imputation + SMOTE"),
        ("6️⃣", "XGBoost + Optuna", "ROC-AUC 98.88% | F1 92.23%"),
        ("7️⃣", "K-Means Clustering", "Elbow + Silhouette selection"),
        ("8️⃣", "Risk Tiers", "High / Medium / Low tiers"),
        ("9️⃣", "Survival Analysis", "Kaplan-Meier + Cox PH"),
        ("🔟", "Streamlit App", "This dashboard!"),
    ]
    cols_steps = st.columns(5)
    for i, (icon, title, desc) in enumerate(steps):
        with cols_steps[i % 5]:
            st.markdown(f"""
            <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;
                        padding:0.75rem;margin:0.25rem 0;text-align:center;">
                <div style="font-size:1.5rem">{icon}</div>
                <div style="font-weight:700;font-size:0.85rem;color:#1e293b;margin-top:0.25rem">{title}</div>
                <div style="font-size:0.72rem;color:#64748b;margin-top:0.2rem">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="footer">Churn Alert v1.0 | Built with XGBoost, Optuna, SHAP, lifelines & Streamlit</div>',
                unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# Page 2: EDA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 EDA":
    st.markdown('<div class="main-header"><h1>🔍 Exploratory Data Analysis</h1><p>Deep dive into customer behaviour, correlations, and churn drivers</p></div>', unsafe_allow_html=True)

    if not os.path.exists(os.path.join(DATA, "ecommerce_churn_clean.csv")):
        st.warning("Run the pipeline first to generate analysis data.")
        st.stop()

    df = load_raw()
    df["Churn"] = df["Churn"].astype(int)

    tab1, tab2, tab3, tab4 = st.tabs(["📉 Tenure & Satisfaction", "🗺️ Correlations", "👤 Customer Profile", "📦 Missing Data"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Churn Rate by Tenure Group")
            df2 = df.copy()
            df2["Tenure_Bin"] = pd.cut(df2["Tenure"],
                bins=[-1, 3, 6, 12, 24, 100],
                labels=["0-3m", "3-6m", "6-12m", "12-24m", "24m+"])
            tenure_churn = df2.groupby("Tenure_Bin", observed=True)["Churn"].mean().reset_index()
            tenure_churn.columns = ["Tenure", "ChurnRate"]
            tenure_churn["ChurnRate"] *= 100
            fig = px.bar(tenure_churn, x="Tenure", y="ChurnRate",
                         color="ChurnRate", color_continuous_scale=["#22c55e", "#f97316", "#ef4444"],
                         text=tenure_churn["ChurnRate"].apply(lambda x: f"{x:.1f}%"))
            fig.add_hline(y=df["Churn"].mean()*100, line_dash="dash",
                          line_color="gray", annotation_text="Overall avg")
            fig.update_traces(textposition="outside")
            fig.update_layout(coloraxis_showscale=False, height=380)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('<div class="insight-box">🏔️ <b>"Death Valley"</b>: Customers in the first 0–3 months have the highest churn rate. If they survive past 12 months, churn drops dramatically.</div>', unsafe_allow_html=True)

        with col2:
            st.markdown("#### Churn Rate by Satisfaction Score")
            sat_churn = df.groupby("SatisfactionScore")["Churn"].mean().reset_index()
            sat_churn.columns = ["Score", "ChurnRate"]
            sat_churn["ChurnRate"] *= 100
            fig2 = px.bar(sat_churn, x="Score", y="ChurnRate",
                          color="ChurnRate", color_continuous_scale=["#22c55e", "#f97316", "#ef4444"],
                          text=sat_churn["ChurnRate"].apply(lambda x: f"{x:.1f}%"))
            fig2.add_hline(y=df["Churn"].mean()*100, line_dash="dash", line_color="gray")
            fig2.update_traces(textposition="outside")
            fig2.update_layout(coloraxis_showscale=False, height=380)
            st.plotly_chart(fig2, use_container_width=True)
            st.markdown('<div class="insight-box">⭐ <b>5-star Paradox</b>: Customers who rate 5 stars churn the most (23.8%). Likely one-time buyers who fake high ratings for vouchers.</div>', unsafe_allow_html=True)

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Pearson Correlation (Numeric Features)")
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            corr = df[num_cols].corr()
            fig = px.imshow(corr, color_continuous_scale="RdYlGn", zmin=-1, zmax=1,
                            aspect="auto", text_auto=".2f")
            fig.update_traces(textfont_size=9)
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### Point-Biserial Correlation vs Churn")
            from scipy import stats as scipy_stats
            pb_results = []
            for col in [c for c in num_cols if c not in ["Churn", "CustomerID"]]:
                valid = df[[col, "Churn"]].dropna()
                r, _ = scipy_stats.pointbiserialr(valid[col], valid["Churn"])
                pb_results.append({"Feature": col, "Correlation": r})
            pb_df = pd.DataFrame(pb_results).sort_values("Correlation")
            colors = ["#ef4444" if c > 0 else "#22c55e" for c in pb_df["Correlation"]]
            fig2 = go.Figure(go.Bar(
                x=pb_df["Correlation"], y=pb_df["Feature"],
                orientation="h",
                marker_color=colors,
                text=[f"{v:.3f}" for v in pb_df["Correlation"]],
                textposition="outside"
            ))
            fig2.add_vline(x=0, line_color="black", line_width=1)
            fig2.update_layout(height=500, xaxis_title="Correlation Coefficient",
                               title="Red = increases churn | Green = decreases churn")
            st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        cat_feats = [c for c in ["Gender", "CityTier", "MaritalStatus", "Complain"] if c in df.columns]
        cols = st.columns(len(cat_feats))
        for col_w, feat in zip(cols, cat_feats):
            with col_w:
                st.markdown(f"#### {feat}")
                churn_by = df.groupby(feat)["Churn"].mean().reset_index()
                churn_by.columns = [feat, "ChurnRate"]
                churn_by["ChurnRate"] *= 100
                churn_by = churn_by.sort_values("ChurnRate", ascending=False)
                fig = px.bar(churn_by, x=feat, y="ChurnRate",
                             color="ChurnRate",
                             color_continuous_scale=["#22c55e", "#f97316", "#ef4444"],
                             text=churn_by["ChurnRate"].apply(lambda x: f"{x:.1f}%"))
                fig.update_traces(textposition="outside")
                fig.add_hline(y=df["Churn"].mean()*100, line_dash="dash", line_color="gray")
                fig.update_layout(coloraxis_showscale=False, height=320,
                                  margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="insight-box">🔴 <b>High-risk profile</b>: Male, Single, City Tier 3, with a complaint history. Complainers churn 3× more than non-complainers.</div>', unsafe_allow_html=True)

    with tab4:
        st.markdown("#### Missing Data Overview")
        missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
        missing_pct = missing_pct[missing_pct > 0].reset_index()
        missing_pct.columns = ["Column", "Missing (%)"]
        fig = px.bar(missing_pct, x="Missing (%)", y="Column", orientation="h",
                     color="Missing (%)", color_continuous_scale="Reds",
                     text=missing_pct["Missing (%)"].apply(lambda x: f"{x:.1f}%"))
        fig.update_traces(textposition="outside")
        fig.update_layout(height=400, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="insight-box">⚠️ <b>Informative Missingness</b>: Tenure and WarehouseToHome missing rows have 30–34% churn rates — these are MAR/MNAR signals, not random gaps. KNN Imputation preserves this signal.</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# Page 3: RFM Segmentation
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💎 RFM Segmentation":
    st.markdown('<div class="main-header"><h1>💎 RFM Segmentation</h1><p>Customer scoring via Recency, Frequency & Monetary value using SQLite NTILE(5)</p></div>', unsafe_allow_html=True)

    df = load_raw()
    df["Churn"] = df["Churn"].astype(int)

    if "rfm_segment" not in df.columns:
        st.warning("Run `python src/04_rfm_segmentation.py` to generate RFM features.")
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Segment Distribution")
        seg_counts = df["rfm_segment"].value_counts().reset_index()
        seg_counts.columns = ["Segment", "Count"]
        colors_map = {
            "Champions": "#22c55e", "Loyal": "#3b82f6",
            "Recent Customers": "#f97316", "At Risk": "#ef4444", "Hibernating": "#a855f7"
        }
        colors = [colors_map.get(s, "#94a3b8") for s in seg_counts["Segment"]]
        fig = px.pie(seg_counts, values="Count", names="Segment",
                     color_discrete_sequence=colors, hole=0.4)
        fig.update_traces(textinfo="percent+label")
        fig.update_layout(height=380, margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Churn Rate by RFM Segment")
        seg_churn = df.groupby("rfm_segment").agg(
            count=("Churn", "count"), churn_rate=("Churn", "mean")).reset_index()
        seg_churn["churn_pct"] = seg_churn["churn_rate"] * 100
        seg_churn = seg_churn.sort_values("churn_pct", ascending=False)
        bar_colors = [colors_map.get(s, "#94a3b8") for s in seg_churn["rfm_segment"]]
        fig2 = go.Figure(go.Bar(
            x=seg_churn["rfm_segment"], y=seg_churn["churn_pct"],
            marker_color=bar_colors,
            text=[f"{v:.1f}%<br>(n={c:,})" for v, c in zip(seg_churn["churn_pct"], seg_churn["count"])],
            textposition="outside"
        ))
        fig2.add_hline(y=df["Churn"].mean()*100, line_dash="dash",
                       line_color="gray", annotation_text="Overall avg")
        fig2.update_layout(height=380, yaxis_title="Churn Rate (%)")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="insight-box">💥 <b>Shocking Insight</b>: "Recent Customers" have the <b>highest churn rate</b> — not "At Risk" as expected. This means the business is leaking new customers as fast as it acquires them. Fix onboarding immediately!</div>', unsafe_allow_html=True)

    st.markdown("#### RFM Total Score vs Churn Rate")
    rfm_churn = df.groupby("rfm_total")["Churn"].mean().reset_index()
    rfm_churn.columns = ["RFM Total", "Churn Rate"]
    rfm_churn["Churn Rate"] *= 100
    fig3 = px.line(rfm_churn, x="RFM Total", y="Churn Rate", markers=True,
                   color_discrete_sequence=["#3b82f6"])
    fig3.update_traces(fill="tozeroy", fillcolor="rgba(59,130,246,0.1)", line_width=2.5)
    fig3.update_layout(height=350, xaxis_title="RFM Total Score (3=lowest, 15=highest)",
                       yaxis_title="Churn Rate (%)")
    st.plotly_chart(fig3, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# Page 4: Model Results
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Results":
    st.markdown('<div class="main-header"><h1>🤖 Model Results</h1><p>XGBoost + Optuna (40 trials) + CalibratedClassifier + SHAP explainability</p></div>', unsafe_allow_html=True)

    if not model_ready():
        st.warning("Run `python src/06_modelling.py` to train the model.")
        st.stop()

    meta = load_metadata()
    metrics = meta["metrics"]

    # Metric cards
    st.markdown("#### Performance Metrics")
    metric_cols = st.columns(6)
    metric_names = ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "PR_AUC"]
    colors_m = ["#3b82f6", "#6366f1", "#22c55e", "#f97316", "#ef4444", "#a855f7"]
    for col, name, color in zip(metric_cols, metric_names, colors_m):
        val = metrics.get(name, 0)
        col.markdown(f"""
        <div class="metric-card" style="border-color:{color}33">
            <span class="value" style="color:{color}">{val:.1%}</span>
            <span class="label">{name}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        # ROC Curve (from saved image)
        roc_img = os.path.join(OUTPUT, "06_roc_curve.png")
        if os.path.exists(roc_img):
            st.markdown("#### ROC Curve")
            st.image(roc_img, use_column_width=True)

    with col2:
        # Confusion Matrix
        cm_img = os.path.join(OUTPUT, "06_confusion_matrix.png")
        if os.path.exists(cm_img):
            st.markdown("#### Confusion Matrix")
            st.image(cm_img, use_column_width=True)

    # SHAP
    st.markdown("#### SHAP Feature Importance")
    shap_df = meta.get("shap_df", None)
    if shap_df is not None:
        shap_plot = shap_df.sort_values("MeanSHAP")
        fig = go.Figure(go.Bar(
            x=shap_plot["MeanSHAP"], y=shap_plot["Feature"],
            orientation="h", marker_color="#3b82f6",
            text=[f"{v:.4f}" for v in shap_plot["MeanSHAP"]],
            textposition="outside"
        ))
        fig.update_layout(height=500, xaxis_title="Mean |SHAP| Value",
                          title="Top 15 Most Important Features")
        st.plotly_chart(fig, use_container_width=True)

    shap_img = os.path.join(OUTPUT, "06_shap_summary.png")
    if os.path.exists(shap_img):
        st.markdown("#### SHAP Beeswarm Summary")
        st.image(shap_img, use_column_width=True)

    st.markdown('<div class="insight-box">🏆 <b>Top Features</b>: device_per_tenure (#1), Complain (#2), Tenure (#3) — engineered features outperform raw features. High cashback_per_order is a strong retention anchor.</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# Page 5: Risk Dashboard
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚠️ Risk Dashboard":
    st.markdown('<div class="main-header"><h1>⚠️ Risk Dashboard</h1><p>Actionable churn risk tiers for the business operations team</p></div>', unsafe_allow_html=True)

    action_file = os.path.join(DATA, "targeted_action_list.csv")
    old_file    = os.path.join(DATA, "risk_table.csv")
    if not os.path.exists(action_file) and not os.path.exists(old_file):
        st.warning("Run the pipeline (Step 8) to generate risk data.")
        st.stop()

    risk_df = load_risk_table()
    cluster_summary = load_cluster_summary() if os.path.exists(os.path.join(DATA, "cluster_summary.csv")) else None

    tier_colors = {"High Risk": "#fee2e2", "Medium Risk": "#fef3c7", "Low Risk": "#d1fae5"}
    tier_text_colors = {"High Risk": "#b91c1c", "Medium Risk": "#b45309", "Low Risk": "#065f46"}
    tier_actions = {
        "High Risk":   "📞 Immediate call + 20% voucher",
        "Medium Risk": "📧 Personalised email offer",
        "Low Risk":    "⭐ Enrol in loyalty program"
    }

    # Summary KPIs
    tier_stats = risk_df.groupby("risk_tier").agg(
        count=("y_true", "count"),
        actual_churn=("y_true", "mean")
    ).reset_index()

    tier_order = ["High Risk", "Medium Risk", "Low Risk"]
    cols = st.columns(3)
    for col, tier in zip(cols, tier_order):
        row = tier_stats[tier_stats["risk_tier"] == tier]
        if len(row) == 0:
            continue
        count = int(row["count"].iloc[0])
        actual = row["actual_churn"].iloc[0] * 100
        bg = tier_colors[tier]
        tc = tier_text_colors[tier]
        col.markdown(f"""
        <div style="background:{bg};border-radius:12px;padding:1.25rem;
                    border:1px solid {tc}33;text-align:center;">
            <div style="font-size:2rem;font-weight:800;color:{tc}">{count}</div>
            <div style="font-weight:700;color:{tc};font-size:1rem">{tier}</div>
            <div style="color:{tc};font-size:0.85rem;margin-top:0.5rem">
                Actual churn: <b>{actual:.1f}%</b>
            </div>
            <div style="color:#374151;font-size:0.8rem;margin-top:0.35rem">
                {tier_actions[tier]}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### Risk Tier Distribution")
        fig = px.pie(tier_stats, values="count", names="risk_tier",
                     color="risk_tier",
                     color_discrete_map={"High Risk": "#ef4444", "Medium Risk": "#f97316", "Low Risk": "#22c55e"},
                     hole=0.45)
        fig.update_traces(textinfo="percent+label")
        fig.update_layout(height=360)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Probability Distribution by Tier")
        fig2 = go.Figure()
        colors_tier = {"High Risk": "#ef4444", "Medium Risk": "#f97316", "Low Risk": "#22c55e"}
        for tier in tier_order:
            subset = risk_df[risk_df["risk_tier"] == tier]["churn_probability"]
            fig2.add_trace(go.Histogram(x=subset, name=tier, nbinsx=25,
                                        marker_color=colors_tier[tier], opacity=0.7))
        fig2.add_vline(x=0.70, line_dash="dash", line_color="#ef4444",
                       annotation_text="High threshold")
        fig2.add_vline(x=0.40, line_dash="dash", line_color="#f97316",
                       annotation_text="Med threshold")
        fig2.update_layout(barmode="overlay", height=360, xaxis_title="Churn Probability")
        st.plotly_chart(fig2, use_container_width=True)

    # Cluster profiling scatter — with persona labels from cluster_summary.csv
    if cluster_summary is not None:
        st.markdown("#### Customer Cluster Analysis (K-Means)")
        _pm = load_persona_map()
        if _pm:
            cluster_summary["persona_label"] = cluster_summary["cluster"].apply(
                lambda c: f"{_pm.get(int(c), {}).get('emoji', '⚪')} {_pm.get(int(c), {}).get('persona', f'Cluster {c}')}"
            )
        x_col = next((c for c in ["Tenure_mean", "avg_tenure"] if c in cluster_summary.columns), None)
        if x_col:
            _hover_cols = ["count", "avg_churn_prob"] + (["strategy"] if "strategy" in cluster_summary.columns else [])
            cluster_fig = px.scatter(
                cluster_summary,
                x=x_col, y="churn_rate_pct",
                size="count", color="churn_rate_pct",
                text="persona_label" if "persona_label" in cluster_summary.columns else None,
                color_continuous_scale=["#22c55e", "#f97316", "#ef4444"],
                hover_data=_hover_cols,
                labels={x_col: "Avg Tenure (months)", "churn_rate_pct": "Churn Rate (%)"},
            )
            if "persona_label" in cluster_summary.columns:
                cluster_fig.update_traces(textposition="top center")
            cluster_fig.update_layout(height=440)
            st.plotly_chart(cluster_fig, use_container_width=True)


    # Top at-risk customers table + recommended actions
    st.markdown("#### 🔴 Top High-Risk Customers & Recommended Actions")
    action_src = os.path.join(DATA, "targeted_action_list.csv")
    fallback   = os.path.join(DATA, "high_risk_customers.csv")
    hr_source  = action_src if os.path.exists(action_src) else (fallback if os.path.exists(fallback) else None)
    if hr_source:
        hr = pd.read_csv(hr_source)
        if "risk_tier" in hr.columns:
            hr = hr[hr["risk_tier"] == "High Risk"]
        hr = hr.sort_values("churn_probability", ascending=False).head(25)
        display_cols = [c for c in ["Tenure", "Complain", "churn_probability",
                                     "cluster", "recommended_action", "risk_tier", "y_true"]
                        if c in hr.columns]
        hr_display = hr[display_cols].copy()
        hr_display["churn_probability"] = hr_display["churn_probability"].apply(lambda x: f"{x:.1%}")
        st.dataframe(hr_display, use_container_width=True, height=400)

# ══════════════════════════════════════════════════════════════════════════════
# Page 6: Survival Analysis
# ═# Page 6: Survival Analysis
# ═# Page 6: Survival Analysis
# ═# Page 6: Survival Analysis
# ═# Page 6: Survival Analysis
# ═# Page 6: Survival Analysis
# ═# Page 6: Survival Analysis
# ═# Page 6: Survival Analysis
# ═# Page 6: Survival Analysis
# ═# Page 6: Survival Analysis
# ═# Page 6: Survival Analysis
# ═# Page 6: Survival Analysis
# ═# Page 6: Survival Analysis
# ═# Page 6: Survival Analysis
# ═# Page 6: Survival Analysis
# ═# Page 6: Survival Analysis
# ═# Page 6: Survival Analysis
# ═# Page 6: Survival Analysis
# ═# Page 6: Survival Analysis
# ═# Page 6: Survival Analysis
# ═# Page 6: Survival Analysis
# ═# Page 6: Survival Analysis
# ═# Page 6: Survival Analysis
# ═# Page 6: Survival Analysis
# ═# Page 6: Survival Analysis
# ═# Page 6: Survival Analysis
# ═# Page 6: Survival Analysis
# ═# Page 6: Survival Analysis
# ═# Page 6: Survival Analysis
# ═# Page 6: Survival Analysis
# ═# Page 6: Survival Analysis
# ═# Page 6: Survival Analysis
# ═# Page 6: Survival Analysis
# ═# Page 6: Survival Analysis
# ═# Page 6: Survival Analysis
# ═# Page 6: Survival Analysis
# ═# Page 6: Survival Analysis
# ═# Page 6: Survival Analysis
# ═# Page 6: Survival Analysis
# ═
elif page == "📈 Survival Analysis":
    st.markdown('<div class="main-header"><h1>📈 Survival Analysis</h1><p>Persona-Based Kaplan-Meier — "When does each customer type churn?"</p></div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📊 KM Curves by Persona", "🎯 Action Matrix"])

    with tab1:
        km_path = next((p for p in [
            os.path.join(OUTPUT, "08_km_by_cluster.png"),
            os.path.join(OUTPUT, "09_kaplan_meier.png"),
        ] if os.path.exists(p)), None)

        if km_path:
            st.image(km_path, use_column_width=True)
        else:
            st.warning("Run the pipeline (Step 8) to generate survival plots.")

        st.markdown('<div class="insight-box">📉 <b>Key insight</b>: KM curves grouped by K-Means persona. Each cluster has a distinct churn trajectory. Use the 70% threshold line to time CRM interventions per persona.</div>', unsafe_allow_html=True)

        st.markdown("""#### Cluster Persona Guide
| Cluster | Name | Drop-off Speed | Recommended Action |
|---------|------|----------------|---------------------|
| 0 | Đô thị phàn nàn | Medium | Phone call to resolve complaints |
| 1 | VIP Tỉnh ổn định | Slowest | VIP perks + loyalty rewards |
| 2 | Đô thị trung thành | Moderate | Freeship + flash sale nudges |
| 3 | Săn Sale phàn nàn | **Fastest 🔴** | Immediate cashback voucher via SMS |
""")

    with tab2:
        _action_csv = os.path.join(DATA, "targeted_action_list.csv")
        if os.path.exists(_action_csv):
            _adf = pd.read_csv(_action_csv)

            st.markdown("#### Campaign Priority Summary (by Tier x Cluster)")
            if {"risk_tier", "cluster", "recommended_action"}.issubset(_adf.columns):
                _summary = (_adf
                            .groupby(["risk_tier", "cluster", "recommended_action"])
                            .size().reset_index(name="customers"))
                _sort_key = {"High Risk": 0, "Medium Risk": 1, "Low Risk": 2}
                _summary["_s"] = _summary["risk_tier"].map(_sort_key)
                _summary = _summary.sort_values(["_s", "customers"], ascending=[True, False]).drop("_s", axis=1)
                st.dataframe(_summary, use_container_width=True, height=380)

            st.markdown("#### High-Risk Customer Action List")
            _high = _adf[_adf["risk_tier"] == "High Risk"].sort_values("churn_probability", ascending=False)
            _cols = [c for c in ["Tenure", "Complain", "churn_probability",
                                  "cluster", "risk_tier", "recommended_action"]
                     if c in _high.columns]
            _h = _high[_cols].copy()
            _h["churn_probability"] = _h["churn_probability"].apply(lambda x: f"{x:.1%}")
            st.dataframe(_h, use_container_width=True, height=380)
        else:
            st.warning("Run the pipeline (Step 8) to generate the action matrix.")

        st.markdown('<div class="insight-box">🎯 <b>Risk Matrix Strategy</b>: Combines XGBoost probability (how likely to churn) with K-Means cluster (who they are) for targeted CRM — from phone calls for urban complainers to cashback vouchers for deal-hunters.</div>', unsafe_allow_html=True)

# Page 7: Customer Lookup
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔎 Customer Lookup":
    st.markdown('<div class="main-header"><h1>🔎 Customer Lookup</h1><p>Input any customer\'s features and get real-time churn probability + risk tier</p></div>', unsafe_allow_html=True)

    if not model_ready():
        st.warning("Run the pipeline first to train the model.")
        st.stop()

    model    = load_model()
    meta     = load_metadata()
    enc, scaler, imp, cat_cols, num_cols_orig, ohe_names = load_transformers()

    st.markdown("#### Enter Customer Details")
    col1, col2, col3 = st.columns(3)

    with col1:
        tenure = st.slider("Tenure (months)", 0, 60, 6)
        city_tier = st.selectbox("City Tier", ["1", "2", "3"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])

    with col2:
        complain = st.selectbox("Has Complaint?", ["0", "1"])
        satisfaction = st.slider("Satisfaction Score", 1, 5, 3)
        n_devices = st.slider("Number of Devices", 1, 6, 2)
        n_address = st.slider("Number of Addresses", 1, 20, 2)

    with col3:
        hours_app = st.slider("Hours on App (monthly)", 0.0, 5.0, 2.0)
        order_count = st.slider("Order Count", 1, 16, 3)
        cashback = st.slider("Cashback Amount", 0.0, 325.0, 150.0)
        wh_dist = st.slider("Warehouse to Home (km)", 5, 127, 30)

    order_hike = st.slider("Order Amount Hike from Last Year (%)", 11, 26, 15)
    coupon_used = st.slider("Coupons Used", 0, 16, 2)
    day_last_order = st.slider("Days Since Last Order", 0, 46, 5)
    login_device = st.selectbox("Preferred Login Device", ["Mobile", "Computer"])
    payment_mode = st.selectbox("Preferred Payment Mode",
        ["Credit Card", "Debit Card", "UPI", "Cash on Delivery", "E wallet"])
    order_cat = st.selectbox("Preferred Order Category",
        ["Laptop & Accessory", "Mobile", "Fashion", "Grocery", "Others"])

    if st.button("🔍 Predict Churn Risk", type="primary", use_container_width=True):
        # Build input DataFrame
        input_data = pd.DataFrame([{
            "Tenure": tenure,
            "CityTier": city_tier,
            "Gender": gender,
            "MaritalStatus": marital,
            "Complain": complain,
            "SatisfactionScore": satisfaction,
            "NumberOfDeviceRegistered": n_devices,
            "NumberOfAddress": n_address,
            "HourSpendOnApp": hours_app,
            "OrderCount": order_count,
            "CashbackAmount": cashback,
            "WarehouseToHome": wh_dist,
            "OrderAmountHikeFromlastYear": order_hike,
            "CouponUsed": coupon_used,
            "DaySinceLastOrder": day_last_order,
            "PreferredLoginDevice": login_device,
            "PreferredPaymentMode": payment_mode,
            "PreferedOrderCat": order_cat,
        }])

        # Feature engineering
        input_data["device_per_tenure"]  = input_data["NumberOfDeviceRegistered"] / (input_data["Tenure"] + 1)
        input_data["cashback_per_order"] = input_data["CashbackAmount"] / (input_data["OrderCount"] + 1)
        input_data["inactivity_ratio"]   = input_data["DaySinceLastOrder"] / (input_data["Tenure"] + 1)

        # Add RFM placeholders (median rfm_total=7, neutral segment)
        if "rfm_total" not in input_data.columns:
            input_data["rfm_total"] = 7
        if "rfm_segment" not in input_data.columns:
            input_data["rfm_segment"] = "Loyal"

        # Get feature names from metadata
        feature_names = meta.get("feature_names", [])

        try:
            # ── Step 1: OHE encode categoricals ──────────────────────────────
            cat_encoded = pd.DataFrame(
                enc.transform(input_data[cat_cols]),
                columns=ohe_names,
                index=input_data.index
            ).reset_index(drop=True)

            # ── Step 2: Numeric part ──────────────────────────────────────────
            num_part = input_data[num_cols_orig].reset_index(drop=True)

            # ── Step 3: Combine → reorder to EXACT feature_names order ────────
            #    (scaler & imputer were fitted on all 35 cols together)
            X_input = pd.concat([num_part, cat_encoded], axis=1)
            for c in feature_names:
                if c not in X_input.columns:
                    X_input[c] = 0
            X_input = X_input[feature_names]   # 35 columns, correct order

            # ── Step 4: Scale ALL 35 cols (mirrors preprocessing) ─────────────
            X_scaled  = scaler.transform(X_input)
            X_imputed = imp.transform(X_scaled)
            X_final   = pd.DataFrame(X_imputed, columns=feature_names)

            prob = model.predict_proba(X_final)[0, 1]
            threshold = meta["threshold"]
            prediction = int(prob >= threshold)

            # ── Cluster assignment ─────────────────────────────────────────────
            cluster_id = assign_cluster_to_customer(input_data)

            persona_map_local = load_persona_map()
            cluster_meta = persona_map_local.get(cluster_id, {
                "persona": f"Cluster {cluster_id}" if cluster_id >= 0 else "Unknown",
                "emoji": "⚪",
                "strategy": "",
                "churn_pct": 0.0,
            })

            # Cluster-specific actions from cluster_summary.csv
            action_high   = "🚨 Liên hệ người dùng ngưay"
            action_medium = "📧 Gửi email chăm sóc"
            try:
                _cs = load_cluster_summary()
                if cluster_id >= 0 and "cluster" in _cs.columns:
                    _row = _cs[_cs["cluster"] == cluster_id]
                    if len(_row) > 0:
                        _r = _row.iloc[0]
                        action_high   = _r.get("action_high",   action_high)
                        action_medium = _r.get("action_medium", action_medium)
            except Exception:
                pass

            # Risk tier + cluster-specific action
            if prob >= 0.70:
                tier = "High Risk"
                tier_color = "#ef4444"
                action = action_high
                emoji = "🔴"
            elif prob >= 0.40:
                tier = "Medium Risk"
                tier_color = "#f97316"
                action = action_medium
                emoji = "🟡"
            else:
                tier = "Low Risk"
                tier_color = "#22c55e"
                action = "✅ Duy trì chăm sóc tiêu chuẩn"
                emoji = "🟢"

            st.markdown("---")
            st.markdown("#### 📋 Prediction Result")
            rcol1, rcol2, rcol3, rcol4 = st.columns(4)

            # Card 1 — Churn Probability
            rcol1.markdown(f"""
            <div class="metric-card" style="border-color:{tier_color}33">
                <span class="value" style="color:{tier_color}">{prob:.1%}</span>
                <span class="label">Churn Probability</span>
            </div>
            """, unsafe_allow_html=True)

            # Card 2 — Cluster Persona
            _cid_display = f"Cluster {cluster_id}" if cluster_id >= 0 else "N/A"
            rcol2.markdown(f"""
            <div class="metric-card" style="border-color:#6366f133">
                <span class="value" style="font-size:1.3rem;color:#6366f1">
                    {cluster_meta['emoji']} {cluster_meta['persona']}
                </span>
                <span class="label">Customer Cluster ({_cid_display})</span>
                <span class="delta" style="color:#94a3b8">
                    Cluster avg churn: {cluster_meta['churn_pct']:.1f}%
                </span>
            </div>
            """, unsafe_allow_html=True)

            # Card 3 — Risk Tier
            rcol3.markdown(f"""
            <div class="metric-card" style="border-color:{tier_color}33">
                <span class="value" style="font-size:1.5rem;color:{tier_color}">{emoji} {tier}</span>
                <span class="label">Risk Classification</span>
            </div>
            """, unsafe_allow_html=True)

            # Card 4 — Recommended Action (cluster-specific)
            rcol4.markdown(f"""
            <div class="metric-card">
                <span class="value" style="font-size:1rem;color:#3b82f6">{action}</span>
                <span class="label">Recommended Action</span>
                <span class="delta" style="color:#94a3b8;font-size:0.72rem">{cluster_meta['strategy']}</span>
            </div>
            """, unsafe_allow_html=True)

            # Probability gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                number={"suffix": "%", "font": {"size": 36, "color": tier_color}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1},
                    "bar": {"color": tier_color},
                    "steps": [
                        {"range": [0, 40], "color": "#d1fae5"},
                        {"range": [40, 70], "color": "#fef3c7"},
                        {"range": [70, 100], "color": "#fee2e2"},
                    ],
                    "threshold": {
                        "line": {"color": "#1e293b", "width": 3},
                        "thickness": 0.75,
                        "value": prob * 100
                    }
                },
                title={"text": "Churn Probability Gauge", "font": {"size": 14}}
            ))
            fig.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            st.info("Make sure the full pipeline has been run successfully before using Customer Lookup.")

st.markdown('<div class="footer">🚨 Churn Alert | E-commerce Customer Retention Intelligence | Built with ❤️ using Python & Streamlit</div>', unsafe_allow_html=True)
