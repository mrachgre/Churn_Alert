"""
Churn Alert — Customer Risk Prediction Dashboard
Single-page Customer Lookup powered by XGBoost-Cluster model
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Alert | Customer Risk Prediction",
    page_icon="🚨",
    layout="wide",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE  = os.path.dirname(os.path.abspath(__file__))
DATA  = os.path.join(BASE, "..", "data")
MODEL = os.path.join(BASE, "..", "models")

# ── Premium CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Dark gradient background */
.stApp {
    background: linear-gradient(135deg, #0a0f1e 0%, #0f172a 50%, #0a1628 100%);
    min-height: 100vh;
}

/* Main header */
.main-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 60%, #0f172a 100%);
    padding: 2rem 2.5rem;
    border-radius: 20px;
    margin-bottom: 2rem;
    border: 1px solid rgba(59,130,246,0.3);
    box-shadow: 0 8px 40px rgba(59,130,246,0.15), 0 0 80px rgba(59,130,246,0.05);
}
.main-header h1 {
    font-size: 2.4rem; font-weight: 800;
    background: linear-gradient(90deg, #f8fafc, #93c5fd, #f8fafc);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0; letter-spacing: -0.5px;
}
.main-header p {
    color: #64748b; margin: 0.5rem 0 0; font-size: 0.95rem;
}

/* Glass input card */
.input-card {
    background: rgba(30, 41, 59, 0.6);
    border: 1px solid rgba(71, 85, 105, 0.5);
    border-radius: 16px;
    padding: 1.5rem;
    backdrop-filter: blur(10px);
    margin-bottom: 1rem;
}
.input-card h4 {
    color: #93c5fd; font-size: 0.9rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.08em;
    margin: 0 0 1rem; padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(71,85,105,0.3);
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, rgba(30,41,59,0.8), rgba(15,23,42,0.9));
    border: 1px solid rgba(71,85,105,0.4);
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 28px rgba(0,0,0,0.4);
}
.metric-card .value {
    font-size: 2rem; font-weight: 800; display: block;
}
.metric-card .label {
    font-size: 0.75rem; color: #64748b;
    font-weight: 500; text-transform: uppercase;
    letter-spacing: 0.06em; margin-top: 0.3rem; display: block;
}
.metric-card .sub {
    font-size: 0.8rem; color: #94a3b8; margin-top: 0.2rem; display: block;
}

/* Risk category badge */
.risk-badge {
    display: inline-flex; align-items: center; gap: 0.5rem;
    padding: 0.5rem 1.25rem; border-radius: 50px;
    font-size: 0.95rem; font-weight: 700; letter-spacing: 0.02em;
}

/* Cluster description card */
.cluster-card {
    border-radius: 16px;
    padding: 1.5rem;
    margin-top: 1rem;
}
.cluster-card h3 {
    font-size: 1.15rem; font-weight: 700; margin: 0 0 0.75rem;
}
.cluster-card ul {
    margin: 0; padding-left: 1.2rem; color: #cbd5e1;
    font-size: 0.9rem; line-height: 1.8;
}
.cluster-card .rec-box {
    margin-top: 1rem; padding: 0.75rem 1rem;
    background: rgba(0,0,0,0.2); border-radius: 8px;
    font-size: 0.88rem; color: #94a3b8;
    border-left: 3px solid;
}

/* Streamlit widget overrides */
[data-testid="stSlider"] label,
[data-testid="stSelectbox"] label { color: #94a3b8 !important; font-size: 0.82rem !important; }
div[data-testid="stMetricValue"] { font-size: 1.6rem !important; font-weight: 700 !important; }
.stButton > button {
    background: linear-gradient(135deg, #3b82f6, #2563eb);
    color: white; border: none; border-radius: 10px;
    font-size: 1rem; font-weight: 600; padding: 0.6rem 1.5rem;
    transition: all 0.2s ease;
    box-shadow: 0 4px 15px rgba(59,130,246,0.4);
}
.stButton > button:hover {
    background: linear-gradient(135deg, #60a5fa, #3b82f6);
    box-shadow: 0 6px 20px rgba(59,130,246,0.5);
    transform: translateY(-1px);
}
</style>
""", unsafe_allow_html=True)

# ── Cluster definitions ────────────────────────────────────────────────────────
CLUSTER_INFO = {
    0: {
        "name":           "Cluster 0",
        "risk_level":     "Relatively stable — Low churn risk",
        "emoji":          "🟢",
        "color":          "#22c55e",
        "bg":             "rgba(34,197,94,0.07)",
        "border":         "rgba(34,197,94,0.35)",
        "traits": [
            "Frequently uses discount codes",
            "Average number of complaints",
            "Has not placed an order in a long time (low recency)",
            "Uses devices relatively often; high order count",
            "Long tenure — purchases across all product categories",
        ],
        "recommendation": "Maintain standard care. Consider loyalty rewards and cross-sell offers.",
    },
    1: {
        "name":           "Cluster 1",
        "risk_level":     "Relatively stable — Low churn risk",
        "emoji":          "🔵",
        "color":          "#3b82f6",
        "bg":             "rgba(59,130,246,0.07)",
        "border":         "rgba(59,130,246,0.35)",
        "traits": [
            "Uses discount codes fairly often",
            "Last order placed slightly more recently than Cluster 0",
            "Uses devices less frequently; fewer orders than Cluster 0",
            "Average tenure; frequently purchases electronics",
        ],
        "recommendation": "Nurture with personalised electronics promotions and freeship incentives.",
    },
    2: {
        "name":           "Cluster 2",
        "risk_level":     "Needs monitoring",
        "emoji":          "🟡",
        "color":          "#f59e0b",
        "bg":             "rgba(245,158,11,0.07)",
        "border":         "rgba(245,158,11,0.35)",
        "traits": [
            "Rarely uses discount codes; few complaints",
            "Last-order recency similar to Cluster 1",
            "Places few orders overall",
            "Strong preference for mobile phones (~42% of purchases)",
        ],
        "recommendation": "Monitor closely. Send targeted mobile-phone promotions to re-engage.",
    },
    3: {
        "name":           "Cluster 3",
        "risk_level":     "Priority intervention required",
        "emoji":          "🔴",
        "color":          "#ef4444",
        "bg":             "rgba(239,68,68,0.07)",
        "border":         "rgba(239,68,68,0.35)",
        "traits": [
            "Frequently files complaints",
            "Rarely uses discount codes",
            "Most recent orders (active buyer, but dissatisfied)",
            "Uses many devices; low app engagement; places few orders",
            "Low tenure — highest churn risk in the cohort",
        ],
        "recommendation": "Immediate action: resolve complaints + personalised retention offer (cashback / voucher).",
    },
}

RISK_CONFIG = {
    "Very Low": {"color": "#22c55e", "emoji": "🟢", "label": "Very Low Risk",  "bg": "#052e16"},
    "Low":      {"color": "#84cc16", "emoji": "🟡", "label": "Low Risk",       "bg": "#1a2e05"},
    "Medium":   {"color": "#f59e0b", "emoji": "🟠", "label": "Medium Risk",    "bg": "#2d1b00"},
    "High":     {"color": "#f97316", "emoji": "🔴", "label": "High Risk",      "bg": "#2d0f00"},
    "Critical": {"color": "#ef4444", "emoji": "🚨", "label": "Critical Risk",  "bg": "#2d0505"},
}

# ── Model / data loaders ──────────────────────────────────────────────────────

@st.cache_resource
def load_xgb_clust():
    path = os.path.join(MODEL, "xgb_clust_model.pkl")
    if not os.path.exists(path):
        path = os.path.join(MODEL, "xgb_model.pkl")
    return joblib.load(path)

@st.cache_resource
def load_kmeans_fe():
    """K-Means model fitted on encoded training features (Step 5)."""
    path = os.path.join(MODEL, "kmeans_fe_model.pkl")
    if not os.path.exists(path):
        path = os.path.join(MODEL, "kmeans_model.pkl")
    return joblib.load(path)

@st.cache_resource
def load_transformers():
    enc    = joblib.load(os.path.join(MODEL, "encoder.pkl"))
    scaler = joblib.load(os.path.join(MODEL, "scaler.pkl"))
    imp    = joblib.load(os.path.join(MODEL, "imputer.pkl"))
    x_train_path = os.path.join(DATA, "X_train.csv")
    ohe_names = enc.get_feature_names_out().tolist()
    cat_cols  = enc.feature_names_in_.tolist()
    if os.path.exists(x_train_path):
        sample   = pd.read_csv(x_train_path, nrows=1)
        num_cols = [c for c in sample.columns if c not in ohe_names]
    else:
        num_cols = []
    return enc, scaler, imp, cat_cols, num_cols, ohe_names

@st.cache_resource
def load_metadata():
    return joblib.load(os.path.join(MODEL, "model_metadata.pkl"))

@st.cache_data
def load_risk_meta():
    path = os.path.join(MODEL, "risk_score_meta.pkl")
    return joblib.load(path) if os.path.exists(path) else None


# ── Prediction helper ─────────────────────────────────────────────────────────

def compute_risk_score(churn_prob: float, cashback_pct: float,
                       meta: dict) -> tuple:
    """
    Returns (risk_score_normalised, risk_category).
    Uses population percentile thresholds saved in risk_score_meta.pkl.
    Falls back to a linear heuristic if metadata is unavailable.
    """
    raw = 0.7 * churn_prob + 0.3 * cashback_pct

    if meta:
        rs = np.clip(
            (raw - meta["rs_min"]) / (meta["rs_max"] - meta["rs_min"] + 1e-10),
            0.0, 1.0,
        )
        if rs <= meta["p20"]:   cat = "Very Low"
        elif rs <= meta["p40"]: cat = "Low"
        elif rs <= meta["p60"]: cat = "Medium"
        elif rs <= meta["p80"]: cat = "High"
        else:                   cat = "Critical"
    else:
        # Fallback: normalise raw score by max possible (churn_prob=1, cashback_pct assumed ≤200)
        rs = np.clip(raw / (0.7 + 0.3 * 200), 0.0, 1.0)
        if rs <= 0.20:  cat = "Very Low"
        elif rs <= 0.40: cat = "Low"
        elif rs <= 0.60: cat = "Medium"
        elif rs <= 0.80: cat = "High"
        else:            cat = "Critical"

    return float(rs), cat


def run_prediction(input_data: pd.DataFrame,
                   cashback_amount: float,
                   order_count: int) -> dict:
    """
    Full prediction pipeline for a single customer:
      1. Feature engineering
      2. OHE → scale → impute
      3. K-Means cluster assignment (kmeans_fe_model.pkl)
      4. XGB-Clust prediction
      5. RiskScore computation
    """
    enc, scaler, imp, cat_cols, num_cols, ohe_names = load_transformers()
    meta_model = load_metadata()
    feature_names = meta_model.get("feature_names", [])

    # ── 1  Feature engineering ────────────────────────────────────────────────
    df = input_data.copy()
    tenure     = float(df["Tenure"].iloc[0])
    order_cnt  = max(float(order_count), 1)

    df["device_per_tenure"]  = df["NumberOfDeviceRegistered"] / (tenure + 1)
    df["cashback_per_order"] = cashback_amount / (order_cnt + 1)
    df["inactivity_ratio"]   = df["DaySinceLastOrder"] / (tenure + 1)
    if "rfm_total"   not in df.columns: df["rfm_total"]   = 7
    if "rfm_segment" not in df.columns: df["rfm_segment"] = "Loyal"

    # ── 2  Encode / scale / impute ────────────────────────────────────────────
    cat_encoded = pd.DataFrame(
        enc.transform(df[cat_cols]),
        columns=ohe_names,
        index=df.index,
    ).reset_index(drop=True)
    num_part = df[num_cols].reset_index(drop=True)

    X_input = pd.concat([num_part, cat_encoded], axis=1)
    for c in feature_names:
        if c not in X_input.columns:
            X_input[c] = 0
    X_input = X_input[feature_names]   # 35 cols, exact order

    X_scaled  = scaler.transform(X_input)
    X_imputed = imp.transform(X_scaled)
    X_final   = pd.DataFrame(X_imputed, columns=feature_names)

    # ── 3  K-Means cluster assignment ─────────────────────────────────────────
    km         = load_kmeans_fe()
    cluster_id = int(km.predict(X_final.values)[0])

    # ── 4  XGB-Clust prediction ───────────────────────────────────────────────
    X_clust = X_final.copy()
    X_clust["kmeans_cluster_id"] = cluster_id

    risk_meta = load_risk_meta()
    if risk_meta and "feature_names_clust" in risk_meta:
        # Ensure column order matches training exactly
        fn_clust = risk_meta["feature_names_clust"]
        for c in fn_clust:
            if c not in X_clust.columns:
                X_clust[c] = 0
        X_clust = X_clust[fn_clust]

    xgb_clust  = load_xgb_clust()
    churn_prob = float(xgb_clust.predict_proba(X_clust)[0, 1])

    # ── 5  RiskScore ──────────────────────────────────────────────────────────
    cashback_pct = cashback_amount / order_cnt
    risk_score, risk_category = compute_risk_score(churn_prob, cashback_pct, risk_meta)

    return {
        "churn_prob":    churn_prob,
        "cluster_id":    cluster_id,
        "risk_score":    risk_score,
        "risk_category": risk_category,
        "cashback_pct":  cashback_pct,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Layout
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="main-header">
    <h1>🚨 Churn Alert — Customer Risk Prediction</h1>
    <p>Real-time churn probability &amp; risk scoring powered by XGBoost-Cluster model</p>
</div>
""", unsafe_allow_html=True)

# ── Guard: model files exist? ─────────────────────────────────────────────────
_required = ["xgb_clust_model.pkl", "kmeans_fe_model.pkl",
             "encoder.pkl", "scaler.pkl", "imputer.pkl", "model_metadata.pkl"]
_missing  = [f for f in _required if not os.path.exists(os.path.join(MODEL, f))]
if _missing:
    st.error(f"⚠️  Missing model files: {', '.join(_missing)}. "
             "Run `python run_pipeline.py` first.")
    st.stop()

# ── Input form ────────────────────────────────────────────────────────────────
st.markdown("### 📋 Enter Customer Details")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="input-card"><h4>🧑 Demographics & Tenure</h4>', unsafe_allow_html=True)
    tenure         = st.slider("Tenure (months)", 0, 61, 6, key="tenure")
    city_tier      = st.selectbox("City Tier", ["1", "2", "3"], key="city")
    gender         = st.selectbox("Gender", ["Male", "Female"], key="gender")
    marital        = st.selectbox("Marital Status",
                                  ["Single", "Married", "Divorced"], key="marital")
    wh_dist        = st.slider("Warehouse to Home (km)", 5, 127, 30, key="wh")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="input-card"><h4>📦 Orders & Engagement</h4>', unsafe_allow_html=True)
    order_count    = st.slider("Order Count", 1, 16, 3, key="orders")
    cashback       = st.slider("Cashback Amount ($)", 0.0, 325.0, 150.0, step=0.5, key="cash")
    order_hike     = st.slider("Order Hike from Last Year (%)", 11, 26, 15, key="hike")
    coupon_used    = st.slider("Coupons Used", 0, 16, 2, key="coupon")
    day_last_order = st.slider("Days Since Last Order", 0, 46, 5, key="recency")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="input-card"><h4>📱 Devices & Satisfaction</h4>', unsafe_allow_html=True)
    n_devices      = st.slider("Number of Devices Registered", 1, 6, 2, key="dev")
    n_address      = st.slider("Number of Addresses", 1, 20, 2, key="addr")
    hours_app      = st.slider("Hours on App / Month", 0.0, 5.0, 2.0, step=0.5, key="hrs")
    satisfaction   = st.slider("Satisfaction Score (1–5)", 1, 5, 3, key="sat")
    complain       = st.selectbox("Has Complaint?", ["0", "1"], key="comp")
    st.markdown('</div>', unsafe_allow_html=True)

# Extra fields in a collapsed expander
with st.expander("⚙️  Preferences (optional — defaults applied if skipped)"):
    ec1, ec2, ec3 = st.columns(3)
    with ec1:
        login_device = st.selectbox("Preferred Login Device",
                                    ["Mobile", "Computer"], key="login")
    with ec2:
        payment_mode = st.selectbox("Preferred Payment Mode",
                                    ["Credit Card", "Debit Card", "UPI",
                                     "Cash on Delivery", "E wallet"], key="pay")
    with ec3:
        order_cat = st.selectbox("Preferred Order Category",
                                 ["Laptop & Accessory", "Mobile", "Fashion",
                                  "Grocery", "Others", "Mobile Phone"], key="cat")

# ── Predict button ────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
predict_btn = st.button("🔍  Predict Churn Risk", type="primary",
                        use_container_width=True, key="predict_btn")

# ── Prediction results ────────────────────────────────────────────────────────
if predict_btn:
    # Build raw input
    input_data = pd.DataFrame([{
        "Tenure":                    tenure,
        "CityTier":                  city_tier,
        "Gender":                    gender,
        "MaritalStatus":             marital,
        "Complain":                  complain,
        "SatisfactionScore":         satisfaction,
        "NumberOfDeviceRegistered":  n_devices,
        "NumberOfAddress":           n_address,
        "HourSpendOnApp":            hours_app,
        "OrderCount":                order_count,
        "CashbackAmount":            cashback,
        "WarehouseToHome":           wh_dist,
        "OrderAmountHikeFromlastYear": order_hike,
        "CouponUsed":                coupon_used,
        "DaySinceLastOrder":         day_last_order,
        "PreferredLoginDevice":      login_device,
        "PreferredPaymentMode":      payment_mode,
        "PreferedOrderCat":          order_cat,
    }])

    with st.spinner("Analysing customer profile…"):
        try:
            result = run_prediction(input_data, cashback, order_count)
        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.info("Ensure the full pipeline has been run before using Customer Lookup.")
            st.stop()

    churn_prob    = result["churn_prob"]
    risk_score    = result["risk_score"]
    risk_category = result["risk_category"]
    cluster_id    = result["cluster_id"]
    rc_cfg        = RISK_CONFIG.get(risk_category, RISK_CONFIG["Medium"])
    cluster_info  = CLUSTER_INFO.get(cluster_id, {
        "name": f"Cluster {cluster_id}", "risk_level": "Unknown",
        "emoji": "⚪", "color": "#94a3b8", "bg": "#1e293b",
        "border": "#334155", "traits": [], "recommendation": "",
    })

    st.markdown("---")
    st.markdown("### 📊 Prediction Results")

    # ── Row 1: three summary cards ────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)

    c1.markdown(f"""
    <div class="metric-card" style="border-color:{rc_cfg['color']}55">
        <span class="value" style="color:{rc_cfg['color']}">{churn_prob:.1%}</span>
        <span class="label">Churn Probability</span>
        <span class="sub">XGBoost-Cluster model</span>
    </div>
    """, unsafe_allow_html=True)

    c2.markdown(f"""
    <div class="metric-card" style="border-color:{rc_cfg['color']}55">
        <span class="value" style="color:{rc_cfg['color']}">{risk_score:.3f}</span>
        <span class="label">Risk Score  <small>[0 – 1]</small></span>
        <span class="sub">= 0.7×churn + 0.3×cashback/order, normalised</span>
    </div>
    """, unsafe_allow_html=True)

    c3.markdown(f"""
    <div class="metric-card" style="border-color:{rc_cfg['color']}55">
        <span class="value" style="font-size:1.5rem;color:{rc_cfg['color']}">
            {rc_cfg['emoji']} {rc_cfg['label']}
        </span>
        <span class="label">Risk Category</span>
        <span class="sub">Population percentile-based quintile</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 2: gauges + cluster card ──────────────────────────────────────────
    g1, g2 = st.columns(2)

    # Churn Probability gauge
    with g1:
        fig_churn = go.Figure(go.Indicator(
            mode  = "gauge+number",
            value = churn_prob * 100,
            number = {"suffix": "%", "font": {"size": 34, "color": rc_cfg["color"]}},
            gauge = {
                "axis":  {"range": [0, 100], "tickwidth": 1, "tickcolor": "#334155"},
                "bar":   {"color": rc_cfg["color"], "thickness": 0.28},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0,  20], "color": "rgba(34,197,94,0.12)"},
                    {"range": [20, 40], "color": "rgba(132,204,22,0.12)"},
                    {"range": [40, 60], "color": "rgba(245,158,11,0.12)"},
                    {"range": [60, 80], "color": "rgba(249,115,22,0.12)"},
                    {"range": [80, 100], "color": "rgba(239,68,68,0.15)"},
                ],
                "threshold": {
                    "line": {"color": "#f8fafc", "width": 2},
                    "thickness": 0.75,
                    "value": churn_prob * 100,
                },
            },
            title = {"text": "Churn Probability", "font": {"size": 13, "color": "#94a3b8"}},
            domain = {"x": [0, 1], "y": [0, 1]},
        ))
        fig_churn.update_layout(
            height=280,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#e2e8f0"},
            margin=dict(l=20, r=20, t=50, b=20),
        )
        st.plotly_chart(fig_churn, use_container_width=True)

    # Risk Score gauge
    with g2:
        fig_risk = go.Figure(go.Indicator(
            mode  = "gauge+number",
            value = risk_score * 100,
            number = {"suffix": "", "valueformat": ".1f",
                      "font": {"size": 34, "color": rc_cfg["color"]}},
            gauge = {
                "axis":  {"range": [0, 100], "tickwidth": 1, "tickcolor": "#334155",
                          "tickformat": ".0f"},
                "bar":   {"color": rc_cfg["color"], "thickness": 0.28},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0,  20], "color": "rgba(34,197,94,0.12)"},
                    {"range": [20, 40], "color": "rgba(132,204,22,0.12)"},
                    {"range": [40, 60], "color": "rgba(245,158,11,0.12)"},
                    {"range": [60, 80], "color": "rgba(249,115,22,0.12)"},
                    {"range": [80, 100], "color": "rgba(239,68,68,0.15)"},
                ],
                "threshold": {
                    "line": {"color": "#f8fafc", "width": 2},
                    "thickness": 0.75,
                    "value": risk_score * 100,
                },
            },
            title = {"text": "Risk Score (×100 for display)", "font": {"size": 13, "color": "#94a3b8"}},
            domain = {"x": [0, 1], "y": [0, 1]},
        ))
        fig_risk.update_layout(
            height=280,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#e2e8f0"},
            margin=dict(l=20, r=20, t=50, b=20),
        )
        st.plotly_chart(fig_risk, use_container_width=True)

    # ── Row 3: Risk scale bar ─────────────────────────────────────────────────
    cats    = ["Very Low", "Low", "Medium", "High", "Critical"]
    colors  = [RISK_CONFIG[c]["color"] for c in cats]
    idx     = cats.index(risk_category) if risk_category in cats else 2

    bar_html = '<div style="display:flex;gap:4px;margin:0.5rem 0 1.5rem;">'
    for i, (c, col) in enumerate(zip(cats, colors)):
        is_active = (i == idx)
        bar_html += (
            f'<div style="flex:1;padding:6px 4px;border-radius:6px;text-align:center;'
            f'background:{col if is_active else "rgba(71,85,105,0.2)"};'
            f'border:2px solid {col if is_active else "transparent"};'
            f'color:{"#fff" if is_active else "#64748b"};'
            f'font-size:0.75rem;font-weight:{"700" if is_active else "400"};">'
            f'{RISK_CONFIG[c]["emoji"]} {c}</div>'
        )
    bar_html += '</div>'
    st.markdown(bar_html, unsafe_allow_html=True)

    # ── Row 4: Cluster description card ───────────────────────────────────────
    c_color  = cluster_info["color"]
    c_bg     = cluster_info["bg"]
    c_border = cluster_info["border"]
    traits_html = "".join(f"<li>{t}</li>" for t in cluster_info.get("traits", []))

    st.markdown(f"""
    <div class="cluster-card"
         style="background:{c_bg};border:1px solid {c_border};">
        <h3 style="color:{c_color}">
            {cluster_info['emoji']}  {cluster_info['name']}
            &nbsp;<small style="font-size:0.8rem;color:#94a3b8;font-weight:400">
                — {cluster_info['risk_level']}
            </small>
        </h3>
        <ul>{traits_html}</ul>
        <div class="rec-box" style="border-left-color:{c_color}">
            💡 <b>Recommended action:</b> {cluster_info['recommendation']}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Row 5: Score breakdown ────────────────────────────────────────────────
    with st.expander("🔢 Score Breakdown"):
        cashback_pct = result["cashback_pct"]
        st.markdown(f"""
| Component | Raw Value | Weight | Contribution |
|-----------|-----------|--------|-------------|
| Churn Probability | `{churn_prob:.4f}` | 0.7 | `{0.7*churn_prob:.4f}` |
| Cashback per Order (cashback_pct) | `{cashback_pct:.4f}` | 0.3 | `{0.3*cashback_pct:.4f}` |
| **Raw Score** | `{0.7*churn_prob + 0.3*cashback_pct:.4f}` | — | — |
| **Normalised RiskScore** | `{risk_score:.4f}` | — | Population min-max |
| **Risk Category** | **{risk_category}** | — | Quintile percentile |
        """)
        rm = load_risk_meta()
        if rm:
            st.caption(
                f"Population thresholds (test set):  "
                f"p20={rm['p20']:.3f} | p40={rm['p40']:.3f} | "
                f"p60={rm['p60']:.3f} | p80={rm['p80']:.3f}"
            )
        else:
            st.warning("risk_score_meta.pkl not found — run Step 8 for calibrated thresholds.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;color:#334155;font-size:0.78rem;
            padding:2rem 0 0.5rem;margin-top:2rem;border-top:1px solid #1e293b;">
    🚨 Churn Alert | XGBoost-Cluster Model | K-Means Feature Engineering
</div>
""", unsafe_allow_html=True)
