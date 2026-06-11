"""
Modify Customer Lookup page:
  1. Add MinMaxScaler import
  2. Add load_kmeans() + load_xtest_raw_for_scaler() loaders
  3. Add assign_cluster_to_customer() helper
  4. Inject cluster assignment into the prediction block
  5. Replace 3-column result with 4-column result including Cluster card
"""
fpath = "app/streamlit_app.py"
with open(fpath, "r", encoding="utf-8") as f:
    content = f.read()

# ── 1. Add MinMaxScaler import ────────────────────────────────────────────────
OLD_IMPORT = "import joblib\nimport os\nimport warnings"
NEW_IMPORT = "import joblib\nimport os\nimport warnings\nfrom sklearn.preprocessing import MinMaxScaler"

if "from sklearn.preprocessing import MinMaxScaler" in content:
    print("[1] MinMaxScaler already imported")
elif OLD_IMPORT in content:
    content = content.replace(OLD_IMPORT, NEW_IMPORT, 1)
    print("[1] Added MinMaxScaler import")
else:
    print("[1] ERROR: import anchor not found")

# ── 2. Add load_kmeans() + load_xtest_raw_for_scaler() after load_persona_map ──
OLD_LOADER_END = "@st.cache_resource\ndef load_model():"
NEW_LOADERS = """@st.cache_resource
def load_kmeans():
    km_path = os.path.join(MODEL, "kmeans_model.pkl")
    if not os.path.exists(km_path):
        return None
    return joblib.load(km_path)

@st.cache_data
def load_xtest_raw_for_scaler():
    \"\"\"Load X_test_raw to re-fit the MinMaxScaler used during clustering.\"\"\"
    return pd.read_csv(os.path.join(DATA, "X_test_raw.csv"))

@st.cache_resource
def load_model():"""

if "def load_kmeans" in content:
    print("[2] load_kmeans already exists")
elif OLD_LOADER_END in content:
    content = content.replace(OLD_LOADER_END, NEW_LOADERS, 1)
    print("[2] Added load_kmeans() and load_xtest_raw_for_scaler()")
else:
    print("[2] ERROR: loader anchor not found")

# ── 3. Add assign_cluster_to_customer() before Page 7 block ──────────────────
PAGE7_MARKER = "# Page 7: Customer Lookup"
HELPER_FUNC = """def assign_cluster_to_customer(input_raw_df: pd.DataFrame) -> int:
    \"\"\"
    Assigns a K-Means cluster to a new customer.
    Re-fits MinMaxScaler on X_test_raw.csv (same columns as training) then predicts.
    Returns cluster ID (int) or -1 if K-Means model unavailable.
    \"\"\"
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

# Page 7: Customer Lookup"""

if "def assign_cluster_to_customer" in content:
    print("[3] assign_cluster_to_customer already exists")
elif PAGE7_MARKER in content:
    content = content.replace(PAGE7_MARKER, HELPER_FUNC, 1)
    print("[3] Added assign_cluster_to_customer()")
else:
    print("[3] ERROR: Page 7 marker not found")

# ── 4. Inject cluster assignment + replace tier block ─────────────────────────
OLD_TIER = """            prob = model.predict_proba(X_final)[0, 1]
            threshold = meta["threshold"]
            prediction = int(prob >= threshold)

            # Risk tier
            if prob >= 0.70:
                tier = "High Risk"
                tier_color = "#ef4444"
                action = "\U0001f4de Immediate phone call + 20% discount voucher"
                emoji = "\U0001f534"
            elif prob >= 0.40:
                tier = "Medium Risk"
                tier_color = "#f97316"
                action = "\U0001f4e7 Send personalised email offer"
                emoji = "\U0001f7e1"
            else:
                tier = "Low Risk"
                tier_color = "#22c55e"
                action = "\u2b50 Enrol in loyalty program"
                emoji = "\U0001f7e2\""""

NEW_TIER = """            prob = model.predict_proba(X_final)[0, 1]
            threshold = meta["threshold"]
            prediction = int(prob >= threshold)

            # ── Cluster assignment ─────────────────────────────────────────────
            cluster_id = assign_cluster_to_customer(input_data)

            persona_map_local = load_persona_map()
            cluster_meta = persona_map_local.get(cluster_id, {
                "persona": f"Cluster {cluster_id}" if cluster_id >= 0 else "Unknown",
                "emoji": "\u26aa",
                "strategy": "",
                "churn_pct": 0.0,
            })

            # Cluster-specific actions from cluster_summary.csv
            action_high   = "\U0001f6a8 Li\u00ean h\u1ec7 ng\u01b0\u1eddi d\u00f9ng ng\u01b0ay"
            action_medium = "\U0001f4e7 G\u1eedi email ch\u0103m s\u00f3c"
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
                emoji = "\U0001f534"
            elif prob >= 0.40:
                tier = "Medium Risk"
                tier_color = "#f97316"
                action = action_medium
                emoji = "\U0001f7e1"
            else:
                tier = "Low Risk"
                tier_color = "#22c55e"
                action = "\u2705 Duy tr\u00ec ch\u0103m s\u00f3c ti\u00eau chu\u1ea9n"
                emoji = "\U0001f7e2\""""

if OLD_TIER in content:
    content = content.replace(OLD_TIER, NEW_TIER, 1)
    print("[4] Injected cluster assignment and updated tier block")
else:
    print("[4] ERROR: tier block not found — checking partial match")
    if "prob = model.predict_proba(X_final)[0, 1]" in content:
        print("   predict_proba line found")
    if "Immediate phone call" in content:
        print("   'Immediate phone call' found")

# ── 5. Replace 3-column result display with 4-column version ─────────────────
OLD_CARDS = """            st.markdown("---")
            st.markdown("#### \U0001f4cb Prediction Result")
            rcol1, rcol2, rcol3 = st.columns(3)
            rcol1.markdown(f\"\"\"
            <div class="metric-card" style="border-color:{tier_color}33">
                <span class="value" style="color:{tier_color}">{prob:.1%}</span>
                <span class="label">Churn Probability</span>
            </div>
            \"\"\", unsafe_allow_html=True)
            rcol2.markdown(f\"\"\"
            <div class="metric-card" style="border-color:{tier_color}33">
                <span class="value" style="font-size:1.5rem;color:{tier_color}">{emoji} {tier}</span>
                <span class="label">Risk Classification</span>
            </div>
            \"\"\", unsafe_allow_html=True)
            rcol3.markdown(f\"\"\"
            <div class="metric-card">
                <span class="value" style="font-size:1.1rem;color:#3b82f6">{action}</span>
                <span class="label">Recommended Action</span>
            </div>
            \"\"\", unsafe_allow_html=True)"""

NEW_CARDS = """            st.markdown("---")
            st.markdown("#### \U0001f4cb Prediction Result")
            rcol1, rcol2, rcol3, rcol4 = st.columns(4)

            # Card 1 — Churn Probability
            rcol1.markdown(f\"\"\"
            <div class="metric-card" style="border-color:{tier_color}33">
                <span class="value" style="color:{tier_color}">{prob:.1%}</span>
                <span class="label">Churn Probability</span>
            </div>
            \"\"\", unsafe_allow_html=True)

            # Card 2 — Cluster Persona
            _cid_display = f"Cluster {cluster_id}" if cluster_id >= 0 else "N/A"
            rcol2.markdown(f\"\"\"
            <div class="metric-card" style="border-color:#6366f133">
                <span class="value" style="font-size:1.3rem;color:#6366f1">
                    {cluster_meta['emoji']} {cluster_meta['persona']}
                </span>
                <span class="label">Customer Cluster ({_cid_display})</span>
                <span class="delta" style="color:#94a3b8">
                    Cluster avg churn: {cluster_meta['churn_pct']:.1f}%
                </span>
            </div>
            \"\"\", unsafe_allow_html=True)

            # Card 3 — Risk Tier
            rcol3.markdown(f\"\"\"
            <div class="metric-card" style="border-color:{tier_color}33">
                <span class="value" style="font-size:1.5rem;color:{tier_color}">{emoji} {tier}</span>
                <span class="label">Risk Classification</span>
            </div>
            \"\"\", unsafe_allow_html=True)

            # Card 4 — Recommended Action (cluster-specific)
            rcol4.markdown(f\"\"\"
            <div class="metric-card">
                <span class="value" style="font-size:1rem;color:#3b82f6">{action}</span>
                <span class="label">Recommended Action</span>
                <span class="delta" style="color:#94a3b8;font-size:0.72rem">{cluster_meta['strategy']}</span>
            </div>
            \"\"\", unsafe_allow_html=True)"""

if OLD_CARDS in content:
    content = content.replace(OLD_CARDS, NEW_CARDS, 1)
    print("[5] Replaced 3-column cards with 4-column version")
else:
    print("[5] ERROR: result cards block not found")
    if 'rcol1, rcol2, rcol3 = st.columns(3)' in content:
        print("   3-column line found")

with open(fpath, "w", encoding="utf-8") as f:
    f.write(content)
print(f"\nDone. Total lines: {content.count(chr(10))}")
