"""Fix 3: Update streamlit_app.py for persona-aware dashboard."""
fpath = "app/streamlit_app.py"
with open(fpath, "r", encoding="utf-8") as f:
    content = f.read()

# ── Fix 3a: Add load_persona_map() after load_cluster_summary() ──────────────
OLD_A = '@st.cache_data\ndef load_cluster_summary():\n    return pd.read_csv(os.path.join(DATA, "cluster_summary.csv"))'
NEW_A = '@st.cache_data\ndef load_cluster_summary():\n    return pd.read_csv(os.path.join(DATA, "cluster_summary.csv"))\n\n@st.cache_data\ndef load_persona_map():\n    """Build cluster_id \u2192 persona metadata dict from cluster_summary.csv."""\n    cs_path = os.path.join(DATA, "cluster_summary.csv")\n    if not os.path.exists(cs_path):\n        return {}\n    cs = pd.read_csv(cs_path)\n    result = {}\n    for _, row in cs.iterrows():\n        cid = int(row["cluster"])\n        result[cid] = {\n            "persona":   row.get("persona",       f"Cluster {cid}"),\n            "emoji":     row.get("persona_emoji",  "\u26aa"),\n            "strategy":  row.get("strategy",       ""),\n            "churn_pct": float(row.get("churn_rate_pct", 0)),\n        }\n    return result'

if OLD_A in content:
    content = content.replace(OLD_A, NEW_A, 1)
    print("[3a] Added load_persona_map()")
else:
    print("[3a] SKIP (may already be applied)")

# ── Fix 3b: Cluster scatter with persona labels ────────────────────────────
OLD_B = '    # Cluster profiling scatter\n    if cluster_summary is not None:\n        st.markdown("#### Customer Cluster Analysis (K-Means)")\n        _x = next((c for c in ["Tenure_mean", "avg_tenure"] if c in cluster_summary.columns), None)\n        if _x:\n            _hover = ["count", "avg_churn_prob"] + (["strategy"] if "strategy" in cluster_summary.columns else [])\n            cluster_fig = px.scatter(\n                cluster_summary, x=_x, y="churn_rate_pct",\n                size="count", color="churn_rate_pct",\n                text="persona_label" if "persona_label" in cluster_summary.columns else None,\n                color_continuous_scale=["#22c55e", "#f97316", "#ef4444"],\n                hover_data=_hover,\n                labels={_x: "Avg Tenure (months)", "churn_rate_pct": "Churn Rate (%)"},\n            )\n            if "persona_label" in cluster_summary.columns:\n                cluster_fig.update_traces(textposition="top center")\n            cluster_fig.update_layout(height=440)\n            st.plotly_chart(cluster_fig, use_container_width=True)'

# Check if Fix 3b was already applied (has _hover)
if '_hover' in content and 'persona_label' in content:
    print("[3b] Already applied, skipping")
elif '# Cluster profiling scatter\n    if cluster_summary is not None:' in content:
    OLD_B2 = '    # Cluster profiling scatter\n    if cluster_summary is not None:\n        st.markdown("#### Customer Cluster Analysis (K-Means)")\n        _x = next((c for c in ["Tenure_mean", "avg_tenure"] if c in cluster_summary.columns), None)\n        if _x:\n            cluster_fig = px.scatter(\n                cluster_summary, x=_x, y="churn_rate_pct",\n                size="count", color="churn_rate_pct",\n                color_continuous_scale=["#22c55e", "#f97316", "#ef4444"],\n                hover_data=["count", "avg_churn_prob"],\n                labels={_x: "Avg Tenure (months)", "churn_rate_pct": "Churn Rate (%)"},\n            )\n            cluster_fig.update_layout(height=380)\n            st.plotly_chart(cluster_fig, use_container_width=True)'
    NEW_B2 = '    # Cluster profiling scatter - with persona labels\n    if cluster_summary is not None:\n        st.markdown("#### Customer Cluster Analysis (K-Means)")\n        _pm = load_persona_map()\n        if _pm:\n            cluster_summary["persona_label"] = cluster_summary["cluster"].apply(\n                lambda c: f"{_pm.get(int(c), {}).get(\'emoji\', \'\u26aa\')} {_pm.get(int(c), {}).get(\'persona\', f\'Cluster {c}\')}"\n            )\n        _x = next((c for c in ["Tenure_mean", "avg_tenure"] if c in cluster_summary.columns), None)\n        if _x:\n            _hover_cols = ["count", "avg_churn_prob"] + (["strategy"] if "strategy" in cluster_summary.columns else [])\n            cluster_fig = px.scatter(\n                cluster_summary, x=_x, y="churn_rate_pct",\n                size="count", color="churn_rate_pct",\n                text="persona_label" if "persona_label" in cluster_summary.columns else None,\n                color_continuous_scale=["#22c55e", "#f97316", "#ef4444"],\n                hover_data=_hover_cols,\n                labels={_x: "Avg Tenure (months)", "churn_rate_pct": "Churn Rate (%)"},\n            )\n            if "persona_label" in cluster_summary.columns:\n                cluster_fig.update_traces(textposition="top center")\n            cluster_fig.update_layout(height=440)\n            st.plotly_chart(cluster_fig, use_container_width=True)'
    if OLD_B2 in content:
        content = content.replace(OLD_B2, NEW_B2, 1)
        print("[3b] Updated cluster scatter with persona labels")
    else:
        print("[3b] ERROR: scatter target not found — check manually")
else:
    print("[3b] ERROR: pattern not found")

# ── Fix 3c: Dynamic Persona Guide ──────────────────────────────────────────
HARDCODED_GUIDE_MARKER = "#### Cluster Persona Guide"
DYNAMIC_GUIDE_MARKER   = "# Dynamic Persona Guide"

if DYNAMIC_GUIDE_MARKER in content:
    print("[3c] Already dynamic, skipping")
elif HARDCODED_GUIDE_MARKER in content:
    OLD_C = '        st.markdown("""#### Cluster Persona Guide\n| Cluster | Name | Drop-off Speed | Recommended Action |\n|---------|------|----------------|---------------------|\n| 0 | \u0110\u00f4 th\u1ecb ph\u00e0n n\u00e0n | Medium | Phone call to resolve complaints |\n| 1 | VIP T\u1ec9nh \u1ed5n \u0111\u1ecbnh | Slowest | VIP perks + loyalty rewards |\n| 2 | \u0110\u00f4 th\u1ecb trung th\u00e0nh | Moderate | Freeship + flash sale nudges |\n| 3 | S\u0103n Sale ph\u00e0n n\u00e0n | **Fastest \U0001f534** | Immediate cashback voucher via SMS |\n""")'
    NEW_C = '        # Dynamic Persona Guide (from cluster_summary.csv)\n        st.markdown("#### Cluster Persona Guide")\n        _pm2 = load_persona_map()\n        if _pm2:\n            _gd = [\n                {\n                    "Cluster":    cid,\n                    "Persona":    f\"{m[\'emoji\']} {m[\'persona\']}\",\n                    "Churn Rate": f\"{m[\'churn_pct\']:.1f}%\",\n                    "Chi\u1ebfn l\u01b0\u1ee3c": m["strategy"],\n                }\n                for cid, m in sorted(_pm2.items())\n            ]\n            _gdf = pd.DataFrame(_gd)\n            def _cr(row):\n                pct = float(str(row["Churn Rate"]).replace("%",""))\n                bg = "#fee2e2" if pct > 35 else "#fef3c7" if pct > 15 else "#d1fae5"\n                return [f"background-color: {bg}"] * len(row)\n            st.dataframe(_gdf.style.apply(_cr, axis=1), use_container_width=True, hide_index=True)\n        else:\n            st.info("Run the full pipeline to generate cluster personas.")'
    if OLD_C in content:
        content = content.replace(OLD_C, NEW_C, 1)
        print("[3c] Updated Persona Guide to dynamic version")
    else:
        # Try to find by partial match
        idx = content.find(HARDCODED_GUIDE_MARKER)
        print(f"[3c] Hardcoded guide found at {idx}, but exact match failed.")
        print("     First 300 chars around marker:")
        print(repr(content[idx-50:idx+300]))
else:
    print("[3c] No persona guide section found at all")

with open(fpath, "w", encoding="utf-8") as f:
    f.write(content)
print(f"\nDone. Total lines: {content.count(chr(10))}")
