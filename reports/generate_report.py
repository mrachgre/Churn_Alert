"""
Generate a self-contained, printer-friendly HTML report
from the Churn Alert pipeline results.
All charts are embedded as base64 so the file is portable.
"""

import base64, os, re, sys

# ── Paths ─────────────────────────────────────────────────────────────────────
THIS    = os.path.dirname(os.path.abspath(__file__))
ROOT    = os.path.normpath(os.path.join(THIS, ".."))
OUT_DIR = os.path.join(ROOT, "outputs", "E_Commer_Data")
DATA    = os.path.join(ROOT, "data",    "E_Commer_Data")
MODEL   = os.path.join(ROOT, "models",  "E_Commer_Data")
HTML_OUT = os.path.join(THIS, "pipeline_report.html")

# ── Charts to embed ───────────────────────────────────────────────────────────
CHARTS = {
    "03_class_distribution":  os.path.join(OUT_DIR, "03_class_distribution.png"),
    "04_rfm_churn_rate":       os.path.join(OUT_DIR, "04_rfm_churn_rate.png"),
    "05b_chi2_scores":         os.path.join(OUT_DIR, "05b_chi2_scores.png"),
    "06b_model_comparison":    os.path.join(OUT_DIR, "06b_model_comparison.png"),
    "06_roc_curve":            os.path.join(OUT_DIR, "06_roc_curve.png"),
    "06_shap_bar":             os.path.join(OUT_DIR, "06_shap_bar.png"),
    "07_cluster_profiling":    os.path.join(OUT_DIR, "07_cluster_profiling.png"),
    "08_km_by_cluster":        os.path.join(OUT_DIR, "08_km_by_cluster.png"),
}

def b64img(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()

# ── Model comparison table from CSV ──────────────────────────────────────────
def model_table():
    import csv
    p = os.path.join(DATA, "model_comparison.csv")
    if not os.path.exists(p):
        return ""
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    headers = list(rows[0].keys())
    medals = {"XGB-Base": "🥇", "RF-Clust": "🥈", "XGB-Clust": "⭐"}
    hl     = {"XGB-Base", "RF-Clust", "XGB-Clust"}
    th = "".join(f"<th>{h}</th>" for h in headers)
    body = ""
    for r in rows:
        cls = ' class="highlight"' if r["Model"] in hl else ""
        medal = medals.get(r["Model"], "")
        cells = "".join(
            f"<td>{medal + ' ' if h == 'Model' and medal else ''}{r[h]}</td>"
            for h in headers
        )
        body += f"<tr{cls}>{cells}</tr>\n"
    return f"<table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>"

# ── Cluster table from CSV ────────────────────────────────────────────────────
def cluster_table():
    import csv
    p = os.path.join(DATA, "cluster_summary.csv")
    if not os.path.exists(p):
        return ""
    rows  = list(csv.DictReader(open(p, encoding="utf-8")))
    cols  = ["cluster","count","churn_rate_pct","persona_emoji","persona","strategy"]
    avail = [c for c in cols if c in rows[0]]
    th    = "".join(f"<th>{c.replace('_',' ').title()}</th>" for c in avail)
    body  = ""
    for r in rows:
        cells = "".join(f"<td>{r.get(c,'')}</td>" for c in avail)
        body += f"<tr>{cells}</tr>\n"
    return f"<table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>"

# ── Build HTML ────────────────────────────────────────────────────────────────
imgs = {k: b64img(v) for k, v in CHARTS.items()}

def img_tag(key, caption=""):
    src = imgs.get(key)
    if not src:
        return f'<p class="warn">⚠ Chart not found: {key}</p>'
    cap = f"<figcaption>{caption}</figcaption>" if caption else ""
    return f'<figure><img src="{src}" alt="{key}">{cap}</figure>'

html = f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Churn Alert — Báo Cáo Kỹ Thuật Pipeline</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root {{
  --bg:      #0f172a;
  --surface: #1e293b;
  --border:  #334155;
  --text:    #e2e8f0;
  --muted:   #94a3b8;
  --blue:    #3b82f6;
  --green:   #22c55e;
  --yellow:  #f59e0b;
  --orange:  #f97316;
  --red:     #ef4444;
  --purple:  #a855f7;
}}

* {{ box-sizing: border-box; margin: 0; padding: 0; }}

body {{
  font-family: 'Inter', sans-serif;
  background: var(--bg);
  color: var(--text);
  line-height: 1.7;
  padding: 0;
}}

/* ── Cover page ── */
.cover {{
  min-height: 100vh;
  display: flex; flex-direction: column;
  justify-content: center; align-items: center;
  background: linear-gradient(135deg, #0a0f1e 0%, #0f172a 50%, #0a1628 100%);
  border-bottom: 2px solid var(--blue);
  text-align: center;
  padding: 4rem 2rem;
  page-break-after: always;
}}
.cover h1 {{
  font-size: 3rem; font-weight: 800;
  background: linear-gradient(90deg, #f8fafc, #93c5fd, #f8fafc);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  margin-bottom: 1rem;
}}
.cover .subtitle {{ color: var(--muted); font-size: 1.1rem; margin-bottom: 2rem; }}
.cover .meta {{
  display: flex; gap: 2rem; flex-wrap: wrap; justify-content: center;
  margin-top: 2rem;
}}
.cover .meta-card {{
  background: rgba(59,130,246,0.1);
  border: 1px solid rgba(59,130,246,0.3);
  border-radius: 12px; padding: 1rem 1.5rem;
  font-size: 0.95rem;
}}
.cover .meta-card strong {{ color: var(--blue); display: block; font-size: 1.4rem; }}

/* ── Layout ── */
.container {{
  max-width: 1100px; margin: 0 auto;
  padding: 3rem 2rem;
}}

/* ── Section ── */
.section {{
  margin-bottom: 4rem;
  page-break-inside: avoid;
}}
.section-header {{
  display: flex; align-items: center; gap: 0.75rem;
  border-bottom: 2px solid var(--border);
  padding-bottom: 0.75rem; margin-bottom: 1.5rem;
}}
.section-header h2 {{
  font-size: 1.5rem; font-weight: 700; color: var(--text);
}}
.step-badge {{
  background: var(--blue);
  color: #fff; font-size: 0.75rem; font-weight: 700;
  padding: 0.2rem 0.6rem; border-radius: 6px;
  white-space: nowrap;
}}

h3 {{ color: var(--blue); font-size: 1.1rem; margin: 1.5rem 0 0.75rem; }}
h4 {{ color: var(--muted); font-size: 0.95rem; margin: 1rem 0 0.5rem; }}

p {{ margin-bottom: 0.75rem; color: var(--text); }}

/* ── Tables ── */
table {{
  width: 100%; border-collapse: collapse;
  margin: 1rem 0 1.5rem; font-size: 0.88rem;
}}
th {{
  background: rgba(59,130,246,0.15);
  color: var(--blue); font-weight: 600;
  padding: 0.6rem 0.8rem; text-align: left;
  border-bottom: 2px solid var(--border);
}}
td {{
  padding: 0.5rem 0.8rem;
  border-bottom: 1px solid rgba(51,65,85,0.5);
}}
tr:nth-child(even) td {{ background: rgba(30,41,59,0.4); }}
tr.highlight td {{
  background: rgba(59,130,246,0.08);
  border-left: 3px solid var(--blue);
}}

/* ── Code blocks ── */
pre, code {{
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.83rem;
}}
pre {{
  background: #0d1117; border: 1px solid var(--border);
  border-radius: 10px; padding: 1.2rem 1.4rem;
  overflow-x: auto; margin: 1rem 0 1.5rem;
  line-height: 1.6;
}}
code {{
  background: rgba(59,130,246,0.12);
  color: #93c5fd;
  padding: 0.15em 0.4em; border-radius: 4px;
}}

/* ── Formula block ── */
.formula {{
  background: linear-gradient(135deg, rgba(59,130,246,0.08), rgba(168,85,247,0.08));
  border: 1px solid rgba(59,130,246,0.3);
  border-radius: 12px; padding: 1.5rem;
  margin: 1rem 0 1.5rem; text-align: center;
  font-size: 1rem;
}}
.formula .eq {{
  font-family: 'JetBrains Mono', monospace;
  color: #c4b5fd; font-size: 1.1rem;
  line-height: 2;
}}

/* ── Alert boxes ── */
.alert {{
  border-radius: 10px; padding: 1rem 1.2rem;
  margin: 1rem 0 1.5rem;
  border-left: 4px solid;
  font-size: 0.9rem;
}}
.alert-note    {{ background: rgba(59,130,246,0.08);  border-color: var(--blue);   color: #93c5fd; }}
.alert-warn    {{ background: rgba(245,158,11,0.08);  border-color: var(--yellow); color: #fcd34d; }}
.alert-danger  {{ background: rgba(239,68,68,0.08);   border-color: var(--red);    color: #fca5a5; }}
.alert-success {{ background: rgba(34,197,94,0.08);   border-color: var(--green);  color: #86efac; }}

/* ── Charts ── */
figure {{
  margin: 1.5rem 0;
  text-align: center;
  page-break-inside: avoid;
}}
figure img {{
  max-width: 100%; border-radius: 12px;
  border: 1px solid var(--border);
  box-shadow: 0 4px 20px rgba(0,0,0,0.4);
}}
figcaption {{
  color: var(--muted); font-size: 0.8rem;
  margin-top: 0.5rem; font-style: italic;
}}

/* ── Cluster cards ── */
.cluster-grid {{
  display: grid; grid-template-columns: repeat(2, 1fr);
  gap: 1rem; margin: 1rem 0 1.5rem;
}}
.cluster-card {{
  border-radius: 14px; padding: 1.25rem 1.5rem;
  border: 1px solid;
}}
.c0 {{ border-color: #22c55e; background: rgba(34,197,94,0.06); }}
.c1 {{ border-color: #3b82f6; background: rgba(59,130,246,0.06); }}
.c2 {{ border-color: #f59e0b; background: rgba(245,158,11,0.06); }}
.c3 {{ border-color: #ef4444; background: rgba(239,68,68,0.06); }}
.cluster-card h4 {{ font-size: 1rem; font-weight: 700; margin: 0 0 0.5rem; }}
.cluster-card ul {{ margin: 0; padding-left: 1.2rem; font-size: 0.87rem; color: var(--muted); line-height: 1.8; }}

/* ── Risk scale ── */
.risk-scale {{
  display: flex; gap: 4px; margin: 1rem 0 1.5rem;
}}
.risk-chip {{
  flex: 1; padding: 0.6rem 4px;
  border-radius: 8px; text-align: center;
  font-size: 0.78rem; font-weight: 600;
}}

/* ── Priority matrix ── */
.priority-grid {{
  display: grid; gap: 0.6rem; margin: 1rem 0 1.5rem;
}}
.priority-row {{
  display: grid; grid-template-columns: 140px 1fr;
  gap: 0.6rem; align-items: center;
}}
.p-badge {{
  border-radius: 8px; padding: 0.4rem 0.8rem;
  font-weight: 700; font-size: 0.85rem; text-align: center;
}}
.p1 {{ background: rgba(239,68,68,0.2);   color: #fca5a5; }}
.p2 {{ background: rgba(249,115,22,0.2);  color: #fdba74; }}
.p3 {{ background: rgba(245,158,11,0.2);  color: #fcd34d; }}
.p4 {{ background: rgba(132,204,22,0.2);  color: #bef264; }}
.p5 {{ background: rgba(34,197,94,0.2);   color: #86efac; }}
.p-desc {{ font-size: 0.87rem; color: var(--text); }}

/* ── Print ── */
@media print {{
  body {{ background: #fff; color: #111; }}
  .cover {{ background: #f8fafc; color: #111; min-height: auto; padding: 3rem; }}
  .cover h1 {{ -webkit-text-fill-color: #1e3a5f; background: none; }}
  pre {{ background: #f1f5f9; border-color: #cbd5e1; }}
  table {{ font-size: 0.78rem; }}
  .section {{ page-break-inside: avoid; }}
}}
</style>
</head>
<body>

<!-- ══ COVER ═══════════════════════════════════════════════════════════════ -->
<div class="cover">
  <div>🚨</div>
  <h1>Churn Alert</h1>
  <p class="subtitle">Báo Cáo Kỹ Thuật Pipeline — E-Commerce Customer Churn</p>
  <div class="meta">
    <div class="meta-card"><strong>5,630</strong>Khách hàng</div>
    <div class="meta-card"><strong>16.8%</strong>Churn Rate</div>
    <div class="meta-card"><strong>9 Steps</strong>Pipeline</div>
    <div class="meta-card"><strong>0.979</strong>XGB-Clust ROC-AUC</div>
    <div class="meta-card"><strong>K = 4</strong>Clusters</div>
  </div>
</div>

<div class="container">

<!-- ══ STEP 1 ══════════════════════════════════════════════════════════════ -->
<div class="section">
  <div class="section-header">
    <span class="step-badge">Step 1</span>
    <h2>Data Cleaning</h2>
  </div>
  <p>Load Excel (sheet <code>E Comm</code>) → 5,630 rows × 20 cột. Chuẩn hóa typo categorical và ép kiểu dữ liệu.</p>
  <table>
    <thead><tr><th>Trước</th><th>Sau</th><th>Cột</th></tr></thead>
    <tbody>
      <tr><td>"Mobile Phone", "Phone"</td><td>"Mobile"</td><td>PreferedOrderCat</td></tr>
      <tr><td>"CC"</td><td>"Credit Card"</td><td>PreferredPaymentMode</td></tr>
      <tr><td>"COD"</td><td>"Cash on Delivery"</td><td>PreferredPaymentMode</td></tr>
    </tbody>
  </table>
  <div class="alert alert-success">✅ <strong>Kết quả:</strong> ecommerce_churn_clean.csv — 5,630 rows, không duplicate, category nhất quán.</div>
</div>

<!-- ══ STEP 2 ══════════════════════════════════════════════════════════════ -->
<div class="section">
  <div class="section-header">
    <span class="step-badge">Step 2</span>
    <h2>Missing Data Analysis</h2>
  </div>
  <p>Phân loại MCAR/MAR/MNAR: so sánh churn rate giữa nhóm có và không có missing (|Δchurn| > 5% → MAR/MNAR).</p>
  <table>
    <thead><tr><th>Cột</th><th>Missing</th><th>Phân loại</th><th>Xử lý</th></tr></thead>
    <tbody>
      <tr><td>DaySinceLastOrder</td><td>5.45%</td><td>MCAR</td><td>KNN Imputer k=5</td></tr>
      <tr><td>OrderAmountHikeFromlastYear</td><td>4.71%</td><td><strong>MAR/MNAR</strong></td><td>KNN Imputer k=5</td></tr>
      <tr><td>Tenure</td><td>4.69%</td><td><strong>MAR/MNAR</strong></td><td>KNN Imputer k=5</td></tr>
      <tr><td>OrderCount</td><td>4.58%</td><td><strong>MAR/MNAR</strong></td><td>KNN Imputer k=5</td></tr>
      <tr><td>CouponUsed</td><td>4.55%</td><td><strong>MAR/MNAR</strong></td><td>KNN Imputer k=5</td></tr>
      <tr><td>HourSpendOnApp</td><td>4.53%</td><td><strong>MAR/MNAR</strong></td><td>KNN Imputer k=5</td></tr>
      <tr><td>WarehouseToHome</td><td>4.46%</td><td><strong>MAR/MNAR</strong></td><td>KNN Imputer k=5</td></tr>
    </tbody>
  </table>
</div>

<!-- ══ STEP 3 ══════════════════════════════════════════════════════════════ -->
<div class="section">
  <div class="section-header">
    <span class="step-badge">Step 3</span>
    <h2>Exploratory Data Analysis</h2>
  </div>
  <table>
    <thead><tr><th>Phân tích</th><th>Phương pháp</th><th>Phát hiện chính</th></tr></thead>
    <tbody>
      <tr><td>Phân bố nhãn</td><td>Bar + Pie</td><td><strong>Mất cân bằng:</strong> 83.2% retained vs 16.8% churn → cần SMOTE</td></tr>
      <tr><td>Churn theo Tenure</td><td>Bar binned</td><td>Churn cao nhất ở 0–3 tháng (~35%)</td></tr>
      <tr><td>Churn theo Satisfaction</td><td>Bar</td><td>Score 1 → ~25%; Score 5 → ~10%</td></tr>
      <tr><td>Pearson Heatmap</td><td>Numeric corr</td><td>Tenure tương quan âm mạnh với churn</td></tr>
      <tr><td>Cramér's V</td><td>Categorical assoc</td><td>Complain liên kết mạnh nhất với churn</td></tr>
      <tr><td>Point-Biserial</td><td>Numeric vs Churn</td><td>Top: Tenure, CashbackAmount, SatisfactionScore</td></tr>
    </tbody>
  </table>
  {img_tag("03_class_distribution", "Phân bố nhãn Churn/Retained")}
</div>

<!-- ══ STEP 4 ══════════════════════════════════════════════════════════════ -->
<div class="section">
  <div class="section-header">
    <span class="step-badge">Step 4</span>
    <h2>RFM Segmentation</h2>
  </div>
  <p>Dùng <strong>SQLite in-memory + NTILE(5)</strong> window function để chấm điểm R/F/M từ 1–5.</p>
  <table>
    <thead><tr><th>Chiều</th><th>Mapping</th><th>Hướng điểm</th></tr></thead>
    <tbody>
      <tr><td><strong>R</strong> — Recency</td><td>DaySinceLastOrder</td><td>Ít ngày → điểm cao</td></tr>
      <tr><td><strong>F</strong> — Frequency</td><td>OrderCount</td><td>Nhiều đơn → điểm cao</td></tr>
      <tr><td><strong>M</strong> — Monetary</td><td>CashbackAmount</td><td>Cashback cao → điểm cao</td></tr>
    </tbody>
  </table>
  <table>
    <thead><tr><th>Segment</th><th>Điều kiện</th><th>Số KH</th></tr></thead>
    <tbody>
      <tr><td>Champions</td><td>F ≥ 4 và M ≥ 4</td><td>1,505</td></tr>
      <tr><td>Recent Customers</td><td>R ≥ 4 và F &lt; 3</td><td>1,319</td></tr>
      <tr><td>Hibernating</td><td>Còn lại</td><td>1,270</td></tr>
      <tr><td>Loyal</td><td>F ≥ 3 và M ≥ 3</td><td>1,244</td></tr>
      <tr><td>At Risk</td><td>R ≤ 2 và F ≥ 3</td><td>292</td></tr>
    </tbody>
  </table>
  <div class="alert alert-note">ℹ RFM standalone ROC-AUC = <strong>0.5503</strong> — yếu độc lập, nhưng rfm_segment + rfm_total là input features quan trọng cho model.</div>
  {img_tag("04_rfm_churn_rate", "Churn rate theo RFM Segment")}
</div>

<!-- ══ STEP 5 ══════════════════════════════════════════════════════════════ -->
<div class="section">
  <div class="section-header">
    <span class="step-badge">Step 5</span>
    <h2>Preprocessing — Đường Tree Models</h2>
  </div>
  <h3>Feature Engineering</h3>
  <pre>device_per_tenure  = NumberOfDeviceRegistered / (Tenure + 1)
cashback_per_order = CashbackAmount / (OrderCount + 1)
inactivity_ratio   = DaySinceLastOrder / (Tenure + 1)</pre>
  <h3>Pipeline</h3>
  <table>
    <thead><tr><th>Bước</th><th>Kỹ thuật</th><th>Chi tiết</th></tr></thead>
    <tbody>
      <tr><td>Train/Test Split</td><td>Stratified</td><td>80/20 | random_state=42 | Train: 4,504 / Test: 1,126</td></tr>
      <tr><td>OHE</td><td>OneHotEncoder</td><td>8 categorical → 27 OHE features | fit on train only</td></tr>
      <tr><td>Scaling</td><td>StandardScaler</td><td>Fit on train only</td></tr>
      <tr><td>Imputation</td><td>KNNImputer k=5</td><td>Fit on train only</td></tr>
      <tr><td>SMOTE</td><td>Synthetic Minority Over-sampling</td><td>Train only: 758 → 3,746 (Class 1)</td></tr>
      <tr><td>KMeans FE</td><td>K=4, silhouette=0.1352</td><td>Fit on 4,504 pre-SMOTE | Strategy A (distances) + B (cluster_id)</td></tr>
    </tbody>
  </table>
  <div class="alert alert-warn">⚠ KMeans chỉ fit một lần trên training data. Test set dùng km.predict() — không có data leakage.</div>
</div>

<!-- ══ STEP 5b ═════════════════════════════════════════════════════════════ -->
<div class="section">
  <div class="section-header">
    <span class="step-badge">Step 5b</span>
    <h2>Preprocessing — Đường Linear Models</h2>
  </div>
  <p>Linear models nhạy cảm với outlier và scale → pipeline riêng biệt.</p>
  <table>
    <thead><tr><th>Bước</th><th>Kỹ thuật</th><th>Chỉ fit train?</th></tr></thead>
    <tbody>
      <tr><td>Outlier clip</td><td>IQR × 1.5 (clip, không drop rows)</td><td>✅</td></tr>
      <tr><td>OHE</td><td>Reuse encoder.pkl từ Step 5</td><td>—</td></tr>
      <tr><td>MinMaxScaler</td><td>Chi² yêu cầu giá trị ≥ 0</td><td>✅</td></tr>
      <tr><td>KNN Imputer</td><td>k=5</td><td>✅</td></tr>
      <tr><td>Chi² Selection</td><td>SelectKBest(chi2), p &lt; 0.05</td><td>✅</td></tr>
      <tr><td>SMOTE</td><td>Sau selection</td><td>✅</td></tr>
    </tbody>
  </table>
  <div class="alert alert-success">✅ Base: 23/35 features | Dist variant: 25/39 features được giữ lại (p &lt; 0.05)</div>
  {img_tag("05b_chi2_scores", "Chi-Square Feature Selection — Base vs Dist variant")}
</div>

<!-- ══ STEP 6 ══════════════════════════════════════════════════════════════ -->
<div class="section">
  <div class="section-header">
    <span class="step-badge">Step 6</span>
    <h2>Modelling — 6 Models So Sánh</h2>
  </div>
  <h3>XGBoost Base — Bayesian Optimization</h3>
  <pre>Optuna: 40 trials, CV=5-fold stratified
Best: n_estimators=463, max_depth=8, lr=0.0834, subsample=0.679
Calibration: CalibratedClassifierCV(cv=5)
Threshold: 0.119 (Youden's J)</pre>

  <h3>Kết quả so sánh</h3>
  {model_table()}
  <p style="font-size:0.85rem;color:var(--muted);">⭐ XGB-Clust = model production | 🥇 XGB-Base = benchmark accuracy | 🥈 RF-Clust = best ROC-AUC</p>

  {img_tag("06b_model_comparison", "So sánh 6 models theo Accuracy, F1, ROC-AUC, PR-AUC")}
  {img_tag("06_roc_curve", "ROC Curve — XGBoost Base (AUC=0.969)")}
  {img_tag("06_shap_bar", "SHAP Feature Importance — Top predictors")}
</div>

<!-- ══ STEP 7 ══════════════════════════════════════════════════════════════ -->
<div class="section">
  <div class="section-header">
    <span class="step-badge">Step 7</span>
    <h2>Cluster Profiling — Business Personas</h2>
  </div>
  <div class="cluster-grid">
    <div class="cluster-card c0">
      <h4>🟢 Cluster 0 — Khách Hàng Thân Thiết</h4>
      <ul>
        <li>Tenure dài nhất: <strong>14.9 tháng</strong></li>
        <li>OrderCount cao nhất: <strong>6.3 đơn</strong></li>
        <li>CashbackAmount cao nhất: <strong>$236.6</strong></li>
        <li>Churn rate: 13% | Churn prob: 0.10</li>
        <li>Hành động: VIP perks + Up-sell</li>
      </ul>
    </div>
    <div class="cluster-card c1">
      <h4>🔵 Cluster 1 — Mua Sắm Đều Đặn</h4>
      <ul>
        <li>Tenure: <strong>8.6 tháng</strong></li>
        <li>OrderCount: 3.3 đơn | Coupon: 2.1</li>
        <li>Risk score thấp nhất: <strong>0.23</strong></li>
        <li>Churn rate: 13% | Churn prob: 0.11</li>
        <li>Hành động: Freeship + Flash sale</li>
      </ul>
    </div>
    <div class="cluster-card c2">
      <h4>🟡 Cluster 2 — Người Mua Thụ Động</h4>
      <ul>
        <li>OrderCount thấp: <strong>1.9 đơn</strong></li>
        <li>Coupon ít: 1.0 | Complain thấp: 25.7%</li>
        <li>Vừa mua gần đây: 5 ngày</li>
        <li>Churn rate: 15% | Risk score: 0.39</li>
        <li>Hành động: Kích hoạt lại + Promotions</li>
      </ul>
    </div>
    <div class="cluster-card c3">
      <h4>🔴 Cluster 3 — Khách Hàng Sắp Rời Bỏ</h4>
      <ul>
        <li>Tenure ngắn nhất: <strong>7.9 tháng</strong></li>
        <li>Vừa mua: <strong>1.4 ngày</strong> ⚠ (nghịch lý!)</li>
        <li>Complain cao nhất: <strong>36%</strong></li>
        <li>Churn rate: <strong>27%</strong> | Prob: <strong>0.25</strong></li>
        <li>Hành động: Can thiệp ngay lập tức</li>
      </ul>
    </div>
  </div>
  <div class="alert alert-danger">🚨 <strong>Nghịch lý Cluster 3:</strong> DaySinceLastOrder = 1.4 ngày (vừa mua nhất) nhưng churn probability cao nhất (0.25). Đây là dấu hiệu "mua xong rồi bỏ" — cần can thiệp trong vài giờ.</div>
  {img_tag("07_cluster_profiling", "Cluster Profiling — Phân bố theo features")}
</div>

<!-- ══ STEP 8 ══════════════════════════════════════════════════════════════ -->
<div class="section">
  <div class="section-header">
    <span class="step-badge">Step 8</span>
    <h2>Risk Scoring</h2>
  </div>
  <div class="formula">
    <div class="eq">RawScore = 0.7 × P(churn) + 0.3 × (CashbackAmount / OrderCount)</div>
    <br>
    <div class="eq">RiskScore = (RawScore − RawScore_min) / (RawScore_max − RawScore_min)</div>
  </div>
  <table>
    <thead><tr><th>Risk Category</th><th>Ngưỡng RiskScore</th><th>Số KH</th><th>Tỷ lệ</th></tr></thead>
    <tbody>
      <tr><td>🟢 Very Low</td><td>≤ 0.1403</td><td>226</td><td>20.1%</td></tr>
      <tr><td>🟡 Low</td><td>0.1403 – 0.2546</td><td>225</td><td>20.0%</td></tr>
      <tr><td>🟠 Medium</td><td>0.2546 – 0.3438</td><td>225</td><td>20.0%</td></tr>
      <tr><td>🔴 High</td><td>0.3438 – 0.4864</td><td>225</td><td>20.0%</td></tr>
      <tr><td>🚨 Critical</td><td>&gt; 0.4864</td><td>225</td><td>20.0%</td></tr>
    </tbody>
  </table>
  <div class="risk-scale">
    <div class="risk-chip" style="background:rgba(34,197,94,0.2);color:#86efac;">🟢 Very Low</div>
    <div class="risk-chip" style="background:rgba(132,204,22,0.2);color:#bef264;">🟡 Low</div>
    <div class="risk-chip" style="background:rgba(245,158,11,0.2);color:#fcd34d;">🟠 Medium</div>
    <div class="risk-chip" style="background:rgba(249,115,22,0.2);color:#fdba74;">🔴 High</div>
    <div class="risk-chip" style="background:rgba(239,68,68,0.3);color:#fca5a5;">🚨 Critical</div>
  </div>
</div>

<!-- ══ STEP 9 ══════════════════════════════════════════════════════════════ -->
<div class="section">
  <div class="section-header">
    <span class="step-badge">Step 9</span>
    <h2>Kaplan-Meier Survival Analysis</h2>
  </div>
  <div class="formula">
    <div class="eq">Log-rank χ² = 40.42  |  p-value = 8.69 × 10⁻⁹</div>
  </div>
  <div class="alert alert-success">✅ Rất có ý nghĩa thống kê (p &lt;&lt; 0.001) — 4 clusters có survival curve khác biệt rõ ràng.</div>
  <table>
    <thead><tr><th>Cluster</th><th>Đường cong</th><th>Ý nghĩa</th></tr></thead>
    <tbody>
      <tr><td>🟢 0 — Thân Thiết</td><td>Giảm chậm nhất</td><td>Gắn bó lâu dài, tenure cao</td></tr>
      <tr><td>🔵 1 — Đều Đặn</td><td>Giảm chậm</td><td>Ổn định, ít rủi ro</td></tr>
      <tr><td>🟡 2 — Thụ Động</td><td>Giảm trung bình</td><td>Cần theo dõi</td></tr>
      <tr><td>🔴 3 — Sắp Rời</td><td><strong>Giảm nhanh nhất</strong></td><td>Rời sớm trong vòng đời</td></tr>
    </tbody>
  </table>
  {img_tag("08_km_by_cluster", "Kaplan-Meier Survival Curves by Cluster (log-rank p=8.69e-9)")}
</div>

<!-- ══ CONCLUSION ══════════════════════════════════════════════════════════ -->
<div class="section">
  <div class="section-header">
    <h2>✅ Kết Luận — Chiến Lược Can Thiệp</h2>
  </div>
  <h3>Ma Trận Can Thiệp Ưu Tiên</h3>
  <table>
    <thead><tr><th>Risk</th><th>Cluster</th><th>Ưu tiên</th><th>Hành động</th></tr></thead>
    <tbody>
      <tr class="highlight"><td>🚨 Critical</td><td>🔴 3 — Sắp Rời</td><td><strong>P1 — Ngay lập tức</strong></td><td>📞 Gọi điện + Xử lý khiếu nại + Cashback voucher</td></tr>
      <tr class="highlight"><td>🚨 Critical</td><td>🟢 0 — VIP</td><td><strong>P1 — Giữ chân VIP</strong></td><td>🎁 Liên hệ cá nhân + Quà tri ân + Up-sell</td></tr>
      <tr><td>🚨 Critical</td><td>🟡 2 / 🔵 1</td><td>P1 — Kích hoạt lại</td><td>💸 Voucher lớn + Gợi ý cá nhân hoá</td></tr>
      <tr><td>🔴 High</td><td>🔴 3</td><td>P2 — Trong 24–48h</td><td>⚠️ Email thăm dò + Mã giảm giá</td></tr>
      <tr><td>🔴 High</td><td>🟢 0</td><td>P2 — VIP Alert</td><td>🎯 Flash sale độc quyền VIP</td></tr>
      <tr><td>🔴 High</td><td>🔵 1 / 🟡 2</td><td>P2 — Trong 48h</td><td>💳 Freeship + Promo theo category</td></tr>
      <tr><td>🟠 Medium</td><td>Bất kỳ</td><td>P3 — Tuần tới</td><td>📧 Email cá nhân hoá + Cross-sell</td></tr>
      <tr><td>🟡 Low</td><td>Bất kỳ</td><td>P4 — Tháng tới</td><td>📊 Monitor + Newsletter</td></tr>
      <tr><td>🟢 Very Low</td><td>Bất kỳ</td><td>P5 — Duy trì</td><td>✅ Chăm sóc tiêu chuẩn</td></tr>
    </tbody>
  </table>

  <h3>Điều Chỉnh Theo Tenure (Survival)</h3>
  <table>
    <thead><tr><th>Tenure</th><th>Giai đoạn</th><th>Điều chỉnh chiến lược</th></tr></thead>
    <tbody>
      <tr><td>0–3 tháng</td><td>Nguy hiểm nhất</td><td>Ưu tiên onboarding + Welcome voucher ngay sau mua đầu</td></tr>
      <tr><td>3–12 tháng</td><td>Quyết định gắn bó</td><td>Loyalty program + Reward tích điểm</td></tr>
      <tr><td>&gt;12 tháng</td><td>Khách VIP</td><td>Retention qua Up-sell + Đặc quyền VIP</td></tr>
    </tbody>
  </table>

  <div class="alert alert-danger">
    🚨 <strong>Nhóm nguy hiểm tuyệt đối:</strong> Critical Risk + Cluster 3 + Tenure &lt; 3 tháng<br>
    → Tất cả dấu hiệu xấu đồng thời: churn prob 0.25, phàn nàn 36%, tenure ngắn, survival giảm nhanh nhất.<br>
    → <strong>Can thiệp ngay trong vài giờ.</strong>
  </div>
</div>

<!-- ══ ARTIFACTS ══════════════════════════════════════════════════════════ -->
<div class="section">
  <div class="section-header">
    <h2>📁 Artifacts</h2>
  </div>
  <table>
    <thead><tr><th>File</th><th>Nội dung</th></tr></thead>
    <tbody>
      <tr><td><code>data/E_Commer_Data/model_comparison.csv</code></td><td>Metrics 6 models</td></tr>
      <tr><td><code>data/E_Commer_Data/cluster_summary.csv</code></td><td>Personas + actions 4 clusters</td></tr>
      <tr><td><code>data/E_Commer_Data/targeted_action_list.csv</code></td><td>Toàn bộ KH + RiskScore + action</td></tr>
      <tr><td><code>models/E_Commer_Data/xgb_clust_model.pkl</code></td><td>Model production (XGB-Clust)</td></tr>
      <tr><td><code>models/E_Commer_Data/risk_score_meta.pkl</code></td><td>Ngưỡng phân vị Risk Score</td></tr>
      <tr><td><code>models/E_Commer_Data/kmeans_fe_model.pkl</code></td><td>K-Means K=4</td></tr>
    </tbody>
  </table>
  <p style="color:var(--muted);font-size:0.8rem;margin-top:2rem;text-align:center;">
    Churn Alert Pipeline — E_Commer_Data | Log-rank p = 8.69×10⁻⁹ | XGB-Clust ROC-AUC = 0.979
  </p>
</div>

</div><!-- /container -->
</body>
</html>
"""

with open(HTML_OUT, "w", encoding="utf-8") as f:
    f.write(html)

print(f"✅ Saved: {HTML_OUT}")
print(f"   Open in browser → File → Print → Save as PDF")
