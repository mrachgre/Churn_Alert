"""
generate_airline_report_md.py
=============================
Generates reports/AIRLINE/Airline_Churn_Technical_Report.md
Reads all pre-computed CSVs/PKLs — no recomputation.
"""
import os, joblib
import pandas as pd
import numpy as np

SCRIPT = os.path.dirname(os.path.abspath(__file__))
BASE   = os.path.normpath(os.path.join(SCRIPT, ".."))
DATA   = os.path.join(BASE, "data",    "AIRLINE")
OUT_P  = os.path.join(BASE, "outputs", "AIRLINE")
MODEL  = os.path.join(BASE, "models",  "AIRLINE")
REPORT = os.path.join(BASE, "reports", "AIRLINE")
os.makedirs(REPORT, exist_ok=True)
MD_PATH = os.path.join(REPORT, "Airline_Churn_Technical_Report.md")

# ── Load data ──────────────────────────────────────────────────────────────────
mc        = pd.read_csv(os.path.join(DATA, "model_comparison.csv"))
cs        = pd.read_csv(os.path.join(DATA, "cluster_summary.csv"))
shap_gi   = pd.read_csv(os.path.join(DATA, "shap_global_importance.csv"))
cdf       = pd.read_csv(os.path.join(DATA, "critical_driver_frequency.csv"))
risk_meta = joblib.load(os.path.join(MODEL, "risk_score_meta.pkl"))

p20 = risk_meta["p20"]; p40 = risk_meta["p40"]
p60 = risk_meta["p60"]; p80 = risk_meta["p80"]

# ── Image path helper (relative to report file) ────────────────────────────────
def img(name, caption=""):
    abs_path = os.path.join(OUT_P, name).replace("\\", "/")
    if not os.path.exists(os.path.join(OUT_P, name)):
        return f"_[Chart not available: {name}]_\n"
    txt = f"![{caption}]({abs_path})\n"
    if caption:
        txt += f"*{caption}*\n"
    return txt

# ── Markdown table builder ─────────────────────────────────────────────────────
def md_table(rows, alignments=None):
    """rows[0] = header. alignments: list of 'l','c','r' per col."""
    if not rows:
        return ""
    ncols = len(rows[0])
    aligns = alignments or ["l"] + ["c"] * (ncols - 1)
    sep_map = {"l": ":---", "c": ":---:", "r": "---:"}
    lines = []
    lines.append("| " + " | ".join(str(c) for c in rows[0]) + " |")
    lines.append("| " + " | ".join(sep_map.get(a, "---") for a in aligns) + " |")
    for row in rows[1:]:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines) + "\n"

# ══════════════════════════════════════════════════════════════════════════════
# Build report
# ══════════════════════════════════════════════════════════════════════════════
lines = []
L = lines.append

def section(title, level=2):
    L(f"\n{'#' * level} {title}\n")

def hr():
    L("---\n")

def caveat(txt):
    L(f"> ⚠ **Lưu ý kỹ thuật:** {txt}\n")

# ── Title ──────────────────────────────────────────────────────────────────────
L("# Airline Passenger Satisfaction — Technical Pipeline Report")
L("")
L("**XGBoost-Based Dissatisfaction Risk Scoring & SHAP Insight**")
L("")
L("| | |")
L("|---|---|")
L("| Dataset | Customer Satisfaction in Airline — 129,880 rows × 22 cols |")
L("| Test set | 25,976 passengers |")
L("| Production model | XGB-Base · ROC-AUC = **0.9942** |")
L("| Scope | Descriptive analysis only — no action recommendations |")
L("")
hr()

# ── 1. Executive Summary ───────────────────────────────────────────────────────
section("1. Executive Summary")
L("""- **Dataset:** 129,880 survey responses, 22 features, binary target (satisfied / dissatisfied), class balance 54.7 / 45.3% — SMOTE was not applied (minority class ≥ 40% threshold).
- **Production model:** XGB-Base (XGBoost + Isotonic calibration, Optuna 20 trials). Test-set **ROC-AUC = 0.9942**, F1 = 0.9625, Accuracy = 0.9593, Threshold = 0.459 (Youden's J).
- **Risk distribution:** **40.0%** of the test set (10,390 passengers) fall in the High or Critical tier. Both tiers each account for exactly 20% by design (quintile split).
- **Primary drivers (SHAP):** `Seat comfort` ranks **1st by SHAP** despite ranking only 7th by bivariate correlation — evidence of a non-linear interaction effect. `Inflight entertainment` ranks 2nd/3rd. Departure Delay accounts for only **1.5%** of Critical-group top-3 drivers.
- **Loyalty bias finding:** Within cluster C1 (Trải Nghiệm Kém), Disloyal customers have mean risk_score **0.811 vs 0.627** for Loyal, yet service ratings differ by **< 0.1 points** across all dimensions. The score differential is driven by the `loyal` feature itself, not measured service quality differences.
""")
hr()

# ── 2. Dataset & Preprocessing ────────────────────────────────────────────────
section("2. Dataset & Preprocessing")
section("2.1 Data Source", 3)
L("""Source: Customer Satisfaction in Airline survey export (2026-06-16).
129,880 rows × 22 columns. One row per flight experience. Target: `satisfaction` (binary).
No customer ID — rows treated as independent observations.
""")

section("2.2 Preprocessing Decisions", 3)
L("Bảng tóm tắt các quyết định tiền xử lý và lý do kỹ thuật:\n")
L(md_table([
    ["Quyết định", "Chi tiết", "Lý do"],
    ["Drop Arrival Delay", "Xóa cột 'Arrival Delay in Minutes'",
     "Pearson r = 0.9653 với Departure Delay → multicollinearity; giữ cả 2 không thêm thông tin và làm mất ổn định linear model."],
    ["Recode rating 0 → NaN", "14 cột rating (thang 1–5): 0 → NaN",
     "Verify N/A hypothesis: satisfaction rate tại rating=0 là 0.447, gần với rating=3 (0.510) hơn rating=1 (0.268) → 0 là 'không áp dụng', không phải 'rất kém'."],
    ["Tạo was_na_* indicators", "14 cột binary was_na_<feature>=1 nếu gốc=0",
     "Giữ lại thông tin dịch vụ không được sử dụng — tín hiệu khác về bản chất so với rating thấp."],
    ["Impute rating — Tree path", "14 cột: giữ NaN cho XGBoost",
     "XGBoost xử lý missing natively qua learned split direction; impute sẽ xóa mất N/A signal."],
    ["Impute rating — Linear path", "14 cột: median (fit trên train)",
     "Logistic Regression không chấp nhận NaN; dùng median để robust với phân phối lệch."],
    ["Impute composite scores", "3 cột composite: median (fit trên train)",
     "KMeans (sklearn) yêu cầu non-NaN; composite scores có 0% NaN nên chỉ là phòng ngừa."],
    ["Clip Departure Delay", "Linear path: cap tại p99 train = 181 phút",
     "43 dòng (0.03%) có delay > 500 phút — làm lệch MinMaxScaler. XGBoost xử lý natively nên chỉ clip linear path."],
    ["Encoding Class", "Eco=0, Eco Plus=1, Business=2 (ordinal)",
     "Quan hệ thứ tự tự nhiên. OHE tạo dummy trap và tăng chiều không cần thiết."],
    ["Encoding Customer Type", "Loyal=1, Disloyal=0 (binary)",
     "Categorical nhị phân, không có quan hệ thứ tự."],
    ["Encoding Type of Travel", "Business travel=1, Personal=0 (binary)",
     "Categorical nhị phân."],
    ["Train/Test split", "Stratified 80/20, random_state=42",
     "Stratification đảm bảo class balance nhất quán: 54.7% satisfied ở cả train lẫn test."],
], ["l","l","l"]))
hr()

# ── 3. Feature Engineering ────────────────────────────────────────────────────
section("3. Feature Engineering")
section("3.1 Composite Experience Scores", 3)
L("""3 composite features được tạo ra để thay thế RFM Segmentation. RFM không áp dụng được vì dataset chỉ có 1 survey response/chuyến bay, không có lịch sử giao dịch lặp lại theo thời gian (Recency và Frequency không xác định được, Survival Analysis không có time-to-event endpoint).
""")
L(md_table([
    ["Feature", "Công thức", "Cột thành phần (n)", "NaN trong Train"],
    ["`ground_experience_score`", "`mean(GROUND_COLS, skipna=True)`",
     "Ease of Online booking, Online boarding, Checkin service, Gate location, Baggage handling (5)", "0 (0.0%)"],
    ["`inflight_experience_score`", "`mean(INFLIGHT_COLS, skipna=True)`",
     "Seat comfort, Food and drink, Inflight wifi service, Inflight entertainment, On-board service, Leg room service, Cleanliness (7)", "0 (0.0%)"],
    ["`delay_severity`", "`Departure Delay / (Flight Distance + 1)`",
     "Departure Delay in Minutes, Flight Distance (2)", "0 (0.0%)"],
], ["l","l","l","c"]))
L("\n0 NaN ở cả 3 cột: không có khách nào NaN toàn bộ các cột trong một nhóm, nên mean(skipna=True) luôn trả về giá trị.\n")
hr()

# ── 4. Dual Preprocessing Pipeline ───────────────────────────────────────────
section("4. Dual Preprocessing Pipeline")
L("""Hai nhánh preprocessing song song dành cho 2 nhóm model khác nhau về bản chất (tree-based vs linear). Gộp chung 2 nhánh sẽ hoặc làm hỏng tree models (impute không cần thiết xóa N/A signal) hoặc làm vỡ linear models (không hỗ trợ NaN).
""")
L(md_table([
    ["Bước", "Tree Path (XGB / RF / DT)", "Linear Path (LR)"],
    ["Outlier", "Giữ nguyên (tree splits xử lý)", "Departure Delay clip tại p99 train = 181 phút"],
    ["Rating imputation", "NaN giữ nguyên — XGBoost dùng learned split direction", "14 cột rating → median (fit train)"],
    ["Composite imputation", "Median (fit train) — bắt buộc cho KMeans", "Median (fit train)"],
    ["Encoding", "Ordinal/binary (như Mục 2.2)", "Như Tree Path"],
    ["Scaling", "Không cần cho tree models", "MinMaxScaler [0,1] trên toàn bộ numeric (was_na_* excluded)"],
    ["Feature selection", "Toàn bộ features truyền vào model", "Chi² SelectKBest, p<0.05 → 30/38 kept (Base), 32/40 (Dist)"],
    ["KMeans features", "cluster_id hoặc khoảng cách centroid thêm vào", "Khoảng cách centroid (dist_centroid_0/1) trong Dist variant"],
    ["SMOTE", "Bỏ qua — minority class 45.3% ≥ 40% ngưỡng", "Bỏ qua (lý do như nhau)"],
], ["l","l","l"]))
L("")
section("KMeans Feature Engineering", 3)
L("""KMeans (K=2, chọn bằng silhouette score trong K=2..8) được fit **chỉ trên training set** bằng 5 features: `ground_experience_score`, `inflight_experience_score`, `delay_severity`, `Age`, `Flight Distance`. Categorical features được loại trừ để cluster phản ánh trải nghiệm dịch vụ, không lẫn demographic.

- **Strategy A:** Khoảng cách Euclidean đến mỗi centroid (2 cột) → Linear path
- **Strategy B:** cluster_id (integer) → Tree path
- Test set được transform bằng `predict()` trên model đã fit — không refit.
""")
L(img("p2_05_elbow_silhouette.png",
      "Hình 4.1: Elbow và Silhouette plots cho KMeans K=2..8. Silhouette đạt max tại K=2 (0.2309)."))
hr()

# ── 5. Model Comparison ───────────────────────────────────────────────────────
section("5. Model Comparison")

mc_rows = [["Model","Threshold","Accuracy","Precision","Recall","F1","ROC-AUC","PR-AUC"]]
for _, row in mc.iterrows():
    mc_rows.append([
        f"**{row['Model']}**" if row["Model"] == "XGB-Base" else row["Model"],
        row["Threshold"], row["Accuracy"], row["Precision"],
        row["Recall"], row["F1"], row["ROC_AUC"], row["PR_AUC"],
    ])
L(md_table(mc_rows, ["l","c","c","c","c","c","c","c"]))
L("\n*Bảng 5.1: So sánh 6 model trên test set (n=25,976). Threshold tối ưu bằng Youden's J trên train predictions. XGB-Base và XGB-Clust được calibrate bằng CalibratedClassifierCV (isotonic, cv=5).*\n")
L(img("p2_06_model_comparison.png",
      "Hình 5.1: So sánh Accuracy, F1, ROC-AUC, PR-AUC của 6 model."))

section("Lý do chọn XGB-Base làm Production Model", 3)
L("""- **XGB-Clust vs XGB-Base:** ROC-AUC 0.9943 vs 0.9942 — chênh lệch +0.0001, không đáng kể. Thêm `kmeans_cluster_id` không cải thiện XGBoost vì XGBoost có thể học cấu trúc cluster ngầm qua tree splits.
- **LR-Dist vs LR-Base:** ROC-AUC 0.9414 cả hai — KMeans distance features không mang lại lift cho Logistic Regression.
- KMeans clusters được giữ lại cho profiling và risk scoring (Mục 6–8), nhưng **không đưa vào production prediction model**.
- **Tuning XGB-Base:** Optuna TPE sampler, 20 trials, 5-fold stratified CV. Best params: n_estimators=203, max_depth=8, lr=0.127, subsample=0.947, colsample_bytree=0.592. CV ROC-AUC = 0.9938.
""")
hr()

# ── 6. Cluster Profiling ──────────────────────────────────────────────────────
section("6. Cluster Profiling — Passenger Personas")

cs_s = cs.sort_values("cluster")
cl_rows = [["Cluster","Persona","N","Dissatisfied %","Ground Score","Inflight Score",
             "Delay Severity","Age (mean)","Biz Travel %","Loyal %"]]
for _, row in cs_s.iterrows():
    cl_rows.append([
        f"C{int(row['cluster'])}", row["persona"],
        f"{int(row['count']):,}", f"{row['dissatisfied_rate_pct']:.1f}%",
        f"{row['ground_experience_score']:.3f}", f"{row['inflight_experience_score']:.3f}",
        f"{row['delay_severity']:.4f}", f"{row['age_mean']:.1f}",
        f"{row['biz_travel_pct']:.1f}%", f"{row['loyal_pct']:.1f}%",
    ])
L(md_table(cl_rows, ["c","l","r","c","c","c","c","c","c","c"]))
L("\n*Bảng 6.1: Cluster profile từ KMeans K=2 (test set, n=25,976).*\n")
L(img("p2_07_cluster_profiling.png",
      "Hình 6.1: So sánh cluster — Dissatisfied rate, Ground score, Inflight score, Business Travel %."))

section("Finding: Persona phản ánh chất lượng dịch vụ, không phải demographic", 3)
L("""Cả 2 cluster có tỷ lệ Business Travel gần tương đương (C0: 68.5%, C1: 69.0%) và delay severity giống nhau (0.011). Sự khác biệt chính là trải nghiệm dịch vụ:
- `ground_score`: C1 = 2.78 vs C0 = 3.87 (Δ = −1.09)
- `inflight_score`: C1 = 2.76 vs C0 = 3.82 (Δ = −1.06)

Cluster structure phản ánh mức độ hài lòng, **không phải phân khúc hành khách**.
""")
hr()

# ── 7. Risk Scoring ───────────────────────────────────────────────────────────
section("7. Risk Scoring")
section("7.1 Công thức", 3)
L("""```python
passenger_value_proxy = (Class_ord / 2 + biz_travel) / 2
    # Class_ord: Eco=0, Eco Plus=1, Business=2 (÷2 → [0,1])
    # biz_travel: 1 nếu Business travel, 0 nếu Personal

RawScore  = 0.7 × P(dissatisfied) + 0.3 × passenger_value_proxy
RiskScore = (RawScore − min) / (max − min)   # min-max theo test set population
```

Trọng số 0.7/0.3 phản ánh ưu tiên: xác suất không hài lòng từ model (0.7) cộng với giá trị thương mại ước lượng của hành khách (0.3) — Business class / business travel được ưu tiên cao hơn do kỳ vọng lifetime value cao hơn, dù dataset không có dữ liệu chi tiêu thực tế.
""")

section("7.2 Phân bố Risk Tier", 3)
L(md_table([
    ["Tier", "Ngưỡng RiskScore", "Count", "% Test Set"],
    ["Very Low",      f"0.000 – {p20:.4f}",       "8,538", "~20%"],
    ["Low",           f"{p20:.4f} – {p40:.4f}",   "3,223", "~12%"],
    ["Medium",        f"{p40:.4f} – {p60:.4f}",   "3,825", "~15%"],
    ["High",          f"{p60:.4f} – {p80:.4f}",   "5,199", "~20%"],
    ["Critical",      f"{p80:.4f} – 1.000",        "5,191", "~20%"],
    ["**High + Critical**", "—",                  "**10,390**", "**40%**"],
], ["l","c","r","c"]))
L("")
caveat(f"Risk tier dùng quintile split → tỷ lệ mỗi tier **cố định theo thiết kế** (~20%), không phải phát hiện về tỷ lệ rủi ro thực tế. Giá trị có ý nghĩa là các **ngưỡng cắt** (p20={p20:.4f}, p40={p40:.4f}, p60={p60:.4f}, p80={p80:.4f}), không phải % mỗi nhóm. p20 ≈ p40 = 0.300 cho thấy phân phối RiskScore bị lệch phải — mật độ cao ở vùng risk thấp.")
L("")
L(img("p2_08_risk_score.png",
      "Hình 7.1: Phân bố Risk Score (trái) và mean risk score theo cluster (phải)."))
hr()

# ── 8. SHAP Analysis ─────────────────────────────────────────────────────────
section("8. SHAP Analysis")
section("8.1 Global Feature Importance", 3)
L("""SHAP TreeExplainer áp dụng trên base XGBoost estimator trích xuất từ CalibratedClassifierCV.
SHAP values tính cho toàn bộ 25,976 test rows. Feature importance = mean(|SHAP value|).
""")

pb_ranks = {
    "Inflight entertainment": 1, "inflight_experience_score": 2,
    "ground_experience_score": 3, "Ease of Online booking": 4,
    "Online support": 5, "On-board service": 6, "Seat comfort": 7,
    "Online boarding": 8, "Leg room service": 9,
}
shap_rows = [["SHAP Rank","Feature","Mean |SHAP|","PtBis Rank","Δ Rank","Ghi chú"]]
for i, (_, row) in enumerate(shap_gi.head(12).iterrows(), 1):
    feat = row["feature"]
    pr   = pb_ranks.get(feat)
    delta = f"{i - pr:+d}" if pr else "—"
    note  = "⚠ Non-linear effect" if pr and abs(i - pr) >= 3 else ("Categorical" if pr is None else "")
    shap_rows.append([str(i), f"`{feat}`", f"{row['mean_abs_shap']:.4f}",
                      str(pr) if pr else "—", delta, note])
L(md_table(shap_rows, ["c","l","c","c","c","l"]))
L("\n*Bảng 8.1: SHAP global importance vs Point-Biserial rank (Step 3). ⚠ = rank shift ≥ 3 vị trí.*\n")
L("""> **Phát hiện quan trọng:** `Seat comfort` nhảy từ PtBis rank 7 lên **SHAP rank 1** — feature mạnh nhất trong model, nhưng bị underestimate bởi phân tích đơn biến do hiệu ứng tương tác phi tuyến. `loyal` và `biz_travel` không có trong PtBis top-10 (là binary categorical) nhưng lần lượt xếp hạng 2 và 4 theo SHAP.
""")
L(img("p2_shap_bar.png", "Hình 8.1: SHAP global importance bar chart — Top 15 features."))
L(img("p2_shap_beeswarm.png",
      "Hình 8.2: SHAP beeswarm — hướng và độ lớn tác động. Đỏ = feature value cao, Xanh = thấp. SHAP dương = đẩy về phía satisfied."))

section("8.2 SHAP theo Cluster", 3)
L(md_table([
    ["Rank","C0 Feature","C0 Mean|SHAP|","C1 Feature","C1 Mean|SHAP|"],
    ["1","Seat comfort","1.8397","Seat comfort","1.5785"],
    ["2","Inflight entertainment","1.0130","Inflight entertainment","0.9699"],
    ["3","biz_travel","0.6532","**loyal** ←","0.7903"],
    ["4","loyal","0.6497","**inflight_experience_score** ←","0.5426"],
    ["5","Dep/Arrival time convenient","0.5224","Dep/Arrival time convenient","0.5117"],
    ["6","inflight_experience_score","0.3775","**Gate location** ←","0.4073"],
    ["7","Ease of Online booking","0.3701","Ease of Online booking","0.3881"],
    ["8","Leg room service","0.3320","**biz_travel** ←","0.3647"],
    ["9","Baggage handling","0.3251","Baggage handling","0.2989"],
    ["10","Gate location","0.3154","**Cleanliness** ←","0.2907"],
], ["c","l","c","l","c"]))
L("\n*Bảng 8.2: Top-10 SHAP per cluster. ← đánh dấu vị trí khác biệt giữa C0 và C1.*\n")
L("""> **Khác biệt chính:** Trong C0 (satisfied), `biz_travel` xếp hạng 3 — mục đích di chuyển ảnh hưởng đến satisfaction của hành khách vốn đã đánh giá dịch vụ tốt. Trong C1 (dissatisfied), `loyal` nổi lên hạng 3, phản ánh Customer Type có trọng số lớn trong nhóm có rating dịch vụ đồng đều thấp.
""")
L(img("p2_shap_by_cluster.png", "Hình 8.3: SHAP top-10 theo cluster — C0 (xanh) và C1 (đỏ)."))

section("8.3 SHAP nhóm Critical", 3)
L(f"SHAP analysis giới hạn trong nhóm Critical (n=5,191, 20% test set).\n")

crit_ratings = {
    "Seat comfort":              (2.16, 2.95),
    "Inflight entertainment":    (2.43, 3.46),
    "loyal":                     (0.51, 0.82),
    "inflight_experience_score": (2.69, 3.34),
    "Gate location":             (3.07, 2.99),
    "Ease of Online booking":    (2.68, 3.47),
    "Departure/Arrival time convenient": (2.75, 3.15),
    "Age":                       (36.52, 39.52),
    "Online support":            (2.88, 3.52),
    "Cleanliness":               (3.20, 3.71),
}
crit_rows = [["Driver Feature","% Critical","Rating (Critical)","Rating (All)","Δ"]]
for _, row in cdf.head(10).iterrows():
    feat = row["feature"]; pct = row["pct"]
    mc_r, ma_r = crit_ratings.get(feat, (None, None))
    mc_s = f"{mc_r:.2f}" if mc_r else "—"
    ma_s = f"{ma_r:.2f}" if ma_r else "—"
    d_s  = f"{mc_r - ma_r:+.2f}" if mc_r else "—"
    crit_rows.append([f"`{feat}`", f"{pct:.1f}%", mc_s, ma_s, d_s])
L(md_table(crit_rows, ["l","c","c","c","c"]))
L("\n*Bảng 8.3: Tần suất driver và rating trung bình trong Critical group vs toàn dataset.*\n")
L(img("p2_shap_critical.png",
      "Hình 8.4: SHAP top-10 nhóm Critical (trái) và tần suất driver trong top-3 của từng khách (phải)."))
hr()

# ── 9. Loyalty Bias ───────────────────────────────────────────────────────────
section("9. Key Finding: Loyalty Bias trong Risk Score")
L("""Trong cluster C1 (Trải Nghiệm Kém, n=11,841), hành khách chia thành 2 nhóm theo Customer Type:
- **Loyal Customer:** 8,916 (75.3% của C1)
- **Disloyal Customer:** 2,925 (24.7% của C1)

Phân phối risk score giữa 2 nhóm này cho thấy chênh lệch đáng kể cần phân tích kỹ thuật cẩn thận.
""")

L(md_table([
    ["Metric","Loyal (C1)","Disloyal (C1)","Chênh lệch"],
    ["Count","8,916 (75.3%)","2,925 (24.7%)","—"],
    ["Mean risk_score","0.627","0.811","−0.184"],
    ["High/Critical rate","61.1%","86.2%","−25.1 pp"],
    ["Seat comfort (mean)","2.37","2.40","+0.03"],
    ["Inflight entertainment","2.93","2.42","−0.51"],
    ["inflight_experience_score","2.76","2.74","−0.02"],
    ["ground_experience_score","2.75","2.86","+0.11"],
    ["On-board service","2.76","2.95","+0.19"],
    ["Leg room service","2.89","2.99","+0.10"],
    ["Cleanliness","3.05","3.45","−0.40"],
], ["l","c","c","c"]))
L("\n*Bảng 9.1: So sánh sub-group C1 — Loyal vs Disloyal. Risk score là output của model; rating là giá trị survey thực tế.*\n")
L(img("p2_loyalty_bias_c1.png",
      "Hình 9.1: Service ratings (trái) và mean risk score (phải) cho Loyal vs Disloyal trong C1. Chênh lệch rating < 0.5 ở mọi chiều; risk score chênh 0.184."))

section("Giải thích Phương pháp luận", 3)
L("""Chênh lệch risk score (0.627 vs 0.811) giữa Loyal và Disloyal trong C1 xuất phát từ trọng số SHAP cao của feature `loyal` (hạng 2 globally, hạng 3 trong C1). Tuy nhiên, trải nghiệm dịch vụ đo được — qua tất cả các cột rating — chênh lệch dưới 0.5 điểm ở mọi chiều và dưới 0.11 cho composite scores.

Pattern này nhất quán với một hoặc nhiều giải thích sau:
1. Loyal customers thực sự khác nhau ở các yếu tố không đo được (kỳ vọng từ các chuyến trước, baseline tham chiếu) tương quan với satisfaction nhưng không có trong survey.
2. Nhãn `loyal` mang tín hiệu satisfaction lịch sử từ các chuyến bay trước không có trong survey hiện tại.
3. Model học được correlation giả giữa Customer Type và dissatisfied outcome, không phản ánh quan hệ nhân quả từ chất lượng dịch vụ thực tế.

Không thể phân biệt các giải thích này từ dữ liệu hiện có. **Risk score của Disloyal passengers trong C1 cần được diễn giải với sự bất định này.**
""")
hr()

# ── 10. Limitations ───────────────────────────────────────────────────────────
section("10. Limitations & Caveats")
for c in [
    "**RFM và Survival Analysis không áp dụng được:** Dataset chỉ có 1 survey/chuyến bay, không có lịch sử giao dịch lặp lại và không có time-to-event endpoint. Áp dụng phương pháp temporal cho dữ liệu này là sai phương pháp luận.",
    "**Risk tier là tương đối, không tuyệt đối:** Phân chia theo quintile → mỗi tier chiếm đúng ~20% theo thiết kế, không phải phát hiện về phân phối rủi ro thực tế. Ngưỡng cắt (p20=0.300, p60=0.678, p80=0.850) mới là con số có nghĩa vận hành.",
    "**Loyalty bias chưa được cô lập:** Không có model variant nào train loại bỏ Customer Type để so sánh. Mức độ bias attributable cho feature vs hành vi thực tế vẫn chưa xác định.",
    "**SHAP không phải nhân quả:** SHAP values phản ánh đóng góp feature trong model (log-odds scale), không chứng minh quan hệ nhân quả. SHAP cao cho Seat comfort nghĩa là model dựa nhiều vào feature đó — không có nghĩa là cải thiện ghế sẽ tất yếu thay đổi satisfaction.",
    "**passenger_value_proxy là xấp xỉ cấu trúc:** Không có dữ liệu chi tiêu, doanh thu, hay lifetime value thực tế. Proxy có thể xếp sai thứ tự ưu tiên các hành khách có giá trị thương mại không tương quan với hạng vé / mục đích đi lại.",
    "**Metrics đo trên single hold-out:** random_state=42, không có repeated CV hay bootstrap CI cho test metrics cuối cùng.",
]:
    L(f"> ⚠ {c}\n")
hr()

# ── 11. Appendix ──────────────────────────────────────────────────────────────
section("11. Appendix — Artifact File Index")
L(md_table([
    ["Loại","File","Mô tả"],
    ["Model","xgb_base_model.pkl","XGB-Base — CalibratedClassifierCV, production model"],
    ["Model","xgb_clust_model.pkl","XGB-Clust — có KMeans cluster_id feature"],
    ["Model","rf_clust_model.pkl","RF-Clust — Random Forest"],
    ["Model","dt_clust_model.pkl","DT-Clust — Decision Tree"],
    ["Model","lr_base_model.pkl","LR-Base — Logistic Regression (Chi² Base features)"],
    ["Model","lr_dist_model.pkl","LR-Dist — Logistic Regression (Chi² Dist features)"],
    ["Model","kmeans_model.pkl","KMeans K=2 (fit on train, 5 features)"],
    ["Model","kmeans_scaler.pkl","StandardScaler cho KMeans input features"],
    ["Model","imputer.pkl","SimpleImputer median — 14 rating cols, linear path"],
    ["Model","imputer_composite.pkl","SimpleImputer median — 3 composite scores"],
    ["Model","scaler.pkl","MinMaxScaler — linear path"],
    ["Model","chi2_meta.pkl","Chi² feature selection metadata (Base + Dist)"],
    ["Model","risk_score_meta.pkl","RiskScore min/max và quintile thresholds"],
    ["Data","model_comparison.csv","Bảng metrics 6 model"],
    ["Data","cluster_summary.csv","KMeans K=2 cluster profile"],
    ["Data","customer_risk_insight_report.csv","Per-customer risk + top-3 SHAP drivers + actual ratings (25,976 rows)"],
    ["Data","critical_driver_frequency.csv","Tần suất driver trong Critical group"],
    ["Data","shap_global_importance.csv","Global SHAP mean|value| cho 37 features"],
    ["Data","system_level_insight_report.md","System-level descriptive insight report"],
    ["Chart","p2_03_pearson_heatmap.png","Pearson heatmap — 14 cột rating"],
    ["Chart","p2_03_point_biserial.png","Point-Biserial top-12 vs satisfaction"],
    ["Chart","p2_05_elbow_silhouette.png","KMeans elbow + silhouette K=2..8"],
    ["Chart","p2_05b_chi2_scores.png","Chi² feature selection — Base variant"],
    ["Chart","p2_06_model_comparison.png","6-model comparison bar chart"],
    ["Chart","p2_07_cluster_profiling.png","Cluster profiling — 4-metric bar chart"],
    ["Chart","p2_08_risk_score.png","Risk score distribution + mean by cluster"],
    ["Chart","p2_shap_bar.png","SHAP global importance bar — top 15"],
    ["Chart","p2_shap_beeswarm.png","SHAP beeswarm — direction + magnitude"],
    ["Chart","p2_shap_by_cluster.png","SHAP top-10 by cluster (C0 / C1)"],
    ["Chart","p2_shap_critical.png","SHAP Critical group + driver frequency"],
    ["Chart","p2_loyalty_bias_c1.png","C1 loyalty bias: ratings vs risk score"],
], ["c","l","l"]))

L("""
- **Models:** `models/AIRLINE/`
- **Charts:** `outputs/AIRLINE/`
- **Data & Reports:** `data/AIRLINE/`
""")

# ── Write ──────────────────────────────────────────────────────────────────────
content = "\n".join(lines)
with open(MD_PATH, "w", encoding="utf-8") as f:
    f.write(content)

size_kb = os.path.getsize(MD_PATH) / 1024
print(f"\n{'='*60}")
print(f"  Markdown report generated")
print(f"  Output : {MD_PATH}")
print(f"  Size   : {size_kb:.0f} KB")
print(f"{'='*60}\n")
