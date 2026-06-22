# 🚨 Churn Alert — Báo Cáo Kỹ Thuật Pipeline

> **Dataset:** E-Commerce Customer Churn (Kaggle)  
> **Mẫu:** 5,630 khách hàng × 20 đặc trưng  |  **Churn rate:** 16.8%  
> **Mục tiêu:** Dự đoán khả năng rời bỏ, phân nhóm hành vi, và ra chiến lược can thiệp ưu tiên theo từng khách hàng.

---

## Tổng Quan Pipeline

```
Raw Data (Excel)
     │
     ▼
[Step 1]  Data Cleaning
     │
     ▼
[Step 2]  Missing Data Analysis
     │
     ▼
[Step 3]  EDA (Exploratory Data Analysis)
     │
     ▼
[Step 4]  RFM Segmentation (SQLite NTILE)
     │
     ├──────────────────────────────────────────────────────┐
     ▼                                                      ▼
[Step 5]  Preprocessing (Tree path)              [Step 5b]  Preprocessing (Linear path)
  KMeans Feature Engineering                       IQR Clip + MinMaxScale + Chi²(χ²)
     │                                                      │
     └────────────────────┬─────────────────────────────────┘
                          ▼
              [Step 6]  Modelling
     XGB-Base | XGB-Clust | RF-Clust | DT-Clust | LR-Base | LR-Dist
                          │
                          ▼
              [Step 7]  Cluster Profiling (Personas)
                          │
                          ▼
              [Step 8]  Risk Scoring + Action Matrix
                          │
                          ▼
              [Step 9]  Kaplan-Meier Survival Analysis
                          │
                          ▼
          🎯  RiskScore + Cluster + Survival → Chiến lược can thiệp
```

---

## Step 1 — Data Cleaning

**Kỹ thuật:**
- Load Excel (sheet `E Comm`) → 5,630 rows × 20 cột.
- Chuyển `CityTier`, `Complain` → kiểu `object` (categorical).
- Chuẩn hóa typo trong categorical:

| Trước | Sau |
|-------|-----|
| `"Mobile Phone"`, `"Phone"` | `"Mobile"` |
| `"CC"` | `"Credit Card"` |
| `"COD"` | `"Cash on Delivery"` |

**Kết quả:** `ecommerce_churn_clean.csv` — 5,630 rows, không duplicate, category nhất quán.

---

## Step 2 — Missing Data Analysis

**Kỹ thuật:**
- Visualize missingness matrix bằng `missingno`.
- Phân loại MCAR / MAR / MNAR: so sánh churn rate giữa nhóm có và không có missing — nếu |Δchurn| > 5% → **MAR/MNAR**.

**Kết quả:**

| Cột | Missing | Loại |
|-----|---------|-------|
| DaySinceLastOrder | 5.45% | MCAR |
| OrderAmountHikeFromlastYear | 4.71% | **MAR/MNAR** |
| Tenure | 4.69% | **MAR/MNAR** |
| OrderCount | 4.58% | **MAR/MNAR** |
| CouponUsed | 4.55% | **MAR/MNAR** |
| HourSpendOnApp | 4.53% | **MAR/MNAR** |
| WarehouseToHome | 4.46% | **MAR/MNAR** |

> [!NOTE]
> Phần lớn missing là MAR/MNAR → điền bằng **KNN Imputer (k=5)** thay vì mean/median để bảo toàn cấu trúc tương quan giữa các biến.

---

## Step 3 — Exploratory Data Analysis

**Kỹ thuật & phát hiện chính:**

| Phân tích | Phương pháp | Phát hiện |
|-----------|------------|-----------|
| Phân bố nhãn | Bar + Pie | **Mất cân bằng:** 83.2% retained vs 16.8% churn → cần SMOTE |
| Churn theo Tenure | Bar binned | Churn cao nhất ở nhóm 0–3 tháng (~35%) |
| Churn theo Satisfaction | Bar | Score 1 → churn ≈25%; Score 5 → churn ≈10% |
| Pearson Heatmap | Numeric corr | `Tenure` tương quan âm mạnh với churn |
| Cramér's V | Categorical assoc | `Complain` liên kết mạnh nhất với churn |
| Point-Biserial | Numeric vs Churn | Top: `Tenure`, `CashbackAmount`, `SatisfactionScore` |

![Class Distribution](C:\Users\pham tien dai\.gemini\antigravity-ide\brain\bda7326d-1074-4947-b331-e5010b5a9cc3\03_class_distribution.png)

---

## Step 4 — RFM Segmentation

**Kỹ thuật:**  
Dùng **SQLite in-memory + NTILE(5)** window function để chấm điểm R/F/M từ 1–5.

| Chiều | Mapping | Hướng điểm |
|-------|---------|-----------|
| **R** (Recency) | `DaySinceLastOrder` | Ít ngày → điểm cao |
| **F** (Frequency) | `OrderCount` | Nhiều đơn → điểm cao |
| **M** (Monetary) | `CashbackAmount` | Nhiều cashback → điểm cao |

**Quy tắc phân đoạn:**

| Segment | Điều kiện | Số KH |
|---------|-----------|-------|
| Champions | F ≥ 4 **và** M ≥ 4 | 1,505 |
| Recent Customers | R ≥ 4 **và** F < 3 | 1,319 |
| Hibernating | Còn lại | 1,270 |
| Loyal | F ≥ 3 **và** M ≥ 3 | 1,244 |
| At Risk | R ≤ 2 **và** F ≥ 3 | 292 |

> RFM standalone ROC-AUC = **0.5503** — yếu độc lập, nhưng `rfm_segment` + `rfm_total` là input features quan trọng cho model.

![RFM Churn Rate](C:\Users\pham tien dai\.gemini\antigravity-ide\brain\bda7326d-1074-4947-b331-e5010b5a9cc3\04_rfm_churn_rate.png)

---

## Step 5 — Preprocessing (Đường Tree Models)

**Kỹ thuật:**

**① Feature Engineering** — 3 đặc trưng tổng hợp:
```python
device_per_tenure  = NumberOfDeviceRegistered / (Tenure + 1)  # mật độ thiết bị
cashback_per_order = CashbackAmount / (OrderCount + 1)        # giá trị mỗi đơn
inactivity_ratio   = DaySinceLastOrder / (Tenure + 1)         # tỷ lệ không hoạt động
```

**② Train/Test Split** — Stratified, test_size=0.2, random_state=42  
Train: **4,504** | Test: **1,126**

**③ OHE + StandardScaler + KNN Imputer** — fit trên train only → transform cả hai.

**④ SMOTE** — chỉ trên training set:

| | Class 0 | Class 1 |
|-|---------|---------|
| Trước SMOTE | 3,746 | 758 |
| Sau SMOTE | 3,746 | 3,746 |

**⑤ K-Means Feature Engineering:**
- Elbow + Silhouette → **K = 4** (silhouette score = 0.1352).
- Fit trên **4,504 pre-SMOTE** training samples — không có data leakage vào test/SMOTE.
- Tạo 2 chiến lược feature song song:
  - **Strategy A** (`X_train_dist`): 4 cột khoảng cách Euclidean đến centroid.
  - **Strategy B** (`X_train_clust`): 1 cột `kmeans_cluster_id`.

> [!IMPORTANT]
> KMeans **chỉ được fit một lần** trên training data. Test set dùng `km.predict()` — tuyệt đối không có data leakage.

---

## Step 5b — Preprocessing (Đường Linear Models)

> [!NOTE]
> Linear models nhạy cảm với outlier và scale → cần pipeline riêng biệt, không dùng chung với tree models.

**Kỹ thuật (theo thứ tự, chỉ fit trên train):**

| Bước | Kỹ thuật | Lý do |
|------|---------|-------|
| Outlier clip | IQR × 1.5 (clip, không drop rows) | Giảm ảnh hưởng giá trị cực |
| OHE | Reuse `encoder.pkl` từ Step 5 | Đảm bảo cùng category mapping |
| MinMaxScaler | Đưa về [0, 1] | Chi² yêu cầu giá trị không âm |
| KNN Imputer | k=5 | Xử lý missing sau scaling |
| **Chi² Selection** | `SelectKBest(chi2)`, p < 0.05 | Giảm chiều, loại noise |
| SMOTE | Sau selection | Cân bằng nhãn |

**Kết quả Chi-Square Selection:**
- **Base variant**: **23 / 35 features** giữ lại (p < 0.05)
- **Dist variant**: **25 / 39 features** (thêm 4 KMeans distance features)

**Top features được chọn (theo χ² score):**  
`device_per_tenure`, `Complain_1`, `Tenure`, `inactivity_ratio`, `DaySinceLastOrder`, `SatisfactionScore`, `CashbackAmount`, `WarehouseToHome`, `NumberOfDeviceRegistered`, `MaritalStatus_Single/Married`, `PreferedOrderCat_*`, `rfm_segment_*`...

![Chi-Square Feature Selection](C:\Users\pham tien dai\.gemini\antigravity-ide\brain\bda7326d-1074-4947-b331-e5010b5a9cc3\05b_chi2_scores.png)

---

## Step 6 — Modelling

**6 models được huấn luyện, tuned và so sánh:**

### XGBoost Base — Model benchmark
- **Bayesian Optimization (Optuna)**: 40 trials, CV=5-fold stratified.
- Best: `n_estimators=463, max_depth=8, lr=0.0834, subsample=0.679`
- Hiệu chỉnh xác suất: `CalibratedClassifierCV(cv=5)`.
- Optimal threshold: **0.119** (Youden's J).

### Logistic Regression (Base / Dist)
- Input từ Step 5b — chi²-selected, MinMaxScaled.
- `GridSearchCV(C ∈ {0.001, 0.01, 0.1, 1, 10, 100})`.

### Tree variants (DT / RF / XGB-Clust)
- Input từ Step 5 + `kmeans_cluster_id`.
- XGB-Clust reuse Optuna params từ XGB-Base.

### Kết quả so sánh

| Model | Accuracy | F1 | ROC-AUC | PR-AUC | Threshold |
|-------|----------|----|---------|--------|-----------|
| **XGB-Base** | **0.9813** | **0.9446** | 0.9689 | 0.9333 | 0.119 |
| RF-Clust | 0.9725 | 0.9246 | **0.9978** | **0.9891** | 0.345 |
| **XGB-Clust ⭐** | 0.9547 | 0.8747 | 0.9790 | 0.9501 | 0.136 |
| DT-Clust | 0.7993 | 0.6103 | 0.9177 | 0.7436 | 0.083 |
| LR-Dist | 0.8446 | 0.6187 | 0.8699 | 0.6802 | 0.568 |
| LR-Base | 0.8224 | 0.5968 | 0.8679 | 0.6761 | 0.497 |

⭐ **Model production: XGB-Clust** — tích hợp cluster context, ROC-AUC 0.979, cân bằng tốt giữa Precision (82%) và Recall (93.7%).

![Model Comparison](C:\Users\pham tien dai\.gemini\antigravity-ide\brain\bda7326d-1074-4947-b331-e5010b5a9cc3\06b_model_comparison.png)

![ROC Curve](C:\Users\pham tien dai\.gemini\antigravity-ide\brain\bda7326d-1074-4947-b331-e5010b5a9cc3\06_roc_curve.png)

**Top features theo SHAP (XGBoost-Base):**

![SHAP Bar](C:\Users\pham tien dai\.gemini\antigravity-ide\brain\bda7326d-1074-4947-b331-e5010b5a9cc3\06_shap_bar.png)

---

## Step 7 — Cluster Profiling (Business Personas)

**Kỹ thuật:**
- Load `kmeans_fe_model.pkl` (không refit), assign cluster bằng `km.predict(X_test_enc)`.
- Profile mỗi cluster: mean features, tỷ lệ categorical, churn rate.

### Đặc trưng thực tế từng Cluster (Test set — 1,126 KH)

| Feature | Cluster 0 | Cluster 1 | Cluster 2 | Cluster 3 |
|---------|:---------:|:---------:|:---------:|:---------:|
| **Tenure (tháng)** | **14.9 ↑** | 8.6 | 8.7 | 7.9 ↓ |
| **CashbackAmount ($)** | **236.6 ↑** | 167.2 | 155.1 | 144.2 ↓ |
| **OrderCount** | **6.3 ↑** | 3.3 | 1.9 | 1.5 ↓ |
| **CouponUsed** | **3.1 ↑** | 2.1 | 1.0 | 0.8 ↓ |
| **DaySinceLastOrder** | 7.1 | 5.2 | 5.0 | **1.4 ↓** (vừa mua!) |
| **SatisfactionScore** | 3.16 | 2.99 | 2.98 | 3.12 |
| **Complain rate** | 29.4% | 30.2% | 25.7% | **36.0% ↑** |
| **Churn probability** | 0.10 | 0.11 | 0.12 | **0.25 ↑** |
| **Risk score (mean)** | 0.30 | 0.23 ↓ | 0.39 | **0.42 ↑** |
| **Churn rate thực tế** | 13% | 13% | 15% | **27%** |
| **Số KH (test set)** | 293 | 281 | 269 | 283 |

### 4 Personas Chính Xác

| Cluster | Persona | Đặc điểm |
|---------|---------|----------|
| 🟢 **0** | **Khách Hàng Thân Thiết** | Tenure dài nhất (14.9T), mua nhiều nhất (6.3 đơn), cashback cao nhất (237$), dùng coupon nhiều (3.1). Nhóm VIP lâu năm có giá trị cao nhất. |
| 🔵 **1** | **Mua Sắm Đều Đặn** | Tenure trung bình (8.6T), 3.3 đơn/kỳ, Risk score thấp nhất (0.23). Ổn định, ít biến động, gần như không có rủi ro. |
| 🟡 **2** | **Người Mua Thụ Động** | Ít đơn (1.9), ít coupon (1.0), vừa mua gần đây (5 ngày) nhưng frequency rất thấp. Complain ít nhất (25.7%) nhưng risk ngầm cao (0.39). |
| 🔴 **3** | **Khách Hàng Sắp Rời Bỏ** | Tenure ngắn nhất (7.9T), đơn ít nhất (1.5), phàn nàn nhiều nhất (36%), churn prob cao nhất (0.25). **Nghịch lý:** vừa mua nhất (1.4 ngày) nhưng đang rời bỏ. |

> [!WARNING]
> Cluster 3 có một đặc điểm đáng chú ý: `DaySinceLastOrder` thấp nhất (1.4 ngày) — có nghĩa họ vừa mua rất gần đây — nhưng churn probability lại cao nhất (0.25). Đây là dấu hiệu **"mua xong rồi bỏ"**: khách hàng vẫn còn active nhưng đang chuẩn bị rời, cần can thiệp ngay trước khi họ biến mất.

![Cluster Profiling](C:\Users\pham tien dai\.gemini\antigravity-ide\brain\bda7326d-1074-4947-b331-e5010b5a9cc3\07_cluster_profiling.png)

---

## Step 8 — Risk Scoring

### Công thức RiskScore

**Bước 1 — Raw Score:**

$$\text{RawScore} = 0.7 \times P(\text{churn}) + 0.3 \times \frac{\text{CashbackAmount}}{\text{OrderCount}}$$

| Thành phần | Trọng số | Ý nghĩa |
|-----------|---------|---------|
| `P(churn)` từ XGB-Clust | **0.7** | Xác suất rời bỏ (đã calibrate) |
| `CashbackAmount / OrderCount` | **0.3** | Giá trị kinh tế trung bình mỗi đơn |

> **Tại sao thêm cashback_pct?**  
> Hai khách hàng cùng P(churn)=0.5, nhưng người có cashback_pct=200$ gây tổn thất lớn hơn nhiều so với người cashback_pct=50$ → risk tổng thể phải cao hơn.

**Bước 2 — Chuẩn hóa Min-Max theo population test set:**

$$\text{RiskScore} = \frac{\text{RawScore} - \text{RawScore}_{\min}}{\text{RawScore}_{\max} - \text{RawScore}_{\min}} \in [0, 1]$$

Tham số từ test set: `rs_min = 0.0054`, `rs_max = 89.76`

**Bước 3 — Phân ngưỡng theo quintile (phân vị 20%):**

| Risk Category | Ngưỡng RiskScore | Số KH | Tỷ lệ |
|--------------|-----------------|-------|-------|
| 🟢 Very Low | ≤ 0.1403 | 226 | 20.1% |
| 🟡 Low | 0.1403 – 0.2546 | 225 | 20.0% |
| 🟠 Medium | 0.2546 – 0.3438 | 225 | 20.0% |
| 🔴 High | 0.3438 – 0.4864 | 225 | 20.0% |
| 🚨 Critical | > 0.4864 | 225 | 20.0% |

---

## Step 9 — Kaplan-Meier Survival Analysis

**Kỹ thuật:**
- `lifelines.KaplanMeierFitter` — xác suất "sống sót" (chưa churn) theo Tenure (tháng).
- Event: `Churn = 1` | Duration: `Tenure`.
- Log-rank test để kiểm định sự khác biệt giữa clusters.

**Kết quả kiểm định:**

$$\chi^2 = 40.42 \quad | \quad p\text{-value} = 8.69 \times 10^{-9}$$

✅ **Rất có ý nghĩa thống kê** — các cluster có survival curve khác nhau rõ rệt (p << 0.001).

| Cluster | Đường cong | Ý nghĩa |
|---------|-----------|---------|
| 🟢 0 — Khách Hàng Thân Thiết | Giảm chậm nhất | Gắn bó lâu dài, tenure cao |
| 🔵 1 — Mua Sắm Đều Đặn | Giảm chậm | Ổn định, ít rủi ro |
| 🟡 2 — Người Mua Thụ Động | Giảm trung bình | Cần theo dõi |
| 🔴 3 — Sắp Rời Bỏ | **Giảm nhanh nhất** | Rời sớm trong vòng đời |

![Kaplan-Meier Survival Curves](C:\Users\pham tien dai\.gemini\antigravity-ide\brain\bda7326d-1074-4947-b331-e5010b5a9cc3\08_km_by_cluster.png)

---

## ✅ Kết Luận — Chiến Lược Can Thiệp Tổng Hợp

### Ba Trụ Cột Quyết Định Hành Động

```
 ╔══════════════════════════════════════════════════════════════╗
 ║  RiskScore ∈ [0,1]   → Mức độ nguy hiểm tổng thể          ║
 ║  Cluster ∈ {0,1,2,3} → Nhóm hành vi & persona             ║
 ║  Survival (Tenure)   → Tốc độ rời bỏ trong vòng đời       ║
 ╚══════════════════════════╦═══════════════════════════════════╝
                            │
                            ▼
               Chiến lược can thiệp cá nhân hoá
```

### Ma Trận Can Thiệp Đầy Đủ

| Risk | Cluster | Ưu tiên | Hành động |
|------|---------|---------|-----------|
| 🚨 **Critical** | 🔴 3 (Sắp Rời) | **P1 — Ngay lập tức** | 📞 Gọi điện CSKH + Xử lý khiếu nại + Cashback voucher |
| 🚨 **Critical** | 🟢 0 (VIP) | **P1 — Giữ chân VIP** | 🎁 Liên hệ cá nhân + Quà tri ân + Up-sell đặc quyền |
| 🚨 **Critical** | 🟡 2 (Thụ Động) | **P1 — Kích hoạt lại** | 💸 Voucher lớn + Gợi ý cá nhân hoá theo hành vi |
| 🚨 **Critical** | 🔵 1 (Đều Đặn) | **P1 — Giữ ổn định** | ⚡ Flash sale + Email khẩn cấp |
| 🔴 **High** | 🔴 3 | **P2 — Trong 24–48h** | ⚠️ Email thăm dò + Mã giảm giá |
| 🔴 **High** | 🟢 0 | **P2 — VIP Alert** | 🎯 Flash sale độc quyền VIP |
| 🔴 **High** | 🔵 1 / 🟡 2 | **P2 — Trong 48h** | 💳 Freeship + Promo theo category |
| 🟠 **Medium** | Bất kỳ | **P3 — Tuần tới** | 📧 Email cá nhân hoá + Cross-sell |
| 🟡 **Low** | Bất kỳ | **P4 — Tháng tới** | 📊 Monitor + Newsletter |
| 🟢 **Very Low** | Bất kỳ | **P5 — Duy trì** | ✅ Chăm sóc tiêu chuẩn |

### Điều Chỉnh Theo Survival (Tenure)

| Tenure | Survival context | Điều chỉnh |
|--------|-----------------|-----------|
| **0–3 tháng** | Cluster 3 rời nhanh nhất ở giai đoạn này | Ưu tiên onboarding + Welcome voucher ngay sau mua đầu |
| **3–12 tháng** | Giai đoạn quyết định có gắn bó không | Loyalty program + Reward tích điểm |
| **> 12 tháng** | Khách VIP đã "sống sót" | Retention qua Up-sell + Đặc quyền VIP |

### Luồng Quyết Định Cuối

```
Khách hàng X
       │
       ├─[XGB-Clust]→ P(churn) = 0.xx
       ├─[K-Means]──→ Cluster ID = {0,1,2,3}
       │
       ▼
  RiskScore = normalize( 0.7 × P(churn) + 0.3 × cashback_pct )
       │
  ┌────┴────────────────────────────────────────┐
  │         So sánh với ngưỡng phân vị         │
  └────┬────────────────────────────────────────┘
       │
  ┌────▼────────────────────────────────────────────────┐
  │  > 0.4864  → 🚨 Critical → Xem Cluster → P1 action │
  │  > 0.3438  → 🔴 High     → Xem Cluster → P2 action │
  │  > 0.2546  → 🟠 Medium   → P3 action               │
  │  > 0.1403  → 🟡 Low      → P4 action               │
  │  ≤ 0.1403  → 🟢 Very Low → P5 action               │
  └─────────────────────────────────────────────────────┘
       │
       ▼
  Điều chỉnh theo Tenure (survival context)
       │
       ▼
  ✅ Hành động cụ thể được giao cho CSKH
```

> [!IMPORTANT]
> **Nhóm nguy hiểm tuyệt đối:** `Critical Risk + Cluster 3 + Tenure < 3 tháng`  
> → Tất cả dấu hiệu xấu đồng thời: churn prob cao (25%), phàn nàn nhiều (36%), tenure ngắn, đường survival giảm nhanh nhất.  
> → **Can thiệp ngay trong vài giờ** — không phải ngày.

---

## 📁 Artifacts Quan Trọng

| File | Nội dung |
|------|---------|
| `data/E_Commer_Data/model_comparison.csv` | Metrics 6 models |
| `data/E_Commer_Data/cluster_summary.csv` | Personas + actions 4 clusters |
| `data/E_Commer_Data/targeted_action_list.csv` | Toàn bộ KH + RiskScore + action |
| `models/E_Commer_Data/xgb_clust_model.pkl` | Model production (XGB-Clust) |
| `models/E_Commer_Data/risk_score_meta.pkl` | Ngưỡng phân vị Risk Score |
| `models/E_Commer_Data/kmeans_fe_model.pkl` | K-Means K=4 |
| `models/E_Commer_Data/encoder.pkl` | OHE encoder |
| `models/E_Commer_Data/scaler.pkl` + `imputer.pkl` | Preprocessing artifacts |

---

*Churn Alert Pipeline — E_Commer_Data*  
*Log-rank p-value = 8.69 × 10⁻⁹ | XGB-Clust ROC-AUC = 0.979 | K = 4 clusters*
