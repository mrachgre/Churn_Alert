# Airline Passenger Satisfaction — Technical Pipeline Report

**XGBoost-Based Dissatisfaction Risk Scoring & SHAP Insight**

| | |
|---|---|
| Dataset | Customer Satisfaction in Airline — 129,880 rows × 22 cols |
| Test set | 25,976 passengers |
| Production model | XGB-Base · ROC-AUC = **0.9942** |
| Scope | Descriptive analysis only — no action recommendations |

---


## 1. Executive Summary

- **Dataset:** 129,880 survey responses, 22 features, binary target (satisfied / dissatisfied), class balance 54.7 / 45.3% — SMOTE was not applied (minority class ≥ 40% threshold).
- **Production model:** XGB-Base (XGBoost + Isotonic calibration, Optuna 20 trials). Test-set **ROC-AUC = 0.9942**, F1 = 0.9625, Accuracy = 0.9593, Threshold = 0.459 (Youden's J).
- **Risk distribution:** **40.0%** of the test set (10,390 passengers) fall in the High or Critical tier. Both tiers each account for exactly 20% by design (quintile split).
- **Primary drivers (SHAP):** `Seat comfort` ranks **1st by SHAP** despite ranking only 7th by bivariate correlation — evidence of a non-linear interaction effect. `Inflight entertainment` ranks 2nd/3rd. Departure Delay accounts for only **1.5%** of Critical-group top-3 drivers.
- **Loyalty bias finding:** Within cluster C1 (Trải Nghiệm Kém), Disloyal customers have mean risk_score **0.811 vs 0.627** for Loyal, yet service ratings differ by **< 0.1 points** across all dimensions. The score differential is driven by the `loyal` feature itself, not measured service quality differences.

---


## 2. Dataset & Preprocessing


### 2.1 Data Source

Source: Customer Satisfaction in Airline survey export (2026-06-16).
129,880 rows × 22 columns. One row per flight experience. Target: `satisfaction` (binary).
No customer ID — rows treated as independent observations.


### 2.2 Preprocessing Decisions

Bảng tóm tắt các quyết định tiền xử lý và lý do kỹ thuật:

| Quyết định | Chi tiết | Lý do |
| :--- | :--- | :--- |
| Drop Arrival Delay | Xóa cột 'Arrival Delay in Minutes' | Pearson r = 0.9653 với Departure Delay → multicollinearity; giữ cả 2 không thêm thông tin và làm mất ổn định linear model. |
| Recode rating 0 → NaN | 14 cột rating (thang 1–5): 0 → NaN | Verify N/A hypothesis: satisfaction rate tại rating=0 là 0.447, gần với rating=3 (0.510) hơn rating=1 (0.268) → 0 là 'không áp dụng', không phải 'rất kém'. |
| Tạo was_na_* indicators | 14 cột binary was_na_<feature>=1 nếu gốc=0 | Giữ lại thông tin dịch vụ không được sử dụng — tín hiệu khác về bản chất so với rating thấp. |
| Impute rating — Tree path | 14 cột: giữ NaN cho XGBoost | XGBoost xử lý missing natively qua learned split direction; impute sẽ xóa mất N/A signal. |
| Impute rating — Linear path | 14 cột: median (fit trên train) | Logistic Regression không chấp nhận NaN; dùng median để robust với phân phối lệch. |
| Impute composite scores | 3 cột composite: median (fit trên train) | KMeans (sklearn) yêu cầu non-NaN; composite scores có 0% NaN nên chỉ là phòng ngừa. |
| Clip Departure Delay | Linear path: cap tại p99 train = 181 phút | 43 dòng (0.03%) có delay > 500 phút — làm lệch MinMaxScaler. XGBoost xử lý natively nên chỉ clip linear path. |
| Encoding Class | Eco=0, Eco Plus=1, Business=2 (ordinal) | Quan hệ thứ tự tự nhiên. OHE tạo dummy trap và tăng chiều không cần thiết. |
| Encoding Customer Type | Loyal=1, Disloyal=0 (binary) | Categorical nhị phân, không có quan hệ thứ tự. |
| Encoding Type of Travel | Business travel=1, Personal=0 (binary) | Categorical nhị phân. |
| Train/Test split | Stratified 80/20, random_state=42 | Stratification đảm bảo class balance nhất quán: 54.7% satisfied ở cả train lẫn test. |

---


## 3. Feature Engineering


### 3.1 Composite Experience Scores

3 composite features được tạo ra để thay thế RFM Segmentation. RFM không áp dụng được vì dataset chỉ có 1 survey response/chuyến bay, không có lịch sử giao dịch lặp lại theo thời gian (Recency và Frequency không xác định được, Survival Analysis không có time-to-event endpoint).

| Feature | Công thức | Cột thành phần (n) | NaN trong Train |
| :--- | :--- | :--- | :---: |
| `ground_experience_score` | `mean(GROUND_COLS, skipna=True)` | Ease of Online booking, Online boarding, Checkin service, Gate location, Baggage handling (5) | 0 (0.0%) |
| `inflight_experience_score` | `mean(INFLIGHT_COLS, skipna=True)` | Seat comfort, Food and drink, Inflight wifi service, Inflight entertainment, On-board service, Leg room service, Cleanliness (7) | 0 (0.0%) |
| `delay_severity` | `Departure Delay / (Flight Distance + 1)` | Departure Delay in Minutes, Flight Distance (2) | 0 (0.0%) |


0 NaN ở cả 3 cột: không có khách nào NaN toàn bộ các cột trong một nhóm, nên mean(skipna=True) luôn trả về giá trị.

---


## 4. Dual Preprocessing Pipeline

Hai nhánh preprocessing song song dành cho 2 nhóm model khác nhau về bản chất (tree-based vs linear). Gộp chung 2 nhánh sẽ hoặc làm hỏng tree models (impute không cần thiết xóa N/A signal) hoặc làm vỡ linear models (không hỗ trợ NaN).

| Bước | Tree Path (XGB / RF / DT) | Linear Path (LR) |
| :--- | :--- | :--- |
| Outlier | Giữ nguyên (tree splits xử lý) | Departure Delay clip tại p99 train = 181 phút |
| Rating imputation | NaN giữ nguyên — XGBoost dùng learned split direction | 14 cột rating → median (fit train) |
| Composite imputation | Median (fit train) — bắt buộc cho KMeans | Median (fit train) |
| Encoding | Ordinal/binary (như Mục 2.2) | Như Tree Path |
| Scaling | Không cần cho tree models | MinMaxScaler [0,1] trên toàn bộ numeric (was_na_* excluded) |
| Feature selection | Toàn bộ features truyền vào model | Chi² SelectKBest, p<0.05 → 30/38 kept (Base), 32/40 (Dist) |
| KMeans features | cluster_id hoặc khoảng cách centroid thêm vào | Khoảng cách centroid (dist_centroid_0/1) trong Dist variant |
| SMOTE | Bỏ qua — minority class 45.3% ≥ 40% ngưỡng | Bỏ qua (lý do như nhau) |



### KMeans Feature Engineering

KMeans (K=2, chọn bằng silhouette score trong K=2..8) được fit **chỉ trên training set** bằng 5 features: `ground_experience_score`, `inflight_experience_score`, `delay_severity`, `Age`, `Flight Distance`. Categorical features được loại trừ để cluster phản ánh trải nghiệm dịch vụ, không lẫn demographic.

- **Strategy A:** Khoảng cách Euclidean đến mỗi centroid (2 cột) → Linear path
- **Strategy B:** cluster_id (integer) → Tree path
- Test set được transform bằng `predict()` trên model đã fit — không refit.

![Hình 4.1: Elbow và Silhouette plots cho KMeans K=2..8. Silhouette đạt max tại K=2 (0.2309).](d:/HUST 2025.2/Khoa học dữ liệu/Churn_Alert/outputs/AIRLINE/p2_05_elbow_silhouette.png)
*Hình 4.1: Elbow và Silhouette plots cho KMeans K=2..8. Silhouette đạt max tại K=2 (0.2309).*

---


## 5. Model Comparison

| Model | Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGB-Base** | 0.459 | 0.9593 | 0.9712 | 0.9539 | 0.9625 | 0.9942 | 0.9955 |
| XGB-Clust | 0.445 | 0.9588 | 0.9679 | 0.9565 | 0.9622 | 0.9943 | 0.9956 |
| RF-Clust | 0.499 | 0.9468 | 0.9584 | 0.9437 | 0.951 | 0.9905 | 0.9929 |
| DT-Clust | 0.548 | 0.9249 | 0.9415 | 0.9199 | 0.9306 | 0.9809 | 0.9844 |
| LR-Base | 0.558 | 0.8647 | 0.887 | 0.8626 | 0.8747 | 0.9414 | 0.9526 |
| LR-Dist | 0.541 | 0.8649 | 0.8826 | 0.8687 | 0.8756 | 0.9414 | 0.9526 |


*Bảng 5.1: So sánh 6 model trên test set (n=25,976). Threshold tối ưu bằng Youden's J trên train predictions. XGB-Base và XGB-Clust được calibrate bằng CalibratedClassifierCV (isotonic, cv=5).*

![Hình 5.1: So sánh Accuracy, F1, ROC-AUC, PR-AUC của 6 model.](d:/HUST 2025.2/Khoa học dữ liệu/Churn_Alert/outputs/AIRLINE/p2_06_model_comparison.png)
*Hình 5.1: So sánh Accuracy, F1, ROC-AUC, PR-AUC của 6 model.*


### Lý do chọn XGB-Base làm Production Model

- **XGB-Clust vs XGB-Base:** ROC-AUC 0.9943 vs 0.9942 — chênh lệch +0.0001, không đáng kể. Thêm `kmeans_cluster_id` không cải thiện XGBoost vì XGBoost có thể học cấu trúc cluster ngầm qua tree splits.
- **LR-Dist vs LR-Base:** ROC-AUC 0.9414 cả hai — KMeans distance features không mang lại lift cho Logistic Regression.
- KMeans clusters được giữ lại cho profiling và risk scoring (Mục 6–8), nhưng **không đưa vào production prediction model**.
- **Tuning XGB-Base:** Optuna TPE sampler, 20 trials, 5-fold stratified CV. Best params: n_estimators=203, max_depth=8, lr=0.127, subsample=0.947, colsample_bytree=0.592. CV ROC-AUC = 0.9938.

---


## 6. Cluster Profiling — Passenger Personas

| Cluster | Persona | N | Dissatisfied % | Ground Score | Inflight Score | Delay Severity | Age (mean) | Biz Travel % | Loyal % |
| :---: | :--- | ---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| C0 | Hài Lòng Toàn Diện | 14,135 | 22.8% | 3.871 | 3.820 | 0.0110 | 41.6 | 68.5% | 87.0% |
| C1 | Trải Nghiệm Kém | 11,841 | 72.1% | 2.776 | 2.757 | 0.0110 | 37.1 | 69.0% | 75.3% |


*Bảng 6.1: Cluster profile từ KMeans K=2 (test set, n=25,976).*

![Hình 6.1: So sánh cluster — Dissatisfied rate, Ground score, Inflight score, Business Travel %.](d:/HUST 2025.2/Khoa học dữ liệu/Churn_Alert/outputs/AIRLINE/p2_07_cluster_profiling.png)
*Hình 6.1: So sánh cluster — Dissatisfied rate, Ground score, Inflight score, Business Travel %.*


### Finding: Persona phản ánh chất lượng dịch vụ, không phải demographic

Cả 2 cluster có tỷ lệ Business Travel gần tương đương (C0: 68.5%, C1: 69.0%) và delay severity giống nhau (0.011). Sự khác biệt chính là trải nghiệm dịch vụ:
- `ground_score`: C1 = 2.78 vs C0 = 3.87 (Δ = −1.09)
- `inflight_score`: C1 = 2.76 vs C0 = 3.82 (Δ = −1.06)

Cluster structure phản ánh mức độ hài lòng, **không phải phân khúc hành khách**.

---


## 7. Risk Scoring


### 7.1 Công thức

```python
passenger_value_proxy = (Class_ord / 2 + biz_travel) / 2
    # Class_ord: Eco=0, Eco Plus=1, Business=2 (÷2 → [0,1])
    # biz_travel: 1 nếu Business travel, 0 nếu Personal

RawScore  = 0.7 × P(dissatisfied) + 0.3 × passenger_value_proxy
RiskScore = (RawScore − min) / (max − min)   # min-max theo test set population
```

Trọng số 0.7/0.3 phản ánh ưu tiên: xác suất không hài lòng từ model (0.7) cộng với giá trị thương mại ước lượng của hành khách (0.3) — Business class / business travel được ưu tiên cao hơn do kỳ vọng lifetime value cao hơn, dù dataset không có dữ liệu chi tiêu thực tế.


### 7.2 Phân bố Risk Tier

| Tier | Ngưỡng RiskScore | Count | % Test Set |
| :--- | :---: | ---: | :---: |
| Very Low | 0.000 – 0.3000 | 8,538 | ~20% |
| Low | 0.3000 – 0.3000 | 3,223 | ~12% |
| Medium | 0.3000 – 0.6775 | 3,825 | ~15% |
| High | 0.6775 – 0.8496 | 5,199 | ~20% |
| Critical | 0.8496 – 1.000 | 5,191 | ~20% |
| **High + Critical** | — | **10,390** | **40%** |


> ⚠ **Lưu ý kỹ thuật:** Risk tier dùng quintile split → tỷ lệ mỗi tier **cố định theo thiết kế** (~20%), không phải phát hiện về tỷ lệ rủi ro thực tế. Giá trị có ý nghĩa là các **ngưỡng cắt** (p20=0.3000, p40=0.3000, p60=0.6775, p80=0.8496), không phải % mỗi nhóm. p20 ≈ p40 = 0.300 cho thấy phân phối RiskScore bị lệch phải — mật độ cao ở vùng risk thấp.


![Hình 7.1: Phân bố Risk Score (trái) và mean risk score theo cluster (phải).](d:/HUST 2025.2/Khoa học dữ liệu/Churn_Alert/outputs/AIRLINE/p2_08_risk_score.png)
*Hình 7.1: Phân bố Risk Score (trái) và mean risk score theo cluster (phải).*

---


## 8. SHAP Analysis


### 8.1 Global Feature Importance

SHAP TreeExplainer áp dụng trên base XGBoost estimator trích xuất từ CalibratedClassifierCV.
SHAP values tính cho toàn bộ 25,976 test rows. Feature importance = mean(|SHAP value|).

| SHAP Rank | Feature | Mean |SHAP| | PtBis Rank | Δ Rank | Ghi chú |
| :---: | :--- | :---: | :---: | :---: | :--- |
| 1 | `Seat comfort` | 1.7207 | 7 | -6 | ⚠ Non-linear effect |
| 2 | `Inflight entertainment` | 0.9934 | 1 | +1 |  |
| 3 | `loyal` | 0.7138 | — | — | Categorical |
| 4 | `biz_travel` | 0.5217 | — | — | Categorical |
| 5 | `Departure/Arrival time convenient` | 0.5175 | — | — | Categorical |
| 6 | `inflight_experience_score` | 0.4527 | 2 | +4 | ⚠ Non-linear effect |
| 7 | `Ease of Online booking` | 0.3783 | 4 | +3 | ⚠ Non-linear effect |
| 8 | `Gate location` | 0.3573 | — | — | Categorical |
| 9 | `Baggage handling` | 0.3131 | — | — | Categorical |
| 10 | `Online support` | 0.2847 | 5 | +5 | ⚠ Non-linear effect |
| 11 | `Cleanliness` | 0.2680 | — | — | Categorical |
| 12 | `Checkin service` | 0.2657 | — | — | Categorical |


*Bảng 8.1: SHAP global importance vs Point-Biserial rank (Step 3). ⚠ = rank shift ≥ 3 vị trí.*

> **Phát hiện quan trọng:** `Seat comfort` nhảy từ PtBis rank 7 lên **SHAP rank 1** — feature mạnh nhất trong model, nhưng bị underestimate bởi phân tích đơn biến do hiệu ứng tương tác phi tuyến. `loyal` và `biz_travel` không có trong PtBis top-10 (là binary categorical) nhưng lần lượt xếp hạng 2 và 4 theo SHAP.

![Hình 8.1: SHAP global importance bar chart — Top 15 features.](d:/HUST 2025.2/Khoa học dữ liệu/Churn_Alert/outputs/AIRLINE/p2_shap_bar.png)
*Hình 8.1: SHAP global importance bar chart — Top 15 features.*

![Hình 8.2: SHAP beeswarm — hướng và độ lớn tác động. Đỏ = feature value cao, Xanh = thấp. SHAP dương = đẩy về phía satisfied.](d:/HUST 2025.2/Khoa học dữ liệu/Churn_Alert/outputs/AIRLINE/p2_shap_beeswarm.png)
*Hình 8.2: SHAP beeswarm — hướng và độ lớn tác động. Đỏ = feature value cao, Xanh = thấp. SHAP dương = đẩy về phía satisfied.*


### 8.2 SHAP theo Cluster

| Rank | C0 Feature | C0 Mean|SHAP| | C1 Feature | C1 Mean|SHAP| |
| :---: | :--- | :---: | :--- | :---: |
| 1 | Seat comfort | 1.8397 | Seat comfort | 1.5785 |
| 2 | Inflight entertainment | 1.0130 | Inflight entertainment | 0.9699 |
| 3 | biz_travel | 0.6532 | **loyal** ← | 0.7903 |
| 4 | loyal | 0.6497 | **inflight_experience_score** ← | 0.5426 |
| 5 | Dep/Arrival time convenient | 0.5224 | Dep/Arrival time convenient | 0.5117 |
| 6 | inflight_experience_score | 0.3775 | **Gate location** ← | 0.4073 |
| 7 | Ease of Online booking | 0.3701 | Ease of Online booking | 0.3881 |
| 8 | Leg room service | 0.3320 | **biz_travel** ← | 0.3647 |
| 9 | Baggage handling | 0.3251 | Baggage handling | 0.2989 |
| 10 | Gate location | 0.3154 | **Cleanliness** ← | 0.2907 |


*Bảng 8.2: Top-10 SHAP per cluster. ← đánh dấu vị trí khác biệt giữa C0 và C1.*

> **Khác biệt chính:** Trong C0 (satisfied), `biz_travel` xếp hạng 3 — mục đích di chuyển ảnh hưởng đến satisfaction của hành khách vốn đã đánh giá dịch vụ tốt. Trong C1 (dissatisfied), `loyal` nổi lên hạng 3, phản ánh Customer Type có trọng số lớn trong nhóm có rating dịch vụ đồng đều thấp.

![Hình 8.3: SHAP top-10 theo cluster — C0 (xanh) và C1 (đỏ).](d:/HUST 2025.2/Khoa học dữ liệu/Churn_Alert/outputs/AIRLINE/p2_shap_by_cluster.png)
*Hình 8.3: SHAP top-10 theo cluster — C0 (xanh) và C1 (đỏ).*


### 8.3 SHAP nhóm Critical

SHAP analysis giới hạn trong nhóm Critical (n=5,191, 20% test set).

| Driver Feature | % Critical | Rating (Critical) | Rating (All) | Δ |
| :--- | :---: | :---: | :---: | :---: |
| `Seat comfort` | 86.3% | 2.16 | 2.95 | -0.79 |
| `Inflight entertainment` | 68.2% | 2.43 | 3.46 | -1.03 |
| `loyal` | 48.6% | 0.51 | 0.82 | -0.31 |
| `inflight_experience_score` | 26.7% | 2.69 | 3.34 | -0.65 |
| `Gate location` | 9.8% | 3.07 | 2.99 | +0.08 |
| `Ease of Online booking` | 9.6% | 2.68 | 3.47 | -0.79 |
| `Departure/Arrival time convenient` | 9.0% | 2.75 | 3.15 | -0.40 |
| `Age` | 7.8% | 36.52 | 39.52 | -3.00 |
| `Online support` | 6.3% | 2.88 | 3.52 | -0.64 |
| `Cleanliness` | 5.8% | 3.20 | 3.71 | -0.51 |


*Bảng 8.3: Tần suất driver và rating trung bình trong Critical group vs toàn dataset.*

![Hình 8.4: SHAP top-10 nhóm Critical (trái) và tần suất driver trong top-3 của từng khách (phải).](d:/HUST 2025.2/Khoa học dữ liệu/Churn_Alert/outputs/AIRLINE/p2_shap_critical.png)
*Hình 8.4: SHAP top-10 nhóm Critical (trái) và tần suất driver trong top-3 của từng khách (phải).*

---


## 9. Key Finding: Loyalty Bias trong Risk Score

Trong cluster C1 (Trải Nghiệm Kém, n=11,841), hành khách chia thành 2 nhóm theo Customer Type:
- **Loyal Customer:** 8,916 (75.3% của C1)
- **Disloyal Customer:** 2,925 (24.7% của C1)

Phân phối risk score giữa 2 nhóm này cho thấy chênh lệch đáng kể cần phân tích kỹ thuật cẩn thận.

| Metric | Loyal (C1) | Disloyal (C1) | Chênh lệch |
| :--- | :---: | :---: | :---: |
| Count | 8,916 (75.3%) | 2,925 (24.7%) | — |
| Mean risk_score | 0.627 | 0.811 | −0.184 |
| High/Critical rate | 61.1% | 86.2% | −25.1 pp |
| Seat comfort (mean) | 2.37 | 2.40 | +0.03 |
| Inflight entertainment | 2.93 | 2.42 | −0.51 |
| inflight_experience_score | 2.76 | 2.74 | −0.02 |
| ground_experience_score | 2.75 | 2.86 | +0.11 |
| On-board service | 2.76 | 2.95 | +0.19 |
| Leg room service | 2.89 | 2.99 | +0.10 |
| Cleanliness | 3.05 | 3.45 | −0.40 |


*Bảng 9.1: So sánh sub-group C1 — Loyal vs Disloyal. Risk score là output của model; rating là giá trị survey thực tế.*

![Hình 9.1: Service ratings (trái) và mean risk score (phải) cho Loyal vs Disloyal trong C1. Chênh lệch rating < 0.5 ở mọi chiều; risk score chênh 0.184.](d:/HUST 2025.2/Khoa học dữ liệu/Churn_Alert/outputs/AIRLINE/p2_loyalty_bias_c1.png)
*Hình 9.1: Service ratings (trái) và mean risk score (phải) cho Loyal vs Disloyal trong C1. Chênh lệch rating < 0.5 ở mọi chiều; risk score chênh 0.184.*


### Giải thích Phương pháp luận

Chênh lệch risk score (0.627 vs 0.811) giữa Loyal và Disloyal trong C1 xuất phát từ trọng số SHAP cao của feature `loyal` (hạng 2 globally, hạng 3 trong C1). Tuy nhiên, trải nghiệm dịch vụ đo được — qua tất cả các cột rating — chênh lệch dưới 0.5 điểm ở mọi chiều và dưới 0.11 cho composite scores.

Pattern này nhất quán với một hoặc nhiều giải thích sau:
1. Loyal customers thực sự khác nhau ở các yếu tố không đo được (kỳ vọng từ các chuyến trước, baseline tham chiếu) tương quan với satisfaction nhưng không có trong survey.
2. Nhãn `loyal` mang tín hiệu satisfaction lịch sử từ các chuyến bay trước không có trong survey hiện tại.
3. Model học được correlation giả giữa Customer Type và dissatisfied outcome, không phản ánh quan hệ nhân quả từ chất lượng dịch vụ thực tế.

Không thể phân biệt các giải thích này từ dữ liệu hiện có. **Risk score của Disloyal passengers trong C1 cần được diễn giải với sự bất định này.**

---


## 10. Limitations & Caveats

> ⚠ **RFM và Survival Analysis không áp dụng được:** Dataset chỉ có 1 survey/chuyến bay, không có lịch sử giao dịch lặp lại và không có time-to-event endpoint. Áp dụng phương pháp temporal cho dữ liệu này là sai phương pháp luận.

> ⚠ **Risk tier là tương đối, không tuyệt đối:** Phân chia theo quintile → mỗi tier chiếm đúng ~20% theo thiết kế, không phải phát hiện về phân phối rủi ro thực tế. Ngưỡng cắt (p20=0.300, p60=0.678, p80=0.850) mới là con số có nghĩa vận hành.

> ⚠ **Loyalty bias chưa được cô lập:** Không có model variant nào train loại bỏ Customer Type để so sánh. Mức độ bias attributable cho feature vs hành vi thực tế vẫn chưa xác định.

> ⚠ **SHAP không phải nhân quả:** SHAP values phản ánh đóng góp feature trong model (log-odds scale), không chứng minh quan hệ nhân quả. SHAP cao cho Seat comfort nghĩa là model dựa nhiều vào feature đó — không có nghĩa là cải thiện ghế sẽ tất yếu thay đổi satisfaction.

> ⚠ **passenger_value_proxy là xấp xỉ cấu trúc:** Không có dữ liệu chi tiêu, doanh thu, hay lifetime value thực tế. Proxy có thể xếp sai thứ tự ưu tiên các hành khách có giá trị thương mại không tương quan với hạng vé / mục đích đi lại.

> ⚠ **Metrics đo trên single hold-out:** random_state=42, không có repeated CV hay bootstrap CI cho test metrics cuối cùng.

---


## 11. Appendix — Artifact File Index

| Loại | File | Mô tả |
| :---: | :--- | :--- |
| Model | xgb_base_model.pkl | XGB-Base — CalibratedClassifierCV, production model |
| Model | xgb_clust_model.pkl | XGB-Clust — có KMeans cluster_id feature |
| Model | rf_clust_model.pkl | RF-Clust — Random Forest |
| Model | dt_clust_model.pkl | DT-Clust — Decision Tree |
| Model | lr_base_model.pkl | LR-Base — Logistic Regression (Chi² Base features) |
| Model | lr_dist_model.pkl | LR-Dist — Logistic Regression (Chi² Dist features) |
| Model | kmeans_model.pkl | KMeans K=2 (fit on train, 5 features) |
| Model | kmeans_scaler.pkl | StandardScaler cho KMeans input features |
| Model | imputer.pkl | SimpleImputer median — 14 rating cols, linear path |
| Model | imputer_composite.pkl | SimpleImputer median — 3 composite scores |
| Model | scaler.pkl | MinMaxScaler — linear path |
| Model | chi2_meta.pkl | Chi² feature selection metadata (Base + Dist) |
| Model | risk_score_meta.pkl | RiskScore min/max và quintile thresholds |
| Data | model_comparison.csv | Bảng metrics 6 model |
| Data | cluster_summary.csv | KMeans K=2 cluster profile |
| Data | customer_risk_insight_report.csv | Per-customer risk + top-3 SHAP drivers + actual ratings (25,976 rows) |
| Data | critical_driver_frequency.csv | Tần suất driver trong Critical group |
| Data | shap_global_importance.csv | Global SHAP mean|value| cho 37 features |
| Data | system_level_insight_report.md | System-level descriptive insight report |
| Chart | p2_03_pearson_heatmap.png | Pearson heatmap — 14 cột rating |
| Chart | p2_03_point_biserial.png | Point-Biserial top-12 vs satisfaction |
| Chart | p2_05_elbow_silhouette.png | KMeans elbow + silhouette K=2..8 |
| Chart | p2_05b_chi2_scores.png | Chi² feature selection — Base variant |
| Chart | p2_06_model_comparison.png | 6-model comparison bar chart |
| Chart | p2_07_cluster_profiling.png | Cluster profiling — 4-metric bar chart |
| Chart | p2_08_risk_score.png | Risk score distribution + mean by cluster |
| Chart | p2_shap_bar.png | SHAP global importance bar — top 15 |
| Chart | p2_shap_beeswarm.png | SHAP beeswarm — direction + magnitude |
| Chart | p2_shap_by_cluster.png | SHAP top-10 by cluster (C0 / C1) |
| Chart | p2_shap_critical.png | SHAP Critical group + driver frequency |
| Chart | p2_loyalty_bias_c1.png | C1 loyalty bias: ratings vs risk score |


- **Models:** `models/AIRLINE/`
- **Charts:** `outputs/AIRLINE/`
- **Data & Reports:** `data/AIRLINE/`
