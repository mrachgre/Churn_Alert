# Churn Alert - Customer Churn Prediction & Segmentation

## Giới Thiệu Dự Án

Churn Alert là một hệ thống dự đoán rời bỏ khách hàng (churn) trong lĩnh vực thương mại điện tử. Thay vì chỉ dự đoán khách hàng nào có nguy cơ cao, dự án này tiến thêm một bước bằng cách phân nhóm khách hàng dựa trên hành vi và đặc trưng, từ đó xác định root cause của churn trong từng phân khúc và đề xuất các biện pháp cải thiện cụ thể.

## Vấn Đề và Giải Pháp

### Vấn đề Kinh Doanh

Các công ty thương mại điện tử thường gặp phải tỷ lệ churn cao mà không biết chính xác tại sao. Một số khách hàng rời bỏ do chất lượng dịch vụ, số khác do giá cả, còn một số chỉ là khách hàng "one-time buyer". Nếu áp dụng chiến lược giữ khách hàng chung cho tất cả, sẽ lãng phí chi phí.

### Giải Pháp của Dự Án

Dự án xây dựng một pipeline phân tích hoàn chỉnh có thể:

1. Dự đoán xác suất churn của mỗi khách hàng
2. Phân chia khách hàng thành các nhóm tương đồng dựa trên các features khác nhau
3. Phân tích đặc trưng riêng biệt của từng nhóm
4. Xác định yếu tố chính ảnh hưởng đến churn trong từng phân khúc
5. Đề xuất chiến lược cải thiện phù hợp cho mỗi nhóm

## Phương Pháp Tiếp Cận

### 1. Dữ Liệu và Chuẩn Bị

Dự án sử dụng dataset e-commerce với khoảng 5,000+ khách hàng và 20+ features ban đầu (tenure, số lần mua, giá trị đơn hàng, tỷ lệ hoàn lại, khiếu nại, mức độ hài lòng, v.v.).

Các bước chuẩn bị:

- Làm sạch dữ liệu (kiểm tra kiểu, sửa typo)
- Phân tích dữ liệu thiếu bằng MCAR/MAR/MNAR framework
- Khám phá mối tương quan bằng Pearson, Cramér's V
- Xử lý mất cân bằng lớp bằng SMOTE

### 2. Tính Năng (Feature Engineering)

Từ 20 features ban đầu, dự án tạo ra 40+ features qua:

- One-Hot Encoding cho biến categorical
- KNN Imputation cho dữ liệu thiếu
- Scaling/Normalization cho các biến numeric
- Tính toán K-Means distance features (khoảng cách tới từng centroid)

### 3. Dự Đoán Churn

Dự án huấn luyện 6 mô hình khác nhau:

- Logistic Regression (base features): Đơn giản, dễ giải thích
- Logistic Regression (+ K-Means distances): Tăng tính phi tuyến
- Decision Tree + Random Forest: Học patterns phức tạp
- XGBoost (base features): Gradient boosting mạnh mẽ
- XGBoost (+ cluster ID): Kết hợp clustering information

Mỗi mô hình được tối ưu hóa siêu tham số qua Optuna (40 trials) để tìm ra hyperparameter tốt nhất. Mô hình được calibrated để xác suất dự đoán có ý nghĩa thực tế.

Kết quả: ROC-AUC ~98.88%, F1-Score ~92.23%, Precision ~93.99%, Recall ~90.53%

### 4. Phân Cụm Khách Hàng

Sử dụng K-Means clustering trên 40 features để chia khách hàng thành K nhóm. Số lượng cụm được xác định qua:

- Elbow method (tìm điểm "khuỷu tay" trong inertia)
- Silhouette score (đo độ tập trung của từng cụm)

Mỗi cụm đại diện cho một phân khúc khách hàng có đặc trưng chung.

### 5. Phân Tích Cụm và Xác Định Risk Score

Với mỗi cụm, dự án tính toán:

- Tỷ lệ churn (%)
- Xác suất churn trung bình
- Giá trị trung bình của mỗi feature
- Phân phối của các biến categorical

Dựa vào tỷ lệ churn, mỗi cụm được xếp vào một Risk Tier:

- LOW RISK: churn_rate < 20%
- MEDIUM RISK: churn_rate 20-40%
- HIGH RISK: churn_rate > 40%

### 6. Phân Tích Nhân Tố và Hành Động

Từ đặc trưng của từng cụm, dự án xác định:

- Yếu tố khác biệt nhất của mỗi cụm (top 5 features)
- Feature nào có ảnh hưởng tích cực hoặc tiêu cực tới churn
- Hành động cụ thể nên áp dụng cho từng phân khúc

Ví dụ:

- Cụm khách hàng mới (Tenure < 3 tháng) có churn cao → cần tăng cường onboarding
- Cụm khách hàng inactive nhưng giá trị cao → cần re-engagement campaigns
- Cụm khách hàng loyal → cần loyalty program

### 7. Survival Analysis (Tùy Chọn)

Sử dụng Kaplan-Meier estimator để ước tính tỷ lệ tồn tại theo thời gian và Cox Proportional Hazards model để xác định hazard ratios của từng yếu tố.

## Các Insights Chính

### Insight 1: Death Valley của Khách Hàng Mới

Khách hàng trong 3 tháng đầu có tỷ lệ churn từ 40-50%, sau đó giảm xuống 10-15% nếu họ vượt qua giai đoạn này. Điều này cho thấy onboarding experience là crítico.

### Insight 2: Paradox của Mức Độ Hài Lòng Cao

Khách hàng có mức độ hài lòng 5 sao lại có tỷ lệ churn cao nhất (25-30%), cao hơn cả khách hàng không hài lòng. Phân tích sâu hơn cho thấy đây là "one-time buyers" - họ mua một lần, rất hài lòng, nhưng không có nhu cầu mua lại trong tương lai gần.

### Insight 3: Khiếu Nại là Dấu Hiệu Churn Rõ Ràng

Khách hàng có khiếu nại có tỷ lệ churn 2.17 lần cao hơn. Tuy nhiên, khách hàng có khiếu nại nhưng được xử lý tốt lại trở thành những người trung thành nhất.

### Insight 4: Giá Trị Đơn Hàng Không Phải Tất Cả

Khách hàng có giá trị cao không nhất thiết trung thành. Tần suất mua hàng (Frequency) lại có tương quan âm mạnh mẽ với churn.

### Insight 5: Cashback là Yếu Tố Giữ Chân Chính

SHAP analysis cho thấy Cashback amount là feature có ảnh hưởng tiêu cực (tích cực) nhất tới churn. Mỗi 100 đơn vị tăng Cashback làm giảm xác suất churn ~5%.

## Tác Dụng và Ứng Dụng Thực Tế

### Tác Dụng Trực Tiếp

1. **Nhận diện sớm khách hàng nguy cơ cao**: Có thể tác động trước khi họ rời đi, không phải phát hiện sau này.

2. **Tối ưu hóa chi phí marketing**: Thay vì dùng chiến lược chung, công ty có thể tập trung ngân sách vào các phân khúc có ROI cao nhất.

3. **Cá nhân hóa chiến lược**: Mỗi phân khúc nhận chiến lược khác nhau phù hợp với nguyên nhân churn của chúng.

4. **Theo dõi hiệu quả**: So sánh tỷ lệ churn trước và sau can thiệp cho mỗi phân khúc.

### Các Use Cases Cụ Thể

- **Chuyên viên bán hàng**: Ưu tiên liên hệ khách hàng high-risk để cứu lấy họ
- **Quản lý sản phẩm**: Điều chỉnh chính sách cashback, ưu đãi dựa trên insights
- **Quản lý dịch vụ khách hàng**: Tăng cường support cho khách hàng mới trong 3 tháng đầu
- **Ban quản lý**: Theo dõi KPI churn rate theo phân khúc để đánh giá chiến lược
- **Quản lý sản phẩm**: Thiết kế onboarding experience mới dựa trên root causes của churn

## Cấu Trúc Dự Án

```
Churn_Alert/
├── data/E_Commer_Data/        # Dữ liệu đầu vào và trung gian
├── src/                        # 8 modules xử lý tuần tự
├── models/                     # Các mô hình đã huấn luyện
├── outputs/                    # Biểu đồ và báo cáo
└── run_pipeline.py            # Script chạy toàn bộ pipeline
```

## Tech Stack

Data processing: pandas, numpy
Machine learning: scikit-learn, XGBoost, Optuna
Statistics: lifelines (survival analysis)
Visualization: matplotlib, seaborn, plotly, SHAP
Others: joblib (model serialization), sqlite3 (optional RFM)

## Output Chính

- cluster_summary.csv: Thống kê mỗi cụm (churn rate, feature means)
- cluster_profiles.csv: Hồ sơ chi tiết từng khách hàng + cluster ID
- predictions.csv: Xác suất churn và cluster của mỗi khách hàng
- Biểu đồ phân tích: Đặc trưng numeric/categorical của từng cụm
- SHAP plots: Feature importance và ảnh hưởng của từng feature

## Kết Luận

Churn Alert không chỉ dự đoán mà còn giải thích. Thay vì một con số xác suất churn, nó cung cấp:

- Tại sao khách hàng có nguy cơ rời bỏ
- Họ thuộc nhóm nào có đặc điểm tương tự
- Nên làm gì để giữ lại họ

Điều này giúp các quyết định kinh doanh trở nên dựa trên dữ liệu và hành động cụ thể hơn là cảm giác.
