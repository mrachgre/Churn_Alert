# COFINFAD — Colombian Fintech Financial Analytics Dataset

## README & Data Dictionary

---

## 1. README

| | |
|---|---|
| **Tên dataset** | COFINFAD (Colombian Fintech Financial Analytics Dataset) |
| **Tác giả** | Luis Eduardo Muñoz Guerrero, Yony Fernando Ceballos, Luis David Trejos Rojas |
| **Đơn vị** | Universidad Tecnológica de Pereira |
| **Ngày công bố** | 06/06/2025 (Version 1) |
| **Nguồn chính** | [Mendeley Data — DOI: 10.17632/mhb4zn3258.1](https://data.mendeley.com/datasets/mhb4zn3258/1) |
| **Nguồn phụ (ML)** | [Hugging Face Datasets](https://huggingface.co/datasets/luisdavidtrejosrojas/cofinfad) (DOI: 10.57967/hf/2942) |
| **Giấy phép** | CC BY 4.0 (Mendeley) / ODC-BY (Hugging Face) |

### Mô tả tổng quan

COFINFAD là dữ liệu hành vi & giao dịch thực tế (đã ẩn danh) thu thập từ
**48.723 khách hàng** (toàn bộ cơ sở khách hàng đang hoạt động) của một
công ty fintech Colombia, trong giai đoạn 12 tháng từ **04/01/2023 đến
29/12/2023**. Dataset bao gồm **3.159.157 giao dịch** cá nhân và **57 biến**,
kết hợp dữ liệu:

- Nhân khẩu học (demographics)
- Hành vi giao dịch (transactional behavior)
- Mức độ sử dụng & tương tác app (digital engagement)
- Mức độ hài lòng khách hàng / NPS (satisfaction & NPS)
- Chỉ số dự báo churn & giá trị khách hàng (CLV, churn probability)

> Mục đích nghiên cứu: phân tích giữ chân khách hàng (customer retention),
> hành vi tài chính, và mức độ chấp nhận dịch vụ tài chính số tại các thị
> trường mới nổi ở Mỹ Latinh.

### Cấu trúc file

Dataset gồm **2 file dữ liệu chính** (cộng lại đủ 57 biến mô tả):

1. **`customer_profile_data.csv`**
   - Mức chi tiết: 1 dòng = 1 khách hàng (48.723 dòng)
   - 53 cột: nhân khẩu học, sản phẩm sử dụng, hành vi app, hài lòng, NPS, các
     chỉ số giao dịch đã tổng hợp, CLV, churn...

2. **`transactions_data.csv`**
   - Mức chi tiết: 1 dòng = 1 giao dịch (3.159.157 dòng)
   - 4 cột: `customer_id`, `date`, `amount`, `type`
   - Liên kết với file (1) qua khóa `customer_id`

### Phương pháp thu thập dữ liệu

- Trích xuất API tự động hàng ngày từ CRM & cơ sở dữ liệu giao dịch của công
  ty (Python + requests), thực hiện vào giờ thấp tải; thu 42 biến liên quan
  giao dịch (số tiền, thời gian, loại giao dịch, danh mục merchant, trạng
  thái hoàn tất).
- Khảo sát trong app theo quý (tháng 3, 6, 9, 12/2023), tỷ lệ phản hồi trung
  bình 14,3% (6.965 phản hồi); 5 câu hỏi hài lòng cốt lõi theo thang điểm
  1–6, cộng đánh giá theo sản phẩm và 3 câu hỏi mở.
- Phân tích sử dụng app qua hạ tầng mobile analytics có sẵn (thời lượng/tần
  suất session, khung giờ dùng, tương tác tính năng, lượt xem màn hình,
  hiệu năng).
- Kiểm định dữ liệu: đối chiếu số lượng record với log hệ thống nguồn, loại
  bỏ tài khoản test/giao dịch nội bộ, loại trùng lặp, phân tích ngoại lai
  thống kê, kiểm tra quy tắc nghiệp vụ.
- Ẩn danh hóa theo Luật bảo vệ dữ liệu Colombia (*Ley 1581 de 2012*): loại
  bỏ định danh cá nhân, mã hóa không thể đảo ngược, gộp địa lý theo cấp
  thành phố, các biến đổi bảo vệ quyền riêng tư khác.
- Tích hợp dữ liệu: hợp nhất nhiều nguồn theo `customer_id`, kiểm tra tính
  toàn vẹn tham chiếu, chuẩn hóa định dạng, đồng bộ dữ liệu theo thời gian.
- Tính các chỉ số dẫn xuất: customer lifetime value, các chỉ số hành vi,
  tổng hợp điểm hài lòng từ khảo sát, tần suất giao dịch, dự báo xác suất
  churn theo cửa sổ 30 ngày.

### Giới hạn / Lưu ý quan trọng

- Dữ liệu mang tính quan sát (observational) — **không** thể tái tạo chính
  xác lại từ đầu.
- Giá trị tiền tệ tính bằng **Peso Colombia (COP)**.
- Thang điểm hài lòng khách hàng: **6 mức (1–6)**.
- NPS (Net Promoter Score): có thể nhận giá trị âm/dương theo công thức
  chuẩn (%Promoters − %Detractors).
- Phạm vi địa lý: các thành phố của Colombia (Bogotá, Medellín, Cali,
  Barranquilla, Cartagena, Bucaramanga, Pereira, Manizales, v.v.) — dữ liệu
  **chỉ thuộc một công ty fintech Colombia**, không đại diện toàn bộ khu
  vực Mỹ Latinh, dù được định vị là nghiên cứu mẫu cho khu vực này.
- Một số cột có thể chứa giá trị **null** (ví dụ `credit_utilization_ratio`
  khi khách hàng không có thẻ tín dụng; `feature_requests`/`complaint_topics`
  khi không có phản hồi mở).

### Gợi ý sử dụng

- Phân tích & dự báo churn (cột `churn_probability` là target gợi ý).
- Phân khúc khách hàng (`customer_segment`, `clv_segment`).
- Phân tích hài lòng khách hàng & NPS theo sản phẩm/kênh.
- Phân tích hành vi giao dịch theo thời gian (kết hợp 2 file).
- Mô hình Customer Lifetime Value (CLV).

---

## 2. Data Dictionary

### File 1: `customer_profile_data.csv` (1 dòng / khách hàng — 53 cột)

#### A. Nhân khẩu học (Demographics)

| Cột | Kiểu | Mô tả |
|---|---|---|
| `customer_id` | int | Mã định danh khách hàng (đã ẩn danh, duy nhất). Khóa liên kết với `transactions_data.csv`. |
| `age` | int | Tuổi khách hàng. |
| `gender` | string | Giới tính (Male / Female). |
| `location` | string | Thành phố & tỉnh cư trú (gộp ở cấp thành phố để ẩn danh), ví dụ "Bogotá, Cundinamarca". |
| `income_bracket` | string | Mức thu nhập theo nhóm (Low / Medium / High / Very High). |
| `occupation` | string | Nghề nghiệp (Engineer, Teacher, Nurse, Lawyer, Driver, v.v.). |
| `education_level` | string | Trình độ học vấn (High School / Bachelor / Master / PhD). |
| `marital_status` | string | Tình trạng hôn nhân (Single / Married / Divorced / Widowed). |
| `household_size` | int | Số người trong hộ gia đình. |
| `acquisition_channel` | string | Kênh thu hút khách hàng (Organic / Referral / Paid Ad / Partnership). |
| `customer_segment` | string | Phân khúc hành vi khách hàng (power / regular / occasional / inactive). |

#### B. Sản phẩm sử dụng (Product Adoption)

| Cột | Kiểu | Mô tả |
|---|---|---|
| `savings_account` | bool | Có tài khoản tiết kiệm hay không. |
| `credit_card` | bool | Có thẻ tín dụng hay không. |
| `personal_loan` | bool | Có khoản vay cá nhân hay không. |
| `investment_account` | bool | Có tài khoản đầu tư hay không. |
| `insurance_product` | bool | Có sản phẩm bảo hiểm hay không. |
| `active_products` | int | Tổng số sản phẩm đang hoạt động của khách hàng. |

#### C. Hành vi sử dụng app (Digital Engagement)

| Cột | Kiểu | Mô tả |
|---|---|---|
| `app_logins_frequency` | int | Tần suất đăng nhập app (số lần / kỳ quan sát). |
| `feature_usage_diversity` | int | Số lượng tính năng khác nhau đã sử dụng. |
| `bill_payment_user` | bool | Có sử dụng tính năng thanh toán hóa đơn hay không. |
| `auto_savings_enabled` | bool | Có bật tính năng tiết kiệm tự động hay không. |

#### D. Chỉ số rủi ro & giao dịch tổng hợp (Risk & Transaction Summary)

| Cột | Kiểu | Mô tả |
|---|---|---|
| `credit_utilization_ratio` | float | Tỷ lệ sử dụng hạn mức tín dụng (null nếu không có thẻ tín dụng). |
| `international_transactions` | int | Số giao dịch quốc tế trong kỳ. |
| `failed_transactions` | int | Số giao dịch thất bại. |
| `tx_count` | int | Tổng số giao dịch (bản tổng hợp ban đầu). |
| `avg_tx_value` | float | Giá trị giao dịch trung bình (COP) (bản tổng hợp ban đầu). |
| `total_tx_volume` | int | Tổng khối lượng giao dịch (COP) (bản tổng hợp ban đầu). |
| `first_tx` | date | Ngày giao dịch đầu tiên (bản tổng hợp ban đầu). |
| `last_tx` | date | Ngày giao dịch gần nhất (bản tổng hợp ban đầu). |

#### E. Mức độ hài lòng & NPS (Satisfaction & NPS)

| Cột | Kiểu | Mô tả |
|---|---|---|
| `base_satisfaction` | float | Điểm hài lòng cơ sở (chỉ số nội bộ, dải mở rộng). |
| `tx_satisfaction` | float | Điểm hài lòng liên quan giao dịch. |
| `product_satisfaction` | float | Điểm hài lòng liên quan sản phẩm (thang 0–1). |
| `satisfaction_score` | int | Điểm hài lòng tổng hợp từ khảo sát (thang 1–6). |
| `nps_score` | int | Net Promoter Score của khách hàng (có thể âm). |
| `last_survey_date` | date | Ngày khảo sát gần nhất khách hàng phản hồi. |
| `support_tickets_count` | int | Số yêu cầu hỗ trợ (ticket) đã tạo. |
| `resolved_tickets_ratio` | float | Tỷ lệ ticket hỗ trợ đã được giải quyết. |
| `app_store_rating` | float | Đánh giá app trên store (thang sao, ví dụ 1–5). |
| `feedback_sentiment` | string | Cảm xúc phản hồi định tính (Positive / Neutral / Negative). |
| `feature_requests` | string | Tính năng khách hàng yêu cầu thêm (văn bản mở, có thể null). |
| `complaint_topics` | string | Chủ đề khiếu nại (văn bản mở, có thể null). |

#### F. Phân khúc giá trị & giao dịch chi tiết (CLV & Transaction Detail)

| Cột | Kiểu | Mô tả |
|---|---|---|
| `clv_segment` | string | Phân khúc giá trị khách hàng (Bronze / Silver / Gold / Platinum). |
| `monthly_transaction_count` | float | Số giao dịch trung bình mỗi tháng. |
| `average_transaction_value` | float | Giá trị giao dịch trung bình (COP) — tính lại từ dữ liệu giao dịch đầy đủ. |
| `total_transaction_volume` | int | Tổng khối lượng giao dịch (COP) — tính lại đầy đủ. |
| `transaction_frequency` | float | Tần suất giao dịch (số giao dịch / ngày, trung bình). |
| `last_transaction_date` | date | Ngày giao dịch cuối cùng (tính lại đầy đủ). |
| `preferred_transaction_type` | string | Loại giao dịch thực hiện nhiều nhất (Transfer / Payment / Withdrawal...). |
| `first_transaction_date` | date | Ngày giao dịch đầu tiên (tính lại đầy đủ). |
| `weekend_transaction_ratio` | float | Tỷ lệ giao dịch thực hiện vào cuối tuần. |
| `avg_daily_transactions` | float | Số giao dịch trung bình mỗi ngày. |
| `customer_tenure` | float | Thời gian là khách hàng (tháng), từ ngày đăng ký. |
| `churn_probability` | float | Xác suất rời bỏ dịch vụ (dự báo cửa sổ 30 ngày), thang 0–1. |
| `customer_lifetime_value` | float | Giá trị khách hàng trọn đời (CLV), đơn vị COP. |

### File 2: `transactions_data.csv` (1 dòng / giao dịch — 4 cột)

| Cột | Kiểu | Mô tả |
|---|---|---|
| `customer_id` | int | Mã khách hàng — khóa liên kết với `customer_profile_data.csv`. |
| `date` | date | Ngày thực hiện giao dịch (YYYY-MM-DD). |
| `amount` | int | Giá trị giao dịch (đơn vị: Peso Colombia – COP). |
| `type` | string | Loại giao dịch (Transfer / Payment / Withdrawal, v.v.). |

---

## 3. Nguồn tham khảo

- Bài báo mô tả dataset (ScienceDirect / *Data in Brief*):
  ["A comprehensive dataset of customer behavior in Latin American Fintech"](https://www.sciencedirect.com/science/article/pii/S2352340926000375)
- Mendeley Data (file gốc, DOI): [data.mendeley.com/datasets/mhb4zn3258/1](https://data.mendeley.com/datasets/mhb4zn3258/1)
- Hugging Face (định dạng cho Machine Learning): [huggingface.co/datasets/luisdavidtrejosrojas/cofinfad](https://huggingface.co/datasets/luisdavidtrejosrojas/cofinfad)

> **Lưu ý:** Tài liệu này được tổng hợp từ thông tin công khai về dataset
> (mô tả, phương pháp thu thập, và schema cột quan sát được từ bản preview
> công khai trên Hugging Face). Khi sử dụng cho công việc/nghiên cứu chính
> thức, nên tải trực tiếp file gốc từ Mendeley Data để đối chiếu lại tên
> cột, kiểu dữ liệu và giá trị thực tế trước khi phân tích.
