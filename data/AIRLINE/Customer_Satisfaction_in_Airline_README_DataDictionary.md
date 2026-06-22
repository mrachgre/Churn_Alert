# Customer Satisfaction in Airline (Invistico Airline Dataset)

## README & Data Dictionary

---

## 1. README

| | |
|---|---|
| **Tên dataset** | Customer Satisfaction in Airline (còn gọi là "Invistico Airline Dataset" / "Airline Passenger Satisfaction") |
| **Nguồn chính** | [Kaggle — yakhyojon/customer-satisfaction-in-airline](https://www.kaggle.com/datasets/yakhyojon/customer-satisfaction-in-airline) |
| **Nguồn gốc số liệu** | Khảo sát hài lòng khách hàng của hãng hàng không ẩn danh "Invistico Airlines" |
| **Tag** | Tabular, Logistic Regression, Decision Tree, Aviation |
| **File bạn cung cấp** | `Customer_Satisfaction_in_Airline_export_2026-06-16_07-20-47.csv` |

> Dữ liệu cũng lưu hành dưới tên gốc `Invistico_Airline.csv` trên nhiều bản
> đăng lại khác trên Kaggle, ví dụ: `sjleshrac/airlines-customer-satisfaction`,
> `likhari/invistico-airline-dataset`, `mohaimenalrashid/invistico-airline`.

### Mô tả tổng quan

Dataset chứa kết quả khảo sát mức độ hài lòng của **129.880 khách hàng** đã
đi máy bay với hãng hàng không "Invistico Airlines" (tên ẩn danh). Mỗi dòng
đại diện cho một lượt khảo sát của một khách hàng, gồm thông tin nhân khẩu
học cơ bản, đặc điểm chuyến đi, và **14 tiêu chí đánh giá dịch vụ** trong
suốt hành trình (thang điểm 0–5), cùng biến mục tiêu (target) là
`satisfaction` (hài lòng / không hài lòng).

Dataset thường được dùng cho:
- **Phân loại nhị phân**: dự đoán khách hàng có hài lòng hay không
  (Logistic Regression, Decision Tree, Random Forest...).
- **Phân tích yếu tố ảnh hưởng** đến sự hài lòng (feature importance, SHAP).
- **Phân khúc khách hàng** theo loại hành khách, hạng vé, mục đích chuyến đi.

### Thống kê thực tế (đã kiểm tra trực tiếp trên file bạn cung cấp)

| Chỉ số | Giá trị |
|---|---|
| Tổng số dòng | 129.880 bản ghi |
| Tổng số cột | 22 cột (1 biến mục tiêu + 21 biến đầu vào) |
| `satisfaction` | satisfied = 71.087 \| dissatisfied = 58.793 |
| `Customer Type` | Loyal Customer = 106.100 \| disloyal Customer = 23.780 |
| `Type of Travel` | Business travel = 89.693 \| Personal Travel = 40.187 |
| `Class` | Business = 62.160 \| Eco = 58.309 \| Eco Plus = 9.411 |
| `Age` | 7 – 85 tuổi |
| `Flight Distance` | 50 – 6.951 (đơn vị khoảng cách, thường là dặm/miles) |
| Tiêu chí đánh giá dịch vụ | chủ yếu thang 0–5 (riêng `Baggage handling` quan sát được min = 1) |
| `Departure Delay in Minutes` | 0 – 1.592 phút, không có giá trị thiếu |
| `Arrival Delay in Minutes` | 0 – 1.584 phút, **có 393 giá trị thiếu** |

> **Lưu ý:** Đây là số liệu thống kê tính trực tiếp từ file CSV bạn đã
> upload, không phải số liệu mô tả trên trang Kaggle (Kaggle hiện chặn truy
> cập tự động nên không lấy được mô tả/schema gốc đầy đủ — phần Data
> Dictionary dưới đây được xây dựng dựa trên cấu trúc cột thực tế của file
> + cách diễn giải chuẩn được dùng phổ biến trong các phân tích công khai
> về dataset này).

### Khác biệt so với bản gốc phổ biến trên Kaggle

Bản phổ biến nhất của dataset này (`Invistico_Airline.csv`, ~129.880 dòng,
23 cột) thường có thêm cột **`Gender`** (Male/Female) ở đầu. Bản bạn cung
cấp **không có** cột Gender — như vậy file của bạn có 22 cột (1 target + 21
input), ít hơn 1 cột nhân khẩu học so với bản gốc phổ biến. Ngoài ra một số
bản khác của dataset này có cột "Inflight service" thay vì "Online support"
— file của bạn dùng tên cột **"Online support"**, khớp với bản gốc
"Invistico Airline" nguyên thủy (không phải bản phái sinh "Airline Passenger
Satisfaction" của teejmahal20, vốn có thêm ID/Gender/Inflight service và bỏ
Online support).

### Giới hạn / Lưu ý quan trọng

- Tên hãng hàng không "Invistico Airlines" được cho là ẩn danh/hư cấu —
  không có xác nhận công khai đây là dữ liệu thật của một hãng cụ thể.
- Đơn vị `Flight Distance` không được công bố rõ ràng trên Kaggle gốc,
  thường được suy luận là dặm (miles) dựa theo các phân tích cộng đồng.
- Thang điểm đánh giá dịch vụ 0–5, trong đó **0** thường được hiểu là
  "Not Applicable / không áp dụng" (ví dụ khách không dùng wifi).
- Cột `Arrival Delay in Minutes` có giá trị thiếu (missing) — cần xử lý
  (loại bỏ hoặc impute) trước khi đưa vào mô hình.
- Dataset đã qua tiền xử lý cơ bản (không có giá trị âm, không có ký tự
  lỗi), phù hợp sử dụng ngay cho bài tập phân loại/học máy.

### Gợi ý sử dụng

- Bài toán phân loại: dự đoán `satisfaction` (satisfied / dissatisfied) từ
  các biến còn lại — phù hợp cho Logistic Regression, Decision Tree, Random
  Forest, XGBoost.
- Phân tích yếu tố ảnh hưởng mạnh nhất đến sự hài lòng (ví dụ: `Inflight
  wifi service`, `Online boarding`, `Ease of Online booking` thường được
  ghi nhận có ảnh hưởng lớn trong các phân tích công khai về dataset này).
- Phân khúc theo `Class` / `Type of Travel` / `Customer Type` để đưa ra
  khuyến nghị cải thiện dịch vụ theo từng nhóm khách hàng.
- Phân tích tương quan giữa `Departure Delay` và `Arrival Delay` (tương
  quan rất cao trong các nghiên cứu công khai, ~0.96).

---

## 2. Data Dictionary

(1 dòng = 1 lượt khảo sát của 1 khách hàng; 22 cột; 129.880 dòng)

### A. Biến mục tiêu (Target)

| Cột | Kiểu | Mô tả |
|---|---|---|
| `satisfaction` | string | Kết quả hài lòng tổng thể của khách hàng. Giá trị: "satisfied" / "dissatisfied". |

### B. Thông tin khách hàng & chuyến đi (Customer & Trip Info)

| Cột | Kiểu | Mô tả |
|---|---|---|
| `Customer Type` | string | Loại khách hàng. "Loyal Customer" (thân thiết) / "disloyal Customer" (không thân thiết). |
| `Age` | int | Tuổi khách hàng. Khoảng giá trị: 7–85. |
| `Type of Travel` | string | Mục đích chuyến đi. "Business travel" (công vụ) / "Personal Travel" (cá nhân). |
| `Class` | string | Hạng vé. "Business" / "Eco" / "Eco Plus". |
| `Flight Distance` | int | Khoảng cách chuyến bay (đơn vị suy luận: dặm/miles). Khoảng giá trị: 50–6.951. |

### C. Đánh giá dịch vụ trong chuyến bay (Service Ratings — thang điểm 0–5)

> Ghi chú chung: 0 = Không áp dụng/không đánh giá; 1 = Rất kém; 5 = Rất tốt.

| Cột | Kiểu | Mô tả |
|---|---|---|
| `Seat comfort` | int | Mức độ hài lòng về độ thoải mái của ghế. |
| `Departure/Arrival time convenient` | int | Mức độ thuận tiện của giờ khởi hành/đến. |
| `Food and drink` | int | Mức độ hài lòng về đồ ăn, thức uống. |
| `Gate location` | int | Mức độ thuận tiện của vị trí cổng ra máy bay. |
| `Inflight wifi service` | int | Mức độ hài lòng về dịch vụ wifi trên máy bay. |
| `Inflight entertainment` | int | Mức độ hài lòng về giải trí trên máy bay. |
| `Online support` | int | Mức độ hài lòng về hỗ trợ trực tuyến (CSKH online). |
| `Ease of Online booking` | int | Mức độ dễ dàng khi đặt vé trực tuyến. |
| `On-board service` | int | Mức độ hài lòng về dịch vụ phục vụ trên máy bay. |
| `Leg room service` | int | Mức độ hài lòng về không gian để chân. |
| `Baggage handling` | int | Mức độ hài lòng về xử lý hành lý (khoảng giá trị quan sát: 1–5). |
| `Checkin service` | int | Mức độ hài lòng về dịch vụ làm thủ tục check-in. |
| `Cleanliness` | int | Mức độ hài lòng về độ sạch sẽ trên máy bay. |
| `Online boarding` | int | Mức độ hài lòng về quy trình lên máy bay trực tuyến (boarding pass điện tử). |

### D. Thông tin trễ chuyến (Delay Info)

| Cột | Kiểu | Mô tả |
|---|---|---|
| `Departure Delay in Minutes` | int | Số phút trễ giờ khởi hành. Khoảng giá trị: 0–1.592. Không có giá trị thiếu. |
| `Arrival Delay in Minutes` | float | Số phút trễ giờ hạ cánh. Khoảng giá trị: 0–1.584. **Có 393 giá trị thiếu** — cần xử lý trước khi phân tích/mô hình hóa. |

---

## 3. Nguồn tham khảo

- Trang dataset chính (khớp tên file bạn cung cấp):
  [kaggle.com/datasets/yakhyojon/customer-satisfaction-in-airline](https://www.kaggle.com/datasets/yakhyojon/customer-satisfaction-in-airline)
- Các bản đăng lại / phái sinh phổ biến của cùng dữ liệu gốc trên Kaggle:
  - [sjleshrac/airlines-customer-satisfaction](https://www.kaggle.com/datasets/sjleshrac/airlines-customer-satisfaction)
  - [likhari/invistico-airline-dataset](https://www.kaggle.com/datasets/likhari/invistico-airline-dataset)
  - [mohaimenalrashid/invistico-airline](https://www.kaggle.com/datasets/mohaimenalrashid/invistico-airline)
- Phân tích công khai tham khảo về dataset (bối cảnh, ý nghĩa biến, NPS...):
  - [Invistico Airlines - Understanding Customer Satisfaction (Medium)](https://harshalvaza.medium.com/invistico-airlines-understanding-customer-satisfaction-6108b500e592)
  - [Airline Passenger Satisfaction Analysis](https://ruruth.github.io/Airline-Passenger-Satisfaction-Analysis/)

> **Lưu ý:** Trang Kaggle chặn truy cập tự động nên không lấy được trực tiếp
> phần mô tả gốc và Data Dictionary chính thức do tác giả công bố. Tài liệu
> này được xây dựng bằng cách: (1) đối chiếu tên file/tên cột thực tế trong
> dữ liệu bạn tải lên, (2) tham khảo các bài phân tích công khai về cùng
> nguồn dữ liệu gốc "Invistico Airline", và (3) tính toán trực tiếp số liệu
> thống kê (số dòng, giá trị min/max, số lượng theo nhóm) từ chính file CSV
> bạn đã cung cấp. Khuyến nghị: nếu cần đối chiếu chính thức, hãy truy cập
> trực tiếp trang Kaggle nêu trên để xem mô tả gốc của tác giả.
