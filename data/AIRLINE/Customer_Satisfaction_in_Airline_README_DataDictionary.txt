================================================================================
Customer Satisfaction in Airline (Invistico Airline Dataset)
README & DATA DICTIONARY
================================================================================

--------------------------------------------------------------------------------
1. README
--------------------------------------------------------------------------------

TÊN DATASET   : Customer Satisfaction in Airline
                (còn được biết đến rộng rãi với tên "Invistico Airline Dataset" /
                "Airline Passenger Satisfaction")
NGUỒN CHÍNH   : Kaggle — tác giả đăng tải: yakhyojon
                https://www.kaggle.com/datasets/yakhyojon/customer-satisfaction-in-airline
NGUỒN GỐC SỐ LIỆU : Khảo sát hài lòng khách hàng của một hãng hàng không hư cấu/
                ẩn danh tên "Invistico Airlines" (dữ liệu cũng được lưu hành dưới
                tên file gốc "Invistico_Airline.csv" trên nhiều bản đăng lại khác
                trên Kaggle, ví dụ: sjleshrac/airlines-customer-satisfaction,
                likhari/invistico-airline-dataset, mohaimenalrashid/invistico-airline).
THỂ LOẠI/TAG  : Tabular, Logistic Regression, Decision Tree, Aviation.
TÊN FILE BẠN CUNG CẤP:
   Customer_Satisfaction_in_Airline_export_2026-06-16_07-20-47.csv

MÔ TẢ TỔNG QUAN
----------------
Dataset chứa kết quả khảo sát mức độ hài lòng của 129.880 khách hàng đã đi
máy bay với hãng hàng không "Invistico Airlines" (tên ẩn danh). Mỗi dòng đại
diện cho một lượt khảo sát của một khách hàng, gồm thông tin nhân khẩu học cơ
bản, đặc điểm chuyến đi, và 14 tiêu chí đánh giá dịch vụ trong suốt hành trình
(thang điểm 0–5), cùng với biến mục tiêu (target) là "satisfaction"
(hài lòng / không hài lòng).

Dataset thường được dùng cho các bài toán:
  - Phân loại nhị phân (binary classification): dự đoán khách hàng có hài
    lòng hay không (Logistic Regression, Decision Tree, Random Forest...).
  - Phân tích yếu tố ảnh hưởng đến sự hài lòng (feature importance, SHAP).
  - Phân khúc khách hàng theo loại hành khách, hạng vé, mục đích chuyến đi.

THỐNG KÊ THỰC TẾ (đã kiểm tra trực tiếp trên file bạn cung cấp)
------------------------------------------------------------------
- Tổng số dòng           : 129.880 bản ghi
- Tổng số cột            : 22 cột (1 biến mục tiêu + 21 biến đầu vào)
- satisfaction           : satisfied = 71.087 | dissatisfied = 58.793
- Customer Type          : Loyal Customer = 106.100 | disloyal Customer = 23.780
- Type of Travel         : Business travel = 89.693 | Personal Travel = 40.187
- Class                  : Business = 62.160 | Eco = 58.309 | Eco Plus = 9.411
- Age                    : 7 – 85 tuổi
- Flight Distance        : 50 – 6.951 (đơn vị khoảng cách, thường là dặm/miles)
- Các tiêu chí đánh giá dịch vụ (thang điểm) : chủ yếu 0–5 (riêng "Baggage
  handling" quan sát được min = 1)
- Departure Delay in Minutes : 0 – 1.592 phút, không có giá trị thiếu
- Arrival Delay in Minutes   : 0 – 1.584 phút, CÓ 393 giá trị thiếu (missing)

LƯU Ý: Đây là số liệu thống kê tính trực tiếp từ file CSV bạn đã upload,
không phải số liệu mô tả trên trang Kaggle (Kaggle hiện chặn truy cập tự
động nên không lấy được mô tả/schema gốc đầy đủ — phần Data Dictionary dưới
đây được xây dựng dựa trên cấu trúc cột thực tế của file + cách diễn giải
chuẩn được dùng phổ biến trong các phân tích công khai về dataset này).

KHÁC BIỆT SO VỚI BẢN GỐC PHỔ BIẾN TRÊN KAGGLE
------------------------------------------------
Bản phổ biến nhất của dataset này (Invistico_Airline.csv, ~129.880 dòng,
23 cột) thường có thêm cột "Gender" (Male/Female) ở đầu. Bản bạn cung cấp
KHÔNG có cột Gender — như vậy file của bạn có 22 cột (1 target + 21 input),
ít hơn 1 cột nhân khẩu học so với bản gốc phổ biến. Ngoài ra một số bản khác
của dataset này có cột "Inflight service" thay vì "Online support" — file
của bạn dùng tên cột "Online support", khớp với bản gốc "Invistico Airline"
nguyên thủy (không phải bản phái sinh "Airline Passenger Satisfaction" của
teejmahal20, vốn có thêm ID/Gender/Inflight service và bỏ Online support).

GIỚI HẠN / LƯU Ý QUAN TRỌNG
-----------------------------
- Tên hãng hàng không "Invistico Airlines" được cho là ẩn danh/hư cấu —
  không có xác nhận công khai đây là dữ liệu thật của một hãng cụ thể.
- Đơn vị "Flight Distance" không được công bố rõ ràng trên Kaggle gốc,
  thường được suy luận là dặm (miles) dựa theo các phân tích cộng đồng.
- Thang điểm đánh giá dịch vụ 0–5, trong đó 0 thường được hiểu là
  "Not Applicable / không áp dụng" (ví dụ khách không dùng wifi).
- Cột "Arrival Delay in Minutes" có giá trị thiếu (missing) — cần xử lý
  (loại bỏ hoặc impute) trước khi đưa vào mô hình.
- Dataset đã qua tiền xử lý cơ bản (không có giá trị âm, không có ký tự lỗi),
  phù hợp sử dụng ngay cho bài tập phân loại/học máy.

GỢI Ý SỬ DỤNG
-------------
- Bài toán phân loại: dự đoán "satisfaction" (satisfied / dissatisfied) từ
  các biến còn lại — phù hợp cho Logistic Regression, Decision Tree,
  Random Forest, XGBoost.
- Phân tích yếu tố ảnh hưởng mạnh nhất đến sự hài lòng (ví dụ: Inflight
  wifi service, Online boarding, Ease of Online booking thường được ghi
  nhận có ảnh hưởng lớn trong các phân tích công khai về dataset này).
- Phân khúc theo Class / Type of Travel / Customer Type để đưa ra khuyến
  nghị cải thiện dịch vụ theo từng nhóm khách hàng.
- Phân tích tương quan giữa Departure Delay và Arrival Delay (tương quan
  rất cao trong các nghiên cứu công khai, ~0.96).

--------------------------------------------------------------------------------
2. DATA DICTIONARY
--------------------------------------------------------------------------------
(1 dòng = 1 lượt khảo sát của 1 khách hàng; 22 cột; 129.880 dòng)

A. BIẾN MỤC TIÊU (TARGET)
----------------------------
satisfaction              | string | Kết quả hài lòng tổng thể của khách hàng.
                                      Giá trị: "satisfied" / "dissatisfied".

B. THÔNG TIN KHÁCH HÀNG & CHUYẾN ĐI (CUSTOMER & TRIP INFO)
--------------------------------------------------------------
Customer Type             | string | Loại khách hàng. Giá trị: "Loyal Customer"
                                      (khách hàng thân thiết) / "disloyal Customer"
                                      (khách hàng không thân thiết).
Age                        | int    | Tuổi khách hàng. Khoảng giá trị: 7–85.
Type of Travel             | string | Mục đích chuyến đi. Giá trị: "Business travel"
                                      (công vụ) / "Personal Travel" (cá nhân).
Class                      | string | Hạng vé. Giá trị: "Business" / "Eco" /
                                      "Eco Plus".
Flight Distance            | int    | Khoảng cách chuyến bay (đơn vị suy luận:
                                      dặm/miles). Khoảng giá trị: 50–6.951.

C. ĐÁNH GIÁ DỊCH VỤ TRONG CHUYẾN BAY (SERVICE RATINGS — thang điểm 0–5)
----------------------------------------------------------------------------
Ghi chú chung: 0 = Không áp dụng/không đánh giá; 1 = Rất kém; 5 = Rất tốt.

Seat comfort                       | int | Mức độ hài lòng về độ thoải mái của ghế.
Departure/Arrival time convenient  | int | Mức độ thuận tiện của giờ khởi hành/đến.
Food and drink                     | int | Mức độ hài lòng về đồ ăn, thức uống.
Gate location                      | int | Mức độ thuận tiện của vị trí cổng ra máy bay.
Inflight wifi service              | int | Mức độ hài lòng về dịch vụ wifi trên máy bay.
Inflight entertainment             | int | Mức độ hài lòng về giải trí trên máy bay.
Online support                     | int | Mức độ hài lòng về hỗ trợ trực tuyến (CSKH online).
Ease of Online booking             | int | Mức độ dễ dàng khi đặt vé trực tuyến.
On-board service                   | int | Mức độ hài lòng về dịch vụ phục vụ trên máy bay.
Leg room service                   | int | Mức độ hài lòng về không gian để chân.
Baggage handling                   | int | Mức độ hài lòng về xử lý hành lý (khoảng giá trị quan sát: 1–5).
Checkin service                    | int | Mức độ hài lòng về dịch vụ làm thủ tục check-in.
Cleanliness                        | int | Mức độ hài lòng về độ sạch sẽ trên máy bay.
Online boarding                    | int | Mức độ hài lòng về quy trình lên máy bay trực tuyến (boarding pass điện tử).

D. THÔNG TIN TRỄ CHUYẾN (DELAY INFO)
----------------------------------------
Departure Delay in Minutes | int   | Số phút trễ giờ khởi hành. Khoảng giá trị:
                                      0–1.592. Không có giá trị thiếu.
Arrival Delay in Minutes   | float | Số phút trễ giờ hạ cánh. Khoảng giá trị:
                                      0–1.584. CÓ 393 giá trị thiếu (missing) —
                                      cần xử lý trước khi phân tích/mô hình hóa.

--------------------------------------------------------------------------------
3. NGUỒN THAM KHẢO
--------------------------------------------------------------------------------
- Trang dataset chính (khớp tên file bạn cung cấp):
  https://www.kaggle.com/datasets/yakhyojon/customer-satisfaction-in-airline

- Các bản đăng lại / phái sinh phổ biến của cùng dữ liệu gốc trên Kaggle:
  https://www.kaggle.com/datasets/sjleshrac/airlines-customer-satisfaction
  https://www.kaggle.com/datasets/likhari/invistico-airline-dataset
  https://www.kaggle.com/datasets/mohaimenalrashid/invistico-airline

- Phân tích công khai tham khảo về dataset (bối cảnh, ý nghĩa biến, NPS...):
  https://harshalvaza.medium.com/invistico-airlines-understanding-customer-satisfaction-6108b500e592
  https://ruruth.github.io/Airline-Passenger-Satisfaction-Analysis/

LƯU Ý: Trang Kaggle chặn truy cập tự động nên không lấy được trực tiếp phần
mô tả gốc và Data Dictionary chính thức do tác giả công bố. Tài liệu này được
xây dựng bằng cách: (1) đối chiếu tên file/tên cột thực tế trong dữ liệu bạn
tải lên, (2) tham khảo các bài phân tích công khai về cùng nguồn dữ liệu gốc
"Invistico Airline", và (3) tính toán trực tiếp số liệu thống kê (số dòng,
giá trị min/max, số lượng theo nhóm) từ chính file CSV bạn đã cung cấp.
Khuyến nghị: nếu cần đối chiếu chính thức, hãy truy cập trực tiếp trang
Kaggle nêu trên để xem mô tả gốc của tác giả.
================================================================================
