# HỆ THỐNG KHAI THÁC DỮ LIỆU & PHÂN TÍCH TRUYỀN THÔNG E-COMMERCE

> Đồ án môn học Khai thác dữ liệu và Truyền thông xã hội — UIT, ĐHQG TP.HCM

## Thành viên

| STT | Họ và tên | MSSV | Phần việc chính |
|---|---|---|---|
| 1 | Trần Đình Huy | 24730103 | Kiến trúc KDD, Flask API và hồi quy tốc độ bán |
| 2 | Lê Thanh Trúc Vi | 24730150 | Tiền xử lý tiếng Việt, TF-IDF, sentiment và ABSA |
| 3 | Vũ Hoàng Thiên Ân | 24730155 | Nhập dữ liệu, K-Means, PCA và Apriori |
| 4 | Dương Phương Anh | 24730156 | Dashboard, kiểm thử và đánh giá mô hình |

## Dữ liệu sử dụng

Project dùng ba nguồn công khai độc lập; downloader ghim version 1 và kiểm SHA-256 trước khi xử lý.

| Nhánh phân tích | Nguồn | Dữ liệu sau chuẩn hóa | Cách dùng |
|---|---|---:|---|
| Sản phẩm | [Shopee Sales Apr–May 2023](https://www.kaggle.com/datasets/yoongsin/shopee-sample-data) | 16.563 sản phẩm từ 20.312 snapshot, 9.221 shop | KPI, hồi quy sales velocity, K-Means/PCA |
| Review | [ABSA Vietnamese](https://www.kaggle.com/datasets/cthng123/absa-vietnamese) | 11.700 review; 9.326 có nhãn sentiment tổng hợp | Sentiment ba lớp và tám mô hình khía cạnh |
| Đơn hàng | [Shopee Seller Transaction](https://www.kaggle.com/datasets/nugrahmaindonesa/shopee-seller-transaction) | 325 đơn; 67 đơn nhiều sản phẩm, 261 đơn hoàn thành | Apriori sau khi loại dữ liệu người mua |

Các nguồn không có khóa chung. Project không nối review hay đơn hàng vào sản phẩm. Snapshot sản phẩm thuộc Shopee Malaysia và dùng MYR; ABSA là review giày bằng tiếng Việt; giao dịch thuộc một seller Indonesia.

Thông tin chi tiết về version, ngày cập nhật, hash nguồn, quy tắc chuẩn hóa và giới hạn nằm trong [`data/provenance.json`](data/provenance.json).

## Chức năng

1. Dashboard mô tả 16.563 listing: lượt bán tích lũy, ngành hàng, rating và dữ liệu thiếu.
2. Sentiment tổng quát bằng TF-IDF với Multinomial Naive Bayes/Logistic Regression.
3. ABSA bằng tám mô hình cho giá, giao hàng, hình thức, chất lượng, kích cỡ, dịch vụ shop, trải nghiệm chung và khía cạnh khác.
4. Hồi quy tốc độ bán/ngày trên 2.263 sản phẩm có delta bán không âm giữa ít nhất hai ngày snapshot.
5. Kiểm tra chất lượng listing bằng quy tắc trên trường quan sát; không gắn nhãn rủi ro kinh doanh giả.
6. K-Means bốn cụm trên tám thuộc tính, kèm phép chiếu PCA hai chiều.
7. Apriori hai-itemset trên 261 đơn hoàn thành đã ẩn danh.
8. Báo cáo nghiệm thu và trang lý thuyết lấy metric trực tiếp từ artifact hiện hành.

Project không có target tăng trưởng, hoàn hàng hoặc rủi ro vận hành. Vì vậy không huấn luyện hay hiển thị classifier cho ba khái niệm đó.

## Khởi chạy trên Windows

### Cách 1: chạy tự động

Nhấp đúp [`run.bat`](run.bat). Script tạo `.venv`, cài dependency, tải dữ liệu nếu chưa có, xác minh dữ liệu/artifact rồi mở Flask tại <http://127.0.0.1:5000>.

Lần đầu cần Internet để tải package, ba archive Kaggle và asset giao diện từ CDN. Kaggle public API của ba dataset này không yêu cầu file credential.

### Cách 2: dòng lệnh

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe import_public_data.py --download
.\.venv\Scripts\python.exe model_training.py
.\.venv\Scripts\python.exe app.py
```

Yêu cầu Python 3.11 trở lên. `scikit-learn==1.9.0` được khóa vì artifact joblib phụ thuộc phiên bản thư viện. Underthesea là tùy chọn:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-optional.txt
```

Nếu không cài Underthesea, pipeline dùng tokenizer regex tích hợp.

## Nhập và bảo vệ dữ liệu

```powershell
.\.venv\Scripts\python.exe import_public_data.py --download
```

Pipeline thực hiện:

- kiểm hash archive và từng CSV nguồn;
- nhận dạng sản phẩm bằng `shop_id + item_id`, giữ snapshot mới nhất;
- tính `daily_sold_rate = delta(total_sold) / số ngày quan sát`, loại hai delta âm;
- giữ split gốc và tám nhãn ABSA `-1/0/1/2`;
- tạo sentiment tổng hợp từ nhãn General hoặc các nhãn người gán đồng thuận, để trống trường hợp xung đột;
- chỉ xuất allowlist trường giao dịch, băm order ID và xóa raw transaction archive sau import;
- ghi hash các CSV kết quả vào `data/provenance.json`.

Không chạy `data_processing.py` để tạo dữ liệu: generator mô phỏng cũ đã bị vô hiệu hóa để không thể ghi đè CSV thật.

## Huấn luyện và đánh giá

```powershell
.\.venv\Scripts\python.exe model_training.py
```

- Sentiment: train + validation của nguồn dùng để chọn mô hình; test gốc chỉ dùng đánh giá. Câu trùng chéo split và câu có nhãn xung đột bị loại.
- ABSA: tám Logistic Regression đa lớp, cùng chính sách chống rò rỉ.
- Sales velocity: Random Forest và Extra Trees học `log1p(daily_sold_rate)`; chọn mô hình theo MAE trên holdout seed 42.
- K-Means: điền median, `log1p` trường lệch phải, StandardScaler, `k=4`; PCA chỉ dùng hiển thị.
- Apriori: chỉ dùng đơn `Hoàn thành`, support tối thiểu 1,5% và confidence tối thiểu 15%.

Metric hiện hành được ghi trong [`models/model_benchmarks.json`](models/model_benchmarks.json). Artifact tăng trưởng/rủi ro của pipeline mô phỏng cũ không còn được sử dụng.

## REST API

| Method | Endpoint | Chức năng |
|---|---|---|
| GET | `/` | Dashboard |
| GET | `/report` | Báo cáo nghiệm thu |
| GET | `/theory` | Cơ sở lý thuyết |
| GET | `/api/overview_stats` | KPI, phân bố và độ phủ dữ liệu |
| GET | `/api/product_lookup?reference=...` | Tra bằng product_id, item_id hoặc URL thật trong snapshot |
| POST | `/api/predict_product` | Ước lượng SP/ngày và nhịp 30 ngày |
| POST | `/api/analyze_sentiment` | Sentiment và tám kết quả ABSA |
| GET | `/api/wordcloud` | Tần suất từ trong review |
| GET | `/api/model_benchmarks` | Metric và provenance |
| GET | `/api/clustering` | Profile cụm và tọa độ PCA |
| GET | `/api/market_basket` | Luật kết hợp |
| GET | `/api/sample_products` | Listing mẫu thật |
| GET | `/api/sample_comments` | Review mẫu thật |

Ví dụ:

```powershell
Invoke-RestMethod "http://127.0.0.1:5000/api/product_lookup?reference=1710706663"

Invoke-RestMethod -Method Post `
  -Uri "http://127.0.0.1:5000/api/analyze_sentiment" `
  -ContentType "application/json" `
  -Body '{"text":"Giày đẹp, giao nhanh, shop tư vấn tốt"}'
```

## Kiểm thử

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Test kiểm schema và số dòng, hash provenance, việc loại PII, khả năng nạp artifact, split sentiment không rò rỉ, tổng số phần tử K-Means, luật Apriori và hợp đồng HTML/JSON của Flask.

## Cấu trúc chính

```text
shopee/
├── import_public_data.py
├── data_processing.py
├── model_training.py
├── app.py
├── run.bat
├── requirements.txt
├── data/
│   ├── provenance.json
│   ├── shopee_products.csv
│   ├── shopee_reviews.csv
│   └── shopee_transactions.csv
├── models/
│   ├── sentiment_model.joblib
│   ├── aspect_models.joblib
│   ├── sales_regressor.joblib
│   ├── kmeans_model.joblib
│   └── *.json
├── templates/
├── static/
└── tests/
```

## Giới hạn diễn giải

- `historical_sold` là tổng bán tích lũy tại snapshot, không phải bán trong tháng.
- Nhịp 30 ngày là phép nhân tốc độ ước lượng với 30 để dễ đọc, không phải dự báo chuỗi thời gian.
- R² của hồi quy được báo nguyên trạng; không suy diễn thành độ chính xác phần trăm.
- Tám dự đoán ABSA có thể trả “Không đề cập”; confidence là xác suất mô hình, không phải xác suất đúng đã hiệu chỉnh.
- Feature importance không biểu diễn quan hệ nhân quả.
- Lift lớn hơn 1 chỉ biểu diễn đồng xuất hiện trong 261 đơn hoàn thành của một seller.
- Ứng dụng không có đăng nhập, cơ sở dữ liệu giao dịch trực tuyến hoặc kết nối Shopee API; chỉ nên chạy cục bộ cho đồ án.
