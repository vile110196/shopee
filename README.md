# HỆ THỐNG KHAI THÁC DỮ LIỆU & PHÂN TÍCH TRUYỀN THÔNG MẠNG XÃ HỘI E-COMMERCE SHOPEE
> **Đồ Án Môn Học:** Khai thác dữ liệu và Truyền thông xã hội (Data Mining & Social Media)  
> **Đơn vị:** Trường Đại học Công nghệ Thông tin - ĐHQG TP.HCM (UIT)

---

## 👥 THÀNH VIÊN NHÓM THỰC HIỆN
| STT | Họ và Tên | MSSV | Vai Trò & Nhiệm Vụ Phụ Trách | Đóng Góp |
|---|---|---|---|---|
| 1 | **Trần Đình Huy** | 24730103 | **Nhóm trưởng** - Thiết kế kiến trúc tổng thể KDD, phát triển backend Flask API, xây dựng mô hình Phân lớp Doanh số & Rủi ro Vận hành (Random Forest, XGBoost), tổng hợp báo cáo. | **100%** |
| 2 | **Lê Thanh Trúc Vi** | 24730150 | **Thành viên** - Thiết kế pipeline Tiền xử lý Dữ liệu Tiếng Việt & Teencode E-Commerce, Xây dựng mô hình Phân tích Sắc thái Cảm xúc NLP (Naive Bayes / TF-IDF), trực quan hóa Word Cloud. | **100%** |
| 3 | **Vũ Hoàng Thiên Ân** | 24730155 | **Thành viên** - Xây dựng mô hình Gom cụm K-Means phân khúc sản phẩm, Khai phá luật kết hợp Apriori giỏ hàng, thu thập & chuẩn hóa bộ dữ liệu Shopee E-Commerce. | **100%** |
| 4 | **Dương Phương Anh** | 24730156 | **Thành viên** - Thiết kế giao diện Web Dashboard UI/UX trực quan, tích hợp biểu đồ phân tích thời gian thực, thực hiện kiểm thử và đánh giá độ đo hiệu suất mô hình (F1, ROC-AUC). | **100%** |

---

## 🚀 HƯỚNG DẪN KHỞI CHẠY NHANH CHỈ VỚI 1 CLICK (NO-CLI)

### Cách 1: Khởi chạy bằng file 1-Click (Khuyên dùng trên Windows)
1. Nhấp đúp chuột vào file **`run.bat`** trong thư mục dự án.
2. Trình duyệt Web sẽ tự động mở tại địa chỉ: **`http://127.0.0.1:5000`**.

### Cách 2: Khởi chạy bằng lệnh Python
```bash
python app.py
```
Sau đó mở trình duyệt và truy cập: `http://127.0.0.1:5000`.

---

## 🌟 CÁC TÍNH NĂNG VÀ PHÂN HỆ KHAI THÁC TRÊN WEB

1. 📊 **Dashboard Toàn Cảnh (Overview Analytics):**
   - Thống kê toàn cảnh hơn 1,200 sản phẩm và 3,500 review mạng xã hội.
   - Biểu đồ Doanh số bán theo từng Ngành hàng Shopee.
   - Biểu đồ Doughnut tỷ lệ Sắc thái Cảm xúc (Tích cực, Trung lập, Tiêu cực).
   - Biểu đồ phân bố điểm đánh giá Rating (1 - 5 sao) và Bảng xếp hạng Top 5 Best-Seller.

2. 🔍 **Dự Đoán Doanh Số & Tiềm Năng Sản Phẩm (Sales & Growth Forecaster):**
   - Ô nhập link Shopee hoặc chọn nhanh các sản phẩm mẫu có sẵn.
   - Ứng dụng mô hình **Random Forest & XGBoost** dự báo số lượng bán kỳ tới và doanh thu ước tính.
   - Giải thích trọng số các yếu tố ảnh hưởng chính (**Feature Importance**) và gợi ý chiến lược kích cầu dành cho gian hàng.

3. 💬 **Khai Phá Cảm Xúc Đánh Giá & MXH (Vietnamese NLP Sentiment Mining):**
   - Khung kiểm tra review tiếng Việt tự do (hỗ trợ giải mã teencode: *sp, shop, auth, fake, ship nhanh, dỏm, ưng ý, móp méo...*).
   - Dự đoán nhãn cảm xúc bằng **Multinomial Naive Bayes**, tính điểm Polarity Score (-1.0 đến +1.0).
   - Tự động phát hiện và tô màu từ khóa khen/chê và nhận diện khía cạnh (Giao hàng, Chất lượng, Đóng gói, Phục vụ, Giá cả).
   - Trực quan hóa **Word Cloud** và xu hướng thảo luận mạng xã hội.

4. ⚠️ **Giám Sát Rủi Ro Vận Hành & Chuỗi Cung Ứng:**
   - Cảnh báo sớm các sản phẩm có nguy cơ hoàn tiền cao, giao trễ hoặc bị khiếu nại chất lượng.
   - Bảng phân tích nguyên nhân gốc rễ (Root Cause) và giải pháp khắc phục.

5. 🧺 **Khai Phá Luật Kết Hợp Giỏ Hàng (Apriori Algorithm):**
   - Khám phá các cặp sản phẩm thường xuyên được mua cùng nhau.
   - Bảng tính toán các chỉ số học thuật: **Support (%)**, **Confidence (%)**, **Lift (Độ tương quan)**.
   - Ứng dụng thiết lập Combo Cross-Selling tăng giá trị trung bình đơn hàng (AOV).

6. 🔮 **Phân Khúc Sản Phẩm Bằng K-Means & 2D PCA (Clustering):**
   - Gom cụm tự động không giám sát thành 4 phân khúc sản phẩm.
   - Biểu đồ phân tán Scatter Plot 2 chiều (PCA Dimensionality Reduction) tương tác trực tiếp.

7. 🧪 **Đấu Trường Đánh Giá Hiệu Suất Mô Hình (Benchmark Arena):**
   - Bảng so sánh đa chiều các thuật toán: **Decision Tree, Random Forest, XGBoost, Naive Bayes**.
   - Trực quan hóa **Ma trận nhầm lẫn (Confusion Matrix)** dạng Heatmap và **Đường cong ROC-AUC**.

8. 📖 **Báo Cáo Nghiệm Thu & Cơ Sở Lý Thuyết Học Thuật:**
   - Trang báo cáo đầy đủ chương hồi, công thức toán học và hướng dẫn câu hỏi vấn đáp bảo vệ đồ án trước hội đồng.

---

## 🛠️ CẤU TRÚC THƯ MỤC NGUỒN (PROJECT STRUCTURE)
```
KHAI THÁC DỮ LIỆU SHOPEE/
├── app.py                      # Flask Web Engine & REST API Controller
├── data_processing.py          # Pipeline tiền xử lý tiếng Việt, từ điển Teencode & Dataset Generator
├── model_training.py           # Script huấn luyện, đánh giá và lưu trữ các mô hình ML
├── run.bat                     # File 1-click khởi chạy ứng dụng trên Windows
├── README.md                   # Tài liệu hướng dẫn sử dụng và báo cáo tổng quan
├── data/                       # Thư mục lưu trữ dữ liệu E-Commerce Shopee
│   ├── shopee_products.csv     # 1,200 sản phẩm với đầy đủ chỉ số kinh doanh & vận hành
│   ├── shopee_reviews.csv      # 3,500 đánh giá & thảo luận MXH (Shopee, TikTok, FB)
│   └── shopee_transactions.csv # 1,500 giao dịch giỏ hàng để chạy Apriori
├── models/                     # Thư mục chứa các mô hình đã được huấn luyện (.joblib / .json)
│   ├── sentiment_model.joblib  # Mô hình phân loại cảm xúc NLP (Naive Bayes)
│   ├── tfidf_vectorizer.joblib # TF-IDF N-Grams Vectorizer tiếng Việt
│   ├── growth_classifier.joblib# Mô hình phân loại tiềm năng (Random Forest / XGBoost)
│   ├── sales_regressor.joblib  # Mô hình dự báo doanh số tháng
│   ├── risk_classifier.joblib   # Mô hình dự báo rủi ro hoàn hàng
│   ├── kmeans_model.joblib     # Mô hình gom cụm K-Means
│   ├── clustering_data.json    # Dữ liệu phân bố cụm và tọa độ 2D PCA
│   ├── market_basket_rules.json# Danh sách luật kết hợp Apriori
│   └── model_benchmarks.json   # Toàn bộ số liệu Confusion Matrix & ROC-AUC
├── templates/                  # Giao diện HTML (Jinja2 Templates)
│   ├── base.html               # Khung sườn chung, Header nhóm sinh viên, Navigation
│   ├── index.html              # Dashboard chính với 7 tab tương tác trực tiếp
│   ├── report.html             # Báo cáo đồ án nghiệm thu chuẩn học thuật UIT
│   └── theory.html             # Tổng hợp cơ sở lý thuyết bám sát slide môn học
└── static/                     # Tài nguyên tĩnh
    ├── css/
    │   └── style.css           # Giao diện Cyberpunk Shopee Dark Theme cao cấp
    └── js/
        └── app.js              # Xử lý tương tác AJAX, vẽ Chart.js, Confetti & Gauge
```

---

## 🎓 CÔNG NGHỆ & THUẬT TOÁN SỬ DỤNG
- **Ngôn ngữ & Thư viện lõi:** Python 3.11, Scikit-learn, XGBoost, Pandas, NumPy, Joblib, Underthesea.
- **Web Backend & REST API:** Flask, Werkzeug, Jinja2.
- **Frontend & Visualizations:** HTML5 Semantic, CSS3 Modern Glassmorphism Variables, Chart.js, FontAwesome 6, Canvas Confetti.
