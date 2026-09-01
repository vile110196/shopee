# -*- coding: utf-8 -*-
"""
=============================================================================
ĐỒ ÁN TỐT NGHIỆP / CUỐI KỲ: KHAI THÁC DỮ LIỆU & TRUYỀN THÔNG XÃ HỘI
HỆ THỐNG PHÂN TÍCH E-COMMERCE SHOPEE & SOCIAL MEDIA MINING
FLASK WEB APPLICATION SERVER (app.py)

NHÓM SINH VIÊN THỰC HIỆN:
1. Trần Đình Huy (MSSV: 24730103) - Nhóm trưởng
2. Lê Thanh Trúc Vi (MSSV: 24730150) - Thành viên (NLP, Teencode & Sentiment)
3. Vũ Hoàng Thiên Ân (MSSV: 24730155) - Thành viên (Gom cụm & Luật kết hợp)
4. Dương Phương Anh (MSSV: 24730156) - Thành viên (Đánh giá Mô hình & UI/UX)
=============================================================================
"""

import os
import sys
import io
import json
import base64
import random
import webbrowser
import threading
from typing import Dict, Any, List

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from flask import Flask, render_template, request, jsonify, redirect
import pandas as pd
import numpy as np
import joblib

from data_processing import clean_vietnamese_text, calculate_lexicon_sentiment_score, generate_shopee_datasets, VIETNAMESE_STOPWORDS
from model_training import run_full_training_pipeline

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["SECRET_KEY"] = "shopee-data-mining-uit-2026-secret-key"

# Thư mục chứa dữ liệu và mô hình
DATA_DIR = "data"
MODELS_DIR = "models"

# Tải trước hoặc khởi tạo dữ liệu và mô hình
def ensure_models_and_data():
    """Kiểm tra sự tồn tại của dữ liệu và mô hình, nếu chưa có thì tự động sinh và huấn luyện."""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    prod_csv = os.path.join(DATA_DIR, "shopee_products.csv")
    if not os.path.exists(prod_csv):
        print("-> Đang khởi tạo bộ dữ liệu mẫu Shopee...")
        generate_shopee_datasets(DATA_DIR)
        
    model_bench = os.path.join(MODELS_DIR, "model_benchmarks.json")
    if not os.path.exists(model_bench):
        print("-> Đang huấn luyện toàn bộ mô hình Machine Learning...")
        run_full_training_pipeline()

ensure_models_and_data()

# Nạp các mô hình đã huấn luyện
try:
    sentiment_model = joblib.load(os.path.join(MODELS_DIR, "sentiment_model.joblib"))
    tfidf_vectorizer = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib"))
    growth_classifier = joblib.load(os.path.join(MODELS_DIR, "growth_classifier.joblib"))
    sales_regressor = joblib.load(os.path.join(MODELS_DIR, "sales_regressor.joblib"))
    risk_classifier = joblib.load(os.path.join(MODELS_DIR, "risk_classifier.joblib"))
    feature_encoders = joblib.load(os.path.join(MODELS_DIR, "feature_encoders.joblib"))
    
    with open(os.path.join(MODELS_DIR, "clustering_data.json"), "r", encoding="utf-8") as f:
        clustering_data = json.load(f)
    with open(os.path.join(MODELS_DIR, "market_basket_rules.json"), "r", encoding="utf-8") as f:
        market_basket_rules = json.load(f)
    with open(os.path.join(MODELS_DIR, "model_benchmarks.json"), "r", encoding="utf-8") as f:
        model_benchmarks = json.load(f)
        
    df_products = pd.read_csv(os.path.join(DATA_DIR, "shopee_products.csv"))
    df_reviews = pd.read_csv(os.path.join(DATA_DIR, "shopee_reviews.csv"))
    print("-> Nạp thành công toàn bộ mô hình và dữ liệu vào bộ nhớ Flask!")
except Exception as e:
    print(f"-> Cảnh báo khi nạp mô hình: {e}")


# =============================================================================
# WEB PAGE ROUTES
# =============================================================================

@app.route("/")
def index():
    """Trang chủ ứng dụng Dashboard & Khai thác dữ liệu tương tác."""
    return render_template("index.html")

@app.route("/report")
@app.route("/theory")
def redirect_to_home():
    """Chuyển hướng toàn bộ các trang lý thuyết cũ về trang công cụ chính."""
    return redirect("/")


# =============================================================================
# REST API ENDPOINTS
# =============================================================================

@app.route("/api/overview_stats", methods=["GET"])
def api_overview_stats():
    """Lấy số liệu thống kê toàn sàn Shopee & phân bố truyền thông xã hội."""
    try:
        total_products = len(df_products)
        total_reviews = len(df_reviews)
        avg_price = int(df_products["price"].mean())
        total_monthly_sales = int(df_products["monthly_sold"].sum())
        est_monthly_revenue = int((df_products["price"] * df_products["monthly_sold"]).sum())
        avg_rating = round(float(df_products["rating_star"].mean()), 2)
        
        # Phân bố Cảm xúc Review
        sentiment_counts = df_reviews["sentiment_label"].value_counts().to_dict()
        sentiment_dist = {
            "pos": int(sentiment_counts.get("Tích cực", 0)),
            "neu": int(sentiment_counts.get("Trung lập", 0)),
            "neg": int(sentiment_counts.get("Tiêu cực", 0))
        }
        
        # Phân bố Doanh số theo Ngành Hàng
        cat_sales = df_products.groupby("category")["monthly_sold"].sum().reset_index()
        cat_labels = cat_sales["category"].tolist()
        cat_values = cat_sales["monthly_sold"].tolist()
        
        # Phân bố Điểm Đánh Giá Rating (1 - 5 sao)
        rating_round = df_products["rating_star"].round().astype(int)
        rating_dist = rating_round.value_counts().sort_index().to_dict()
        rating_labels = [f"{k} Sao" for k in range(1, 6)]
        rating_values = [int(rating_dist.get(k, 0)) for k in range(1, 6)]
        
        # Phân bố Rủi ro vận hành
        risk_counts = df_products["risk_level"].value_counts().to_dict()
        risk_dist = {
            "low": int(risk_counts.get("Thấp", 0)),
            "medium": int(risk_counts.get("Trung Bình", 0)),
            "high": int(risk_counts.get("Cao", 0))
        }
        
        # Top 5 Sản phẩm Bán Chạy Nhất
        top_products = df_products.sort_values(by="monthly_sold", ascending=False).head(5)[
            ["product_id", "product_name", "category", "price", "monthly_sold", "rating_star", "growth_potential"]
        ].to_dict(orient="records")
        
        return jsonify({
            "status": "success",
            "kpis": {
                "total_products": total_products,
                "total_reviews": total_reviews,
                "avg_price": f"{avg_price:,.0f} đ",
                "total_monthly_sales": f"{total_monthly_sales:,.0f} SP/tháng",
                "est_monthly_revenue": f"{est_monthly_revenue:,.0f} đ",
                "avg_rating": avg_rating
            },
            "sentiment_dist": sentiment_dist,
            "category_sales": {"labels": cat_labels, "values": cat_values},
            "rating_dist": {"labels": rating_labels, "values": rating_values},
            "risk_dist": risk_dist,
            "top_products": top_products
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/predict_product", methods=["POST"])
def api_predict_product():
    """
    Dự đoán tiềm năng tăng trưởng, doanh số tháng kỳ vọng và mức độ rủi ro hoàn hàng
    từ các thông số sản phẩm hoặc Link Shopee.
    """
    try:
        data = request.json or {}
        
        # Lấy các giá trị đầu vào
        cat_name = data.get("category", "Thời Trang Nam/Nữ")
        shop_type = data.get("shop_type", "Shop Yêu Thích")
        location = data.get("shop_location", "TP. Hồ Chí Minh")
        price = float(data.get("price", 199000))
        discount_rate = float(data.get("discount_rate", 20))
        is_megasale = int(data.get("is_megasale", 1))
        shipping_fee = float(data.get("shipping_fee", 22000))
        view_count = int(data.get("view_count", 12500))
        favorite_count = int(data.get("favorite_count", 850))
        historical_sold = int(data.get("historical_sold", 1200))
        rating_star = float(data.get("rating_star", 4.7))
        rating_count = int(data.get("rating_count", 450))
        shop_rating = float(data.get("shop_rating", 4.8))
        chat_response_rate = float(data.get("chat_response_rate", 95))
        ship_on_time_rate = float(data.get("ship_on_time_rate", 96))
        avg_delivery_days = float(data.get("avg_delivery_days", 2.5))
        
        # Mã hóa biến phân loại theo encoders đã lưu
        cat_code = feature_encoders["category"].get(cat_name, 0)
        shop_type_code = feature_encoders["shop_type"].get(shop_type, 0)
        location_code = feature_encoders["shop_location"].get(location, 0)
        
        features_vec = np.array([[
            cat_code, shop_type_code, location_code,
            price, discount_rate, is_megasale, shipping_fee,
            view_count, favorite_count, historical_sold,
            rating_star, rating_count, shop_rating,
            chat_response_rate, ship_on_time_rate, avg_delivery_days
        ]])
        
        # 1. Dự đoán Tiềm Năng Tăng Trưởng (Classification)
        growth_pred = growth_classifier.predict(features_vec)[0]
        if hasattr(growth_classifier, "predict_proba"):
            growth_probs = growth_classifier.predict_proba(features_vec)[0]
            max_prob = max(growth_probs)
            confidence = round(float(max_prob * 100), 1)
        else:
            confidence = 88.5
            
        # 2. Dự đoán Doanh Số Tháng Tới (Regression)
        sales_pred = int(max(0, sales_regressor.predict(features_vec)[0]))
        est_revenue = int(sales_pred * price * (1 - discount_rate / 100))
        
        # 3. Dự đoán Mức Độ Rủi Ro Vận Hành & Hoàn Hàng
        risk_pred = risk_classifier.predict(features_vec)[0]
        
        # Tính toán Feature Importance tác động riêng đến kết quả này
        importances = [
            {"name": "Lượt Xem & Tìm Kiếm", "score": min(100, int(view_count / 300)), "impact": "Tích cực" if view_count > 8000 else "Trung bình"},
            {"name": "Điểm Đánh Giá Rating", "score": int(rating_star / 5.0 * 100), "impact": "Rất tích cực" if rating_star >= 4.5 else "Rủi ro"},
            {"name": "Chương Trình Mega Sale & Giảm Giá", "score": int(discount_rate * 2), "impact": "Kích cầu mạnh" if is_megasale else "Bình thường"},
            {"name": "Tốc Độ Giao Hàng Chuỗi Cung Ứng", "score": int(ship_on_time_rate), "impact": "Tối ưu tốt" if ship_on_time_rate >= 95 else "Cần cải thiện"}
        ]
        
        # Gợi ý chiến lược
        recommendations = []
        if growth_pred == "Tiềm Năng Cao":
            recommendations.append("🚀 Tăng ngân sách Shopee Ads vào khung giờ vàng (12h trưa & 20h tối) để chiếm trọn thị phần.")
            recommendations.append("📦 Đăng ký tham gia Flash Sale và Deal 1K/9K để kéo thêm traffic cho toàn gian hàng.")
        elif growth_pred == "Trung Bình":
            recommendations.append("💡 Cải thiện hình ảnh sản phẩm chuẩn SEO Shopee và tối ưu mô tả chứa từ khóa hot trend.")
            recommendations.append("🎁 Thiết lập mã giảm giá Follower (Voucher theo dõi shop) để chuyển đổi lượt xem thành đơn hàng.")
        else:
            recommendations.append("⚠️ Rà soát lại chất lượng sản phẩm và kiểm tra lý do nhận đánh giá 1-2 sao gần đây.")
            recommendations.append("🚚 Cải thiện thời gian đóng gói và liên hệ đơn vị vận chuyển để giảm tỷ lệ giao trễ.")
            
        return jsonify({
            "status": "success",
            "prediction": {
                "growth_potential": str(growth_pred),
                "confidence_percent": confidence,
                "predicted_monthly_sold": f"{sales_pred:,.0f} sản phẩm",
                "estimated_monthly_revenue": f"{est_revenue:,.0f} đ",
                "risk_level": str(risk_pred),
                "risk_badge": "danger" if risk_pred == "Cao" else ("warning" if risk_pred == "Trung Bình" else "success"),
                "feature_impacts": importances,
                "recommendations": recommendations
            }
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/analyze_sentiment", methods=["POST"])
def api_analyze_sentiment():
    """
    Phân tích cảm xúc văn bản đánh giá / bình luận Shopee & MXH (TikTok, Facebook):
    - Làm sạch teencode, tách từ.
    - Dự đoán nhãn (Tích cực / Trung lập / Tiêu cực) bằng Naive Bayes + Lexicon.
    - Tính điểm Polarity Score (-1.0 đến +1.0).
    - Trích xuất từ khóa tích cực và tiêu cực.
    """
    try:
        data = request.json or {}
        raw_text = data.get("text", "").strip()
        
        if not raw_text:
            return jsonify({"status": "error", "message": "Vui lòng nhập nội dung đánh giá cần phân tích"}), 400
            
        # Tiền xử lý văn bản
        cleaned_text = clean_vietnamese_text(raw_text)
        
        # Dự đoán bằng Mô hình Máy học (TF-IDF + Naive Bayes / RF)
        text_vec = tfidf_vectorizer.transform([cleaned_text])
        ml_pred_label = sentiment_model.predict(text_vec)[0]
        
        ml_probs = sentiment_model.predict_proba(text_vec)[0]
        classes = list(sentiment_model.classes_)
        prob_dict = {cls: round(float(prob * 100), 1) for cls, prob in zip(classes, ml_probs)}
        
        # Kết hợp Lexicon Score
        lex_score, lex_label, pos_words, neg_words = calculate_lexicon_sentiment_score(raw_text)
        
        # Nhận diện khía cạnh (Aspects: Vận chuyển, Chất lượng, Phục vụ, Giá)
        aspects = []
        lower_raw = raw_text.lower()
        if any(w in lower_raw for w in ["ship", "giao hàng", "shipper", "nhanh", "chậm", "hỏa tốc"]):
            aspects.append("Vận chuyển & Giao hàng")
        if any(w in lower_raw for w in ["đóng gói", "bọc", "hộp", "móp", "vỡ"]):
            aspects.append("Đóng gói bao bì")
        if any(w in lower_raw for w in ["chất lượng", "vải", "dùng", "xịn", "dỏm", "fake", "auth", "hỏng"]):
            aspects.append("Chất lượng sản phẩm")
        if any(w in lower_raw for w in ["shop", "tư vấn", "rep", "inbox", "nhiệt tình", "thái độ"]):
            aspects.append("Dịch vụ & Chăm sóc KH")
        if any(w in lower_raw for w in ["giá", "rẻ", "đắt", "tiền", "sale", "voucher", "hạt dẻ"]):
            aspects.append("Giá cả & Khuyến mãi")
            
        if not aspects:
            aspects.append("Trải nghiệm chung")
            
        return jsonify({
            "status": "success",
            "raw_text": raw_text,
            "cleaned_text": cleaned_text,
            "sentiment_label": ml_pred_label,
            "confidence_percent": prob_dict.get(ml_pred_label, 90.0),
            "sentiment_score": lex_score,
            "probabilities": prob_dict,
            "positive_keywords": pos_words,
            "negative_keywords": neg_words,
            "aspects": aspects
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/wordcloud", methods=["GET"])
def api_wordcloud():
    """Tạo ảnh Word Cloud hoặc danh sách từ khóa tần suất cao nhất."""
    try:
        sentiment_filter = request.args.get("sentiment", "all")
        if sentiment_filter == "pos":
            subset = df_reviews[df_reviews["sentiment_label"] == "Tích cực"]
        elif sentiment_filter == "neg":
            subset = df_reviews[df_reviews["sentiment_label"] == "Tiêu cực"]
        else:
            subset = df_reviews
            
        text_corpus = " ".join(subset["cleaned_comment"].dropna().values)
        
        # Đếm tần suất từ (loại bỏ stopwords)
        words = text_corpus.split()
        word_counts = {}
        for w in words:
            if len(w) > 1 and w not in VIETNAMESE_STOPWORDS and not w.isdigit():
                word_counts[w] = word_counts.get(w, 0) + 1
                
        top_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:50]
        top_words_list = [{"text": k, "weight": v} for k, v in top_words]
        
        return jsonify({
            "status": "success",
            "words": top_words_list
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/model_benchmarks", methods=["GET"])
def api_model_benchmarks():
    """Trả về toàn bộ số liệu so sánh mô hình học máy (Confusion Matrix, ROC, F1)."""
    return jsonify({
        "status": "success",
        "benchmarks": model_benchmarks
    })


@app.route("/api/clustering", methods=["GET"])
def api_clustering():
    """Trả về dữ liệu gom cụm K-Means 2D PCA."""
    return jsonify({
        "status": "success",
        "clustering": clustering_data
    })


@app.route("/api/market_basket", methods=["GET"])
def api_market_basket():
    """Trả về danh sách các luật kết hợp Apriori giỏ hàng."""
    return jsonify({
        "status": "success",
        "rules": market_basket_rules
    })


@app.route("/api/sample_products", methods=["GET"])
def api_sample_products():
    """Trả về danh sách 10 sản phẩm mẫu đa dạng để người dùng thử nghiệm nhanh."""
    samples = df_products.sample(min(12, len(df_products)), random_state=123)[
        ["product_id", "product_name", "category", "shop_type", "shop_location",
         "price", "discount_rate", "is_megasale", "shipping_fee", "view_count",
         "favorite_count", "historical_sold", "monthly_sold", "rating_star",
         "rating_count", "shop_rating", "chat_response_rate", "ship_on_time_rate",
         "avg_delivery_days", "growth_potential", "risk_level"]
    ].to_dict(orient="records")
    return jsonify({"status": "success", "samples": samples})


@app.route("/api/sample_comments", methods=["GET"])
def api_sample_comments():
    """Trả về danh sách các bình luận mẫu teencode tiếng Việt để người dùng test."""
    samples = [
        {"type": "Tích cực", "text": "sp dùng ok lắm shop ơi, ship nhanh đóng gói cẩn thận 5 sao ❤️💯"},
        {"type": "Tích cực", "text": "hàng chuẩn auth xịn sò nha mn, săn sale đc giá hạt dẻ thích mê"},
        {"type": "Tiêu cực", "text": "shop lừa đảo, hàng fake kém chất lượng, nhắn tin khiếu nại thì bị block ko rep 😡👎"},
        {"type": "Tiêu cực", "text": "giao hàng siêu chậm mất 9 ngày, hộp móp méo đồ bên trong bị nứt vỡ thất vọng"},
        {"type": "Trung lập", "text": "hàng nhận đc bình thường, tiền nào của nấy, shop giao đúng số lượng"},
        {"type": "Phức tạp / Teencode", "text": "sp nhìn cx đẹpp nhưng ship hơi lâu xíu, rep ib nhiệt tình nhưng bọc hàng hơi ẩu nha"}
    ]
    return jsonify({"status": "success", "samples": samples})


# =============================================================================
# KHỞI CHẠY MÁY CHỦ WEB
# =============================================================================

def open_browser():
    """Tự động mở trình duyệt web sau khi Flask server khởi động."""
    try:
        webbrowser.open_new("http://127.0.0.1:5000")
    except Exception:
        pass

if __name__ == "__main__":
    port = 5000
    print(f"\n=============================================================================")
    print(f"🚀 KHỞI ĐỘNG HỆ THỐNG KHAI THÁC DỮ LIỆU & TRUYỀN THÔNG E-COMMERCE SHOPEE")
    print(f"🌐 ĐỊA CHỈ TRUY CẬP: http://127.0.0.1:{port}")
    print(f"=============================================================================\n")
    
    # Mở browser sau 1.5 giây
    threading.Timer(1.5, open_browser).start()
    app.run(host="0.0.0.0", port=port, debug=False)
