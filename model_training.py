# -*- coding: utf-8 -*-
"""
=============================================================================
ĐỒ ÁN: KHAI THÁC DỮ LIỆU & TRUYỀN THÔNG XÃ HỘI E-COMMERCE SHOPEE
MODULE: HUẤN LUYỆN & ĐÁNH GIÁ MÔ HÌNH MÁY HỌC (model_training.py)
NHÓM THỰC HIỆN:
- Trần Đình Huy (MSSV: 24730103) - Nhóm trưởng
- Lê Thanh Trúc Vi (MSSV: 24730150) - Phụ trách Tiền xử lý NLP & Phân loại Cảm xúc
- Vũ Hoàng Thiên Ân (MSSV: 24730155) - Phụ trách Gom cụm & Luật kết hợp
- Dương Phương Anh (MSSV: 24730156) - Phụ trách Đánh giá Mô hình & Độ đo
=============================================================================
"""

import os
import sys
import json
import itertools
from collections import defaultdict
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import joblib

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    mean_squared_error, mean_absolute_error, r2_score
)

# Thử import XGBoost nếu có
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False

from data_processing import clean_vietnamese_text, VIETNAMESE_STOPWORDS

# =============================================================================
# 1. HUẤN LUYỆN MÔ HÌNH PHÂN LOẠI CẢM XÚC (SENTIMENT ANALYSIS)
# =============================================================================

def train_sentiment_models(data_path: str = "data/shopee_reviews.csv", models_dir: str = "models") -> Dict[str, Any]:
    """
    Huấn luyện và so sánh các mô hình phân loại cảm xúc đánh giá Shopee & MXH:
    - Multinomial Naive Bayes (Thuật toán cốt lõi bài 5.1/7)
    - Random Forest
    - Logistic Regression
    - Decision Tree
    """
    print("\n--- ĐANG HUẤN LUYỆN MÔ HÌNH PHÂN TÍCH CẢM XÚC REVIEW (NLP) ---")
    df = pd.read_csv(data_path)
    
    # Đảm bảo dữ liệu đã làm sạch
    df["cleaned_comment"] = df["raw_comment"].apply(clean_vietnamese_text)
    
    X_text = df["cleaned_comment"].values
    y = df["sentiment_label"].values
    
    # Chia tập train / test
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # TF-IDF Vectorization
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=3000,
        min_df=2,
        sublinear_tf=True
    )
    X_train_vec = tfidf.fit_transform(X_train_text)
    X_test_vec = tfidf.transform(X_test_text)
    
    # Các thuật toán thử nghiệm
    classifiers = {
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.5),
        "Random Forest (NLP)": RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.5, random_state=42),
        "Decision Tree (NLP)": DecisionTreeClassifier(max_depth=15, random_state=42)
    }
    
    labels_order = ["Tích cực", "Trung lập", "Tiêu cực"]
    benchmark_results = {}
    best_model_name = "Multinomial Naive Bayes"
    best_f1 = 0
    best_model = None
    
    for name, clf in classifiers.items():
        clf.fit(X_train_vec, y_train)
        y_pred = clf.predict(X_test_vec)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        cm = confusion_matrix(y_test, y_pred, labels=labels_order).tolist()
        
        # ROC Curve cho bài toán đa lớp (One-vs-Rest)
        roc_data = {}
        if hasattr(clf, "predict_proba"):
            y_prob = clf.predict_proba(X_test_vec)
            # Map classes
            class_indices = {cls: idx for idx, cls in enumerate(clf.classes_)}
            for lbl in labels_order:
                if lbl in class_indices:
                    idx = class_indices[lbl]
                    y_true_binary = (y_test == lbl).astype(int)
                    fpr, tpr, _ = roc_curve(y_true_binary, y_prob[:, idx])
                    roc_auc_val = auc(fpr, tpr)
                    # Lấy mẫu 20 điểm để JSON nhẹ hơn
                    step = max(1, len(fpr) // 20)
                    roc_data[lbl] = {
                        "fpr": [round(float(x), 4) for x in fpr[::step]],
                        "tpr": [round(float(x), 4) for x in tpr[::step]],
                        "auc": round(float(roc_auc_val), 4)
                    }
        
        benchmark_results[name] = {
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "confusion_matrix": cm,
            "labels": labels_order,
            "roc_data": roc_data
        }
        
        print(f"  + {name:25s} | Acc: {acc:.4f} | F1-Score: {f1:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_model = clf
            
    # Lưu mô hình tốt nhất & Vectorizer
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(best_model, os.path.join(models_dir, "sentiment_model.joblib"))
    joblib.dump(tfidf, os.path.join(models_dir, "tfidf_vectorizer.joblib"))
    
    print(f"-> Đã lưu Sentiment Model tốt nhất ({best_model_name}) tại '{models_dir}/sentiment_model.joblib'")
    return {
        "best_model_name": best_model_name,
        "best_f1": best_f1,
        "benchmark": benchmark_results
    }


# =============================================================================
# 2. HUẤN LUYỆN MÔ HÌNH DỰ ĐOÁN TIỀM NĂNG & DOANH SỐ (SALES & GROWTH)
# =============================================================================

def train_product_sales_and_risk_models(data_path: str = "data/shopee_products.csv", models_dir: str = "models") -> Dict[str, Any]:
    """
    Huấn luyện các mô hình:
    1. Phân loại Tiềm năng Tăng trưởng: Random Forest vs XGBoost vs Decision Tree
    2. Dự đoán Doanh số Tháng (Hồi quy): Random Forest Regressor vs XGBoost Regressor
    3. Phân loại Rủi ro Vận hành & Hoàn hàng: Gradient Boosting vs Random Forest
    """
    print("\n--- ĐANG HUẤN LUYỆN MÔ HÌNH DỰ ĐOÁN DOANH SỐ & RỦI RO SẢN PHẨM ---")
    df = pd.read_csv(data_path)
    
    # Mã hóa các biến phân loại
    le_category = LabelEncoder()
    df["category_code"] = le_category.fit_transform(df["category"])
    
    le_shop_type = LabelEncoder()
    df["shop_type_code"] = le_shop_type.fit_transform(df["shop_type"])
    
    le_location = LabelEncoder()
    df["location_code"] = le_location.fit_transform(df["shop_location"])
    
    # Danh sách đặc trưng đầu vào
    feature_cols = [
        "category_code", "shop_type_code", "location_code",
        "price", "discount_rate", "is_megasale", "shipping_fee",
        "view_count", "favorite_count", "historical_sold",
        "rating_star", "rating_count", "shop_rating",
        "chat_response_rate", "ship_on_time_rate", "avg_delivery_days"
    ]
    
    X = df[feature_cols].values
    
    # -------------------------------------------------------------------------
    # A. PHÂN LOẠI TIỀM NĂNG TĂNG TRƯỞNG (GROWTH POTENTIAL)
    # -------------------------------------------------------------------------
    y_growth = df["growth_potential"].values
    growth_labels = ["Tiềm Năng Cao", "Trung Bình", "Rủi Ro / Kém"]
    
    X_train, X_test, y_g_train, y_g_test = train_test_split(
        X, y_growth, test_size=0.2, random_state=42, stratify=y_growth
    )
    
    growth_classifiers = {
        "Random Forest Classifier": RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42),
        "Decision Tree (C4.5/CART)": DecisionTreeClassifier(max_depth=8, criterion="gini", random_state=42)
    }
    
    if HAS_XGBOOST:
        le_target = LabelEncoder()
        y_g_train_num = le_target.fit_transform(y_g_train)
        y_g_test_num = le_target.transform(y_g_test)
        
        xgb_clf = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, eval_metric="mlogloss")
        xgb_clf.fit(X_train, y_g_train_num)
        y_pred_num = xgb_clf.predict(X_test)
        y_pred_xgb = le_target.inverse_transform(y_pred_num)
        
        acc_xgb = accuracy_score(y_g_test, y_pred_xgb)
        f1_xgb = f1_score(y_g_test, y_pred_xgb, average="weighted")
        growth_classifiers["XGBoost Classifier"] = (xgb_clf, le_target)
    
    growth_benchmarks = {}
    best_growth_clf = None
    best_g_f1 = 0
    best_g_name = ""
    
    for name, clf_obj in growth_classifiers.items():
        if name == "XGBoost Classifier":
            clf, le_t = clf_obj
            y_pred = le_t.inverse_transform(clf.predict(X_test))
            clf_actual = clf
        else:
            clf_obj.fit(X_train, y_g_train)
            y_pred = clf_obj.predict(X_test)
            clf_actual = clf_obj
            
        acc = accuracy_score(y_g_test, y_pred)
        prec = precision_score(y_g_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_g_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_g_test, y_pred, average="weighted", zero_division=0)
        cm = confusion_matrix(y_g_test, y_pred, labels=growth_labels).tolist()
        
        # Feature importance
        if hasattr(clf_actual, "feature_importances_"):
            feat_imp = [
                {"feature": col, "importance": round(float(imp), 4)}
                for col, imp in sorted(zip(feature_cols, clf_actual.feature_importances_), key=lambda x: x[1], reverse=True)
            ]
        else:
            feat_imp = []
            
        growth_benchmarks[name] = {
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "confusion_matrix": cm,
            "labels": growth_labels,
            "feature_importance": feat_imp[:8]
        }
        print(f"  + {name:25s} | Acc: {acc:.4f} | F1-Score: {f1:.4f}")
        
        if f1 > best_g_f1:
            best_g_f1 = f1
            best_g_name = name
            best_growth_clf = clf_actual
            
    # -------------------------------------------------------------------------
    # B. DỰ ĐOÁN DOANH SỐ THÁNG (SALES REGRESSOR)
    # -------------------------------------------------------------------------
    y_sales = df["monthly_sold"].values
    X_s_train, X_s_test, y_s_train, y_s_test = train_test_split(X, y_sales, test_size=0.2, random_state=42)
    
    rf_reg = RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42)
    rf_reg.fit(X_s_train, y_s_train)
    y_s_pred = rf_reg.predict(X_s_test)
    
    r2 = r2_score(y_s_test, y_s_pred)
    rmse = np.sqrt(mean_squared_error(y_s_test, y_s_pred))
    mae = mean_absolute_error(y_s_test, y_s_pred)
    print(f"  + RF Sales Regressor        | R² Score: {r2:.4f} | RMSE: {rmse:.2f} | MAE: {mae:.2f}")
    
    # -------------------------------------------------------------------------
    # C. PHÂN LOẠI RỦI RO HOÀN HÀNG (RISK CLASSIFIER)
    # -------------------------------------------------------------------------
    y_risk = df["risk_level"].values
    risk_labels = ["Thấp", "Trung Bình", "Cao"]
    X_r_train, X_r_test, y_r_train, y_r_test = train_test_split(X, y_risk, test_size=0.2, random_state=42, stratify=y_risk)
    
    gb_risk = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
    gb_risk.fit(X_r_train, y_r_train)
    y_r_pred = gb_risk.predict(X_r_test)
    
    risk_acc = accuracy_score(y_r_test, y_r_pred)
    risk_f1 = f1_score(y_r_test, y_r_pred, average="weighted")
    print(f"  + GB Risk Classifier        | Acc: {risk_acc:.4f} | F1-Score: {risk_f1:.4f}")
    
    # Lưu các encoders và models
    encoders = {
        "category": {label: int(code) for label, code in zip(le_category.classes_, range(len(le_category.classes_)))},
        "shop_type": {label: int(code) for label, code in zip(le_shop_type.classes_, range(len(le_shop_type.classes_)))},
        "shop_location": {label: int(code) for label, code in zip(le_location.classes_, range(len(le_location.classes_)))},
        "feature_cols": feature_cols
    }
    
    joblib.dump(best_growth_clf, os.path.join(models_dir, "growth_classifier.joblib"))
    joblib.dump(rf_reg, os.path.join(models_dir, "sales_regressor.joblib"))
    joblib.dump(gb_risk, os.path.join(models_dir, "risk_classifier.joblib"))
    joblib.dump(encoders, os.path.join(models_dir, "feature_encoders.joblib"))
    
    return {
        "growth_benchmarks": growth_benchmarks,
        "sales_regressor_metrics": {"r2": round(float(r2), 4), "rmse": round(float(rmse), 2), "mae": round(float(mae), 2)},
        "risk_classifier_metrics": {"accuracy": round(float(risk_acc), 4), "f1_score": round(float(risk_f1), 4)}
    }


# =============================================================================
# 3. GOM CỤM SẢN PHẨM & SHOP (K-MEANS CLUSTERING)
# =============================================================================

def train_kmeans_clustering(data_path: str = "data/shopee_products.csv", models_dir: str = "models") -> Dict[str, Any]:
    """
    Gom cụm K-Means phân khúc sản phẩm Shopee (Bài 6 trong giáo trình):
    - Sử dụng các đặc trưng chuẩn hóa: Giá, Lượt bán, Rating, Discount, Ship On Time.
    - Giảm chiều PCA 2D để trực quan hóa biểu đồ phân bố cụm.
    - Gán nhãn ngữ nghĩa:
      0: Best-Seller / Ngôi Sao Doanh Số
      1: Tiềm Năng Tăng Trưởng Cao
      2: Hàng Bán Chậm / Cạnh Tranh Gay Gắt
      3: Rủi Ro Hoàn Hàng & Khiếu Nại
    """
    print("\n--- ĐANG GOM CỤM SẢN PHẨM BẰNG THUẬT TOÁN K-MEANS (K=4) ---")
    df = pd.read_csv(data_path)
    
    cluster_features = [
        "price", "monthly_sold", "rating_star", "discount_rate",
        "view_count", "favorite_count", "ship_on_time_rate", "return_rate"
    ]
    
    X_cluster = df[cluster_features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)
    
    # Huấn luyện K-Means với k=4
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    df["cluster"] = cluster_labels
    
    # Giảm chiều PCA xuống 2D
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    df["pca_x"] = X_pca[:, 0]
    df["pca_y"] = X_pca[:, 1]
    
    # Định nghĩa tên cụm dựa trên trung bình các chỉ số
    cluster_profiles = {}
    cluster_names = {
        0: "Cụm 1: Best-Seller / Ngôi Sao Doanh Thu",
        1: "Cụm 2: Hàng Mới / Tiềm Năng Tăng Trưởng",
        2: "Cụm 3: Hàng Phổ Thông / Cạnh Tranh Cao",
        3: "Cụm 4: Cần Cải Thiện Vận Hành / Rủi Ro"
    }
    
    for c_id in range(4):
        c_df = df[df["cluster"] == c_id]
        cluster_profiles[c_id] = {
            "name": cluster_names[c_id],
            "count": int(len(c_df)),
            "avg_price": round(float(c_df["price"].mean()), 0),
            "avg_monthly_sold": round(float(c_df["monthly_sold"].mean()), 1),
            "avg_rating": round(float(c_df["rating_star"].mean()), 2),
            "avg_discount": round(float(c_df["discount_rate"].mean()), 1),
            "avg_return_rate": round(float(c_df["return_rate"].mean()), 2)
        }
        
    # Tạo danh sách 150 điểm mẫu đại diện để hiển thị trên biểu đồ Web
    sample_points = []
    for _, row in df.sample(min(150, len(df)), random_state=42).iterrows():
        sample_points.append({
            "product_id": row["product_id"],
            "product_name": row["product_name"][:35] + "...",
            "category": row["category"],
            "cluster": int(row["cluster"]),
            "cluster_name": cluster_names[int(row["cluster"])],
            "pca_x": round(float(row["pca_x"]), 3),
            "pca_y": round(float(row["pca_y"]), 3),
            "price": int(row["price"]),
            "monthly_sold": int(row["monthly_sold"]),
            "rating": float(row["rating_star"])
        })
        
    # Lưu K-Means model và dữ liệu PCA
    joblib.dump(kmeans, os.path.join(models_dir, "kmeans_model.joblib"))
    joblib.dump(scaler, os.path.join(models_dir, "cluster_scaler.joblib"))
    
    pca_data = {
        "cluster_profiles": cluster_profiles,
        "sample_points": sample_points,
        "explained_variance_ratio": [round(float(v), 4) for v in pca.explained_variance_ratio_]
    }
    
    with open(os.path.join(models_dir, "clustering_data.json"), "w", encoding="utf-8") as f:
        json.dump(pca_data, f, ensure_ascii=False, indent=2)
        
    print(f"-> Đã lưu K-Means Clustering và PCA data tại '{models_dir}/clustering_data.json'")
    return pca_data


# =============================================================================
# 4. KHAI PHÁ LUẬT KẾT HỢP GIỎ HÀNG (APRIORI ALGORITHM)
# =============================================================================

def mine_association_rules_apriori(data_path: str = "data/shopee_transactions.csv", min_support: float = 0.04, min_confidence: float = 0.4, models_dir: str = "models") -> List[Dict[str, Any]]:
    """
    Thuật toán Apriori tìm tập phổ biến & luật kết hợp (Bài 2 trong giáo trình):
    - Tính Support, Confidence, Lift.
    - Phát hiện sản phẩm mua kèm / combo khuyến mãi cross-selling.
    """
    print("\n--- ĐANG KHAI PHÁ TẬP PHỔ BIẾN & LUẬT KẾT HỢP (APRIORI) ---")
    df = pd.read_csv(data_path)
    
    transactions = [t.split(", ") for t in df["items"].values]
    num_trans = len(transactions)
    
    # 1. Đếm tần suất các item đơn
    item_counts = defaultdict(int)
    for t in transactions:
        for item in set(t):
            item_counts[item] += 1
            
    # Lọc 1-itemsets thỏa min_support
    freq_1 = {item: count / num_trans for item, count in item_counts.items() if count / num_trans >= min_support}
    
    # 2. Tạo 2-itemsets
    pair_counts = defaultdict(int)
    for t in transactions:
        t_set = sorted(list(set(t)))
        for pair in itertools.combinations(t_set, 2):
            pair_counts[pair] += 1
            
    freq_2 = {pair: count / num_trans for pair, count in pair_counts.items() if count / num_trans >= min_support}
    
    # 3. Tạo luật kết hợp A -> B và B -> A
    rules = []
    for (item_a, item_b), supp_ab in freq_2.items():
        # Luật 1: A => B
        supp_a = freq_1.get(item_a, 0)
        supp_b = freq_1.get(item_b, 0)
        if supp_a > 0 and supp_b > 0:
            conf_a_to_b = supp_ab / supp_a
            lift_a_to_b = conf_a_to_b / supp_b
            if conf_a_to_b >= min_confidence:
                rules.append({
                    "antecedent": item_a,
                    "consequent": item_b,
                    "rule_text": f"Nếu mua '{item_a}' => Có xu hướng mua '{item_b}'",
                    "support": round(float(supp_ab * 100), 2),
                    "confidence": round(float(conf_a_to_b * 100), 2),
                    "lift": round(float(lift_a_to_b), 2),
                    "insight": f"Khách hàng mua {item_a} có khả năng mua thêm {item_b} gấp {lift_a_to_b:.1f} lần so với ngẫu nhiên."
                })
                
        # Luật 2: B => A
        if supp_b > 0 and supp_a > 0:
            conf_b_to_a = supp_ab / supp_b
            lift_b_to_a = conf_b_to_a / supp_a
            if conf_b_to_a >= min_confidence:
                rules.append({
                    "antecedent": item_b,
                    "consequent": item_a,
                    "rule_text": f"Nếu mua '{item_b}' => Có xu hướng mua '{item_a}'",
                    "support": round(float(supp_ab * 100), 2),
                    "confidence": round(float(conf_b_to_a * 100), 2),
                    "lift": round(float(lift_b_to_a), 2),
                    "insight": f"Khách hàng mua {item_b} có khả năng mua thêm {item_a} gấp {lift_b_to_a:.1f} lần so với ngẫu nhiên."
                })
                
    # Sắp xếp theo Lift giảm dần
    rules = sorted(rules, key=lambda x: x["lift"], reverse=True)
    
    with open(os.path.join(models_dir, "market_basket_rules.json"), "w", encoding="utf-8") as f:
        json.dump(rules, f, ensure_ascii=False, indent=2)
        
    print(f"-> Đã khai phá được {len(rules)} luật kết hợp mạnh. Lưu tại '{models_dir}/market_basket_rules.json'")
    return rules


# =============================================================================
# 5. TỔNG HỢP VÀ THỰC THI TOÀN BỘ QUY TRÌNH HUẤN LUYỆN
# =============================================================================

def run_full_training_pipeline():
    """
    Chạy toàn bộ pipeline huấn luyện các mô hình Machine Learning & Data Mining.
    """
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    # 1. NLP Sentiment Models
    sentiment_res = train_sentiment_models("data/shopee_reviews.csv", models_dir)
    
    # 2. Sales & Growth Potential & Risk Models
    product_res = train_product_sales_and_risk_models("data/shopee_products.csv", models_dir)
    
    # 3. K-Means Clustering
    clustering_res = train_kmeans_clustering("data/shopee_products.csv", models_dir)
    
    # 4. Market Basket Apriori Rules
    rules_res = mine_association_rules_apriori("data/shopee_transactions.csv", min_support=0.03, min_confidence=0.35, models_dir=models_dir)
    
    # Tổng hợp file model_benchmarks.json cho Web Dashboard
    all_benchmarks = {
        "sentiment_models": sentiment_res["benchmark"],
        "growth_models": product_res["growth_benchmarks"],
        "sales_regressor": product_res["sales_regressor_metrics"],
        "risk_classifier": product_res["risk_classifier_metrics"],
        "top_rules_count": len(rules_res),
        "clustering_clusters_count": 4
    }
    
    with open(os.path.join(models_dir, "model_benchmarks.json"), "w", encoding="utf-8") as f:
        json.dump(all_benchmarks, f, ensure_ascii=False, indent=2)
        
    print("\n=============================================================================")
    print("🎉 HOÀN TẤT HUẤN LUYỆN TOÀN BỘ MÔ HÌNH KHAI THÁC DỮ LIỆU SHOPEE THÀNH CÔNG!")
    print("=============================================================================")

if __name__ == "__main__":
    run_full_training_pipeline()
