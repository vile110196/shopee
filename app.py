# -*- coding: utf-8 -*-
"""Flask dashboard cho ba bộ dữ liệu Shopee công khai đã chuẩn hóa."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
import sys
import threading
import webbrowser
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

from data_processing import (
    VIETNAMESE_STOPWORDS,
    calculate_lexicon_sentiment_score,
    clean_vietnamese_text,
)
from model_training import ASPECT_LABEL_NAMES, run_full_training_pipeline


if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["SECRET_KEY"] = os.environ.get("SHOPEE_SECRET_KEY") or secrets.token_hex(32)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REQUIRED_DATA_FILES = (
    "shopee_products.csv",
    "shopee_reviews.csv",
    "shopee_transactions.csv",
    "provenance.json",
)
REQUIRED_MODEL_FILES = (
    "sentiment_model.joblib",
    "tfidf_vectorizer.joblib",
    "aspect_models.joblib",
    "sales_regressor.joblib",
    "feature_encoders.joblib",
    "kmeans_model.joblib",
    "cluster_scaler.joblib",
    "clustering_data.json",
    "market_basket_rules.json",
    "model_benchmarks.json",
)

PRODUCT_PAYLOAD_FIELDS = [
    "product_id",
    "source_item_id",
    "source_shop_id",
    "product_url",
    "product_name",
    "category",
    "subcategory",
    "shop_name",
    "shop_location",
    "currency",
    "price",
    "original_price",
    "discount_rate",
    "rating_star",
    "rating_count",
    "historical_sold",
    "favorite_count",
    "snapshot_count",
    "first_seen_date",
    "last_seen_date",
    "observation_days",
    "observed_sold_delta",
    "daily_sold_rate",
    "description",
    "source_dataset",
]
FEATURE_LABELS = {
    "category_code": "Ngành hàng",
    "location_code": "Khu vực shop",
    "price": "Giá bán",
    "discount_rate": "Tỷ lệ giảm giá",
    "rating_star": "Điểm đánh giá",
    "rating_count": "Số lượt đánh giá",
    "historical_sold": "Lượt bán tích lũy",
    "favorite_count": "Lượt yêu thích",
    "snapshot_count": "Số snapshot",
}
ASPECT_DISPLAY_NAMES = {
    "price_sentiment": "Giá",
    "shipping_sentiment": "Giao hàng",
    "outlook_sentiment": "Hình thức",
    "quality_sentiment": "Chất lượng",
    "size_sentiment": "Kích cỡ",
    "shop_service_sentiment": "Dịch vụ shop",
    "general_sentiment": "Trải nghiệm chung",
    "others_sentiment": "Khác",
}


def ensure_models_and_data() -> None:
    """Chỉ tự huấn luyện từ dữ liệu thật; không âm thầm sinh dữ liệu giả."""
    missing_data = [name for name in REQUIRED_DATA_FILES if not (DATA_DIR / name).is_file()]
    if missing_data:
        joined = ", ".join(missing_data)
        raise RuntimeError(
            f"Thiếu dữ liệu đã chuẩn hóa: {joined}. "
            "Chạy '.venv\\Scripts\\python.exe import_public_data.py --download' trước."
        )
    provenance = json.loads((DATA_DIR / "provenance.json").read_text(encoding="utf-8"))
    if provenance.get("source_type") != "kaggle_real_shopee_multi_source":
        raise RuntimeError("provenance.json không xác nhận bộ dữ liệu Kaggle thật")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    missing_models = [name for name in REQUIRED_MODEL_FILES if not (MODELS_DIR / name).is_file()]
    models_stale = bool(missing_models)
    if not models_stale:
        try:
            benchmark = json.loads((MODELS_DIR / "model_benchmarks.json").read_text(encoding="utf-8"))
            expected = benchmark["metadata"]["data_sha256"]
            current = {
                name: hashlib.sha256((DATA_DIR / name).read_bytes()).hexdigest()
                for name in REQUIRED_DATA_FILES
                if name.endswith(".csv")
            }
            models_stale = benchmark["metadata"]["data_provenance"] != provenance["source_type"] or expected != current
        except (KeyError, OSError, TypeError, json.JSONDecodeError):
            models_stale = True
    if models_stale:
        print("-> Artifact thiếu hoặc cũ; huấn luyện lại từ dữ liệu Kaggle đã xác minh...")
        run_full_training_pipeline()


ensure_models_and_data()

try:
    sentiment_model = joblib.load(MODELS_DIR / "sentiment_model.joblib")
    tfidf_vectorizer = joblib.load(MODELS_DIR / "tfidf_vectorizer.joblib")
    aspect_models = joblib.load(MODELS_DIR / "aspect_models.joblib")
    sales_regressor = joblib.load(MODELS_DIR / "sales_regressor.joblib")
    feature_encoders = joblib.load(MODELS_DIR / "feature_encoders.joblib")
    clustering_data = json.loads((MODELS_DIR / "clustering_data.json").read_text(encoding="utf-8"))
    market_basket_rules = json.loads((MODELS_DIR / "market_basket_rules.json").read_text(encoding="utf-8"))
    model_benchmarks = json.loads((MODELS_DIR / "model_benchmarks.json").read_text(encoding="utf-8"))
    provenance = json.loads((DATA_DIR / "provenance.json").read_text(encoding="utf-8"))
    df_products = pd.read_csv(DATA_DIR / "shopee_products.csv")
    df_reviews = pd.read_csv(DATA_DIR / "shopee_reviews.csv")
    df_transactions = pd.read_csv(DATA_DIR / "shopee_transactions.csv")
except Exception as exc:
    raise RuntimeError("Không thể nạp dữ liệu hoặc artifact; chạy model_training.py để dựng lại.") from exc


def _native(value: Any) -> Any:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return value.item() if isinstance(value, np.generic) else value


def _product_payload(row: pd.Series) -> dict[str, Any]:
    return {field: _native(row[field]) for field in PRODUCT_PAYLOAD_FIELDS}


def _parse_shopee_identity(reference: str) -> tuple[str, str] | None:
    match = re.search(r"(?:i\.|/product/)(\d+)[./](\d+)", reference, flags=re.IGNORECASE)
    if not match:
        match = re.fullmatch(r"MY_(\d+)_(\d+)", reference.strip(), flags=re.IGNORECASE)
    return (match.group(1), match.group(2)) if match else None


def _find_product(reference: str) -> pd.Series | None:
    reference = str(reference or "").strip()
    if not reference:
        return None
    direct = df_products[df_products["product_id"].str.casefold().eq(reference.casefold())]
    if not direct.empty:
        return direct.iloc[0]
    if reference.isdigit():
        item = df_products[pd.to_numeric(df_products["source_item_id"], errors="coerce").eq(int(reference))]
        if not item.empty:
            return item.iloc[0]
    identity = _parse_shopee_identity(reference)
    if identity:
        shop_id, item_id = map(int, identity)
        matches = df_products[
            pd.to_numeric(df_products["source_shop_id"], errors="coerce").eq(shop_id)
            & pd.to_numeric(df_products["source_item_id"], errors="coerce").eq(item_id)
        ]
        if not matches.empty:
            return matches.iloc[0]
    exact_url = df_products[df_products["product_url"].str.rstrip("/").eq(reference.rstrip("/"))]
    return None if exact_url.empty else exact_url.iloc[0]


def _number_field(
    data: dict[str, Any],
    name: str,
    default: float,
    minimum: float,
    maximum: float,
    *,
    integer: bool = False,
) -> float | int:
    value = data.get(name, default)
    if value is None or value == "":
        value = default
    if isinstance(value, bool):
        raise ValueError(f"'{name}' phải là số")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"'{name}' phải là số") from exc
    if not math.isfinite(parsed) or not minimum <= parsed <= maximum:
        raise ValueError(f"'{name}' phải nằm trong [{minimum}, {maximum}]")
    if integer and not parsed.is_integer():
        raise ValueError(f"'{name}' phải là số nguyên")
    return int(parsed) if integer else parsed


def _category_field(data: dict[str, Any], name: str, default: str, encoder_name: str) -> str:
    value = str(data.get(name, default) or default).strip()
    if value not in feature_encoders[encoder_name]:
        raise ValueError(f"'{name}' không tồn tại trong tập huấn luyện")
    return value


def _sales_feature_impacts(limit: int = 6) -> list[dict[str, Any]]:
    importances = getattr(sales_regressor, "feature_importances_", None)
    feature_names = feature_encoders.get("feature_cols", [])
    if importances is None or len(importances) != len(feature_names):
        return []
    ranked = sorted(zip(feature_names, importances), key=lambda item: float(item[1]), reverse=True)[:limit]
    maximum = max((float(value) for _, value in ranked), default=1.0)
    return [
        {
            "name": FEATURE_LABELS.get(feature, feature),
            "score": round(float(value) / maximum * 100, 1) if maximum else 0.0,
            "importance_percent": round(float(value) * 100, 2),
            "impact": "Trọng số toàn cục của mô hình; không phải tác động nhân quả.",
        }
        for feature, value in ranked
    ]


def _listing_quality_alerts(limit: int = 8) -> list[dict[str, Any]]:
    price_p99 = float(df_products["price"].quantile(0.99))
    candidates = []
    for _, row in df_products.iterrows():
        issues: list[str] = []
        score = 0
        if pd.isna(row["rating_star"]):
            issues.append("Thiếu điểm đánh giá")
            score += 2
        elif float(row["rating_star"]) <= 3 and int(row["rating_count"]) >= 5:
            issues.append("Điểm đánh giá thấp")
            score += 3
        if float(row["price"]) > price_p99:
            issues.append("Giá nằm trên phân vị 99%")
            score += 2
        if not str(row.get("description", "") or "").strip():
            issues.append("Thiếu mô tả")
            score += 1
        if pd.isna(row["shop_location"]) or not str(row["shop_location"]).strip():
            issues.append("Thiếu vị trí shop")
            score += 1
        if issues:
            candidates.append((score, row, issues))
    candidates.sort(key=lambda item: (item[0], float(item[1]["historical_sold"])), reverse=True)
    alerts = []
    for score, row, issues in candidates[:limit]:
        alerts.append(
            {
                "product_id": row["product_id"],
                "product_name": row["product_name"],
                "quality_level": "Cần kiểm tra" if score >= 3 else "Thiếu dữ liệu",
                "issues": issues,
                "price": round(float(row["price"]), 2),
                "rating_star": _native(row["rating_star"]),
                "rating_count": int(row["rating_count"]),
                "recommended_action": "Đối chiếu lại listing gốc trước khi dùng cho phân tích.",
            }
        )
    return alerts


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/report")
def report():
    return render_template(
        "report.html",
        dataset_summary={
            "products": int(len(df_products)),
            "reviews": int(len(df_reviews)),
            "labeled_reviews": int(df_reviews["sentiment_label"].notna().sum()),
            "transactions": int(len(df_transactions)),
            "completed_transactions": int(df_transactions["order_status"].eq("Hoàn thành").sum()),
            "velocity_products": int(df_products["daily_sold_rate"].notna().sum()),
            "unique_comments": int(df_reviews["cleaned_comment"].nunique()),
        },
        benchmarks=model_benchmarks,
        provenance=provenance,
    )


@app.route("/theory")
def theory():
    return render_template("theory.html")


@app.route("/api/overview_stats", methods=["GET"])
def api_overview_stats():
    try:
        sentiment_counts = df_reviews["sentiment_label"].value_counts().to_dict()
        sentiment_dist = {
            "pos": int(sentiment_counts.get("Tích cực", 0)),
            "neu": int(sentiment_counts.get("Trung lập", 0)),
            "neg": int(sentiment_counts.get("Tiêu cực", 0)),
            "unlabeled": int(df_reviews["sentiment_label"].isna().sum()),
        }
        category_sales = (
            df_products.groupby("category", as_index=False)["historical_sold"]
            .sum()
            .sort_values("historical_sold", ascending=False)
        )
        rating_values = pd.to_numeric(df_products["rating_star"], errors="coerce").dropna().round().astype(int)
        rating_counts = rating_values.value_counts().to_dict()
        quality_dist = {
            "with_velocity": int(df_products["daily_sold_rate"].notna().sum()),
            "without_velocity": int(df_products["daily_sold_rate"].isna().sum()),
            "missing_rating": int(df_products["rating_star"].isna().sum()),
            "missing_location": int(df_products["shop_location"].isna().sum()),
        }
        top_products = (
            df_products.sort_values("historical_sold", ascending=False)
            .head(5)[
                [
                    "product_id",
                    "product_url",
                    "product_name",
                    "category",
                    "currency",
                    "price",
                    "historical_sold",
                    "rating_star",
                    "daily_sold_rate",
                ]
            ]
            .replace({np.nan: None})
            .to_dict(orient="records")
        )
        return jsonify(
            {
                "status": "success",
                "kpis": {
                    "total_products": int(len(df_products)),
                    "total_reviews": int(len(df_reviews)),
                    "median_price": f"RM {df_products['price'].median():,.2f}",
                    "historical_sold": f"{int(df_products['historical_sold'].sum()):,} SP",
                    "velocity_coverage": f"{quality_dist['with_velocity']:,} SP",
                    "avg_rating": round(float(df_products["rating_star"].mean()), 2),
                },
                "sentiment_dist": sentiment_dist,
                "category_sales": {
                    "labels": category_sales["category"].tolist(),
                    "values": [int(value) for value in category_sales["historical_sold"]],
                },
                "rating_dist": {
                    "labels": [f"{star} Sao" for star in range(1, 6)],
                    "values": [int(rating_counts.get(star, 0)) for star in range(1, 6)],
                },
                "quality_dist": quality_dist,
                "listing_alerts": _listing_quality_alerts(),
                "top_products": top_products,
                "currency": "MYR",
                "snapshot_period": "2023-04-24 đến 2023-05-13",
            }
        )
    except Exception as exc:
        app.logger.exception("Lỗi tổng hợp dashboard")
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/api/product_lookup", methods=["GET"])
def api_product_lookup():
    reference = request.args.get("reference", "").strip()
    if not reference:
        return jsonify({"status": "error", "message": "Thiếu mã item, product_id hoặc URL Shopee"}), 400
    row = _find_product(reference)
    if row is None:
        return jsonify({"status": "error", "message": "Không tìm thấy sản phẩm trong snapshot công khai"}), 404
    return jsonify({"status": "success", "product": _product_payload(row)})


@app.route("/api/predict_product", methods=["POST"])
def api_predict_product():
    """Ước lượng nhịp bán từ biến quan sát; không dự báo tăng trưởng hay rủi ro."""
    try:
        data = request.get_json(silent=True) or {}
        if not isinstance(data, dict):
            raise ValueError("Nội dung JSON phải là object")

        defaults = feature_encoders["numeric_medians"]
        default_category = next(iter(feature_encoders["category"]))
        default_location = "Không rõ" if "Không rõ" in feature_encoders["shop_location"] else next(iter(feature_encoders["shop_location"]))
        category = _category_field(data, "category", default_category, "category")
        location = _category_field(data, "shop_location", default_location, "shop_location")
        values = {
            "category_code": feature_encoders["category"][category],
            "location_code": feature_encoders["shop_location"][location],
            "price": _number_field(data, "price", defaults["price"], 0.01, 1_000_000_000),
            "discount_rate": _number_field(data, "discount_rate", defaults["discount_rate"], 0, 100),
            "rating_star": _number_field(data, "rating_star", defaults["rating_star"], 1, 5),
            "rating_count": _number_field(data, "rating_count", defaults["rating_count"], 0, 1_000_000_000, integer=True),
            "historical_sold": _number_field(data, "historical_sold", defaults["historical_sold"], 0, 1_000_000_000, integer=True),
            "favorite_count": _number_field(data, "favorite_count", defaults["favorite_count"], 0, 1_000_000_000, integer=True),
            "snapshot_count": _number_field(data, "snapshot_count", defaults["snapshot_count"], 1, 10_000, integer=True),
        }
        vector = np.array([[values[name] for name in feature_encoders["feature_cols"]]], dtype=float)
        daily_rate = max(0.0, float(np.expm1(sales_regressor.predict(vector)[0])))
        rate_30d = daily_rate * 30
        price = float(values["price"])
        thresholds = feature_encoders["performance_band_thresholds"]
        if daily_rate <= thresholds["low_max_daily_rate"]:
            band = "Nhịp bán thấp trong mẫu quan sát"
        elif daily_rate <= thresholds["medium_max_daily_rate"]:
            band = "Nhịp bán trung bình trong mẫu quan sát"
        else:
            band = "Nhịp bán cao trong mẫu quan sát"
        return jsonify(
            {
                "status": "success",
                "prediction": {
                    "performance_band": band,
                    "estimated_daily_sold_rate": round(daily_rate, 4),
                    "estimated_30d_sales_pace": round(rate_30d, 2),
                    "estimated_30d_gross_value_myr": round(rate_30d * price, 2),
                    "currency": "MYR",
                    "feature_impacts": _sales_feature_impacts(),
                    "sales_model": feature_encoders["selected_sales_model"],
                    "training_rows": feature_encoders["observed_target_rows"],
                    "method_note": "Ước lượng mô tả từ delta tổng bán giữa snapshot; không phải dự báo nhân quả hoặc cam kết doanh số tương lai.",
                    "input_reference": str(data.get("product_url") or data.get("product_id") or "").strip(),
                },
            }
        )
    except ValueError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400
    except Exception:
        app.logger.exception("Lỗi ước lượng nhịp bán")
        return jsonify({"status": "error", "message": "Không thể thực hiện ước lượng"}), 500


@app.route("/api/analyze_sentiment", methods=["POST"])
def api_analyze_sentiment():
    try:
        data = request.get_json(silent=True) or {}
        if not isinstance(data, dict) or not isinstance(data.get("text", ""), str):
            return jsonify({"status": "error", "message": "'text' phải là chuỗi"}), 400
        raw_text = data.get("text", "").strip()
        if not raw_text:
            return jsonify({"status": "error", "message": "Vui lòng nhập nội dung đánh giá"}), 400
        if len(raw_text) > 5_000:
            return jsonify({"status": "error", "message": "Nội dung tối đa 5.000 ký tự"}), 400
        cleaned = clean_vietnamese_text(raw_text)
        vector = tfidf_vectorizer.transform([cleaned])
        prediction = str(sentiment_model.predict(vector)[0])
        probabilities = sentiment_model.predict_proba(vector)[0]
        probability_map = {
            str(label): round(float(probability * 100), 1)
            for label, probability in zip(sentiment_model.classes_, probabilities)
        }
        lexicon_score, lexicon_label, positive, negative = calculate_lexicon_sentiment_score(raw_text)

        aspects = []
        for name, model in aspect_models.items():
            code = int(model.predict(vector)[0])
            aspect_probability = model.predict_proba(vector)[0]
            class_index = list(model.classes_).index(code)
            aspects.append(
                {
                    "key": name,
                    "name": ASPECT_DISPLAY_NAMES.get(name, name),
                    "code": code,
                    "label": ASPECT_LABEL_NAMES[str(code)],
                    "confidence_percent": round(float(aspect_probability[class_index] * 100), 1),
                }
            )
        return jsonify(
            {
                "status": "success",
                "raw_text": raw_text,
                "cleaned_text": cleaned,
                "sentiment_label": prediction,
                "confidence_percent": probability_map[prediction],
                "probabilities": probability_map,
                "lexicon_score": lexicon_score,
                "lexicon_label": lexicon_label,
                "positive_keywords": positive,
                "negative_keywords": negative,
                "aspects": aspects,
                "aspect_method": "Tám bộ phân loại Logistic Regression huấn luyện từ nhãn ABSA do người gán.",
            }
        )
    except Exception:
        app.logger.exception("Lỗi phân tích sentiment")
        return jsonify({"status": "error", "message": "Không thể phân tích nội dung"}), 500


@app.route("/api/wordcloud", methods=["GET"])
def api_wordcloud():
    sentiment_filter = request.args.get("sentiment", "all")
    if sentiment_filter == "pos":
        subset = df_reviews[df_reviews["sentiment_label"].eq("Tích cực")]
    elif sentiment_filter == "neg":
        subset = df_reviews[df_reviews["sentiment_label"].eq("Tiêu cực")]
    else:
        subset = df_reviews
    counts: dict[str, int] = {}
    for word in " ".join(subset["cleaned_comment"].dropna().astype(str)).split():
        if len(word) > 1 and word not in VIETNAMESE_STOPWORDS and not word.isdigit():
            counts[word] = counts.get(word, 0) + 1
    words = [
        {"text": word, "weight": weight}
        for word, weight in sorted(counts.items(), key=lambda item: item[1], reverse=True)[:50]
    ]
    return jsonify({"status": "success", "words": words})


@app.route("/api/model_benchmarks", methods=["GET"])
def api_model_benchmarks():
    return jsonify({"status": "success", "benchmarks": model_benchmarks})


@app.route("/api/clustering", methods=["GET"])
def api_clustering():
    return jsonify({"status": "success", "clustering": clustering_data})


@app.route("/api/market_basket", methods=["GET"])
def api_market_basket():
    return jsonify({"status": "success", "rules": market_basket_rules})


@app.route("/api/sample_products", methods=["GET"])
def api_sample_products():
    sample = df_products.sample(min(12, len(df_products)), random_state=123)
    return jsonify({"status": "success", "samples": [_product_payload(row) for _, row in sample.iterrows()]})


@app.route("/api/sample_comments", methods=["GET"])
def api_sample_comments():
    samples = []
    labeled = df_reviews[df_reviews["sentiment_label"].notna()]
    for label in ("Tích cực", "Trung lập", "Tiêu cực"):
        subset = labeled[labeled["sentiment_label"].eq(label)]
        for _, row in subset.sample(min(2, len(subset)), random_state=123).iterrows():
            samples.append({"type": label, "text": row["raw_comment"], "review_id": row["review_id"]})
    return jsonify({"status": "success", "samples": samples})


def open_browser() -> None:
    try:
        webbrowser.open_new("http://127.0.0.1:5000")
    except Exception:
        pass


if __name__ == "__main__":
    port = int(os.environ.get("SHOPEE_PORT", "5000"))
    print(f"Shopee Analytics Lab: http://127.0.0.1:{port}")
    threading.Timer(1.5, open_browser).start()
    app.run(host=os.environ.get("SHOPEE_HOST", "127.0.0.1"), port=port, debug=False)
