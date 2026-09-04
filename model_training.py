# -*- coding: utf-8 -*-
"""Huấn luyện các mô hình trên ba bộ dữ liệu Shopee công khai đã chuẩn hóa."""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_curve,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler

from data_processing import clean_vietnamese_text


if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
RANDOM_SEED = 42

SENTIMENT_LABELS = ["Tích cực", "Trung lập", "Tiêu cực"]
ASPECT_COLUMNS = [
    "price_sentiment",
    "shipping_sentiment",
    "outlook_sentiment",
    "quality_sentiment",
    "size_sentiment",
    "shop_service_sentiment",
    "general_sentiment",
    "others_sentiment",
]
ASPECT_LABELS = [-1, 0, 1, 2]
ASPECT_LABEL_NAMES = {
    "-1": "Không đề cập",
    "0": "Tiêu cực",
    "1": "Tích cực",
    "2": "Trung lập",
}
PRODUCT_FEATURES = [
    "category_code",
    "location_code",
    "price",
    "discount_rate",
    "rating_star",
    "rating_count",
    "historical_sold",
    "favorite_count",
    "snapshot_count",
]
CLUSTER_FEATURES = [
    "price",
    "historical_sold",
    "rating_star",
    "discount_rate",
    "favorite_count",
    "rating_count",
    "daily_sold_rate",
    "snapshot_count",
]
CLUSTER_LOG_FEATURES = {
    "price",
    "historical_sold",
    "favorite_count",
    "rating_count",
    "daily_sold_rate",
}


def _project_path(path: str | os.PathLike[str]) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def _build_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=15_000,
        min_df=2,
        sublinear_tf=True,
    )


def _deduplicated_split(df: pd.DataFrame, label_column: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Giữ test gốc độc lập, loại câu trùng chéo split và câu có nhãn xung đột."""
    usable = df.loc[
        df[label_column].notna() & df["cleaned_comment"].str.len().gt(0),
        ["cleaned_comment", "source_split", label_column],
    ].copy()
    conflicts = usable.groupby("cleaned_comment")[label_column].nunique()
    usable = usable[usable["cleaned_comment"].isin(conflicts[conflicts == 1].index)]

    test = (
        usable[usable["source_split"].eq("test")]
        .drop_duplicates("cleaned_comment", keep="first")
        .reset_index(drop=True)
    )
    test_texts = set(test["cleaned_comment"])
    train = (
        usable[~usable["source_split"].eq("test")]
        .loc[lambda frame: ~frame["cleaned_comment"].isin(test_texts)]
        .drop_duplicates("cleaned_comment", keep="first")
        .reset_index(drop=True)
    )
    deployment = usable.drop_duplicates("cleaned_comment", keep="first").reset_index(drop=True)
    if train.empty or test.empty:
        raise ValueError(f"Không đủ train/test độc lập cho {label_column}")
    return train, test, deployment


def _classification_metrics(model, x_test, y_test, labels: list[Any]) -> dict[str, Any]:
    prediction = model.predict(x_test)
    result: dict[str, Any] = {
        "accuracy": round(float(accuracy_score(y_test, prediction)), 4),
        "precision": round(float(precision_score(y_test, prediction, average="weighted", zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, prediction, average="weighted", zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, prediction, average="weighted", zero_division=0)), 4),
        "macro_f1": round(float(f1_score(y_test, prediction, average="macro", zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_test, prediction, labels=labels).tolist(),
        "labels": labels,
        "test_samples": int(len(y_test)),
    }
    roc_data: dict[str, Any] = {}
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x_test)
        indices = {label: index for index, label in enumerate(model.classes_)}
        for label in labels:
            if label not in indices:
                continue
            truth = (np.asarray(y_test) == label).astype(int)
            if truth.min() == truth.max():
                continue
            fpr, tpr, _ = roc_curve(truth, probabilities[:, indices[label]])
            step = max(1, len(fpr) // 30)
            roc_data[str(label)] = {
                "fpr": [round(float(value), 4) for value in fpr[::step]],
                "tpr": [round(float(value), 4) for value in tpr[::step]],
                "auc": round(float(auc(fpr, tpr)), 4),
            }
    result["roc_data"] = roc_data
    return result


def train_sentiment_and_aspect_models(
    data_path: str | os.PathLike[str] = "data/shopee_reviews.csv",
    models_dir: str | os.PathLike[str] = "models",
) -> dict[str, Any]:
    """Huấn luyện sentiment tổng quát và tám bộ phân loại ABSA."""
    print("\n--- HUẤN LUYỆN SENTIMENT VÀ TÁM KHÍA CẠNH ABSA ---")
    data_path = _project_path(data_path)
    models_dir = _project_path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    reviews = pd.read_csv(data_path)
    reviews["cleaned_comment"] = reviews["raw_comment"].fillna("").map(clean_vietnamese_text)

    overall_train, overall_test, overall_deployment = _deduplicated_split(reviews, "sentiment_label")
    evaluation_vectorizer = _build_vectorizer()
    x_train = evaluation_vectorizer.fit_transform(overall_train["cleaned_comment"])
    x_test = evaluation_vectorizer.transform(overall_test["cleaned_comment"])
    y_train = overall_train["sentiment_label"].to_numpy()
    y_test = overall_test["sentiment_label"].to_numpy()

    candidates = {
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.5),
        "Logistic Regression": LogisticRegression(
            max_iter=1_000,
            C=2.0,
            class_weight="balanced",
            random_state=RANDOM_SEED,
        ),
    }
    benchmarks: dict[str, Any] = {}
    best_name = ""
    best_template = None
    best_macro_f1 = -1.0
    for name, template in candidates.items():
        model = clone(template)
        model.fit(x_train, y_train)
        metrics = _classification_metrics(model, x_test, y_test, SENTIMENT_LABELS)
        benchmarks[name] = metrics
        print(
            f"  + {name:27s} | Accuracy {metrics['accuracy']:.4f} | "
            f"Macro-F1 {metrics['macro_f1']:.4f}"
        )
        if metrics["macro_f1"] > best_macro_f1:
            best_name = name
            best_template = template
            best_macro_f1 = metrics["macro_f1"]

    if best_template is None:
        raise RuntimeError("Không chọn được mô hình sentiment")

    # Artifact triển khai dùng toàn bộ văn bản để xây vocabulary; metric phía trên chỉ dùng test gốc.
    all_text = reviews.loc[reviews["cleaned_comment"].str.len().gt(0), "cleaned_comment"].drop_duplicates()
    deployment_vectorizer = _build_vectorizer()
    deployment_vectorizer.fit(all_text)
    sentiment_model = clone(best_template)
    sentiment_model.fit(
        deployment_vectorizer.transform(overall_deployment["cleaned_comment"]),
        overall_deployment["sentiment_label"],
    )

    aspect_models: dict[str, Any] = {}
    aspect_benchmarks: dict[str, Any] = {}
    for aspect in ASPECT_COLUMNS:
        aspect_train, aspect_test, aspect_deployment = _deduplicated_split(reviews, aspect)
        model = LogisticRegression(
            max_iter=1_000,
            C=2.0,
            class_weight="balanced",
            random_state=RANDOM_SEED,
        )
        model.fit(evaluation_vectorizer.transform(aspect_train["cleaned_comment"]), aspect_train[aspect].astype(int))
        metrics = _classification_metrics(
            model,
            evaluation_vectorizer.transform(aspect_test["cleaned_comment"]),
            aspect_test[aspect].astype(int).to_numpy(),
            ASPECT_LABELS,
        )
        final_model = clone(model)
        final_model.fit(
            deployment_vectorizer.transform(aspect_deployment["cleaned_comment"]),
            aspect_deployment[aspect].astype(int),
        )
        aspect_models[aspect] = final_model
        aspect_benchmarks[aspect] = metrics
        print(f"  + {aspect:27s} | Macro-F1 {metrics['macro_f1']:.4f}")

    joblib.dump(sentiment_model, models_dir / "sentiment_model.joblib")
    joblib.dump(deployment_vectorizer, models_dir / "tfidf_vectorizer.joblib")
    joblib.dump(aspect_models, models_dir / "aspect_models.joblib")
    return {
        "selected_model": best_name,
        "benchmark": benchmarks,
        "aspect_benchmark": aspect_benchmarks,
        "source_rows": int(len(reviews)),
        "overall_labeled_rows": int(reviews["sentiment_label"].notna().sum()),
        "train_samples": int(len(overall_train)),
        "test_samples": int(len(overall_test)),
        "deployment_samples": int(len(overall_deployment)),
        "split_policy": "source_train_plus_validation_vs_source_test; conflicting_and_cross_split_duplicate_text_removed",
        "aspect_label_names": ASPECT_LABEL_NAMES,
    }


def _product_matrix(df: pd.DataFrame, metadata: dict[str, Any]) -> np.ndarray:
    prepared = pd.DataFrame(index=df.index)
    prepared["category_code"] = (
        df["category"].fillna("Không rõ").astype(str).map(metadata["category"]).fillna(metadata["unknown_code"])
    )
    prepared["location_code"] = (
        df["shop_location"].fillna("Không rõ").astype(str).map(metadata["shop_location"]).fillna(metadata["unknown_code"])
    )
    for feature in PRODUCT_FEATURES[2:]:
        prepared[feature] = pd.to_numeric(df[feature], errors="coerce").fillna(metadata["numeric_medians"][feature])
    return prepared[PRODUCT_FEATURES].to_numpy(dtype=float)


def train_sales_velocity_model(
    data_path: str | os.PathLike[str] = "data/shopee_products.csv",
    models_dir: str | os.PathLike[str] = "models",
) -> dict[str, Any]:
    """Ước lượng tốc độ bán/ngày từ các delta tổng bán quan sát được giữa snapshot."""
    print("\n--- HUẤN LUYỆN HỒI QUY TỐC ĐỘ BÁN QUAN SÁT ---")
    data_path = _project_path(data_path)
    models_dir = _project_path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    products = pd.read_csv(data_path)
    observed = products[pd.to_numeric(products["daily_sold_rate"], errors="coerce").notna()].copy()
    observed["daily_sold_rate"] = pd.to_numeric(observed["daily_sold_rate"], errors="coerce")
    observed = observed[np.isfinite(observed["daily_sold_rate"]) & observed["daily_sold_rate"].ge(0)].copy()
    if len(observed) < 100:
        raise ValueError("Không đủ sản phẩm có tốc độ bán quan sát để huấn luyện")

    categories = sorted(products["category"].fillna("Không rõ").astype(str).unique())
    locations = sorted(products["shop_location"].fillna("Không rõ").astype(str).unique())
    metadata: dict[str, Any] = {
        "category": {value: index for index, value in enumerate(categories)},
        "shop_location": {value: index for index, value in enumerate(locations)},
        "unknown_code": -1,
        "numeric_medians": {
            feature: float(pd.to_numeric(observed[feature], errors="coerce").median())
            for feature in PRODUCT_FEATURES[2:]
        },
        "feature_cols": PRODUCT_FEATURES,
        "target": "daily_sold_rate",
        "target_transform": "log1p",
        "currency": "MYR",
    }
    x = _product_matrix(observed, metadata)
    y = observed["daily_sold_rate"].to_numpy(dtype=float)
    rng = np.random.default_rng(RANDOM_SEED)
    indices = np.arange(len(observed))
    rng.shuffle(indices)
    split = int(len(indices) * 0.8)
    train_idx, test_idx = indices[:split], indices[split:]
    x_train, x_test = x[train_idx], x[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    y_train_log = np.log1p(y_train)

    candidates = {
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=220,
            max_depth=14,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=RANDOM_SEED,
        ),
        "Extra Trees Regressor": ExtraTreesRegressor(
            n_estimators=220,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=RANDOM_SEED,
        ),
    }
    benchmarks: dict[str, Any] = {}
    best_name = ""
    best_model = None
    best_mae = float("inf")
    for name, model in candidates.items():
        model.fit(x_train, y_train_log)
        prediction = np.maximum(0.0, np.expm1(model.predict(x_test)))
        metrics = {
            "r2": round(float(r2_score(y_test, prediction)), 4),
            "rmse": round(float(np.sqrt(mean_squared_error(y_test, prediction))), 4),
            "mae": round(float(mean_absolute_error(y_test, prediction)), 4),
        }
        benchmarks[name] = metrics
        print(f"  + {name:27s} | MAE {metrics['mae']:.4f} SP/ngày | R² {metrics['r2']:.4f}")
        if metrics["mae"] < best_mae:
            best_name = name
            best_model = model
            best_mae = metrics["mae"]

    if best_model is None:
        raise RuntimeError("Không chọn được mô hình hồi quy")
    best_model.fit(x, np.log1p(y))
    lower, upper = np.quantile(y, [1 / 3, 2 / 3])
    metadata.update(
        {
            "selected_sales_model": best_name,
            "observed_target_rows": int(len(observed)),
            "test_samples": int(len(test_idx)),
            "performance_band_thresholds": {
                "low_max_daily_rate": round(float(lower), 6),
                "medium_max_daily_rate": round(float(upper), 6),
            },
        }
    )
    joblib.dump(best_model, models_dir / "sales_regressor.joblib")
    joblib.dump(metadata, models_dir / "feature_encoders.joblib")
    return {
        "selected_model": best_name,
        **benchmarks[best_name],
        "models": benchmarks,
        "observed_target_rows": int(len(observed)),
        "test_samples": int(len(test_idx)),
        "target": "daily_sold_rate",
        "target_unit": "products_per_day",
        "target_transform": "log1p",
        "performance_band_thresholds": metadata["performance_band_thresholds"],
    }


def train_kmeans_clustering(
    data_path: str | os.PathLike[str] = "data/shopee_products.csv",
    models_dir: str | os.PathLike[str] = "models",
) -> dict[str, Any]:
    """Gom bốn cụm sản phẩm theo tám thuộc tính quan sát được."""
    print("\n--- GOM CỤM SẢN PHẨM BẰNG K-MEANS (K=4) ---")
    data_path = _project_path(data_path)
    models_dir = _project_path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    products = pd.read_csv(data_path)
    matrix = pd.DataFrame(index=products.index)
    medians: dict[str, float] = {}
    for feature in CLUSTER_FEATURES:
        values = pd.to_numeric(products[feature], errors="coerce")
        median = float(values.median())
        medians[feature] = median
        matrix[feature] = values.fillna(median).clip(lower=0)
        if feature in CLUSTER_LOG_FEATURES:
            matrix[feature] = np.log1p(matrix[feature])

    scaler = StandardScaler()
    scaled = scaler.fit_transform(matrix[CLUSTER_FEATURES])
    kmeans = KMeans(n_clusters=4, random_state=RANDOM_SEED, n_init=10)
    labels = kmeans.fit_predict(scaled)
    products["cluster"] = labels
    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    coordinates = pca.fit_transform(scaled)
    products["pca_x"] = coordinates[:, 0]
    products["pca_y"] = coordinates[:, 1]

    profile = products.groupby("cluster").agg(
        avg_historical_sold=("historical_sold", "mean"),
        avg_daily_sold_rate=("daily_sold_rate", "mean"),
        avg_favorite_count=("favorite_count", "mean"),
        avg_rating_count=("rating_count", "mean"),
    )
    remaining = set(range(4))
    velocity_id = max(
        remaining,
        key=lambda cid: float(profile.loc[cid, "avg_daily_sold_rate"])
        if pd.notna(profile.loc[cid, "avg_daily_sold_rate"])
        else -1.0,
    )
    remaining.remove(velocity_id)
    history_id = max(remaining, key=lambda cid: float(profile.loc[cid, "avg_historical_sold"]))
    remaining.remove(history_id)
    engagement_id = max(
        remaining,
        key=lambda cid: float(profile.loc[cid, "avg_favorite_count"] + profile.loc[cid, "avg_rating_count"]),
    )
    remaining.remove(engagement_id)
    cluster_names = {
        velocity_id: "Tốc độ bán quan sát cao",
        history_id: "Lượt bán tích lũy cao",
        engagement_id: "Mức quan tâm và đánh giá cao",
        remaining.pop(): "Nhóm phổ thông hoặc dữ liệu hạn chế",
    }

    profiles: dict[int, Any] = {}
    for cluster_id in range(4):
        subset = products[products["cluster"].eq(cluster_id)]
        profiles[cluster_id] = {
            "cluster_id": cluster_id,
            "name": cluster_names[cluster_id],
            "count": int(len(subset)),
            "avg_price": round(float(subset["price"].mean()), 2),
            "avg_historical_sold": round(float(subset["historical_sold"].mean()), 2),
            "avg_daily_sold_rate": round(float(subset["daily_sold_rate"].mean()), 4)
            if subset["daily_sold_rate"].notna().any()
            else None,
            "avg_rating": round(float(subset["rating_star"].mean()), 3),
            "avg_discount": round(float(subset["discount_rate"].mean()), 2),
            "avg_favorite_count": round(float(subset["favorite_count"].mean()), 2),
            "velocity_coverage": round(float(subset["daily_sold_rate"].notna().mean()), 4),
        }

    sample_points = []
    for _, row in products.sample(min(250, len(products)), random_state=RANDOM_SEED).iterrows():
        sample_points.append(
            {
                "product_id": row["product_id"],
                "product_name": str(row["product_name"])[:70],
                "category": row["category"],
                "cluster": int(row["cluster"]),
                "cluster_name": cluster_names[int(row["cluster"])],
                "pca_x": round(float(row["pca_x"]), 4),
                "pca_y": round(float(row["pca_y"]), 4),
                "price": round(float(row["price"]), 2),
                "historical_sold": int(row["historical_sold"]),
                "daily_sold_rate": round(float(row["daily_sold_rate"]), 4)
                if pd.notna(row["daily_sold_rate"])
                else None,
                "rating": round(float(row["rating_star"]), 2) if pd.notna(row["rating_star"]) else None,
            }
        )

    output = {
        "cluster_profiles": profiles,
        "sample_points": sample_points,
        "explained_variance_ratio": [round(float(value), 4) for value in pca.explained_variance_ratio_],
        "feature_columns": CLUSTER_FEATURES,
        "numeric_medians": medians,
        "log1p_features": sorted(CLUSTER_LOG_FEATURES),
        "currency": "MYR",
    }
    joblib.dump(kmeans, models_dir / "kmeans_model.joblib")
    joblib.dump(scaler, models_dir / "cluster_scaler.joblib")
    (models_dir / "clustering_data.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return output


def mine_association_rules_apriori(
    data_path: str | os.PathLike[str] = "data/shopee_transactions.csv",
    min_support: float = 0.015,
    min_confidence: float = 0.15,
    models_dir: str | os.PathLike[str] = "models",
) -> list[dict[str, Any]]:
    """Khai phá luật hai sản phẩm từ các đơn hoàn thành bằng support/confidence/lift."""
    print("\n--- KHAI PHÁ LUẬT KẾT HỢP TỪ ĐƠN HOÀN THÀNH ---")
    data_path = _project_path(data_path)
    models_dir = _project_path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    orders = pd.read_csv(data_path)
    orders = orders[orders["order_status"].eq("Hoàn thành")]
    transactions: list[list[str]] = []
    for raw in orders["items_json"].dropna():
        items = json.loads(raw)
        if isinstance(items, list):
            cleaned = sorted({str(item).strip() for item in items if str(item).strip()})
            if cleaned:
                transactions.append(cleaned)
    if not transactions:
        raise ValueError("Không có đơn hoàn thành hợp lệ cho Apriori")

    total = len(transactions)
    item_counts: defaultdict[str, int] = defaultdict(int)
    pair_counts: defaultdict[tuple[str, str], int] = defaultdict(int)
    for transaction in transactions:
        for item in transaction:
            item_counts[item] += 1
        for pair in itertools.combinations(transaction, 2):
            pair_counts[pair] += 1
    item_support = {item: count / total for item, count in item_counts.items()}

    rules: list[dict[str, Any]] = []
    for (item_a, item_b), count in pair_counts.items():
        pair_support = count / total
        if pair_support < min_support:
            continue
        for antecedent, consequent in ((item_a, item_b), (item_b, item_a)):
            confidence = pair_support / item_support[antecedent]
            lift = confidence / item_support[consequent]
            if confidence < min_confidence:
                continue
            rules.append(
                {
                    "antecedent": antecedent,
                    "consequent": consequent,
                    "rule_text": f"Nếu có '{antecedent}' → thường kèm '{consequent}'",
                    "support": round(float(pair_support * 100), 2),
                    "confidence": round(float(confidence * 100), 2),
                    "lift": round(float(lift), 3),
                    "cooccurrence_orders": int(count),
                    "completed_order_count": int(total),
                    "insight": "Đây là liên hệ đồng xuất hiện trong mẫu đơn, không phải quan hệ nhân quả.",
                }
            )
    rules.sort(key=lambda rule: (rule["lift"], rule["confidence"], rule["support"]), reverse=True)
    (models_dir / "market_basket_rules.json").write_text(
        json.dumps(rules, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"  + {len(rules)} luật từ {total} đơn hoàn thành")
    return rules


def run_full_training_pipeline() -> dict[str, Any]:
    """Huấn luyện toàn bộ artifact và ghi benchmark kèm provenance dữ liệu."""
    DEFAULT_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    provenance_path = DEFAULT_DATA_DIR / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("source_type") != "kaggle_real_shopee_multi_source":
        raise ValueError("Pipeline chỉ chấp nhận bộ dữ liệu Kaggle thật đã chuẩn hóa")

    sentiment = train_sentiment_and_aspect_models(
        DEFAULT_DATA_DIR / "shopee_reviews.csv", DEFAULT_MODELS_DIR
    )
    sales = train_sales_velocity_model(
        DEFAULT_DATA_DIR / "shopee_products.csv", DEFAULT_MODELS_DIR
    )
    clustering = train_kmeans_clustering(
        DEFAULT_DATA_DIR / "shopee_products.csv", DEFAULT_MODELS_DIR
    )
    rules = mine_association_rules_apriori(
        DEFAULT_DATA_DIR / "shopee_transactions.csv",
        min_support=0.015,
        min_confidence=0.15,
        models_dir=DEFAULT_MODELS_DIR,
    )

    csv_paths = [
        DEFAULT_DATA_DIR / "shopee_products.csv",
        DEFAULT_DATA_DIR / "shopee_reviews.csv",
        DEFAULT_DATA_DIR / "shopee_transactions.csv",
    ]
    benchmarks = {
        "metadata": {
            "data_provenance": provenance["source_type"],
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "random_seed": RANDOM_SEED,
            "sklearn_version": sklearn.__version__,
            "data_rows": {path.name: int(provenance["processed"][path.name]["rows"]) for path in csv_paths},
            "data_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in csv_paths},
            "source_refs": {
                key: value["kaggle_ref"] for key, value in provenance["sources"].items()
            },
            "sentiment_source_rows": sentiment["source_rows"],
            "sentiment_labeled_rows": sentiment["overall_labeled_rows"],
            "sentiment_train_samples": sentiment["train_samples"],
            "sentiment_test_samples": sentiment["test_samples"],
            "sentiment_deployment_samples": sentiment["deployment_samples"],
            "sentiment_split_policy": sentiment["split_policy"],
        },
        "sentiment_models": sentiment["benchmark"],
        "selected_sentiment_model": sentiment["selected_model"],
        "aspect_models": sentiment["aspect_benchmark"],
        "aspect_label_names": sentiment["aspect_label_names"],
        "sales_velocity_regressor": sales,
        "clustering": {
            "clusters_count": len(clustering["cluster_profiles"]),
            "feature_columns": clustering["feature_columns"],
        },
        "market_basket": {
            "rules_count": len(rules),
            "completed_orders": 261,
            "min_support": 0.015,
            "min_confidence": 0.15,
        },
    }
    (DEFAULT_MODELS_DIR / "model_benchmarks.json").write_text(
        json.dumps(benchmarks, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Hai artifact này thuộc pipeline mô phỏng cũ và không có target trong dữ liệu công khai.
    for obsolete in ("growth_classifier.joblib", "risk_classifier.joblib"):
        path = DEFAULT_MODELS_DIR / obsolete
        if path.exists():
            path.unlink()

    print("\nHoàn tất pipeline dữ liệu Kaggle thật.")
    return benchmarks


if __name__ == "__main__":
    run_full_training_pipeline()
