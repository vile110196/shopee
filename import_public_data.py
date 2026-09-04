# -*- coding: utf-8 -*-
"""Tải và chuẩn hóa ba bộ dữ liệu Shopee công bố trên Kaggle.

Mỗi nguồn phục vụ một nhánh độc lập; không ghép thực thể giữa các nguồn:
- Shopee Sales Apr-May 2023: sản phẩm, giá, rating và số bán tích lũy.
- ABSA Vietnamese: review tiếng Việt với tám nhãn khía cạnh do người gán.
- Shopee Seller Transaction: giỏ hàng; chỉ giữ allowlist không chứa PII.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import tempfile
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data_processing import clean_vietnamese_text


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_CACHE_DIR = DEFAULT_DATA_DIR / "external"

KAGGLE_SOURCES: dict[str, dict[str, Any]] = {
    "products": {
        "dataset": "Shopee Sales Data Apr - May 2023",
        "ref": "yoongsin/shopee-sample-data",
        "version": 1,
        "last_updated": "2024-04-04T06:28:25.503Z",
        "license": "Other (specified in description)",
        "archive": "sales-v1.zip",
        "archive_sha256": "c82bd733ba8813fe08f934f8c68c88d9e2defe8e3b14953c55c5b9815b142c27",
        "files": {
            "20240121_shopee_sample_data (1).csv":
                "afed3932287c81df7eefcd7397b723cc2e2ca71d73690dc661ad250c7c78bc69",
        },
    },
    "reviews": {
        "dataset": "ABSA Vietnamese",
        "ref": "cthng123/absa-vietnamese",
        "version": 1,
        "last_updated": "2024-05-14T15:27:14.390Z",
        "license": "CC BY 4.0",
        "archive": "absa-v1.zip",
        "archive_sha256": "c494fba611da8ff89b9b198900a340ae48c3f0e131ade8d1c5fca877fa7cdf97",
        "files": {
            "train_data.csv":
                "58350af4348ac3edec26278d98e249f556e3437545d5b1dd278a499418cc4c4f",
            "val_data.csv":
                "d128932e9fa59ddb736b449c2241718f7be4b590d258e5519187368f3ff0b824",
            "test_data.csv":
                "397bce8c6f45c9b8ff1d358e3356aeda7185acec28c9c5e154f13ee48bef6166",
        },
    },
    "transactions": {
        "dataset": "Shopee Seller Transaction",
        "ref": "nugrahmaindonesa/shopee-seller-transaction",
        "version": 1,
        "last_updated": "2025-12-20T02:19:21.037Z",
        "license": "Apache 2.0",
        "archive": "seller-transactions-v1.zip",
        "archive_sha256": "fd9b0c2a2a92dbfedb927ad299d81ebf06e7ba6b70b175ee09b9717691be5e46",
        "files": {
            "Shopee Seller Transaction.csv":
                "b02d725d1bd2ff8e7ce401bac17684d74f5464c992edcd7a33f6e3459658504c",
        },
    },
}

SALES_REQUIRED_COLUMNS = {
    "price_ori",
    "delivery",
    "item_category_detail",
    "specification",
    "title",
    "w_date",
    "link_ori",
    "item_rating",
    "seller_name",
    "price_actual",
    "total_rating",
    "total_sold",
    "favorite",
    "desc",
}

ABSA_ASPECT_COLUMNS = {
    "Price": "price_sentiment",
    "Shipping": "shipping_sentiment",
    "Outlook": "outlook_sentiment",
    "Quality": "quality_sentiment",
    "Size": "size_sentiment",
    "Shop_Service": "shop_service_sentiment",
    "General": "general_sentiment",
    "Others": "others_sentiment",
}

TRANSACTION_REQUIRED_COLUMNS = {
    "No. Pesanan",
    "Status Pesanan",
    "Waktu Pesanan Dibuat",
    "Waktu Pesanan Selesai",
    "Metode Pembayaran",
    "Nama Produk",
    "Jumlah",
}

TRANSACTION_PII_COLUMNS = {
    "No. Pesanan",
    "No. Resi",
    "Catatan dari Pembeli",
    "Username (Pembeli)",
    "Nama Penerima",
    "No. Telepon",
    "Alamat Pengiriman",
    "Kota/Kabupaten",
    "Provinsi",
}

SENTIMENT_NAMES = {
    0: "Tiêu cực",
    1: "Tích cực",
    2: "Trung lập",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download_verified(url: str, destination: Path, expected_sha256: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and sha256_file(destination) == expected_sha256:
        return

    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.unlink(missing_ok=True)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "UIT-Shopee-coursework-real-data/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    actual_sha256 = sha256_file(temporary)
    if actual_sha256 != expected_sha256:
        temporary.unlink(missing_ok=True)
        raise ValueError(
            f"SHA-256 archive không khớp cho {destination.name}: "
            f"cần {expected_sha256}, nhận {actual_sha256}"
        )
    temporary.replace(destination)


def ensure_kaggle_archives(cache_dir: Path = DEFAULT_CACHE_DIR, download: bool = False) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    for source in KAGGLE_SOURCES.values():
        archive = cache_dir / source["archive"]
        valid = archive.is_file() and sha256_file(archive) == source["archive_sha256"]
        if valid:
            continue
        if not download:
            raise FileNotFoundError(
                f"Thiếu archive Kaggle đã xác minh: {archive}. "
                "Chạy import_public_data.py --download."
            )
        url = (
            f"https://www.kaggle.com/api/v1/datasets/download/{source['ref']}"
            f"?datasetVersionNumber={source['version']}"
        )
        _download_verified(url, archive, source["archive_sha256"])


def _extract_verified_source(source_key: str, cache_dir: Path, destination: Path) -> dict[str, Path]:
    source = KAGGLE_SOURCES[source_key]
    archive_path = cache_dir / source["archive"]
    extracted: dict[str, Path] = {}
    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        for member_name, expected_sha256 in source["files"].items():
            if member_name not in names:
                raise ValueError(f"{archive_path.name} thiếu file {member_name}")
            output_path = destination / source_key / Path(member_name).name
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member_name) as source_handle, output_path.open("wb") as output:
                shutil.copyfileobj(source_handle, output)
            actual_sha256 = sha256_file(output_path)
            if actual_sha256 != expected_sha256:
                raise ValueError(
                    f"SHA-256 file nguồn không khớp cho {member_name}: "
                    f"cần {expected_sha256}, nhận {actual_sha256}"
                )
            extracted[member_name] = output_path
    return extracted


def _require_columns(frame: pd.DataFrame, required: set[str], dataset_name: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{dataset_name} thiếu cột: {', '.join(missing)}")


def _parse_compact_number(value: Any) -> float:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return np.nan
    text = str(value).strip().lower().replace(",", "")
    match = re.search(r"(-?\d+(?:\.\d+)?)\s*([km]?)", text)
    if not match:
        return np.nan
    number = float(match.group(1))
    multiplier = {"": 1.0, "k": 1_000.0, "m": 1_000_000.0}[match.group(2)]
    return number * multiplier


def _last_valid(series: pd.Series, default: float = np.nan) -> float:
    values = series.dropna()
    return default if values.empty else float(values.iloc[-1])


def _text_value(value: Any) -> str:
    # CSV nguồn có mô tả nhiều dòng với khoảng trắng cuối dòng. Chuẩn hóa về một dòng
    # giúp output ổn định giữa hệ điều hành và không làm thay đổi nội dung từ vựng.
    return "" if pd.isna(value) else re.sub(r"\s+", " ", str(value)).strip()


def _split_category(value: Any) -> tuple[str, str]:
    parts = [part.strip() for part in _text_value(value).split("|") if part.strip()]
    if parts and parts[0].casefold() == "shopee":
        parts = parts[1:]
    if not parts:
        return "Không xác định", "Không xác định"
    return parts[0], " > ".join(parts[1:]) if len(parts) > 1 else parts[0]


def _extract_source_ids(url: str, fallback: str) -> tuple[str, str, str]:
    match = re.search(r"-i\.(\d+)\.(\d+)(?:\?|$)", url)
    if match:
        shop_id, item_id = match.groups()
        return f"MY_{shop_id}_{item_id}", item_id, shop_id
    token = hashlib.sha256(fallback.encode("utf-8")).hexdigest()[:20].upper()
    return f"MY_HASH_{token}", "", ""


def build_products(source_path: Path, output_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = pd.read_csv(source_path, low_memory=False)
    _require_columns(raw, SALES_REQUIRED_COLUMNS, "Shopee Sales Apr-May 2023")

    raw["_crawl_date"] = pd.to_datetime(raw["w_date"], errors="coerce")
    raw["_price"] = pd.to_numeric(raw["price_actual"], errors="coerce")
    raw["_original_price"] = pd.to_numeric(raw["price_ori"], errors="coerce")
    raw["_rating"] = pd.to_numeric(raw["item_rating"], errors="coerce")
    raw["_rating_count"] = pd.to_numeric(raw["total_rating"], errors="coerce")
    raw["_sold"] = pd.to_numeric(raw["total_sold"], errors="coerce")
    raw["_favorite"] = raw["favorite"].map(_parse_compact_number)
    raw["_source_order"] = np.arange(len(raw))
    source_ids = raw["link_ori"].astype(str).str.extract(r"-i\.(\d+)\.(\d+)(?:\?|$)")
    raw["_source_shop_id"] = source_ids[0]
    raw["_source_item_id"] = source_ids[1]
    raw["_product_key"] = np.where(
        raw["_source_shop_id"].notna() & raw["_source_item_id"].notna(),
        raw["_source_shop_id"].astype(str) + ":" + raw["_source_item_id"].astype(str),
        "hash:" + raw["idHash"].astype(str),
    )
    raw = raw.sort_values(["_product_key", "_crawl_date", "_source_order"])

    output_rows: list[dict[str, Any]] = []
    negative_deltas = 0
    for _, group in raw.groupby("_product_key", sort=False, dropna=False):
        latest = group.iloc[-1]
        category, subcategory = _split_category(latest["item_category_detail"])
        url = _text_value(latest["link_ori"])
        fallback_id = str(latest.get("idHash") or latest.get("id") or latest["_source_order"])
        product_id, source_item_id, source_shop_id = _extract_source_ids(url, fallback_id)

        sold_observations = (
            group.loc[group["_crawl_date"].notna() & group["_sold"].notna(), ["_crawl_date", "_sold"]]
            .groupby("_crawl_date", as_index=False)
            .last()
            .sort_values("_crawl_date")
        )
        observation_days = 0
        observed_sold_delta = np.nan
        daily_sold_rate = np.nan
        if len(sold_observations) >= 2:
            observation_days = int(
                (sold_observations["_crawl_date"].iloc[-1] - sold_observations["_crawl_date"].iloc[0]).days
            )
            delta = float(sold_observations["_sold"].iloc[-1] - sold_observations["_sold"].iloc[0])
            if observation_days > 0 and delta >= 0:
                observed_sold_delta = delta
                daily_sold_rate = delta / observation_days
            elif delta < 0:
                negative_deltas += 1

        price = _last_valid(group["_price"])
        original_price = _last_valid(group["_original_price"], price)
        if not np.isfinite(original_price) or original_price <= 0:
            original_price = price
        discount_rate = 0.0
        if np.isfinite(price) and np.isfinite(original_price) and original_price > 0:
            discount_rate = max(0.0, min(100.0, (original_price - price) / original_price * 100.0))

        crawl_dates = group["_crawl_date"].dropna()
        output_rows.append({
            "product_id": product_id,
            "source_item_id": source_item_id,
            "source_shop_id": source_shop_id,
            "product_url": url,
            "product_name": _text_value(latest["title"]),
            "category": category,
            "subcategory": subcategory,
            "shop_name": _text_value(latest["seller_name"]),
            "shop_location": _text_value(latest["delivery"]),
            "currency": "MYR",
            "price": price,
            "original_price": original_price,
            "discount_rate": round(discount_rate, 4),
            "rating_star": _last_valid(group["_rating"]),
            "rating_count": int(max(0, _last_valid(group["_rating_count"], 0))),
            "historical_sold": int(max(0, _last_valid(group["_sold"], 0))),
            "favorite_count": int(max(0, _last_valid(group["_favorite"], 0))),
            "snapshot_count": int(len(group)),
            "first_seen_date": crawl_dates.min().date().isoformat() if not crawl_dates.empty else "",
            "last_seen_date": crawl_dates.max().date().isoformat() if not crawl_dates.empty else "",
            "observation_days": observation_days,
            "observed_sold_delta": observed_sold_delta,
            "daily_sold_rate": daily_sold_rate,
            "description": _text_value(latest["desc"]),
            "source_dataset": "Shopee Sales Apr-May 2023",
        })

    products = pd.DataFrame(output_rows)
    products = products.loc[products["price"].notna() & (products["price"] > 0)].copy()
    products = products.sort_values("product_id").reset_index(drop=True)
    if products["product_id"].duplicated().any():
        raise ValueError("Mã sản phẩm chuẩn hóa bị trùng")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    products.to_csv(output_path, index=False, encoding="utf-8-sig", lineterminator="\n")
    summary = {
        "raw_snapshot_rows": int(len(raw)),
        "processed_unique_products": int(len(products)),
        "unique_shops": int(products["source_shop_id"].replace("", np.nan).nunique()),
        "crawl_date_min": products["first_seen_date"].replace("", np.nan).min(),
        "crawl_date_max": products["last_seen_date"].replace("", np.nan).max(),
        "products_with_observed_velocity": int(products["daily_sold_rate"].notna().sum()),
        "negative_sold_deltas_excluded": int(negative_deltas),
        "currency": "MYR",
        "deduplication_key": "shop_id_plus_item_id_keep_latest_snapshot",
        "velocity_rule": "nonnegative_change_in_total_sold_divided_by_distinct_snapshot_day_span",
    }
    return products, summary


def _overall_sentiment(row: pd.Series) -> tuple[str | None, str | None]:
    general = int(row["General"])
    if general in SENTIMENT_NAMES:
        return SENTIMENT_NAMES[general], "human_general_aspect"
    labels = [int(row[column]) for column in ABSA_ASPECT_COLUMNS if int(row[column]) in SENTIMENT_NAMES]
    if labels and len(set(labels)) == 1:
        return SENTIMENT_NAMES[labels[0]], "unanimous_human_aspects"
    return None, None


def build_reviews(
    source_paths: dict[str, Path],
    output_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    split_files = (
        ("train_data.csv", "train"),
        ("val_data.csv", "validation"),
        ("test_data.csv", "test"),
    )
    frames: list[pd.DataFrame] = []
    split_counts: dict[str, int] = {}
    for filename, split in split_files:
        frame = pd.read_csv(source_paths[filename])
        _require_columns(
            frame,
            {"Review", *ABSA_ASPECT_COLUMNS.keys()},
            f"ABSA Vietnamese/{filename}",
        )
        split_counts[split] = int(len(frame))
        frame["source_split"] = split
        frame["source_row"] = np.arange(1, len(frame) + 1)
        frames.append(frame)

    raw = pd.concat(frames, ignore_index=True)
    overall = raw.apply(_overall_sentiment, axis=1, result_type="expand")
    reviews = pd.DataFrame({
        "review_id": [
            f"ABSA_{split.upper()}_{row:05d}"
            for split, row in zip(raw["source_split"], raw["source_row"])
        ],
        "source_split": raw["source_split"],
        "raw_comment": raw["Review"].astype(str).str.strip(),
    })
    reviews["cleaned_comment"] = reviews["raw_comment"].map(clean_vietnamese_text)
    reviews["sentiment_label"] = overall[0]
    reviews["sentiment_label_source"] = overall[1]
    reviews["category"] = "Giày dép"
    reviews["channel"] = "Shopee"
    for source_column, output_column in ABSA_ASPECT_COLUMNS.items():
        reviews[output_column] = raw[source_column].astype(int)
    reviews["source_dataset"] = "ABSA Vietnamese"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    reviews.to_csv(output_path, index=False, encoding="utf-8-sig", lineterminator="\n")
    summary = {
        "source_rows": split_counts,
        "processed_rows": int(len(reviews)),
        "duplicate_cleaned_reviews": int(reviews["cleaned_comment"].duplicated().sum()),
        "overall_labeled_rows": int(reviews["sentiment_label"].notna().sum()),
        "mixed_or_unlabeled_overall_rows": int(reviews["sentiment_label"].isna().sum()),
        "overall_label_counts": {
            str(key): int(value)
            for key, value in reviews["sentiment_label"].value_counts().items()
        },
        "overall_label_rule": (
            "human General aspect when present; otherwise unanimous non-None human aspect labels; "
            "mixed labels remain null"
        ),
        "aspect_encoding": {"none": -1, "negative": 0, "positive": 1, "neutral": 2},
    }
    return reviews, summary


def _parse_indonesian_integer(value: Any) -> int:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return 0
    text = re.sub(r"[^0-9-]", "", str(value))
    return int(text) if text and text != "-" else 0


def _first_nonempty(series: pd.Series) -> str:
    values = series.dropna().astype(str).map(str.strip)
    values = values[values != ""]
    return "" if values.empty else values.iloc[0]


def build_transactions(source_path: Path, output_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = pd.read_csv(source_path)
    _require_columns(raw, TRANSACTION_REQUIRED_COLUMNS, "Shopee Seller Transaction")

    output_rows: list[dict[str, Any]] = []
    namespace = "nugrahmaindonesa/shopee-seller-transaction:v1:"
    for raw_order_id, group in raw.groupby("No. Pesanan", sort=False):
        product_names = [
            name
            for name in dict.fromkeys(
                group["Nama Produk"].dropna().astype(str).map(str.strip).tolist()
            )
            if name
        ]
        token = hashlib.sha256(f"{namespace}{raw_order_id}".encode("utf-8")).hexdigest()[:20].upper()
        status_raw = _first_nonempty(group["Status Pesanan"])
        order_status = {"Selesai": "Hoàn thành", "Batal": "Đã hủy"}.get(status_raw, status_raw)
        created_values = pd.to_datetime(group["Waktu Pesanan Dibuat"], errors="coerce").dropna()
        completed_values = pd.to_datetime(group["Waktu Pesanan Selesai"], errors="coerce").dropna()
        quantities = group["Jumlah"].map(_parse_indonesian_integer)
        output_rows.append({
            "transaction_id": f"ORD_{token}",
            "order_status": order_status,
            "created_at": created_values.min().isoformat(sep=" ") if not created_values.empty else "",
            "completed_at": completed_values.max().isoformat(sep=" ") if not completed_values.empty else "",
            "payment_method": _first_nonempty(group["Metode Pembayaran"]),
            "items_json": json.dumps(product_names, ensure_ascii=False, separators=(",", ":")),
            "items": " | ".join(product_names),
            "item_count": len(product_names),
            "total_quantity": int(quantities.sum()),
            "source_dataset": "Shopee Seller Transaction",
        })

    transactions = pd.DataFrame(output_rows)
    transactions = transactions.sort_values(["created_at", "transaction_id"]).reset_index(drop=True)
    if set(transactions.columns) & TRANSACTION_PII_COLUMNS:
        raise ValueError("CSV giao dịch đầu ra còn cột PII")
    if transactions["transaction_id"].duplicated().any():
        raise ValueError("Mã giao dịch băm bị trùng")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    transactions.to_csv(output_path, index=False, encoding="utf-8-sig", lineterminator="\n")
    summary = {
        "raw_line_items": int(len(raw)),
        "processed_orders": int(len(transactions)),
        "unique_products": int(raw["Nama Produk"].nunique()),
        "orders_with_multiple_products": int((transactions["item_count"] >= 2).sum()),
        "completed_orders": int((transactions["order_status"] == "Hoàn thành").sum()),
        "cancelled_orders": int((transactions["order_status"] == "Đã hủy").sum()),
        "privacy_rule": "allowlist_only; raw order IDs hashed; buyer, recipient, phone, address and tracking fields dropped",
        "retained_columns": list(transactions.columns),
    }
    return transactions, summary


def _source_provenance(source_key: str, summary: dict[str, Any]) -> dict[str, Any]:
    source = KAGGLE_SOURCES[source_key]
    return {
        "dataset": source["dataset"],
        "kaggle_ref": source["ref"],
        "homepage": f"https://www.kaggle.com/datasets/{source['ref']}",
        "version": source["version"],
        "last_updated": source["last_updated"],
        "license": source["license"],
        "archive_sha256": source["archive_sha256"],
        "source_file_sha256": source["files"],
        **summary,
    }


def write_provenance(
    staging_dir: Path,
    product_summary: dict[str, Any],
    review_summary: dict[str, Any],
    transaction_summary: dict[str, Any],
) -> dict[str, Any]:
    processed_names = (
        "shopee_products.csv",
        "shopee_reviews.csv",
        "shopee_transactions.csv",
    )
    provenance = {
        "source_type": "kaggle_real_shopee_multi_source",
        "generated_by": "import_public_data.py",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "joinability": (
            "Three independent source namespaces; no cross-source product, review, user, "
            "or order identity joins are claimed."
        ),
        "sources": {
            "product_analytics": _source_provenance("products", product_summary),
            "review_and_aspect_analytics": _source_provenance("reviews", review_summary),
            "market_basket": _source_provenance("transactions", transaction_summary),
        },
        "processed": {
            name: {
                "rows": int(len(pd.read_csv(staging_dir / name))),
                "sha256": sha256_file(staging_dir / name),
            }
            for name in processed_names
        },
        "limitations": [
            "Product snapshots are from Shopee Malaysia and use MYR.",
            "ABSA reviews are Vietnamese shoe-product comments and have no product IDs.",
            "Seller transactions are from one Indonesian shop and are not linked to the product snapshots.",
            "Sales velocity exists only for products observed on at least two distinct dates.",
            "No return-rate, delivery-SLA, causal growth, or operational-risk target is available.",
        ],
    }
    (staging_dir / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return provenance


def _purge_sensitive_cache(cache_dir: Path) -> None:
    source = KAGGLE_SOURCES["transactions"]
    (cache_dir / source["archive"]).unlink(missing_ok=True)
    extracted_dir = cache_dir / Path(source["archive"]).stem
    if extracted_dir.is_dir():
        shutil.rmtree(extracted_dir)


def import_kaggle_data(
    data_dir: Path = DEFAULT_DATA_DIR,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    *,
    download: bool = False,
) -> dict[str, Any]:
    data_dir = data_dir.resolve()
    cache_dir = cache_dir.resolve()
    ensure_kaggle_archives(cache_dir, download=download)

    try:
        with tempfile.TemporaryDirectory(prefix="shopee-kaggle-source-") as raw_tmp:
            raw_root = Path(raw_tmp)
            product_files = _extract_verified_source("products", cache_dir, raw_root)
            review_files = _extract_verified_source("reviews", cache_dir, raw_root)
            transaction_files = _extract_verified_source("transactions", cache_dir, raw_root)

            data_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(
                prefix=".shopee-import-",
                dir=data_dir,
            ) as staging_tmp:
                staging = Path(staging_tmp)
                products, product_summary = build_products(
                    product_files["20240121_shopee_sample_data (1).csv"],
                    staging / "shopee_products.csv",
                )
                reviews, review_summary = build_reviews(
                    review_files,
                    staging / "shopee_reviews.csv",
                )
                transactions, transaction_summary = build_transactions(
                    transaction_files["Shopee Seller Transaction.csv"],
                    staging / "shopee_transactions.csv",
                )
                provenance = write_provenance(
                    staging,
                    product_summary,
                    review_summary,
                    transaction_summary,
                )

                for filename in (
                    "shopee_products.csv",
                    "shopee_reviews.csv",
                    "shopee_transactions.csv",
                    "provenance.json",
                ):
                    (staging / filename).replace(data_dir / filename)
    finally:
        _purge_sensitive_cache(cache_dir)

    print(
        f"Shopee Sales: {len(products):,} sản phẩm duy nhất; "
        f"{product_summary['products_with_observed_velocity']:,} có tốc độ bán quan sát được."
    )
    print(
        f"ABSA Vietnamese: {len(reviews):,} review; "
        f"{review_summary['overall_labeled_rows']:,} có nhãn cảm xúc tổng hợp từ nhãn người gán."
    )
    print(
        f"Shopee Seller Transaction: {len(transactions):,} đơn đã ẩn danh; "
        f"{transaction_summary['orders_with_multiple_products']:,} đơn có nhiều sản phẩm."
    )
    print("Đã xóa raw cache giao dịch chứa thông tin người mua.")
    return {
        "products": products,
        "reviews": reviews,
        "transactions": transactions,
        "provenance": provenance,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--download",
        action="store_true",
        help="Tải lại archive Kaggle version đã khóa khi cache chưa có hoặc sai hash.",
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    args = parser.parse_args()
    import_kaggle_data(args.data_dir, args.cache_dir, download=args.download)


if __name__ == "__main__":
    main()
