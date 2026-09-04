import hashlib
import json
import re
import unittest
from pathlib import Path

import joblib
import pandas as pd

from data_processing import clean_vietnamese_text, generate_shopee_datasets


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"


class DataAndModelContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.products = pd.read_csv(DATA_DIR / "shopee_products.csv")
        cls.reviews = pd.read_csv(DATA_DIR / "shopee_reviews.csv")
        cls.transactions = pd.read_csv(DATA_DIR / "shopee_transactions.csv")
        cls.provenance = json.loads((DATA_DIR / "provenance.json").read_text(encoding="utf-8"))
        cls.benchmark = json.loads((MODELS_DIR / "model_benchmarks.json").read_text(encoding="utf-8"))

    def test_real_dataset_counts_and_schema(self):
        self.assertEqual(len(self.products), 16_563)
        self.assertEqual(len(self.reviews), 11_700)
        self.assertEqual(len(self.transactions), 325)
        self.assertTrue(
            {
                "source_shop_id",
                "source_item_id",
                "product_url",
                "currency",
                "historical_sold",
                "snapshot_count",
                "daily_sold_rate",
            }.issubset(self.products.columns)
        )
        self.assertTrue(
            {
                "source_split",
                "raw_comment",
                "cleaned_comment",
                "sentiment_label",
                "price_sentiment",
                "shipping_sentiment",
                "quality_sentiment",
                "shop_service_sentiment",
            }.issubset(self.reviews.columns)
        )
        self.assertTrue({"items_json", "item_count", "order_status"}.issubset(self.transactions.columns))
        self.assertFalse({"monthly_sold", "growth_potential", "risk_level", "return_rate"} & set(self.products.columns))
        self.assertEqual(set(self.products["currency"]), {"MYR"})
        self.assertEqual(int(self.products["daily_sold_rate"].notna().sum()), 2_263)

    def test_provenance_hashes_match_processed_files(self):
        self.assertEqual(self.provenance["source_type"], "kaggle_real_shopee_multi_source")
        expected_refs = {
            "yoongsin/shopee-sample-data",
            "cthng123/absa-vietnamese",
            "nugrahmaindonesa/shopee-seller-transaction",
        }
        self.assertEqual(
            {source["kaggle_ref"] for source in self.provenance["sources"].values()}, expected_refs
        )
        for filename, metadata in self.provenance["processed"].items():
            path = DATA_DIR / filename
            self.assertEqual(metadata["rows"], len(pd.read_csv(path)))
            self.assertEqual(metadata["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())

    def test_transaction_allowlist_and_raw_pii_cache_removal(self):
        allowed = {
            "transaction_id",
            "order_status",
            "created_at",
            "completed_at",
            "payment_method",
            "items_json",
            "items",
            "item_count",
            "total_quantity",
            "source_dataset",
        }
        self.assertEqual(set(self.transactions.columns), allowed)
        self.assertTrue(self.transactions["transaction_id"].map(lambda value: bool(re.fullmatch(r"ORD_[0-9A-F]{20}", value))).all())
        self.assertFalse((DATA_DIR / "external" / "seller-transactions-v1.zip").exists())
        self.assertEqual(int(self.transactions["item_count"].gt(1).sum()), 67)
        self.assertEqual(int(self.transactions["order_status"].eq("Hoàn thành").sum()), 261)

    def test_text_normalization_and_synthetic_generator_is_blocked(self):
        cleaned = clean_vietnamese_text("SP đẹpppp, shop ship nhanh ❤️ https://example.com")
        self.assertIn("sản phẩm", cleaned.replace("_", " "))
        self.assertNotIn("https", cleaned)
        self.assertNotIn("đẹpp", cleaned)
        with self.assertRaisesRegex(RuntimeError, "vô hiệu hóa generator mô phỏng"):
            generate_shopee_datasets("unused")

    def test_runtime_artifacts_load_and_obsolete_models_are_absent(self):
        names = (
            "sentiment_model.joblib",
            "tfidf_vectorizer.joblib",
            "aspect_models.joblib",
            "sales_regressor.joblib",
            "feature_encoders.joblib",
            "kmeans_model.joblib",
            "cluster_scaler.joblib",
        )
        artifacts = {name: joblib.load(MODELS_DIR / name) for name in names}
        encoders = artifacts["feature_encoders.joblib"]
        self.assertEqual(len(encoders["feature_cols"]), 9)
        self.assertEqual(artifacts["sales_regressor.joblib"].n_features_in_, 9)
        self.assertEqual(artifacts["kmeans_model.joblib"].n_features_in_, 8)
        self.assertEqual(len(artifacts["aspect_models.joblib"]), 8)
        self.assertFalse((MODELS_DIR / "growth_classifier.joblib").exists())
        self.assertFalse((MODELS_DIR / "risk_classifier.joblib").exists())

    def test_benchmark_uses_source_test_and_real_provenance(self):
        metadata = self.benchmark["metadata"]
        self.assertEqual(metadata["data_provenance"], "kaggle_real_shopee_multi_source")
        self.assertEqual(metadata["sentiment_source_rows"], 11_700)
        self.assertGreater(metadata["sentiment_train_samples"], 7_000)
        self.assertGreater(metadata["sentiment_test_samples"], 1_800)
        self.assertIn("source_test", metadata["sentiment_split_policy"])
        for filename, expected_hash in metadata["data_sha256"].items():
            self.assertEqual(expected_hash, hashlib.sha256((DATA_DIR / filename).read_bytes()).hexdigest())
        selected = self.benchmark["sentiment_models"][self.benchmark["selected_sentiment_model"]]
        self.assertEqual(sum(map(sum, selected["confusion_matrix"])), metadata["sentiment_test_samples"])
        self.assertEqual(self.benchmark["sales_velocity_regressor"]["observed_target_rows"], 2_263)
        self.assertEqual(len(self.benchmark["aspect_models"]), 8)

    def test_clustering_and_apriori_outputs(self):
        clustering = json.loads((MODELS_DIR / "clustering_data.json").read_text(encoding="utf-8"))
        profiles = clustering["cluster_profiles"]
        self.assertEqual(len(profiles), 4)
        self.assertEqual(sum(profile["count"] for profile in profiles.values()), 16_563)
        self.assertEqual(len({profile["name"] for profile in profiles.values()}), 4)
        self.assertEqual(len(clustering["feature_columns"]), 8)

        rules = json.loads((MODELS_DIR / "market_basket_rules.json").read_text(encoding="utf-8"))
        self.assertGreater(len(rules), 0)
        self.assertTrue(all(rule["support"] >= 1.5 for rule in rules))
        self.assertTrue(all(rule["confidence"] >= 15 for rule in rules))
        self.assertTrue(all(rule["completed_order_count"] == 261 for rule in rules))


if __name__ == "__main__":
    unittest.main()
