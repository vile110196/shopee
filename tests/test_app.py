import math
import unittest
from pathlib import Path

import app as app_module


class AppContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app_module.app.config.update(TESTING=True)
        cls.client = app_module.app.test_client()
        cls.first_product = app_module.df_products.iloc[0]

    def test_html_pages_exist(self):
        expected_markers = {
            "/": "Tổng quan dữ liệu",
            "/report": "Báo cáo nghiệm thu kỹ thuật",
            "/theory": "Cơ sở lý thuyết và công thức sử dụng",
        }
        for path, marker in expected_markers.items():
            with self.subTest(path=path):
                response = self.client.get(path)
                self.assertEqual(response.status_code, 200)
                self.assertIn(marker, response.get_data(as_text=True))

    def test_responsive_shell_and_mathjax_contract(self):
        homepage = self.client.get("/").get_data(as_text=True)
        theory = self.client.get("/theory").get_data(as_text=True)
        report = self.client.get("/report").get_data(as_text=True)

        for marker in ("app-sidebar", "sidebar-backdrop", "sidebar-open", "sidebar-close"):
            self.assertIn(marker, homepage)
        self.assertIn('<html class="light" lang="vi">', homepage)
        self.assertIn('id="MathJax-script"', theory)
        self.assertIn(r"\operatorname{tfidf}", theory)
        self.assertIn(r"\arg\max", theory)
        self.assertIn(r"\operatorname{support}", theory)
        self.assertIn(r"\operatorname{MAE}", report)

    def test_ui_sources_do_not_contain_common_mojibake(self):
        project_root = Path(app_module.__file__).resolve().parent
        ui_files = [
            project_root / "templates" / "base.html",
            project_root / "templates" / "index.html",
            project_root / "templates" / "report.html",
            project_root / "templates" / "theory.html",
            project_root / "static" / "js" / "app.js",
        ]
        mojibake_markers = ("Ã¡", "Ã¢", "Ã ", "Ã©", "Ãª", "Ä‘", "Æ°", "áº", "á»", "â€“", "â€”", "â†’", "ðŸ", "�")
        for path in ui_files:
            text = path.read_text(encoding="utf-8")
            for marker in mojibake_markers:
                with self.subTest(path=path.name, marker=marker):
                    self.assertNotIn(marker, text)

    def test_overview_uses_real_dataset_and_observable_quality_fields(self):
        response = self.client.get("/api/overview_stats")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["status"], "success")
        self.assertEqual(payload["kpis"]["total_products"], 16_563)
        self.assertEqual(payload["kpis"]["total_reviews"], 11_700)
        self.assertEqual(payload["currency"], "MYR")
        self.assertEqual(payload["quality_dist"]["with_velocity"], 2_263)
        self.assertEqual(
            payload["quality_dist"]["with_velocity"] + payload["quality_dist"]["without_velocity"],
            16_563,
        )
        self.assertGreater(len(payload["listing_alerts"]), 0)
        self.assertEqual(len(payload["top_products"]), 5)
        self.assertNotIn("risk_dist", payload)

    def test_read_only_api_contracts(self):
        contracts = {
            "/api/wordcloud": "words",
            "/api/model_benchmarks": "benchmarks",
            "/api/clustering": "clustering",
            "/api/market_basket": "rules",
            "/api/sample_products": "samples",
            "/api/sample_comments": "samples",
        }
        for path, key in contracts.items():
            with self.subTest(path=path):
                response = self.client.get(path)
                self.assertEqual(response.status_code, 200)
                payload = response.get_json()
                self.assertEqual(payload["status"], "success")
                self.assertIn(key, payload)

    def test_product_lookup_by_product_id_item_id_and_real_url(self):
        expected_id = self.first_product["product_id"]
        references = (
            expected_id,
            str(int(self.first_product["source_item_id"])),
            self.first_product["product_url"],
        )
        for reference in references:
            with self.subTest(reference=reference):
                response = self.client.get("/api/product_lookup", query_string={"reference": reference})
                self.assertEqual(response.status_code, 200)
                product = response.get_json()["product"]
                self.assertEqual(product["product_id"], expected_id)
                self.assertEqual(product["currency"], "MYR")
        self.assertEqual(
            self.client.get("/api/product_lookup", query_string={"reference": "MY_0_0"}).status_code,
            404,
        )

    def test_sales_velocity_prediction_contract(self):
        sample = self.client.get("/api/sample_products").get_json()["samples"][0]
        response = self.client.post("/api/predict_product", json=sample)
        self.assertEqual(response.status_code, 200)
        prediction = response.get_json()["prediction"]
        self.assertGreaterEqual(prediction["estimated_daily_sold_rate"], 0)
        self.assertAlmostEqual(
            prediction["estimated_30d_sales_pace"],
            prediction["estimated_daily_sold_rate"] * 30,
            delta=0.02,
        )
        self.assertEqual(prediction["currency"], "MYR")
        self.assertEqual(prediction["training_rows"], 2_263)
        self.assertGreater(len(prediction["feature_impacts"]), 0)
        self.assertNotIn("growth_potential", prediction)
        self.assertNotIn("risk_level", prediction)
        self.assertNotIn("confidence_percent", prediction)
        self.assertTrue(math.isfinite(prediction["estimated_30d_gross_value_myr"]))

    def test_prediction_rejects_invalid_inputs(self):
        invalid_cases = [
            ({"price": -1}, "price"),
            ({"rating_star": 6}, "rating_star"),
            ({"discount_rate": 101}, "discount_rate"),
            ({"snapshot_count": 0}, "snapshot_count"),
            ({"category": "Không tồn tại"}, "category"),
        ]
        for payload, field in invalid_cases:
            with self.subTest(field=field):
                response = self.client.post("/api/predict_product", json=payload)
                self.assertEqual(response.status_code, 400)
                self.assertIn(field, response.get_json()["message"])

    def test_sentiment_and_eight_model_based_aspects(self):
        response = self.client.post(
            "/api/analyze_sentiment",
            json={"text": "Giày đẹp, đi êm, giá hợp lý và shop giao hàng nhanh"},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertIn(payload["sentiment_label"], {"Tích cực", "Trung lập", "Tiêu cực"})
        self.assertGreaterEqual(payload["confidence_percent"], 0)
        self.assertLessEqual(payload["confidence_percent"], 100)
        self.assertEqual(len(payload["aspects"]), 8)
        self.assertEqual(
            {aspect["label"] for aspect in payload["aspects"]}
            <= {"Không đề cập", "Tiêu cực", "Tích cực", "Trung lập"},
            True,
        )
        self.assertIn("Logistic Regression", payload["aspect_method"])
        self.assertEqual(self.client.post("/api/analyze_sentiment", json={}).status_code, 400)
        self.assertEqual(self.client.post("/api/analyze_sentiment", json={"text": 123}).status_code, 400)
        self.assertEqual(
            self.client.post("/api/analyze_sentiment", json={"text": "x" * 5_001}).status_code,
            400,
        )


if __name__ == "__main__":
    unittest.main()
