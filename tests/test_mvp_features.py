import json
import unittest

from mvp.agent import MVPCommerceAgent, MVPConfig, PipelineStage
from mvp.router import MVPRouter
from src.schema import DecomposedQuery, IntentType


TEST_CONFIG = {
    "databases": {
        "sqlite": {"path": "data/processed/catalog.db"},
        "lancedb": {"path": "data/processed/lancedb", "table_name": "product_vectors"},
    },
    "embeddings": {
        "visual": {"dimension": 768},
        "semantic": {"dimension": 1024},
    },
}


class FakeSQLite:
    def get_all_categories(self):
        return ["Clothing & Accessories", "Electronics"]


class MVPFeatureTests(unittest.TestCase):
    def setUp(self):
        self.agent = MVPCommerceAgent(TEST_CONFIG, MVPConfig())
        self.agent.sqlite = FakeSQLite()

    def test_live_text_search_product_normalization_prefers_image_field(self):
        product = {
            "title": "Spring Jacket",
            "image": "https://example.com/jacket.jpg",
            "image_urls": "amazon/amz_123.jpg",
            "brand": "Test Brand",
        }

        normalized = self.agent._normalize_product(product)

        self.assertEqual(normalized["image"], "https://example.com/jacket.jpg")
        self.assertEqual(normalized["image_urls"], "https://example.com/jacket.jpg")
        self.assertEqual(normalized["merchant"], "Test Brand")

    def test_live_text_search_product_normalization_resolves_relative_gcs_image(self):
        product = {
            "title": "Plaid Shacket",
            "image_urls": "amazon/amz_B08JGH3FQ7.jpg",
        }

        normalized = self.agent._normalize_product(product)

        self.assertEqual(
            normalized["image"],
            "https://storage.googleapis.com/web-agent-data-caleb-2026/amazon/amz_B08JGH3FQ7.jpg",
        )

    def test_text_query_sanitizer_drops_unknown_category_filter(self):
        query = DecomposedQuery(
            intent=IntentType.TEXT_SEARCH,
            original_query="find a jacket under 120",
            tags=["casual"],
            filters={"category": "jacket", "price_max": 120},
            rewritten_query="casual jacket under 120",
        )

        sanitized = self.agent._sanitize_query(query)

        self.assertEqual(sanitized.filters, {"price_max": 120})
        self.assertIn("jacket", sanitized.tags)

    def test_keyword_boosts_prioritize_matching_product_type(self):
        products = [
            {
                "product_id": "1",
                "title": "Casual Harrington Jacket",
                "subcategory": "Lightweight Jackets",
                "category": "Clothing & Accessories",
                "retrieval_score": 0.01,
            },
            {
                "product_id": "2",
                "title": "Baseball Cap",
                "subcategory": "Hats",
                "category": "Clothing & Accessories",
                "retrieval_score": 0.02,
            },
        ]

        ranked = self.agent._apply_keyword_type_boosts(
            "recommend me a casual spring jacket",
            products,
            limit=2,
        )

        self.assertEqual(ranked[0]["product_id"], "1")
        self.assertGreater(ranked[0]["ranking_score"], ranked[1]["ranking_score"])

    def test_general_talk_route_detected_without_backend(self):
        router = MVPRouter(api_key="")
        intent = router.detect_intent("hello there")
        self.assertEqual(intent, IntentType.GENERAL_TALK)

    def test_text_search_route_detected_without_backend(self):
        router = MVPRouter(api_key="")
        intent = router.detect_intent("recommend me some jeans under 80")
        self.assertEqual(intent, IntentType.TEXT_SEARCH)

    def test_image_search_route_detected_from_uploaded_image_flag(self):
        router = MVPRouter(api_key="")
        intent = router.detect_intent("find something like this", has_image=True)
        self.assertEqual(intent, IntentType.IMAGE_SEARCH)

    def test_web_result_cap_defaults_to_one(self):
        config = MVPConfig()
        self.assertEqual(config.web_num_results, 1)

    def test_pipeline_stage_serializes_to_sse_status_message(self):
        payload = json.loads(PipelineStage.to_sse(PipelineStage.SOURCING_WEB))
        self.assertEqual(payload["type"], "status")
        self.assertEqual(payload["stage"], "sourcing_web")


if __name__ == "__main__":
    unittest.main()
