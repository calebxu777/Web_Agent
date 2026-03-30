import json
import unittest
import asyncio
from unittest.mock import patch

from mvp.agent import MVPCommerceAgent, MVPConfig, PipelineStage
from mvp.api import resolve_env_flag, resolve_mvp_act_mode
from mvp.router import MVPRouter
from mvp.worksheet_store import InMemoryWorksheetStore
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


class FakeRouter:
    def __init__(self, query: DecomposedQuery):
        self.query = query

    def decompose_query(self, _message: str) -> DecomposedQuery:
        return self.query

    def rerank(
        self,
        _query: str,
        products: list[dict],
        top_k: int = 10,
        preference_context: str = "",
    ) -> list[dict]:
        return products[:top_k]


class FakeMasterBrain:
    async def grounded_synthesize_stream(
        self,
        _query,
        _acts,
        _chat_history=None,
        _memory_context="",
        extra_instructions="",
    ):
        for token in ["Here is ", "the comparison."]:
            yield token

    async def synthesize_stream(
        self,
        _query,
        _products,
        _chat_history=None,
        _memory_context="",
        extra_instructions="",
    ):
        for token in ["Here is ", "a result."]:
            yield token


class FakeRetriever:
    def search_text(self, _query):
        return [
            {
                "product_id": "p1",
                "title": "Budget Jacket",
                "price": 99.0,
                "brand": "Test Brand",
            }
        ]


class MVPFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._worksheet_store_patch = patch(
            "mvp.agent.build_worksheet_store",
            side_effect=lambda _config: InMemoryWorksheetStore(),
        )
        cls._worksheet_store_patch.start()

    @classmethod
    def tearDownClass(cls):
        cls._worksheet_store_patch.stop()

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

    def test_live_text_search_product_normalization_uses_image_url_fallback(self):
        product = {
            "title": "Everyday Tee",
            "image_url": "amazon/amz_B012345678.jpg",
        }

        normalized = self.agent._normalize_product(product)

        self.assertEqual(
            normalized["image"],
            "https://storage.googleapis.com/web-agent-data-caleb-2026/amazon/amz_B012345678.jpg",
        )

    def test_live_text_search_product_normalization_reads_json_image_list(self):
        product = {
            "title": "Graphic Tee",
            "image_urls": "[\"amazon/amz_B000000001.jpg\", \"amazon/amz_B000000002.jpg\"]",
        }

        normalized = self.agent._normalize_product(product)

        self.assertEqual(
            normalized["image"],
            "https://storage.googleapis.com/web-agent-data-caleb-2026/amazon/amz_B000000001.jpg",
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

    def test_keyword_boosts_penalize_other_product_types(self):
        products = [
            {
                "product_id": "1",
                "title": "Red Lightweight Jacket",
                "subcategory": "Lightweight Jackets",
                "category": "Clothing & Accessories",
                "retrieval_score": 0.05,
            },
            {
                "product_id": "2",
                "title": "Black Pullover Hoodie",
                "subcategory": "Active Hoodies",
                "category": "Clothing & Accessories",
                "retrieval_score": 0.01,
            },
        ]

        ranked = self.agent._apply_keyword_type_boosts(
            "recommend me hoodies",
            products,
            limit=2,
        )

        self.assertEqual(ranked[0]["product_id"], "2")

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

    def test_resolve_env_flag_accepts_common_truthy_values(self):
        self.assertTrue(resolve_env_flag("true"))
        self.assertTrue(resolve_env_flag("YES"))
        self.assertFalse(resolve_env_flag("false"))
        self.assertFalse(resolve_env_flag(None))

    def test_resolve_mvp_act_mode_accepts_known_values(self):
        self.assertEqual(resolve_mvp_act_mode("dynamic", use_agent_acts=True), "dynamic")
        self.assertEqual(resolve_mvp_act_mode("hardcoded", use_agent_acts=True), "hardcoded")

    def test_resolve_mvp_act_mode_falls_back_based_on_switch(self):
        self.assertEqual(resolve_mvp_act_mode("something-else", use_agent_acts=False), "off")
        self.assertEqual(resolve_mvp_act_mode(None, use_agent_acts=False), "off")
        self.assertEqual(resolve_mvp_act_mode(None, use_agent_acts=True), "dynamic")
        self.assertEqual(resolve_mvp_act_mode("off", use_agent_acts=True), "dynamic")

    def test_pipeline_stage_serializes_to_sse_status_message(self):
        payload = json.loads(PipelineStage.to_sse(PipelineStage.SOURCING_WEB))
        self.assertEqual(payload["type"], "status")
        self.assertEqual(payload["stage"], "sourcing_web")

    def test_text_search_worksheet_allows_partial_query_without_product_type(self):
        agent = MVPCommerceAgent(TEST_CONFIG, MVPConfig(use_worksheets=True, emit_worksheet_events=True))
        agent.sqlite = FakeSQLite()
        agent.router = FakeRouter(
            DecomposedQuery(
                intent=IntentType.TEXT_SEARCH,
                original_query="show me something cheaper",
                tags=["cheaper"],
                filters={"price_max": 100},
                rewritten_query="something cheaper under 100",
            )
        )
        agent.retriever = FakeRetriever()
        agent.master_brain = FakeMasterBrain()

        async def collect_events():
            return [
                json.loads(event)
                async for event in agent._workflow_text_search(
                    user_id="user-1",
                    session_id="session-1",
                    message="show me something cheaper",
                    include_web=False,
                )
            ]

        events = asyncio.run(collect_events())

        token_events = [event for event in events if event.get("type") == "token"]
        worksheet_events = [event for event in events if event.get("type") == "worksheet_state"]
        product_events = [event for event in events if event.get("type") == "products"]

        self.assertTrue(worksheet_events)
        self.assertEqual(worksheet_events[-1]["worksheet"]["status"], "active")
        self.assertEqual(worksheet_events[-1]["worksheet"]["missing_required_fields"], [])
        self.assertTrue(product_events)
        self.assertEqual("".join(event["content"] for event in token_events), "Here is a result.")

    def test_text_search_worksheet_does_not_emit_frontend_events_by_default(self):
        agent = MVPCommerceAgent(TEST_CONFIG, MVPConfig(use_worksheets=True))
        agent.sqlite = FakeSQLite()
        agent.router = FakeRouter(
            DecomposedQuery(
                intent=IntentType.TEXT_SEARCH,
                original_query="show me something cheaper",
                tags=["cheaper"],
                filters={"price_max": 100},
                rewritten_query="something cheaper under 100",
            )
        )
        agent.retriever = FakeRetriever()
        agent.master_brain = FakeMasterBrain()

        async def collect_events():
            return [
                json.loads(event)
                async for event in agent._workflow_text_search(
                    user_id="user-1",
                    session_id="session-hidden-worksheet",
                    message="show me something cheaper",
                    include_web=False,
                )
            ]

        events = asyncio.run(collect_events())
        worksheet_events = [event for event in events if event.get("type") == "worksheet_state"]
        product_events = [event for event in events if event.get("type") == "products"]

        self.assertFalse(worksheet_events)
        self.assertTrue(product_events)

    def test_compare_workflow_uses_last_search_results(self):
        agent = MVPCommerceAgent(
            TEST_CONFIG,
            MVPConfig(use_worksheets=True, emit_worksheet_events=True, use_agent_acts=True, act_mode="dynamic"),
        )
        search_definition = agent.worksheet_registry.get("product_search")
        search_instance = agent.worksheet_engine.create_instance(search_definition)
        search_instance.result_refs["last_products"] = [
            {"product_id": "p1", "title": "First Jacket", "price": 100.0},
            {"product_id": "p2", "title": "Second Jacket", "price": 120.0},
            {"product_id": "p3", "title": "Third Jacket", "price": 150.0},
        ]
        agent.worksheet_store.save("session-compare", search_instance)
        agent.master_brain = FakeMasterBrain()

        async def collect_events():
            return [
                json.loads(event)
                async for event in agent._workflow_compare(
                    user_id="user-1",
                    session_id="session-compare",
                    message="compare 1 and 2",
                )
            ]

        events = asyncio.run(collect_events())

        worksheet_events = [event for event in events if event.get("type") == "worksheet_state"]
        product_events = [event for event in events if event.get("type") == "products"]
        token_events = [event for event in events if event.get("type") == "token"]

        self.assertTrue(worksheet_events)
        self.assertEqual(worksheet_events[-1]["worksheet"]["name"], "compare_products")
        self.assertTrue(product_events)
        self.assertEqual(len(product_events[-1]["items"]), 2)
        self.assertEqual("".join(event["content"] for event in token_events), "Here is the comparison.")

    def test_text_search_without_worksheets_skips_clarification(self):
        agent = MVPCommerceAgent(TEST_CONFIG, MVPConfig(use_worksheets=False))
        agent.sqlite = FakeSQLite()
        agent.router = FakeRouter(
            DecomposedQuery(
                intent=IntentType.TEXT_SEARCH,
                original_query="show me something cheaper",
                tags=["cheaper"],
                filters={"price_max": 100},
                rewritten_query="something cheaper under 100",
            )
        )
        agent.retriever = FakeRetriever()
        agent.master_brain = FakeMasterBrain()

        async def collect_events():
            return [
                json.loads(event)
                async for event in agent._workflow_text_search(
                    user_id="user-1",
                    session_id="session-no-worksheet",
                    message="show me something cheaper",
                    include_web=False,
                )
            ]

        events = asyncio.run(collect_events())

        worksheet_events = [event for event in events if event.get("type") == "worksheet_state"]
        product_events = [event for event in events if event.get("type") == "products"]
        token_events = [event for event in events if event.get("type") == "token"]

        self.assertFalse(worksheet_events)
        self.assertTrue(product_events)
        self.assertEqual("".join(event["content"] for event in token_events), "Here is a result.")

    def test_compare_workflow_without_agent_acts_uses_plain_synthesis(self):
        agent = MVPCommerceAgent(
            TEST_CONFIG,
            MVPConfig(use_worksheets=True, emit_worksheet_events=True, use_agent_acts=False, act_mode="off"),
        )
        search_definition = agent.worksheet_registry.get("product_search")
        search_instance = agent.worksheet_engine.create_instance(search_definition)
        search_instance.result_refs["last_products"] = [
            {"product_id": "p1", "title": "First Jacket", "price": 100.0},
            {"product_id": "p2", "title": "Second Jacket", "price": 120.0},
        ]
        agent.worksheet_store.save("session-plain-compare", search_instance)
        agent.master_brain = FakeMasterBrain()

        async def collect_events():
            return [
                json.loads(event)
                async for event in agent._workflow_compare(
                    user_id="user-1",
                    session_id="session-plain-compare",
                    message="compare 1 and 2",
                )
            ]

        events = asyncio.run(collect_events())
        token_events = [event for event in events if event.get("type") == "token"]

        self.assertEqual("".join(event["content"] for event in token_events), "Here is a result.")


if __name__ == "__main__":
    unittest.main()
