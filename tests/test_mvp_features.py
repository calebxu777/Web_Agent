import json
import unittest
import asyncio
import types
from unittest.mock import patch

from mvp.agent import MVPCommerceAgent, MVPConfig, PipelineStage
from mvp.api import resolve_env_flag, resolve_mvp_act_mode
from mvp.preference_models import TurnAnalysisResult
from mvp.router import MVPRouter
from mvp.worksheet_store import InMemoryWorksheetStore
from src.agent_acts import RecommendAct
from src.master_brain import MasterBrain
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

    def analyze_turn(self, _message: str, has_image: bool = False, recent_context: dict | None = None) -> TurnAnalysisResult:
        return TurnAnalysisResult(
            intent=self.query.intent,
            tags=list(self.query.tags),
            filters=dict(self.query.filters),
            rewritten_query=self.query.rewritten_query,
        )

    def rerank(
        self,
        _query: str,
        products: list[dict],
        top_k: int = 10,
        preference_context: str = "",
    ) -> list[dict]:
        return products[:top_k]


class RecordingRouter(FakeRouter):
    def __init__(self, query: DecomposedQuery):
        super().__init__(query)
        self.last_top_k = None

    def rerank(
        self,
        _query: str,
        products: list[dict],
        top_k: int = 10,
        preference_context: str = "",
    ) -> list[dict]:
        self.last_top_k = top_k
        return super().rerank(_query, products, top_k=top_k, preference_context=preference_context)


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
    def __init__(self):
        self.last_query = None

    def search_text(self, _query):
        self.last_query = _query
        return [
            {
                "product_id": "p1",
                "title": "Budget Jacket",
                "price": 99.0,
                "brand": "Test Brand",
            }
        ]


class ConfigurableRetriever(FakeRetriever):
    def __init__(self, products: list[dict]):
        super().__init__()
        self.products = products

    def search_text(self, _query):
        self.last_query = _query
        return list(self.products)


class FollowupRouter(FakeRouter):
    def __init__(self, analyses: list[TurnAnalysisResult]):
        self.analyses = analyses
        self.query = None

    def analyze_turn(self, _message: str, has_image: bool = False, recent_context: dict | None = None) -> TurnAnalysisResult:
        if self.analyses:
            result = self.analyses.pop(0)
            self.query = result.to_decomposed_query(_message, default_intent=IntentType.TEXT_SEARCH)
            return result
        return TurnAnalysisResult(intent=IntentType.GENERAL_TALK)

    def decompose_query(self, _message: str) -> DecomposedQuery:
        return self.query or DecomposedQuery(
            intent=IntentType.TEXT_SEARCH,
            original_query=_message,
            rewritten_query=_message,
        )


class CapturingImageRouter(FakeRouter):
    def __init__(self):
        super().__init__(
            DecomposedQuery(
                intent=IntentType.IMAGE_SEARCH,
                original_query="find something similar to this",
                rewritten_query="",
            )
        )
        self.last_recent_context = None
        self.last_has_image = None

    def analyze_turn(self, _message: str, has_image: bool = False, recent_context: dict | None = None) -> TurnAnalysisResult:
        self.last_recent_context = recent_context
        self.last_has_image = has_image
        return TurnAnalysisResult(intent=IntentType.IMAGE_SEARCH)


class FakeImagePipeline:
    def __init__(self, products=None, caption="blue jeans", tags=None):
        self.products = list(products or [])
        self.caption = caption
        self.tags = list(tags or ["jeans"])

    def search(self, image_source, user_text="", filters=None):
        return {
            "products": list(self.products),
            "caption": self.caption,
            "tags": list(self.tags),
        }


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

    def test_live_text_search_product_normalization_uses_thumbnail_fallback(self):
        product = {
            "title": "Rain Shell",
            "thumbnail": "amazon/amz_B999999999.jpg",
        }

        normalized = self.agent._normalize_product(product)

        self.assertEqual(
            normalized["image"],
            "https://storage.googleapis.com/web-agent-data-caleb-2026/amazon/amz_B999999999.jpg",
        )

    def test_prioritize_products_with_images_moves_imageless_items_back(self):
        products = [
            {"product_id": "1", "title": "No Image Jacket"},
            {"product_id": "2", "title": "Image Jacket", "image_urls": "amazon/amz_B000000010.jpg"},
        ]

        ranked = self.agent._prioritize_products_with_images(products)

        self.assertEqual(ranked[0]["product_id"], "2")

    def test_generation_stage_uses_non_personalized_copy_for_anonymous_users(self):
        self.assertEqual(
            self.agent._generation_stage("anon_session-123", mode="recommendations"),
            PipelineStage.GENERATING_RECOMMENDATIONS,
        )
        self.assertEqual(
            self.agent._generation_stage("caleb", mode="recommendations"),
            PipelineStage.GENERATING_PERSONALIZED,
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

    def test_keyword_boosts_penalize_gender_mismatch_from_preference_context(self):
        products = [
            {
                "product_id": "1",
                "title": "Women's Maternity Hoodie",
                "subcategory": "Maternity Hoodies",
                "category": "Clothing & Accessories",
                "retrieval_score": 0.05,
            },
            {
                "product_id": "2",
                "title": "Men's Pullover Hoodie",
                "subcategory": "Active Hoodies",
                "category": "Clothing & Accessories",
                "retrieval_score": 0.01,
            },
        ]

        ranked = self.agent._apply_keyword_type_boosts(
            "recommend me some hoodies",
            products,
            limit=2,
            preference_context="User preference context:\n- gender: male",
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

    def test_active_search_followup_short_color_reuses_previous_product_type(self):
        agent = MVPCommerceAgent(TEST_CONFIG, MVPConfig(use_worksheets=True))
        agent.sqlite = FakeSQLite()
        agent.router = FollowupRouter(
            [
                TurnAnalysisResult(
                    intent=IntentType.TEXT_SEARCH,
                    tags=["t-shirt"],
                    filters={"category": "t-shirt"},
                    rewritten_query="t-shirt",
                ),
                TurnAnalysisResult(
                    intent=IntentType.GENERAL_TALK,
                    tags=[],
                    filters={},
                    rewritten_query="",
                ),
            ]
        )
        agent.retriever = FakeRetriever()
        agent.master_brain = FakeMasterBrain()
        agent._initialized = True

        async def collect(message: str):
            return [
                json.loads(event)
                async for event in agent.handle_message(
                    user_id="user-1",
                    session_id="session-followup",
                    message=message,
                    image_bytes=None,
                    web_search_enabled=False,
                )
            ]

        asyncio.run(collect("recommend some t-shirts"))
        events = asyncio.run(collect("blue"))

        product_events = [event for event in events if event.get("type") == "products"]
        self.assertTrue(product_events)
        self.assertEqual(agent.retriever.last_query.filters.get("color"), "blue")
        self.assertIn("shirt", agent.retriever.last_query.rewritten_query.lower())

    def test_followup_recommendation_question_uses_recent_results_compare(self):
        agent = MVPCommerceAgent(TEST_CONFIG, MVPConfig(use_worksheets=True))
        search_definition = agent.worksheet_registry.get("product_search")
        search_instance = agent.worksheet_engine.create_instance(search_definition)
        search_instance.result_refs["last_products"] = [
            {"product_id": "p1", "title": "First Jacket", "price": 100.0},
            {"product_id": "p2", "title": "Second Jacket", "price": 120.0},
            {"product_id": "p3", "title": "Third Jacket", "price": 150.0},
        ]
        agent.worksheet_store.save("session-followup-compare", search_instance)

        self.assertTrue(
            agent._should_handle_as_compare(
                "which one do you recommend based on reviews",
                search_instance,
            )
        )

        compare_definition, compare_instance = agent._prepare_compare_worksheet(
            "session-followup-compare",
            "which one do you recommend based on reviews",
        )

        self.assertEqual(compare_definition.name, "compare_products")
        self.assertEqual(len(compare_instance.result_refs["comparison_products"]), 3)
        self.assertIn("reviews", compare_instance.values["comparison_dimensions"])

    def test_image_turn_does_not_inherit_active_search_filters(self):
        agent = MVPCommerceAgent(TEST_CONFIG, MVPConfig(use_worksheets=True))
        search_definition = agent.worksheet_registry.get("product_search")
        search_instance = agent.worksheet_engine.create_instance(search_definition)
        search_instance.values.update(
            {
                "product_type": "shirt",
                "color": "blue",
                "rewritten_query": "blue shirts",
            }
        )
        agent.worksheet_store.save("session-image-isolation", search_instance)

        agent.router = CapturingImageRouter()
        agent._initialized = True

        async def fake_image_workflow(self, user_id, session_id, message, image_bytes, include_web=False, analyzed_query=None):
            yield json.dumps({"type": "products", "items": [], "tags": [], "caption": ""})

        agent._workflow_image_search = types.MethodType(fake_image_workflow, agent)

        async def collect_events():
            return [
                json.loads(event)
                async for event in agent.handle_message(
                    user_id="user-1",
                    session_id="session-image-isolation",
                    message="find something similar to this",
                    image_bytes=b"fake-image",
                    web_search_enabled=False,
                )
            ]

        events = asyncio.run(collect_events())

        self.assertTrue(events)
        self.assertTrue(agent.router.last_has_image)
        self.assertIsInstance(agent.router.last_recent_context, dict)
        self.assertNotIn("active_product_search", agent.router.last_recent_context)

    def test_image_search_returns_no_match_fallback_when_no_products_found(self):
        agent = MVPCommerceAgent(TEST_CONFIG, MVPConfig(use_florence=True, use_memory=False))
        agent.sqlite = FakeSQLite()
        agent.image_pipeline = FakeImagePipeline(products=[], caption="blue jeans", tags=["jeans"])
        agent.router = FakeRouter(
            DecomposedQuery(
                intent=IntentType.IMAGE_SEARCH,
                original_query="recommend me some jeans like this",
                rewritten_query="recommend me some jeans like this",
            )
        )

        async def collect_events():
            return [
                json.loads(event)
                async for event in agent._workflow_image_search(
                    user_id="user-1",
                    session_id="session-image-empty",
                    message="recommend me some jeans like this",
                    image_bytes=b"fake-image",
                    include_web=False,
                )
            ]

        events = asyncio.run(collect_events())
        product_events = [event for event in events if event.get("type") == "products"]
        token_events = [event for event in events if event.get("type") == "token"]

        self.assertTrue(product_events)
        self.assertEqual(product_events[-1]["items"], [])
        text = "".join(event["content"] for event in token_events).lower()
        self.assertIn("couldn't find a strong visual match", text)
        self.assertIn("turning on web search", text)

    def test_image_search_returns_final_apology_when_web_enabled_and_no_products_found(self):
        agent = MVPCommerceAgent(TEST_CONFIG, MVPConfig(use_florence=True, use_memory=False))
        agent.sqlite = FakeSQLite()
        agent.image_pipeline = FakeImagePipeline(products=[], caption="blue jeans", tags=["jeans"])
        agent.router = FakeRouter(
            DecomposedQuery(
                intent=IntentType.IMAGE_SEARCH,
                original_query="recommend me some jeans like this",
                rewritten_query="recommend me some jeans like this",
            )
        )

        async def collect_events():
            return [
                json.loads(event)
                async for event in agent._workflow_image_search(
                    user_id="user-1",
                    session_id="session-image-empty-web",
                    message="recommend me some jeans like this",
                    image_bytes=b"fake-image",
                    include_web=True,
                )
            ]

        events = asyncio.run(collect_events())
        token_events = [event for event in events if event.get("type") == "token"]

        self.assertIn("sorry", "".join(event["content"] for event in token_events).lower())
        self.assertIn("web search", "".join(event["content"] for event in token_events).lower())

    def test_text_search_returns_web_search_suggestion_when_no_catalog_results(self):
        agent = MVPCommerceAgent(TEST_CONFIG, MVPConfig(use_worksheets=False, use_memory=False))
        agent.sqlite = FakeSQLite()
        agent.router = FakeRouter(
            DecomposedQuery(
                intent=IntentType.TEXT_SEARCH,
                original_query="recommend some niche jackets",
                tags=["jacket"],
                filters={},
                rewritten_query="niche jackets",
            )
        )
        agent.retriever = ConfigurableRetriever([])
        agent.master_brain = FakeMasterBrain()

        async def collect_events():
            return [
                json.loads(event)
                async for event in agent._workflow_text_search(
                    user_id="user-1",
                    session_id="session-text-empty",
                    message="recommend some niche jackets",
                    include_web=False,
                )
            ]

        events = asyncio.run(collect_events())
        token_events = [event for event in events if event.get("type") == "token"]

        self.assertIn("turning on web search", "".join(event["content"] for event in token_events).lower())

    def test_text_search_returns_final_apology_when_web_enabled_and_no_results_found(self):
        agent = MVPCommerceAgent(TEST_CONFIG, MVPConfig(use_worksheets=False, use_memory=False))
        agent.sqlite = FakeSQLite()
        agent.router = FakeRouter(
            DecomposedQuery(
                intent=IntentType.TEXT_SEARCH,
                original_query="recommend some niche jackets",
                tags=["jacket"],
                filters={},
                rewritten_query="niche jackets",
            )
        )
        agent.retriever = ConfigurableRetriever([])
        agent.master_brain = FakeMasterBrain()

        async def collect_events():
            return [
                json.loads(event)
                async for event in agent._workflow_text_search(
                    user_id="user-1",
                    session_id="session-text-empty-web",
                    message="recommend some niche jackets",
                    include_web=True,
                )
            ]

        events = asyncio.run(collect_events())
        token_events = [event for event in events if event.get("type") == "token"]
        text = "".join(event["content"] for event in token_events).lower()

        self.assertIn("sorry", text)
        self.assertIn("web search", text)

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

    def test_text_search_anonymous_status_is_not_personalized(self):
        agent = MVPCommerceAgent(TEST_CONFIG, MVPConfig(use_worksheets=False))
        agent.sqlite = FakeSQLite()
        agent.router = FakeRouter(
            DecomposedQuery(
                intent=IntentType.TEXT_SEARCH,
                original_query="recommend some jackets",
                tags=["jacket"],
                filters={},
                rewritten_query="jackets",
            )
        )
        agent.retriever = FakeRetriever()
        agent.master_brain = FakeMasterBrain()

        async def collect_events():
            return [
                json.loads(event)
                async for event in agent._workflow_text_search(
                    user_id="anon_session-xyz",
                    session_id="session-anon-status",
                    message="recommend some jackets",
                    include_web=False,
                )
            ]

        events = asyncio.run(collect_events())
        generating_events = [
            event for event in events if event.get("type") == "status" and event.get("stage") == "generating"
        ]

        self.assertTrue(generating_events)
        self.assertEqual(generating_events[-1]["message"], "Generating recommendations...")

    def test_text_search_prefers_image_bearing_products_from_larger_rerank_window(self):
        agent = MVPCommerceAgent(
            TEST_CONFIG,
            MVPConfig(use_worksheets=False, top_k_final=2, top_k_reranked=6),
        )
        agent.sqlite = FakeSQLite()
        agent.router = RecordingRouter(
            DecomposedQuery(
                intent=IntentType.TEXT_SEARCH,
                original_query="recommend some t-shirts",
                tags=["t-shirt"],
                filters={},
                rewritten_query="t-shirts",
            )
        )
        agent.retriever = ConfigurableRetriever(
            [
                {"product_id": "p1", "title": "Plain Tee 1"},
                {"product_id": "p2", "title": "Plain Tee 2"},
                {"product_id": "p3", "title": "Plain Tee 3"},
                {"product_id": "p4", "title": "Plain Tee 4"},
                {"product_id": "p5", "title": "Plain Tee 5"},
                {
                    "product_id": "p6",
                    "title": "Blue Tee With Image",
                    "image_urls": "amazon/amz_B000000111.jpg",
                },
            ]
        )
        agent.master_brain = FakeMasterBrain()

        async def collect_events():
            return [
                json.loads(event)
                async for event in agent._workflow_text_search(
                    user_id="anon_session-images",
                    session_id="session-image-window",
                    message="recommend some t-shirts",
                    include_web=False,
                )
            ]

        events = asyncio.run(collect_events())
        product_events = [event for event in events if event.get("type") == "products"]

        self.assertTrue(product_events)
        self.assertEqual(agent.router.last_top_k, 6)
        self.assertEqual(product_events[-1]["items"][0]["product_id"], "p6")

    def test_recommend_act_acknowledges_total_surface_count_when_subset_is_recommended(self):
        prompt = RecommendAct(
            ranked_products=[
                {"product_id": "p1", "title": "Item 1"},
                {"product_id": "p2", "title": "Item 2"},
                {"product_id": "p3", "title": "Item 3"},
                {"product_id": "p4", "title": "Item 4"},
                {"product_id": "p5", "title": "Item 5"},
            ],
            user_query="recommend some hoodies",
            max_recommendations=2,
            total_candidates=5,
        ).to_prompt_block()

        self.assertIn("Total surfaced matches in [REPORT]: 5", prompt)
        self.assertIn("I found 5 strong matches", prompt)

    def test_master_brain_prompt_mentions_total_surfaced_matches(self):
        brain = MasterBrain(api_base="http://localhost:1", model_name="test-model")
        try:
            messages = brain._build_messages(
                user_query="recommend some hoodies",
                products=[
                    {"product_id": "p1", "title": "Item 1"},
                    {"product_id": "p2", "title": "Item 2"},
                    {"product_id": "p3", "title": "Item 3"},
                    {"product_id": "p4", "title": "Item 4"},
                    {"product_id": "p5", "title": "Item 5"},
                ],
                chat_history=[],
            )
        finally:
            brain.close()

        self.assertIn("Available Products (5 surfaced matches):", messages[-1]["content"])
        self.assertIn("I found 5 strong matches", messages[-1]["content"])

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
