import asyncio
import json
import unittest
from pathlib import Path
from uuid import uuid4
from unittest.mock import patch

from mvp.agent import MVPCommerceAgent, MVPConfig
from mvp.preference_models import (
    PreferenceItem,
    StoredPreferenceProfile,
    TurnAnalysisResult,
)
from mvp.preference_store import (
    InMemorySessionPreferenceStore,
    PreferenceStore,
    SQLitePreferenceProfileStore,
)
from mvp.router import MVPRouter
from src.schema import ChatMessage
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


class FakeRetriever:
    def search_text(self, _query):
        return [
            {
                "product_id": "p1",
                "title": "Everyday Tee",
                "price": 35.0,
                "brand": "Test Brand",
            }
        ]


class FakeMasterBrain:
    async def synthesize_stream(
        self,
        query,
        _products,
        _chat_history=None,
        _memory_context="",
    ):
        for token in ["Here is ", "a result."]:
            yield token


class AnalysisRouter:
    def __init__(self, analyses: dict[str, TurnAnalysisResult] | None = None):
        self.analyses = analyses or {}
        self.last_preference_context = ""

    def analyze_turn(
        self,
        message: str,
        has_image: bool = False,
        recent_context: dict | None = None,
    ) -> TurnAnalysisResult:
        if has_image:
            return self.analyses.get(
                message,
                TurnAnalysisResult(intent=IntentType.IMAGE_SEARCH),
            )
        return self.analyses.get(
            message,
            TurnAnalysisResult(
                intent=IntentType.TEXT_SEARCH,
                rewritten_query=message,
            ),
        )

    def decompose_query(self, message: str) -> DecomposedQuery:
        return self.analyze_turn(message).to_decomposed_query(message)

    def rerank(
        self,
        _query: str,
        products: list[dict],
        top_k: int = 10,
        preference_context: str = "",
    ) -> list[dict]:
        self.last_preference_context = preference_context
        return products[:top_k]


class FakeGCSSyncClient:
    def __init__(self):
        self.uploaded_files: list[tuple[str, str, str]] = []
        self.appended_jsonl_records: list[tuple[str, dict]] = []

    def upload_file(self, local_path, blob_name: str, content_type: str = "application/octet-stream") -> str:
        self.uploaded_files.append((str(local_path), blob_name, content_type))
        return f"gs://test-bucket/{blob_name}"

    def append_jsonl_record(self, blob_name: str, record: dict) -> str:
        self.appended_jsonl_records.append((blob_name, record))
        return f"gs://test-bucket/{blob_name}"


class MVPPreferenceTests(unittest.TestCase):
    def setUp(self):
        processed_dir = Path("data/processed")
        processed_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = str((processed_dir / f"test_user_preferences_{uuid4().hex}.db").resolve())
        runtime_dir = Path("data/test_runtime")
        runtime_dir.mkdir(parents=True, exist_ok=True)
        self.evaluation_path = str((runtime_dir / f"conversation_recordings_{uuid4().hex}.jsonl").resolve())
        self._stores_to_close: list[PreferenceStore] = []

    def tearDown(self):
        for store in self._stores_to_close:
            store.close()
        for suffix in ("", "-shm", "-wal", "-journal"):
            path = Path(f"{self.db_path}{suffix}")
            if path.exists():
                path.unlink()
        eval_path = Path(self.evaluation_path)
        if eval_path.exists():
            eval_path.unlink()

    def build_store(self) -> PreferenceStore:
        store = PreferenceStore(
            session_store=InMemorySessionPreferenceStore(),
            durable_store=SQLitePreferenceProfileStore(self.db_path),
        )
        self._stores_to_close.append(store)
        return store

    def test_turn_analysis_parses_preferences(self):
        router = MVPRouter(api_key="")
        raw = json.dumps(
            {
                "intent": "text_search",
                "tags": ["casual"],
                "filters": {"price_max": 120, "color": "Red"},
                "rewritten_query": "red casual shirt under 120",
                "preferences": [
                    {
                        "kind": "color",
                        "value": " Red ",
                        "polarity": "positive",
                        "confidence": 0.91,
                        "source_text": "I love red",
                    },
                    {
                        "kind": "budget_max",
                        "value": "$120",
                        "polarity": "positive",
                        "confidence": 0.84,
                        "source_text": "Keep it under $120",
                    },
                    {
                        "kind": "gender",
                        "value": " Male ",
                        "polarity": "positive",
                        "confidence": 0.88,
                        "source_text": "I am male",
                    },
                ],
            }
        )
        with patch.object(router, "_chat_completion", return_value=raw):
            result = router.analyze_turn("I love red. Keep it under $120.")

        extracted = {(item.kind, str(item.value)): item for item in result.preferences}
        self.assertEqual(result.intent, IntentType.TEXT_SEARCH)
        self.assertEqual(result.filters["price_max"], 120)
        self.assertIn(("color", "red"), extracted)
        self.assertIn(("budget_max", "120"), extracted)
        self.assertIn(("gender", "male"), extracted)

    def test_session_preference_merge_is_deterministic(self):
        store = self.build_store()

        store.update_session_preferences(
            user_id="caleb",
            session_id="session-1",
            preferences=[
                PreferenceItem(kind="color", value="Red", confidence=0.92, source_text="I like red"),
                PreferenceItem(kind="brand", value="Nike", confidence=0.88, source_text="Nike works for me"),
            ],
        )
        profile = store.update_session_preferences(
            user_id="caleb",
            session_id="session-1",
            preferences=[
                PreferenceItem(kind="color", value="red", confidence=0.70, source_text="still red"),
                PreferenceItem(kind="budget_max", value=95, confidence=0.81, source_text="under 95"),
            ],
        )

        self.assertEqual(
            profile.preferences,
            {
                "color": ["red"],
                "brand": ["nike"],
                "budget_max": 95,
            },
        )
        self.assertEqual(len(profile.evidence), 4)

    def test_sqlite_persistence_merges_session_and_durable_preferences(self):
        store = self.build_store()

        store.durable_store.save(
            StoredPreferenceProfile(
                user_id="caleb",
                preferences={"brand": ["nike"]},
                updated_at=1.0,
                source_session_id="session-old",
            )
        )
        store.update_session_preferences(
            user_id="caleb",
            session_id="session-new",
            preferences=[
                PreferenceItem(kind="color", value="red", confidence=0.9, source_text="I like red"),
                PreferenceItem(kind="budget_max", value=120, confidence=0.8, source_text="under 120"),
            ],
        )

        merged = store.finalize_session("caleb", "session-new")
        reloaded = store.get_stored_profile("caleb")

        self.assertIsNotNone(merged)
        self.assertEqual(
            merged.preferences,
            {
                "brand": ["nike"],
                "color": ["red"],
                "budget_max": 120,
            },
        )
        self.assertIsNone(store.get_session_profile("session-new"))
        self.assertEqual(reloaded.preferences, merged.preferences)

    def test_rerank_prompt_includes_preferences_when_available(self):
        router = MVPRouter(api_key="")
        prompt = router.build_rerank_prompt(
            query="show me black shirts",
            products=[{"title": "Black Tee", "price": 40, "brand": "Test Brand"}],
            top_k=3,
            preference_context="User preference context:\n- colors: red",
        )

        self.assertIn("User preference context", prompt)
        self.assertIn("follow the current query", prompt)
        self.assertIn("weak tie-breaker", prompt)
        self.assertIn("Never use saved preferences to change the requested product type", prompt)

    def test_rerank_prompt_omits_preference_section_when_absent(self):
        router = MVPRouter(api_key="")
        prompt = router.build_rerank_prompt(
            query="show me black shirts",
            products=[{"title": "Black Tee", "price": 40, "brand": "Test Brand"}],
            top_k=3,
        )

        self.assertNotIn("User preference context", prompt)

    def test_next_turn_rerank_receives_session_preferences(self):
        agent = MVPCommerceAgent(
            TEST_CONFIG,
            MVPConfig(
                use_preference_inference=True,
                use_preference_reranking=True,
                user_preferences_db_path=self.db_path,
            ),
        )
        if agent.preference_store:
            agent.preference_store.close()
        agent.router = AnalysisRouter(
            analyses={
                "I like red": TurnAnalysisResult(
                    intent=IntentType.TEXT_SEARCH,
                    rewritten_query="I like red",
                    preferences=[
                        PreferenceItem(
                            kind="color",
                            value="red",
                            confidence=0.95,
                            source_text="I like red",
                        )
                    ],
                ),
                "recommend me some t-shirts": TurnAnalysisResult(
                    intent=IntentType.TEXT_SEARCH,
                    rewritten_query="recommend me some t-shirts",
                    preferences=[],
                ),
            }
        )
        agent.retriever = FakeRetriever()
        agent.master_brain = FakeMasterBrain()
        agent.preference_store = self.build_store()
        agent._initialized = True

        async def run_turn(message: str):
            return [
                event
                async for event in agent.handle_message(
                    user_id="caleb",
                    session_id="session-1",
                    message=message,
                    web_search_enabled=False,
                )
            ]

        asyncio.run(run_turn("I like red"))
        asyncio.run(run_turn("recommend me some t-shirts"))

        self.assertIn("- colors: red", agent.router.last_preference_context)

    def test_partial_search_turn_still_stores_preferences_from_turn_analysis(self):
        agent = MVPCommerceAgent(
            TEST_CONFIG,
            MVPConfig(
                use_worksheets=True,
                emit_worksheet_events=True,
                use_preference_inference=True,
                user_preferences_db_path=self.db_path,
            ),
        )
        if agent.preference_store:
            agent.preference_store.close()
        agent.preference_store = self.build_store()
        agent.router = AnalysisRouter(
            analyses={
                "I like red": TurnAnalysisResult(
                    intent=IntentType.TEXT_SEARCH,
                    filters={"color": "red"},
                    rewritten_query="red",
                    preferences=[
                        PreferenceItem(
                            kind="color",
                            value="red",
                            confidence=0.95,
                            source_text="I like red",
                        )
                    ],
                )
            }
        )
        agent.retriever = FakeRetriever()
        agent.master_brain = FakeMasterBrain()
        agent._initialized = True

        async def collect_events():
            return [
                json.loads(event)
                async for event in agent.handle_message(
                    user_id="caleb",
                    session_id="session-clarify",
                    message="I like red",
                    web_search_enabled=False,
                )
            ]

        events = asyncio.run(collect_events())
        token_events = [event for event in events if event.get("type") == "token"]
        product_events = [event for event in events if event.get("type") == "products"]
        worksheet_events = [event for event in events if event.get("type") == "worksheet_state"]
        session_profile = agent.preference_store.get_session_profile("session-clarify")

        self.assertTrue(token_events)
        self.assertTrue(product_events)
        self.assertTrue(worksheet_events)
        self.assertEqual(worksheet_events[-1]["worksheet"]["status"], "active")
        self.assertEqual("".join(event["content"] for event in token_events), "Here is a result.")
        self.assertEqual(session_profile.preferences, {"color": ["red"]})

    def test_agent_finalize_session_persists_preferences(self):
        agent = MVPCommerceAgent(
            TEST_CONFIG,
            MVPConfig(
                use_preference_inference=True,
                use_preference_reranking=True,
                user_preferences_db_path=self.db_path,
                local_evaluation_recordings_path=self.evaluation_path,
            ),
        )
        if agent.preference_store:
            agent.preference_store.close()
        agent.preference_store = self.build_store()
        agent.gcs_sync = FakeGCSSyncClient()
        agent.conversation_recording_store.add_message(
            "session-42",
            ChatMessage(role="user", content="show me minimalist looks", timestamp=1.0),
        )
        agent.conversation_recording_store.add_message(
            "session-42",
            ChatMessage(role="assistant", content="Here are some minimalist options.", timestamp=2.0),
        )
        agent.preference_store.update_session_preferences(
            user_id="caleb",
            session_id="session-42",
            preferences=[
                PreferenceItem(kind="style", value="minimalist", confidence=0.9, source_text="I like minimalist looks")
            ],
        )

        result = asyncio.run(agent.finalize_session("caleb", "session-42"))

        self.assertEqual(result["status"], "finalized")
        self.assertEqual(result["preferences"], {"style": ["minimalist"]})
        self.assertIsNone(agent.preference_store.get_session_profile("session-42"))
        self.assertEqual(result["evaluation_gcs_uri"], "gs://test-bucket/evaluations/recording.jsonl")
        self.assertIsNone(result["evaluation_gcs_error"])
        self.assertEqual(result["preference_db_gcs_uri"], "gs://test-bucket/preference/user_preferences.db")
        self.assertIsNone(result["preference_db_gcs_error"])

        local_recording = Path(self.evaluation_path).read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(local_recording), 1)
        payload = json.loads(local_recording[0])
        self.assertEqual(payload["type"], "mvp")
        self.assertEqual(payload["inferred_preferences"], {"style": ["minimalist"]})
        self.assertEqual(payload["conversation"][0]["user"], "show me minimalist looks")
        self.assertEqual(payload["conversation"][1]["agent"], "Here are some minimalist options.")


if __name__ == "__main__":
    unittest.main()
