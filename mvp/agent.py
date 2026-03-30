"""
agent.py - MVP agent orchestrator
================================
Separate MVP workflow that keeps the current backend intact while using an
API LLM for routing/reranking and Master Brain generation.
"""

from __future__ import annotations

import base64
import json
import re
import tempfile
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Optional
from urllib.parse import quote, urlparse

import httpx

from mvp.evaluation.record_for_evaluation import (
    InMemoryConversationRecordingStore,
    append_record_to_jsonl,
    build_conversation_record,
)
from mvp.gcs_sync import MVPGCSSyncClient
from mvp.preference_models import (
    TurnAnalysisResult,
    build_preference_context,
)
from mvp.preference_store import (
    PreferenceStore,
    SQLitePreferenceProfileStore,
    build_session_preference_store,
)
from mvp.router import MVPRouter
from mvp.worksheet_engine import WorksheetEngine
from mvp.worksheet_models import WorksheetDefinition, WorksheetInstance
from mvp.worksheet_registry import WorksheetRegistry
from mvp.worksheet_router import WorksheetRouter
from mvp.worksheet_store import InMemoryWorksheetStore, build_worksheet_store
from src.master_brain import MasterBrain
from src.schema import ChatMessage, DecomposedQuery, IntentType


@dataclass
class MVPConfig:
    master_brain_model_name: str = "gpt-4o-mini"
    router_model_name: str = "gpt-4o-mini"
    reranker_model_name: str = "gpt-4o-mini"
    api_base: str = "https://api.openai.com/v1"
    api_key: str = ""

    use_florence: bool = False
    use_web_search: bool = True
    use_visual_verifier: bool = False
    verifier_threshold: float = 0.5

    serpapi_api_key: str = ""
    serpapi_api_base: str = "https://serpapi.com"
    serpapi_location: str = ""
    serpapi_gl: str = "us"
    serpapi_hl: str = "en"
    serpapi_mock_results_path: str = ""
    web_num_results: int = 1

    top_k_initial: int = 50
    top_k_reranked: int = 10
    top_k_final: int = 5
    rrf_k: int = 60

    use_memory: bool = True
    use_preference_inference: bool = False
    use_preference_reranking: bool = False
    preference_redis_ttl_seconds: int = 3600
    user_preferences_db_path: str = "data/processed/user_preferences.db"
    local_evaluation_recordings_path: str = "data/evaluation/conversation_recordings.jsonl"
    image_storage_provider: str = "gcs"
    gcs_public_url: str = "https://storage.googleapis.com/web-agent-data-caleb-2026"
    gcs_bucket_name: str = "web-agent-data-caleb-2026"
    gcs_project_id: str = "webagent2026"
    gcs_preferences_blob_path: str = "preference/user_preferences.db"
    gcs_evaluation_blob_path: str = "evaluations/recording.jsonl"
    sync_preferences_to_gcs: bool = True
    sync_evaluations_to_gcs: bool = True
    recording_type: str = "mvp"
    catalog_db_url: str = "https://storage.googleapis.com/web-agent-data-caleb-2026/metadata/catalog.db"
    lancedb_public_prefix: str = "https://storage.googleapis.com/web-agent-data-caleb-2026/data/processed/lancedb"
    lancedb_manifest_url: str = ""

    log_timing: bool = False
    include_debug_metadata: bool = False

    # ---- Worksheet/Acts Feature Switches ----
    use_worksheets: bool = False
    emit_worksheet_events: bool = False
    use_agent_acts: bool = False

    # ---- Grounded Response Acts ----
    # "dynamic":   Selects acts based on user intent. Best for capable API LLMs.
    # "hardcoded": Fixed act combo (Report+Recommend+Style). Best for OSS models.
    # "off":       Bypass acts, use original free-form synthesis.
    act_mode: str = "off"

    def resolve_image_url(self, value: str) -> str:
        if not value:
            return ""
        if value.startswith("http://") or value.startswith("https://"):
            return value
        return f"{self.gcs_public_url.rstrip('/')}/{value.lstrip('/')}"


class PipelineStage:
    COLD_START = ("cold_start", "Loading services for MVP cold start...")
    INTENT_DETECTION = ("intent_detection", "Understanding your request...")
    PREPARING_COMPARE = ("preparing_compare", "Preparing a side-by-side comparison...")
    ANALYZING_IMAGE = ("analyzing_image", "Analyzing your image...")
    DECOMPOSING_QUERY = ("decomposing_query", "Breaking down your query...")
    SOURCING_LOCAL = ("sourcing_local", "Sourcing matches from catalog...")
    SOURCING_WEB = ("sourcing_web", "Searching the web via Google Shopping...")
    RERANKING = ("reranking", "Ranking products by relevance...")
    RECALLING_MEMORY = ("recalling_memory", "Remembering your preferences...")
    GENERATING = ("generating", "Generating your personalized recommendations...")

    @staticmethod
    def to_sse(stage: tuple[str, str]) -> str:
        return json.dumps({"type": "status", "stage": stage[0], "message": stage[1]})


class MVPCommerceAgent:
    def __init__(self, config: dict, agent_config: MVPConfig | None = None):
        self.config = config
        self.ac = agent_config or MVPConfig()
        self._initialized = False
        self._stage_times: dict[str, float] = {}

        self.router = None
        self.master_brain = None
        self.semantic_embedder = None
        self.visual_embedder = None
        self.florence_tagger = None
        self.sqlite = None
        self.lancedb = None
        self.retriever = None
        self.image_pipeline = None
        self.memory = None
        self.web_pipeline = None
        self.preference_store: PreferenceStore | None = None
        self.gcs_sync: MVPGCSSyncClient | None = (
            MVPGCSSyncClient(
                bucket_name=self.ac.gcs_bucket_name,
                project_id=self.ac.gcs_project_id,
            )
            if self.ac.gcs_bucket_name
            else None
        )
        self.conversation_recording_store = InMemoryConversationRecordingStore()
        self.worksheet_registry = WorksheetRegistry()
        self.worksheet_router = WorksheetRouter(self.worksheet_registry)
        self.worksheet_engine = WorksheetEngine()
        self.worksheet_store = (
            build_worksheet_store(config)
            if self.ac.use_worksheets
            else InMemoryWorksheetStore()
        )
        self._ensure_preference_services()

    def _log_stage(self, name: str, start: float):
        elapsed = time.time() - start
        self._stage_times[name] = elapsed
        if self.ac.log_timing:
            print(f"  [timing] {name}: {elapsed:.3f}s")

    def _acts_enabled(self) -> bool:
        return self.ac.use_agent_acts and self.ac.act_mode != "off"

    def _worksheet_events_enabled(self) -> bool:
        return self.ac.use_worksheets and self.ac.emit_worksheet_events

    def _preferences_enabled(self) -> bool:
        return self.ac.use_preference_inference or self.ac.use_preference_reranking

    def _serialize_worksheet_event(
        self,
        definition: WorksheetDefinition,
        instance: WorksheetInstance,
    ) -> str | None:
        if not self._worksheet_events_enabled():
            return None
        return json.dumps(self.worksheet_engine.build_event_payload(definition, instance))

    def _ensure_preference_services(self) -> None:
        if not self._preferences_enabled():
            return

        if self.preference_store is None:
            self.preference_store = PreferenceStore(
                session_store=build_session_preference_store(
                    self.config,
                    ttl_seconds=self.ac.preference_redis_ttl_seconds,
                ),
                durable_store=SQLitePreferenceProfileStore(
                    db_path=self.ac.user_preferences_db_path,
                ),
            )

    def _record_assistant_message(self, session_id: str, content: str) -> None:
        assistant_message = ChatMessage(role="assistant", content=content, timestamp=time.time())
        self._record_conversation_message(session_id, assistant_message)
        if not self.memory:
            return
        self.memory.on_assistant_message(
            session_id,
            assistant_message,
        )

    def _record_conversation_message(self, session_id: str, message: ChatMessage) -> None:
        self.conversation_recording_store.add_message(session_id, message)

    def _store_captured_preferences(
        self,
        user_id: str,
        session_id: str,
        preferences,
    ) -> None:
        if not self.ac.use_preference_inference:
            return

        self._ensure_preference_services()
        if not self.preference_store or not preferences:
            return

        try:
            self.preference_store.update_session_preferences(
                user_id=user_id,
                session_id=session_id,
                preferences=preferences,
            )
        except Exception as exc:
            if self.ac.log_timing:
                print(f"  [preferences] capture skipped: {exc}")

    def _store_preferences_from_analysis(
        self,
        user_id: str,
        session_id: str,
        turn_analysis: TurnAnalysisResult | None,
    ) -> None:
        if not turn_analysis:
            return
        self._store_captured_preferences(
            user_id=user_id,
            session_id=session_id,
            preferences=turn_analysis.preferences,
        )

    def _get_preference_context(self, user_id: str, session_id: str) -> str:
        if not self.ac.use_preference_reranking:
            return ""

        self._ensure_preference_services()
        if not self.preference_store:
            return ""

        try:
            profile = self.preference_store.get_combined_profile(user_id, session_id)
            return build_preference_context(profile)
        except Exception as exc:
            if self.ac.log_timing:
                print(f"  [preferences] context unavailable: {exc}")
            return ""

    def _get_full_conversation_messages(self, session_id: str) -> list[ChatMessage]:
        if self.memory:
            try:
                messages = self.memory.episodic.get_full_conversation(session_id)
                if messages:
                    return messages
            except Exception as exc:
                if self.ac.log_timing:
                    print(f"  [evaluation] redis conversation fetch skipped: {exc}")
        return self.conversation_recording_store.get_messages(session_id)

    def _append_evaluation_record(
        self,
        user_id: str,
        session_id: str,
        preferences: dict[str, object],
    ) -> tuple[str | None, str | None, str | None]:
        messages = self._get_full_conversation_messages(session_id)
        if not messages:
            return None, None, None

        record = build_conversation_record(
            user_id=user_id,
            session_id=session_id,
            messages=messages,
            inferred_preferences=preferences,
            record_type=self.ac.recording_type,
        )
        local_path = append_record_to_jsonl(self.ac.local_evaluation_recordings_path, record)

        gcs_uri = None
        gcs_error = None
        if self.ac.sync_evaluations_to_gcs and self.gcs_sync:
            try:
                gcs_uri = self.gcs_sync.append_jsonl_record(
                    self.ac.gcs_evaluation_blob_path,
                    record,
                )
            except Exception as exc:
                gcs_error = str(exc)
                if self.ac.log_timing:
                    print(f"  [evaluation] gcs sync skipped: {exc}")

        return str(local_path), gcs_uri, gcs_error

    def _sync_preferences_db_to_gcs(self) -> tuple[str | None, str | None]:
        if not self.ac.sync_preferences_to_gcs or not self.gcs_sync:
            return None, None

        db_path = Path(self.ac.user_preferences_db_path)
        if not db_path.exists():
            return None, None

        try:
            return self.gcs_sync.upload_file(
                db_path,
                self.ac.gcs_preferences_blob_path,
                content_type="application/x-sqlite3",
            ), None
        except Exception as exc:
            if self.ac.log_timing:
                print(f"  [preferences] gcs db sync skipped: {exc}")
            return None, str(exc)

    def _clear_session_artifacts(self, session_id: str) -> None:
        self.conversation_recording_store.clear_session(session_id)
        if self.memory:
            try:
                self.memory.episodic.clear_session(session_id)
            except Exception as exc:
                if self.ac.log_timing:
                    print(f"  [evaluation] session cleanup skipped: {exc}")

    @staticmethod
    def _build_compare_synthesis_prompt(message: str, comparison_dimensions: list[str]) -> str:
        dimensions = [value for value in comparison_dimensions if value]
        if dimensions:
            return (
                "Compare the selected products side by side for the user. "
                f"Focus on these dimensions: {', '.join(dimensions)}. "
                f"User request: {message}"
            )
        return f"Compare the selected products side by side for the user. User request: {message}"

    def _prepare_product_search_worksheet(
        self,
        session_id: str,
        message: str,
        include_web: bool,
        decomposed: DecomposedQuery | None = None,
    ) -> tuple[WorksheetDefinition, WorksheetInstance, DecomposedQuery]:
        active_instance: WorksheetInstance | None = self.worksheet_store.get(session_id)
        definition = self.worksheet_router.resolve(IntentType.TEXT_SEARCH, active_instance)
        if definition is None or definition.name != "product_search":
            definition = self.worksheet_registry.get("product_search")
            active_instance = None

        instance = active_instance
        if not instance or instance.worksheet_name != definition.name:
            instance = self.worksheet_engine.create_instance(definition)

        decomposed = decomposed or self.router.decompose_query(message)
        instance = self.worksheet_engine.apply_product_search_turn(
            definition,
            instance,
            message,
            decomposed,
            include_web=include_web,
        )
        self.worksheet_store.save(session_id, instance)
        return definition, instance, decomposed

    def _should_handle_as_compare(
        self,
        message: str,
        active_instance: WorksheetInstance | None,
    ) -> bool:
        if not active_instance:
            return False
        if active_instance.worksheet_name not in {"product_search", "compare_products"}:
            return False
        if not self.worksheet_engine.is_compare_message(message):
            return False

        source_products = []
        if active_instance.worksheet_name == "compare_products":
            source_products = list(active_instance.result_refs.get("source_products", []))
        else:
            source_products = list(active_instance.result_refs.get("last_products", []))

        return len(source_products) >= 2

    def _prepare_compare_worksheet(
        self,
        session_id: str,
        message: str,
    ) -> tuple[WorksheetDefinition, WorksheetInstance]:
        active_instance: WorksheetInstance | None = self.worksheet_store.get(session_id)
        definition = self.worksheet_registry.get("compare_products")
        instance = self.worksheet_engine.create_compare_instance(definition, active_instance, message)
        self.worksheet_store.save(session_id, instance)
        return definition, instance

    def _ensure_catalog_db(self):
        db_path = Path(self.config["databases"]["sqlite"]["path"])
        if db_path.exists():
            return
        db_path.parent.mkdir(parents=True, exist_ok=True)
        response = httpx.get(self.ac.catalog_db_url, timeout=60)
        response.raise_for_status()
        db_path.write_bytes(response.content)
        print(f"  Downloaded catalog DB from {self.ac.catalog_db_url}")

    def _has_local_lancedb(self) -> bool:
        lancedb_path = Path(self.config["databases"]["lancedb"]["path"])
        table_name = self.config["databases"]["lancedb"]["table_name"]
        table_dir = lancedb_path / f"{table_name}.lance"
        if not table_dir.exists() or not any(table_dir.rglob("*")):
            return False

        try:
            from src.database import LanceDBCatalog

            catalog = LanceDBCatalog(
                db_path=str(lancedb_path),
                table_name=table_name,
                visual_dim=self.config["embeddings"]["visual"]["dimension"],
                semantic_dim=self.config["embeddings"]["semantic"]["dimension"],
            )
            return catalog.count() > 0
        except Exception:
            return False

    def _parse_storage_public_url(self, url: str) -> tuple[str, str]:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or parsed.netloc != "storage.googleapis.com":
            raise ValueError(
                "GCS public URLs must look like https://storage.googleapis.com/<bucket>/<prefix>"
            )

        trimmed = parsed.path.lstrip("/")
        if not trimmed:
            raise ValueError("GCS public URL is missing the bucket name.")

        parts = trimmed.split("/", 1)
        bucket = parts[0]
        prefix = parts[1] if len(parts) > 1 else ""
        return bucket, prefix.rstrip("/")

    def _list_public_gcs_objects(self, prefix_url: str) -> list[str]:
        bucket, prefix = self._parse_storage_public_url(prefix_url)
        client = httpx.Client(timeout=60)
        object_names: list[str] = []
        continuation_token = ""

        while True:
            params = {"prefix": prefix, "list-type": "2"}
            if continuation_token:
                params["continuation-token"] = continuation_token

            response = client.get(f"https://storage.googleapis.com/{bucket}", params=params)
            response.raise_for_status()

            root = ET.fromstring(response.text)
            namespace = {"s3": "http://doc.s3.amazonaws.com/2006-03-01"}

            keys = [
                node.text or ""
                for node in root.findall(".//s3:Contents/s3:Key", namespace)
            ]
            object_names.extend([key for key in keys if key and not key.endswith("/")])

            next_token = root.findtext(".//s3:NextContinuationToken", default="", namespaces=namespace)
            if not next_token:
                break
            continuation_token = next_token

        return object_names

    def _download_url_to_path(self, url: str, destination: Path):
        destination.parent.mkdir(parents=True, exist_ok=True)
        with httpx.stream("GET", url, timeout=120, follow_redirects=True) as response:
            response.raise_for_status()
            with destination.open("wb") as f:
                for chunk in response.iter_bytes():
                    if chunk:
                        f.write(chunk)

    def _download_lancedb_from_manifest(self, manifest_url: str, destination_root: Path) -> int:
        response = httpx.get(manifest_url, timeout=60)
        response.raise_for_status()
        payload = response.json()

        files = payload.get("files")
        if not isinstance(files, list) or not files:
            raise ValueError("LanceDB manifest must contain a non-empty 'files' list.")

        downloaded = 0
        for entry in files:
            if not isinstance(entry, dict):
                continue

            relative_path = entry.get("path", "").strip().replace("\\", "/")
            source_url = entry.get("url", "").strip()
            if not relative_path or not source_url:
                continue

            self._download_url_to_path(source_url, destination_root / relative_path)
            downloaded += 1

        return downloaded

    def _download_lancedb_from_public_prefix(self, prefix_url: str, destination_root: Path) -> int:
        bucket, prefix = self._parse_storage_public_url(prefix_url)
        object_names = self._list_public_gcs_objects(prefix_url)
        if not object_names:
            raise RuntimeError(f"No LanceDB objects found under {prefix_url}")

        prefix_with_slash = f"{prefix}/" if prefix else ""
        downloaded = 0
        for object_name in object_names:
            if prefix_with_slash and not object_name.startswith(prefix_with_slash):
                continue

            relative_path = object_name[len(prefix_with_slash):] if prefix_with_slash else object_name
            if not relative_path:
                continue

            object_url = f"https://storage.googleapis.com/{bucket}/{quote(object_name, safe='/')}"
            self._download_url_to_path(object_url, destination_root / relative_path)
            downloaded += 1

        return downloaded

    def _ensure_lancedb(self):
        lancedb_path = Path(self.config["databases"]["lancedb"]["path"])
        if self._has_local_lancedb():
            return

        lancedb_path.mkdir(parents=True, exist_ok=True)

        downloaded = 0
        if self.ac.lancedb_manifest_url:
            downloaded = self._download_lancedb_from_manifest(self.ac.lancedb_manifest_url, lancedb_path)
            source = self.ac.lancedb_manifest_url
        else:
            downloaded = self._download_lancedb_from_public_prefix(self.ac.lancedb_public_prefix, lancedb_path)
            source = self.ac.lancedb_public_prefix

        if not self._has_local_lancedb():
            raise RuntimeError(
                f"Finished downloading LanceDB artifacts from {source}, but the product vector table is still missing."
            )

        print(f"  Downloaded {downloaded} LanceDB files from {source}")

    def _normalize_product(self, product: dict) -> dict:
        item = dict(product)
        explicit_image = str(item.get("image", "") or item.get("image_url", "") or "").strip()
        raw_images = item.get("image_urls", "")
        first_image = ""

        if explicit_image:
            first_image = explicit_image
        elif isinstance(raw_images, list):
            first_image = raw_images[0] if raw_images else ""
        elif isinstance(raw_images, str):
            cleaned = raw_images.strip()
            if cleaned.startswith("[") and cleaned.endswith("]"):
                try:
                    parsed = json.loads(cleaned)
                    if isinstance(parsed, list) and parsed:
                        first_image = str(parsed[0]).strip()
                except Exception:
                    first_image = cleaned.split(",")[0].strip() if cleaned else ""
            else:
                first_image = cleaned.split(",")[0].strip() if cleaned else ""

        item["image"] = self.ac.resolve_image_url(first_image)
        item["image_urls"] = (
            item["image"]
            or self.ac.resolve_image_url(str(item.get("image_url", "") or "").strip())
            or raw_images
        )
        item["merchant"] = item.get("merchant") or item.get("brand") or ""
        return item

    def _sanitize_query(self, query):
        if not self.sqlite:
            return query

        filters = dict(query.filters or {})
        category = (filters.get("category") or "").strip()
        if category:
            known_categories = {
                str(value).strip().lower()
                for value in self.sqlite.get_all_categories()
                if str(value).strip()
            }
            if category.lower() not in known_categories:
                filters.pop("category", None)
                tags = list(query.tags or [])
                if category.lower() not in {tag.lower() for tag in tags}:
                    tags.append(category)
                query.tags = tags

        query.filters = filters
        return query

    def _fuse_ranked_product_lists(
        self,
        ranked_lists: list[tuple[str, list[dict]]],
        limit: int,
    ) -> list[dict]:
        scored_products: dict[str, dict] = {}
        scores: dict[str, float] = {}

        for source_name, products in ranked_lists:
            for rank, product in enumerate(products):
                key = (
                    product.get("product_id")
                    or product.get("url")
                    or f"{source_name}:{product.get('title', '')}:{rank}"
                )
                score = 1.0 / (self.ac.rrf_k + rank + 1)
                scores[key] = scores.get(key, 0.0) + score
                merged = dict(product)
                merged["source"] = merged.get("source") or source_name
                merged["retrieval_score"] = round(scores[key], 6)
                scored_products[key] = merged

        ordered = sorted(
            scored_products.values(),
            key=lambda item: -item.get("retrieval_score", 0.0),
        )
        return ordered[:limit]

    @staticmethod
    def _tokenize_keywords(text: str) -> list[str]:
        return [token for token in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(token) > 2]

    def _apply_keyword_type_boosts(self, query_text: str, products: list[dict], limit: int) -> list[dict]:
        if not products:
            return products

        product_type_markers = {
            "jacket",
            "jackets",
            "coat",
            "coats",
            "blazer",
            "blazers",
            "shacket",
            "shackets",
            "parka",
            "parkas",
            "hoodie",
            "hoodies",
            "windbreaker",
            "windbreakers",
            "bomber",
            "bombers",
            "shirt",
            "shirts",
            "dress",
            "dresses",
            "shoe",
            "shoes",
            "sneaker",
            "sneakers",
            "bag",
            "bags",
            "boot",
            "boots",
        }

        keyword_candidates = {
            token
            for token in self._tokenize_keywords(query_text)
            if token in product_type_markers
        }
        if not keyword_candidates:
            return products[:limit]

        boosted = []
        for product in products:
            title = (product.get("title") or "").lower()
            subcategory = (product.get("subcategory") or "").lower()
            category = (product.get("category") or "").lower()
            combined = " ".join([title, subcategory, category])

            boost = 0.0
            for keyword in keyword_candidates:
                singular = keyword[:-1] if keyword.endswith("s") else keyword
                variants = {keyword, singular}
                if any(variant and variant in title for variant in variants):
                    boost += 3.0
                if any(variant and variant in subcategory for variant in variants):
                    boost += 2.0
                if any(variant and variant in category for variant in variants):
                    boost += 0.5

            # Stronger penalty when the product clearly matches a different product type
            # than the one requested by the current query.
            if boost == 0.0:
                other_type_match = False
                for marker in product_type_markers:
                    singular = marker[:-1] if marker.endswith("s") else marker
                    if singular in combined or marker in combined:
                        other_type_match = True
                        break
                if other_type_match:
                    boost -= 3.0
                elif any(
                    marker in combined
                    for marker in ["cap", "hat", "skirt", "dress", "sandal", "bag", "shirt"]
                ):
                    boost -= 1.0

            updated = dict(product)
            updated["keyword_boost"] = boost
            updated["ranking_score"] = updated.get("retrieval_score", 0.0) + boost
            boosted.append(updated)

        boosted.sort(key=lambda item: item.get("ranking_score", 0.0), reverse=True)
        return boosted[:limit]

    def initialize(self):
        if self._initialized:
            return

        cfg = self.config
        ac = self.ac

        print(f"\n{'='*60}")
        print("  MVPCommerceAgent Cold Start")
        print(f"  Master Brain:      {ac.master_brain_model_name}")
        print(f"  Router Model:      {ac.router_model_name}")
        print(f"  API Base:          {ac.api_base}")
        print(f"  Use Web Search:    {ac.use_web_search}")
        print(f"  Use Memory:        {ac.use_memory}")
        print(f"  Use Worksheets:    {ac.use_worksheets}")
        print(f"  Use Agent Acts:    {ac.use_agent_acts} ({ac.act_mode})")
        print(f"  Preference Infer:  {ac.use_preference_inference}")
        print(f"  Preference Rerank: {ac.use_preference_reranking}")
        print(f"{'='*60}\n")

        from src.database import LanceDBCatalog, LanceDBMemoryStore, SQLiteCatalog
        from src.embeddings import BGEM3Embedder, DINOv2Embedder
        from src.image_search import Florence2Tagger, ImageSearchPipeline
        from src.memory import (
            ConversationalMemory,
            EpisodicMemory,
            MemoryManager,
            SemanticMemory,
        )
        from src.retrieval import HybridRetriever
        from src.web_search import SerpApiGoogleShoppingSearcher, WebSearchPipeline

        self._ensure_preference_services()
        self._ensure_catalog_db()
        self._ensure_lancedb()

        self.sqlite = SQLiteCatalog(cfg["databases"]["sqlite"]["path"])
        self.lancedb = LanceDBCatalog(
            db_path=cfg["databases"]["lancedb"]["path"],
            table_name=cfg["databases"]["lancedb"]["table_name"],
            visual_dim=cfg["embeddings"]["visual"]["dimension"],
            semantic_dim=cfg["embeddings"]["semantic"]["dimension"],
        )

        self.semantic_embedder = BGEM3Embedder(
            model_id=cfg["embeddings"]["semantic"]["model_id"],
            batch_size=cfg["embeddings"]["semantic"]["batch_size"],
            use_fp16=cfg["embeddings"]["semantic"]["use_fp16"],
        )
        self.visual_embedder = DINOv2Embedder(
            model_id=cfg["embeddings"]["visual"]["model_id"],
            batch_size=cfg["embeddings"]["visual"]["batch_size"],
        )

        if ac.use_florence:
            self.florence_tagger = Florence2Tagger(
                model_id=cfg["models"]["florence"]["model_id"],
            )

        self.router = MVPRouter(
            api_base=ac.api_base,
            api_key=ac.api_key,
            model_name=ac.router_model_name,
            reranker_model_name=ac.reranker_model_name,
        )
        self.master_brain = MasterBrain(
            api_base=ac.api_base,
            api_key=ac.api_key,
            model_name=ac.master_brain_model_name,
        )

        self.retriever = HybridRetriever(
            sqlite_catalog=self.sqlite,
            lancedb_catalog=self.lancedb,
            semantic_embedder=self.semantic_embedder,
            visual_embedder=self.visual_embedder,
            handyman=self.router,
            top_k_initial=ac.top_k_initial,
            top_k_reranked=ac.top_k_reranked,
            rrf_k=ac.rrf_k,
        )

        if self.florence_tagger:
            self.image_pipeline = ImageSearchPipeline(
                florence_tagger=self.florence_tagger,
                visual_embedder=self.visual_embedder,
                semantic_embedder=self.semantic_embedder,
                hybrid_retriever=self.retriever,
            )

        if ac.use_web_search and (ac.serpapi_api_key or ac.serpapi_mock_results_path):
            searcher = SerpApiGoogleShoppingSearcher(
                api_key=ac.serpapi_api_key,
                api_base=ac.serpapi_api_base,
                location=ac.serpapi_location,
                gl=ac.serpapi_gl,
                hl=ac.serpapi_hl,
                mock_results_path=ac.serpapi_mock_results_path,
            )
            self.web_pipeline = WebSearchPipeline(searcher=searcher)

        if ac.use_memory:
            mem_cfg = cfg["memory"]
            episodic = EpisodicMemory(
                host=mem_cfg["episodic"]["host"],
                port=mem_cfg["episodic"]["port"],
                db=mem_cfg["episodic"]["db"],
                max_turns=mem_cfg["episodic"]["max_turns"],
                ttl_seconds=mem_cfg["episodic"]["ttl_seconds"],
            )
            memory_store = LanceDBMemoryStore(
                db_path=mem_cfg["semantic"]["path"],
                embedding_dim=cfg["embeddings"]["semantic"]["dimension"],
            )
            semantic = SemanticMemory(memory_store)
            conversational = ConversationalMemory(
                memory_store,
                max_summaries_per_user=mem_cfg["conversational"]["max_summaries_per_user"],
            )
            self.memory = MemoryManager(episodic, semantic, conversational)

        self._initialized = True
        print("\n[ok] MVPCommerceAgent initialized.\n")

    async def handle_message(
        self,
        user_id: str,
        session_id: str,
        message: str,
        image_bytes: Optional[bytes] = None,
        web_search_enabled: bool = False,
    ) -> AsyncIterator[str]:
        self._stage_times = {}

        if not self._initialized:
            yield PipelineStage.to_sse(PipelineStage.COLD_START)
            self.initialize()

        user_message = ChatMessage(role="user", content=message, timestamp=time.time())
        self._record_conversation_message(session_id, user_message)
        if self.memory:
            self.memory.on_user_message(
                session_id,
                user_message,
            )

        active_instance = self.worksheet_store.get(session_id) if self.ac.use_worksheets else None
        compare_override = (
            self.ac.use_worksheets
            and image_bytes is None
            and self._should_handle_as_compare(message, active_instance)
        )
        turn_analysis: TurnAnalysisResult | None = None

        t0 = time.time()
        yield PipelineStage.to_sse(PipelineStage.INTENT_DETECTION)
        if compare_override:
            intent = IntentType.TEXT_SEARCH
        else:
            turn_analysis = self.router.analyze_turn(
                message,
                has_image=image_bytes is not None,
            )
            self._store_preferences_from_analysis(user_id, session_id, turn_analysis)
            intent = turn_analysis.intent
        self._log_stage("intent_detection", t0)

        if compare_override:
            async for event in self._workflow_compare(user_id, session_id, message):
                yield event
        elif intent == IntentType.GENERAL_TALK:
            async for event in self._workflow_general(user_id, session_id, message):
                yield event
        elif intent in (IntentType.TEXT_SEARCH, IntentType.WEB_SEARCH):
            async for event in self._workflow_text_search(
                user_id,
                session_id,
                message,
                include_web=(web_search_enabled or intent == IntentType.WEB_SEARCH),
                analyzed_query=(
                    turn_analysis.to_decomposed_query(message, default_intent=IntentType.TEXT_SEARCH)
                    if turn_analysis
                    else None
                ),
            ):
                yield event
        elif intent == IntentType.IMAGE_SEARCH:
            async for event in self._workflow_image_search(
                user_id,
                session_id,
                message,
                image_bytes,
                include_web=web_search_enabled,
                analyzed_query=(
                    turn_analysis.to_decomposed_query(message, default_intent=IntentType.IMAGE_SEARCH)
                    if turn_analysis
                    else None
                ),
            ):
                yield event

        if self.ac.include_debug_metadata:
            yield json.dumps({"type": "debug", "timings": self._stage_times})

        yield json.dumps({"type": "done"})

    async def _workflow_general(self, user_id: str, session_id: str, message: str) -> AsyncIterator[str]:
        memory_context = ""
        chat_history = []
        if self.memory:
            yield PipelineStage.to_sse(PipelineStage.RECALLING_MEMORY)
            query_emb = self.semantic_embedder.embed_query(message)
            memory_context = self.memory.get_memory_context(user_id, query_emb.tolist())
            chat_history = self.memory.get_chat_history(session_id)

        t0 = time.time()
        yield PipelineStage.to_sse(PipelineStage.GENERATING)
        full_response = ""
        async for token in self.master_brain.general_chat_stream(message, chat_history, memory_context):
            full_response += token
            yield json.dumps({"type": "token", "content": token})
        self._log_stage("generation", t0)

        self._record_assistant_message(session_id, full_response)

    async def _workflow_compare(
        self,
        user_id: str,
        session_id: str,
        message: str,
    ) -> AsyncIterator[str]:
        t0 = time.time()
        yield PipelineStage.to_sse(PipelineStage.PREPARING_COMPARE)
        definition, worksheet_instance = self._prepare_compare_worksheet(session_id, message)
        self._log_stage("prepare_compare", t0)
        worksheet_event = self._serialize_worksheet_event(definition, worksheet_instance)
        if worksheet_event:
            yield worksheet_event

        comparison_products = list(worksheet_instance.result_refs.get("comparison_products", []))
        if len(comparison_products) < 2:
            clarification = self.worksheet_engine.build_compare_clarification_question(worksheet_instance)
            t0 = time.time()
            yield PipelineStage.to_sse(PipelineStage.GENERATING)
            yield json.dumps({"type": "token", "content": clarification})
            self._log_stage("generation", t0)
            self._record_assistant_message(session_id, clarification)
            return

        memory_context = ""
        chat_history = []
        if self.memory:
            yield PipelineStage.to_sse(PipelineStage.RECALLING_MEMORY)
            query_emb = self.semantic_embedder.embed_query(message)
            memory_context = self.memory.get_memory_context(user_id, query_emb.tolist())
            chat_history = self.memory.get_chat_history(session_id)

        frontend_items = [self._normalize_product(item) for item in comparison_products]
        yield json.dumps(
            {
                "type": "products",
                "items": frontend_items,
                "tags": worksheet_instance.values.get("comparison_dimensions", []),
                "mode": "compare",
            }
        )

        t0 = time.time()
        yield PipelineStage.to_sse(PipelineStage.GENERATING)
        full_response = ""
        if self._acts_enabled():
            from src.agent_acts import ActBuilder

            acts = (
                ActBuilder()
                .report(
                    frontend_items,
                    query=message,
                    source=worksheet_instance.values.get("source_label", "recent results"),
                )
                .compare(
                    indices=list(range(1, len(frontend_items) + 1)),
                    dimensions=worksheet_instance.values.get("comparison_dimensions", []),
                    query=message,
                    pick_winner=True,
                )
                .style(
                    tone="analytical but friendly",
                    format_hint="Use a structured comparison, then give a clear verdict.",
                    followup=True,
                )
                .build()
            )
            async for token in self.master_brain.grounded_synthesize_stream(
                message,
                acts,
                chat_history,
                memory_context,
            ):
                full_response += token
                yield json.dumps({"type": "token", "content": token})
        else:
            compare_prompt = self._build_compare_synthesis_prompt(
                message,
                worksheet_instance.values.get("comparison_dimensions", []),
            )
            async for token in self.master_brain.synthesize_stream(
                compare_prompt,
                frontend_items,
                chat_history,
                memory_context,
            ):
                full_response += token
                yield json.dumps({"type": "token", "content": token})
        self._log_stage("generation", t0)

        self._record_assistant_message(session_id, full_response)

    async def _workflow_text_search(
        self,
        user_id: str,
        session_id: str,
        message: str,
        include_web: bool = False,
        analyzed_query: DecomposedQuery | None = None,
    ) -> AsyncIterator[str]:
        t0 = time.time()
        yield PipelineStage.to_sse(PipelineStage.DECOMPOSING_QUERY)
        definition = None
        worksheet_instance = None
        if self.ac.use_worksheets:
            definition, worksheet_instance, _raw_query = self._prepare_product_search_worksheet(
                session_id,
                message,
                include_web,
                decomposed=analyzed_query,
            )
            query = self.worksheet_engine.build_query_from_instance(worksheet_instance, message)
            query = self._sanitize_query(query)
            self._log_stage("decomposition", t0)
            worksheet_event = self._serialize_worksheet_event(definition, worksheet_instance)
            if worksheet_event:
                yield worksheet_event

            search_query_text = query.rewritten_query or message

            if worksheet_instance.missing_required_fields:
                clarification = self.worksheet_engine.build_clarification_question(
                    definition,
                    worksheet_instance,
                )
                t0 = time.time()
                yield PipelineStage.to_sse(PipelineStage.GENERATING)
                yield json.dumps({"type": "token", "content": clarification})
                self._log_stage("generation", t0)
                self._record_assistant_message(session_id, clarification)
                return
        else:
            query = analyzed_query or self.router.decompose_query(message)
            query = self._sanitize_query(query)
            self._log_stage("decomposition", t0)
            search_query_text = query.rewritten_query or message

        t0 = time.time()
        yield PipelineStage.to_sse(PipelineStage.SOURCING_LOCAL)
        local_products = self.retriever.search_text(query)
        for product in local_products:
            product["source"] = "local"
        self._log_stage("local_retrieval", t0)

        web_products = []
        if include_web and self.web_pipeline:
            t0 = time.time()
            yield PipelineStage.to_sse(PipelineStage.SOURCING_WEB)
            web_result = self.web_pipeline.search(
                query=search_query_text,
                num_results=self.ac.web_num_results,
            )
            web_products = web_result["products"]
            for product in web_products:
                product["source"] = "web"
            self._log_stage("web_retrieval", t0)

        t0 = time.time()
        yield PipelineStage.to_sse(PipelineStage.RERANKING)
        fused_candidates = self._fuse_ranked_product_lists(
            [("local", local_products), ("web", web_products)],
            limit=max(self.ac.top_k_reranked, self.ac.top_k_final * 3),
        )
        fused_candidates = self._apply_keyword_type_boosts(
            search_query_text,
            fused_candidates,
            limit=max(self.ac.top_k_reranked, self.ac.top_k_final * 3),
        )
        preference_context = self._get_preference_context(user_id, session_id)
        reranked = self.router.rerank(
            search_query_text,
            fused_candidates,
            top_k=self.ac.top_k_final,
            preference_context=preference_context,
        )
        self._log_stage("reranking", t0)

        if self.ac.use_worksheets and definition and worksheet_instance:
            worksheet_instance = self.worksheet_engine.update_result_refs(
                worksheet_instance,
                local_products=local_products,
                web_products=web_products,
                reranked_products=reranked,
                query=query,
                include_web=include_web,
            )
            self.worksheet_store.save(session_id, worksheet_instance)

        memory_context = ""
        chat_history = []
        if self.memory:
            yield PipelineStage.to_sse(PipelineStage.RECALLING_MEMORY)
            query_emb = self.semantic_embedder.embed_query(search_query_text)
            memory_context = self.memory.get_memory_context(user_id, query_emb.tolist())
            chat_history = self.memory.get_chat_history(session_id)

        frontend_items = [self._normalize_product(item) for item in reranked[: self.ac.top_k_final]]
        if self.ac.use_worksheets and definition and worksheet_instance:
            worksheet_event = self._serialize_worksheet_event(definition, worksheet_instance)
            if worksheet_event:
                yield worksheet_event
        yield json.dumps({"type": "products", "items": frontend_items, "tags": query.tags, "filters": query.filters})

        t0 = time.time()
        yield PipelineStage.to_sse(PipelineStage.GENERATING)
        if not frontend_items:
            no_results = (
                "I couldn't find strong matches in the current catalog for that request. "
                "Try broadening the category or removing one constraint like budget or product type."
            )
            yield json.dumps({"type": "token", "content": no_results})
            self._log_stage("generation", t0)
            self._record_assistant_message(session_id, no_results)
            return

        full_response = ""
        if self._acts_enabled():
            from src.agent_acts import select_acts

            source = "catalog + web" if include_web else "local catalog"
            acts = select_acts(
                mode=self.ac.act_mode,
                message=search_query_text,
                products=frontend_items,
                source=source,
            )
            if acts:
                async for token in self.master_brain.grounded_synthesize_stream(
                    search_query_text, acts, chat_history, memory_context
                ):
                    full_response += token
                    yield json.dumps({"type": "token", "content": token})
            else:
                async for token in self.master_brain.synthesize_stream(
                    search_query_text,
                    frontend_items,
                    chat_history,
                    memory_context,
                ):
                    full_response += token
                    yield json.dumps({"type": "token", "content": token})
        else:
            async for token in self.master_brain.synthesize_stream(
                search_query_text,
                frontend_items,
                chat_history,
                memory_context,
            ):
                full_response += token
                yield json.dumps({"type": "token", "content": token})
        self._log_stage("generation", t0)

        self._record_assistant_message(session_id, full_response)

    async def _workflow_image_search(
        self,
        user_id: str,
        session_id: str,
        message: str,
        image_bytes: Optional[bytes],
        include_web: bool = False,
        analyzed_query: DecomposedQuery | None = None,
    ) -> AsyncIterator[str]:
        t0 = time.time()
        yield PipelineStage.to_sse(PipelineStage.ANALYZING_IMAGE)

        filters = {}
        if analyzed_query is not None:
            filters = dict(analyzed_query.filters or {})
        elif message:
            filters = self.router.decompose_query(message).filters

        caption = ""
        tags = []
        products = []

        if self.ac.use_florence and self.image_pipeline:
            result = self.image_pipeline.search(
                image_source=image_bytes,
                user_text=message,
                filters=filters,
            )
            caption = result["caption"]
            tags = result["tags"]
            products = result.get("products", [])
        elif image_bytes:
            image_data_url = f"data:image/jpeg;base64,{base64.b64encode(image_bytes).decode('utf-8')}"
            caption = self.router._chat_completion(
                system_prompt=(
                    "Describe this product image in detail. Include item type, color, material, style, and brand if visible. "
                    "Output a concise description."
                ),
                user_message=[
                    {"type": "text", "text": "Describe this product image:"},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
                model=self.ac.router_model_name,
                max_tokens=200,
            )
            tags = [part.strip() for part in caption.split(",") if part.strip()]

            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp.write(image_bytes)
                tmp_path = tmp.name

            visual_embedding = self.visual_embedder.embed_single(tmp_path)
            products = self.retriever.search_hybrid(
                text_query=caption or message or "similar products",
                image_embedding=visual_embedding,
                tags=tags,
                filters=filters,
            )

        for product in products:
            product["source"] = "local"
        self._log_stage("image_analysis", t0)

        web_products = []
        if include_web and self.web_pipeline and caption:
            t0 = time.time()
            yield PipelineStage.to_sse(PipelineStage.SOURCING_WEB)
            web_result = self.web_pipeline.search(
                query=caption,
                num_results=self.ac.web_num_results,
                include_visual=True,
            )
            web_products = web_result["products"]
            for product in web_products:
                product["source"] = "web"
            self._log_stage("web_image_retrieval", t0)

        combined = self._fuse_ranked_product_lists(
            [("local", products), ("web", web_products)],
            limit=max(self.ac.top_k_reranked, self.ac.top_k_final * 3),
        )
        combined = self._apply_keyword_type_boosts(
            caption or message,
            combined,
            limit=max(self.ac.top_k_reranked, self.ac.top_k_final * 3),
        )

        t0 = time.time()
        yield PipelineStage.to_sse(PipelineStage.RERANKING)
        preference_context = self._get_preference_context(user_id, session_id)
        reranked = self.router.rerank(
            caption or message,
            combined,
            top_k=self.ac.top_k_final,
            preference_context=preference_context,
        )
        self._log_stage("reranking", t0)

        memory_context = ""
        chat_history = []
        if self.memory:
            yield PipelineStage.to_sse(PipelineStage.RECALLING_MEMORY)
            query_emb = self.semantic_embedder.embed_query(caption or message)
            memory_context = self.memory.get_memory_context(user_id, query_emb.tolist())
            chat_history = self.memory.get_chat_history(session_id)

        frontend_items = [self._normalize_product(item) for item in reranked[: self.ac.top_k_final]]
        yield json.dumps({"type": "products", "items": frontend_items, "tags": tags, "caption": caption})

        t0 = time.time()
        yield PipelineStage.to_sse(PipelineStage.GENERATING)
        synth_query = f"The user uploaded an image that shows: {caption}."
        if message:
            synth_query += f" They also said: {message}"

        full_response = ""
        if self._acts_enabled():
            from src.agent_acts import select_acts

            source = "image search + web" if include_web else "image search"
            acts = select_acts(
                mode=self.ac.act_mode,
                message=synth_query,
                products=frontend_items,
                source=source,
                is_image_search=True,
            )
            if acts:
                async for token in self.master_brain.grounded_synthesize_stream(
                    synth_query, acts, chat_history, memory_context
                ):
                    full_response += token
                    yield json.dumps({"type": "token", "content": token})
            else:
                async for token in self.master_brain.synthesize_stream(
                    synth_query,
                    frontend_items,
                    chat_history,
                    memory_context,
                ):
                    full_response += token
                    yield json.dumps({"type": "token", "content": token})
        else:
            async for token in self.master_brain.synthesize_stream(
                synth_query,
                frontend_items,
                chat_history,
                memory_context,
            ):
                full_response += token
                yield json.dumps({"type": "token", "content": token})
        self._log_stage("generation", t0)

        self._record_assistant_message(session_id, full_response)

    async def finalize_session(self, user_id: str, session_id: str) -> dict:
        self._ensure_preference_services()

        stored_profile = None
        if self.preference_store:
            stored_profile = self.preference_store.finalize_session(user_id, session_id)

        preferences = stored_profile.preferences if stored_profile else {}
        evaluation_record_path, evaluation_gcs_uri, evaluation_gcs_error = self._append_evaluation_record(
            user_id=user_id,
            session_id=session_id,
            preferences=preferences,
        )
        preference_db_gcs_uri, preference_db_gcs_error = self._sync_preferences_db_to_gcs()

        if self.ac.use_worksheets:
            self.worksheet_store.clear(session_id)
        self._clear_session_artifacts(session_id)

        return {
            "status": "finalized",
            "user_id": user_id,
            "session_id": session_id,
            "preferences": preferences,
            "updated_at": stored_profile.updated_at if stored_profile else None,
            "evaluation_record_path": evaluation_record_path,
            "evaluation_gcs_uri": evaluation_gcs_uri,
            "evaluation_gcs_error": evaluation_gcs_error,
            "preference_db_gcs_uri": preference_db_gcs_uri,
            "preference_db_gcs_error": preference_db_gcs_error,
        }
