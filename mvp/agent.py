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

from mvp.router import MVPRouter
from src.master_brain import MasterBrain
from src.schema import ChatMessage, IntentType


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
    image_storage_provider: str = "gcs"
    gcs_public_url: str = "https://storage.googleapis.com/web-agent-data-caleb-2026"
    catalog_db_url: str = "https://storage.googleapis.com/web-agent-data-caleb-2026/metadata/catalog.db"
    lancedb_public_prefix: str = "https://storage.googleapis.com/web-agent-data-caleb-2026/data/processed/lancedb"
    lancedb_manifest_url: str = ""

    log_timing: bool = False
    include_debug_metadata: bool = False

    def resolve_image_url(self, value: str) -> str:
        if not value:
            return ""
        if value.startswith("http://") or value.startswith("https://"):
            return value
        return f"{self.gcs_public_url.rstrip('/')}/{value.lstrip('/')}"


class PipelineStage:
    COLD_START = ("cold_start", "Loading services for MVP cold start...")
    INTENT_DETECTION = ("intent_detection", "Understanding your request...")
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

    def _log_stage(self, name: str, start: float):
        elapsed = time.time() - start
        self._stage_times[name] = elapsed
        if self.ac.log_timing:
            print(f"  [timing] {name}: {elapsed:.3f}s")

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
        explicit_image = str(item.get("image", "") or "").strip()
        raw_images = item.get("image_urls", "")
        first_image = ""

        if explicit_image:
            first_image = explicit_image
        elif isinstance(raw_images, list):
            first_image = raw_images[0] if raw_images else ""
        elif isinstance(raw_images, str):
            first_image = raw_images.split(",")[0].strip() if raw_images else ""

        item["image"] = self.ac.resolve_image_url(first_image)
        item["image_urls"] = item["image"] or raw_images
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

        keyword_candidates = {
            token
            for token in self._tokenize_keywords(query_text)
            if token in {
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

            # Mild penalty for obvious misses when a strong product type is requested.
            if boost == 0.0 and any(
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

        if self.memory:
            self.memory.on_user_message(
                session_id,
                ChatMessage(role="user", content=message, timestamp=time.time()),
            )

        t0 = time.time()
        yield PipelineStage.to_sse(PipelineStage.INTENT_DETECTION)
        intent = self.router.detect_intent(message, has_image=image_bytes is not None)
        self._log_stage("intent_detection", t0)

        if intent == IntentType.GENERAL_TALK:
            async for event in self._workflow_general(user_id, session_id, message):
                yield event
        elif intent in (IntentType.TEXT_SEARCH, IntentType.WEB_SEARCH):
            async for event in self._workflow_text_search(
                user_id, session_id, message, include_web=web_search_enabled
            ):
                yield event
        elif intent == IntentType.IMAGE_SEARCH:
            async for event in self._workflow_image_search(
                user_id, session_id, message, image_bytes, include_web=web_search_enabled
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

        if self.memory:
            self.memory.on_assistant_message(
                session_id,
                ChatMessage(role="assistant", content=full_response, timestamp=time.time()),
            )

    async def _workflow_text_search(
        self,
        user_id: str,
        session_id: str,
        message: str,
        include_web: bool = False,
    ) -> AsyncIterator[str]:
        t0 = time.time()
        yield PipelineStage.to_sse(PipelineStage.DECOMPOSING_QUERY)
        query = self.router.decompose_query(message)
        query = self._sanitize_query(query)
        self._log_stage("decomposition", t0)

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
                query=query.rewritten_query or message,
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
            query.rewritten_query or message,
            fused_candidates,
            limit=max(self.ac.top_k_reranked, self.ac.top_k_final * 3),
        )
        reranked = self.router.rerank(message, fused_candidates, top_k=self.ac.top_k_final)
        self._log_stage("reranking", t0)

        memory_context = ""
        chat_history = []
        if self.memory:
            yield PipelineStage.to_sse(PipelineStage.RECALLING_MEMORY)
            query_emb = self.semantic_embedder.embed_query(message)
            memory_context = self.memory.get_memory_context(user_id, query_emb.tolist())
            chat_history = self.memory.get_chat_history(session_id)

        frontend_items = [self._normalize_product(item) for item in reranked[: self.ac.top_k_final]]
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
            if self.memory:
                self.memory.on_assistant_message(
                    session_id,
                    ChatMessage(role="assistant", content=no_results, timestamp=time.time()),
                )
            return

        full_response = ""
        async for token in self.master_brain.synthesize_stream(message, frontend_items, chat_history, memory_context):
            full_response += token
            yield json.dumps({"type": "token", "content": token})
        self._log_stage("generation", t0)

        if self.memory:
            self.memory.on_assistant_message(
                session_id,
                ChatMessage(role="assistant", content=full_response, timestamp=time.time()),
            )

    async def _workflow_image_search(
        self,
        user_id: str,
        session_id: str,
        message: str,
        image_bytes: Optional[bytes],
        include_web: bool = False,
    ) -> AsyncIterator[str]:
        t0 = time.time()
        yield PipelineStage.to_sse(PipelineStage.ANALYZING_IMAGE)

        filters = {}
        if message:
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
        reranked = self.router.rerank(caption or message, combined, top_k=self.ac.top_k_final)
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
        async for token in self.master_brain.synthesize_stream(
            synth_query,
            frontend_items,
            chat_history,
            memory_context,
        ):
            full_response += token
            yield json.dumps({"type": "token", "content": token})
        self._log_stage("generation", t0)

        if self.memory:
            self.memory.on_assistant_message(
                session_id,
                ChatMessage(role="assistant", content=full_response, timestamp=time.time()),
            )
