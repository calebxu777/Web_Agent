"""
agent.py — Top-Level Agent Orchestrator
=========================================
The CommerceAgent ties everything together:
- Receives user input (text and/or image)
- Routes via Handyman (intent detection + decomposition)
- Dispatches to the appropriate workflow (local, web, image search)
- Manages memory read/write
- Returns streaming responses from Master Brain

Experiment Switches:
  AgentConfig controls all toggleable features for A/B testing:
    - use_florence: use Florence-2 for image captioning vs Handyman VLM
    - use_web_search: enable Firecrawl web search pipeline
    - use_visual_verifier: enable LoRA-Verifier for image match filtering
    - master_brain_model_name: swap different models as the synthesis engine
    - handyman_model_name: swap router model
    - top_k_final: number of products to present
    - etc.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional

# Heavy module imports are LAZY — only loaded in production mode inside initialize().
# This lets mock mode run on a laptop with zero GPU/ML dependencies installed.
# Only lightweight schema types are imported eagerly.
from src.schema import ChatMessage, IntentType


# ================================================================
# Experiment Configuration
# ================================================================

@dataclass
class AgentConfig:
    """
    Central experiment switches for A/B testing and benchmarking.
    All toggles are documented for README.md performance tables.

    Usage:
        config = AgentConfig(
            use_florence=False,
            master_brain_model_name="master_brain_v2",
            top_k_final=5,
        )
        agent = CommerceAgent(settings, agent_config=config)
    """

    # ---- Model Selection ----
    # Name of the Master Brain model served by SGLang.
    # Change this to test different fine-tuned checkpoints.
    # Examples: "master_brain", "master_brain_sft_ep3", "qwen3.5-9b-chat"
    master_brain_model_name: str = "master_brain"

    # Name of the Handyman base model served by SGLang.
    # LoRA adapters are selected per-task automatically.
    handyman_model_name: str = "handyman"

    # ---- Florence-2 Toggle ----
    # True:  Use Florence-2 for image captioning (fine-grained tags).
    # False: Use Handyman VLM for captioning (fewer attributes, one fewer model).
    use_florence: bool = True

    # Florence model ID (only loaded if use_florence=True)
    florence_model_id: str = "microsoft/Florence-2-base"

    # ---- Web Search Toggle ----
    # True: enable Firecrawl web search pipeline
    # False: local catalog search only (even when frontend toggle is ON)
    use_web_search: bool = True

    # Firecrawl API key (only used if use_web_search=True)
    firecrawl_api_key: str = ""
    firecrawl_api_base: str = "https://api.firecrawl.dev/v1"

    # ---- Visual Verifier Toggle ----
    # True:  After image search, use Handyman LoRA-Verifier to filter mismatches.
    # False: Trust vector similarity results without VLM verification.
    use_visual_verifier: bool = True

    # Confidence threshold for the verifier (0.0-1.0)
    # Products scoring below this are filtered out.
    verifier_threshold: float = 0.5

    # ---- Retrieval Tuning ----
    # Number of initial candidates from retrieval
    top_k_initial: int = 50

    # Number of products after Handyman reranking
    top_k_reranked: int = 10

    # Number of products to present to the user
    top_k_final: int = 5

    # RRF k parameter (controls how aggressively we discount lower-ranked items)
    rrf_k: int = 60

    # ---- Memory Toggle ----
    # True:  Use memory recall (episodic + semantic + conversational).
    # False: Stateless — no memory context in prompts (faster, cleaner benchmarks).
    use_memory: bool = True

    # ---- Image & Data Storage ----
    # "local": serve from local disk (fast, for local demo / recording)
    # "gcs":   serve from Google Cloud Storage publicly (industry standard, for Vercel deployment)
    image_storage_provider: str = "local"

    # Local: base path for product images
    local_image_base_path: str = r"C:\Users\Caleb\Desktop\product_images"

    # Google Cloud Storage: public URL prefix for the bucket
    # Images served via: https://storage.googleapis.com/{gcs_bucket_name}/{path}
    gcs_public_url: str = "https://storage.googleapis.com/web-agent-data-caleb-2026"

    # ---- Latency/Debug ----
    # True:  Log timing for each pipeline stage to stdout.
    log_timing: bool = False

    # True:  Include metadata in SSE events (timing, source, model used).
    include_debug_metadata: bool = False

    def resolve_image_url(self, relative_path: str) -> str:
        """
        Resolve a product image path to a full URL based on the active storage provider.

        Args:
            relative_path: e.g. "amazon/amz_B001234.jpg" or "hm/hm_0108775015.jpg"

        Returns:
            Local:  "/api/images/amazon/amz_B001234.jpg"
            GCS:    "https://storage.googleapis.com/commerce-agent-images/amazon/amz_B001234.jpg"
        """
        if not relative_path:
            return ""

        if self.image_storage_provider == "gcs" and self.gcs_public_url:
            return f"{self.gcs_public_url.rstrip('/')}/{relative_path}"
        else:
            # Local: served via Next.js API route or static files
            return f"/api/images/{relative_path}"

    def to_benchmark_header(self) -> dict:
        """Export the config as a dict for benchmark logging."""
        return {
            "master_brain": self.master_brain_model_name,
            "use_florence": self.use_florence,
            "use_web_search": self.use_web_search,
            "use_visual_verifier": self.use_visual_verifier,
            "verifier_threshold": self.verifier_threshold,
            "top_k_initial": self.top_k_initial,
            "top_k_reranked": self.top_k_reranked,
            "top_k_final": self.top_k_final,
            "rrf_k": self.rrf_k,
            "use_memory": self.use_memory,
            "image_storage": self.image_storage_provider,
        }


# ================================================================
# Pipeline Stage Events (sent to frontend via SSE)
# ================================================================

class PipelineStage:
    """Stage events for the frontend status pipeline."""

    COLD_START = ("cold_start", "Loading models for cold start...")
    INTENT_DETECTION = ("intent_detection", "Understanding your request...")
    ANALYZING_IMAGE = ("analyzing_image", "Analyzing your image...")
    DECOMPOSING_QUERY = ("decomposing_query", "Breaking down your query...")
    SOURCING_LOCAL = ("sourcing_local", "Sourcing matches from catalog...")
    SOURCING_WEB = ("sourcing_web", "Searching the web via Firecrawl...")
    EXTRACTING_WEB = ("extracting_web", "Extracting product data from web results...")
    APPLYING_FILTERS = ("applying_filters", "Applying your preferences...")
    RERANKING = ("reranking", "Ranking products by relevance...")
    VERIFYING = ("verifying", "Verifying visual matches...")
    RECALLING_MEMORY = ("recalling_memory", "Remembering your preferences...")
    GENERATING = ("generating", "Generating your personalized recommendations...")
    COMPLETE = ("complete", "Done!")

    @staticmethod
    def to_sse(stage: tuple[str, str]) -> str:
        return json.dumps({"type": "status", "stage": stage[0], "message": stage[1]})


# ================================================================
# The Agent
# ================================================================

class CommerceAgent:
    """
    The single-agent orchestrator with a dual-brain architecture.

    Workflows:
    1. General Conversation → Handyman detects intent → Master Brain responds
    2. Text-Based Search   → Decompose → BGE-M3 retrieve → Rerank → Synthesize
    3. Image-Based Search  → Florence-2 / Handyman VLM tag → DINOv2+BGE-M3 → Verify → Synthesize
    4. Web Search          → Firecrawl → embed → RRF merge with local → Rerank → Synthesize
    """

    def __init__(self, config: dict, agent_config: AgentConfig | None = None):
        self.config = config
        self.ac = agent_config or AgentConfig()
        self._initialized = False
        self._stage_times: dict[str, float] = {}

        # Will be initialized lazily (cold start)
        self.handyman: Optional[HandymanRouter] = None
        self.master_brain: Optional[MasterBrain] = None
        self.semantic_embedder: Optional[BGEM3Embedder] = None
        self.visual_embedder: Optional[DINOv2Embedder] = None
        self.florence_tagger: Optional[Florence2Tagger] = None
        self.sqlite: Optional[SQLiteCatalog] = None
        self.lancedb: Optional[LanceDBCatalog] = None
        self.retriever: Optional[HybridRetriever] = None
        self.image_pipeline: Optional[ImageSearchPipeline] = None
        self.memory: Optional[MemoryManager] = None
        self.web_pipeline: Optional[WebSearchPipeline] = None

    def _log_stage(self, name: str, start: float):
        """Log timing for a pipeline stage if enabled."""
        elapsed = time.time() - start
        self._stage_times[name] = elapsed
        if self.ac.log_timing:
            print(f"  ⏱  {name}: {elapsed:.3f}s")

    def initialize(self):
        """
        Cold start — load all models and connect to databases.
        Called once on first request.
        """
        if self._initialized:
            return

        cfg = self.config
        ac = self.ac

        print(f"\n{'='*60}")
        print(f"  CommerceAgent Cold Start")
        print(f"  Master Brain:      {ac.master_brain_model_name}")
        print(f"  Use Florence:      {ac.use_florence}")
        print(f"  Use Web Search:    {ac.use_web_search}")
        print(f"  Use Verifier:      {ac.use_visual_verifier}")
        print(f"  Use Memory:        {ac.use_memory}")
        print(f"  Top-K Final:       {ac.top_k_final}")
        print(f"{'='*60}\n")

        # Lazy imports — heavy deps loaded at init time, not module import time
        from src.database import LanceDBCatalog, LanceDBMemoryStore, SQLiteCatalog
        from src.embeddings import BGEM3Embedder, DINOv2Embedder
        from src.image_search import Florence2Tagger, ImageSearchPipeline
        from src.master_brain import MasterBrain
        from src.memory import (
            ConversationalMemory,
            EpisodicMemory,
            MemoryManager,
            SemanticMemory,
        )
        from src.retrieval import HybridRetriever
        from src.router import HandymanRouter
        from src.web_search import FirecrawlSearcher, WebSearchPipeline

        # --- Databases ---
        self.sqlite = SQLiteCatalog(cfg["databases"]["sqlite"]["path"])
        self.lancedb = LanceDBCatalog(
            db_path=cfg["databases"]["lancedb"]["path"],
            table_name=cfg["databases"]["lancedb"]["table_name"],
            visual_dim=cfg["embeddings"]["visual"]["dimension"],
            semantic_dim=cfg["embeddings"]["semantic"]["dimension"],
        )

        # --- Embedding Models ---
        self.semantic_embedder = BGEM3Embedder(
            model_id=cfg["embeddings"]["semantic"]["model_id"],
            batch_size=cfg["embeddings"]["semantic"]["batch_size"],
            use_fp16=cfg["embeddings"]["semantic"]["use_fp16"],
        )
        self.visual_embedder = DINOv2Embedder(
            model_id=cfg["embeddings"]["visual"]["model_id"],
            batch_size=cfg["embeddings"]["visual"]["batch_size"],
        )

        # --- Florence-2 (conditional) ---
        if ac.use_florence:
            print("  Loading Florence-2...")
            self.florence_tagger = Florence2Tagger(
                model_id=ac.florence_model_id,
            )
        else:
            print("  Skipping Florence-2 (use_florence=False, Handyman VLM will caption)")
            self.florence_tagger = None

        # --- SGLang Model Endpoints ---
        sglang_cfg = cfg["inference"]["sglang"]
        self.handyman = HandymanRouter(
            api_base=f"http://localhost:{sglang_cfg['handyman_port']}/v1",
        )
        self.master_brain = MasterBrain(
            api_base=f"http://localhost:{sglang_cfg['master_brain_port']}/v1",
            model_name=ac.master_brain_model_name,
        )

        # --- Retrieval ---
        self.retriever = HybridRetriever(
            sqlite_catalog=self.sqlite,
            lancedb_catalog=self.lancedb,
            semantic_embedder=self.semantic_embedder,
            visual_embedder=self.visual_embedder,
            handyman=self.handyman,
            top_k_initial=ac.top_k_initial,
            top_k_reranked=ac.top_k_reranked,
            rrf_k=ac.rrf_k,
        )

        # --- Image Search Pipeline ---
        self.image_pipeline = ImageSearchPipeline(
            florence_tagger=self.florence_tagger,
            visual_embedder=self.visual_embedder,
            semantic_embedder=self.semantic_embedder,
            hybrid_retriever=self.retriever,
        )

        # --- Web Search Pipeline (conditional) ---
        if ac.use_web_search and ac.firecrawl_api_key:
            firecrawl = FirecrawlSearcher(
                api_key=ac.firecrawl_api_key,
                api_base=ac.firecrawl_api_base,
            )
            self.web_pipeline = WebSearchPipeline(
                firecrawl=firecrawl,
                semantic_embedder=self.semantic_embedder,
                visual_embedder=self.visual_embedder,
            )
            print("  Web search pipeline enabled (Firecrawl)")
        else:
            self.web_pipeline = None
            if ac.use_web_search and not ac.firecrawl_api_key:
                print("  ⚠️  Web search enabled but no API key — web search disabled")
            else:
                print("  Web search pipeline disabled")

        # --- Memory (conditional) ---
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
        else:
            self.memory = None
            print("  Memory disabled (stateless mode)")

        self._initialized = True
        print("\n✅ CommerceAgent initialized.\n")

    # ================================================================
    # Main Entry Point
    # ================================================================

    async def handle_message(
        self,
        user_id: str,
        session_id: str,
        message: str,
        image_bytes: Optional[bytes] = None,
        web_search_enabled: bool = False,
    ) -> AsyncIterator[str]:
        """
        Main entry point — handles a user message and yields SSE events.

        Args:
            web_search_enabled: True when the frontend toggle is ON
        """
        self._stage_times = {}

        # Cold start if needed
        if not self._initialized:
            yield PipelineStage.to_sse(PipelineStage.COLD_START)
            self.initialize()

        # Record user message in memory
        if self.memory:
            user_msg = ChatMessage(
                role="user",
                content=message,
                timestamp=time.time(),
            )
            self.memory.on_user_message(session_id, user_msg)

        # --- Step 1: Intent Detection ---
        t0 = time.time()
        yield PipelineStage.to_sse(PipelineStage.INTENT_DETECTION)
        has_image = image_bytes is not None
        intent = self.handyman.detect_intent(message, has_image=has_image)
        self._log_stage("intent_detection", t0)

        # --- Step 2: Route to workflow ---
        if intent == IntentType.GENERAL_TALK:
            async for event in self._workflow_general(user_id, session_id, message):
                yield event

        elif intent in (IntentType.TEXT_SEARCH, IntentType.WEB_SEARCH):
            async for event in self._workflow_text_search(
                user_id, session_id, message,
                include_web=web_search_enabled,
            ):
                yield event

        elif intent == IntentType.IMAGE_SEARCH:
            async for event in self._workflow_image_search(
                user_id, session_id, message, image_bytes,
                include_web=web_search_enabled,
            ):
                yield event

        # Emit debug metadata if enabled
        if self.ac.include_debug_metadata:
            yield json.dumps({
                "type": "debug",
                "config": self.ac.to_benchmark_header(),
                "timings": self._stage_times,
            })

        yield json.dumps({"type": "done"})

    # ================================================================
    # Workflow: General Conversation
    # ================================================================

    async def _workflow_general(
        self,
        user_id: str,
        session_id: str,
        message: str,
    ) -> AsyncIterator[str]:
        """General conversation workflow — no product search needed."""
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
        async for token in self.master_brain.general_chat_stream(
            message, chat_history, memory_context
        ):
            full_response += token
            yield json.dumps({"type": "token", "content": token})
        self._log_stage("generation", t0)

        if self.memory:
            assistant_msg = ChatMessage(role="assistant", content=full_response, timestamp=time.time())
            self.memory.on_assistant_message(session_id, assistant_msg)

    # ================================================================
    # Workflow: Text Search (+ optional Web Search)
    # ================================================================

    async def _workflow_text_search(
        self,
        user_id: str,
        session_id: str,
        message: str,
        include_web: bool = False,
    ) -> AsyncIterator[str]:
        """Text-based recommendation workflow with optional web search fusion."""
        # Decompose query
        t0 = time.time()
        yield PipelineStage.to_sse(PipelineStage.DECOMPOSING_QUERY)
        query = self.handyman.decompose_query(message)
        self._log_stage("decomposition", t0)

        # Local retrieval
        t0 = time.time()
        yield PipelineStage.to_sse(PipelineStage.SOURCING_LOCAL)
        local_products = self.retriever.search_text(query)
        for p in local_products:
            p["source"] = "local"
        self._log_stage("local_retrieval", t0)

        # Web retrieval (if toggle ON and pipeline available)
        web_products = []
        if include_web and self.web_pipeline:
            t0 = time.time()
            yield PipelineStage.to_sse(PipelineStage.SOURCING_WEB)
            web_result = self.web_pipeline.search(
                query=query.rewritten_query or message,
                num_results=self.ac.top_k_initial,
            )
            web_products = web_result["products"]
            for p in web_products:
                p["source"] = "web"
            self._log_stage("web_retrieval", t0)

        # Merge and rerank the combined candidate pool
        t0 = time.time()
        yield PipelineStage.to_sse(PipelineStage.RERANKING)
        combined = local_products + web_products
        reranked = self.handyman.rerank(message, combined, top_k=self.ac.top_k_final)
        self._log_stage("reranking", t0)

        # Memory context
        memory_context = ""
        chat_history = []
        if self.memory:
            yield PipelineStage.to_sse(PipelineStage.RECALLING_MEMORY)
            query_emb = self.semantic_embedder.embed_query(message)
            memory_context = self.memory.get_memory_context(user_id, query_emb.tolist())
            chat_history = self.memory.get_chat_history(session_id)

        # Send product data to frontend
        yield json.dumps({
            "type": "products",
            "items": reranked[:self.ac.top_k_final],
            "tags": query.tags,
            "filters": query.filters,
        })

        # Synthesize recommendation
        t0 = time.time()
        yield PipelineStage.to_sse(PipelineStage.GENERATING)
        full_response = ""
        async for token in self.master_brain.synthesize_stream(
            message, reranked, chat_history, memory_context
        ):
            full_response += token
            yield json.dumps({"type": "token", "content": token})
        self._log_stage("generation", t0)

        if self.memory:
            assistant_msg = ChatMessage(role="assistant", content=full_response, timestamp=time.time())
            self.memory.on_assistant_message(session_id, assistant_msg)

    # ================================================================
    # Workflow: Image Search (+ optional Verification + Web)
    # ================================================================

    async def _workflow_image_search(
        self,
        user_id: str,
        session_id: str,
        message: str,
        image_bytes: Optional[bytes] = None,
        include_web: bool = False,
    ) -> AsyncIterator[str]:
        """Image-based search workflow with Florence/Handyman and optional verification."""
        t0 = time.time()
        yield PipelineStage.to_sse(PipelineStage.ANALYZING_IMAGE)

        # Extract filters from text (if any)
        filters = {}
        if message:
            query = self.handyman.decompose_query(message)
            filters = query.filters

        # Image captioning — Florence-2 or Handyman VLM
        if self.ac.use_florence and self.florence_tagger:
            # Use dedicated Florence-2 model
            result = self.image_pipeline.search(
                image_source=image_bytes,
                user_text=message,
                filters=filters,
            )
            caption = result["caption"]
            tags = result["tags"]
        else:
            # Use Handyman VLM for captioning (no Florence-2 loaded)
            # Handyman generates a caption via its base VLM capabilities
            caption = self.handyman._chat_completion(
                system_prompt="Describe this product image in detail. Include: item type, color, material, style, brand if visible. Output a concise description.",
                user_message=[
                    {"type": "text", "text": "Describe this product image:"},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},  # Will be replaced with actual bytes
                ],
                model=self.handyman.BASE_MODEL,
                max_tokens=200,
            )
            tags = caption.split(", ") if caption else []
            # Run DINOv2 + BGE-M3 search manually
            result = self.retriever.search_image(
                image_bytes=image_bytes,
                text_query=caption,
                filters=filters,
            )

        products = result.get("products", [])
        for p in products:
            p["source"] = "local"
        self._log_stage("image_analysis", t0)

        # Local results done — now optionally search the web too
        yield PipelineStage.to_sse(PipelineStage.SOURCING_LOCAL)
        web_products = []
        if include_web and self.web_pipeline and caption:
            t0 = time.time()
            yield PipelineStage.to_sse(PipelineStage.SOURCING_WEB)
            web_result = self.web_pipeline.search(
                query=caption,
                num_results=self.ac.top_k_initial,
                include_visual=True,
            )
            web_products = web_result["products"]
            for p in web_products:
                p["source"] = "web"
            self._log_stage("web_image_retrieval", t0)

        # Merge local + web
        combined = products + web_products

        # Visual Verifier (conditional)
        if self.ac.use_visual_verifier and image_bytes:
            t0 = time.time()
            yield PipelineStage.to_sse(PipelineStage.VERIFYING)
            candidate_urls = []
            for p in combined:
                urls = p.get("image_urls", "").split(",")
                candidate_urls.append(urls[0].strip() if urls and urls[0].strip() else "")

            valid_urls = [u for u in candidate_urls if u]
            if valid_urls:
                # Verifier checks: does each candidate visually match the query image?
                verifier_results = self.handyman.verify_image_match(
                    query_image_url="<user_uploaded>",  # Passed as bytes in real impl
                    candidate_image_urls=valid_urls,
                    threshold=self.ac.verifier_threshold,
                )
                # Filter out mismatches
                verified_urls = {
                    r["url"] for r in verifier_results
                    if r["match"] or r["confidence"] >= self.ac.verifier_threshold
                }
                combined = [
                    p for p in combined
                    if not p.get("image_urls") or
                    p["image_urls"].split(",")[0].strip() in verified_urls or
                    p["image_urls"].split(",")[0].strip() == ""
                ]
            self._log_stage("visual_verification", t0)

        # Rerank the combined pool
        t0 = time.time()
        yield PipelineStage.to_sse(PipelineStage.RERANKING)
        reranked = self.handyman.rerank(
            caption or message, combined, top_k=self.ac.top_k_final
        )
        self._log_stage("reranking", t0)

        # Memory context
        memory_context = ""
        chat_history = []
        if self.memory:
            yield PipelineStage.to_sse(PipelineStage.RECALLING_MEMORY)
            query_emb = self.semantic_embedder.embed_query(caption or message)
            memory_context = self.memory.get_memory_context(user_id, query_emb.tolist())
            chat_history = self.memory.get_chat_history(session_id)

        # Send products + tags
        yield json.dumps({
            "type": "products",
            "items": reranked[:self.ac.top_k_final],
            "tags": tags,
            "caption": caption,
        })

        # Synthesize
        t0 = time.time()
        yield PipelineStage.to_sse(PipelineStage.GENERATING)
        synth_query = f"The user uploaded an image that shows: {caption}."
        if message:
            synth_query += f" They also said: {message}"

        full_response = ""
        async for token in self.master_brain.synthesize_stream(
            synth_query, reranked, chat_history, memory_context
        ):
            full_response += token
            yield json.dumps({"type": "token", "content": token})
        self._log_stage("generation", t0)

        if self.memory:
            assistant_msg = ChatMessage(role="assistant", content=full_response, timestamp=time.time())
            self.memory.on_assistant_message(session_id, assistant_msg)

    # ================================================================
    # Session Management
    # ================================================================

    async def end_session(self, user_id: str, session_id: str):
        """
        End a session — summarize and store in ConversationalMemory.
        Called when the user closes the chat or after inactivity.
        """
        if not self.memory:
            return

        full_history = self.memory.episodic.get_full_conversation(session_id)
        if not full_history or len(full_history) < 2:
            self.memory.episodic.clear_session(session_id)
            return

        summary = self.master_brain.summarize_conversation(full_history)
        summary_embedding = self.semantic_embedder.embed_query(summary)

        key_products = []
        key_topics = []

        self.memory.end_session(
            user_id=user_id,
            session_id=session_id,
            summary=summary,
            key_products=key_products,
            key_topics=key_topics,
            summary_embedding=summary_embedding.tolist(),
        )
