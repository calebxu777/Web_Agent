"""
mock.py — Mock Implementations for Local MVP Testing
======================================================
Lightweight stand-ins for every heavy component so the full
API pipeline (FastAPI ↔ SSE ↔ Next.js) can run on a laptop
without SGLang, Redis, LanceDB, DINOv2, BGE-M3, or Florence-2.

Usage:
    In api.py, set AgentConfig(mock_mode=True).
    CommerceAgent.initialize() will use these mocks instead of real models.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from typing import AsyncIterator, Optional

from src.schema import ChatMessage, DecomposedQuery, IntentType


# ================================================================
# Mock Products — Realistic catalog entries
# ================================================================

MOCK_PRODUCTS = [
    {
        "product_id": "mock_001",
        "title": "Alpine Frost Insulated Parka",
        "description": "Premium down-insulated parka with waterproof shell, faux fur-trimmed hood, and fleece-lined pockets. Perfect for harsh winter conditions.",
        "category": "Clothing & Accessories",
        "subcategory": "Outerwear",
        "price": 189.99,
        "brand": "NorthPeak",
        "image_urls": "",
        "rating": 4.7,
        "review_count": 1243,
        "reviews_summary": "Customers love the warmth and build quality. Some note it runs slightly large.",
        "in_stock": 1,
        "source": "amazon",
    },
    {
        "product_id": "mock_002",
        "title": "Heritage Leather Crossbody Bag",
        "description": "Full-grain Italian leather crossbody with adjustable strap, brass hardware, and organized interior with RFID-blocking pocket.",
        "category": "Clothing & Accessories",
        "subcategory": "Bags",
        "price": 145.00,
        "brand": "Marcello",
        "image_urls": "",
        "rating": 4.5,
        "review_count": 876,
        "reviews_summary": "Beautiful leather that ages well. Compact but fits essentials perfectly.",
        "in_stock": 1,
        "source": "amazon",
    },
    {
        "product_id": "mock_003",
        "title": "CloudStep Running Shoes",
        "description": "Responsive foam midsole with engineered mesh upper. Lightweight at 8.2oz with reflective accents for night visibility.",
        "category": "Clothing & Accessories",
        "subcategory": "Athletic Shoes",
        "price": 129.95,
        "brand": "StrideX",
        "image_urls": "",
        "rating": 4.8,
        "review_count": 2156,
        "reviews_summary": "Incredibly comfortable for long runs. The arch support is excellent.",
        "in_stock": 1,
        "source": "amazon",
    },
    {
        "product_id": "mock_004",
        "title": "ProNoise ANC Wireless Earbuds",
        "description": "Active noise cancelling wireless earbuds with 32-hour battery life, spatial audio, and IPX5 water resistance.",
        "category": "Electronics",
        "subcategory": "Audio",
        "price": 79.99,
        "brand": "SoundCore",
        "image_urls": "",
        "rating": 4.6,
        "review_count": 3421,
        "reviews_summary": "Best ANC in this price range. Call quality could be better in wind.",
        "in_stock": 1,
        "source": "amazon",
    },
    {
        "product_id": "mock_005",
        "title": "Minimalist Ceramic Pour-Over Set",
        "description": "Handcrafted ceramic dripper with thermal carafe, reusable stainless steel filter, and bamboo stand.",
        "category": "Home & Kitchen",
        "subcategory": "Coffee & Tea",
        "price": 64.50,
        "brand": "BrewCraft",
        "image_urls": "",
        "rating": 4.9,
        "review_count": 542,
        "reviews_summary": "Restaurant-quality pour-over at home. The ceramic retains heat beautifully.",
        "in_stock": 1,
        "source": "hm",
    },
    {
        "product_id": "mock_006",
        "title": "Merino Wool Crew Neck Sweater",
        "description": "Ultra-soft 100% merino wool sweater with ribbed cuffs and hem. Machine washable. Available in 12 colors.",
        "category": "Clothing & Accessories",
        "subcategory": "Sweaters",
        "price": 89.00,
        "brand": "Everloom",
        "image_urls": "",
        "rating": 4.4,
        "review_count": 987,
        "reviews_summary": "Soft, warm, and doesn't pill. True to size. Great layering piece.",
        "in_stock": 1,
        "source": "hm",
    },
    {
        "product_id": "mock_007",
        "title": "Smart LED Desk Lamp Pro",
        "description": "Adjustable color temperature (2700K-6500K), dimmable, wireless charging base, USB-C port, and memory function.",
        "category": "Home & Kitchen",
        "subcategory": "Lighting",
        "price": 54.99,
        "brand": "LumiTech",
        "image_urls": "",
        "rating": 4.7,
        "review_count": 1567,
        "reviews_summary": "Perfect for home office. The wireless charging base is a nice touch.",
        "in_stock": 1,
        "source": "amazon",
    },
]


# ================================================================
# Mock Handyman Router
# ================================================================

class MockHandymanRouter:
    """
    Keyword-based intent detection and hardcoded query decomposition.
    Mimics the real Handyman's API surface exactly.
    """

    ROUTER_MODEL = "handyman-router"
    RERANKER_MODEL = "handyman-reranker"
    VERIFIER_MODEL = "handyman-verifier"
    BASE_MODEL = "handyman"

    def detect_intent(self, user_message: str, has_image: bool = False) -> IntentType:
        if has_image:
            return IntentType.IMAGE_SEARCH

        msg = user_message.lower()

        # Web search indicators
        web_keywords = ["search the web", "look online", "google", "web search"]
        if any(kw in msg for kw in web_keywords):
            return IntentType.WEB_SEARCH

        # Product search indicators
        search_keywords = [
            "find", "looking for", "recommend", "suggest", "show me",
            "buy", "shop", "need", "want", "best", "top", "cheap",
            "under $", "jacket", "shoes", "bag", "headphones", "laptop",
            "dress", "shirt", "pants", "watch", "phone", "camera",
            "similar", "alternative", "compare",
        ]
        if any(kw in msg for kw in search_keywords):
            return IntentType.TEXT_SEARCH

        return IntentType.GENERAL_TALK

    def decompose_query(self, user_message: str) -> DecomposedQuery:
        msg = user_message.lower()

        # Extract tags from keywords
        tag_map = {
            "warm": "warm", "cozy": "cozy", "winter": "winter",
            "summer": "summer", "casual": "casual", "formal": "formal",
            "lightweight": "lightweight", "waterproof": "waterproof",
            "breathable": "breathable", "elegant": "elegant",
            "minimalist": "minimalist", "vintage": "vintage",
        }
        tags = [v for k, v in tag_map.items() if k in msg]

        # Extract filters
        filters = {}
        import re
        price_match = re.search(r"under\s*\$?\s*(\d+)", msg)
        if price_match:
            filters["price_max"] = float(price_match.group(1))
        price_min_match = re.search(r"over\s*\$?\s*(\d+)", msg)
        if price_min_match:
            filters["price_min"] = float(price_min_match.group(1))

        # Brand extraction
        brands = ["nike", "adidas", "apple", "samsung", "sony", "northpeak", "everloom"]
        for brand in brands:
            if brand in msg:
                filters["brand"] = brand.capitalize()
                break

        return DecomposedQuery(
            intent=IntentType.TEXT_SEARCH,
            original_query=user_message,
            tags=tags or ["general"],
            filters=filters,
            rewritten_query=user_message,
        )

    def rerank(self, query: str, products: list[dict], top_k: int = 10) -> list[dict]:
        """Passthrough — trust the retriever ordering in mock mode."""
        return products[:top_k]

    def verify_image_match(self, query_image_url, candidate_image_urls, threshold=0.5):
        return [
            {"url": url, "match": True, "confidence": 0.85}
            for url in candidate_image_urls
        ]

    def close(self):
        pass


# ================================================================
# Mock Master Brain
# ================================================================

MOCK_GENERAL_RESPONSES = [
    "Hey there! 👋 I'm your shopping assistant. I can help you find the perfect products, "
    "compare options, or just chat about what you're looking for. What's on your mind today?",

    "Happy to help! Whether you're looking for fashion, electronics, home goods, or anything "
    "else — just describe what you need and I'll find the best options for you.",

    "Great question! I'm here to make your shopping experience as smooth as possible. "
    "I can search our catalog, compare products, remember your preferences, and give you "
    "honest recommendations. Try asking me to find something specific!",
]


class MockMasterBrain:
    """
    Streams realistic shopping-assistant text token-by-token.
    Mimics the real MasterBrain's full API surface.
    """

    def __init__(self, **kwargs):
        self.model_name = kwargs.get("model_name", "master_brain")

    def _build_messages(self, *args, **kwargs):
        return []

    def synthesize(self, user_query, products, chat_history=None, memory_context=""):
        return self._generate_recommendation(user_query, products)

    async def synthesize_stream(
        self,
        user_query: str,
        products: list[dict],
        chat_history: list[ChatMessage] = None,
        memory_context: str = "",
    ) -> AsyncIterator[str]:
        response = self._generate_recommendation(user_query, products)
        # Stream token-by-token with realistic delays
        words = response.split(" ")
        for i, word in enumerate(words):
            token = word if i == 0 else " " + word
            yield token
            await asyncio.sleep(random.uniform(0.01, 0.04))

    def general_chat(self, user_message, chat_history=None, memory_context=""):
        return random.choice(MOCK_GENERAL_RESPONSES)

    async def general_chat_stream(
        self,
        user_message: str,
        chat_history: list[ChatMessage] = None,
        memory_context: str = "",
    ) -> AsyncIterator[str]:
        response = random.choice(MOCK_GENERAL_RESPONSES)
        words = response.split(" ")
        for i, word in enumerate(words):
            token = word if i == 0 else " " + word
            yield token
            await asyncio.sleep(random.uniform(0.01, 0.04))

    def summarize_conversation(self, messages):
        return "Mock conversation summary: user discussed product preferences."

    def _generate_recommendation(self, query: str, products: list[dict]) -> str:
        if not products:
            return (
                "I searched through our catalog but couldn't find an exact match for that. "
                "Could you tell me more about what you're looking for? For example, any "
                "specific brand, price range, or features that are important to you?"
            )

        # Build a natural recommendation from the products
        parts = []
        parts.append(
            f"Great question! I found some excellent options that match what you're looking for. "
            f"Here are my top picks:\n\n"
        )

        for i, p in enumerate(products[:3], 1):
            title = p.get("title", "Unknown Product")
            price = p.get("price")
            rating = p.get("rating")
            summary = p.get("reviews_summary", "")
            brand = p.get("brand", "")

            parts.append(f"**{i}. {title}**")
            if brand:
                parts.append(f"by {brand}")
            if price:
                parts.append(f" — ${price:.2f}")
            parts.append("\n")
            if rating:
                parts.append(f"⭐ {rating}/5")
                if p.get("review_count"):
                    parts.append(f" ({p['review_count']:,} reviews)")
                parts.append("\n")
            if summary:
                parts.append(f"{summary}\n")
            parts.append("\n")

        parts.append(
            "Would you like me to dive deeper into any of these, or would you like me to "
            "refine the search with different criteria? I can also look for alternatives "
            "in a different price range or style."
        )

        return "".join(parts)

    def close(self):
        pass

    async def aclose(self):
        pass


# ================================================================
# Mock Retriever
# ================================================================

class MockRetriever:
    """Returns filtered mock products. Mimics HybridRetriever's API."""

    def search_text(self, query: DecomposedQuery) -> list[dict]:
        products = list(MOCK_PRODUCTS)

        # Apply basic filters
        if query.filters.get("price_max"):
            products = [p for p in products if (p.get("price") or 0) <= query.filters["price_max"]]
        if query.filters.get("price_min"):
            products = [p for p in products if (p.get("price") or 0) >= query.filters["price_min"]]
        if query.filters.get("brand"):
            brand = query.filters["brand"].lower()
            products = [p for p in products if brand in p.get("brand", "").lower()]
        if query.filters.get("category"):
            cat = query.filters["category"].lower()
            products = [p for p in products if cat in p.get("category", "").lower()]

        # If filters eliminated everything, return shuffled full set
        if not products:
            products = list(MOCK_PRODUCTS)
            random.shuffle(products)

        return products[:5]

    def search_visual(self, image_embedding, text_query=None, filters=None):
        products = list(MOCK_PRODUCTS)
        random.shuffle(products)
        return products[:5]

    def search_hybrid(self, text_query, image_embedding=None, tags=None, filters=None):
        products = list(MOCK_PRODUCTS)
        random.shuffle(products)
        return products[:5]

    def search_image(self, image_bytes=None, text_query=None, filters=None):
        products = list(MOCK_PRODUCTS)
        random.shuffle(products)
        return {"products": products[:5]}


# ================================================================
# Mock Embedder
# ================================================================

class MockEmbedder:
    """
    Replaces both BGEM3Embedder and DINOv2Embedder.
    Returns random vectors of the right dimension.
    """

    def __init__(self, dimension: int = 1024):
        self.dimension = dimension
        import numpy as np
        self._np = np

    def embed_query(self, text: str):
        vec = self._np.random.randn(self.dimension).astype("float32")
        vec /= self._np.linalg.norm(vec)
        return vec

    def embed_single(self, text_or_url: str):
        return self.embed_query(text_or_url)

    def embed_batch(self, items: list, show_progress=True, **kwargs):
        vecs = self._np.random.randn(len(items), self.dimension).astype("float32")
        norms = self._np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1
        vecs /= norms
        return {"dense": vecs}


# ================================================================
# Mock Memory Manager
# ================================================================

class MockMemoryManager:
    """
    In-memory dict-based memory. No Redis or LanceDB needed.
    Mimics MemoryManager's full API surface.
    """

    def __init__(self):
        self._sessions: dict[str, list[ChatMessage]] = {}

    def on_user_message(self, session_id: str, message: ChatMessage):
        if session_id not in self._sessions:
            self._sessions[session_id] = []
        self._sessions[session_id].append(message)

    def on_assistant_message(self, session_id: str, message: ChatMessage):
        if session_id not in self._sessions:
            self._sessions[session_id] = []
        self._sessions[session_id].append(message)

    def get_chat_history(self, session_id: str) -> list[ChatMessage]:
        return self._sessions.get(session_id, [])[-20:]

    def get_memory_context(self, user_id: str, query_embedding: list[float]) -> str:
        return ""

    def end_session(self, **kwargs):
        session_id = kwargs.get("session_id", "")
        self._sessions.pop(session_id, None)

    @property
    def episodic(self):
        return self

    def get_full_conversation(self, session_id):
        return self._sessions.get(session_id, [])

    def clear_session(self, session_id):
        self._sessions.pop(session_id, None)
