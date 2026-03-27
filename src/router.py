"""
router.py — The Handyman (Qwen 3.5-0.8B) with Multi-LoRA
==========================================================
Single base model, three task-specific LoRA adapters:
- LoRA-Router: Intent detection (general_talk | text_search | image_search | web_search)
              + Query decomposition (tags, filters, rewritten_query)
- LoRA-Reranker: Product relevance scoring
- LoRA-Verifier: Visual product match verification
"""

from __future__ import annotations

import json
from typing import Optional

import httpx

from src.schema import DecomposedQuery, IntentType


# ================================================================
# System Prompts
# ================================================================

INTENT_SYSTEM_PROMPT = """You are a commerce intent classifier. Given a user message, classify the intent into exactly one of:
- "general_talk": casual conversation, greetings, questions about the assistant
- "text_search": the user wants to find/buy a product using text description
- "image_search": the user has uploaded an image and wants to find similar products
- "web_search": the user explicitly asks to search the web, or wants items not in a typical catalog

Respond with ONLY a JSON object: {"intent": "<intent_type>"}"""

DECOMPOSITION_SYSTEM_PROMPT = """You are a commerce query decomposition engine. Given a user's shopping query, extract:
1. **tags**: semantic descriptors (e.g., "warm", "casual", "winter", "elegant", "breathable")
2. **filters**: hard constraints as key-value pairs:
   - price_max: maximum price in USD (number)
   - price_min: minimum price in USD (number)  
   - brand: specific brand name (string)
   - color: specific color (string)
   - category: product category (string)
   - size: size specification (string)
3. **rewritten_query**: a clean, search-optimized version of the query

Respond with ONLY a JSON object:
{
  "tags": ["tag1", "tag2"],
  "filters": {"price_max": 200, "color": "red"},
  "rewritten_query": "red winter jacket under $200"
}"""


class HandymanRouter:
    """
    The 0.8B Handyman with multi-LoRA serving.
    All calls hit the same SGLang server; per-request LoRA selection
    via the 'model' field in the API call.
    """

    # LoRA adapter names (must match --lora-names in 08_launch_sglang.py)
    ROUTER_MODEL = "handyman-router"
    RERANKER_MODEL = "handyman-reranker"
    VERIFIER_MODEL = "handyman-verifier"
    BASE_MODEL = "handyman"  # Fallback if no LoRAs loaded

    def __init__(
        self,
        api_base: str = "http://localhost:30000/v1",
        timeout: float = 5.0,
    ):
        self.api_base = api_base.rstrip("/")
        self.timeout = timeout
        self.client = httpx.Client(timeout=timeout)

    def _chat_completion(
        self,
        system_prompt: str,
        user_message,
        temperature: float = 0.1,
        model: str = None,
        max_tokens: int = 256,
    ) -> str:
        """Call the SGLang endpoint with a specific LoRA adapter."""
        response = self.client.post(
            f"{self.api_base}/chat/completions",
            json={
                "model": model or self.BASE_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    def detect_intent(self, user_message: str, has_image: bool = False) -> IntentType:
        """
        Classify the user's intent.
        If an image is attached, shortcut to image_search.
        """
        if has_image:
            return IntentType.IMAGE_SEARCH

        try:
            raw = self._chat_completion(
                INTENT_SYSTEM_PROMPT, user_message, model=self.ROUTER_MODEL
            )
            parsed = json.loads(raw)
            intent_str = parsed.get("intent", "general_talk")
            return IntentType(intent_str)
        except (json.JSONDecodeError, ValueError):
            return IntentType.GENERAL_TALK

    def decompose_query(self, user_message: str) -> DecomposedQuery:
        """
        Break a shopping query into structured tags and filters.
        Only called when intent is text_search or image_search.
        """
        try:
            raw = self._chat_completion(
                DECOMPOSITION_SYSTEM_PROMPT, user_message, model=self.ROUTER_MODEL
            )
            parsed = json.loads(raw)

            return DecomposedQuery(
                intent=IntentType.TEXT_SEARCH,
                original_query=user_message,
                tags=parsed.get("tags", []),
                filters=parsed.get("filters", {}),
                rewritten_query=parsed.get("rewritten_query", user_message),
            )
        except (json.JSONDecodeError, ValueError):
            # Fallback: use the original query as-is
            return DecomposedQuery(
                intent=IntentType.TEXT_SEARCH,
                original_query=user_message,
                tags=[],
                filters={},
                rewritten_query=user_message,
            )

    def rerank(self, query: str, products: list[dict], top_k: int = 10) -> list[dict]:
        """
        Use the Handyman as a lightweight reranker.
        Takes Top-50 candidates and selects Top-10.
        """
        # Build a concise product summary for the model
        product_summaries = []
        for i, p in enumerate(products):
            summary = f"[{i}] {p.get('title', '')} | ${p.get('price', 'N/A')} | {p.get('brand', '')}"
            product_summaries.append(summary)

        rerank_prompt = f"""Given the search query: "{query}"

Rank the following products by relevance. Return ONLY a JSON array of the top {top_k} indices, most relevant first.

Products:
{chr(10).join(product_summaries)}

Response format: [3, 7, 1, ...]"""

        try:
            raw = self._chat_completion(
                "You are a product relevance ranker. Return only a JSON array of indices.",
                rerank_prompt,
                temperature=0.1,
                model=self.RERANKER_MODEL,
            )
            indices = json.loads(raw)
            reranked = []
            for idx in indices[:top_k]:
                if 0 <= idx < len(products):
                    reranked.append(products[idx])
            return reranked
        except (json.JSONDecodeError, IndexError):
            return products[:top_k]

    def verify_image_match(
        self,
        query_image_url: str,
        candidate_image_urls: list[str],
        threshold: float = 0.5,
    ) -> list[dict]:
        """
        Visual Verifier — uses LoRA-Verifier to compare query image
        against candidate product images.
        Returns list of {url, match, confidence} dicts.
        """
        results = []
        for candidate_url in candidate_image_urls:
            try:
                user_content = [
                    {"type": "text", "text": "Do these two product images show the same or very similar product?"},
                    {"type": "image_url", "image_url": {"url": query_image_url}},
                    {"type": "image_url", "image_url": {"url": candidate_url}},
                ]
                raw = self._chat_completion(
                    "You are a visual product matching model. Respond with JSON: {\"match\": true/false, \"confidence\": 0.0-1.0}",
                    user_content,
                    temperature=0.1,
                    model=self.VERIFIER_MODEL,
                    max_tokens=128,
                )
                parsed = json.loads(raw)
                results.append({
                    "url": candidate_url,
                    "match": parsed.get("match", False),
                    "confidence": parsed.get("confidence", 0.0),
                })
            except Exception:
                results.append({"url": candidate_url, "match": True, "confidence": 0.5})

        return results

    def close(self):
        self.client.close()
