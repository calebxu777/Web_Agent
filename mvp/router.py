"""
router.py - MVP LLM router/reranker
===================================
Uses an OpenAI-compatible chat completion API for:
- intent detection
- query decomposition
- product reranking
- optional vision-assisted image matching
"""

from __future__ import annotations

import json
import re

import httpx

from src.schema import DecomposedQuery, IntentType


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


class MVPRouter:
    """LLM-backed router/reranker for the MVP backend."""

    def __init__(
        self,
        api_base: str = "https://api.openai.com/v1",
        api_key: str = "",
        model_name: str = "gpt-4o-mini",
        reranker_model_name: str = "",
        verifier_model_name: str = "",
        timeout: float = 30.0,
    ):
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model_name = model_name
        self.reranker_model_name = reranker_model_name or model_name
        self.verifier_model_name = verifier_model_name or model_name
        self.client = httpx.Client(timeout=timeout)

    @staticmethod
    def _heuristic_intent(user_message: str) -> IntentType | None:
        message = user_message.strip().lower()
        if not message:
            return IntentType.GENERAL_TALK

        web_patterns = [
            "search the web",
            "search online",
            "on the web",
            "online only",
            "from the internet",
            "google shopping",
            "google for",
            "search google",
        ]
        if any(pattern in message for pattern in web_patterns):
            return IntentType.WEB_SEARCH

        search_patterns = [
            "find me",
            "show me",
            "looking for",
            "search for",
            "recommend",
            "need a",
            "need an",
            "need some",
            "shop for",
            "buy",
            "under $",
            "under ",
            "less than",
            "within my budget",
            "similar products",
            "jacket",
            "dress",
            "shoes",
            "shirt",
            "pants",
            "coat",
            "sneakers",
            "bag",
        ]
        if any(pattern in message for pattern in search_patterns):
            return IntentType.TEXT_SEARCH

        if re.search(r"\$\s*\d+", message) or re.search(r"\b\d+\s*dollars?\b", message):
            return IntentType.TEXT_SEARCH

        return None

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    @staticmethod
    def _extract_json(text: str):
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        return json.loads(text.strip())

    def _chat_completion(
        self,
        system_prompt: str,
        user_message,
        model: str = "",
        temperature: float = 0.1,
        max_tokens: int = 256,
    ) -> str:
        response = self.client.post(
            f"{self.api_base}/chat/completions",
            headers=self._headers(),
            json={
                "model": model or self.model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def detect_intent(self, user_message: str, has_image: bool = False) -> IntentType:
        if has_image:
            return IntentType.IMAGE_SEARCH

        heuristic = self._heuristic_intent(user_message)
        if heuristic is not None:
            return heuristic

        try:
            raw = self._chat_completion(
                INTENT_SYSTEM_PROMPT,
                user_message,
                model=self.model_name,
                max_tokens=64,
            )
            parsed = self._extract_json(raw)
            return IntentType(parsed.get("intent", "general_talk"))
        except Exception:
            return IntentType.GENERAL_TALK

    def decompose_query(self, user_message: str) -> DecomposedQuery:
        try:
            raw = self._chat_completion(
                DECOMPOSITION_SYSTEM_PROMPT,
                user_message,
                model=self.model_name,
                max_tokens=256,
            )
            parsed = self._extract_json(raw)
            return DecomposedQuery(
                intent=IntentType.TEXT_SEARCH,
                original_query=user_message,
                tags=parsed.get("tags", []),
                filters=parsed.get("filters", {}),
                rewritten_query=parsed.get("rewritten_query", user_message),
            )
        except Exception:
            return DecomposedQuery(
                intent=IntentType.TEXT_SEARCH,
                original_query=user_message,
                tags=[],
                filters={},
                rewritten_query=user_message,
            )

    def rerank(self, query: str, products: list[dict], top_k: int = 10) -> list[dict]:
        product_summaries = []
        for i, product in enumerate(products):
            summary = f"[{i}] {product.get('title', '')} | ${product.get('price', 'N/A')} | {product.get('brand', '')}"
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
                model=self.reranker_model_name,
                temperature=0.1,
                max_tokens=128,
            )
            indices = self._extract_json(raw)
            reranked = []
            for idx in indices[:top_k]:
                if isinstance(idx, int) and 0 <= idx < len(products):
                    reranked.append(products[idx])
            return reranked or products[:top_k]
        except Exception:
            return products[:top_k]

    def verify_image_match(
        self,
        query_image_url: str,
        candidate_image_urls: list[str],
        threshold: float = 0.5,
    ) -> list[dict]:
        results = []
        for candidate_url in candidate_image_urls:
            try:
                user_content = [
                    {
                        "type": "text",
                        "text": "Do these two product images show the same or very similar product? Respond with JSON: {\"match\": true/false, \"confidence\": 0.0-1.0}",
                    },
                    {"type": "image_url", "image_url": {"url": query_image_url}},
                    {"type": "image_url", "image_url": {"url": candidate_url}},
                ]
                raw = self._chat_completion(
                    "You are a visual product matching model.",
                    user_content,
                    model=self.verifier_model_name,
                    temperature=0.1,
                    max_tokens=128,
                )
                parsed = self._extract_json(raw)
                results.append(
                    {
                        "url": candidate_url,
                        "match": parsed.get("match", False),
                        "confidence": parsed.get("confidence", 0.0),
                    }
                )
            except Exception:
                results.append(
                    {
                        "url": candidate_url,
                        "match": True,
                        "confidence": threshold,
                    }
                )
        return results

    def close(self):
        self.client.close()
