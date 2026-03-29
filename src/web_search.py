"""
web_search.py - SerpApi Google Shopping integration
===================================================
Fetches structured shopping results from SerpApi and normalizes them into the
same product-like shape used by the local catalog.
"""

from __future__ import annotations

import hashlib
import json
from typing import Optional


class SerpApiGoogleShoppingSearcher:
    """Thin client for SerpApi Google Shopping results."""

    def __init__(
        self,
        api_key: str,
        api_base: str = "https://serpapi.com",
        location: str = "",
        gl: str = "us",
        hl: str = "en",
        mock_results_path: str = "",
        timeout: float = 30.0,
    ):
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        self.location = location
        self.gl = gl
        self.hl = hl
        self.mock_results_path = mock_results_path
        self.client = None
        self.timeout = timeout

    def search(self, query: str, num_results: int = 10) -> list[dict]:
        """Return raw shopping results from SerpApi."""
        if self.mock_results_path:
            try:
                with open(self.mock_results_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                results = data.get("shopping_results", []) or data.get("inline_shopping_results", [])
                return results[:num_results]
            except Exception as e:
                print(f"[SerpApi Mock] Failed to load {self.mock_results_path}: {e}")
                return []

        params = {
            "engine": "google_shopping",
            "q": query,
            "api_key": self.api_key,
            "gl": self.gl,
            "hl": self.hl,
            "num": min(max(num_results, 1), 100),
            "direct_link": "true",
            "no_cache": "false",
        }
        if self.location:
            params["location"] = self.location

        try:
            if self.client is None:
                import httpx

                self.client = httpx.Client(timeout=self.timeout)
            response = self.client.get(f"{self.api_base}/search.json", params=params)
            response.raise_for_status()
            data = response.json()
            return data.get("shopping_results", []) or data.get("inline_shopping_results", [])
        except Exception as e:
            print(f"[SerpApi] Search failed: {e}")
            return []


class WebProductExtractor:
    """Normalize SerpApi shopping results into the app's product shape."""

    @staticmethod
    def extract_products(search_results: list[dict]) -> list[dict]:
        products = []

        for idx, result in enumerate(search_results):
            title = result.get("title") or "Untitled"
            product_url = result.get("product_link") or result.get("link") or ""
            external_id = str(result.get("product_id") or result.get("position") or idx)
            dedupe_seed = product_url or f"{title}|{external_id}"
            product_id = f"web_{hashlib.sha256(dedupe_seed.encode('utf-8')).hexdigest()[:12]}"

            thumbnail = (
                result.get("thumbnail")
                or result.get("serpapi_thumbnail")
                or ""
            )
            merchant = result.get("source", "")
            price = result.get("extracted_price")
            if price is None:
                price = WebProductExtractor._coerce_price(result.get("price"))

            rating = result.get("rating")
            review_count = result.get("reviews")
            description_parts = []
            if result.get("snippet"):
                description_parts.append(result["snippet"])
            if merchant:
                description_parts.append(f"Merchant: {merchant}")
            if result.get("delivery"):
                description_parts.append(f"Delivery: {result['delivery']}")
            if result.get("extensions"):
                description_parts.append(" | ".join(result["extensions"]))

            product = {
                "product_id": product_id,
                "external_id": external_id,
                "title": title,
                "description": " ".join(description_parts).strip(),
                "price": price,
                "brand": merchant,
                "category": "",
                "image": thumbnail,
                "image_urls": thumbnail,
                "source": "web",
                "merchant": merchant,
                "url": product_url,
                "rating": rating,
                "review_count": review_count or 0,
                "in_stock": 1,
                "web_position": result.get("position", idx + 1),
            }
            products.append(product)

        return products

    @staticmethod
    def _coerce_price(raw_price) -> Optional[float]:
        if raw_price is None:
            return None
        if isinstance(raw_price, (int, float)):
            return float(raw_price)

        cleaned = str(raw_price).replace("$", "").replace(",", "").strip()
        try:
            return float(cleaned)
        except ValueError:
            return None


class WebSearchPipeline:
    """
    Structured web search pipeline:
    1. Search SerpApi Google Shopping
    2. Normalize results to product-like records
    3. Return a ranked candidate list for later fusion
    """

    def __init__(self, searcher: SerpApiGoogleShoppingSearcher):
        self.searcher = searcher
        self.extractor = WebProductExtractor()

    def search(
        self,
        query: str,
        num_results: int = 10,
        include_visual: bool = False,
    ) -> dict:
        del include_visual  # Present for API compatibility with the agent.

        raw_results = self.searcher.search(query, num_results=num_results)
        products = self.extractor.extract_products(raw_results)
        return {"products": products}
