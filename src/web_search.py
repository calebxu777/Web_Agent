"""
web_search.py — Firecrawl Web Search Integration
===================================================
Searches the web for products using Firecrawl,
extracts structured product data, embeds them,
and returns candidates ready for RRF fusion with local results.
"""

from __future__ import annotations

import json
import re
from typing import Optional

import httpx
import numpy as np

from src.embeddings import BGEM3Embedder, DINOv2Embedder


class FirecrawlSearcher:
    """
    Uses Firecrawl API to search the web and extract structured product data.

    Flow:
    1. Search query → Firecrawl search endpoint → result URLs
    2. Firecrawl scrapes each URL → structured markdown/JSON
    3. Parse product information (title, price, description, image URLs)
    4. Embed products for hybrid search
    """

    def __init__(
        self,
        api_key: str,
        api_base: str = "https://api.firecrawl.dev/v1",
        timeout: float = 30.0,
    ):
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        self.client = httpx.Client(
            timeout=timeout,
            headers={"Authorization": f"Bearer {api_key}"},
        )

    def search(self, query: str, num_results: int = 10) -> list[dict]:
        """
        Search the web and return structured results.
        Firecrawl's /search endpoint handles both search and scraping.
        """
        try:
            response = self.client.post(
                f"{self.api_base}/search",
                json={
                    "query": query,
                    "limit": num_results,
                    "scrapeOptions": {
                        "formats": ["markdown"],
                    },
                },
            )
            response.raise_for_status()
            data = response.json()
            return data.get("data", [])
        except Exception as e:
            print(f"[Firecrawl] Search failed: {e}")
            return []

    def scrape_url(self, url: str) -> dict | None:
        """Scrape a single URL for product data."""
        try:
            response = self.client.post(
                f"{self.api_base}/scrape",
                json={
                    "url": url,
                    "formats": ["markdown"],
                },
            )
            response.raise_for_status()
            return response.json().get("data", {})
        except Exception as e:
            print(f"[Firecrawl] Scrape failed for {url}: {e}")
            return None


class WebProductExtractor:
    """
    Parses Firecrawl search results into structured product data
    compatible with the local catalog format.
    """

    @staticmethod
    def extract_products(search_results: list[dict]) -> list[dict]:
        """
        Extract product-like items from Firecrawl search results.
        Each result has: url, title, description, markdown content.
        """
        products = []

        for i, result in enumerate(search_results):
            title = result.get("title", "")
            url = result.get("url", "")
            description = result.get("description", "")
            markdown = result.get("markdown", "")

            # Try to extract price from content
            price = WebProductExtractor._extract_price(markdown or description)

            # Try to extract image URLs from markdown
            image_urls = WebProductExtractor._extract_images(markdown)

            # Build a product-like dict compatible with local search results
            product = {
                "product_id": f"web_{i}_{hash(url) % 100000}",
                "title": title or "Untitled",
                "description": description,
                "price": price,
                "brand": "",
                "category": "",
                "image_urls": ",".join(image_urls[:3]),
                "source": "web",
                "url": url,
                "in_stock": 1,
                # Full markdown for the Master Brain to reference
                "full_content": (markdown or description)[:2000],
            }
            products.append(product)

        return products

    @staticmethod
    def _extract_price(text: str) -> float | None:
        """Extract price from text using common patterns."""
        patterns = [
            r'\$(\d{1,5}(?:\.\d{2})?)',        # $99.99
            r'USD\s*(\d{1,5}(?:\.\d{2})?)',     # USD 99.99
            r'Price:\s*\$?(\d{1,5}(?:\.\d{2})?)', # Price: 99.99
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue
        return None

    @staticmethod
    def _extract_images(markdown: str) -> list[str]:
        """Extract image URLs from markdown content."""
        # Match ![alt](url) pattern
        pattern = r'!\[.*?\]\((https?://[^\s\)]+)\)'
        urls = re.findall(pattern, markdown)
        # Also match raw image URLs
        img_pattern = r'(https?://\S+\.(?:jpg|jpeg|png|webp|gif))'
        urls.extend(re.findall(img_pattern, markdown, re.IGNORECASE))
        # Deduplicate
        seen = set()
        unique = []
        for u in urls:
            if u not in seen:
                seen.add(u)
                unique.append(u)
        return unique


class WebSearchPipeline:
    """
    Full web search pipeline:
    1. Firecrawl search → structured results
    2. Extract products from results
    3. Embed products (BGE-M3 text + optional DINOv2 visual)
    4. Return candidates ready for RRF fusion with local results
    """

    def __init__(
        self,
        firecrawl: FirecrawlSearcher,
        semantic_embedder: BGEM3Embedder,
        visual_embedder: Optional[DINOv2Embedder] = None,
    ):
        self.firecrawl = firecrawl
        self.semantic_embedder = semantic_embedder
        self.visual_embedder = visual_embedder
        self.extractor = WebProductExtractor()

    def search(
        self,
        query: str,
        num_results: int = 10,
        include_visual: bool = False,
    ) -> dict:
        """
        Run the full web search pipeline.

        Returns:
            dict with 'products' (list), 'text_embeddings' (np.ndarray),
            and optionally 'visual_embeddings' (np.ndarray)
        """
        # Step 1: Firecrawl search
        raw_results = self.firecrawl.search(query, num_results=num_results)

        if not raw_results:
            return {"products": [], "text_embeddings": None, "visual_embeddings": None}

        # Step 2: Extract products
        products = self.extractor.extract_products(raw_results)

        if not products:
            return {"products": [], "text_embeddings": None, "visual_embeddings": None}

        # Step 3: Embed text (title + description)
        texts = [
            f"{p['title']} | {p['description'][:300]}"
            for p in products
        ]
        text_result = self.semantic_embedder.embed_batch(texts, show_progress=False)
        text_embeddings = text_result["dense"]

        # Step 4: Optionally embed images
        visual_embeddings = None
        if include_visual and self.visual_embedder:
            image_urls = []
            for p in products:
                urls = p.get("image_urls", "").split(",")
                image_urls.append(urls[0].strip() if urls and urls[0].strip() else "")

            # Only embed products that have images
            valid_urls = [u for u in image_urls if u]
            if valid_urls:
                visual_embeddings = self.visual_embedder.embed_batch(
                    valid_urls, show_progress=False
                )

        return {
            "products": products,
            "text_embeddings": text_embeddings,
            "visual_embeddings": visual_embeddings,
        }
