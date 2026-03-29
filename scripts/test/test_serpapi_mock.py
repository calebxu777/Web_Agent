"""
Smoke test for the SerpApi Google Shopping mock fixture.

Runs without making network calls and verifies that the normalized web-search
records match the shape expected by the frontend/backend pipeline.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.web_search import SerpApiGoogleShoppingSearcher, WebSearchPipeline


def main() -> None:
    fixture = Path("scripts/test/mock_serpapi_google_shopping.json")
    searcher = SerpApiGoogleShoppingSearcher(
        api_key="",
        mock_results_path=str(fixture),
    )
    pipeline = WebSearchPipeline(searcher=searcher)
    result = pipeline.search("wireless noise cancelling headphones", num_results=3)
    products = result["products"]

    assert len(products) == 3, f"expected 3 products, got {len(products)}"

    required_keys = {
        "product_id",
        "title",
        "price",
        "image",
        "image_urls",
        "source",
        "merchant",
        "url",
    }
    for idx, product in enumerate(products, start=1):
        missing = required_keys - product.keys()
        assert not missing, f"product {idx} missing keys: {sorted(missing)}"

    print("Mock SerpApi smoke test passed.")
    for product in products:
        print(
            f"- {product['title']} | ${product['price']:.2f} | "
            f"{product['merchant']} | {product['url']}"
        )


if __name__ == "__main__":
    main()
