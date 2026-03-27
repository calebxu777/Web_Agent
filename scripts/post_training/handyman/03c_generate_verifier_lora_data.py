"""
03c_generate_verifier_lora_data.py — Visual Verifier LoRA Training Data
=========================================================================
Generates training data for the Handyman Visual Verifier LoRA.

This is the EASIEST dataset to generate — no teacher model needed!
We create image pairs automatically from the catalog:
- Positive pairs: two images of the SAME product → match=true
- Hard negatives: two images from the SAME category but DIFFERENT products → match=false
- Easy negatives: two images from DIFFERENT categories → match=false

The VLM learns to compare two product images and output a match score.

Output: data/training/verifier_lora_train.jsonl

Usage:
    python scripts/03c_generate_verifier_lora_data.py
    python scripts/03c_generate_verifier_lora_data.py --num-samples 4000
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.database import SQLiteCatalog


def load_config(p):
    with open(p) as f:
        return yaml.safe_load(f)


VERIFIER_SYSTEM_PROMPT = (
    "You are a visual product matching model. Given two product images, "
    "determine if they show the same or very similar product. "
    "Respond with JSON: {\"match\": true/false, \"confidence\": 0.0-1.0, \"reason\": \"brief reason\"}"
)


class VerifierDataGenerator:
    def __init__(self, sqlite: SQLiteCatalog):
        self.sqlite = sqlite

    def _get_products_with_images(self, category: str, n: int = 10) -> list[dict]:
        """Get products that have image URLs."""
        products = self.sqlite.get_random_products_by_category(category, n=n * 2)
        return [p for p in products if p.get("image_urls") and p["image_urls"].strip()]

    def generate_positive_pair(self, category: str) -> dict | None:
        """
        Positive: same product, two different images (if available)
        or same product image described differently.
        """
        products = self._get_products_with_images(category, n=5)
        if not products:
            return None

        product = random.choice(products)
        image_urls = [u.strip() for u in product["image_urls"].split(",") if u.strip()]

        if len(image_urls) >= 2:
            img_a, img_b = random.sample(image_urls, 2)
        elif len(image_urls) == 1:
            # Same image — model should still say "match"
            img_a = img_b = image_urls[0]
        else:
            return None

        return {
            "messages": [
                {"role": "system", "content": VERIFIER_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Do these two product images show the same or very similar product?"},
                        {"type": "image_url", "image_url": {"url": img_a}},
                        {"type": "image_url", "image_url": {"url": img_b}},
                    ],
                },
                {
                    "role": "assistant",
                    "content": json.dumps({
                        "match": True,
                        "confidence": round(random.uniform(0.85, 0.99), 2),
                        "reason": f"Both images show the same product: {product['title']}",
                    }),
                },
            ],
            "metadata": {"pair_type": "positive", "product_id": product["product_id"]},
        }

    def generate_hard_negative(self, category: str) -> dict | None:
        """
        Hard negative: same category, different products.
        These look similar but aren't the same product.
        """
        products = self._get_products_with_images(category, n=10)
        if len(products) < 2:
            return None

        p_a, p_b = random.sample(products, 2)
        urls_a = [u.strip() for u in p_a["image_urls"].split(",") if u.strip()]
        urls_b = [u.strip() for u in p_b["image_urls"].split(",") if u.strip()]

        if not urls_a or not urls_b:
            return None

        return {
            "messages": [
                {"role": "system", "content": VERIFIER_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Do these two product images show the same or very similar product?"},
                        {"type": "image_url", "image_url": {"url": random.choice(urls_a)}},
                        {"type": "image_url", "image_url": {"url": random.choice(urls_b)}},
                    ],
                },
                {
                    "role": "assistant",
                    "content": json.dumps({
                        "match": False,
                        "confidence": round(random.uniform(0.60, 0.85), 2),
                        "reason": f"Different products in same category. Image A: {p_a['title']}, Image B: {p_b['title']}",
                    }),
                },
            ],
            "metadata": {
                "pair_type": "hard_negative",
                "product_a": p_a["product_id"],
                "product_b": p_b["product_id"],
            },
        }

    def generate_easy_negative(self, categories: list[str]) -> dict | None:
        """
        Easy negative: different categories entirely.
        Obviously different products.
        """
        if len(categories) < 2:
            return None

        cat_a, cat_b = random.sample(categories, 2)
        products_a = self._get_products_with_images(cat_a, 3)
        products_b = self._get_products_with_images(cat_b, 3)

        if not products_a or not products_b:
            return None

        p_a = random.choice(products_a)
        p_b = random.choice(products_b)
        urls_a = [u.strip() for u in p_a["image_urls"].split(",") if u.strip()]
        urls_b = [u.strip() for u in p_b["image_urls"].split(",") if u.strip()]

        if not urls_a or not urls_b:
            return None

        return {
            "messages": [
                {"role": "system", "content": VERIFIER_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Do these two product images show the same or very similar product?"},
                        {"type": "image_url", "image_url": {"url": random.choice(urls_a)}},
                        {"type": "image_url", "image_url": {"url": random.choice(urls_b)}},
                    ],
                },
                {
                    "role": "assistant",
                    "content": json.dumps({
                        "match": False,
                        "confidence": round(random.uniform(0.90, 0.99), 2),
                        "reason": f"Completely different products. Image A: {p_a['title']} ({cat_a}), Image B: {p_b['title']} ({cat_b})",
                    }),
                },
            ],
            "metadata": {
                "pair_type": "easy_negative",
                "product_a": p_a["product_id"],
                "product_b": p_b["product_id"],
            },
        }


def main():
    parser = argparse.ArgumentParser(description="Generate Verifier LoRA training data")
    parser.add_argument("--config", default="config/settings.yaml")
    parser.add_argument("--num-samples", type=int, default=4000)
    parser.add_argument("--output", default="data/training/verifier_lora_train.jsonl")
    args = parser.parse_args()

    config = load_config(args.config)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    sqlite = SQLiteCatalog(config["databases"]["sqlite"]["path"])
    generator = VerifierDataGenerator(sqlite)
    categories = sqlite.get_all_categories()

    if not categories:
        print("❌ No categories found. Run 01_normalize_catalog.py first.")
        return

    # Distribution: 25% positive, 50% hard negative, 25% easy negative
    num_pos = int(args.num_samples * 0.25)
    num_hard = int(args.num_samples * 0.50)
    num_easy = args.num_samples - num_pos - num_hard

    samples = []
    failures = 0

    with open(args.output, "w") as f:
        for _ in tqdm(range(num_pos), desc="Positive pairs"):
            s = generator.generate_positive_pair(random.choice(categories))
            if s:
                f.write(json.dumps(s) + "\n")
                samples.append(s)
            else:
                failures += 1

        for _ in tqdm(range(num_hard), desc="Hard negatives"):
            s = generator.generate_hard_negative(random.choice(categories))
            if s:
                f.write(json.dumps(s) + "\n")
                samples.append(s)
            else:
                failures += 1

        for _ in tqdm(range(num_easy), desc="Easy negatives"):
            s = generator.generate_easy_negative(categories)
            if s:
                f.write(json.dumps(s) + "\n")
                samples.append(s)
            else:
                failures += 1

    print(f"\n✅ Generated {len(samples)} Verifier LoRA samples ({failures} failures)")
    print(f"   Positive: {num_pos}, Hard neg: {num_hard}, Easy neg: {num_easy}")
    print(f"   Output: {args.output}")
    sqlite.close()


if __name__ == "__main__":
    main()
