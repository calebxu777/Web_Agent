"""
03b_generate_reranker_lora_data.py — Reranker LoRA Training Data
==================================================================
Generates training data for the Handyman Reranker LoRA.

For each sample:
1. Pick a category, sample products, generate a user query
2. Use 70B teacher to score each product's relevance (0-10)
3. Train the 0.8B to output relevance-ordered indices

Output: data/training/reranker_lora_train.jsonl

Usage:
    python scripts/03b_generate_reranker_lora_data.py
    python scripts/03b_generate_reranker_lora_data.py --num-samples 3000
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import httpx
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.database import SQLiteCatalog


def load_config(p):
    with open(p) as f:
        return yaml.safe_load(f)


RERANK_SCORING_PROMPT = """You are a product relevance scorer for a commerce search engine.

Customer query: "{query}"

Score each product on relevance to the query (0 = completely irrelevant, 10 = perfect match).
Consider: category match, feature alignment, price appropriateness, brand relevance.

Products:
{products}

Respond with ONLY a JSON array of objects:
[{{"index": 0, "score": 8, "reason": "brief reason"}}, ...]

Order by score descending (most relevant first)."""

QUERY_GEN_PROMPT = """Given these products from the "{category}" category, write a natural shopping query.
Products: {titles}
Output ONLY the query text, no quotes:"""


class RerankerDataGenerator:
    def __init__(self, teacher_api_base, teacher_model, sqlite):
        self.api_base = teacher_api_base.rstrip("/")
        self.model = teacher_model
        self.sqlite = sqlite
        self.client = httpx.Client(timeout=120.0)

    def _call_teacher(self, system, user, temperature=0.7, max_tokens=1024):
        r = self.client.post(f"{self.api_base}/chat/completions", json={
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        })
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    def generate_sample(self, category):
        """Generate one reranking training sample."""
        # Get 8-15 products (mix of relevant and irrelevant)
        products = self.sqlite.get_random_products_by_category(category, n=8)

        # Also pull some random products from OTHER categories (noise)
        all_cats = self.sqlite.get_all_categories()
        other_cats = [c for c in all_cats if c != category]
        if other_cats:
            noise_cat = random.choice(other_cats)
            noise_products = self.sqlite.get_random_products_by_category(noise_cat, n=4)
            products.extend(noise_products)

        random.shuffle(products)

        if len(products) < 5:
            return None

        # Generate a query
        titles = ", ".join(p["title"] for p in products[:5])
        try:
            query = self._call_teacher(
                "Output only a shopping query.",
                QUERY_GEN_PROMPT.format(category=category, titles=titles),
                temperature=0.9, max_tokens=100,
            ).strip().strip('"')
        except Exception:
            return None

        # Format products for scoring
        product_list = ""
        for i, p in enumerate(products):
            product_list += f"[{i}] {p.get('title', '')} | ${p.get('price', 'N/A')} | {p.get('brand', '')} | {p.get('category', '')}\n"

        # Get relevance scores from teacher
        try:
            raw = self._call_teacher(
                "You are a relevance scorer. Output only JSON.",
                RERANK_SCORING_PROMPT.format(query=query, products=product_list),
                temperature=0.3, max_tokens=1024,
            )
            scores = json.loads(raw)
            # Sort by score descending
            scores.sort(key=lambda x: x.get("score", 0), reverse=True)
            ranked_indices = [s["index"] for s in scores if 0 <= s["index"] < len(products)]
        except (json.JSONDecodeError, KeyError):
            return None

        # Build training sample
        # Input: query + product list → Output: ranked indices
        input_text = f"Query: {query}\n\nProducts:\n{product_list}\nRank by relevance (most relevant first). Output JSON array of indices:"

        return {
            "messages": [
                {"role": "system", "content": "You are a product relevance ranker. Given a query and product list, output a JSON array of product indices ordered by relevance (most relevant first)."},
                {"role": "user", "content": input_text},
                {"role": "assistant", "content": json.dumps(ranked_indices)},
            ],
            "metadata": {
                "category": category,
                "num_products": len(products),
            },
        }


def main():
    parser = argparse.ArgumentParser(description="Generate Reranker LoRA training data")
    parser.add_argument("--config", default="config/settings.yaml")
    parser.add_argument("--num-samples", type=int, default=3000)
    parser.add_argument("--output", default="data/training/reranker_lora_train.jsonl")
    args = parser.parse_args()

    config = load_config(args.config)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    sqlite = SQLiteCatalog(config["databases"]["sqlite"]["path"])
    teacher = config["models"]["teacher"]
    generator = RerankerDataGenerator(teacher["api_base"], teacher["model_id"], sqlite)

    categories = sqlite.get_all_categories()
    if not categories:
        print("❌ No categories found. Run 01_normalize_catalog.py first.")
        return

    samples = []
    failures = 0

    with open(args.output, "w") as f:
        for i in tqdm(range(args.num_samples), desc="Reranker samples"):
            cat = random.choice(categories)
            sample = generator.generate_sample(cat)
            if sample:
                f.write(json.dumps(sample) + "\n")
                samples.append(sample)
            else:
                failures += 1

    print(f"\n✅ Generated {len(samples)} Reranker LoRA samples ({failures} failures)")
    print(f"   Output: {args.output}")
    sqlite.close()


if __name__ == "__main__":
    main()
