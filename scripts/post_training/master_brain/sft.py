"""
03_generate_sft_data.py — SFT Data Generation (8,000 samples)
================================================================
Uses Llama 3.1-70B (4-bit quantized, via vLLM on cluster) as teacher
to generate supervised fine-tuning data using the "Product-First" approach.

Thematic Clustering:
1. Pick a random category
2. Sample 5 products
3. Prompt the 70B model to write a query that fits 3 and excludes 2
4. Generate a gold response grounded in product JSON

Output: data/training/sft_train.jsonl

Usage:
    python scripts/03_generate_sft_data.py
    python scripts/03_generate_sft_data.py --num-samples 100 --config config/settings.yaml
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


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


# ================================================================
# Prompt Templates
# ================================================================

QUERY_GENERATION_PROMPT = """You are a helpful dataset creator for a commerce AI assistant.

I will give you 5 products from the same category. Your task:
1. Write a NATURAL shopping query that a real customer might ask
2. The query should clearly match 3 of the products but NOT match the other 2
3. Make the query specific enough to differentiate (mention use-case, style, budget, etc.)
4. Vary the query style: sometimes casual, sometimes detailed, sometimes with budget constraints

Products:
{products_json}

Products that SHOULD match (indices): {matching_indices}
Products that should NOT match (indices): {exclude_indices}

Respond with ONLY the customer query (no explanation, no quotes). Make it sound like a real person typing:"""

RESPONSE_GENERATION_PROMPT = """You are a warm, knowledgeable shopping assistant. A customer asked:

"{query}"

Here are the products that match their needs:
{matching_products}

Instructions:
1. Recommend the best 2-3 products from the matching list
2. Explain WHY each product fits their needs
3. Be specific — cite actual features from the product data
4. Be warm and conversational, not robotic
5. Do NOT make up features — ONLY use what's in the product data
6. If relevant, mention trade-offs or alternatives
7. End with a helpful follow-up question

Respond as the assistant:"""

SYSTEM_PROMPT = """You are a warm, knowledgeable shopping assistant. You combine deep product \
expertise with genuine empathy. You never hallucinate features — you only recommend products from \
the provided context. When uncertain, you ask clarifying questions. You speak naturally, like a \
trusted friend who happens to know everything about shopping."""


class SFTDataGenerator:
    """Generates SFT training data using the teacher model."""

    def __init__(self, teacher_api_base: str, teacher_model: str, sqlite: SQLiteCatalog):
        self.api_base = teacher_api_base.rstrip("/")
        self.teacher_model = teacher_model
        self.sqlite = sqlite
        self.client = httpx.Client(timeout=120.0)

    def _call_teacher(self, system: str, user: str, temperature: float = 0.8, max_tokens: int = 1024) -> str:
        """Call the teacher model via vLLM OpenAI-compatible API."""
        response = self.client.post(
            f"{self.api_base}/chat/completions",
            json={
                "model": self.teacher_model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def _format_product_for_context(self, product: dict) -> dict:
        """Format a product dict for inclusion in prompts."""
        return {
            "title": product.get("title", ""),
            "price": product.get("price"),
            "brand": product.get("brand", ""),
            "category": product.get("category", ""),
            "description": (product.get("description", "") or "")[:300],
            "rating": product.get("rating"),
            "review_count": product.get("review_count", 0),
        }

    def generate_sample(self, category: str) -> dict | None:
        """
        Generate one SFT sample using thematic clustering.
        Returns {"messages": [...]} or None if generation fails.
        """
        # Step 1: Pull 5 random products from the category
        products = self.sqlite.get_random_products_by_category(category, n=5)
        if len(products) < 5:
            return None

        # Step 2: Randomly select 3 matching and 2 excluded
        indices = list(range(5))
        random.shuffle(indices)
        matching_indices = sorted(indices[:3])
        exclude_indices = sorted(indices[3:])

        # Step 3: Generate the user query
        products_for_prompt = [self._format_product_for_context(p) for p in products]
        query_prompt = QUERY_GENERATION_PROMPT.format(
            products_json=json.dumps(products_for_prompt, indent=2),
            matching_indices=matching_indices,
            exclude_indices=exclude_indices,
        )

        try:
            user_query = self._call_teacher(
                "You are a dataset creator. Output only the shopping query.",
                query_prompt,
                temperature=0.9,
                max_tokens=200,
            ).strip().strip('"')
        except Exception as e:
            print(f"  [ERROR] Query generation failed: {e}")
            return None

        # Step 4: Generate the gold response (grounded in matching products only)
        matching_products = [products[i] for i in matching_indices]
        matching_formatted = json.dumps(
            [self._format_product_for_context(p) for p in matching_products], indent=2
        )

        response_prompt = RESPONSE_GENERATION_PROMPT.format(
            query=user_query,
            matching_products=matching_formatted,
        )

        try:
            assistant_response = self._call_teacher(
                SYSTEM_PROMPT,
                response_prompt,
                temperature=0.7,
                max_tokens=1024,
            ).strip()
        except Exception as e:
            print(f"  [ERROR] Response generation failed: {e}")
            return None

        # Step 5: Build the training sample
        # Include product context in the system message (gold context)
        context_system = (
            SYSTEM_PROMPT + "\n\n"
            "Available products for this query:\n"
            + matching_formatted
        )

        return {
            "messages": [
                {"role": "system", "content": context_system},
                {"role": "user", "content": user_query},
                {"role": "assistant", "content": assistant_response},
            ],
            "metadata": {
                "category": category,
                "matching_product_ids": [products[i]["product_id"] for i in matching_indices],
                "excluded_product_ids": [products[i]["product_id"] for i in exclude_indices],
            },
        }


def main():
    parser = argparse.ArgumentParser(description="Generate SFT training data")
    parser.add_argument("--config", default="config/settings.yaml")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    num_samples = args.num_samples or config["training"]["sft"]["num_samples"]
    output_path = args.output or config["training"]["sft"]["data_path"]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Initialize
    sqlite = SQLiteCatalog(config["databases"]["sqlite"]["path"])
    teacher_cfg = config["models"]["teacher"]

    generator = SFTDataGenerator(
        teacher_api_base=teacher_cfg["api_base"],
        teacher_model=teacher_cfg["model_id"],
        sqlite=sqlite,
    )

    categories = sqlite.get_all_categories()
    if not categories:
        print("❌ No categories found in catalog. Run 01_normalize_catalog.py first.")
        return

    print(f"Found {len(categories)} categories")
    print(f"Generating {num_samples} SFT samples...")

    samples = []
    failures = 0

    with open(output_path, "w") as f:
        for i in tqdm(range(num_samples), desc="Generating SFT data"):
            category = random.choice(categories)
            sample = generator.generate_sample(category)

            if sample is None:
                failures += 1
                continue

            f.write(json.dumps(sample) + "\n")
            samples.append(sample)

    print(f"\n✅ Generated {len(samples)} SFT samples ({failures} failures)")
    print(f"   Output: {output_path}")
    sqlite.close()


if __name__ == "__main__":
    main()
