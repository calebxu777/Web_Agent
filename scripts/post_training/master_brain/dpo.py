"""
03a_generate_router_lora_data.py — Router LoRA Training Data
==============================================================
Generates training data for the Handyman Router LoRA:
- Intent classification: general_talk | text_search | image_search | web_search
- Query decomposition: tags, filters, rewritten_query

Uses Llama 3.1-70B teacher to generate diverse (query → structured output) pairs.

Output: data/training/router_lora_train.jsonl

Usage:
    python scripts/03a_generate_router_lora_data.py
    python scripts/03a_generate_router_lora_data.py --num-samples 5000
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


# ================================================================
# Prompt Templates
# ================================================================

INTENT_DATA_PROMPT = """You are generating training data for an intent classifier.
Generate a realistic customer message and classify it.

Category context (for shopping queries): {category}
Example products: {product_titles}

Generate ONE example for the intent type: {intent_type}

Intent types:
- general_talk: greetings, questions about the assistant, chit-chat
- text_search: user wants to find/buy something using text description
- image_search: user describes wanting to find products similar to an image they have
- web_search: user explicitly asks to search the web, or wants products not in a typical catalog

Respond with ONLY this JSON:
{{
  "message": "the customer message",
  "intent": "{intent_type}"
}}"""

DECOMPOSITION_DATA_PROMPT = """You are generating training data for a query decomposition model.
Given a product category and examples, generate a realistic shopping query and decompose it.

Category: {category}
Example products: {product_titles}

Generate a shopping query and extract structured information.
Respond with ONLY this JSON:
{{
  "message": "the shopping query",
  "decomposition": {{
    "intent": "text_search",
    "tags": ["tag1", "tag2", "tag3"],
    "filters": {{"price_max": null, "price_min": null, "brand": null, "color": null, "category": null}},
    "rewritten_query": "cleaned search query"
  }}
}}

Make the query natural and varied. Include budget constraints ~40% of the time.
Include brand preferences ~20% of the time. Include color/style ~50% of the time."""


class RouterDataGenerator:
    def __init__(self, teacher_api_base, teacher_model, sqlite):
        self.api_base = teacher_api_base.rstrip("/")
        self.model = teacher_model
        self.sqlite = sqlite
        self.client = httpx.Client(timeout=120.0)

    def _call_teacher(self, system, user, temperature=0.9, max_tokens=512):
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

    def generate_intent_sample(self, intent_type, category="", product_titles=""):
        """Generate one intent classification sample."""
        prompt = INTENT_DATA_PROMPT.format(
            intent_type=intent_type,
            category=category or "general",
            product_titles=product_titles or "N/A",
        )
        try:
            raw = self._call_teacher("Output only JSON.", prompt)
            data = json.loads(raw)
            # Build training format: input message → output JSON
            return {
                "messages": [
                    {"role": "system", "content": "Classify the user's intent into: general_talk, text_search, image_search, or web_search. Respond with JSON: {\"intent\": \"...\"}"},
                    {"role": "user", "content": data["message"]},
                    {"role": "assistant", "content": json.dumps({"intent": data["intent"]})},
                ]
            }
        except (json.JSONDecodeError, KeyError):
            return None

    def generate_decomposition_sample(self, category, product_titles):
        """Generate one query decomposition sample."""
        prompt = DECOMPOSITION_DATA_PROMPT.format(
            category=category,
            product_titles=product_titles,
        )
        try:
            raw = self._call_teacher("Output only JSON.", prompt)
            data = json.loads(raw)
            return {
                "messages": [
                    {"role": "system", "content": "Decompose the shopping query into tags, filters, and a rewritten query. Respond with JSON."},
                    {"role": "user", "content": data["message"]},
                    {"role": "assistant", "content": json.dumps(data["decomposition"])},
                ]
            }
        except (json.JSONDecodeError, KeyError):
            return None


def main():
    parser = argparse.ArgumentParser(description="Generate Router LoRA training data")
    parser.add_argument("--config", default="config/settings.yaml")
    parser.add_argument("--num-samples", type=int, default=5000)
    parser.add_argument("--output", default="data/training/router_lora_train.jsonl")
    args = parser.parse_args()

    config = load_config(args.config)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    sqlite = SQLiteCatalog(config["databases"]["sqlite"]["path"])
    teacher = config["models"]["teacher"]
    generator = RouterDataGenerator(teacher["api_base"], teacher["model_id"], sqlite)

    categories = sqlite.get_all_categories()
    intent_types = ["general_talk", "text_search", "image_search", "web_search"]
    # Distribution: 15% general, 40% text, 20% image, 25% web
    intent_weights = [0.15, 0.40, 0.20, 0.25]

    # Split: 60% intent classification, 40% decomposition
    num_intent = int(args.num_samples * 0.6)
    num_decomp = args.num_samples - num_intent

    samples = []
    failures = 0

    with open(args.output, "w") as f:
        # Generate intent samples
        for i in tqdm(range(num_intent), desc="Intent samples"):
            intent = random.choices(intent_types, weights=intent_weights, k=1)[0]
            cat = random.choice(categories) if categories else ""
            products = sqlite.get_random_products_by_category(cat, 3) if cat else []
            titles = ", ".join(p["title"] for p in products[:3])

            sample = generator.generate_intent_sample(intent, cat, titles)
            if sample:
                f.write(json.dumps(sample) + "\n")
                samples.append(sample)
            else:
                failures += 1

        # Generate decomposition samples
        for i in tqdm(range(num_decomp), desc="Decomposition samples"):
            cat = random.choice(categories) if categories else "General"
            products = sqlite.get_random_products_by_category(cat, 5) if cat else []
            titles = ", ".join(p["title"] for p in products[:5])

            sample = generator.generate_decomposition_sample(cat, titles)
            if sample:
                f.write(json.dumps(sample) + "\n")
                samples.append(sample)
            else:
                failures += 1

    print(f"\n✅ Generated {len(samples)} Router LoRA samples ({failures} failures)")
    print(f"   Intent: {num_intent - failures if failures < num_intent else num_intent}")
    print(f"   Decomposition: {len(samples) - (num_intent - min(failures, num_intent))}")
    print(f"   Output: {args.output}")
    sqlite.close()


if __name__ == "__main__":
    main()
