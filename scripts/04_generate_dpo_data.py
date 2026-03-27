"""
04_generate_dpo_data.py — DPO Pair Generation (2,000 pairs)
==============================================================
Generates contrastive alignment pairs using Llama 3.1-70B as critic.

For each pair:
1. Generate/reuse a user prompt + product context
2. Generate two candidate responses:
   - "High-EQ & Grounded" (Chosen)
   - "Robotic or Hallucinated" (Rejected)
3. Use 70B as "Commerce Critic" to validate chosen/rejected labeling

Output: data/training/dpo_train.jsonl

Usage:
    python scripts/04_generate_dpo_data.py
    python scripts/04_generate_dpo_data.py --num-pairs 100
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

CHOSEN_RESPONSE_PROMPT = """You are a warm, empathetic shopping assistant. A customer asked:

"{query}"

Here are the matching products:
{products}

Write a HIGH-QUALITY response that:
1. Shows genuine empathy and understanding of the customer's needs
2. Recommends 2-3 products with specific, FACTUAL reasons from the product data
3. Uses a warm, conversational tone (like a knowledgeable friend)
4. Mentions relevant trade-offs honestly
5. NEVER makes up features — only cite what's in the data
6. Ends with a helpful follow-up question

Respond as the assistant:"""

REJECTED_RESPONSE_PROMPT = """You are generating a DELIBERATELY POOR shopping assistant response for training data.
The response should have ONE OR MORE of these flaws:

A customer asked: "{query}"

Available products:
{products}

Generate a response that is INTENTIONALLY FLAWED in one of these ways (pick randomly):
1. ROBOTIC: Overly formal, uses bullet-point lists, sounds like a product database dump
2. HALLUCINATED: Mentions features or specs that are NOT in the product data
3. GENERIC: Gives vague advice like "this is a great choice" without specific reasoning
4. PUSHY: Overly salesy, ignores the customer's stated preferences
5. INCOMPLETE: Only mentions one product, ignores budget or other constraints

Make it subtly bad — not obviously broken, but clearly worse than a great response.

Respond as the flawed assistant:"""

CRITIC_PROMPT = """You are an expert Commerce Critic evaluating two shopping assistant responses.

Customer query: "{query}"

Available product data:
{products}

Response A:
{response_a}

Response B:
{response_b}

Evaluate both responses on these criteria (1-5 each):
1. **Empathy**: Does the assistant understand and address the customer's underlying needs?
2. **Accuracy**: Are all mentioned product features factually present in the data?
3. **Specificity**: Does the response cite specific, relevant details from the products?
4. **Tone**: Is the tone warm and conversational (not robotic or pushy)?
5. **Helpfulness**: Does the response help the customer make a decision?

Then decide:
- Which response is CHOSEN (better overall)?
- Which is REJECTED (worse overall)?

Respond with ONLY this JSON:
{{
  "scores_a": {{"empathy": X, "accuracy": X, "specificity": X, "tone": X, "helpfulness": X}},
  "scores_b": {{"empathy": X, "accuracy": X, "specificity": X, "tone": X, "helpfulness": X}},
  "chosen": "A" or "B",
  "reason": "Brief explanation"
}}"""


class DPODataGenerator:
    """Generates DPO training pairs using the teacher model as both generator and critic."""

    def __init__(self, teacher_api_base: str, teacher_model: str, sqlite: SQLiteCatalog):
        self.api_base = teacher_api_base.rstrip("/")
        self.teacher_model = teacher_model
        self.sqlite = sqlite
        self.client = httpx.Client(timeout=120.0)

    def _call_teacher(self, system: str, user: str, temperature: float = 0.8, max_tokens: int = 1024) -> str:
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

    def _format_products(self, products: list[dict]) -> str:
        formatted = []
        for p in products:
            parts = [f"- {p.get('title', '')}"]
            if p.get("price"):
                parts[0] += f" (${p['price']})"
            if p.get("brand"):
                parts.append(f"  Brand: {p['brand']}")
            if p.get("description"):
                parts.append(f"  Description: {(p['description'] or '')[:200]}")
            if p.get("rating"):
                parts.append(f"  Rating: {p['rating']}/5")
            formatted.append("\n".join(parts))
        return "\n\n".join(formatted)

    def generate_pair(self, category: str) -> dict | None:
        """Generate one DPO pair."""
        # Get products
        products = self.sqlite.get_random_products_by_category(category, n=5)
        if len(products) < 3:
            return None

        matching = random.sample(products, 3)
        products_text = self._format_products(matching)

        # Generate a query
        query_prompt = f"""Write a natural shopping query that a customer might ask about these products:
{products_text}

Output ONLY the query (no quotes, no explanation):"""

        try:
            query = self._call_teacher(
                "Output only a single shopping query.",
                query_prompt,
                temperature=0.9,
                max_tokens=150,
            ).strip().strip('"')
        except Exception as e:
            print(f"  [ERROR] Query gen: {e}")
            return None

        # Generate CHOSEN response (High-EQ & Grounded)
        try:
            chosen = self._call_teacher(
                "You are the world's best shopping assistant.",
                CHOSEN_RESPONSE_PROMPT.format(query=query, products=products_text),
                temperature=0.7,
                max_tokens=1024,
            ).strip()
        except Exception as e:
            print(f"  [ERROR] Chosen gen: {e}")
            return None

        # Generate REJECTED response (Robotic/Hallucinated)
        try:
            rejected = self._call_teacher(
                "Generate a deliberately flawed response.",
                REJECTED_RESPONSE_PROMPT.format(query=query, products=products_text),
                temperature=0.9,
                max_tokens=1024,
            ).strip()
        except Exception as e:
            print(f"  [ERROR] Rejected gen: {e}")
            return None

        # Use critic to validate and potentially swap labels
        try:
            # Randomly assign A/B to avoid position bias
            if random.random() > 0.5:
                response_a, response_b = chosen, rejected
                mapping = {"A": "chosen", "B": "rejected"}
            else:
                response_a, response_b = rejected, chosen
                mapping = {"A": "rejected", "B": "chosen"}

            critic_response = self._call_teacher(
                "You are an expert Commerce Critic. Respond only with JSON.",
                CRITIC_PROMPT.format(
                    query=query,
                    products=products_text,
                    response_a=response_a,
                    response_b=response_b,
                ),
                temperature=0.1,
                max_tokens=512,
            )

            critic_data = json.loads(critic_response)
            critic_chosen = critic_data.get("chosen", "A")

            # Map critic's choice back
            if mapping[critic_chosen] == "chosen":
                # Critic agrees with our labels
                final_chosen, final_rejected = chosen, rejected
            else:
                # Critic disagrees — swap labels
                final_chosen, final_rejected = rejected, chosen

        except (json.JSONDecodeError, KeyError):
            # If critic fails, use our original labels
            final_chosen, final_rejected = chosen, rejected

        # Build the training pair
        system_prompt = (
            "You are a warm, knowledgeable shopping assistant. You combine deep product "
            "expertise with genuine empathy. You never hallucinate features."
        )

        return {
            "prompt": f"[System: {system_prompt}]\n\nProduct context:\n{products_text}\n\nCustomer: {query}",
            "chosen": final_chosen,
            "rejected": final_rejected,
            "metadata": {
                "category": category,
                "product_ids": [p["product_id"] for p in matching],
            },
        }


def main():
    parser = argparse.ArgumentParser(description="Generate DPO training pairs")
    parser.add_argument("--config", default="config/settings.yaml")
    parser.add_argument("--num-pairs", type=int, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    num_pairs = args.num_pairs or config["training"]["dpo"]["num_pairs"]
    output_path = args.output or config["training"]["dpo"]["data_path"]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    sqlite = SQLiteCatalog(config["databases"]["sqlite"]["path"])
    teacher_cfg = config["models"]["teacher"]

    generator = DPODataGenerator(
        teacher_api_base=teacher_cfg["api_base"],
        teacher_model=teacher_cfg["model_id"],
        sqlite=sqlite,
    )

    categories = sqlite.get_all_categories()
    if not categories:
        print("❌ No categories found. Run 01_normalize_catalog.py first.")
        return

    print(f"Found {len(categories)} categories")
    print(f"Generating {num_pairs} DPO pairs...")

    pairs = []
    failures = 0

    with open(output_path, "w") as f:
        for i in tqdm(range(num_pairs), desc="Generating DPO pairs"):
            category = random.choice(categories)
            pair = generator.generate_pair(category)

            if pair is None:
                failures += 1
                continue

            f.write(json.dumps(pair) + "\n")
            pairs.append(pair)

    print(f"\n✅ Generated {len(pairs)} DPO pairs ({failures} failures)")
    print(f"   Output: {output_path}")
    sqlite.close()


if __name__ == "__main__":
    main()
