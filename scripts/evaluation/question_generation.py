"""
question_generation.py
======================
Uses an LLM (via OpenAI-compatible API) to generate two sets of evaluation prompts:

  Set 1 — Initial shopping questions (diverse product types, constraints, personas)
  Set 2 — Follow-up questions based on 4 archetypes (Refiner, Budgeter, Comparer, Skeptic)

Output: scripts/evaluation/testing_prompts.jsonl
Each line is a JSON object with:
  {"set": 1, "question": "...", "metadata": {...}}
  {"set": 2, "question": "...", "archetype": "...", "metadata": {...}}

Usage:
  # Set your API key and optionally a custom base URL
  set OPENAI_API_KEY=sk-...
  set OPENAI_BASE_URL=https://api.together.xyz/v1   (optional, defaults to OpenAI)

  python scripts/evaluation/question_generation.py --num-initial 80 --num-followups 60
"""

import argparse
import json
import os
import sys
from pathlib import Path

import httpx

# ── Config ──────────────────────────────────────────────────────────
API_KEY = os.environ.get("OPENAI_API_KEY", "")
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
MODEL = os.environ.get("EVAL_MODEL", "gpt-4o-mini")  # cheap + fast for generation

OUTPUT_PATH = Path(__file__).parent / "testing_prompts.jsonl"


def call_llm(system_prompt: str, user_prompt: str, temperature: float = 0.9) -> str:
    """Call an OpenAI-compatible chat completion endpoint."""
    if not API_KEY:
        print("ERROR: Set OPENAI_API_KEY environment variable.")
        print("  For OpenAI:   set OPENAI_API_KEY=sk-...")
        print("  For Together:  set OPENAI_API_KEY=... and set OPENAI_BASE_URL=https://api.together.xyz/v1")
        print("  For Groq:     set OPENAI_API_KEY=... and set OPENAI_BASE_URL=https://api.groq.com/openai/v1")
        sys.exit(1)

    url = f"{BASE_URL.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": 4096,
    }

    resp = httpx.post(url, json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


# ── Set 1: Initial Shopping Questions ───────────────────────────────
SET1_SYSTEM = """You are a test data generator for an AI-powered e-commerce shopping assistant.
Your job is to create diverse, realistic shopping queries that a real user would type into a conversational shopping assistant.

Rules:
- Each question should be a single, natural user message
- Vary the product categories: clothing, shoes, accessories, outerwear, activewear, formalwear, etc.
- Vary the specificity: some very specific ("blue Nike Air Max 90 size 11"), some vague ("something nice for a date night")
- Include different intents: product search, style advice, gift recommendations, occasion-based shopping
- Include price constraints in ~30% of questions
- Include brand preferences in ~25% of questions
- Include color/material preferences in ~40% of questions
- Make them feel like real human queries (casual tone, typos OK occasionally)
- Do NOT repeat the same product category more than 3 times
"""

SET1_USER_TEMPLATE = """Generate exactly {n} diverse shopping questions. Output them as a JSON array of strings, nothing else.

Example format:
["recommend me some blue nike hoodies", "I need a warm winter jacket under $200", "what are some good running shoes for flat feet?"]

Generate {n} questions now:"""


# ── Set 2: Follow-up Questions ──────────────────────────────────────
SET2_SYSTEM = """You are a test data generator for an AI-powered e-commerce shopping assistant.
Your job is to create realistic FOLLOW-UP questions that a user would ask AFTER receiving product recommendations.

There are 4 user archetypes for follow-ups:

1. **The Refiner** — Wants to narrow down by color, material, size, fit, style variant
   Example: "I like the first one, but does it come in blue?"

2. **The Budgeter** — Wants cheaper alternatives or price comparisons
   Example: "These are too expensive. Anything similar under $50?"

3. **The Comparer** — Wants to understand differences between recommended products
   Example: "What's the main difference between the top two?"

4. **The Skeptic** — Questions quality, reviews, durability, return policy
   Example: "Are the reviews actually good for the one you suggested?"

Rules:
- Each follow-up should be a single, natural user message
- They should make sense as a RESPONSE to product recommendations (not standalone questions)
- Vary the phrasing — don't repeat the same structure
- Include casual/conversational tone
- Some can reference specific products ("the second one", "the Nike one", "that jacket")
"""

SET2_USER_TEMPLATE = """Generate exactly {n} follow-up questions, distributed roughly equally across all 4 archetypes.
Output as a JSON array of objects with "archetype" and "question" fields.

Example format:
[
  {{"archetype": "Refiner", "question": "Does that come in a darker shade?"}},
  {{"archetype": "Budgeter", "question": "Anything cheaper that looks similar?"}}
]

Generate {n} follow-up questions now:"""


def generate_set1(n: int) -> list[dict]:
    """Generate initial shopping questions."""
    print(f"Generating {n} initial shopping questions (Set 1)...")

    # Generate in batches to avoid hitting token limits
    batch_size = min(n, 25)
    all_questions = []

    while len(all_questions) < n:
        remaining = n - len(all_questions)
        batch = min(batch_size, remaining)
        raw = call_llm(SET1_SYSTEM, SET1_USER_TEMPLATE.format(n=batch))

        # Parse the JSON array from the LLM response
        try:
            # Find the JSON array in the response (LLM might add text around it)
            start = raw.index("[")
            end = raw.rindex("]") + 1
            questions = json.loads(raw[start:end])
            for q in questions:
                all_questions.append({
                    "set": 1,
                    "question": q.strip(),
                    "metadata": {"batch": len(all_questions) // batch_size},
                })
        except (ValueError, json.JSONDecodeError) as e:
            print(f"  Warning: Failed to parse batch, retrying... ({e})")
            continue

        print(f"  Generated {len(all_questions)}/{n}")

    return all_questions[:n]


def generate_set2(n: int) -> list[dict]:
    """Generate follow-up questions with archetypes."""
    print(f"Generating {n} follow-up questions (Set 2)...")

    batch_size = min(n, 20)
    all_followups = []

    while len(all_followups) < n:
        remaining = n - len(all_followups)
        batch = min(batch_size, remaining)
        raw = call_llm(SET2_SYSTEM, SET2_USER_TEMPLATE.format(n=batch))

        try:
            start = raw.index("[")
            end = raw.rindex("]") + 1
            followups = json.loads(raw[start:end])
            for f in followups:
                all_followups.append({
                    "set": 2,
                    "question": f["question"].strip(),
                    "archetype": f["archetype"].strip(),
                    "metadata": {"batch": len(all_followups) // batch_size},
                })
        except (ValueError, json.JSONDecodeError) as e:
            print(f"  Warning: Failed to parse batch, retrying... ({e})")
            continue

        print(f"  Generated {len(all_followups)}/{n}")

    return all_followups[:n]


def main():
    parser = argparse.ArgumentParser(description="Generate evaluation test prompts")
    parser.add_argument("--num-initial", type=int, default=80,
                        help="Number of initial shopping questions (Set 1)")
    parser.add_argument("--num-followups", type=int, default=60,
                        help="Number of follow-up questions (Set 2)")
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH),
                        help="Output JSONL file path")
    args = parser.parse_args()

    print(f"Using model: {MODEL}")
    print(f"Using API:   {BASE_URL}")
    print()

    set1 = generate_set1(args.num_initial)
    set2 = generate_set2(args.num_followups)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for item in set1 + set2:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n✅ Saved {len(set1)} initial + {len(set2)} follow-up questions to:")
    print(f"   {output_path.absolute()}")
    print(f"\n   Set 1 (initial):   {len(set1)} questions")
    print(f"   Set 2 (follow-up): {len(set2)} questions")

    # Print a few samples
    print("\n── Sample Set 1 Questions ──")
    for item in set1[:5]:
        print(f"   • {item['question']}")

    print("\n── Sample Set 2 Questions ──")
    for item in set2[:5]:
        print(f"   [{item['archetype']}] {item['question']}")


if __name__ == "__main__":
    main()
