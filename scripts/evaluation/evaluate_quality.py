"""
evaluate_quality.py
===================
LLM-as-Judge evaluation pipeline for the Commerce Agent.

Reads conversations.jsonl (produced by run_evaluation.py), sends each conversation
to a large critic LLM, and evaluates the agent across a multi-aspect rubric.

Rubric Aspects (each scored 1-10):
  1. Emotional Intelligence (High EQ)
  2. Query Relevance (Actually Answering the Question)
  3. Customer Service Standards
  4. Product Knowledge & Accuracy
  5. Conversational Flow & Context Retention
  6. Helpfulness & Proactivity
  7. Handling Constraints (Budget, Preferences)

Output:
  - Per-conversation scores + comments
  - Aggregate report with average scores, latency analysis, and overall grade

Usage:
  set OPENAI_API_KEY=sk-...
  set CRITIC_MODEL=gpt-4o           (optional, defaults to gpt-4o)

  python scripts/evaluation/evaluate_quality.py --input results/conversations_sim_20260328.jsonl
  python scripts/evaluation/evaluate_quality.py --input results/conversations_sim_20260328.jsonl --model gpt-4o-mini
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx

# ── Config ──────────────────────────────────────────────────────────
API_KEY = os.environ.get("OPENAI_API_KEY", "")
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
CRITIC_MODEL = os.environ.get("CRITIC_MODEL", "gpt-4o")

EVAL_DIR = Path(__file__).parent
RESULTS_DIR = EVAL_DIR / "results"

# ── Rubric Definition ───────────────────────────────────────────────

RUBRIC = [
    {
        "id": "emotional_intelligence",
        "name": "Emotional Intelligence (High EQ)",
        "description": (
            "Does the agent demonstrate empathy, warmth, and emotional awareness? "
            "Does it acknowledge user frustrations (e.g., price too high) with understanding "
            "rather than robotic responses? Does it celebrate good finds with genuine enthusiasm? "
            "Does it adapt its tone to the user's mood?"
        ),
    },
    {
        "id": "query_relevance",
        "name": "Query Relevance & Accuracy",
        "description": (
            "Does the agent actually answer the user's question? Are the product recommendations "
            "relevant to what the user asked for? Does it address specific criteria (color, size, "
            "brand, price range) mentioned by the user? Does it avoid recommending irrelevant products?"
        ),
    },
    {
        "id": "customer_service",
        "name": "Customer Service Standards",
        "description": (
            "Does the agent follow good customer service practices? Is it polite, professional, "
            "and helpful? Does it offer alternatives when the exact request can't be fulfilled? "
            "Does it avoid being pushy, dismissive, or condescending? Does it handle 'difficult' "
            "user requests gracefully?"
        ),
    },
    {
        "id": "product_knowledge",
        "name": "Product Knowledge & Detail",
        "description": (
            "Does the agent provide useful, specific product details (materials, features, "
            "use cases, pros/cons)? Does it demonstrate genuine understanding of the products? "
            "Are comparisons between products informative and fair? Does it avoid generic or "
            "clearly hallucinated product information?"
        ),
    },
    {
        "id": "context_retention",
        "name": "Conversational Flow & Context Retention",
        "description": (
            "Does the agent remember what the user said earlier in the conversation? "
            "Does it build on previous turns rather than starting fresh each time? "
            "Are follow-up responses coherent with the conversation history? "
            "Does the conversation feel natural and connected?"
        ),
    },
    {
        "id": "helpfulness",
        "name": "Helpfulness & Proactivity",
        "description": (
            "Does the agent go above and beyond to help? Does it proactively suggest things "
            "the user might not have thought of (e.g., 'you might also want to consider...')? "
            "Does it explain trade-offs clearly? Does it help the user make a decision rather "
            "than just listing options?"
        ),
    },
    {
        "id": "constraint_handling",
        "name": "Handling Constraints (Budget, Preferences)",
        "description": (
            "When the user specifies constraints (price limit, brand preference, color, size), "
            "does the agent respect them? Does it acknowledge when it can't perfectly match all "
            "constraints? Does it offer the closest alternatives? Does it avoid recommending "
            "products that clearly violate the user's stated constraints?"
        ),
    },
]

# ── Critic Prompt ───────────────────────────────────────────────────

CRITIC_SYSTEM = """You are an expert evaluator for AI shopping assistants. Your job is to critically evaluate a conversation between a user and an AI commerce agent.

You will be given:
1. A complete multi-turn conversation
2. A rubric with 7 evaluation aspects

For EACH aspect in the rubric, you must:
1. Carefully analyze the agent's behavior in the conversation
2. Assign a score from 1 to 10 (1 = terrible, 5 = mediocre, 10 = exceptional)
3. Write a brief but specific comment (1-3 sentences) explaining your score, referencing specific moments in the conversation

Be honest and critical. Do NOT give everything high scores. Differentiate between mediocre and excellent performance.

Scoring guide:
  1-3: Poor — Major issues, harmful or unhelpful responses
  4-5: Below Average — Misses key aspects, generic responses
  6-7: Good — Solid performance with room for improvement
  8-9: Excellent — Strong performance, minor nitpicks only
  10:  Exceptional — Could not be meaningfully improved

You MUST output valid JSON in exactly this format (no markdown, no extra text):
{
  "scores": {
    "ASPECT_ID": {
      "score": NUMBER_1_TO_10,
      "comment": "Your specific comment here"
    }
  },
  "overall_comment": "A 2-3 sentence summary of the agent's overall performance in this conversation."
}"""

CRITIC_USER_TEMPLATE = """Here is the conversation to evaluate:

{conversation_text}

---

Evaluate the agent's responses using this rubric:

{rubric_text}

Output your evaluation as JSON:"""


def format_conversation_for_critic(record: dict) -> str:
    """Format a conversation record into readable text for the critic LLM."""
    lines = []
    lines.append(f"[Conversation ID: {record['conversation_id']}]")
    lines.append(f"[Total turns: {record['num_turns']}]")
    lines.append("")

    for entry in record["dialogue"]:
        role = entry["role"].upper()
        content = entry["content"]

        if role == "USER":
            lines.append(f"USER: {content}")
        else:
            latency = entry.get("latency_ms")
            products = entry.get("products_shown", 0)
            meta_parts = []
            if latency is not None:
                meta_parts.append(f"response time: {latency}ms")
            if products > 0:
                meta_parts.append(f"{products} products shown")
            meta_str = f" [{', '.join(meta_parts)}]" if meta_parts else ""

            lines.append(f"AGENT{meta_str}: {content}")

        lines.append("")

    return "\n".join(lines)


def format_rubric() -> str:
    """Format the rubric into text for the critic prompt."""
    lines = []
    for i, aspect in enumerate(RUBRIC, 1):
        lines.append(f"{i}. **{aspect['name']}** (ID: {aspect['id']})")
        lines.append(f"   {aspect['description']}")
        lines.append("")
    return "\n".join(lines)


def call_critic(conversation_text: str, rubric_text: str) -> dict:
    """Send a conversation to the critic LLM for evaluation."""
    if not API_KEY:
        print("ERROR: Set OPENAI_API_KEY environment variable.")
        sys.exit(1)

    url = f"{BASE_URL.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": CRITIC_MODEL,
        "messages": [
            {"role": "system", "content": CRITIC_SYSTEM},
            {"role": "user", "content": CRITIC_USER_TEMPLATE.format(
                conversation_text=conversation_text,
                rubric_text=rubric_text,
            )},
        ],
        "temperature": 0.3,  # Low temp for consistent grading
        "max_tokens": 2048,
    }

    resp = httpx.post(url, json=payload, headers=headers, timeout=90)
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"]["content"].strip()

    # Parse JSON from the response (handle potential markdown wrapping)
    try:
        # Try direct parse first
        return json.loads(raw)
    except json.JSONDecodeError:
        # LLM may have wrapped in ```json ... ```
        if "```" in raw:
            start = raw.index("```") + 3
            if raw[start:start + 4] == "json":
                start += 4
            end = raw.rindex("```")
            return json.loads(raw[start:end].strip())
        raise ValueError(f"Could not parse critic response as JSON:\n{raw[:500]}")


def evaluate_conversations(conversations_path: Path) -> dict:
    """Run the critic LLM on all conversations and build the report."""
    records = []
    with open(conversations_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        print("ERROR: No conversations found in the file.")
        sys.exit(1)

    print(f"Loaded {len(records)} conversations")
    print(f"Critic model: {CRITIC_MODEL}")
    print(f"API: {BASE_URL}")
    print()

    rubric_text = format_rubric()
    conversation_evaluations = []
    all_latencies = []

    for i, record in enumerate(records):
        convo_id = record["conversation_id"][:8]
        num_turns = record["num_turns"]

        print(f"  [{i + 1}/{len(records)}] Evaluating conversation {convo_id}... "
              f"({num_turns} turns)", end="", flush=True)

        # Collect latencies for the report
        for entry in record["dialogue"]:
            if entry["role"] == "agent" and entry.get("latency_ms") is not None:
                all_latencies.append(entry["latency_ms"])

        conversation_text = format_conversation_for_critic(record)

        try:
            start = time.perf_counter()
            evaluation = call_critic(conversation_text, rubric_text)
            critic_time = time.perf_counter() - start

            # Attach conversation metadata
            eval_record = {
                "conversation_id": record["conversation_id"],
                "num_turns": record["num_turns"],
                "metadata": record.get("metadata", {}),
                "scores": evaluation.get("scores", {}),
                "overall_comment": evaluation.get("overall_comment", ""),
                "critic_latency_s": round(critic_time, 1),
            }
            conversation_evaluations.append(eval_record)

            # Quick peek at scores
            scores = [s["score"] for s in evaluation.get("scores", {}).values() if isinstance(s, dict)]
            avg = sum(scores) / len(scores) if scores else 0
            print(f" → avg {avg:.1f}/10 ({critic_time:.1f}s)")

        except Exception as e:
            print(f" → ERROR: {e}")
            conversation_evaluations.append({
                "conversation_id": record["conversation_id"],
                "num_turns": record["num_turns"],
                "metadata": record.get("metadata", {}),
                "error": str(e),
            })

    # ── Build aggregate report ──
    report = build_report(conversation_evaluations, all_latencies)
    return report


def build_report(evaluations: list[dict], all_latencies: list[int]) -> dict:
    """Build the final aggregate quality report."""

    # Collect per-aspect scores
    aspect_scores = {aspect["id"]: [] for aspect in RUBRIC}
    aspect_comments = {aspect["id"]: [] for aspect in RUBRIC}

    successful_evals = [e for e in evaluations if "error" not in e]

    for eval_record in successful_evals:
        for aspect_id, data in eval_record.get("scores", {}).items():
            if isinstance(data, dict) and "score" in data:
                aspect_scores.setdefault(aspect_id, []).append(data["score"])
                aspect_comments.setdefault(aspect_id, []).append(data.get("comment", ""))

    # Build aspect summaries
    aspect_results = {}
    for aspect in RUBRIC:
        aid = aspect["id"]
        scores = aspect_scores.get(aid, [])
        if scores:
            aspect_results[aid] = {
                "name": aspect["name"],
                "avg_score": round(sum(scores) / len(scores), 2),
                "min_score": min(scores),
                "max_score": max(scores),
                "std_dev": round(
                    (sum((s - sum(scores) / len(scores)) ** 2 for s in scores) / len(scores)) ** 0.5,
                    2,
                ),
                "num_evaluated": len(scores),
                "sample_comments": [c for c in aspect_comments.get(aid, [])[:5] if c],
            }
        else:
            aspect_results[aid] = {
                "name": aspect["name"],
                "avg_score": None,
                "num_evaluated": 0,
            }

    # Overall score
    all_scores = []
    for scores in aspect_scores.values():
        all_scores.extend(scores)

    overall_avg = round(sum(all_scores) / len(all_scores), 2) if all_scores else None

    # Latency analysis
    latency_analysis = {}
    if all_latencies:
        sorted_lat = sorted(all_latencies)
        n = len(sorted_lat)
        latency_analysis = {
            "total_responses": n,
            "avg_ms": round(sum(sorted_lat) / n, 1),
            "median_ms": sorted_lat[n // 2],
            "p90_ms": sorted_lat[int(n * 0.9)],
            "p95_ms": sorted_lat[int(n * 0.95)],
            "p99_ms": sorted_lat[int(n * 0.99)] if n >= 100 else sorted_lat[-1],
            "min_ms": sorted_lat[0],
            "max_ms": sorted_lat[-1],
        }

    # Grade letter
    grade = "N/A"
    if overall_avg is not None:
        if overall_avg >= 9.0:
            grade = "A+"
        elif overall_avg >= 8.0:
            grade = "A"
        elif overall_avg >= 7.0:
            grade = "B"
        elif overall_avg >= 6.0:
            grade = "C"
        elif overall_avg >= 5.0:
            grade = "D"
        else:
            grade = "F"

    return {
        "report_generated_at": datetime.now().isoformat(),
        "critic_model": CRITIC_MODEL,
        "total_conversations": len(evaluations),
        "successful_evaluations": len(successful_evals),
        "failed_evaluations": len(evaluations) - len(successful_evals),
        "overall_score": overall_avg,
        "overall_grade": grade,
        "latency_analysis": latency_analysis,
        "aspect_scores": aspect_results,
        "per_conversation": evaluations,
    }


def print_report(report: dict):
    """Pretty-print the evaluation report to stdout."""
    print(f"\n{'='*60}")
    print(f"  AGENT QUALITY REPORT")
    print(f"{'='*60}")
    print(f"  Critic Model:    {report['critic_model']}")
    print(f"  Conversations:   {report['total_conversations']}")
    print(f"  Evaluated:       {report['successful_evaluations']}")
    print(f"  Failed:          {report['failed_evaluations']}")
    print(f"\n  Overall Score:   {report['overall_score']}/10  (Grade: {report['overall_grade']})")

    print(f"\n  ── Aspect Scores ──")
    for aspect in RUBRIC:
        aid = aspect["id"]
        data = report["aspect_scores"].get(aid, {})
        avg = data.get("avg_score")
        if avg is not None:
            bar = "█" * int(avg) + "░" * (10 - int(avg))
            print(f"    {aspect['name'][:40]:40s}  {avg:5.2f}/10  {bar}")
        else:
            print(f"    {aspect['name'][:40]:40s}  N/A")

    lat = report.get("latency_analysis", {})
    if lat:
        print(f"\n  ── Latency Analysis ──")
        print(f"    Total responses:  {lat['total_responses']}")
        print(f"    Average:          {lat['avg_ms']}ms")
        print(f"    Median:           {lat['median_ms']}ms")
        print(f"    P90:              {lat['p90_ms']}ms")
        print(f"    P95:              {lat['p95_ms']}ms")
        print(f"    Min / Max:        {lat['min_ms']}ms / {lat['max_ms']}ms")

    # Top comments per worst-scoring aspects
    sorted_aspects = sorted(
        [(aid, data) for aid, data in report["aspect_scores"].items()
         if data.get("avg_score") is not None],
        key=lambda x: x[1]["avg_score"],
    )
    if sorted_aspects:
        print(f"\n  ── Areas for Improvement (Lowest Scoring) ──")
        for aid, data in sorted_aspects[:3]:
            print(f"    {data['name']} ({data['avg_score']}/10):")
            for comment in data.get("sample_comments", [])[:2]:
                print(f"      • {comment}")

    print(f"\n{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="LLM-as-Judge quality evaluation for Commerce Agent conversations"
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Path to conversations.jsonl from run_evaluation.py")
    parser.add_argument("--model", type=str, default=None,
                        help="Override critic model (default: CRITIC_MODEL env or gpt-4o)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output report JSON path (default: auto-generated)")
    args = parser.parse_args()

    global CRITIC_MODEL
    if args.model:
        CRITIC_MODEL = args.model

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        print(f"Run run_evaluation.py first to generate conversations.jsonl")
        sys.exit(1)

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = RESULTS_DIR / f"quality_report_{timestamp}.json"

    # Run evaluation
    report = evaluate_conversations(input_path)

    # Print report
    print_report(report)

    # Save report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n  Report saved to: {output_path}")


if __name__ == "__main__":
    main()
