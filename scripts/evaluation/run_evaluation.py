"""
run_evaluation.py
=================
Multi-turn evaluation harness for the Commerce Agent.

Uses a small LLM ("User Simulator") to generate contextually relevant follow-up
questions instead of random picks. This mimics real conversation flow:

  1. Pick a random initial question from Set 1
  2. Send to agent, collect full response
  3. Sample number of follow-ups from realistic distribution:
       5%  → 0 follow-ups
       10% → 1 follow-up
       20% → 2 follow-ups
       20% → 3 follow-ups
       20% → 4 follow-ups
       15% → 5 follow-ups
       10% → randint(6, 11) follow-ups
  4. For each follow-up:
     - Feed the User Simulator LLM with conversation history + 20 random Set 2 examples
     - The LLM picks/adapts the most contextually relevant follow-up
  5. Save everything to results JSONL

Usage:
  # Set API key for the User Simulator LLM
  set OPENAI_API_KEY=sk-...
  set OPENAI_BASE_URL=https://api.groq.com/openai/v1   (optional)
  set USER_SIM_MODEL=llama-3.1-8b-instant               (optional, defaults to gpt-4o-mini)

  python scripts/evaluation/run_evaluation.py --sessions 50 --backend http://localhost:8000
  python scripts/evaluation/run_evaluation.py --sessions 100 --backend http://34.123.45.67:8000 --no-simulator
"""

import argparse
import json
import os
import random
import time
import uuid
from datetime import datetime
from pathlib import Path

import httpx

# ── File paths ──────────────────────────────────────────────────────
EVAL_DIR = Path(__file__).parent
PROMPTS_PATH = EVAL_DIR / "testing_prompts.jsonl"
RESULTS_DIR = EVAL_DIR / "results"

# ── User Simulator LLM config ─────────────────────────────────────
SIM_API_KEY = os.environ.get("OPENAI_API_KEY", "")
SIM_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
SIM_MODEL = os.environ.get("USER_SIM_MODEL", "gpt-4o-mini")

# ── Follow-up distribution ──────────────────────────────────────────
FOLLOWUP_DISTRIBUTION = {
    0: 0.05,        # 5%  — user satisfied immediately
    1: 0.10,        # 10% — one clarification
    2: 0.20,        # 20% — moderate conversation
    3: 0.20,        # 20% — typical conversation
    4: 0.20,        # 20% — detailed exploration
    5: 0.15,        # 15% — long conversation
    "long": 0.10,   # 10% — extended session (6-11 turns)
}


def sample_num_followups() -> int:
    """Sample the number of follow-up questions from the defined distribution."""
    r = random.random()
    cumulative = 0.0
    for key, prob in FOLLOWUP_DISTRIBUTION.items():
        cumulative += prob
        if r <= cumulative:
            if key == "long":
                return random.randint(6, 11)
            return key
    return 3  # fallback


def load_prompts(path: Path) -> tuple[list[dict], list[dict]]:
    """Load and split prompts into Set 1 (initial) and Set 2 (follow-up)."""
    set1, set2 = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if item["set"] == 1:
                set1.append(item)
            elif item["set"] == 2:
                set2.append(item)

    if not set1:
        raise ValueError(f"No Set 1 questions found in {path}")
    if not set2:
        raise ValueError(f"No Set 2 questions found in {path}")

    return set1, set2


# ── User Simulator ──────────────────────────────────────────────────

USER_SIM_SYSTEM = """You are simulating a real online shopper in a conversation with an AI shopping assistant.

Your job: Given the conversation so far and a list of possible follow-up styles, pick the SINGLE most natural and relevant follow-up question to ask next.

Rules:
1. Your follow-up MUST be relevant to both the initial question AND the assistant's last response.
2. Pick from or adapt one of the provided example follow-ups — do NOT invent a completely new topic.
3. Make it feel natural and conversational (casual tone, first person).
4. Reference specific products, prices, or details from the assistant's response when possible.
5. Output ONLY the follow-up question text, nothing else. No quotes, no explanation."""

USER_SIM_TEMPLATE = """Here is the conversation so far:

{conversation}

Here are {n} possible follow-up styles to choose from:
{examples}

Based on the conversation above, pick or adapt the SINGLE most natural and relevant follow-up question. Reference specifics from the assistant's response if possible.

Your follow-up question:"""


def call_user_simulator(
    conversation_history: list[dict],
    example_followups: list[dict],
) -> tuple[str, str]:
    """
    Use the User Simulator LLM to generate a contextually relevant follow-up.

    Args:
        conversation_history: List of {"role": "user"|"assistant", "content": "..."}
        example_followups: List of Set 2 dicts with "question" and "archetype"

    Returns:
        (generated_question, archetype_hint)
    """
    # Format conversation for the prompt
    convo_text = ""
    for turn in conversation_history:
        role = "🧑 User" if turn["role"] == "user" else "🤖 Assistant"
        # Truncate long assistant responses to keep the prompt small
        content = turn["content"]
        if len(content) > 600:
            content = content[:600] + "..."
        convo_text += f"{role}: {content}\n\n"

    # Format example follow-ups
    examples_text = ""
    for i, ex in enumerate(example_followups, 1):
        examples_text += f"  {i}. [{ex['archetype']}] {ex['question']}\n"

    user_prompt = USER_SIM_TEMPLATE.format(
        conversation=convo_text.strip(),
        n=len(example_followups),
        examples=examples_text.strip(),
    )

    # Call the LLM
    url = f"{SIM_BASE_URL.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {SIM_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": SIM_MODEL,
        "messages": [
            {"role": "system", "content": USER_SIM_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.7,
        "max_tokens": 150,
    }

    try:
        resp = httpx.post(url, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        generated = resp.json()["choices"][0]["message"]["content"].strip()

        # Try to figure out which archetype it's closest to
        # by checking which example it resembles most
        archetype = "Unknown"
        generated_lower = generated.lower()
        archetype_keywords = {
            "Refiner": ["color", "size", "fit", "material", "shade", "version", "style", "come in"],
            "Budgeter": ["cheap", "price", "expensive", "under", "budget", "cost", "afford", "less"],
            "Comparer": ["difference", "compare", "better", "versus", "vs", "which one", "between"],
            "Skeptic": ["review", "quality", "durable", "worth", "return", "rating", "last", "reliable"],
        }
        best_score = 0
        for arch, keywords in archetype_keywords.items():
            score = sum(1 for kw in keywords if kw in generated_lower)
            if score > best_score:
                best_score = score
                archetype = arch

        return generated, archetype

    except Exception as e:
        # Fallback: just pick a random example directly
        fallback = random.choice(example_followups)
        return fallback["question"], fallback["archetype"]


def generate_followup_random(set2: list[dict]) -> tuple[str, str]:
    """Fallback: just pick a random follow-up (no LLM simulator)."""
    pick = random.choice(set2)
    return pick["question"], pick["archetype"]


# ── Agent Communication ────────────────────────────────────────────

def send_message(
    client: httpx.Client,
    backend_url: str,
    message: str,
    session_id: str,
    user_id: str = "eval_user",
) -> dict:
    """
    Send a message to the agent's SSE endpoint and collect the full response.

    Returns dict with: message, response_text, products, pipeline_stages,
                        latency_ms, token_count, error
    """
    result = {
        "message": message,
        "response_text": "",
        "products": [],
        "pipeline_stages": [],
        "latency_ms": 0,
        "token_count": 0,
        "error": None,
    }

    start = time.perf_counter()

    try:
        with client.stream(
            "POST",
            f"{backend_url}/api/chat",
            json={
                "message": message,
                "session_id": session_id,
                "user_id": user_id,
                "hasImage": False,
                "webSearch": False,
            },
            timeout=120,
        ) as response:
            response.raise_for_status()

            buffer = ""
            for chunk in response.iter_text():
                buffer += chunk
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()

                    if line == "data: [DONE]":
                        break
                    if not line.startswith("data: "):
                        continue

                    try:
                        data = json.loads(line[6:])
                    except json.JSONDecodeError:
                        continue

                    if data.get("type") == "status":
                        result["pipeline_stages"].append({
                            "stage": data.get("stage"),
                            "message": data.get("message"),
                        })
                    elif data.get("type") == "products":
                        result["products"] = data.get("items", [])
                    elif data.get("type") == "token":
                        result["response_text"] += data.get("content", "")
                        result["token_count"] += 1
                    elif data.get("type") == "error":
                        result["error"] = data.get("message", "Unknown error")

    except httpx.HTTPStatusError as e:
        result["error"] = f"HTTP {e.response.status_code}: {e.response.text[:200]}"
    except httpx.ConnectError:
        result["error"] = f"Connection refused — is the backend running at {backend_url}?"
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {str(e)}"

    result["latency_ms"] = int((time.perf_counter() - start) * 1000)
    return result


# ── Session Runner ──────────────────────────────────────────────────

def run_session(
    client: httpx.Client,
    backend_url: str,
    set1: list[dict],
    set2: list[dict],
    session_idx: int,
    use_simulator: bool = True,
) -> dict:
    """Run a single evaluation session (initial + LLM-guided follow-ups)."""
    session_id = str(uuid.uuid4())
    num_followups = sample_num_followups()

    # Pick a random initial question
    initial = random.choice(set1)

    print(f"\n{'─'*60}")
    print(f"  Session {session_idx + 1} | {num_followups} follow-ups | "
          f"{'🤖 Simulated' if use_simulator else '🎲 Random'} | ID: {session_id[:8]}...")
    print(f"  Q: {initial['question'][:70]}")
    print(f"{'─'*60}")

    turns = []
    conversation_history = []  # For the User Simulator context

    # ── Turn 1: Initial question ──
    print(f"  [Turn 1] Sending initial question...")
    result = send_message(client, backend_url, initial["question"], session_id)
    turns.append({
        "turn": 1,
        "set": 1,
        "archetype": None,
        "simulator_used": False,
        **result,
    })
    conversation_history.append({"role": "user", "content": initial["question"]})
    conversation_history.append({"role": "assistant", "content": result["response_text"]})

    print(f"    → {result['token_count']} tokens, {len(result['products'])} products, {result['latency_ms']}ms")
    if result["error"]:
        print(f"    ⚠️  Error: {result['error']}")

    # ── Follow-up turns ──
    for i in range(num_followups):
        turn_num = i + 2

        if use_simulator and SIM_API_KEY:
            # Pick 20 random Set 2 examples for the simulator to choose from
            examples = random.sample(set2, min(20, len(set2)))
            question, archetype = call_user_simulator(conversation_history, examples)
            sim_used = True
        else:
            question, archetype = generate_followup_random(set2)
            sim_used = False

        print(f"  [Turn {turn_num}] [{archetype}] {question[:60]}...")

        result = send_message(client, backend_url, question, session_id)
        turns.append({
            "turn": turn_num,
            "set": 2,
            "archetype": archetype,
            "simulator_used": sim_used,
            **result,
        })
        conversation_history.append({"role": "user", "content": question})
        conversation_history.append({"role": "assistant", "content": result["response_text"]})

        print(f"    → {result['token_count']} tokens, {len(result['products'])} products, {result['latency_ms']}ms")
        if result["error"]:
            print(f"    ⚠️  Error: {result['error']}")

    return {
        "session_id": session_id,
        "session_idx": session_idx,
        "initial_question": initial["question"],
        "num_followups": num_followups,
        "total_turns": len(turns),
        "simulator_used": use_simulator and bool(SIM_API_KEY),
        "simulator_model": SIM_MODEL if (use_simulator and SIM_API_KEY) else None,
        "turns": turns,
        "total_latency_ms": sum(t["latency_ms"] for t in turns),
        "total_tokens": sum(t["token_count"] for t in turns),
        "total_products": sum(len(t["products"]) for t in turns),
        "has_error": any(t["error"] for t in turns),
    }


def print_summary(results: list[dict]):
    """Print aggregate evaluation statistics."""
    total_sessions = len(results)
    total_turns = sum(r["total_turns"] for r in results)
    total_tokens = sum(r["total_tokens"] for r in results)
    total_latency = sum(r["total_latency_ms"] for r in results)
    error_sessions = sum(1 for r in results if r["has_error"])
    sim_sessions = sum(1 for r in results if r["simulator_used"])

    followup_dist = {}
    for r in results:
        n = r["num_followups"]
        followup_dist[n] = followup_dist.get(n, 0) + 1

    archetype_counts = {}
    for r in results:
        for t in r["turns"]:
            if t.get("archetype"):
                arch = t["archetype"]
                archetype_counts[arch] = archetype_counts.get(arch, 0) + 1

    avg_latency_per_turn = total_latency / total_turns if total_turns else 0
    avg_tokens_per_turn = total_tokens / total_turns if total_turns else 0

    print(f"\n{'='*60}")
    print(f"  EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"  Sessions:           {total_sessions}")
    print(f"  Simulated sessions: {sim_sessions} / {total_sessions}")
    print(f"  Total turns:        {total_turns}")
    print(f"  Sessions w/ errors: {error_sessions}")
    print(f"  Total tokens:       {total_tokens}")
    print(f"  Total latency:      {total_latency / 1000:.1f}s")
    print(f"  Avg latency/turn:   {avg_latency_per_turn:.0f}ms")
    print(f"  Avg tokens/turn:    {avg_tokens_per_turn:.1f}")

    print(f"\n  Follow-up distribution (actual):")
    for n in sorted(followup_dist.keys()):
        pct = followup_dist[n] / total_sessions * 100
        bar = "█" * int(pct / 2)
        print(f"    {n:2d} follow-ups: {followup_dist[n]:3d} sessions ({pct:5.1f}%) {bar}")

    if archetype_counts:
        print(f"\n  Archetype distribution:")
        total_arch = sum(archetype_counts.values())
        for arch in sorted(archetype_counts.keys()):
            pct = archetype_counts[arch] / total_arch * 100
            print(f"    {arch:12s}: {archetype_counts[arch]:3d} turns ({pct:5.1f}%)")

    print(f"{'='*60}")


# ── Conversation Record Builder ─────────────────────────────────────

def build_conversation_record(session_result: dict) -> dict:
    """
    Transform a raw session result into a clean conversation record
    for the critic LLM to evaluate.

    Each record in conversations.jsonl looks like:
    {
        "conversation_id": "abc123...",
        "num_turns": 4,
        "dialogue": [
            {"role": "user",  "content": "...", "latency_ms": null},
            {"role": "agent", "content": "...", "latency_ms": 1234, "products_shown": 5},
            {"role": "user",  "content": "...", "latency_ms": null},
            {"role": "agent", "content": "...", "latency_ms": 890, "products_shown": 0},
        ],
        "metadata": {
            "total_latency_ms": 2124,
            "avg_latency_ms": 1062.0,
            "total_agent_tokens": 340,
            "num_followups": 1,
            "simulator_used": true,
            "has_error": false
        }
    }
    """
    dialogue = []
    latencies = []

    for turn in session_result["turns"]:
        # User message
        dialogue.append({
            "role": "user",
            "content": turn["message"],
            "latency_ms": None,  # user messages don't have latency
        })

        # Agent response
        agent_entry = {
            "role": "agent",
            "content": turn["response_text"],
            "latency_ms": turn["latency_ms"],
            "products_shown": len(turn.get("products", [])),
        }
        if turn.get("error"):
            agent_entry["error"] = turn["error"]

        dialogue.append(agent_entry)
        latencies.append(turn["latency_ms"])

    return {
        "conversation_id": session_result["session_id"],
        "num_turns": session_result["total_turns"],
        "dialogue": dialogue,
        "metadata": {
            "total_latency_ms": sum(latencies),
            "avg_latency_ms": round(sum(latencies) / max(1, len(latencies)), 1),
            "total_agent_tokens": session_result["total_tokens"],
            "num_followups": session_result["num_followups"],
            "simulator_used": session_result["simulator_used"],
            "has_error": session_result["has_error"],
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run multi-turn evaluation against the Commerce Agent"
    )
    parser.add_argument("--sessions", type=int, default=50,
                        help="Number of evaluation sessions to run")
    parser.add_argument("--backend", type=str, default="http://localhost:8000",
                        help="Backend API URL")
    parser.add_argument("--prompts", type=str, default=str(PROMPTS_PATH),
                        help="Path to testing_prompts.jsonl")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--no-simulator", action="store_true",
                        help="Disable LLM user simulator, use random follow-ups")
    args = parser.parse_args()

    random.seed(args.seed)

    # Load prompts
    prompts_path = Path(args.prompts)
    if not prompts_path.exists():
        print(f"ERROR: Prompts file not found: {prompts_path}")
        print(f"Run question_generation.py first to create it.")
        return

    set1, set2 = load_prompts(prompts_path)
    print(f"Loaded {len(set1)} initial + {len(set2)} follow-up prompts")

    use_simulator = not args.no_simulator
    if use_simulator and not SIM_API_KEY:
        print("WARNING: OPENAI_API_KEY not set — falling back to random follow-ups.")
        print("  Set OPENAI_API_KEY to enable the LLM User Simulator.")
        use_simulator = False

    if use_simulator:
        print(f"User Simulator: {SIM_MODEL} via {SIM_BASE_URL}")
    else:
        print(f"User Simulator: DISABLED (random follow-ups)")

    # Prepare output
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_tag = "sim" if use_simulator else "rand"
    results_path = RESULTS_DIR / f"eval_{mode_tag}_{timestamp}.jsonl"
    conversations_path = RESULTS_DIR / f"conversations_{mode_tag}_{timestamp}.jsonl"
    summary_path = RESULTS_DIR / f"eval_{mode_tag}_{timestamp}_summary.json"

    print(f"Backend:  {args.backend}")
    print(f"Sessions: {args.sessions}")
    print(f"Output:   {results_path}")
    print(f"Convos:   {conversations_path}")

    # Quick health check
    try:
        r = httpx.get(f"{args.backend}/health", timeout=5)
        print(f"Health:   {r.json()}")
    except Exception as e:
        print(f"WARNING: Backend health check failed: {e}")
        print(f"Proceeding anyway...\n")

    # Run evaluation sessions
    results = []
    client = httpx.Client()

    try:
        for i in range(args.sessions):
            session_result = run_session(
                client, args.backend, set1, set2, i,
                use_simulator=use_simulator,
            )
            results.append(session_result)

            # Stream raw results to file (crash-safe)
            with open(results_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(session_result, ensure_ascii=False) + "\n")

            # Write human-readable conversation record for the critic LLM
            conversation_record = build_conversation_record(session_result)
            with open(conversations_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(conversation_record, ensure_ascii=False) + "\n")

    except KeyboardInterrupt:
        print(f"\n\nInterrupted after {len(results)} sessions.")
    finally:
        client.close()

    # Print summary
    if results:
        print_summary(results)

        # Save summary JSON
        summary = {
            "timestamp": timestamp,
            "backend": args.backend,
            "simulator_used": use_simulator,
            "simulator_model": SIM_MODEL if use_simulator else None,
            "num_sessions": len(results),
            "num_turns": sum(r["total_turns"] for r in results),
            "num_errors": sum(1 for r in results if r["has_error"]),
            "total_tokens": sum(r["total_tokens"] for r in results),
            "total_latency_ms": sum(r["total_latency_ms"] for r in results),
            "avg_latency_per_turn_ms": round(
                sum(r["total_latency_ms"] for r in results) /
                max(1, sum(r["total_turns"] for r in results)),
                1,
            ),
        }
        # Followup distribution
        fd = {}
        for r in results:
            k = str(r["num_followups"])
            fd[k] = fd.get(k, 0) + 1
        summary["followup_distribution"] = fd

        # Archetype distribution
        ad = {}
        for r in results:
            for t in r["turns"]:
                if t.get("archetype"):
                    arch = t["archetype"]
                    ad[arch] = ad.get(arch, 0) + 1
        summary["archetype_distribution"] = ad

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\n  Results:       {results_path}")
        print(f"  Conversations: {conversations_path}")
        print(f"  Summary:       {summary_path}")
    else:
        print("No sessions completed.")


if __name__ == "__main__":
    main()
