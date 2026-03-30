from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import time
from pathlib import Path

import httpx


REPORT_SECTIONS = (
    "catalog_detail_completeness",
    "recommendation_quality",
    "comparison_quality",
    "conversational_eq",
    "closing_helpfulness",
)

LLM_CRITIC_SYSTEM_PROMPT = """You are a strict but fair evaluator for a commerce assistant conversation.

Score each section from 0 to 5 where:
- 5 = excellent
- 4 = strong
- 3 = acceptable
- 2 = weak
- 1 = poor
- 0 = missing or harmful

Rubric:
1. catalog_detail_completeness
   - Did the agent provide concrete product details grounded in the response, such as product name, price, description/features, brand/merchant, ratings/reviews, or trade-offs when relevant?
2. recommendation_quality
   - If the user was shopping or asking for suggestions, did the agent make clear recommendations with rationale and useful trade-offs?
   - If not applicable, mark applicable=false and score=null.
3. comparison_quality
   - If the user asked to compare options, did the agent compare clearly across dimensions like price, quality, fit, style, or value?
   - If not applicable, mark applicable=false and score=null.
4. conversational_eq
   - Was the assistant warm, respectful, clear, attentive to the user's preferences, and emotionally intelligent rather than robotic?
5. closing_helpfulness
   - Did the agent appropriately offer next-step help, such as asking whether the user wants more options, details, or further help, without sounding pushy?

Return ONLY JSON with this shape:
{
  "summary": "overall short critique",
  "overall_score": 0.0,
  "sections": {
    "catalog_detail_completeness": {"applicable": true, "score": 4.0, "review": "..."},
    "recommendation_quality": {"applicable": true, "score": 4.0, "review": "..."},
    "comparison_quality": {"applicable": false, "score": null, "review": "..."},
    "conversational_eq": {"applicable": true, "score": 5.0, "review": "..."},
    "closing_helpfulness": {"applicable": true, "score": 3.0, "review": "..."}
  }
}"""


def _parse_gs_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"Not a GCS URI: {uri}")
    remainder = uri[5:]
    if "/" not in remainder:
        raise ValueError(f"GCS URI must include an object path: {uri}")
    bucket_name, blob_name = remainder.split("/", 1)
    return bucket_name, blob_name


def _download_gcs_text(uri: str) -> str:
    bucket_name, blob_name = _parse_gs_uri(uri)
    from google.cloud import storage

    project_id = os.environ.get("MVP_GCS_PROJECT_ID") or os.environ.get("GCP_PROJECT_ID") or ""
    client_kwargs = {"project": project_id} if project_id else {}
    client = storage.Client(**client_kwargs)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    return blob.download_as_text()


def load_conversation_records(input_source: str | Path) -> list[dict]:
    source = str(input_source)
    raw_text = _download_gcs_text(source) if source.startswith("gs://") else Path(source).read_text(encoding="utf-8")
    return [json.loads(line) for line in raw_text.splitlines() if line.strip()]


def _extract_messages(record: dict) -> tuple[list[str], list[str]]:
    user_messages: list[str] = []
    agent_messages: list[str] = []
    for entry in record.get("conversation", []):
        if "user" in entry:
            user_messages.append(str(entry["user"]))
        if "agent" in entry:
            agent_messages.append(str(entry["agent"]))
    return user_messages, agent_messages


def _conversation_transcript(record: dict) -> str:
    lines: list[str] = []
    for entry in record.get("conversation", []):
        if "user" in entry:
            lines.append(f"User: {entry['user']}")
        elif "agent" in entry:
            lines.append(f"Agent: {entry['agent']}")
    if record.get("inferred_preferences"):
        lines.append(f"Inferred preferences: {json.dumps(record['inferred_preferences'], ensure_ascii=True, sort_keys=True)}")
    return "\n".join(lines)


def _requested_recommendation(user_text: str) -> bool:
    lowered = user_text.lower()
    markers = [
        "recommend",
        "show me",
        "find me",
        "looking for",
        "need a",
        "need some",
        "suggest",
    ]
    return any(marker in lowered for marker in markers)


def _requested_comparison(user_text: str) -> bool:
    lowered = user_text.lower()
    markers = [
        "compare",
        "vs",
        "versus",
        "difference",
        "better",
        "which one",
        "head to head",
    ]
    return any(marker in lowered for marker in markers)


def _section_result(applicable: bool, score: float | None, review: str) -> dict[str, object]:
    return {
        "applicable": applicable,
        "score": score,
        "review": review.strip(),
    }


def _score_catalog_detail(agent_text: str) -> dict[str, object]:
    lowered = agent_text.lower()
    checks = {
        "price": bool(re.search(r"\$\s*\d", agent_text)),
        "description_or_features": any(marker in lowered for marker in ["description", "features", "made from", "great for", "trade-off"]),
        "rating_or_reviews": any(marker in lowered for marker in ["rating", "reviews", "/5"]),
        "multiple_items": bool(re.search(r"(^|\n)\s*[1-5]\.", agent_text)) or lowered.count("price:") >= 2,
        "brand_or_merchant": any(marker in lowered for marker in ["brand", "merchant"]) or bool(re.search(r"\bby [a-z0-9&'\-]+\b", lowered)),
    }
    score = min(5.0, float(sum(1 for value in checks.values() if value)))
    found = [name.replace("_", " ") for name, passed in checks.items() if passed]
    missing = [name.replace("_", " ") for name, passed in checks.items() if not passed]
    review = "Found: " + ", ".join(found) if found else "No concrete catalog details were surfaced."
    if missing:
        review += ". Missing or weak: " + ", ".join(missing)
    return _section_result(True, score, review)


def _score_recommendation_quality(user_text: str, agent_text: str) -> dict[str, object]:
    if not _requested_recommendation(user_text):
        return _section_result(False, None, "No recommendation request detected.")

    lowered = agent_text.lower()
    signals = {
        "specific_items": bool(re.search(r"(^|\n)\s*[1-5]\.", agent_text)) or any(marker in lowered for marker in ["top three", "top picks", "recommendation"]),
        "rationale": any(marker in lowered for marker in ["because", "ideal", "good for", "worth", "best if", "great for", "description", "features"]),
        "tradeoffs": any(marker in lowered for marker in ["trade-off", "downside", "however", "while", "may not"]),
        "constraints_used": any(marker in lowered for marker in ["under $", "budget", "style", "preference", "color", "size"]),
        "actionable_next_step": any(marker in lowered for marker in ["want to explore", "want more", "compare", "more options"]),
    }
    score = min(5.0, float(sum(1 for value in signals.values() if value)))
    found = [name.replace("_", " ") for name, passed in signals.items() if passed]
    missing = [name.replace("_", " ") for name, passed in signals.items() if not passed]
    review = "Recommendation signals present: " + ", ".join(found) if found else "The response did not give clear recommendation signals."
    if missing:
        review += ". Still weak on: " + ", ".join(missing)
    return _section_result(True, score, review)


def _score_comparison_quality(user_text: str, agent_text: str) -> dict[str, object]:
    if not _requested_comparison(user_text):
        return _section_result(False, None, "No comparison request detected.")

    lowered = agent_text.lower()
    signals = {
        "multiple_products": bool(re.search(r"(^|\n)\s*[1-5]\.", agent_text)) or lowered.count("price") >= 2,
        "comparative_language": any(marker in lowered for marker in ["while", "whereas", "better", "more than", "less than", "compared"]),
        "comparison_dimensions": any(marker in lowered for marker in ["price", "quality", "features", "style", "value"]),
        "decision_guidance": any(marker in lowered for marker in ["choose", "best for", "if you want", "go with"]),
    }
    score = min(5.0, float(sum(1 for value in signals.values() if value)) + 1.0)
    found = [name.replace("_", " ") for name, passed in signals.items() if passed]
    missing = [name.replace("_", " ") for name, passed in signals.items() if not passed]
    review = "Comparison signals present: " + ", ".join(found) if found else "The response did not clearly compare the options."
    if missing:
        review += ". Missing: " + ", ".join(missing)
    return _section_result(True, score, review)


def _score_conversational_eq(user_text: str, agent_text: str, preferences: dict[str, object]) -> dict[str, object]:
    lowered = agent_text.lower()
    user_lowered = user_text.lower()
    signals = {
        "warm_tone": any(marker in lowered for marker in ["i found", "here are", "if you're looking", "happy to help", "i can help"]),
        "preference_awareness": any(str(value).lower() in lowered for value in preferences.values()) if preferences else False,
        "constraint_awareness": any(marker in lowered for marker in ["under $", "budget", "style", "preference", "color", "size"]),
        "clear_guidance": any(marker in lowered for marker in ["top", "recommend", "best", "trade-off", "compare"]),
        "responsive_to_user": any(token in lowered for token in re.findall(r"[a-z0-9]+", user_lowered)[:10]),
    }
    score = min(5.0, max(1.0, float(sum(1 for value in signals.values() if value))))
    found = [name.replace("_", " ") for name, passed in signals.items() if passed]
    missing = [name.replace("_", " ") for name, passed in signals.items() if not passed]
    review = "EQ strengths: " + ", ".join(found) if found else "The reply felt generic and not very attuned to the user."
    if missing:
        review += ". Could improve on: " + ", ".join(missing)
    return _section_result(True, score, review)


def _score_closing_helpfulness(agent_text: str) -> dict[str, object]:
    lowered = agent_text.lower().strip()
    signals = {
        "offers_more_help": any(marker in lowered for marker in ["anything else", "want more", "would you like", "want me to", "can help with"]),
        "ends_with_question": lowered.endswith("?") or "?" in lowered[-120:],
        "next_step_prompt": any(marker in lowered for marker in ["compare", "explore more", "specific style", "more options", "details"]),
    }
    score = float(sum(1 for value in signals.values() if value))
    review = "Closing signals: " + ", ".join(name.replace("_", " ") for name, passed in signals.items() if passed)
    if score == 0.0:
        review = "The response did not offer a clear next step or follow-up help."
    return _section_result(True, min(5.0, score + 1.0 if score > 0 else 0.0), review)


def _overall_score(sections: dict[str, dict[str, object]]) -> float:
    scores = [
        float(section["score"])
        for section in sections.values()
        if section.get("applicable") and section.get("score") is not None
    ]
    if not scores:
        return 0.0
    return round(sum(scores) / len(scores), 2)


def evaluate_record_with_python_rubric(record: dict) -> dict[str, object]:
    user_messages, agent_messages = _extract_messages(record)
    user_text = "\n".join(user_messages)
    agent_text = "\n".join(agent_messages)
    preferences = dict(record.get("inferred_preferences") or {})

    sections = {
        "catalog_detail_completeness": _score_catalog_detail(agent_text),
        "recommendation_quality": _score_recommendation_quality(user_text, agent_text),
        "comparison_quality": _score_comparison_quality(user_text, agent_text),
        "conversational_eq": _score_conversational_eq(user_text, agent_text, preferences),
        "closing_helpfulness": _score_closing_helpfulness(agent_text),
    }

    return {
        "mode": "python",
        "summary": "Deterministic rubric based on surfaced product detail, recommendation quality, comparison behavior, EQ, and closing behavior.",
        "overall_score": _overall_score(sections),
        "sections": sections,
    }


def _extract_json(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
    if cleaned.endswith("```"):
        cleaned = cleaned.rsplit("```", 1)[0]
    return json.loads(cleaned.strip())


def _llm_chat_completion(
    *,
    api_base: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_message: str,
    temperature: float = 0.0,
    max_tokens: int = 900,
) -> str:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    response = httpx.post(
        f"{api_base.rstrip('/')}/chat/completions",
        headers=headers,
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        timeout=60.0,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def evaluate_record_with_llm_critic(
    record: dict,
    *,
    api_base: str,
    api_key: str,
    model: str,
) -> dict[str, object]:
    if not api_key:
        raise ValueError("LLM critic mode requires an API key.")

    transcript = _conversation_transcript(record)
    response = _llm_chat_completion(
        api_base=api_base,
        api_key=api_key,
        model=model,
        system_prompt=LLM_CRITIC_SYSTEM_PROMPT,
        user_message=f"Evaluate this commerce conversation.\n\n{transcript}",
    )
    parsed = _extract_json(response)
    return {
        "mode": "llm",
        "summary": parsed.get("summary", ""),
        "overall_score": float(parsed.get("overall_score") or 0.0),
        "sections": parsed.get("sections", {}),
    }


def _aggregate_component_performance(conversation_results: list[dict], result_key: str) -> dict[str, dict[str, object]]:
    component_report: dict[str, dict[str, object]] = {}
    for section in REPORT_SECTIONS:
        section_scores: list[float] = []
        reviews: list[str] = []
        for conversation in conversation_results:
            evaluation = conversation.get(result_key) or {}
            section_payload = (evaluation.get("sections") or {}).get(section)
            if not section_payload or not section_payload.get("applicable"):
                continue
            score = section_payload.get("score")
            if score is None:
                continue
            section_scores.append(float(score))
            review = str(section_payload.get("review") or "").strip()
            if review:
                reviews.append(review)

        component_report[section] = {
            "average_score": round(statistics.mean(section_scores), 2) if section_scores else None,
            "num_applicable_conversations": len(section_scores),
            "sample_reviews": reviews[:3],
        }
    return component_report


def generate_evaluation_report(
    records: list[dict],
    *,
    mode: str = "python",
    api_base: str = "",
    api_key: str = "",
    model: str = "gpt-4o-mini",
    input_source: str = "",
) -> dict[str, object]:
    normalized_mode = mode.strip().lower()
    if normalized_mode not in {"python", "llm", "both"}:
        raise ValueError(f"Unsupported evaluation mode: {mode}")

    conversation_results: list[dict[str, object]] = []
    for record in records:
        item = {
            "user_id": record.get("user_id", ""),
            "session_id": record.get("session_id", ""),
            "type": record.get("type", ""),
            "created_at": record.get("created_at"),
        }
        if normalized_mode in {"python", "both"}:
            item["python_critic"] = evaluate_record_with_python_rubric(record)
        if normalized_mode in {"llm", "both"}:
            item["llm_critic"] = evaluate_record_with_llm_critic(
                record,
                api_base=api_base,
                api_key=api_key,
                model=model,
            )
        conversation_results.append(item)

    report: dict[str, object] = {
        "generated_at": time.time(),
        "input_source": input_source,
        "mode": normalized_mode,
        "num_conversations": len(records),
        "conversations": conversation_results,
    }

    if normalized_mode in {"python", "both"}:
        python_scores = [item["python_critic"]["overall_score"] for item in conversation_results if item.get("python_critic")]
        report["python_critic"] = {
            "overall_average_score": round(statistics.mean(python_scores), 2) if python_scores else None,
            "component_performance": _aggregate_component_performance(conversation_results, "python_critic"),
        }

    if normalized_mode in {"llm", "both"}:
        llm_scores = [item["llm_critic"]["overall_score"] for item in conversation_results if item.get("llm_critic")]
        report["llm_critic"] = {
            "overall_average_score": round(statistics.mean(llm_scores), 2) if llm_scores else None,
            "component_performance": _aggregate_component_performance(conversation_results, "llm_critic"),
            "model": model,
            "api_base": api_base,
        }

    return report


def write_evaluation_report(report: dict[str, object], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate recorded MVP conversations from local JSONL or GCS.")
    parser.add_argument("--input", required=True, help="Local JSONL path or gs://bucket/path JSONL file.")
    parser.add_argument(
        "--mode",
        default="python",
        choices=["python", "llm", "both"],
        help="Evaluation mode: deterministic python rubric, LLM critic, or both.",
    )
    parser.add_argument(
        "--output",
        default="mvp/evaluation/evaluation_report.json",
        help="Output report path.",
    )
    parser.add_argument(
        "--api-base",
        default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        help="OpenAI-compatible API base for LLM critic mode.",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY", ""),
        help="API key for LLM critic mode.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("MVP_EVALUATOR_MODEL", os.environ.get("MVP_MASTER_BRAIN_MODEL", "gpt-4o-mini")),
        help="Model name for LLM critic mode.",
    )

    args = parser.parse_args()

    records = load_conversation_records(args.input)
    report = generate_evaluation_report(
        records,
        mode=args.mode,
        api_base=args.api_base,
        api_key=args.api_key,
        model=args.model,
        input_source=args.input,
    )
    output_path = write_evaluation_report(report, args.output)
    print(f"Wrote evaluation report to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
