"""
agent_acts.py — Grounded Response Acts (Production)
=====================================================
Lightweight act system inspired by GenieWorksheets' AgentActs pattern.
Each act type constrains what the Master Brain is allowed to say.

The flow:
  1. After reranking, the agent pipeline converts products → typed acts
  2. Acts are serialized into a structured prompt section
  3. Master Brain generates a response grounded in ONLY the provided acts
  4. (Optional) A validator checks the response against the acts

Act Modes:
  - "hardcoded": Fixed act combo per workflow. Best for OSS models that
                  need strict constraints (Report + Recommend + Style).
  - "dynamic":   Selects acts based on user intent. Best for capable
                  API LLMs that can follow looser constraints.
  - "off":       Bypass acts entirely, use the original free-form synthesis.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


# ================================================================
# Base Act
# ================================================================

@dataclass
class AgentAct:
    """Base class for all agent acts."""

    def to_prompt_block(self) -> str:
        """Serialize this act into a text block for the synthesis prompt."""
        raise NotImplementedError


# ================================================================
# Report Act — "Here are the facts about these products"
# ================================================================

@dataclass
class ReportAct(AgentAct):
    """
    Report grounded product data to the user.
    The LLM MUST only cite attributes present in `product_data`.
    """

    product_data: list[dict] = field(default_factory=list)
    query: str = ""
    source_label: str = ""

    def to_prompt_block(self) -> str:
        if not self.product_data:
            return ""

        lines = [f'[REPORT] Search results for: "{self.query}"']
        if self.source_label:
            lines.append(f"Source: {self.source_label}")
        lines.append("")

        for i, p in enumerate(self.product_data, 1):
            parts = [f"  Product {i}: {p.get('title', 'Unknown')}"]

            if p.get("brand"):
                parts.append(f"    Brand: {p['brand']}")
            if p.get("price") is not None:
                parts.append(f"    Price: ${p['price']:.2f}")
            if p.get("category"):
                parts.append(f"    Category: {p['category']}")
            if p.get("subcategory"):
                parts.append(f"    Subcategory: {p['subcategory']}")
            if p.get("description"):
                parts.append(f"    Description: {p['description'][:300]}")
            if p.get("rating") is not None:
                parts.append(f"    Rating: {p['rating']}/5 ({p.get('review_count', 0)} reviews)")
            if p.get("reviews_summary"):
                parts.append(f"    Review highlights: {p['reviews_summary'][:200]}")
            if p.get("attributes"):
                attrs = p["attributes"]
                if isinstance(attrs, dict):
                    for k, v in list(attrs.items())[:5]:
                        parts.append(f"    {k}: {v}")
            if p.get("source"):
                parts.append(f"    Source: {p['source']}")

            lines.append("\n".join(parts))

        return "\n".join(lines)


# ================================================================
# Recommend Act — "Pick the best N and explain why"
# ================================================================

@dataclass
class RecommendAct(AgentAct):
    """Instruct the LLM to recommend specific products with reasoning."""

    ranked_products: list[dict] = field(default_factory=list)
    user_query: str = ""
    max_recommendations: int = 3
    total_candidates: int = 0
    reasoning_style: str = "conversational"
    highlight_tradeoffs: bool = True
    suggest_alternatives: bool = True

    def to_prompt_block(self) -> str:
        total_candidates = self.total_candidates or len(self.ranked_products)
        lines = [
            f"[RECOMMEND] Select the best {self.max_recommendations} products for the user.",
            f'User\'s request: "{self.user_query}"',
            f"Total surfaced matches in [REPORT]: {total_candidates}",
            f"Style: {self.reasoning_style}",
        ]
        if self.highlight_tradeoffs:
            lines.append("Include honest trade-offs for each recommendation.")
        if self.suggest_alternatives:
            lines.append("If appropriate, suggest alternatives from the catalog.")
        if total_candidates > self.max_recommendations:
            lines.append(
                f"If you only highlight a subset, first acknowledge the broader result set naturally, for example: "
                f"\"I found {total_candidates} strong matches, and among them I'd especially recommend...\""
            )
        lines.append("")
        lines.append("You may ONLY recommend products from the [REPORT] section above.")
        lines.append("Do NOT invent products, prices, or features not listed above.")
        return "\n".join(lines)


# ================================================================
# Compare Act — "Compare these specific products"
# ================================================================

@dataclass
class CompareAct(AgentAct):
    """Instruct the LLM to compare specific products side-by-side."""

    product_indices: list[int] = field(default_factory=list)
    comparison_dimensions: list[str] = field(default_factory=lambda: [
        "price", "quality", "features", "value_for_money"
    ])
    user_query: str = ""
    pick_winner: bool = True

    def to_prompt_block(self) -> str:
        lines = [
            f"[COMPARE] Compare products at positions: {self.product_indices}",
            f"Compare on: {', '.join(self.comparison_dimensions)}",
        ]
        if self.pick_winner:
            lines.append("End with a clear recommendation of which is the better choice and why.")
        if self.user_query:
            lines.append(f'User\'s comparison request: "{self.user_query}"')
        lines.append("")
        lines.append("Use ONLY the data from the [REPORT] section. Do NOT fabricate specs.")
        return "\n".join(lines)


# ================================================================
# Ask Clarification Act — "Ask the user for more info"
# ================================================================

@dataclass
class AskClarificationAct(AgentAct):
    """Instruct the LLM to ask the user a clarifying question."""

    missing_fields: list[str] = field(default_factory=list)
    context: str = ""

    def to_prompt_block(self) -> str:
        lines = ["[CLARIFY] Ask the user to provide more information."]
        if self.missing_fields:
            lines.append(f"Missing info: {', '.join(self.missing_fields)}")
        if self.context:
            lines.append(f"Context: {self.context}")
        lines.append("Ask naturally — don't list fields robotically.")
        return "\n".join(lines)


# ================================================================
# Style Directive Act — tone/format constraints
# ================================================================

@dataclass
class StyleDirectiveAct(AgentAct):
    """Control the tone and format of the response."""

    tone: str = "warm and conversational"
    format_hint: str = ""
    max_length: Optional[int] = None
    include_followup_question: bool = True

    def to_prompt_block(self) -> str:
        lines = [f"[STYLE] Tone: {self.tone}"]
        if self.format_hint:
            lines.append(f"Format: {self.format_hint}")
        if self.max_length:
            lines.append(f"Keep response under ~{self.max_length} words.")
        if self.include_followup_question:
            lines.append("End with a natural follow-up question to refine the user's preferences.")
        return "\n".join(lines)


# ================================================================
# Act Builder — assemble acts for a given turn
# ================================================================

class ActBuilder:
    """
    Builds a list of acts for a single agent turn.

    Usage:
        acts = (
            ActBuilder()
            .report(products, query="red winter jacket", source="local catalog")
            .recommend(products, query="red winter jacket", max_recs=3)
            .style(tone="warm and conversational")
            .build()
        )
    """

    def __init__(self):
        self._acts: list[AgentAct] = []

    def report(self, products: list[dict], query: str = "", source: str = "") -> "ActBuilder":
        self._acts.append(ReportAct(product_data=products, query=query, source_label=source))
        return self

    def recommend(self, products: list[dict], query: str = "", max_recs: int = 3,
                  style: str = "conversational", tradeoffs: bool = True,
                  alternatives: bool = True) -> "ActBuilder":
        self._acts.append(RecommendAct(
            ranked_products=products, user_query=query, max_recommendations=max_recs,
            total_candidates=len(products),
            reasoning_style=style, highlight_tradeoffs=tradeoffs,
            suggest_alternatives=alternatives,
        ))
        return self

    def compare(self, indices: list[int], dimensions: list[str] | None = None,
                query: str = "", pick_winner: bool = True) -> "ActBuilder":
        self._acts.append(CompareAct(
            product_indices=indices,
            comparison_dimensions=dimensions or ["price", "quality", "features", "value_for_money"],
            user_query=query, pick_winner=pick_winner,
        ))
        return self

    def clarify(self, missing: list[str], context: str = "") -> "ActBuilder":
        self._acts.append(AskClarificationAct(missing_fields=missing, context=context))
        return self

    def style(self, tone: str = "warm and conversational", format_hint: str = "",
              max_length: int | None = None, followup: bool = True) -> "ActBuilder":
        self._acts.append(StyleDirectiveAct(
            tone=tone, format_hint=format_hint, max_length=max_length,
            include_followup_question=followup,
        ))
        return self

    def add(self, act: AgentAct) -> "ActBuilder":
        """Add a custom act type."""
        self._acts.append(act)
        return self

    def build(self) -> list[AgentAct]:
        return list(self._acts)


# ================================================================
# Prompt Serialization
# ================================================================

def acts_to_prompt(acts: list[AgentAct]) -> str:
    """Serialize a list of acts into the grounded prompt section."""
    blocks = []
    for act in acts:
        block = act.to_prompt_block()
        if block:
            blocks.append(block)

    header = (
        "=== GROUNDED CONTEXT ===\n"
        "You MUST only reference data provided in the sections below.\n"
        "Do NOT invent, hallucinate, or assume any product details not listed.\n"
        "If a field is missing for a product, do not guess — omit it or say it's not available.\n"
    )

    return header + "\n\n".join(blocks)


def acts_to_metadata(acts: list[AgentAct]) -> dict:
    """Export acts as a JSON-serializable dict for debug/logging."""
    return {
        "act_types": [type(act).__name__ for act in acts],
        "act_count": len(acts),
        "has_report": any(isinstance(a, ReportAct) for a in acts),
        "has_recommend": any(isinstance(a, RecommendAct) for a in acts),
        "has_compare": any(isinstance(a, CompareAct) for a in acts),
        "has_clarify": any(isinstance(a, AskClarificationAct) for a in acts),
    }


# ================================================================
# Act Selector — dynamic vs hardcoded
# ================================================================

# Simple keyword patterns for dynamic intent detection.
# These are intentionally lightweight — no LLM call needed.
_COMPARE_PATTERNS = re.compile(
    r"\b(compare|versus|vs\.?|which\s+(one|is)\s+better|difference\s+between|"
    r"pros\s+and\s+cons|head\s+to\s+head)\b",
    re.IGNORECASE,
)

_CLARIFY_SIGNALS = [
    # Very short queries with no product specificity
    lambda msg: len(msg.split()) <= 2 and not any(c.isdigit() for c in msg),
    # Pure greetings / meta
    lambda msg: msg.strip().lower() in {
        "hi", "hello", "hey", "help", "what can you do",
        "show me something", "anything", "recommendations",
    },
]


def select_acts(
    mode: str,
    message: str,
    products: list[dict],
    source: str = "local catalog",
    is_image_search: bool = False,
) -> list[AgentAct]:
    """
    Select agent acts based on the configured mode.

    Args:
        mode:   "hardcoded" | "dynamic" | "off"
        message: the user's original message
        products: reranked product list
        source: label for result source
        is_image_search: True if this is an image-based workflow

    Returns:
        list[AgentAct] — acts to pass to grounded_synthesize_stream.
        Empty list if mode == "off" (caller should fall back to free-form).
    """
    if mode == "off":
        return []

    if mode == "hardcoded":
        return _hardcoded_acts(message, products, source, is_image_search)

    if mode == "dynamic":
        return _dynamic_acts(message, products, source, is_image_search)

    # Unknown mode — fall back to hardcoded
    return _hardcoded_acts(message, products, source, is_image_search)


def _hardcoded_acts(
    message: str,
    products: list[dict],
    source: str,
    is_image_search: bool,
) -> list[AgentAct]:
    """
    Fixed act combo — always Report + Recommend + Style.
    Strictest constraint. Best for open-source models.
    """
    builder = ActBuilder()

    if products:
        builder.report(products, query=message, source=source)
        builder.recommend(
            products,
            query=message,
            max_recs=min(3, len(products)),
            style="conversational",
            tradeoffs=True,
            alternatives=False,  # Strict: don't suggest things outside the report
        )
    else:
        builder.clarify(
            missing=["product_type"],
            context="No products found matching the query.",
        )

    builder.style(
        tone="warm and conversational",
        followup=True,
        max_length=300,  # Keep OSS model responses concise
    )

    return builder.build()


def _dynamic_acts(
    message: str,
    products: list[dict],
    source: str,
    is_image_search: bool,
) -> list[AgentAct]:
    """
    Dynamically selects acts based on user intent.
    Looser constraints. Best for capable API LLMs (GPT-4o, Claude, etc.).
    """
    builder = ActBuilder()

    # Always ground with a report if we have products
    if products:
        builder.report(products, query=message, source=source)
    else:
        builder.clarify(
            missing=["product_type", "budget", "use_case"],
            context="No products found. Help the user refine their search.",
        )
        builder.style(tone="warm and helpful", followup=True)
        return builder.build()

    # Detect comparison intent
    if _COMPARE_PATTERNS.search(message) and len(products) >= 2:
        # Compare mode — user wants side-by-side analysis
        indices = list(range(1, min(len(products), 4) + 1))
        builder.compare(indices=indices, query=message, pick_winner=True)
        builder.style(
            tone="analytical but friendly",
            format_hint="Use a structured comparison, then give a clear verdict.",
            followup=True,
        )
        return builder.build()

    # Detect vague/underspecified query
    if any(signal(message) for signal in _CLARIFY_SIGNALS):
        builder.clarify(
            missing=["product_type", "budget", "style_preference"],
            context=f"Query was vague: '{message}'. Still showing results but ask for refinement.",
        )
        builder.recommend(
            products, query=message, max_recs=min(2, len(products)),
            style="brief",
            tradeoffs=False,
            alternatives=True,
        )
        builder.style(tone="warm and curious", followup=True)
        return builder.build()

    # Default: standard recommendation
    builder.recommend(
        products,
        query=message,
        max_recs=min(3, len(products)),
        style="conversational",
        tradeoffs=True,
        alternatives=True,  # Dynamic mode allows broader suggestions
    )
    builder.style(tone="warm and conversational", followup=True)

    return builder.build()
