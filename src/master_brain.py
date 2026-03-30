"""
master_brain.py — The Master Brain (Qwen 3.5-9B)
==================================================
The Voice and Reasoning Engine.
- Synthesizes Top-10 products into High-EQ recommendations (3-5 items)
- Maintains consistent persona across conversations
- Generates conversation summaries for ConversationalMemory
"""

from __future__ import annotations

import json
from typing import AsyncIterator, Optional, TYPE_CHECKING

import httpx

from src.schema import ChatMessage

if TYPE_CHECKING:
    from src.agent_acts import AgentAct


# ================================================================
# System Prompts
# ================================================================

PERSONA_SYSTEM_PROMPT = """You are a warm, knowledgeable shopping assistant. You combine deep product \
expertise with genuine empathy. You never hallucinate features — you only recommend products from \
the provided context. When uncertain, you ask clarifying questions. You speak naturally, like a \
trusted friend who happens to know everything about shopping.

Key traits:
- Empathetic: understand the user's needs beyond what they explicitly state
- Precise: only cite features that exist in the product data
- Conversational: avoid robotic lists; weave recommendations into natural dialog
- Decisive: confidently recommend the "best pick" with clear reasoning
- Helpful: proactively suggest complementary items or alternatives"""

SYNTHESIS_PROMPT_TEMPLATE = """Based on the user's query and the following products, recommend the best 3-5 items.

**User Query:** {query}

**Available Products (Top 10):**
{products_context}

**Instructions:**
1. Select the 3-5 BEST products that match the user's needs
2. For each recommendation, explain WHY it's a great fit
3. If a product has reviews, reference them naturally
4. Mention any trade-offs honestly
5. End with a follow-up question to refine preferences

{memory_context}

Respond naturally — do NOT use numbered lists unless the user asked for them."""


GROUNDED_SYSTEM_PROMPT = """You are a warm, knowledgeable shopping assistant. You combine deep product \
expertise with genuine empathy.

CRITICAL GROUNDING RULES:
- You MUST only reference products, prices, features, and attributes that appear in the \
[REPORT] section of the grounded context.
- NEVER invent, guess, or hallucinate product details.
- If a product field is missing (no rating, no reviews), say so honestly — do not fabricate.
- Follow ALL instruction acts ([RECOMMEND], [COMPARE], [CLARIFY], [STYLE]) exactly.
- When recommending, explain your reasoning using ONLY the provided data.

Key traits:
- Precise: only cite features that exist in the grounded context
- Conversational: weave recommendations into natural dialog
- Decisive: confidently recommend with clear reasoning tied to product data
- Honest: mention trade-offs, don't oversell"""

GROUNDED_SYNTHESIS_TEMPLATE = """**User Query:** {query}

{acts_context}

{memory_context}

Generate your response following the instruction acts above."""

SUMMARY_PROMPT = """Summarize this conversation between a shopping assistant and a customer. \
Focus on:
1. What products were discussed (include specific names/brands)
2. What the customer's preferences are (style, budget, use case)
3. Any decisions made or items the customer showed interest in
4. Key topics or themes

Keep it concise (3-5 sentences). This summary will be used to recall context in future conversations.

Conversation:
{conversation}"""


class MasterBrain:
    """
    The 9B Master Brain — the reasoning and synthesis engine.
    Communicates with the SGLang server via OpenAI-compatible API.
    Supports both synchronous and streaming responses.
    """

    def __init__(
        self,
        api_base: str = "http://localhost:30001/v1",
        model_name: str = "master_brain",
        api_key: str = "",
        timeout: float = 30.0,
    ):
        self.api_base = api_base.rstrip("/")
        self.model_name = model_name
        self.api_key = api_key
        self.timeout = timeout
        self.client = httpx.Client(timeout=timeout)
        self.async_client = httpx.AsyncClient(timeout=timeout)

    def _request_headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _build_messages(
        self,
        user_query: str,
        products: list[dict],
        chat_history: list[ChatMessage],
        memory_context: str = "",
        extra_instructions: str = "",
    ) -> list[dict]:
        """Build the full message list for the synthesis call."""
        messages = [{"role": "system", "content": PERSONA_SYSTEM_PROMPT}]

        # Add chat history (last N turns from episodic memory)
        for msg in chat_history:
            messages.append({"role": msg.role, "content": msg.content})

        # Format product context
        products_context = self._format_products(products)

        # Build the synthesis prompt
        memory_section = ""
        if memory_context:
            memory_section = f"\n**Previous interactions with this customer:**\n{memory_context}\n"

        synthesis_content = SYNTHESIS_PROMPT_TEMPLATE.format(
            query=user_query,
            products_context=products_context,
            memory_context=memory_section,
        )
        if extra_instructions:
            synthesis_content += f"\n\n**Additional Instructions:**\n{extra_instructions}"

        messages.append({"role": "user", "content": synthesis_content})
        return messages

    def _format_products(self, products: list[dict]) -> str:
        """Format product list into readable context for the model."""
        lines = []
        for i, p in enumerate(products, 1):
            parts = [f"**{i}. {p.get('title', 'Unknown')}**"]
            if p.get("brand"):
                parts.append(f"   Brand: {p['brand']}")
            if p.get("price") is not None:
                parts.append(f"   Price: ${p['price']:.2f}")
            if p.get("category"):
                parts.append(f"   Category: {p['category']}")
            if p.get("description"):
                desc = p["description"][:300]  # Truncate long descriptions
                parts.append(f"   Description: {desc}")
            if p.get("rating") is not None:
                parts.append(f"   Rating: {p['rating']}/5 ({p.get('review_count', 0)} reviews)")
            if p.get("reviews_summary"):
                parts.append(f"   Reviews: {p['reviews_summary'][:200]}")
            if p.get("image_urls"):
                urls = p["image_urls"] if isinstance(p["image_urls"], list) else p["image_urls"].split(",")
                if urls and urls[0]:
                    parts.append(f"   Image: {urls[0]}")
            lines.append("\n".join(parts))
        return "\n\n".join(lines)

    def synthesize(
        self,
        user_query: str,
        products: list[dict],
        chat_history: list[ChatMessage] = None,
        memory_context: str = "",
    ) -> str:
        """
        Synchronous synthesis — returns full response.
        Takes Top-10 products and generates High-EQ recommendation.
        """
        messages = self._build_messages(
            user_query, products, chat_history or [], memory_context
        )

        response = self.client.post(
            f"{self.api_base}/chat/completions",
            headers=self._request_headers(),
            json={
                "model": self.model_name,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 2048,
            },
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    async def synthesize_stream(
        self,
        user_query: str,
        products: list[dict],
        chat_history: list[ChatMessage] = None,
        memory_context: str = "",
        extra_instructions: str = "",
    ) -> AsyncIterator[str]:
        """
        Streaming synthesis — yields tokens as they arrive.
        Used by the frontend for real-time display.
        """
        messages = self._build_messages(
            user_query, products, chat_history or [], memory_context, extra_instructions
        )

        async with self.async_client.stream(
            "POST",
            f"{self.api_base}/chat/completions",
            headers=self._request_headers(),
            json={
                "model": self.model_name,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 2048,
                "stream": True,
            },
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        delta = data["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                    except json.JSONDecodeError:
                        continue

    def general_chat(
        self,
        user_message: str,
        chat_history: list[ChatMessage] = None,
        memory_context: str = "",
    ) -> str:
        """Handle general conversation (no product search needed)."""
        messages = [{"role": "system", "content": PERSONA_SYSTEM_PROMPT}]

        for msg in (chat_history or []):
            messages.append({"role": msg.role, "content": msg.content})

        if memory_context:
            user_message += f"\n\n[Context from previous conversations: {memory_context}]"

        messages.append({"role": "user", "content": user_message})

        response = self.client.post(
            f"{self.api_base}/chat/completions",
            headers=self._request_headers(),
            json={
                "model": self.model_name,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 1024,
            },
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    async def general_chat_stream(
        self,
        user_message: str,
        chat_history: list[ChatMessage] = None,
        memory_context: str = "",
        extra_instructions: str = "",
    ) -> AsyncIterator[str]:
        """Streaming general chat."""
        messages = [{"role": "system", "content": PERSONA_SYSTEM_PROMPT}]

        for msg in (chat_history or []):
            messages.append({"role": msg.role, "content": msg.content})

        if memory_context:
            user_message += f"\n\n[Context from previous conversations: {memory_context}]"
        if extra_instructions:
            user_message += f"\n\n[Additional Instructions: {extra_instructions}]"

        messages.append({"role": "user", "content": user_message})

        async with self.async_client.stream(
            "POST",
            f"{self.api_base}/chat/completions",
            headers=self._request_headers(),
            json={
                "model": self.model_name,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 1024,
                "stream": True,
            },
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        delta = data["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                    except json.JSONDecodeError:
                        continue

    def summarize_conversation(self, messages: list[ChatMessage]) -> str:
        """
        Generate a conversation summary for ConversationalMemory.
        Called at session end.
        """
        # Format the conversation
        conversation_text = ""
        for msg in messages:
            role = "Customer" if msg.role == "user" else "Assistant"
            conversation_text += f"{role}: {msg.content}\n\n"

        prompt = SUMMARY_PROMPT.format(conversation=conversation_text)

        response = self.client.post(
            f"{self.api_base}/chat/completions",
            headers=self._request_headers(),
            json={
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": "You are a conversation summarizer."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.3,
                "max_tokens": 512,
            },
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    # ================================================================
    # Grounded Synthesis (Acts-Based)
    # ================================================================

    def _build_grounded_messages(
        self,
        user_query: str,
        acts_prompt: str,
        chat_history: list[ChatMessage],
        memory_context: str = "",
        extra_instructions: str = "",
    ) -> list[dict]:
        """Build message list for grounded (acts-based) synthesis."""
        messages = [{"role": "system", "content": GROUNDED_SYSTEM_PROMPT}]

        for msg in chat_history:
            messages.append({"role": msg.role, "content": msg.content})

        memory_section = ""
        if memory_context:
            memory_section = f"\n**Previous interactions with this customer:**\n{memory_context}\n"

        synthesis_content = GROUNDED_SYNTHESIS_TEMPLATE.format(
            query=user_query,
            acts_context=acts_prompt,
            memory_context=memory_section,
        )
        if extra_instructions:
            synthesis_content += f"\n\nAdditional instructions:\n{extra_instructions}"

        messages.append({"role": "user", "content": synthesis_content})
        return messages

    async def grounded_synthesize_stream(
        self,
        user_query: str,
        acts: list["AgentAct"],
        chat_history: list[ChatMessage] = None,
        memory_context: str = "",
        extra_instructions: str = "",
    ) -> AsyncIterator[str]:
        """
        Acts-based streaming synthesis.
        Instead of raw product dicts, this takes structured acts that
        constrain what the LLM can say.

        Usage:
            from src.agent_acts import ActBuilder, acts_to_prompt

            acts = (
                ActBuilder()
                .report(products, query=message, source="local catalog")
                .recommend(products, query=message, max_recs=3)
                .style(tone="warm and conversational")
                .build()
            )
            async for token in brain.grounded_synthesize_stream(
                message, acts, chat_history, memory_context
            ):
                yield token
        """
        from src.agent_acts import acts_to_prompt

        acts_prompt = acts_to_prompt(acts)
        messages = self._build_grounded_messages(
            user_query, acts_prompt, chat_history or [], memory_context, extra_instructions
        )

        async with self.async_client.stream(
            "POST",
            f"{self.api_base}/chat/completions",
            headers=self._request_headers(),
            json={
                "model": self.model_name,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 2048,
                "stream": True,
            },
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        delta = data["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                    except json.JSONDecodeError:
                        continue

    def close(self):
        self.client.close()

    async def aclose(self):
        await self.async_client.aclose()
