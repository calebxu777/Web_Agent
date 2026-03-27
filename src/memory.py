"""
memory.py — Tiered Memory System
==================================
Three memory tiers:
1. EpisodicMemory (Redis): Last 10 turns of current chat session
2. SemanticMemory (LanceDB): Long-term user preference vectors
3. ConversationalMemory (LanceDB): Cross-session dialog summaries for recall
"""

from __future__ import annotations

import json
import time
from typing import Optional

import redis

from src.database import LanceDBMemoryStore
from src.schema import ChatMessage, ConversationSummary, UserPreference


# ================================================================
# Episodic Memory — Redis
# ================================================================

class EpisodicMemory:
    """
    Short-term memory for the current conversation session.
    Stores the last N turns in Redis with TTL-based expiry.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        max_turns: int = 10,
        ttl_seconds: int = 3600,
    ):
        self.max_turns = max_turns
        self.ttl_seconds = ttl_seconds
        self.redis_client = redis.Redis(
            host=host, port=port, db=db, decode_responses=True
        )

    def _key(self, session_id: str) -> str:
        return f"chat:session:{session_id}"

    def add_message(self, session_id: str, message: ChatMessage):
        """Add a message to the session's chat history."""
        key = self._key(session_id)
        msg_data = json.dumps({
            "role": message.role,
            "content": message.content,
            "image_url": message.image_url,
            "timestamp": message.timestamp or time.time(),
        })
        self.redis_client.rpush(key, msg_data)
        # Trim to max_turns (each turn = 1 user + 1 assistant = 2 messages)
        self.redis_client.ltrim(key, -(self.max_turns * 2), -1)
        # Reset TTL
        self.redis_client.expire(key, self.ttl_seconds)

    def get_history(self, session_id: str) -> list[ChatMessage]:
        """Retrieve the chat history for a session."""
        key = self._key(session_id)
        raw_messages = self.redis_client.lrange(key, 0, -1)
        messages = []
        for raw in raw_messages:
            data = json.loads(raw)
            messages.append(ChatMessage(
                role=data["role"],
                content=data["content"],
                image_url=data.get("image_url"),
                timestamp=data.get("timestamp"),
            ))
        return messages

    def get_full_conversation(self, session_id: str) -> list[ChatMessage]:
        """Get ALL messages (not just last N) — used for summarization at session end."""
        # Store full history in a separate key
        full_key = f"chat:full:{session_id}"
        raw_messages = self.redis_client.lrange(full_key, 0, -1)
        messages = []
        for raw in raw_messages:
            data = json.loads(raw)
            messages.append(ChatMessage(
                role=data["role"],
                content=data["content"],
                image_url=data.get("image_url"),
                timestamp=data.get("timestamp"),
            ))
        return messages

    def add_to_full_history(self, session_id: str, message: ChatMessage):
        """Append to full history (no trimming) — for end-of-session summarization."""
        full_key = f"chat:full:{session_id}"
        msg_data = json.dumps({
            "role": message.role,
            "content": message.content,
            "image_url": message.image_url,
            "timestamp": message.timestamp or time.time(),
        })
        self.redis_client.rpush(full_key, msg_data)
        self.redis_client.expire(full_key, self.ttl_seconds * 2)  # Keep longer for summarization

    def clear_session(self, session_id: str):
        """Clear a session's chat history."""
        self.redis_client.delete(self._key(session_id))
        self.redis_client.delete(f"chat:full:{session_id}")

    def session_exists(self, session_id: str) -> bool:
        return self.redis_client.exists(self._key(session_id)) > 0


# ================================================================
# Semantic Memory — LanceDB
# ================================================================

class SemanticMemory:
    """
    Long-term memory for user preferences.
    Stores preference vectors in LanceDB for ANN-based personalization.
    """

    def __init__(self, memory_store: LanceDBMemoryStore):
        self.store = memory_store

    def record_preference(
        self,
        user_id: str,
        preference_text: str,
        embedding: list[float],
    ):
        """Record a user preference (e.g., "prefers minimalist shoes under $150")."""
        self.store.add_preference(
            user_id=user_id,
            text=preference_text,
            embedding=embedding,
            timestamp=time.time(),
        )

    def get_relevant_preferences(
        self,
        user_id: str,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[str]:
        """Retrieve preferences most relevant to the current query."""
        results = self.store.search_preferences(user_id, query_embedding, top_k)
        return [r["preference_text"] for r in results]


# ================================================================
# Conversational Memory — LanceDB
# ================================================================

class ConversationalMemory:
    """
    Cross-session dialog recall.
    At end of each session, summarize the conversation and store with embeddings.
    On new session start, retrieve relevant past summaries for context.

    Enables queries like:
    "Do you remember the red Nike shoes I asked you to recommend?"
    """

    def __init__(
        self,
        memory_store: LanceDBMemoryStore,
        max_summaries_per_user: int = 50,
    ):
        self.store = memory_store
        self.max_summaries_per_user = max_summaries_per_user

    def save_session_summary(
        self,
        user_id: str,
        session_id: str,
        summary: str,
        key_products: list[str],
        key_topics: list[str],
        embedding: list[float],
    ):
        """
        Save a conversation summary at the end of a session.
        Called by the agent when a session ends.
        """
        self.store.add_conversation_summary(
            user_id=user_id,
            session_id=session_id,
            summary=summary,
            key_products=key_products,
            key_topics=key_topics,
            embedding=embedding,
            timestamp=time.time(),
        )

    def recall(
        self,
        user_id: str,
        query_embedding: list[float],
        top_k: int = 3,
    ) -> str:
        """
        Recall past conversation context relevant to the current query.
        Returns a formatted string of past conversation summaries.
        """
        results = self.store.search_conversation_history(
            user_id, query_embedding, top_k
        )

        if not results:
            return ""

        recall_parts = []
        for r in results:
            summary = r["summary"]
            topics = ", ".join(r.get("key_topics", []))
            products = ", ".join(r.get("key_products", []))
            part = f"- {summary}"
            if topics:
                part += f" (Topics: {topics})"
            if products:
                part += f" (Products: {products})"
            recall_parts.append(part)

        return "Past conversation context:\n" + "\n".join(recall_parts)


# ================================================================
# Unified Memory Manager
# ================================================================

class MemoryManager:
    """
    Unified interface for all three memory tiers.
    The agent uses this single class to manage all memory operations.
    """

    def __init__(
        self,
        episodic: EpisodicMemory,
        semantic: SemanticMemory,
        conversational: ConversationalMemory,
    ):
        self.episodic = episodic
        self.semantic = semantic
        self.conversational = conversational

    def on_user_message(self, session_id: str, message: ChatMessage):
        """Called when a user sends a message."""
        self.episodic.add_message(session_id, message)
        self.episodic.add_to_full_history(session_id, message)

    def on_assistant_message(self, session_id: str, message: ChatMessage):
        """Called when the assistant responds."""
        self.episodic.add_message(session_id, message)
        self.episodic.add_to_full_history(session_id, message)

    def get_chat_history(self, session_id: str) -> list[ChatMessage]:
        """Get recent chat history for context."""
        return self.episodic.get_history(session_id)

    def get_memory_context(
        self,
        user_id: str,
        query_embedding: list[float],
    ) -> str:
        """
        Build a combined memory context from semantic preferences
        and past conversation summaries.
        """
        parts = []

        # Semantic preferences
        prefs = self.semantic.get_relevant_preferences(user_id, query_embedding)
        if prefs:
            parts.append("Your known preferences: " + "; ".join(prefs))

        # Past conversation recall
        recall = self.conversational.recall(user_id, query_embedding)
        if recall:
            parts.append(recall)

        return "\n\n".join(parts)

    def end_session(
        self,
        user_id: str,
        session_id: str,
        summary: str,
        key_products: list[str],
        key_topics: list[str],
        summary_embedding: list[float],
    ):
        """
        Called when a session ends.
        Saves conversation summary and cleans up episodic memory.
        """
        self.conversational.save_session_summary(
            user_id=user_id,
            session_id=session_id,
            summary=summary,
            key_products=key_products,
            key_topics=key_topics,
            embedding=summary_embedding,
        )
        self.episodic.clear_session(session_id)
