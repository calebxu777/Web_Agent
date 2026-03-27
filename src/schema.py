"""
schema.py — Unified Product Schema
====================================
Pydantic models that normalize products from Amazon, H&M, LVIS, and LAION
into a single schema for indexing in SQLite (metadata) and LanceDB (vectors).
"""

from __future__ import annotations

import hashlib
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, computed_field


# ------------------------------------------------------------------
# Enums
# ------------------------------------------------------------------

class DataSource(str, Enum):
    AMAZON = "amazon"
    HM = "hm"
    LVIS = "lvis"
    LAION = "laion"


class IntentType(str, Enum):
    GENERAL_TALK = "general_talk"
    TEXT_SEARCH = "text_search"
    IMAGE_SEARCH = "image_search"
    WEB_SEARCH = "web_search"


# ------------------------------------------------------------------
# Core Product Schema
# ------------------------------------------------------------------

class UnifiedProduct(BaseModel):
    """
    The single canonical representation of any product in the catalog.
    All dataset-specific formats are mapped into this schema.
    """

    product_id: str = Field(..., description="Globally unique product identifier")
    title: str = Field(..., description="Product title / name")
    description: str = Field(default="", description="Product description text")
    category: str = Field(default="", description="Product category hierarchy")
    subcategory: str = Field(default="", description="More specific category")
    price: Optional[float] = Field(default=None, description="Price in USD")
    currency: str = Field(default="USD")
    brand: str = Field(default="", description="Brand name")

    # Images stored in GCP — DB only stores URLs
    image_urls: list[str] = Field(default_factory=list, description="GCS URLs for product images")

    # Flexible key-value attributes (color, material, size, etc.)
    attributes: dict[str, Any] = Field(default_factory=dict)

    reviews_summary: str = Field(default="", description="Aggregated review summary")
    rating: Optional[float] = Field(default=None, description="Average rating 0-5")
    review_count: int = Field(default=0)

    in_stock: bool = Field(default=True)
    source: DataSource = Field(..., description="Which dataset this product came from")

    # Embedding status flags (filled by extract_embeddings step)
    has_visual_embedding: bool = Field(default=False)
    has_semantic_embedding: bool = Field(default=False)

    @computed_field
    @property
    def dedup_hash(self) -> str:
        """Deterministic hash for deduplication by title + brand."""
        raw = f"{self.title.lower().strip()}|{self.brand.lower().strip()}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def to_sqlite_row(self) -> dict:
        """Flatten to a dict suitable for SQLite insertion."""
        return {
            "product_id": self.product_id,
            "title": self.title,
            "description": self.description,
            "category": self.category,
            "subcategory": self.subcategory,
            "price": self.price,
            "currency": self.currency,
            "brand": self.brand,
            "image_urls": ",".join(self.image_urls),
            "attributes": str(self.attributes),
            "reviews_summary": self.reviews_summary,
            "rating": self.rating,
            "review_count": self.review_count,
            "in_stock": int(self.in_stock),
            "source": self.source.value,
            "has_visual_embedding": int(self.has_visual_embedding),
            "has_semantic_embedding": int(self.has_semantic_embedding),
            "dedup_hash": self.dedup_hash,
        }

    def to_embedding_text(self) -> str:
        """Concatenate title + description for semantic embedding."""
        parts = [self.title]
        if self.description:
            parts.append(self.description)
        if self.category:
            parts.append(f"Category: {self.category}")
        if self.brand:
            parts.append(f"Brand: {self.brand}")
        # Include key attributes
        for k, v in self.attributes.items():
            parts.append(f"{k}: {v}")
        return " | ".join(parts)


# ------------------------------------------------------------------
# Query Decomposition Schema (output of the Handyman Router)
# ------------------------------------------------------------------

class DecomposedQuery(BaseModel):
    """Structured output from the 0.8B Handyman router."""

    intent: IntentType
    original_query: str
    tags: list[str] = Field(default_factory=list, description="Semantic tags: warm, casual, winter, etc.")
    filters: dict[str, Any] = Field(
        default_factory=dict,
        description="Hard filters: price_max, brand, color, category, etc."
    )
    rewritten_query: str = Field(default="", description="Cleaned / rewritten search query")


# ------------------------------------------------------------------
# Chat & Memory Schemas
# ------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: str  # "user" | "assistant" | "system"
    content: str
    image_url: Optional[str] = None
    timestamp: Optional[float] = None


class ConversationSummary(BaseModel):
    """Stored at session end for cross-session recall."""
    user_id: str
    session_id: str
    summary: str
    key_products: list[str] = Field(default_factory=list, description="Product IDs discussed")
    key_topics: list[str] = Field(default_factory=list, description="Topics: shoes, winter gear, etc.")
    timestamp: float


class UserPreference(BaseModel):
    """Semantic memory entry for long-term user preferences."""
    user_id: str
    preference_text: str
    embedding: Optional[list[float]] = None
    timestamp: float
