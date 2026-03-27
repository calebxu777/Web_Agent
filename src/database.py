"""
database.py — SQLite + LanceDB Helpers
========================================
Provides thin wrappers around SQLite (metadata/filters) and LanceDB (vector search).
Products are indexed once and queried through both systems.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Optional

import lancedb
import numpy as np
import pyarrow as pa
import yaml


def load_config(config_path: str = "config/settings.yaml") -> dict:
    """Load the central YAML configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ================================================================
# SQLite Catalog
# ================================================================

class SQLiteCatalog:
    """
    Stores product metadata for hard-filter queries (price, stock, brand, etc.).
    This is the relational backbone — vector search happens in LanceDB.
    """

    def __init__(self, db_path: str = "data/processed/catalog.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS products (
                product_id    TEXT PRIMARY KEY,
                title         TEXT NOT NULL,
                description   TEXT,
                category      TEXT,
                subcategory   TEXT,
                price         REAL,
                currency      TEXT DEFAULT 'USD',
                brand         TEXT,
                image_urls    TEXT,
                attributes    TEXT,
                reviews_summary TEXT,
                rating        REAL,
                review_count  INTEGER DEFAULT 0,
                in_stock      INTEGER DEFAULT 1,
                source        TEXT,
                has_visual_embedding  INTEGER DEFAULT 0,
                has_semantic_embedding INTEGER DEFAULT 0,
                dedup_hash    TEXT
            )
        """)
        # Indexes for common filter queries
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_category ON products(category)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_price ON products(price)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_brand ON products(brand)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_in_stock ON products(in_stock)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_dedup ON products(dedup_hash)")
        self.conn.commit()

    def insert_product(self, row: dict) -> bool:
        """Insert a product, skipping duplicates by dedup_hash."""
        existing = self.conn.execute(
            "SELECT 1 FROM products WHERE dedup_hash = ?", (row["dedup_hash"],)
        ).fetchone()
        if existing:
            return False

        cols = ", ".join(row.keys())
        placeholders = ", ".join(["?"] * len(row))
        self.conn.execute(
            f"INSERT OR IGNORE INTO products ({cols}) VALUES ({placeholders})",
            list(row.values()),
        )
        self.conn.commit()
        return True

    def insert_batch(self, rows: list[dict]) -> int:
        """Batch insert, returns count of newly inserted products."""
        inserted = 0
        for row in rows:
            if self.insert_product(row):
                inserted += 1
        return inserted

    def get_product(self, product_id: str) -> Optional[dict]:
        row = self.conn.execute(
            "SELECT * FROM products WHERE product_id = ?", (product_id,)
        ).fetchone()
        return dict(row) if row else None

    def get_products_by_ids(self, product_ids: list[str]) -> list[dict]:
        placeholders = ",".join(["?"] * len(product_ids))
        rows = self.conn.execute(
            f"SELECT * FROM products WHERE product_id IN ({placeholders})", product_ids
        ).fetchall()
        return [dict(r) for r in rows]

    def filter_products(
        self,
        product_ids: list[str],
        in_stock: bool = True,
        max_price: Optional[float] = None,
        category: Optional[str] = None,
        brand: Optional[str] = None,
    ) -> list[dict]:
        """Apply hard filters to a candidate set of product_ids."""
        conditions = ["product_id IN ({})".format(",".join(["?"] * len(product_ids)))]
        params: list[Any] = list(product_ids)

        if in_stock:
            conditions.append("in_stock = 1")
        if max_price is not None:
            conditions.append("price <= ?")
            params.append(max_price)
        if category:
            conditions.append("category = ?")
            params.append(category)
        if brand:
            conditions.append("brand = ?")
            params.append(brand)

        where = " AND ".join(conditions)
        rows = self.conn.execute(
            f"SELECT * FROM products WHERE {where}", params
        ).fetchall()
        return [dict(r) for r in rows]

    def get_random_products_by_category(self, category: str, n: int = 5) -> list[dict]:
        """Pull n random products from a category (used for SFT data generation)."""
        rows = self.conn.execute(
            "SELECT * FROM products WHERE category = ? ORDER BY RANDOM() LIMIT ?",
            (category, n),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_all_categories(self) -> list[str]:
        rows = self.conn.execute(
            "SELECT DISTINCT category FROM products WHERE category != ''"
        ).fetchall()
        return [r["category"] for r in rows]

    def update_embedding_status(self, product_id: str, visual: bool = False, semantic: bool = False):
        updates = []
        params = []
        if visual:
            updates.append("has_visual_embedding = 1")
        if semantic:
            updates.append("has_semantic_embedding = 1")
        if not updates:
            return
        params.append(product_id)
        self.conn.execute(
            f"UPDATE products SET {', '.join(updates)} WHERE product_id = ?", params
        )
        self.conn.commit()

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM products").fetchone()[0]

    def close(self):
        self.conn.close()


# ================================================================
# LanceDB Catalog (Vector Store)
# ================================================================

class LanceDBCatalog:
    """
    Stores product embeddings for ANN (Approximate Nearest Neighbor) search.
    Supports both visual (DINOv2) and semantic (BGE-M3) vectors.
    """

    def __init__(
        self,
        db_path: str = "data/processed/lancedb",
        table_name: str = "product_vectors",
        visual_dim: int = 768,
        semantic_dim: int = 1024,
    ):
        self.db_path = db_path
        self.table_name = table_name
        self.visual_dim = visual_dim
        self.semantic_dim = semantic_dim

        Path(db_path).mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(db_path)
        self._ensure_table()

    def _ensure_table(self):
        """Create the vector table if it doesn't exist."""
        if self.table_name not in self.db.table_names():
            schema = pa.schema([
                pa.field("product_id", pa.string()),
                pa.field("visual_embedding", pa.list_(pa.float32(), self.visual_dim)),
                pa.field("semantic_embedding", pa.list_(pa.float32(), self.semantic_dim)),
            ])
            self.db.create_table(self.table_name, schema=schema)

    @property
    def table(self):
        return self.db.open_table(self.table_name)

    def upsert_embeddings(
        self,
        product_ids: list[str],
        visual_embeddings: Optional[np.ndarray] = None,
        semantic_embeddings: Optional[np.ndarray] = None,
    ):
        """
        Upsert embedding vectors for a batch of products.
        Pass None for embeddings you don't want to update.
        """
        records = []
        for i, pid in enumerate(product_ids):
            record = {"product_id": pid}
            if visual_embeddings is not None:
                record["visual_embedding"] = visual_embeddings[i].tolist()
            else:
                record["visual_embedding"] = [0.0] * self.visual_dim
            if semantic_embeddings is not None:
                record["semantic_embedding"] = semantic_embeddings[i].tolist()
            else:
                record["semantic_embedding"] = [0.0] * self.semantic_dim
            records.append(record)

        # LanceDB merge_insert for upsert behavior
        self.table.merge_insert("product_id").when_matched_update_all().when_not_matched_insert_all().execute(records)

    def search_visual(self, query_vector: list[float], top_k: int = 50) -> list[dict]:
        """ANN search over DINOv2 visual embeddings."""
        results = (
            self.table.search(query_vector, vector_column_name="visual_embedding")
            .metric("cosine")
            .limit(top_k)
            .to_list()
        )
        return results

    def search_semantic(self, query_vector: list[float], top_k: int = 50) -> list[dict]:
        """ANN search over BGE-M3 semantic embeddings."""
        results = (
            self.table.search(query_vector, vector_column_name="semantic_embedding")
            .metric("cosine")
            .limit(top_k)
            .to_list()
        )
        return results

    def count(self) -> int:
        return self.table.count_rows()


# ================================================================
# LanceDB for Memory (User Preferences + Conversation Summaries)
# ================================================================

class LanceDBMemoryStore:
    """
    Separate LanceDB tables for:
    - user_preferences: Semantic memory for personalization
    - conversation_summaries: Cross-session dialog recall
    """

    def __init__(
        self,
        db_path: str = "data/processed/lancedb",
        embedding_dim: int = 1024,
    ):
        self.db = lancedb.connect(db_path)
        self.embedding_dim = embedding_dim
        self._ensure_tables()

    def _ensure_tables(self):
        if "user_preferences" not in self.db.table_names():
            schema = pa.schema([
                pa.field("user_id", pa.string()),
                pa.field("preference_text", pa.string()),
                pa.field("embedding", pa.list_(pa.float32(), self.embedding_dim)),
                pa.field("timestamp", pa.float64()),
            ])
            self.db.create_table("user_preferences", schema=schema)

        if "conversation_summaries" not in self.db.table_names():
            schema = pa.schema([
                pa.field("user_id", pa.string()),
                pa.field("session_id", pa.string()),
                pa.field("summary", pa.string()),
                pa.field("key_products", pa.string()),  # JSON list
                pa.field("key_topics", pa.string()),     # JSON list
                pa.field("embedding", pa.list_(pa.float32(), self.embedding_dim)),
                pa.field("timestamp", pa.float64()),
            ])
            self.db.create_table("conversation_summaries", schema=schema)

    def add_preference(self, user_id: str, text: str, embedding: list[float], timestamp: float):
        table = self.db.open_table("user_preferences")
        table.add([{
            "user_id": user_id,
            "preference_text": text,
            "embedding": embedding,
            "timestamp": timestamp,
        }])

    def search_preferences(self, user_id: str, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        table = self.db.open_table("user_preferences")
        results = (
            table.search(query_embedding, vector_column_name="embedding")
            .where(f"user_id = '{user_id}'")
            .metric("cosine")
            .limit(top_k)
            .to_list()
        )
        return results

    def add_conversation_summary(
        self,
        user_id: str,
        session_id: str,
        summary: str,
        key_products: list[str],
        key_topics: list[str],
        embedding: list[float],
        timestamp: float,
    ):
        table = self.db.open_table("conversation_summaries")
        table.add([{
            "user_id": user_id,
            "session_id": session_id,
            "summary": summary,
            "key_products": json.dumps(key_products),
            "key_topics": json.dumps(key_topics),
            "embedding": embedding,
            "timestamp": timestamp,
        }])

    def search_conversation_history(
        self, user_id: str, query_embedding: list[float], top_k: int = 5
    ) -> list[dict]:
        table = self.db.open_table("conversation_summaries")
        results = (
            table.search(query_embedding, vector_column_name="embedding")
            .where(f"user_id = '{user_id}'")
            .metric("cosine")
            .limit(top_k)
            .to_list()
        )
        # Parse JSON fields
        for r in results:
            r["key_products"] = json.loads(r.get("key_products", "[]"))
            r["key_topics"] = json.loads(r.get("key_topics", "[]"))
        return results
