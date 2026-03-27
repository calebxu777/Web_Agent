"""
retrieval.py — Hybrid Search + RRF + Reranking
================================================
Combines BGE-M3 semantic search with DINOv2 visual search
using Reciprocal Rank Fusion (RRF), then applies hard filters via SQLite.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from src.database import LanceDBCatalog, SQLiteCatalog
from src.embeddings import BGEM3Embedder, DINOv2Embedder
from src.router import HandymanRouter
from src.schema import DecomposedQuery


class HybridRetriever:
    """
    The retrieval pipeline:
    1. BGE-M3 → LanceDB Top-50 (text path)
    2. DINOv2 → LanceDB Top-50 (visual path, optional)
    3. Reciprocal Rank Fusion to merge candidate lists
    4. SQLite hard filters (in_stock, price, brand)
    5. Handyman reranks to Top-10
    """

    def __init__(
        self,
        sqlite_catalog: SQLiteCatalog,
        lancedb_catalog: LanceDBCatalog,
        semantic_embedder: BGEM3Embedder,
        visual_embedder: Optional[DINOv2Embedder] = None,
        handyman: Optional[HandymanRouter] = None,
        top_k_initial: int = 50,
        top_k_reranked: int = 10,
        rrf_k: int = 60,
    ):
        self.sqlite = sqlite_catalog
        self.lancedb = lancedb_catalog
        self.semantic_embedder = semantic_embedder
        self.visual_embedder = visual_embedder
        self.handyman = handyman
        self.top_k_initial = top_k_initial
        self.top_k_reranked = top_k_reranked
        self.rrf_k = rrf_k

    def search_text(
        self,
        query: DecomposedQuery,
    ) -> list[dict]:
        """
        Text-based retrieval pipeline.
        Uses BGE-M3 semantic search → hard filters → Handyman rerank.
        """
        # 1. Embed the rewritten query
        search_text = query.rewritten_query or query.original_query
        if query.tags:
            search_text += " " + " ".join(query.tags)
        query_vector = self.semantic_embedder.embed_query(search_text)

        # 2. ANN search in LanceDB
        candidates = self.lancedb.search_semantic(
            query_vector.tolist(), top_k=self.top_k_initial
        )

        # 3. Get product IDs and fetch metadata from SQLite
        candidate_ids = [c["product_id"] for c in candidates]
        products = self._apply_hard_filters(candidate_ids, query.filters)

        # 4. Rerank via Handyman
        if self.handyman and products:
            products = self.handyman.rerank(
                query.original_query, products, top_k=self.top_k_reranked
            )

        return products

    def search_visual(
        self,
        image_embedding: np.ndarray,
        text_query: Optional[str] = None,
        filters: Optional[dict] = None,
    ) -> list[dict]:
        """
        Visual search pipeline.
        Uses DINOv2 visual vectors, optionally fused with BGE-M3 text vectors via RRF.
        """
        # 1. Visual ANN search
        visual_candidates = self.lancedb.search_visual(
            image_embedding.tolist(), top_k=self.top_k_initial
        )

        if text_query:
            # 2. Text ANN search (for hybrid)
            text_vector = self.semantic_embedder.embed_query(text_query)
            text_candidates = self.lancedb.search_semantic(
                text_vector.tolist(), top_k=self.top_k_initial
            )

            # 3. Reciprocal Rank Fusion
            fused = self._reciprocal_rank_fusion(
                [visual_candidates, text_candidates]
            )
        else:
            fused = visual_candidates

        # 4. Get product IDs and apply hard filters
        candidate_ids = [c["product_id"] for c in fused[:self.top_k_initial]]
        products = self._apply_hard_filters(candidate_ids, filters or {})

        # 5. Rerank via Handyman
        if self.handyman and products:
            products = self.handyman.rerank(
                text_query or "similar products", products, top_k=self.top_k_reranked
            )

        return products

    def search_hybrid(
        self,
        text_query: str,
        image_embedding: Optional[np.ndarray] = None,
        tags: list[str] = None,
        filters: dict = None,
    ) -> list[dict]:
        """
        General hybrid search — works with text-only, image-only, or both.
        """
        candidate_lists = []

        # Semantic search
        search_text = text_query
        if tags:
            search_text += " " + " ".join(tags)
        text_vector = self.semantic_embedder.embed_query(search_text)
        text_candidates = self.lancedb.search_semantic(
            text_vector.tolist(), top_k=self.top_k_initial
        )
        candidate_lists.append(text_candidates)

        # Visual search (if image provided)
        if image_embedding is not None:
            visual_candidates = self.lancedb.search_visual(
                image_embedding.tolist(), top_k=self.top_k_initial
            )
            candidate_lists.append(visual_candidates)

        # RRF fusion
        if len(candidate_lists) > 1:
            fused = self._reciprocal_rank_fusion(candidate_lists)
        else:
            fused = candidate_lists[0]

        # Hard filters
        candidate_ids = [c["product_id"] for c in fused[:self.top_k_initial]]
        products = self._apply_hard_filters(candidate_ids, filters or {})

        # Rerank
        if self.handyman and products:
            products = self.handyman.rerank(
                text_query, products, top_k=self.top_k_reranked
            )

        return products

    def _reciprocal_rank_fusion(
        self,
        candidate_lists: list[list[dict]],
    ) -> list[dict]:
        """
        Reciprocal Rank Fusion (RRF) — merges multiple ranked lists.
        Score(d) = Σ 1 / (k + rank_i(d)) for each list i
        """
        scores: dict[str, float] = {}
        product_map: dict[str, dict] = {}

        for candidates in candidate_lists:
            for rank, candidate in enumerate(candidates):
                pid = candidate["product_id"]
                rrf_score = 1.0 / (self.rrf_k + rank + 1)
                scores[pid] = scores.get(pid, 0.0) + rrf_score
                if pid not in product_map:
                    product_map[pid] = candidate

        # Sort by RRF score descending
        sorted_pids = sorted(scores.keys(), key=lambda pid: scores[pid], reverse=True)

        return [product_map[pid] for pid in sorted_pids]

    def _apply_hard_filters(
        self,
        product_ids: list[str],
        filters: dict,
    ) -> list[dict]:
        """Apply SQLite hard filters to candidate products."""
        if not product_ids:
            return []

        return self.sqlite.filter_products(
            product_ids=product_ids,
            in_stock=True,
            max_price=filters.get("price_max"),
            category=filters.get("category"),
            brand=filters.get("brand"),
        )
