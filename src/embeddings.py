"""
embeddings.py — DINOv2 & BGE-M3 Embedding Wrappers
=====================================================
Extracts visual fingerprints (DINOv2) and semantic fingerprints (BGE-M3)
from product images and text respectively.
"""

from __future__ import annotations

from io import BytesIO
from typing import Optional

import numpy as np
import requests
import torch
from PIL import Image
from tqdm import tqdm


# ================================================================
# DINOv2 — Visual Fingerprint
# ================================================================

class DINOv2Embedder:
    """
    Self-supervised visual feature extractor.
    Produces 768-dim vectors from product images.
    Images are loaded from GCS URLs.
    """

    def __init__(
        self,
        model_id: str = "facebook/dinov2-base",
        device: Optional[str] = None,
        batch_size: int = 64,
    ):
        from transformers import AutoImageProcessor, AutoModel

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size

        print(f"[DINOv2] Loading {model_id} on {self.device}...")
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device)
        self.model.eval()
        self.dimension = self.model.config.hidden_size  # 768

    def _load_image(self, url_or_path: str) -> Optional[Image.Image]:
        """Load image from URL (GCS) or local path."""
        try:
            if url_or_path.startswith("http"):
                response = requests.get(url_or_path, timeout=10)
                response.raise_for_status()
                return Image.open(BytesIO(response.content)).convert("RGB")
            else:
                return Image.open(url_or_path).convert("RGB")
        except Exception as e:
            print(f"[DINOv2] Failed to load image {url_or_path}: {e}")
            return None

    @torch.no_grad()
    def embed_single(self, image_url: str) -> Optional[np.ndarray]:
        """Embed a single image, returns 768-dim vector or None."""
        img = self._load_image(image_url)
        if img is None:
            return None
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        # Use [CLS] token embedding
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()
        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding

    @torch.no_grad()
    def embed_batch(self, image_urls: list[str], show_progress: bool = True) -> np.ndarray:
        """
        Embed a batch of images.
        Returns (N, 768) array. Failed images get zero vectors.
        """
        all_embeddings = []
        iterator = range(0, len(image_urls), self.batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="[DINOv2] Embedding images")

        for start in iterator:
            batch_urls = image_urls[start : start + self.batch_size]
            images = []
            valid_indices = []

            for i, url in enumerate(batch_urls):
                img = self._load_image(url)
                if img is not None:
                    images.append(img)
                    valid_indices.append(i)

            # Initialize batch embeddings with zeros
            batch_embs = np.zeros((len(batch_urls), self.dimension), dtype=np.float32)

            if images:
                inputs = self.processor(images=images, return_tensors="pt").to(self.device)
                outputs = self.model(**inputs)
                cls_embeds = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                # L2 normalize
                norms = np.linalg.norm(cls_embeds, axis=1, keepdims=True)
                norms[norms == 0] = 1
                cls_embeds = cls_embeds / norms

                for j, idx in enumerate(valid_indices):
                    batch_embs[idx] = cls_embeds[j]

            all_embeddings.append(batch_embs)

        return np.vstack(all_embeddings)


# ================================================================
# BGE-M3 — Semantic Fingerprint
# ================================================================

class BGEM3Embedder:
    """
    Dense + sparse + multi-vector semantic embeddings.
    Produces 1024-dim dense vectors from text.
    Supports all three retrieval modes of BGE-M3.
    """

    def __init__(
        self,
        model_id: str = "BAAI/bge-m3",
        device: Optional[str] = None,
        batch_size: int = 128,
        use_fp16: bool = True,
    ):
        from FlagEmbedding import BGEM3FlagModel

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.use_fp16 = use_fp16

        print(f"[BGE-M3] Loading {model_id}...")
        self.model = BGEM3FlagModel(
            model_id,
            use_fp16=use_fp16,
            device=self.device,
        )
        self.dimension = 1024

    def embed_single(self, text: str) -> dict:
        """
        Embed a single text.
        Returns dict with 'dense', 'sparse', 'colbert' vectors.
        """
        output = self.model.encode(
            [text],
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=True,
        )
        return {
            "dense": output["dense_vecs"][0],        # (1024,)
            "sparse": output["lexical_weights"][0],   # dict of token_id -> weight
            "colbert": output["colbert_vecs"][0],     # (seq_len, 1024)
        }

    def embed_batch(
        self,
        texts: list[str],
        return_sparse: bool = False,
        return_colbert: bool = False,
        show_progress: bool = True,
    ) -> dict:
        """
        Embed a batch of texts.
        Returns dict with 'dense' (N, 1024) array and optionally sparse/colbert.
        """
        all_dense = []
        all_sparse = [] if return_sparse else None
        all_colbert = [] if return_colbert else None

        iterator = range(0, len(texts), self.batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="[BGE-M3] Embedding texts")

        for start in iterator:
            batch = texts[start : start + self.batch_size]
            output = self.model.encode(
                batch,
                return_dense=True,
                return_sparse=return_sparse,
                return_colbert_vecs=return_colbert,
            )
            all_dense.append(output["dense_vecs"])
            if return_sparse:
                all_sparse.extend(output["lexical_weights"])
            if return_colbert:
                all_colbert.extend(output["colbert_vecs"])

        result = {"dense": np.vstack(all_dense)}
        if return_sparse:
            result["sparse"] = all_sparse
        if return_colbert:
            result["colbert"] = all_colbert
        return result

    def embed_query(self, query: str) -> np.ndarray:
        """Quick helper: embed a query and return dense vector."""
        output = self.model.encode([query], return_dense=True)
        return output["dense_vecs"][0]
