"""
image_search.py — Florence-2 Image Tagging + Hybrid Visual Search
===================================================================
Pipeline for image-based product search:
1. Florence-2 generates technical tags from uploaded image
2. DINOv2 extracts visual embedding
3. BGE-M3 embeds the tags as text
4. HybridRetriever fuses both signals via RRF
"""

from __future__ import annotations

from io import BytesIO
from typing import Optional

import numpy as np
import requests
import torch
from PIL import Image

from src.embeddings import BGEM3Embedder, DINOv2Embedder
from src.retrieval import HybridRetriever
from src.schema import DecomposedQuery, IntentType


class Florence2Tagger:
    """
    Uses microsoft/Florence-2-large to generate detailed image captions
    and extract technical tags for product images.
    """

    def __init__(
        self,
        model_id: str = "microsoft/Florence-2-large",
        device: Optional[str] = None,
    ):
        from transformers import AutoModelForCausalLM, AutoProcessor

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[Florence-2] Loading {model_id} on {self.device}...")
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()

    def _load_image(self, image_source) -> Image.Image:
        """Load image from URL, path, or PIL Image."""
        if isinstance(image_source, Image.Image):
            return image_source.convert("RGB")
        if isinstance(image_source, str):
            if image_source.startswith("http"):
                response = requests.get(image_source, timeout=10)
                response.raise_for_status()
                return Image.open(BytesIO(response.content)).convert("RGB")
            return Image.open(image_source).convert("RGB")
        if isinstance(image_source, bytes):
            return Image.open(BytesIO(image_source)).convert("RGB")
        raise ValueError(f"Unsupported image source type: {type(image_source)}")

    @torch.no_grad()
    def generate_tags(self, image_source, task: str = "<MORE_DETAILED_CAPTION>") -> dict:
        """
        Run Florence-2 on an image to get a detailed caption and tags.

        Returns:
            dict with 'caption' (str) and 'tags' (list[str])
        """
        image = self._load_image(image_source)

        inputs = self.processor(text=task, images=image, return_tensors="pt").to(self.device)

        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=256,
            num_beams=3,
        )

        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Post-process: extract the caption after the task token
        caption = self.processor.post_process_generation(
            generated_text, task=task, image_size=image.size
        )

        # If caption is a dict (Florence-2 format), extract the text
        if isinstance(caption, dict):
            caption_text = caption.get(task, generated_text)
        else:
            caption_text = str(caption)

        # Extract individual tags from the caption
        tags = self._extract_tags(caption_text)

        return {
            "caption": caption_text,
            "tags": tags,
        }

    def _extract_tags(self, caption: str) -> list[str]:
        """
        Extract semantic tags from a detailed caption.
        Simple heuristic: split on common delimiters and filter.
        """
        import re
        # Split on punctuation and common conjunctions
        words = re.split(r'[,;.\-\s]+', caption.lower())
        # Filter to meaningful terms (3+ chars, not stop words)
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "with", "and", "or",
            "for", "from", "that", "this", "has", "have", "had", "its", "it",
            "can", "not", "but", "also", "very", "much", "more", "some",
        }
        tags = [w.strip() for w in words if len(w.strip()) >= 3 and w.strip() not in stop_words]
        # Deduplicate while preserving order
        seen = set()
        unique_tags = []
        for t in tags:
            if t not in seen:
                seen.add(t)
                unique_tags.append(t)
        return unique_tags


class ImageSearchPipeline:
    """
    Full image search pipeline:
    User uploads image → Florence-2 tags → DINOv2 visual embedding
    → BGE-M3 text embedding of tags → HybridRetriever (RRF fusion)
    """

    def __init__(
        self,
        florence_tagger: Florence2Tagger,
        visual_embedder: DINOv2Embedder,
        semantic_embedder: BGEM3Embedder,
        hybrid_retriever: HybridRetriever,
    ):
        self.tagger = florence_tagger
        self.visual_embedder = visual_embedder
        self.semantic_embedder = semantic_embedder
        self.retriever = hybrid_retriever

    def search(
        self,
        image_source,
        user_text: str = "",
        filters: dict = None,
    ) -> dict:
        """
        Run the full image search pipeline.

        Args:
            image_source: URL, path, bytes, or PIL Image
            user_text: Optional text from the user (e.g., "find me something similar but cheaper")
            filters: Optional hard filters (price_max, brand, etc.)

        Returns:
            dict with 'products', 'tags', 'caption'
        """
        # Step 1: Florence-2 tagging
        tag_result = self.tagger.generate_tags(image_source)
        caption = tag_result["caption"]
        tags = tag_result["tags"]

        # Step 2: DINOv2 visual embedding
        if isinstance(image_source, str):
            visual_embedding = self.visual_embedder.embed_single(image_source)
        else:
            # For bytes or PIL Image, save to temp then embed
            from PIL import Image as PILImage
            import tempfile, os
            img = self.tagger._load_image(image_source)
            tmp_path = os.path.join(tempfile.gettempdir(), "query_image.jpg")
            img.save(tmp_path)
            visual_embedding = self.visual_embedder.embed_single(tmp_path)

        if visual_embedding is None:
            visual_embedding = np.zeros(self.visual_embedder.dimension, dtype=np.float32)

        # Step 3: Build combined text query from tags + user text
        text_query = caption
        if user_text:
            text_query = f"{user_text} | {caption}"

        # Step 4: Hybrid retrieval (RRF fusion of visual + text)
        products = self.retriever.search_hybrid(
            text_query=text_query,
            image_embedding=visual_embedding,
            tags=tags,
            filters=filters or {},
        )

        return {
            "products": products,
            "tags": tags,
            "caption": caption,
        }
