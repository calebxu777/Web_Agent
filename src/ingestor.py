"""
ingestor.py — Async Web Product Ingestion Pipeline
=====================================================
When a user "thumbs up" a web-searched product, this module
asynchronously ingests it into the local catalog:

1. LLM parses raw scrape → structured metadata
2. Image downloaded → local disk + GCS
3. Embeddings generated → LanceDB
4. Metadata inserted → SQLite

All work runs in a background thread — zero latency to the user.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import threading
from pathlib import Path
from typing import TYPE_CHECKING

import requests
from PIL import Image

if TYPE_CHECKING:
    from src.agent import AgentConfig
    from src.database import LanceDBCatalog, SQLiteCatalog

logger = logging.getLogger(__name__)

# Where web-ingested images are stored locally
LOCAL_WEB_IMAGE_DIR = Path(r"C:\Users\Caleb\Desktop\product_images\web")


# ================================================================
# Image Utilities
# ================================================================

def _download_and_resize(url: str, save_path: Path, size: int = 224) -> bool:
    """Download image from URL, resize, save as JPEG."""
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        img = img.resize((size, size), Image.LANCZOS)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(str(save_path), "JPEG", quality=85)
        return True
    except Exception as e:
        logger.warning(f"Failed to download image {url}: {e}")
        return False


def _upload_image_to_gcs(local_path: Path, gcs_key: str) -> bool:
    """Upload a local image file to GCS."""
    try:
        from google.cloud import storage
        project_id = os.environ.get("GCP_PROJECT_ID", "webagent2026")
        client = storage.Client(project=project_id)
        bucket_name = os.environ.get("GCS_BUCKET_NAME", "web-agent-data-caleb-2026")
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(gcs_key)
        blob.upload_from_filename(str(local_path), content_type="image/jpeg")
        logger.info(f"Uploaded to GCS: gs://{bucket_name}/{gcs_key}")
        return True
    except Exception as e:
        logger.warning(f"GCS upload failed for {gcs_key}: {e}")
        return False


# ================================================================
# LLM Metadata Extraction
# ================================================================

EXTRACT_PROMPT = """You are a product data parser. Given a raw product description scraped from the web,
extract structured attributes as JSON. Return ONLY valid JSON, no explanation.

Raw product data:
Title: {title}
Description: {description}
Price: {price}

Extract:
{{
  "brand": "<brand name or empty string>",
  "category": "<product category like 'Clothing & Accessories', 'Electronics', etc.>",
  "subcategory": "<specific sub-category>",
  "color": "<primary color or empty string>",
  "material": "<primary material or empty string>",
  "garment_type": "<type like 'Jacket', 'Shoes', 'Bag' or empty string>",
  "key_features": "<2-3 key product features, comma separated>"
}}"""


def _extract_attributes_with_llm(
    title: str,
    description: str,
    price: float | None,
    handyman_url: str = "http://localhost:30000/v1/chat/completions",
    model_name: str = "handyman-router",
) -> dict:
    """
    Use the Handyman router LoRA to parse raw web product data
    into structured attributes for the catalog DB.
    Falls back to empty attributes on failure.
    """
    prompt = EXTRACT_PROMPT.format(
        title=title,
        description=description[:500],
        price=price if price else "Unknown",
    )

    try:
        resp = requests.post(
            handyman_url,
            json={
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 256,
            },
            timeout=10,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]

        # Strip markdown code fences if present
        content = content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[-1]
        if content.endswith("```"):
            content = content.rsplit("```", 1)[0]

        return json.loads(content.strip())

    except Exception as e:
        logger.warning(f"LLM attribute extraction failed: {e}")
        return {}


# ================================================================
# Core Ingestion
# ================================================================

def _ingest_product_sync(
    product_data: dict,
    agent_config: "AgentConfig",
    sqlite_catalog: "SQLiteCatalog",
    lancedb_catalog: "LanceDBCatalog",
):
    """
    Synchronous ingestion — called from a background thread.
    1) Extract structured attributes via Handyman LLM
    2) Download + resize product image → local disk
    3) Optionally upload image to GCS
    4) Generate embeddings (BGE-M3 + DINOv2)
    5) Insert into SQLite + LanceDB
    """
    title = product_data.get("title", "")
    description = product_data.get("description", "")
    price = product_data.get("price")
    source_url = product_data.get("url", "")
    image_url = product_data.get("image", "")

    if not title:
        logger.warning("Skipping ingestion: no title")
        return False

    # ---- Deterministic product ID based on source URL ----
    url_hash = hashlib.sha256(source_url.encode()).hexdigest()[:12]
    product_id = f"web_{url_hash}"
    dedup_hash = hashlib.sha256(
        f"{title.lower().strip()}|{source_url}".encode()
    ).hexdigest()[:16]

    # Check if already ingested
    existing = sqlite_catalog.get_product(product_id)
    if existing:
        logger.info(f"Product {product_id} already exists, skipping")
        return False

    logger.info(f"Ingesting web product: {title[:60]}...")

    # ---- Step 1: Extract structured metadata via LLM ----
    attributes = _extract_attributes_with_llm(title, description, price)
    category = attributes.pop("category", "")
    subcategory = attributes.pop("subcategory", "")
    brand = attributes.pop("brand", "")

    # ---- Step 2: Download image locally ----
    local_image_path = ""
    if image_url:
        img_filename = f"{product_id}.jpg"
        save_path = LOCAL_WEB_IMAGE_DIR / img_filename
        if _download_and_resize(image_url, save_path):
            local_image_path = f"web/{img_filename}"
            logger.info(f"  Image saved: {save_path}")

            # ---- Step 3: Upload to GCS if configured ----
            if agent_config.image_storage_provider == "gcs":
                gcs_key = f"web/{img_filename}"
                _upload_image_to_gcs(save_path, gcs_key)

    # ---- Step 4: Insert metadata into SQLite ----
    row = {
        "product_id": product_id,
        "title": title[:500],
        "description": description[:2000],
        "category": category,
        "subcategory": subcategory,
        "price": price,
        "currency": "USD",
        "brand": brand,
        "image_urls": local_image_path,
        "attributes": json.dumps(attributes),
        "reviews_summary": "",  # Web products don't have reviews at ingest time
        "rating": None,
        "review_count": 0,
        "in_stock": 1,
        "source": "web_ingested",
        "has_visual_embedding": 0,
        "has_semantic_embedding": 0,
        "dedup_hash": dedup_hash,
    }
    inserted = sqlite_catalog.insert_product(row)
    if not inserted:
        logger.info(f"  Duplicate detected via dedup_hash, skipping")
        return False

    logger.info(f"  Inserted into SQLite: {product_id}")

    # ---- Step 5: Generate embeddings and upsert into LanceDB ----
    try:
        from src.embeddings import BGEM3Embedder, DINOv2Embedder
        import numpy as np

        # Semantic embedding
        text_embedder = BGEM3Embedder()
        text_input = f"{title} | {description[:300]}"
        text_result = text_embedder.embed_batch([text_input], show_progress=False)
        semantic_emb = text_result["dense"]

        # Visual embedding (if image was downloaded)
        visual_emb = None
        if local_image_path:
            visual_embedder = DINOv2Embedder()
            img_path = LOCAL_WEB_IMAGE_DIR / f"{product_id}.jpg"
            visual_emb = visual_embedder.embed_batch(
                [str(img_path)], show_progress=False
            )

        lancedb_catalog.upsert_embeddings(
            product_ids=[product_id],
            visual_embeddings=visual_emb,
            semantic_embeddings=semantic_emb,
        )

        # Update embedding status in SQLite
        sqlite_catalog.update_embedding_status(
            product_id,
            visual=visual_emb is not None,
            semantic=True,
        )

        logger.info(f"  Embeddings stored in LanceDB")

    except Exception as e:
        logger.warning(f"  Embedding generation failed: {e}")
        # Product is still in SQLite, just without vectors

    logger.info(f"✅ Web product ingestion complete: {product_id}")
    return True


# ================================================================
# Public API — Fire-and-Forget
# ================================================================

def ingest_web_product_async(
    product_data: dict,
    agent_config: "AgentConfig",
    sqlite_catalog: "SQLiteCatalog",
    lancedb_catalog: "LanceDBCatalog",
) -> None:
    """
    Fire-and-forget ingestion of a web product.
    Spawns a daemon thread so it completes even if the user closes the page.
    The main request returns immediately with zero added latency.

    Args:
        product_data: dict with keys: title, description, price, url, image
        agent_config: AgentConfig with storage provider settings
        sqlite_catalog: SQLiteCatalog instance
        lancedb_catalog: LanceDBCatalog instance
    """
    thread = threading.Thread(
        target=_ingest_product_sync,
        args=(product_data, agent_config, sqlite_catalog, lancedb_catalog),
        daemon=False,  # Non-daemon so it finishes even if main thread exits
        name=f"ingest-{product_data.get('title', 'unknown')[:30]}",
    )
    thread.start()
    logger.info(
        f"Spawned background ingestion thread for: {product_data.get('title', 'N/A')[:50]}"
    )


def ingest_batch_async(
    products: list[dict],
    agent_config: "AgentConfig",
    sqlite_catalog: "SQLiteCatalog",
    lancedb_catalog: "LanceDBCatalog",
) -> None:
    """Ingest multiple web products asynchronously (one thread each)."""
    for product in products:
        ingest_web_product_async(product, agent_config, sqlite_catalog, lancedb_catalog)
