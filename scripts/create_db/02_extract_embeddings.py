"""
02_extract_embeddings.py — Visual & Semantic Fingerprinting
=============================================================
Iterates over all products in the catalog and extracts:
- DINOv2 visual embeddings from product images (local disk)
- BGE-M3 semantic embeddings from title + description

Updates LanceDB with real vectors and sets embedding status flags in SQLite.

Local test (RTX 3050 4GB):
    python scripts/02_extract_embeddings.py --limit 20 --batch-size 8

Full run (A100 cluster):
    python scripts/02_extract_embeddings.py --batch-size 64
    python scripts/02_extract_embeddings.py --type visual --batch-size 128
    python scripts/02_extract_embeddings.py --type semantic --batch-size 256
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database import LanceDBCatalog, SQLiteCatalog
from src.embeddings import BGEM3Embedder, DINOv2Embedder

LOCAL_IMAGE_DIR = Path(r"C:\Users\Caleb\Desktop\product_images")


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def extract_visual_embeddings(
    sqlite: SQLiteCatalog,
    lancedb: LanceDBCatalog,
    embedder: DINOv2Embedder,
    batch_size: int = 64,
    limit: int | None = None,
    force: bool = False,
):
    """Extract DINOv2 visual embeddings for all products with images."""
    if force:
        query = "SELECT product_id, image_urls FROM products WHERE image_urls != ''"
    else:
        query = "SELECT product_id, image_urls FROM products WHERE has_visual_embedding = 0 AND image_urls != ''"
        
    if limit:
        query += f" LIMIT {limit}"
    rows = sqlite.conn.execute(query).fetchall()

    if not rows:
        print("[Visual] All products already have visual embeddings")
        return

    print(f"[Visual] Extracting embeddings for {len(rows)} products...")

    for start in tqdm(range(0, len(rows), batch_size), desc="[Visual] Batches"):
        batch = rows[start : start + batch_size]
        product_ids = []
        image_paths = []

        for row in batch:
            pid = row["product_id"]
            rel_path = row["image_urls"].split(",")[0].strip()
            if not rel_path:
                continue

            # Resolve to full local path
            full_path = LOCAL_IMAGE_DIR / rel_path
            if full_path.exists():
                product_ids.append(pid)
                image_paths.append(str(full_path))

        if not image_paths:
            continue

        # Batch embed
        embeddings = embedder.embed_batch(image_paths, show_progress=False)

        # Update LanceDB
        lancedb.upsert_embeddings(product_ids, visual_embeddings=embeddings)

        # Update SQLite status
        for pid in product_ids:
            sqlite.update_embedding_status(pid, visual=True)

        # Free VRAM periodically (important for 4GB GPUs)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"[Visual] Done")


def extract_semantic_embeddings(
    sqlite: SQLiteCatalog,
    lancedb: LanceDBCatalog,
    embedder: BGEM3Embedder,
    batch_size: int = 128,
    limit: int | None = None,
):
    """Extract BGE-M3 semantic embeddings for all products."""
    query = (
        "SELECT product_id, title, description, category, brand, attributes "
        "FROM products WHERE has_semantic_embedding = 0"
    )
    if limit:
        query += f" LIMIT {limit}"
    rows = sqlite.conn.execute(query).fetchall()

    if not rows:
        print("[Semantic] All products already have semantic embeddings")
        return

    print(f"[Semantic] Extracting embeddings for {len(rows)} products...")

    for start in tqdm(range(0, len(rows), batch_size), desc="[Semantic] Batches"):
        batch = rows[start : start + batch_size]
        product_ids = []
        texts = []

        for row in batch:
            pid = row["product_id"]
            # Build embedding text: title + description + category + brand
            parts = [row["title"]]
            if row["description"]:
                parts.append(row["description"][:500])
            if row["category"]:
                parts.append(f"Category: {row['category']}")
            if row["brand"]:
                parts.append(f"Brand: {row['brand']}")

            text = " | ".join(parts)
            product_ids.append(pid)
            texts.append(text)

        if not texts:
            continue

        # Batch embed
        result = embedder.embed_batch(texts, show_progress=False)
        embeddings = result["dense"]

        # Update LanceDB
        lancedb.upsert_embeddings(product_ids, semantic_embeddings=embeddings)

        # Update SQLite status
        for pid in product_ids:
            sqlite.update_embedding_status(pid, semantic=True)

        # Free VRAM
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"[Semantic] Done")


def main():
    parser = argparse.ArgumentParser(description="Extract visual & semantic embeddings")
    parser.add_argument("--config", default="config/settings.yaml", help="Config file path")
    parser.add_argument("--type", choices=["visual", "semantic", "all"], default="all",
                        help="Which embeddings to extract")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size (use 8-16 for 4GB VRAM, 64-128 for A100)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of products to process (for testing)")
    parser.add_argument("--image-dir", type=str, default=None,
                        help="Override local image directory")
    parser.add_argument("--force-visual", action="store_true",
                        help="Force re-extraction of visual embeddings even if SQLite says they are already extracted")
    args = parser.parse_args()

    global LOCAL_IMAGE_DIR
    if args.image_dir:
        LOCAL_IMAGE_DIR = Path(args.image_dir)

    config = load_config(args.config)

    # Initialize databases
    sqlite = SQLiteCatalog(config["databases"]["sqlite"]["path"])
    lancedb = LanceDBCatalog(
        db_path=config["databases"]["lancedb"]["path"],
        table_name=config["databases"]["lancedb"]["table_name"],
        visual_dim=config["embeddings"]["visual"]["dimension"],
        semantic_dim=config["embeddings"]["semantic"]["dimension"],
    )

    total = sqlite.count()
    print(f"Catalog size: {total} products")

    if args.limit:
        print(f"  Testing mode: processing {args.limit} products only")

    # Print GPU info
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU: {gpu} ({vram:.1f} GB VRAM)")
    else:
        print("  WARNING: No GPU detected, running on CPU (will be very slow)")

    if args.type in ("visual", "all"):
        vbs = args.batch_size or config["embeddings"]["visual"]["batch_size"]
        print(f"\n--- Visual Embedding (DINOv2, batch_size={vbs}) ---")
        visual_embedder = DINOv2Embedder(
            model_id=config["embeddings"]["visual"]["model_id"],
            batch_size=vbs,
        )
        extract_visual_embeddings(sqlite, lancedb, visual_embedder, batch_size=vbs, limit=args.limit, force=args.force_visual)

        # Free GPU memory before loading next model
        del visual_embedder
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if args.type in ("semantic", "all"):
        sbs = args.batch_size or config["embeddings"]["semantic"]["batch_size"]
        print(f"\n--- Semantic Embedding (BGE-M3, batch_size={sbs}) ---")
        semantic_embedder = BGEM3Embedder(
            model_id=config["embeddings"]["semantic"]["model_id"],
            batch_size=sbs,
            use_fp16=config["embeddings"]["semantic"]["use_fp16"],
        )
        extract_semantic_embeddings(sqlite, lancedb, semantic_embedder, batch_size=sbs, limit=args.limit)

    # Final stats
    total = sqlite.count()
    visual_done = sqlite.conn.execute("SELECT COUNT(*) FROM products WHERE has_visual_embedding = 1").fetchone()[0]
    semantic_done = sqlite.conn.execute("SELECT COUNT(*) FROM products WHERE has_semantic_embedding = 1").fetchone()[0]
    print(f"\nEmbedding coverage:")
    print(f"   Visual:   {visual_done}/{total} ({100*visual_done/total:.1f}%)")
    print(f"   Semantic: {semantic_done}/{total} ({100*semantic_done/total:.1f}%)")
    print(f"   LanceDB:  {lancedb.count()} vectors")

    sqlite.close()


if __name__ == "__main__":
    main()
