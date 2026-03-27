"""
01_normalize_catalog.py — Dataset Normalization Pipeline
=========================================================
Reads manually-downloaded raw datasets (Amazon, H&M, LVIS, LAION)
and normalizes them into the Unified Product Schema.

Outputs:
- SQLite catalog (data/processed/catalog.db)
- LanceDB vector table with placeholder embeddings (data/processed/lancedb/)

Usage:
    python scripts/01_normalize_catalog.py
    python scripts/01_normalize_catalog.py --source amazon
    python scripts/01_normalize_catalog.py --config config/settings.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import uuid
from pathlib import Path

import yaml
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database import LanceDBCatalog, SQLiteCatalog
from src.schema import DataSource, UnifiedProduct


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


# ================================================================
# Amazon Reviews 2023 Normalizer
# ================================================================

def normalize_amazon(raw_path: str, image_url_prefix: str) -> list[UnifiedProduct]:
    """
    Amazon Reviews 2023 format (JSONL):
    {
        "asin": "...", "title": "...", "description": [...],
        "price": "...", "brand": "...", "categories": [...],
        "imageURLs": [...], "features": [...],
        "averageRating": 4.5, "ratingCount": 123
    }
    """
    products = []
    jsonl_files = list(Path(raw_path).glob("*.jsonl")) + list(Path(raw_path).glob("*.json"))

    if not jsonl_files:
        print(f"[Amazon] No JSONL/JSON files found in {raw_path}")
        return products

    for filepath in jsonl_files:
        print(f"[Amazon] Processing {filepath.name}...")
        with open(filepath, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc=f"  {filepath.name}"):
                try:
                    item = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue

                # Parse price
                price = None
                price_str = item.get("price", "")
                if isinstance(price_str, (int, float)):
                    price = float(price_str)
                elif isinstance(price_str, str):
                    price_str = price_str.replace("$", "").replace(",", "").strip()
                    try:
                        price = float(price_str) if price_str else None
                    except ValueError:
                        price = None

                # Parse description
                desc = item.get("description", [])
                if isinstance(desc, list):
                    desc = " ".join(desc)

                # Parse categories
                categories = item.get("categories", [])
                if isinstance(categories, list) and categories:
                    category = categories[0] if isinstance(categories[0], str) else str(categories[0])
                else:
                    category = ""

                # Parse features as attributes
                features = item.get("features", [])
                attributes = {}
                if isinstance(features, list):
                    for feat in features[:10]:  # Limit to 10 features
                        if isinstance(feat, str) and ":" in feat:
                            k, v = feat.split(":", 1)
                            attributes[k.strip()] = v.strip()

                # Image URLs — stored in GCS, map ASIN to GCS path
                raw_image_urls = item.get("imageURLs", item.get("images", []))
                if isinstance(raw_image_urls, list):
                    image_urls = [
                        f"{image_url_prefix}/amazon/{item.get('asin', '')}/{i}.jpg"
                        for i in range(len(raw_image_urls))
                    ]
                else:
                    image_urls = []

                product = UnifiedProduct(
                    product_id=f"amz_{item.get('asin', uuid.uuid4().hex[:12])}",
                    title=item.get("title", ""),
                    description=desc,
                    category=category,
                    price=price,
                    brand=item.get("brand", ""),
                    image_urls=image_urls,
                    attributes=attributes,
                    rating=item.get("averageRating"),
                    review_count=item.get("ratingCount", 0),
                    in_stock=True,
                    source=DataSource.AMAZON,
                )
                products.append(product)

    print(f"[Amazon] Normalized {len(products)} products")
    return products


# ================================================================
# H&M Fashion Normalizer
# ================================================================

def normalize_hm(raw_path: str, image_url_prefix: str) -> list[UnifiedProduct]:
    """
    H&M Personalized Fashion (CSV):
    articles.csv: article_id, prod_name, product_type_name, product_group_name,
                  colour_group_name, department_name, section_name, detail_desc, ...
    """
    products = []
    articles_path = Path(raw_path) / "articles.csv"

    if not articles_path.exists():
        print(f"[H&M] articles.csv not found in {raw_path}")
        return products

    print(f"[H&M] Processing articles.csv...")
    with open(articles_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc="  articles.csv"):
            article_id = row.get("article_id", "")

            attributes = {}
            for key in ["colour_group_name", "department_name", "index_group_name",
                         "section_name", "garment_group_name"]:
                if row.get(key):
                    attributes[key.replace("_name", "")] = row[key]

            # H&M images: typically stored as <article_id prefix>/<article_id>.jpg
            prefix = article_id[:3] if len(article_id) >= 3 else article_id
            image_urls = [f"{image_url_prefix}/hm/{prefix}/{article_id}.jpg"]

            product = UnifiedProduct(
                product_id=f"hm_{article_id}",
                title=row.get("prod_name", ""),
                description=row.get("detail_desc", ""),
                category=row.get("product_group_name", ""),
                subcategory=row.get("product_type_name", ""),
                brand="H&M",
                image_urls=image_urls,
                attributes=attributes,
                in_stock=True,
                source=DataSource.HM,
            )
            products.append(product)

    print(f"[H&M] Normalized {len(products)} products")
    return products


# ================================================================
# LVIS Normalizer
# ================================================================

def normalize_lvis(raw_path: str, image_url_prefix: str) -> list[UnifiedProduct]:
    """
    LVIS (Large Vocabulary Instance Segmentation) annotations:
    Provides long-tail object categories with COCO images.
    We treat each unique category as a "product" template.
    """
    products = []
    annotation_files = list(Path(raw_path).glob("*.json"))

    if not annotation_files:
        print(f"[LVIS] No JSON files found in {raw_path}")
        return products

    for filepath in annotation_files:
        print(f"[LVIS] Processing {filepath.name}...")
        with open(filepath, "r") as f:
            data = json.load(f)

        categories = {c["id"]: c for c in data.get("categories", [])}
        images = {img["id"]: img for img in data.get("images", [])}

        # Group annotations by category
        category_images: dict[int, list[str]] = {}
        for ann in data.get("annotations", []):
            cat_id = ann["category_id"]
            img_id = ann["image_id"]
            if cat_id not in category_images:
                category_images[cat_id] = []
            img_info = images.get(img_id, {})
            if img_info.get("coco_url"):
                category_images[cat_id].append(img_info["coco_url"])

        for cat_id, cat_info in tqdm(categories.items(), desc=f"  {filepath.name}"):
            cat_name = cat_info.get("name", "unknown")
            synonyms = cat_info.get("synonyms", [])

            img_urls = category_images.get(cat_id, [])[:5]  # Max 5 images per category
            gcs_urls = [
                f"{image_url_prefix}/lvis/{cat_name}/{i}.jpg"
                for i in range(len(img_urls))
            ]

            product = UnifiedProduct(
                product_id=f"lvis_{cat_id}",
                title=cat_name.replace("_", " ").title(),
                description=f"Category: {cat_name}. Synonyms: {', '.join(synonyms)}" if synonyms else f"Category: {cat_name}",
                category="Home & Objects",
                attributes={"synonyms": synonyms, "frequency": cat_info.get("frequency", "")},
                image_urls=gcs_urls,
                in_stock=True,
                source=DataSource.LVIS,
            )
            products.append(product)

    print(f"[LVIS] Normalized {len(products)} products")
    return products


# ================================================================
# LAION Normalizer
# ================================================================

def normalize_laion(raw_path: str, image_url_prefix: str) -> list[UnifiedProduct]:
    """
    LAION subset (Parquet):
    Columns typically: url, caption, similarity, width, height, ...
    We use captions as product descriptions.
    """
    products = []

    try:
        import pandas as pd
    except ImportError:
        print("[LAION] pandas not available, skipping")
        return products

    parquet_files = list(Path(raw_path).glob("*.parquet"))
    if not parquet_files:
        print(f"[LAION] No Parquet files found in {raw_path}")
        return products

    for filepath in parquet_files:
        print(f"[LAION] Processing {filepath.name}...")
        df = pd.read_parquet(filepath)

        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"  {filepath.name}"):
            caption = str(row.get("caption", ""))
            url = str(row.get("url", ""))

            product_id = f"laion_{uuid.uuid4().hex[:12]}"
            gcs_url = f"{image_url_prefix}/laion/{product_id}.jpg"

            product = UnifiedProduct(
                product_id=product_id,
                title=caption[:100] if caption else "Untitled",
                description=caption,
                category="General",
                image_urls=[gcs_url],
                attributes={"original_url": url},
                in_stock=True,
                source=DataSource.LAION,
            )
            products.append(product)

    print(f"[LAION] Normalized {len(products)} products")
    return products


# ================================================================
# Main Pipeline
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="Normalize raw datasets into Unified Schema")
    parser.add_argument("--config", default="config/settings.yaml", help="Config file path")
    parser.add_argument("--source", choices=["amazon", "hm", "lvis", "laion", "all"], default="all",
                        help="Which dataset to normalize")
    args = parser.parse_args()

    config = load_config(args.config)
    datasets = config["datasets"]
    image_url_prefix = config["cloud_storage"]["image_url_prefix"]

    # Initialize databases
    sqlite = SQLiteCatalog(config["databases"]["sqlite"]["path"])
    lancedb = LanceDBCatalog(
        db_path=config["databases"]["lancedb"]["path"],
        table_name=config["databases"]["lancedb"]["table_name"],
        visual_dim=config["embeddings"]["visual"]["dimension"],
        semantic_dim=config["embeddings"]["semantic"]["dimension"],
    )

    normalizers = {
        "amazon": (normalize_amazon, datasets["amazon"]["raw_path"]),
        "hm": (normalize_hm, datasets["hm"]["raw_path"]),
        "lvis": (normalize_lvis, datasets["lvis"]["raw_path"]),
        "laion": (normalize_laion, datasets["laion"]["raw_path"]),
    }

    sources = [args.source] if args.source != "all" else list(normalizers.keys())

    total_inserted = 0
    for source in sources:
        normalizer_fn, raw_path = normalizers[source]

        if not Path(raw_path).exists():
            print(f"[{source.upper()}] Raw data path not found: {raw_path} — skipping")
            continue

        products = normalizer_fn(raw_path, image_url_prefix)

        if not products:
            continue

        # Insert into SQLite
        rows = [p.to_sqlite_row() for p in products]
        inserted = sqlite.insert_batch(rows)
        total_inserted += inserted
        print(f"[{source.upper()}] Inserted {inserted} new products (skipped {len(products) - inserted} duplicates)")

        # Insert placeholder vectors into LanceDB
        product_ids = [p.product_id for p in products]
        lancedb.upsert_embeddings(product_ids)  # Zeros — will be filled by 02_extract_embeddings.py

    print(f"\n✅ Total products in catalog: {sqlite.count()}")
    print(f"   New products added: {total_inserted}")
    sqlite.close()


if __name__ == "__main__":
    main()
