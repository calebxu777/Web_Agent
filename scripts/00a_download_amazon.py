"""
00a_download_amazon.py — Download Amazon Reviews 2023 Dataset
===============================================================
Downloads product metadata + reviews from HuggingFace (McAuley Lab).
Downloads product images to local disk + optionally uploads to GCS.
Inserts metadata (with reviews) into the catalog SQLite database.

KEY FIX: Uses HTTP streaming to read JSONL line-by-line from HuggingFace
without downloading the full 18GB file. Only fetches metadata for
max_products items.

Prerequisites:
    pip install huggingface_hub Pillow tqdm google-cloud-storage requests

Usage:
    # Download metadata only (no images)
    python scripts/00a_download_amazon.py --metadata-only

    # Download metadata + images to local disk
    python scripts/00a_download_amazon.py

    # Download metadata + images + upload to GCS
    python scripts/00a_download_amazon.py --upload-gcs

    # Control subset size
    python scripts/00a_download_amazon.py --max-products 10000
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from huggingface_hub import hf_hub_url
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))


# ================================================================
# Configuration
# ================================================================

CATEGORIES = {
    "Clothing_Shoes_and_Jewelry": "Clothing & Accessories",
    "Electronics": "Electronics",
    "Home_and_Kitchen": "Home & Kitchen",
}

# HuggingFace paths for metadata JSONL files
META_PATH_TEMPLATE = "raw/meta_categories/meta_{category}.jsonl"
REVIEW_PATH_TEMPLATE = "raw/review_categories/{category}.jsonl"

LOCAL_IMAGE_DIR = Path(r"C:\Users\Caleb\Desktop\product_images\amazon")
DB_PATH = Path("data/processed/catalog.db")
RAW_DIR = Path("data/raw/amazon")

GCS_BUCKET_DEFAULT = "web-agent-data-caleb-2026"


# ================================================================
# Database Setup
# ================================================================

def init_db(db_path: Path) -> sqlite3.Connection:
    """Create the products table if it doesn't exist."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS products (
            product_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            description TEXT DEFAULT '',
            category TEXT DEFAULT '',
            subcategory TEXT DEFAULT '',
            price REAL,
            currency TEXT DEFAULT 'USD',
            brand TEXT DEFAULT '',
            image_urls TEXT DEFAULT '',
            attributes TEXT DEFAULT '',
            reviews_summary TEXT DEFAULT '',
            rating REAL,
            review_count INTEGER DEFAULT 0,
            in_stock INTEGER DEFAULT 1,
            source TEXT DEFAULT '',
            has_visual_embedding INTEGER DEFAULT 0,
            has_semantic_embedding INTEGER DEFAULT 0,
            dedup_hash TEXT DEFAULT ''
        )
    """)
    conn.commit()
    return conn


# ================================================================
# Streaming JSONL Reader (no full download!)
# ================================================================

def stream_jsonl_from_hf(repo_id: str, filepath: str, max_lines: int = 10000):
    """
    Stream JSONL lines directly from HuggingFace via HTTP.
    Reads line-by-line without downloading the full file.
    This is the key fix — the Clothing JSONL is 18GB, but we only
    need the first 10K lines (~20MB).
    """
    url = hf_hub_url(repo_id, filepath, repo_type="dataset")

    with requests.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        resp.encoding = "utf-8"
        count = 0

        for line in resp.iter_lines(decode_unicode=True):
            if line:
                try:
                    yield json.loads(line)
                    count += 1
                    if count >= max_lines:
                        return
                except json.JSONDecodeError:
                    continue


# ================================================================
# Image Download
# ================================================================

def download_image(url: str, save_path: Path, size: int = 224) -> bool:
    """Download an image, resize to DINOv2 input size, save as JPEG."""
    try:
        resp = requests.get(url, timeout=10, stream=True)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        img = img.resize((size, size), Image.LANCZOS)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(str(save_path), "JPEG", quality=85)
        return True
    except Exception:
        return False


def upload_to_gcs(local_path: Path, gcs_key: str, bucket) -> bool:
    """Upload a local file to Google Cloud Storage."""
    try:
        blob = bucket.blob(gcs_key)
        blob.upload_from_filename(str(local_path), content_type="image/jpeg")
        return True
    except Exception:
        return False


# ================================================================
# Review Aggregation
# ================================================================

def aggregate_reviews(reviews: list[dict], max_reviews: int = 10) -> tuple[str, float, int]:
    """Aggregate reviews into a summary string, avg rating, and count."""
    if not reviews:
        return "", None, 0

    reviews = reviews[:max_reviews]
    total_rating = 0
    count = 0
    snippets = []

    for r in reviews:
        rating = r.get("rating")
        text = r.get("text", "").strip()
        title = r.get("title", "").strip()

        if rating is not None:
            total_rating += float(rating)
            count += 1

        if title and text:
            snippets.append(f"[{rating}★] {title}: {text[:150]}")
        elif text:
            snippets.append(f"[{rating}★] {text[:150]}")

    avg_rating = round(total_rating / count, 1) if count > 0 else None
    summary = " | ".join(snippets[:5])
    return summary, avg_rating, count


# ================================================================
# Main Download Pipeline
# ================================================================

def download_amazon(
    max_products_per_cat: int = 10000,
    metadata_only: bool = False,
    upload_gcs: bool = False,
    image_size: int = 224,
    num_workers: int = 8,
):
    conn = init_db(DB_PATH)
    LOCAL_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Optionally set up GCS client
    bucket = None
    if upload_gcs:
        try:
            from google.cloud import storage
            project_id = os.environ.get("GCP_PROJECT_ID", "webagent2026")
            client = storage.Client(project=project_id)
            bucket_name = os.environ.get("GCS_BUCKET_NAME", GCS_BUCKET_DEFAULT)
            bucket = client.bucket(bucket_name)
            print(f"📦 GCS upload enabled (bucket: {bucket_name})")
        except ImportError:
            print("❌ google-cloud-storage not installed. Run: pip install google-cloud-storage")
            upload_gcs = False
        except Exception as e:
            print(f"❌ GCS auth failed: {e}")
            print("  Run: gcloud auth application-default login")
            upload_gcs = False

    total_inserted = 0
    total_images = 0
    total_image_failures = 0

    for category, cat_name in CATEGORIES.items():
        print(f"\n{'='*60}")
        print(f"  Downloading: {cat_name}")
        print(f"{'='*60}")

        # --- Stream metadata JSONL from HuggingFace ---
        meta_path = META_PATH_TEMPLATE.format(category=category)
        print(f"  Streaming {max_products_per_cat} products from HuggingFace...")
        print(f"  File: {meta_path}")

        meta_items = []
        try:
            for item in tqdm(
                stream_jsonl_from_hf(
                    "McAuley-Lab/Amazon-Reviews-2023",
                    meta_path,
                    max_lines=max_products_per_cat,
                ),
                desc=f"  {cat_name}",
                total=max_products_per_cat,
            ):
                meta_items.append(item)
        except Exception as e:
            print(f"  ❌ Failed to stream {category}: {e}")
            continue

        print(f"  Collected {len(meta_items)} products")

        # Collect ASINs for review lookup
        asin_set = set()
        for item in meta_items:
            asin = item.get("parent_asin") or item.get("asin", "")
            if asin:
                asin_set.add(asin)

        # --- Stream reviews ---
        review_path = REVIEW_PATH_TEMPLATE.format(category=category)
        print(f"  Streaming reviews...")
        reviews_by_asin = {}
        try:
            review_count = 0
            for review in tqdm(
                stream_jsonl_from_hf(
                    "McAuley-Lab/Amazon-Reviews-2023",
                    review_path,
                    max_lines=max_products_per_cat * 10,
                ),
                desc="  Reviews",
            ):
                asin = review.get("parent_asin") or review.get("asin")
                if asin and asin in asin_set:
                    if asin not in reviews_by_asin:
                        reviews_by_asin[asin] = []
                    if len(reviews_by_asin[asin]) < 10:
                        reviews_by_asin[asin].append(review)
                        review_count += 1
                if review_count >= max_products_per_cat * 5:
                    break
            print(f"  Matched reviews for {len(reviews_by_asin)} products")
        except Exception as e:
            print(f"  ⚠️ Could not load reviews: {e}")

        # --- Process each product ---
        batch_rows = []
        image_tasks = []

        for item in tqdm(meta_items, desc=f"  Processing {cat_name}"):
            asin = item.get("parent_asin") or item.get("asin", "")
            if not asin:
                continue

            product_id = f"amz_{asin}"
            title = item.get("title") or ""
            if not title:
                continue

            # Description
            description = ""
            if item.get("description"):
                desc = item["description"]
                if isinstance(desc, list):
                    description = " ".join(str(d) for d in desc)
                else:
                    description = str(desc)

            features = item.get("features") or []
            if isinstance(features, list):
                description += " " + " ".join(str(f) for f in features)

            # Price
            price = None
            if item.get("price"):
                try:
                    price = float(str(item["price"]).replace("$", "").replace(",", ""))
                except (ValueError, TypeError):
                    pass

            brand = item.get("store") or item.get("brand") or ""

            # Subcategory
            subcategory = ""
            cats = item.get("categories") or item.get("main_category") or ""
            if isinstance(cats, list) and len(cats) > 1:
                subcategory = cats[-1] if isinstance(cats[-1], str) else ""

            # Image URLs
            images = item.get("images") or []
            image_urls = []
            if isinstance(images, list):
                for img in images:
                    if isinstance(img, dict):
                        url = img.get("hi_res") or img.get("large") or img.get("thumb")
                        if url:
                            image_urls.append(url)
                    elif isinstance(img, str):
                        image_urls.append(img)
            image_urls = list(dict.fromkeys(image_urls))[:5]

            local_image_path = f"amazon/{product_id}.jpg"

            # Reviews
            product_reviews = reviews_by_asin.get(asin, [])
            reviews_summary, avg_rating, review_count = aggregate_reviews(product_reviews)

            if avg_rating is None and item.get("average_rating"):
                try:
                    avg_rating = float(item["average_rating"])
                except (ValueError, TypeError):
                    pass
            if review_count == 0 and item.get("rating_number"):
                try:
                    review_count = int(item["rating_number"])
                except (ValueError, TypeError):
                    pass

            # Attributes
            attributes = {}
            if item.get("details"):
                details = item["details"]
                if isinstance(details, dict):
                    attributes = {k: str(v)[:100] for k, v in details.items()}

            # Dedup hash
            raw = f"{title.lower().strip()}|{brand.lower().strip()}"
            dedup_hash = hashlib.sha256(raw.encode()).hexdigest()[:16]

            row = {
                "product_id": product_id,
                "title": title[:500],
                "description": description[:2000],
                "category": cat_name,
                "subcategory": subcategory,
                "price": price,
                "currency": "USD",
                "brand": brand,
                "image_urls": local_image_path,
                "attributes": json.dumps(attributes),
                "reviews_summary": reviews_summary[:1000],
                "rating": avg_rating,
                "review_count": review_count,
                "in_stock": 1,
                "source": "amazon",
                "has_visual_embedding": 0,
                "has_semantic_embedding": 0,
                "dedup_hash": dedup_hash,
            }
            batch_rows.append(row)

            if not metadata_only and image_urls:
                image_tasks.append((image_urls[0], LOCAL_IMAGE_DIR / f"{product_id}.jpg", product_id))

        # --- Insert metadata ---
        print(f"  Inserting {len(batch_rows)} products into database...")
        for row in batch_rows:
            try:
                conn.execute(
                    """INSERT OR REPLACE INTO products VALUES (
                        :product_id, :title, :description, :category, :subcategory,
                        :price, :currency, :brand, :image_urls, :attributes,
                        :reviews_summary, :rating, :review_count, :in_stock, :source,
                        :has_visual_embedding, :has_semantic_embedding, :dedup_hash
                    )""",
                    row,
                )
            except Exception:
                pass
        conn.commit()
        total_inserted += len(batch_rows)

        # --- Download images in parallel ---
        if image_tasks:
            print(f"  Downloading {len(image_tasks)} images...")
            successes = 0
            failures = 0

            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                future_map = {}
                for url, path, pid in image_tasks:
                    f = executor.submit(download_image, url, path, image_size)
                    future_map[f] = (url, path, pid)

                for future in tqdm(as_completed(future_map), total=len(future_map), desc="  Images"):
                    url, path, pid = future_map[future]
                    if future.result():
                        successes += 1
                        if upload_gcs and bucket and path.exists():
                            gcs_key = f"amazon/{pid}.jpg"
                            upload_to_gcs(path, gcs_key, bucket)
                    else:
                        failures += 1

            total_images += successes
            total_image_failures += failures
            print(f"  ✅ Images: {successes} downloaded, {failures} failed")

    conn.close()
    print(f"\n{'='*60}")
    print(f"  Amazon Download Complete!")
    print(f"  Products inserted: {total_inserted}")
    print(f"  Images downloaded: {total_images} ({total_image_failures} failed)")
    print(f"  Database: {DB_PATH}")
    print(f"  Images: {LOCAL_IMAGE_DIR}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Download Amazon Reviews 2023 dataset")
    parser.add_argument("--max-products", type=int, default=10000,
                        help="Max products per category")
    parser.add_argument("--metadata-only", action="store_true",
                        help="Download metadata only, skip images")
    parser.add_argument("--upload-gcs", action="store_true",
                        help="Also upload images to Google Cloud Storage")
    parser.add_argument("--image-size", type=int, default=224,
                        help="Resize images to this dimension (default: 224 for DINOv2)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel image download workers")
    args = parser.parse_args()

    download_amazon(
        max_products_per_cat=args.max_products,
        metadata_only=args.metadata_only,
        upload_gcs=args.upload_gcs,
        image_size=args.image_size,
        num_workers=args.workers,
    )


if __name__ == "__main__":
    main()
