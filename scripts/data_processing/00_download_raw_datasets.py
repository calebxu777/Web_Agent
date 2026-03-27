
# ========================================
# ----- MERGED FROM: 00a_download_amazon.py -----
# ========================================

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


def main_00a_download_amazon():
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





# ========================================
# ----- MERGED FROM: 00b_download_hm.py -----
# ========================================

"""
00b_download_hm.py — Download & Process H&M Fashion Dataset
==============================================================
Processes the H&M Personalized Fashion Recommendations dataset from Kaggle.
Images are copied/resized to the local image directory + optionally uploaded to GCS.
Metadata from articles.csv is inserted into the catalog SQLite database.

Prerequisites:
    1. Download the dataset from Kaggle first:
       kaggle competitions download -c h-and-m-personalized-fashion-recommendations
    2. Extract to data/raw/hm/
       Expected structure:
         data/raw/hm/
           articles.csv
           images/
             0/
               0108775015.jpg
               0108775044.jpg
             ...

Usage:
    python scripts/00b_download_hm.py --max-products 20000 --upload-gcs
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))


# ================================================================
# Configuration
# ================================================================

HM_RAW_DIR = Path("data/raw/hm")
LOCAL_IMAGE_DIR = Path(r"C:\Users\Caleb\Desktop\product_images\hm")
DB_PATH = Path("data/processed/catalog.db")


# ================================================================
# Database Setup (same schema as Amazon)
# ================================================================

def init_db(db_path: Path) -> sqlite3.Connection:
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
# Image Processing
# ================================================================

def process_image(src_path: Path, dst_path: Path, size: int = 224) -> bool:
    """Resize an H&M product image and save to destination."""
    try:
        img = Image.open(str(src_path)).convert("RGB")
        img = img.resize((size, size), Image.LANCZOS)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(str(dst_path), "JPEG", quality=85)
        return True
    except Exception:
        return False


def upload_to_gcs(local_path: Path, gcs_key: str, bucket) -> bool:
    try:
        blob = bucket.blob(gcs_key)
        blob.upload_from_filename(str(local_path), content_type="image/jpeg")
        return True
    except Exception:
        return False


# ================================================================
# H&M articles.csv Columns (key ones):
#   article_id, product_code, prod_name, product_type_name,
#   product_group_name, graphical_appearance_name, colour_group_name,
#   perceived_colour_value_name, department_name, index_name,
#   index_group_name, section_name, garment_group_name, detail_desc
# ================================================================

def find_hm_image(article_id: str, hm_images_dir: Path) -> Path | None:
    """
    H&M images are stored as: images/{first_3_digits}/{article_id}.jpg
    e.g. images/010/0108775015.jpg
    """
    # article_id is like "0108775015" (10 digits, zero-padded)
    article_str = str(article_id).zfill(10)
    prefix = article_str[:3]
    img_path = hm_images_dir / prefix / f"{article_str}.jpg"
    return img_path if img_path.exists() else None


def process_hm(
    max_products: int = 20000,
    upload_gcs: bool = False,
    image_size: int = 224,
    num_workers: int = 8,
):
    articles_file = HM_RAW_DIR / "articles.csv"
    hm_images_dir = HM_RAW_DIR / "images"

    if not articles_file.exists():
        print(f"❌ articles.csv not found at {articles_file}")
        print(f"   Download from Kaggle first:")
        print(f"   kaggle competitions download -c h-and-m-personalized-fashion-recommendations")
        print(f"   Then extract to {HM_RAW_DIR}/")
        return

    conn = init_db(DB_PATH)
    LOCAL_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    # Set up GCS
    bucket = None
    if upload_gcs:
        try:
            from google.cloud import storage
            project_id = os.environ.get("GCP_PROJECT_ID", "webagent2026")
            client = storage.Client(project=project_id)
            bucket_name = os.environ.get("GCS_BUCKET_NAME", "web-agent-data-caleb-2026")
            bucket = client.bucket(bucket_name)
            print(f"📦 GCS upload enabled (bucket: {bucket_name})")
        except ImportError:
            print("❌ google-cloud-storage not installed. Run: pip install google-cloud-storage")
            upload_gcs = False

    print(f"\n{'='*60}")
    print(f"  Processing H&M Fashion Dataset")
    print(f"  Source: {articles_file}")
    print(f"  Max products: {max_products}")
    print(f"{'='*60}\n")

    # Read articles.csv
    rows = []
    image_tasks = []
    count = 0

    with open(articles_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for item in tqdm(reader, desc="Reading articles.csv"):
            if count >= max_products:
                break

            article_id = item.get("article_id", "")
            prod_name = item.get("prod_name", "")
            if not article_id or not prod_name:
                continue

            product_id = f"hm_{article_id}"

            # Build description from available fields
            desc_parts = []
            if item.get("detail_desc"):
                desc_parts.append(item["detail_desc"])
            if item.get("product_type_name"):
                desc_parts.append(f"Type: {item['product_type_name']}")
            if item.get("graphical_appearance_name"):
                desc_parts.append(f"Pattern: {item['graphical_appearance_name']}")
            description = " | ".join(desc_parts)

            # Category
            category = item.get("product_group_name", "Fashion")
            subcategory = item.get("section_name", "")

            # Color as attribute
            attributes = {}
            if item.get("colour_group_name"):
                attributes["color"] = item["colour_group_name"]
            if item.get("perceived_colour_value_name"):
                attributes["shade"] = item["perceived_colour_value_name"]
            if item.get("garment_group_name"):
                attributes["garment_type"] = item["garment_group_name"]
            if item.get("index_name"):
                attributes["collection"] = item["index_name"]
            if item.get("department_name"):
                attributes["department"] = item["department_name"]

            # Image path
            local_image_path = f"hm/{product_id}.jpg"

            # Dedup hash
            raw = f"{prod_name.lower().strip()}|hm"
            dedup_hash = hashlib.sha256(raw.encode()).hexdigest()[:16]

            row = {
                "product_id": product_id,
                "title": prod_name[:500],
                "description": description[:2000],
                "category": category,
                "subcategory": subcategory,
                "price": None,  # H&M dataset doesn't include prices
                "currency": "USD",
                "brand": "H&M",
                "image_urls": local_image_path,
                "attributes": json.dumps(attributes),
                "reviews_summary": "",  # No reviews in H&M dataset
                "rating": None,
                "review_count": 0,
                "in_stock": 1,
                "source": "hm",
                "has_visual_embedding": 0,
                "has_semantic_embedding": 0,
                "dedup_hash": dedup_hash,
            }
            rows.append(row)

            # Queue image processing
            src_img = find_hm_image(article_id, hm_images_dir)
            if src_img:
                dst_img = LOCAL_IMAGE_DIR / f"{product_id}.jpg"
                image_tasks.append((src_img, dst_img, product_id))

            count += 1

    # --- Insert into DB ---
    print(f"\n  Inserting {len(rows)} products into database...")
    for row in rows:
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
    conn.close()

    # --- Process images in parallel ---
    if image_tasks:
        print(f"  Processing {len(image_tasks)} images (resize to {image_size}px)...")
        successes = 0
        failures = 0

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_map = {}
            for src, dst, pid in image_tasks:
                f = executor.submit(process_image, src, dst, image_size)
                future_map[f] = (src, dst, pid)

            for future in tqdm(as_completed(future_map), total=len(future_map), desc="  Images"):
                src, dst, pid = future_map[future]
                if future.result():
                    successes += 1
                    if upload_gcs and bucket and dst.exists():
                        gcs_key = f"hm/{pid}.jpg"
                        upload_to_gcs(dst, gcs_key, bucket)
                else:
                    failures += 1

        print(f"  ✅ Images: {successes} processed, {failures} failed")
    else:
        print("  ⚠️ No images found. Make sure images/ folder exists in data/raw/hm/")

    print(f"\n{'='*60}")
    print(f"  H&M Download Complete!")
    print(f"  Products inserted: {len(rows)}")
    print(f"  Database: {DB_PATH}")
    print(f"  Images: {LOCAL_IMAGE_DIR}")
    print(f"{'='*60}")


def main_00b_download_hm():
    parser = argparse.ArgumentParser(description="Process H&M Fashion dataset")
    parser.add_argument("--max-products", type=int, default=20000,
                        help="Max products to process")
    parser.add_argument("--upload-gcs", action="store_true",
                        help="Also upload images to Google Cloud Storage")
    parser.add_argument("--image-size", type=int, default=224,
                        help="Resize images to this dimension")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel workers")
    args = parser.parse_args()

    process_hm(
        max_products=args.max_products,
        upload_gcs=args.upload_gcs,
        image_size=args.image_size,
        num_workers=args.workers,
    )





# ========================================
# ----- MERGED FROM: 00b_extract_hm.py -----
# ========================================

"""
00b_extract_hm.py — Extract H&M data directly from zip (space-efficient)
==========================================================================
Reads articles.csv and images DIRECTLY from the zip file without
extracting the full 30GB. Only saves the 20K resized images we need.

This saves ~30GB of disk space vs full extraction.

Usage:
    python scripts/00b_extract_hm.py --max-products 20000 --upload-gcs
    python scripts/00b_extract_hm.py --max-products 20000 --delete-zip
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import sqlite3
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

# ================================================================
# Configuration
# ================================================================

ZIP_PATH = Path("data/raw/hm/h-and-m-personalized-fashion-recommendations.zip")
LOCAL_IMAGE_DIR = Path(r"C:\Users\Caleb\Desktop\product_images\hm")
DB_PATH = Path("data/processed/catalog.db")
GCS_BUCKET_DEFAULT = "web-agent-data-caleb-2026"


# ================================================================
# Database Setup
# ================================================================

def init_db(db_path: Path) -> sqlite3.Connection:
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
# Image Processing
# ================================================================

def resize_and_save(image_bytes: bytes, dst_path: Path, size: int = 224) -> bool:
    """Resize image from bytes and save as JPEG."""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize((size, size), Image.LANCZOS)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(str(dst_path), "JPEG", quality=85)
        return True
    except Exception:
        return False


def upload_to_gcs(local_path: Path, gcs_key: str, bucket) -> bool:
    try:
        blob = bucket.blob(gcs_key)
        blob.upload_from_filename(str(local_path), content_type="image/jpeg")
        return True
    except Exception:
        return False


# ================================================================
# Main Pipeline
# ================================================================

def process_hm_from_zip(
    max_products: int = 20000,
    upload_gcs: bool = False,
    delete_zip: bool = False,
    image_size: int = 224,
    num_workers: int = 8,
):
    if not ZIP_PATH.exists():
        print(f"❌ Zip not found at {ZIP_PATH}")
        print(f"   Run first: python scripts/00b_kaggle_download_hm.py")
        return

    print(f"📦 Reading from zip: {ZIP_PATH}")
    print(f"   Zip size: {ZIP_PATH.stat().st_size / 1e9:.1f} GB")
    print(f"   Max products: {max_products}")
    print(f"   NO full extraction — reading directly from zip\n")

    conn = init_db(DB_PATH)
    LOCAL_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    # Set up GCS
    bucket = None
    if upload_gcs:
        try:
            from google.cloud import storage
            project_id = os.environ.get("GCP_PROJECT_ID", "webagent2026")
            client = storage.Client(project=project_id)
            bucket_name = os.environ.get("GCS_BUCKET_NAME", GCS_BUCKET_DEFAULT)
            bucket = client.bucket(bucket_name)
            print(f"📦 GCS upload enabled (bucket: {bucket_name})")
        except Exception as e:
            print(f"⚠️ GCS not available: {e}")
            upload_gcs = False

    with zipfile.ZipFile(str(ZIP_PATH), "r") as zf:
        # Find articles.csv in the zip
        csv_name = None
        for name in zf.namelist():
            if name.endswith("articles.csv"):
                csv_name = name
                break

        if not csv_name:
            print("❌ articles.csv not found in zip")
            return

        print(f"  Reading {csv_name}...")

        # Read articles.csv from zip
        with zf.open(csv_name) as f:
            # Wrap in TextIOWrapper for csv.DictReader
            text_f = io.TextIOWrapper(f, encoding="utf-8")
            reader = csv.DictReader(text_f)

            rows = []
            article_ids = []
            count = 0

            for item in tqdm(reader, desc="  Reading articles"):
                if count >= max_products:
                    break

                article_id = item.get("article_id", "")
                prod_name = item.get("prod_name", "")
                if not article_id or not prod_name:
                    continue

                product_id = f"hm_{article_id}"

                # Build description
                desc_parts = []
                if item.get("detail_desc"):
                    desc_parts.append(item["detail_desc"])
                if item.get("product_type_name"):
                    desc_parts.append(f"Type: {item['product_type_name']}")
                if item.get("graphical_appearance_name"):
                    desc_parts.append(f"Pattern: {item['graphical_appearance_name']}")
                description = " | ".join(desc_parts)

                category = item.get("product_group_name", "Fashion")
                subcategory = item.get("section_name", "")

                attributes = {}
                if item.get("colour_group_name"):
                    attributes["color"] = item["colour_group_name"]
                if item.get("perceived_colour_value_name"):
                    attributes["shade"] = item["perceived_colour_value_name"]
                if item.get("garment_group_name"):
                    attributes["garment_type"] = item["garment_group_name"]
                if item.get("index_name"):
                    attributes["collection"] = item["index_name"]
                if item.get("department_name"):
                    attributes["department"] = item["department_name"]

                local_image_path = f"hm/{product_id}.jpg"
                raw = f"{prod_name.lower().strip()}|hm"
                dedup_hash = hashlib.sha256(raw.encode()).hexdigest()[:16]

                row = {
                    "product_id": product_id,
                    "title": prod_name[:500],
                    "description": description[:2000],
                    "category": category,
                    "subcategory": subcategory,
                    "price": None,
                    "currency": "USD",
                    "brand": "H&M",
                    "image_urls": local_image_path,
                    "attributes": json.dumps(attributes),
                    "reviews_summary": "",
                    "rating": None,
                    "review_count": 0,
                    "in_stock": 1,
                    "source": "hm",
                    "has_visual_embedding": 0,
                    "has_semantic_embedding": 0,
                    "dedup_hash": dedup_hash,
                }
                rows.append(row)
                article_ids.append(article_id)
                count += 1

        print(f"  Read {len(rows)} products from articles.csv")

        # Build a lookup of image paths inside the zip
        # H&M images: images/0{first_2_digits}/{article_id}.jpg
        print(f"  Building image index from zip...")
        zip_image_map = {}
        for name in zf.namelist():
            if name.endswith(".jpg") and "images/" in name:
                # Extract just the filename without extension
                basename = name.rsplit("/", 1)[-1].replace(".jpg", "")
                zip_image_map[basename] = name

        print(f"  Found {len(zip_image_map)} images in zip")

        # Extract + resize images for our products
        print(f"  Extracting & resizing {len(article_ids)} product images...")
        successes = 0
        failures = 0
        image_paths_for_gcs = []

        for article_id in tqdm(article_ids, desc="  Images"):
            padded = article_id.zfill(10)
            zip_name = zip_image_map.get(padded)
            if not zip_name:
                failures += 1
                continue

            product_id = f"hm_{article_id}"
            dst_path = LOCAL_IMAGE_DIR / f"{product_id}.jpg"

            try:
                img_bytes = zf.read(zip_name)
                if resize_and_save(img_bytes, dst_path, image_size):
                    successes += 1
                    image_paths_for_gcs.append((dst_path, f"hm/{product_id}.jpg"))
                else:
                    failures += 1
            except Exception:
                failures += 1

        print(f"  ✅ Images: {successes} saved, {failures} failed")

    # Insert metadata into DB
    print(f"  Inserting {len(rows)} products into database...")
    for row in rows:
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
    conn.close()

    # Upload to GCS
    if upload_gcs and bucket and image_paths_for_gcs:
        print(f"\n  Uploading {len(image_paths_for_gcs)} images to GCS...")
        gcs_ok = 0
        gcs_fail = 0

        with ThreadPoolExecutor(max_workers=16) as executor:
            def do_upload(args):
                local, key = args
                return upload_to_gcs(local, key, bucket)

            futures = {executor.submit(do_upload, item): item for item in image_paths_for_gcs}
            for future in tqdm(as_completed(futures), total=len(futures), desc="  GCS Upload"):
                if future.result():
                    gcs_ok += 1
                else:
                    gcs_fail += 1

        print(f"  ✅ GCS: {gcs_ok} uploaded, {gcs_fail} failed")

    # Delete zip to free space
    if delete_zip and ZIP_PATH.exists():
        print(f"\n  Deleting zip to free {ZIP_PATH.stat().st_size / 1e9:.1f} GB...")
        ZIP_PATH.unlink()
        print(f"  ✅ Zip deleted")

    print(f"\n{'='*60}")
    print(f"  H&M Processing Complete!")
    print(f"  Products inserted: {len(rows)}")
    print(f"  Images saved: {successes}")
    print(f"  Database: {DB_PATH}")
    print(f"  Images: {LOCAL_IMAGE_DIR}")
    print(f"{'='*60}")


def main_00b_extract_hm():
    parser = argparse.ArgumentParser(description="Process H&M dataset from zip (space-efficient)")
    parser.add_argument("--max-products", type=int, default=20000)
    parser.add_argument("--upload-gcs", action="store_true")
    parser.add_argument("--delete-zip", action="store_true", help="Delete zip after processing to free ~29GB")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    process_hm_from_zip(
        max_products=args.max_products,
        upload_gcs=args.upload_gcs,
        delete_zip=args.delete_zip,
        image_size=args.image_size,
        num_workers=args.workers,
    )





# ========================================
# ----- MERGED FROM: 00b_kaggle_download_hm.py -----
# ========================================

"""
00b_kaggle_download_hm.py — Download H&M Dataset from Kaggle
================================================================
Downloads and extracts the H&M Personalized Fashion Recommendations
dataset from Kaggle. Run this BEFORE 00b_download_hm.py.

Prerequisites:
    pip install kaggle
    Set KAGGLE_API_TOKEN env var or place kaggle.json in ~/.kaggle/

Usage:
    python scripts/00b_kaggle_download_hm.py
"""

import os
import sys
import zipfile
from pathlib import Path

HM_RAW_DIR = Path("data/raw/hm")
COMPETITION = "h-and-m-personalized-fashion-recommendations"


def download_hm():
    HM_RAW_DIR.mkdir(parents=True, exist_ok=True)

    zip_path = HM_RAW_DIR / f"{COMPETITION}.zip"
    articles_file = HM_RAW_DIR / "articles.csv"

    # Skip if already extracted
    if articles_file.exists():
        print(f"✅ H&M dataset already extracted at {HM_RAW_DIR}")
        print(f"   articles.csv found — skipping download.")
        print(f"   Run 00b_download_hm.py to process it.")
        return

    # Download from Kaggle
    print(f"Downloading H&M dataset from Kaggle...")
    print(f"  Competition: {COMPETITION}")
    print(f"  Destination: {HM_RAW_DIR}")

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        api.competition_download_files(COMPETITION, path=str(HM_RAW_DIR), quiet=False)
    except Exception:
        # Fallback to CLI
        exit_code = os.system(
            f'kaggle competitions download -c {COMPETITION} -p "{HM_RAW_DIR}"'
        )
        if exit_code != 0:
            print("❌ Kaggle download failed. Make sure you:")
            print("   1. Have KAGGLE_API_TOKEN set or kaggle.json in ~/.kaggle/")
            print("   2. Have accepted the competition rules on kaggle.com")
            sys.exit(1)

    # Extract
    if zip_path.exists():
        print(f"\nExtracting {zip_path.name} ({zip_path.stat().st_size / 1e9:.1f} GB)...")
        with zipfile.ZipFile(str(zip_path), "r") as zf:
            zf.extractall(str(HM_RAW_DIR))
        print(f"✅ Extracted to {HM_RAW_DIR}")

        # Clean up zip to save disk space
        print(f"Removing zip to save space...")
        zip_path.unlink()
        print(f"✅ Deleted {zip_path.name}")
    else:
        print(f"❌ Zip file not found at {zip_path}")
        sys.exit(1)

    # Verify
    if articles_file.exists():
        print(f"\n✅ H&M dataset ready!")
        print(f"   articles.csv: {articles_file}")
        images_dir = HM_RAW_DIR / "images"
        if images_dir.exists():
            count = sum(1 for _ in images_dir.rglob("*.jpg"))
            print(f"   images/: {count} images found")
        print(f"\n   Next step: python scripts/00b_download_hm.py --max-products 20000 --upload-gcs")
    else:
        print(f"❌ articles.csv not found after extraction")


if __name__ == "__main__":
    download_hm()



if __name__ == "__main__":
    print("\n\n========== Running 00a_download_amazon ==========")
    main_00a_download_amazon()
    print("\n\n========== Running 00b_download_hm ==========")
    main_00b_download_hm()
    print("\n\n========== Running 00b_extract_hm ==========")
    main_00b_extract_hm()
    print("\n\n========== Running 00b_kaggle_download_hm ==========")
    main_00b_kaggle_download_hm()