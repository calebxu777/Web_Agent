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


def main():
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


if __name__ == "__main__":
    main()
