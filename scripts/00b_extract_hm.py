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


def main():
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


if __name__ == "__main__":
    main()
