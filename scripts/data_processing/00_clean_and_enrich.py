
# ========================================
# ----- MERGED FROM: 00c_cleanup_and_upload.py -----
# ========================================

"""Clean up DB: remove products without local images, then optionally upload to GCS."""
import argparse
import os
import sqlite3
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

LOCAL_IMAGE_DIR = Path(r"C:\Users\Caleb\Desktop\product_images")
DB_PATH = Path("data/processed/catalog.db")


def clean_and_upload(upload_gcs: bool = False):
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    # Count before
    total_before = conn.execute("SELECT COUNT(*) FROM products").fetchone()[0]
    print(f"Products before cleanup: {total_before}")

    # Find products without matching local images
    rows = conn.execute("SELECT product_id, image_urls, source FROM products").fetchall()
    to_delete = []
    valid_images = []

    for row in rows:
        image_rel = row["image_urls"]  # e.g. "amazon/amz_B001234.jpg"
        if not image_rel:
            to_delete.append(row["product_id"])
            continue
        local_path = LOCAL_IMAGE_DIR / image_rel
        if not local_path.exists():
            to_delete.append(row["product_id"])
        else:
            valid_images.append((local_path, image_rel))

    # Delete products without images
    if to_delete:
        placeholders = ",".join(["?"] * len(to_delete))
        conn.execute(f"DELETE FROM products WHERE product_id IN ({placeholders})", to_delete)
        conn.commit()

    total_after = conn.execute("SELECT COUNT(*) FROM products").fetchone()[0]
    print(f"Products removed: {len(to_delete)}")
    print(f"Products remaining: {total_after}")
    print(f"Valid images: {len(valid_images)}")

    # Category breakdown
    cats = conn.execute("SELECT category, COUNT(*) FROM products GROUP BY category").fetchall()
    for cat, cnt in cats:
        print(f"  {cat}: {cnt}")
    conn.close()

    # Upload to GCS
    if upload_gcs and valid_images:
        try:
            from google.cloud import storage
            project_id = os.environ.get("GCP_PROJECT_ID", "webagent2026")
            client = storage.Client(project=project_id)
            bucket_name = os.environ.get("GCS_BUCKET_NAME", "web-agent-data-caleb-2026")
            bucket = client.bucket(bucket_name)
            print(f"\n📦 Uploading {len(valid_images)} images to gs://{bucket_name}/...")

            successes = 0
            failures = 0

            def upload_one(args):
                local_path, gcs_key = args
                try:
                    blob = bucket.blob(gcs_key)
                    blob.upload_from_filename(str(local_path), content_type="image/jpeg")
                    return True
                except Exception:
                    return False

            with ThreadPoolExecutor(max_workers=16) as executor:
                futures = {executor.submit(upload_one, item): item for item in valid_images}
                for future in tqdm(as_completed(futures), total=len(futures), desc="  GCS Upload"):
                    if future.result():
                        successes += 1
                    else:
                        failures += 1

            print(f"  ✅ GCS: {successes} uploaded, {failures} failed")
            print(f"  Public URL example: https://storage.googleapis.com/{bucket_name}/{valid_images[0][1]}")

        except Exception as e:
            print(f"❌ GCS upload failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--upload-gcs", action="store_true", help="Upload images to GCS after cleanup")
    args = parser.parse_args()
    clean_and_upload(upload_gcs=args.upload_gcs)


# ========================================
# ----- MERGED FROM: 00d_cleanup_metadata.py -----
# ========================================

"""Cleanup: remove products without images, check DB size, upload metadata to GCS."""
import os
import sqlite3
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

LOCAL_IMAGE_DIR = Path(r"C:\Users\Caleb\Desktop\product_images")
DB_PATH = Path("data/processed/catalog.db")


def cleanup_and_upload_metadata():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    total_before = conn.execute("SELECT COUNT(*) FROM products").fetchone()[0]
    print(f"Products before cleanup: {total_before}")

    # Find products without matching local images
    rows = conn.execute("SELECT product_id, image_urls FROM products").fetchall()
    to_delete = []
    valid_count = 0

    for row in rows:
        image_rel = row["image_urls"]
        if not image_rel:
            to_delete.append(row["product_id"])
            continue
        local_path = LOCAL_IMAGE_DIR / image_rel
        if not local_path.exists():
            to_delete.append(row["product_id"])
        else:
            valid_count += 1

    if to_delete:
        placeholders = ",".join(["?"] * len(to_delete))
        conn.execute(f"DELETE FROM products WHERE product_id IN ({placeholders})", to_delete)
        conn.commit()

    # Vacuum to reclaim space
    conn.execute("VACUUM")
    conn.close()

    total_after = sqlite3.connect(str(DB_PATH)).execute("SELECT COUNT(*) FROM products").fetchone()[0]
    db_size_mb = DB_PATH.stat().st_size / (1024 * 1024)

    print(f"Products removed (no image): {len(to_delete)}")
    print(f"Products remaining: {total_after}")
    print(f"Database size: {db_size_mb:.1f} MB")

    # Category breakdown
    conn2 = sqlite3.connect(str(DB_PATH))
    cats = conn2.execute("SELECT category, COUNT(*) FROM products GROUP BY category").fetchall()
    for cat, cnt in cats:
        print(f"  {cat}: {cnt}")
    conn2.close()

    # Check if too large for GitHub (>50MB is risky, >100MB fails)
    if db_size_mb > 50:
        print(f"\n⚠️  Database is {db_size_mb:.1f} MB — too large for GitHub!")
        print(f"   Adding to .gitignore and uploading to GCS...")
    else:
        print(f"\n✅ Database is {db_size_mb:.1f} MB — small enough for GitHub")

    # Upload metadata DB to GCS
    print(f"\nUploading catalog.db to GCS...")
    try:
        from google.cloud import storage
        project_id = os.environ.get("GCP_PROJECT_ID", "webagent2026")
        client = storage.Client(project=project_id)
        bucket_name = os.environ.get("GCS_BUCKET_NAME", "web-agent-data-caleb-2026")
        bucket = client.bucket(bucket_name)

        blob = bucket.blob("metadata/catalog.db")
        blob.upload_from_filename(str(DB_PATH))
        print(f"✅ Uploaded to gs://{bucket_name}/metadata/catalog.db")
        print(f"   Public URL: https://storage.googleapis.com/{bucket_name}/metadata/catalog.db")
    except Exception as e:
        print(f"❌ GCS upload failed: {e}")

    return db_size_mb


if __name__ == "__main__":
    size = cleanup_and_upload_metadata()


# ========================================
# ----- MERGED FROM: 00e_enrich_catalog.py -----
# ========================================

"""
00e_enrich_catalog.py — Enrich Catalog with Colors, Sizes & Synthetic Prices
==============================================================================
1. Parses color/size from existing attributes & descriptions for Amazon products
2. Updates dedup_hash to include color (so variants aren't flagged as duplicates)
3. Assigns synthetic prices to H&M products based on garment type
4. Re-uploads enriched catalog.db to GCS
"""

import hashlib
import json
import os
import random
import re
import sqlite3
from pathlib import Path

DB_PATH = Path("data/processed/catalog.db")

# ================================================================
# H&M Synthetic Price Ranges (USD)
# ================================================================

HM_PRICE_RANGES = {
    # Upper body
    "Garment Upper body": {
        "T-shirt": (15, 35),
        "Blouse": (25, 60),
        "Sweater": (30, 80),
        "Jacket": (50, 150),
        "Coat": (80, 200),
        "Hoodie": (30, 70),
        "Cardigan": (30, 75),
        "Vest": (20, 50),
        "Top": (15, 45),
        "Shirt": (20, 55),
        "Polo": (20, 45),
        "_default": (20, 60),
    },
    # Lower body
    "Garment Lower body": {
        "Jeans": (30, 70),
        "Trousers": (25, 65),
        "Shorts": (15, 40),
        "Skirt": (20, 55),
        "Leggings": (15, 35),
        "Pants": (25, 60),
        "_default": (20, 55),
    },
    # Full body
    "Garment Full body": {
        "Dress": (30, 100),
        "Jumpsuit": (35, 90),
        "Overall": (35, 85),
        "Romper": (25, 65),
        "_default": (30, 80),
    },
    # Accessories
    "Accessories": {
        "Bag": (20, 60),
        "Hat": (10, 30),
        "Cap": (10, 25),
        "Scarf": (15, 40),
        "Belt": (12, 35),
        "Gloves": (10, 25),
        "Jewellery": (8, 30),
        "Sunglasses": (10, 25),
        "_default": (10, 35),
    },
    # Underwear
    "Underwear": {
        "Bra": (15, 40),
        "Brief": (8, 20),
        "Boxer": (10, 25),
        "_default": (10, 30),
    },
    # Socks
    "Socks & Tights": {
        "_default": (5, 20),
    },
    # Shoes
    "Shoes": {
        "Sneaker": (30, 80),
        "Boot": (50, 120),
        "Sandal": (20, 50),
        "Slipper": (15, 35),
        "_default": (25, 70),
    },
    # Swimwear
    "Swimwear": {
        "_default": (15, 50),
    },
    # Nightwear
    "Nightwear": {
        "_default": (15, 45),
    },
    # Cosmetic
    "Cosmetic": {
        "_default": (5, 30),
    },
}

# Fallback
DEFAULT_RANGE = (15, 50)


def get_hm_price(category: str, title: str, attributes_json: str) -> float:
    """Assign a realistic synthetic price for an H&M product."""
    random.seed(hash(title))  # Deterministic per product

    cat_ranges = HM_PRICE_RANGES.get(category, {})
    title_lower = title.lower()

    # Try to match garment type
    attrs = {}
    try:
        attrs = json.loads(attributes_json) if attributes_json else {}
    except json.JSONDecodeError:
        pass

    garment_type = attrs.get("garment_type", "")

    # Check title and garment type for matching keywords
    for keyword, (low, high) in cat_ranges.items():
        if keyword == "_default":
            continue
        kw = keyword.lower()
        if kw in title_lower or kw in garment_type.lower():
            return round(random.uniform(low, high), 2)

    # Use category default
    default = cat_ranges.get("_default", DEFAULT_RANGE)
    return round(random.uniform(default[0], default[1]), 2)


# ================================================================
# Color/Size Extraction for Amazon
# ================================================================

COLOR_KEYWORDS = [
    "Black", "White", "Red", "Blue", "Green", "Yellow", "Pink", "Purple",
    "Orange", "Brown", "Grey", "Gray", "Navy", "Beige", "Ivory", "Gold",
    "Silver", "Teal", "Maroon", "Burgundy", "Olive", "Coral", "Turquoise",
    "Charcoal", "Khaki", "Tan", "Cream", "Rose", "Lavender", "Mint",
    "Multi", "Multicolor", "Floral", "Camo", "Plaid", "Striped",
]

SIZE_KEYWORDS = [
    "XXS", "XS", "S", "M", "L", "XL", "XXL", "XXXL", "2XL", "3XL", "4XL",
    "Small", "Medium", "Large", "Extra Large",
    "One Size", "Free Size", "Regular", "Plus Size", "Petite", "Tall",
]


def extract_color_size(title: str, description: str, attributes_json: str):
    """Extract color and size from product data."""
    attrs = {}
    try:
        attrs = json.loads(attributes_json) if attributes_json else {}
    except json.JSONDecodeError:
        pass

    # Try attributes first (most reliable)
    color = attrs.get("color", attrs.get("Color", ""))
    size = attrs.get("size", attrs.get("Size", ""))

    # Then try title/description
    if not color:
        text = f"{title} {description[:300]}"
        for c in COLOR_KEYWORDS:
            if re.search(r'\b' + c + r'\b', text, re.IGNORECASE):
                color = c
                break

    if not size:
        text = f"{title} {description[:300]}"
        for s in SIZE_KEYWORDS:
            if re.search(r'\b' + re.escape(s) + r'\b', text, re.IGNORECASE):
                size = s
                break

    return color, size


# ================================================================
# Main
# ================================================================

def enrich():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    total = conn.execute("SELECT COUNT(*) FROM products").fetchone()[0]
    print(f"Enriching {total} products...\n")

    rows = conn.execute(
        "SELECT product_id, title, description, category, attributes, source, price, dedup_hash "
        "FROM products"
    ).fetchall()

    prices_added = 0
    colors_extracted = 0
    sizes_extracted = 0
    dedup_updated = 0

    for row in rows:
        product_id = row["product_id"]
        title = row["title"]
        desc = row["description"]
        category = row["category"]
        attrs_json = row["attributes"]
        source = row["source"]
        price = row["price"]

        updates = {}
        attrs = {}
        try:
            attrs = json.loads(attrs_json) if attrs_json else {}
        except json.JSONDecodeError:
            attrs = {}

        # --- Extract color/size ---
        color, size = extract_color_size(title, desc, attrs_json)
        attrs_changed = False

        if color and not attrs.get("color"):
            attrs["color"] = color
            colors_extracted += 1
            attrs_changed = True

        if size and not attrs.get("size"):
            attrs["size"] = size
            sizes_extracted += 1
            attrs_changed = True

        if attrs_changed:
            updates["attributes"] = json.dumps(attrs)

        # --- Assign synthetic H&M prices ---
        if source == "hm" and (price is None or price == 0):
            new_price = get_hm_price(category, title, attrs_json)
            updates["price"] = new_price
            prices_added += 1

        # --- Fix dedup hash to include color ---
        color_val = attrs.get("color", "")
        new_raw = f"{title.lower().strip()}|{color_val.lower()}|{source}"
        new_hash = hashlib.sha256(new_raw.encode()).hexdigest()[:16]
        if new_hash != row["dedup_hash"]:
            updates["dedup_hash"] = new_hash
            dedup_updated += 1

        # --- Apply updates ---
        if updates:
            set_clause = ", ".join(f"{k} = ?" for k in updates.keys())
            values = list(updates.values()) + [product_id]
            conn.execute(f"UPDATE products SET {set_clause} WHERE product_id = ?", values)

    conn.commit()

    # Re-check duplicates
    dupes = conn.execute(
        "SELECT dedup_hash, COUNT(*) as cnt FROM products GROUP BY dedup_hash HAVING cnt > 1"
    ).fetchall()
    total_dupes = sum(r["cnt"] - 1 for r in dupes)

    print(f"ENRICHMENT RESULTS:")
    print(f"  H&M prices assigned: {prices_added}")
    print(f"  Colors extracted: {colors_extracted}")
    print(f"  Sizes extracted: {sizes_extracted}")
    print(f"  Dedup hashes updated: {dedup_updated}")
    print(f"  Remaining duplicates: {len(dupes)} groups ({total_dupes} products)")

    # Price stats
    price_stats = conn.execute(
        "SELECT source, COUNT(*) as total, "
        "SUM(CASE WHEN price IS NOT NULL AND price > 0 THEN 1 ELSE 0 END) as has_price, "
        "AVG(CASE WHEN price IS NOT NULL AND price > 0 THEN price END) as avg_price "
        "FROM products GROUP BY source"
    ).fetchall()
    print(f"\n  PRICE COVERAGE (after enrichment):")
    for r in price_stats:
        pct = r["has_price"] / r["total"] * 100
        avg = r["avg_price"] or 0
        print(f"    {r['source']}: {r['has_price']}/{r['total']} ({pct:.1f}%) avg=${avg:.2f}")

    conn.close()

    # Upload enriched DB to GCS
    print(f"\nUploading enriched catalog.db to GCS...")
    try:
        from google.cloud import storage
        project_id = os.environ.get("GCP_PROJECT_ID", "webagent2026")
        client = storage.Client(project=project_id)
        bucket_name = os.environ.get("GCS_BUCKET_NAME", "web-agent-data-caleb-2026")
        bucket = client.bucket(bucket_name)
        blob = bucket.blob("metadata/catalog.db")
        blob.upload_from_filename(str(DB_PATH))
        print(f"  Uploaded to gs://{bucket_name}/metadata/catalog.db")
    except Exception as e:
        print(f"  GCS upload failed: {e}")

    print(f"\nDone!")


if __name__ == "__main__":
    enrich()


# ========================================
# ----- MERGED FROM: 00f_dedup_and_copy.py -----
# ========================================

"""Remove remaining duplicates and copy DB to local product folder."""
import os
import shutil
import sqlite3
from pathlib import Path

DB_PATH = Path("data/processed/catalog.db")
LOCAL_COPY = Path(r"C:\Users\Caleb\Desktop\product_images\catalog.db")

conn = sqlite3.connect(str(DB_PATH))

total_before = conn.execute("SELECT COUNT(*) FROM products").fetchone()[0]
print(f"Products before dedup: {total_before}")

# Find duplicate groups — keep the one with the best data (most fields filled)
dupes = conn.execute("""
    SELECT dedup_hash, GROUP_CONCAT(product_id) as pids, COUNT(*) as cnt
    FROM products
    GROUP BY dedup_hash
    HAVING cnt > 1
""").fetchall()

to_delete = []
for row in dupes:
    pids = row[1].split(",")
    # Score each: prefer ones with price, rating, longer description
    best_pid = None
    best_score = -1
    for pid in pids:
        r = conn.execute(
            "SELECT product_id, price, rating, LENGTH(description) as dlen, LENGTH(reviews_summary) as rlen "
            "FROM products WHERE product_id = ?", (pid,)
        ).fetchone()
        score = 0
        if r[1] is not None and r[1] > 0:  # has price
            score += 3
        if r[2] is not None:  # has rating
            score += 2
        score += (r[3] or 0) / 100  # longer desc = better
        score += (r[4] or 0) / 50   # has reviews = better
        if score > best_score:
            best_score = score
            best_pid = pid
    # Delete all except the best one
    for pid in pids:
        if pid != best_pid:
            to_delete.append(pid)

if to_delete:
    # Delete in batches
    for i in range(0, len(to_delete), 500):
        batch = to_delete[i:i+500]
        placeholders = ",".join(["?"] * len(batch))
        conn.execute(f"DELETE FROM products WHERE product_id IN ({placeholders})", batch)
    conn.commit()

# Vacuum
conn.execute("VACUUM")

total_after = conn.execute("SELECT COUNT(*) FROM products").fetchone()[0]
db_size_mb = DB_PATH.stat().st_size / (1024 * 1024)

print(f"Duplicates removed: {len(to_delete)}")
print(f"Products remaining: {total_after}")
print(f"Database size: {db_size_mb:.1f} MB")

# Category breakdown
cats = conn.execute("SELECT source, category, COUNT(*) as cnt FROM products GROUP BY source, category ORDER BY source, cnt DESC").fetchall()
print(f"\nBreakdown:")
for r in cats:
    print(f"  [{r[0]}] {r[1]}: {r[2]}")

conn.close()

# Copy DB to local product folder
LOCAL_COPY.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(str(DB_PATH), str(LOCAL_COPY))
print(f"\nCopied catalog.db to {LOCAL_COPY}")

# Upload cleaned DB to GCS
print(f"Uploading cleaned catalog.db to GCS...")
try:
    from google.cloud import storage
    project_id = os.environ.get("GCP_PROJECT_ID", "webagent2026")
    client = storage.Client(project=project_id)
    bucket_name = os.environ.get("GCS_BUCKET_NAME", "web-agent-data-caleb-2026")
    bucket = client.bucket(bucket_name)
    blob = bucket.blob("metadata/catalog.db")
    blob.upload_from_filename(str(DB_PATH))
    print(f"  Uploaded to gs://{bucket_name}/metadata/catalog.db")
except Exception as e:
    print(f"  GCS upload failed: {e}")

print(f"\nDone!")



if __name__ == "__main__":
    print("\n\n========== Running 00c_cleanup_and_upload ==========")
    main_00c_cleanup_and_upload()
    print("\n\n========== Running 00d_cleanup_metadata ==========")
    main_00d_cleanup_metadata()
    print("\n\n========== Running 00e_enrich_catalog ==========")
    main_00e_enrich_catalog()
    print("\n\n========== Running 00f_dedup_and_copy ==========")
    main_00f_dedup_and_copy()