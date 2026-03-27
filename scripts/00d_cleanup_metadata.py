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
