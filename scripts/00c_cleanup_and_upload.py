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
