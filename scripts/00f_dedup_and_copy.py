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
