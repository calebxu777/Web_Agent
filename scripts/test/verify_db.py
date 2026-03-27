"""Quality audit of the catalog database."""
import sqlite3
import json
from pathlib import Path

DB_PATH = "data/processed/catalog.db"
LOCAL_IMAGE_DIR = Path(r"C:\Users\Caleb\Desktop\product_images")

conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row

total = conn.execute("SELECT COUNT(*) FROM products").fetchone()[0]
print(f"{'='*70}")
print(f"  CATALOG QUALITY AUDIT — {total} products")
print(f"{'='*70}")

# --- Category breakdown ---
print(f"\n📦 CATEGORY BREAKDOWN:")
cats = conn.execute("SELECT category, COUNT(*) as cnt FROM products GROUP BY category ORDER BY cnt DESC").fetchall()
for r in cats:
    print(f"  {r['category']}: {r['cnt']}")

# --- Source breakdown ---
print(f"\n🏷️ SOURCE BREAKDOWN:")
sources = conn.execute("SELECT source, COUNT(*) as cnt FROM products GROUP BY source ORDER BY cnt DESC").fetchall()
for r in sources:
    print(f"  {r['source']}: {r['cnt']}")

# --- Title quality ---
print(f"\n📝 TITLE QUALITY:")
empty_titles = conn.execute("SELECT COUNT(*) FROM products WHERE title = '' OR title IS NULL").fetchone()[0]
short_titles = conn.execute("SELECT COUNT(*) FROM products WHERE LENGTH(title) < 10").fetchone()[0]
avg_title_len = conn.execute("SELECT AVG(LENGTH(title)) FROM products").fetchone()[0]
print(f"  Empty titles: {empty_titles}")
print(f"  Short titles (<10 chars): {short_titles}")
print(f"  Avg title length: {avg_title_len:.0f} chars")

# --- Description quality ---
print(f"\n📄 DESCRIPTION QUALITY:")
empty_desc = conn.execute("SELECT COUNT(*) FROM products WHERE description = '' OR description IS NULL").fetchone()[0]
has_desc = total - empty_desc
avg_desc_len = conn.execute("SELECT AVG(LENGTH(description)) FROM products WHERE description != ''").fetchone()[0]
print(f"  Has description: {has_desc} ({has_desc/total*100:.1f}%)")
print(f"  Empty description: {empty_desc} ({empty_desc/total*100:.1f}%)")
print(f"  Avg description length: {avg_desc_len:.0f} chars")

# --- Price coverage ---
print(f"\n💰 PRICE COVERAGE:")
has_price = conn.execute("SELECT COUNT(*) FROM products WHERE price IS NOT NULL AND price > 0").fetchone()[0]
no_price = total - has_price
avg_price = conn.execute("SELECT AVG(price) FROM products WHERE price IS NOT NULL AND price > 0").fetchone()[0]
min_price = conn.execute("SELECT MIN(price) FROM products WHERE price IS NOT NULL AND price > 0").fetchone()[0]
max_price = conn.execute("SELECT MAX(price) FROM products WHERE price IS NOT NULL AND price > 0").fetchone()[0]
print(f"  Has price: {has_price} ({has_price/total*100:.1f}%)")
print(f"  No price: {no_price} ({no_price/total*100:.1f}%)")
if avg_price:
    print(f"  Avg price: ${avg_price:.2f}")
    print(f"  Range: ${min_price:.2f} - ${max_price:.2f}")

# By source
for src in ["amazon", "hm"]:
    hp = conn.execute("SELECT COUNT(*) FROM products WHERE source=? AND price IS NOT NULL AND price > 0", (src,)).fetchone()[0]
    t = conn.execute("SELECT COUNT(*) FROM products WHERE source=?", (src,)).fetchone()[0]
    print(f"  {src}: {hp}/{t} have prices ({hp/t*100:.1f}%)" if t > 0 else "")

# --- Rating/Review coverage ---
print(f"\n⭐ RATING & REVIEW COVERAGE:")
has_rating = conn.execute("SELECT COUNT(*) FROM products WHERE rating IS NOT NULL").fetchone()[0]
has_reviews = conn.execute("SELECT COUNT(*) FROM products WHERE review_count > 0").fetchone()[0]
has_summary = conn.execute("SELECT COUNT(*) FROM products WHERE reviews_summary != '' AND reviews_summary IS NOT NULL").fetchone()[0]
avg_rating = conn.execute("SELECT AVG(rating) FROM products WHERE rating IS NOT NULL").fetchone()[0]
print(f"  Has rating: {has_rating} ({has_rating/total*100:.1f}%)")
print(f"  Has review count: {has_reviews} ({has_reviews/total*100:.1f}%)")
print(f"  Has review summary: {has_summary} ({has_summary/total*100:.1f}%)")
if avg_rating:
    print(f"  Avg rating: {avg_rating:.1f}")

# --- Brand coverage ---
print(f"\n🏪 BRAND COVERAGE:")
has_brand = conn.execute("SELECT COUNT(*) FROM products WHERE brand != '' AND brand IS NOT NULL").fetchone()[0]
print(f"  Has brand: {has_brand} ({has_brand/total*100:.1f}%)")
top_brands = conn.execute("SELECT brand, COUNT(*) as cnt FROM products WHERE brand != '' GROUP BY brand ORDER BY cnt DESC LIMIT 10").fetchall()
print(f"  Top brands:")
for r in top_brands:
    print(f"    {r['brand']}: {r['cnt']}")

# --- Attributes quality ---
print(f"\n🔧 ATTRIBUTES:")
has_attrs = conn.execute("SELECT COUNT(*) FROM products WHERE attributes != '' AND attributes != '{}' AND attributes IS NOT NULL").fetchone()[0]
print(f"  Has attributes: {has_attrs} ({has_attrs/total*100:.1f}%)")

# --- Image check (sample) ---
print(f"\n🖼️ IMAGE SAMPLE CHECK:")
samples = conn.execute("SELECT product_id, title, image_urls, source, price, rating FROM products ORDER BY RANDOM() LIMIT 10").fetchall()
for s in samples:
    img_path = LOCAL_IMAGE_DIR / s["image_urls"] if s["image_urls"] else None
    img_ok = "✅" if img_path and img_path.exists() else "❌"
    price_str = f"${s['price']:.2f}" if s['price'] else "N/A"
    rating_str = f"{s['rating']}★" if s['rating'] else "N/A"
    print(f"  {img_ok} [{s['source']}] {s['title'][:50]} | {price_str} | {rating_str}")

# --- Duplicates check ---
print(f"\n🔍 DUPLICATE CHECK:")
dupes = conn.execute("SELECT dedup_hash, COUNT(*) as cnt FROM products GROUP BY dedup_hash HAVING cnt > 1").fetchall()
print(f"  Duplicate groups: {len(dupes)}")
if dupes:
    total_dupes = sum(r['cnt'] - 1 for r in dupes)
    print(f"  Total duplicate products: {total_dupes}")

print(f"\n{'='*70}")
conn.close()
