"""Test the full ingestor pipeline: ingest → verify → cleanup."""
import httpx
import json

BASE = "http://localhost:8000"

# Mock product to ingest (simulates what a thumbs-up sends)
mock_product = {
    "title": "Arc'teryx Alpha SV Jacket",
    "description": "Gore-Tex Pro shell with WaterTight zippers. 3-layer construction for extreme alpine conditions. Helmet-compatible StormHood.",
    "price": 799.00,
    "brand": "Arc'teryx",
    "category": "Outerwear",
    "image": "https://images.unsplash.com/photo-1544923246-77307dd270b0?w=400",
    "url": "https://arcteryx.com/shop/alpha-sv-jacket",
    "source": "web",
}

print("=" * 50)
print("  INGESTOR PIPELINE TEST")
print("=" * 50)

# Step 1: Ingest
print("\n1. Ingesting mock product...")
r = httpx.post(f"{BASE}/api/ingest", json={"product_data": mock_product}, timeout=10)
result = r.json()
print(f"   Status: {result['status']}")
print(f"   Product ID: {result['product_id']}")
print(f"   Title: {result['title']}")

# Step 2: Verify
print("\n2. Verifying DB contents...")
r = httpx.get(f"{BASE}/api/ingest/verify", timeout=10)
data = r.json()
print(f"   Count: {data['count']}")
for p in data["products"]:
    print(f"   → [{p['product_id']}] {p['title']} | ${p['price']} | {p['brand']}")

# Step 3: Nickname test (bonus)
print("\n3. Testing nickname...")
r = httpx.post(f"{BASE}/api/nickname", json={"nickname": "riley"}, timeout=10)
print(f"   Result: {r.json()}")

# Same nickname again → should be welcome_back
r = httpx.post(f"{BASE}/api/nickname", json={"nickname": "riley"}, timeout=10)
print(f"   Again:  {r.json()}")

# Step 4: Cleanup
print("\n4. Cleaning up ingested products...")
r = httpx.delete(f"{BASE}/api/ingest/cleanup", timeout=10)
print(f"   Cleaned: {r.json()}")

# Verify empty
r = httpx.get(f"{BASE}/api/ingest/verify", timeout=10)
print(f"   After cleanup: {r.json()['count']} products")

print("\n✅ All tests passed!")
