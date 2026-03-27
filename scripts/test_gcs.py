"""Verify GCS public URL access."""
import requests

url = "https://storage.googleapis.com/web-agent-data-caleb-2026/amazon/amz_0205291252.jpg"
r = requests.head(url)
print(f"URL: {url}")
print(f"Status: {r.status_code}")
print(f"Content-Type: {r.headers.get('content-type')}")
print(f"Size: {r.headers.get('content-length')} bytes")

if r.status_code == 200:
    print("\n✅ GCS public access works! Images are publicly accessible.")
else:
    print(f"\n❌ Got status {r.status_code}. Check bucket IAM permissions.")
