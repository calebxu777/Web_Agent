"""Quick test to verify Amazon JSONL download works."""
from huggingface_hub import hf_hub_download
import json

path = hf_hub_download(
    "McAuley-Lab/Amazon-Reviews-2023",
    "raw/meta_categories/meta_Clothing_Shoes_and_Jewelry.jsonl",
    repo_type="dataset",
)
print(f"Downloaded to: {path}")

with open(path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 3:
            break
        item = json.loads(line)
        print(f"\nItem {i}:")
        print(f"  keys: {list(item.keys())}")
        title = item.get("title", "")
        print(f"  title: {title[:80]}")
        print(f"  parent_asin: {item.get('parent_asin')}")
        print(f"  price: {item.get('price')}")
        imgs = item.get("images", [])
        print(f"  images: {type(imgs).__name__}, count={len(imgs) if isinstance(imgs, list) else 'N/A'}")
        if isinstance(imgs, list) and imgs:
            print(f"  first_image: {imgs[0]}")

print("\nDone!")
