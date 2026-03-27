"""Quick test: hit the mock FastAPI and print the SSE pipeline."""
import httpx, json

r = httpx.post(
    "http://localhost:8000/api/chat",
    json={"message": "find me a winter jacket under $200", "hasImage": False, "webSearch": False},
    timeout=30,
)

lines = r.text.strip().split("\n\n")
for line in lines:
    data_str = line.replace("data: ", "", 1)
    if data_str == "[DONE]":
        print("  ✅ [DONE]")
        break
    try:
        d = json.loads(data_str)
        t = d.get("type")
        if t == "status":
            print(f"  STAGE: [{d['stage']}] {d['message']}")
        elif t == "products":
            print(f"  PRODUCTS: {len(d['items'])} items")
            for p in d["items"][:3]:
                print(f"    → {p['title']} | ${p['price']} | {p['brand']}")
        elif t == "token":
            pass  # skip token-by-token output
        elif t == "done":
            print("  ✅ done event")
    except json.JSONDecodeError:
        pass

print(f"\nTotal SSE events: {len(lines)}")
