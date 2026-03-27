"""
Quick terminal script to test LanceDB Semantic & Visual Search Quality
"""
import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database import LanceDBCatalog, SQLiteCatalog
from src.embeddings import BGEM3Embedder, DINOv2Embedder


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Test RAG Vectors")
    parser.add_argument("--query", type=str, help="Text to search for (Semantic Search)")
    parser.add_argument("--image", type=str, help="Path to image to search for (Visual Search)")
    parser.add_argument("--k", type=int, default=5, help="Number of results")
    parser.add_argument("--config", default="config/settings.yaml")
    args = parser.parse_args()

    if not args.query and not args.image:
        print("Please provide --query 'some text' or --image 'path/to/img.jpg'")
        return

    config = load_config(args.config)
    sqlite = SQLiteCatalog(config["databases"]["sqlite"]["path"])
    lancedb = LanceDBCatalog(
        db_path=config["databases"]["lancedb"]["path"],
        table_name=config["databases"]["lancedb"]["table_name"],
        visual_dim=config["embeddings"]["visual"]["dimension"],
        semantic_dim=config["embeddings"]["semantic"]["dimension"],
    )

    print(f"\nConnected to LanceDB ({lancedb.count()} vectors loaded)")

    results = []

    # Semantic Search
    if args.query:
        print(f"Loading BGE-M3 to embed query: '{args.query}'...")
        embedder = BGEM3Embedder(model_id=config["embeddings"]["semantic"]["model_id"])
        query_vector = embedder.embed_query(args.query)
        
        print(f"\nSearching LanceDB...")
        results = lancedb.search_semantic(query_vector.tolist(), top_k=args.k)

    # Visual (DINOv2) Search
    elif args.image:
        print(f"Loading DINOv2 to embed image: '{args.image}'...")
        embedder = DINOv2Embedder(model_id=config["embeddings"]["visual"]["model_id"])
        query_vector = embedder.embed_single(args.image)
        if query_vector is None:
            print("Failed to embed image.")
            return

        print(f"\nSearching LanceDB...")
        results = lancedb.search_visual(query_vector.tolist(), top_k=args.k)

    # Display Results
    print(f"\n{'='*60}\nTOP {args.k} MATCHES\n{'='*60}")
    for i, res in enumerate(results):
        pid = res["product_id"]
        # Distance metric depends on setup (often lower is better for L2, or LanceDB cosine implementation)
        score = res.get("_distance", 0) 
        
        # Get actual product info from SQLite
        row = sqlite.get_product(pid)
        if row:
            print(f"[{i+1}] {row['title']}")
            print(f"    ID: {pid} | Score: {score:.4f} | Source: {row['source']}")
            desc = row["description"] or ""
            print(f"    Desc: {desc[:100]}...")
            print(f"    Price: ${row['price']} | Images: {row['image_urls'].split(',')[0][:50]}")
            print("-" * 60)
        else:
            print(f"[{i+1}] {pid} (Not found in metadata DB)")

    sqlite.close()

if __name__ == "__main__":
    main()
