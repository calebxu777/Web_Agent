"""
00b_kaggle_download_hm.py — Download H&M Dataset from Kaggle
================================================================
Downloads and extracts the H&M Personalized Fashion Recommendations
dataset from Kaggle. Run this BEFORE 00b_download_hm.py.

Prerequisites:
    pip install kaggle
    Set KAGGLE_API_TOKEN env var or place kaggle.json in ~/.kaggle/

Usage:
    python scripts/00b_kaggle_download_hm.py
"""

import os
import sys
import zipfile
from pathlib import Path

HM_RAW_DIR = Path("data/raw/hm")
COMPETITION = "h-and-m-personalized-fashion-recommendations"


def download_hm():
    HM_RAW_DIR.mkdir(parents=True, exist_ok=True)

    zip_path = HM_RAW_DIR / f"{COMPETITION}.zip"
    articles_file = HM_RAW_DIR / "articles.csv"

    # Skip if already extracted
    if articles_file.exists():
        print(f"✅ H&M dataset already extracted at {HM_RAW_DIR}")
        print(f"   articles.csv found — skipping download.")
        print(f"   Run 00b_download_hm.py to process it.")
        return

    # Download from Kaggle
    print(f"Downloading H&M dataset from Kaggle...")
    print(f"  Competition: {COMPETITION}")
    print(f"  Destination: {HM_RAW_DIR}")

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        api.competition_download_files(COMPETITION, path=str(HM_RAW_DIR), quiet=False)
    except Exception:
        # Fallback to CLI
        exit_code = os.system(
            f'kaggle competitions download -c {COMPETITION} -p "{HM_RAW_DIR}"'
        )
        if exit_code != 0:
            print("❌ Kaggle download failed. Make sure you:")
            print("   1. Have KAGGLE_API_TOKEN set or kaggle.json in ~/.kaggle/")
            print("   2. Have accepted the competition rules on kaggle.com")
            sys.exit(1)

    # Extract
    if zip_path.exists():
        print(f"\nExtracting {zip_path.name} ({zip_path.stat().st_size / 1e9:.1f} GB)...")
        with zipfile.ZipFile(str(zip_path), "r") as zf:
            zf.extractall(str(HM_RAW_DIR))
        print(f"✅ Extracted to {HM_RAW_DIR}")

        # Clean up zip to save disk space
        print(f"Removing zip to save space...")
        zip_path.unlink()
        print(f"✅ Deleted {zip_path.name}")
    else:
        print(f"❌ Zip file not found at {zip_path}")
        sys.exit(1)

    # Verify
    if articles_file.exists():
        print(f"\n✅ H&M dataset ready!")
        print(f"   articles.csv: {articles_file}")
        images_dir = HM_RAW_DIR / "images"
        if images_dir.exists():
            count = sum(1 for _ in images_dir.rglob("*.jpg"))
            print(f"   images/: {count} images found")
        print(f"\n   Next step: python scripts/00b_download_hm.py --max-products 20000 --upload-gcs")
    else:
        print(f"❌ articles.csv not found after extraction")


if __name__ == "__main__":
    download_hm()
