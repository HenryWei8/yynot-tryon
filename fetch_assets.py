#!/usr/bin/env python3
"""Fetch sample model and garment images from VITON-HD test set on HuggingFace."""
from huggingface_hub import hf_hub_download
import shutil, os

os.makedirs("assets/models", exist_ok=True)
os.makedirs("assets/garments", exist_ok=True)

person_files = [
    "test/image/00008_00.jpg",
    "test/image/00034_00.jpg",
]
garment_files = [
    "test/cloth/00034_00.jpg",
    "test/cloth/00057_00.jpg",
    "test/cloth/00100_00.jpg",
    "test/cloth/00191_00.jpg",
    "test/cloth/00321_00.jpg",
]

for f in person_files:
    slug = os.path.basename(f).replace(".jpg", "")
    try:
        path = hf_hub_download(
            repo_id="SaffalPoosh/VITON-HD-test",
            filename=f,
            repo_type="dataset",
            local_dir="/tmp/vitonhd"
        )
        shutil.copy(path, f"assets/models/{slug}.jpg")
        print(f"✓ model: {slug}.jpg")
    except Exception as e:
        print(f"✗ skip model {slug}: {e}")

for f in garment_files:
    slug = os.path.basename(f).replace(".jpg", "")
    try:
        path = hf_hub_download(
            repo_id="SaffalPoosh/VITON-HD-test",
            filename=f,
            repo_type="dataset",
            local_dir="/tmp/vitonhd"
        )
        shutil.copy(path, f"assets/garments/{slug}.jpg")
        print(f"✓ garment: {slug}.jpg")
    except Exception as e:
        print(f"✗ skip garment {slug}: {e}")

print("Done. Assets ready.")
