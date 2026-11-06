#!/usr/bin/env python3
"""
High-speed parallel model downloader for zai-org/GLM-5.3-Flash and zai-org/GLM-5.3-Flash-BF16
into /mnt/localssd/hf_cache.
"""

import argparse
import importlib.util
import os
import time

if importlib.util.find_spec("hf_transfer") is not None:
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
else:
    os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
os.environ.setdefault("HF_HOME", "/mnt/localssd/hf_cache")

from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="zai-org/GLM-5.3-Flash")
    parser.add_argument("--cache-dir", type=str, default="/mnt/localssd/hf_cache/hub")
    args = parser.parse_args()

    t0 = time.time()
    print(f"Starting download of {args.model} to {args.cache_dir} (16 workers)...")
    local_dir = snapshot_download(
        repo_id=args.model,
        cache_dir=args.cache_dir,
        max_workers=16,
    )
    dt = time.time() - t0
    print(f"Completed download of {args.model} -> {local_dir} in {dt:.1f}s")


if __name__ == "__main__":
    main()
