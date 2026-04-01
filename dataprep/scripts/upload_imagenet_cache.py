#!/usr/bin/env python3
"""Upload ImageNet-1K slipstream caches to S3.

Uploads pre-built caches to the lab's S3 bucket for sharing.

Usage:
    # Upload val caches (JPEG + YUV420)
    python dataprep/scripts/upload_imagenet_cache.py --split val

    # Upload train JPEG only
    python dataprep/scripts/upload_imagenet_cache.py --split train --fmt jpeg

    # Custom source and destination
    python dataprep/scripts/upload_imagenet_cache.py --split val --cache-dir /fast/storage --remote s3://my-bucket/caches/
"""
from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_REMOTE = "s3://visionlab-datasets/slipstream-cache/imagenet1k"
DEFAULT_CACHE_BASE = Path.home() / ".slipstream"


def upload_cache(
    cache_dir: Path,
    remote_path: str,
):
    from slipstream.s3_sync import upload_s3_cache

    if not cache_dir.exists():
        print(f"  Cache not found: {cache_dir}")
        print(f"  Run build_imagenet_cache.py first")
        return False

    print(f"  Local:  {cache_dir}")
    print(f"  Remote: {remote_path}")

    success = upload_s3_cache(cache_dir, remote_path)

    if success:
        print(f"  Upload complete")
    else:
        print(f"  Upload failed")

    return success


def main():
    parser = argparse.ArgumentParser(
        description="Upload ImageNet-1K caches to S3")
    parser.add_argument("--split", type=str, default="val",
                        choices=["train", "val"],
                        help="Dataset split (default: val)")
    parser.add_argument("--fmt", type=str, default="both",
                        choices=["jpeg", "yuv420", "both"],
                        help="Image format (default: both)")
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="Base cache directory (default: ~/.slipstream/)")
    parser.add_argument("--remote", type=str, default=DEFAULT_REMOTE,
                        help=f"S3 base path (default: {DEFAULT_REMOTE})")
    args = parser.parse_args()

    cache_base = Path(args.cache_dir) if args.cache_dir else DEFAULT_CACHE_BASE
    formats = ["jpeg", "yuv420"] if args.fmt == "both" else [args.fmt]

    for fmt in formats:
        cache_name = f"imagenet1k-s256_l512-{fmt}-{args.split}"
        cache_dir = cache_base / cache_name
        remote_path = f"{args.remote.rstrip('/')}/{cache_name}"

        print(f"\nUploading {cache_name}:")
        upload_cache(cache_dir, remote_path)


if __name__ == "__main__":
    main()
