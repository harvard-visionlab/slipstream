#!/usr/bin/env python3
"""Build slipstream cache for ImageNet-1K with s256_l512 preprocessing.

Builds JPEG and/or YUV420 caches with label indexes, using parallel
workers for fast processing.

Usage:
    # Build val set (JPEG + YUV420)
    python dataprep/scripts/build_imagenet_cache.py --root /path/to/imagenet --split val

    # Build train set (JPEG only)
    python dataprep/scripts/build_imagenet_cache.py --root /path/to/imagenet --split train --fmt jpeg

    # Custom output dir and workers
    python dataprep/scripts/build_imagenet_cache.py --root /path/to/imagenet --split val --output-dir /fast/storage --num-workers 24
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path


def build_cache(
    root: str,
    split: str,
    fmt: str,
    output_dir: str | None,
    num_workers: int,
):
    from slipstream.prep import ImageNet1k_s256_l512
    from slipstream.cache import OptimizedCache, write_index

    print("=" * 60)
    print(f"Building ImageNet-1K {split} cache ({fmt})")
    print("=" * 60)

    # Create dataset
    dataset = ImageNet1k_s256_l512(root, split=split)
    print(dataset)
    print()

    # Determine output directory
    if output_dir:
        cache_dir = Path(output_dir) / f"imagenet1k-s256_l512-{fmt}-{split}"
    else:
        cache_dir = Path(dataset.cache_path).parent / f"imagenet1k-s256_l512-{fmt}-{split}"

    print(f"Output: {cache_dir}")
    print(f"Workers: {num_workers}")
    print()

    # Build cache
    t0 = time.time()
    cache = OptimizedCache.build(
        dataset,
        output_dir=cache_dir,
        num_workers=num_workers,
        image_format=fmt,
    )
    elapsed = time.time() - t0
    print(f"\nBuild completed in {elapsed:.1f}s ({len(dataset)/elapsed:.0f} samples/sec)")

    # Build label index
    print("\nBuilding label index...")
    write_index(cache, fields=["label"])
    print("Label index built")

    # Print file sizes
    print(f"\nCache files:")
    total = 0
    for f in sorted(cache.cache_dir.iterdir()):
        if f.is_file():
            size = f.stat().st_size
            total += size
            print(f"  {f.name:30s} {size / 1e6:8.1f} MB")
    size_str = f"{total / 1e9:.2f} GB" if total > 1e9 else f"{total / 1e6:.1f} MB"
    print(f"  {'TOTAL':30s} {size_str}")

    return cache_dir


def main():
    parser = argparse.ArgumentParser(
        description="Build ImageNet-1K slipstream cache")
    parser.add_argument("--root", type=str, required=True,
                        help="ImageNet root directory (contains train/ and val/)")
    parser.add_argument("--split", type=str, default="val",
                        choices=["train", "val"],
                        help="Dataset split (default: val)")
    parser.add_argument("--fmt", type=str, default="both",
                        choices=["jpeg", "yuv420", "both"],
                        help="Image format (default: both)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: ~/.slipstream/)")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="Parallel workers (default: auto = ncpu-1)")
    args = parser.parse_args()

    if args.num_workers == 0:
        try:
            args.num_workers = len(os.sched_getaffinity(0)) - 1
        except AttributeError:
            args.num_workers = (os.cpu_count() or 8) - 1
        args.num_workers = max(args.num_workers, 1)

    formats = ["jpeg", "yuv420"] if args.fmt == "both" else [args.fmt]

    for fmt in formats:
        build_cache(
            root=args.root,
            split=args.split,
            fmt=fmt,
            output_dir=args.output_dir,
            num_workers=args.num_workers,
        )
        print()


if __name__ == "__main__":
    main()
