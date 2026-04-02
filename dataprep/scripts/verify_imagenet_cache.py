#!/usr/bin/env python3
"""Verify ImageNet slipstream cache integrity against source images.

Reads samples from the cache, re-processes the corresponding source image
with the same s256_l512 pipeline, and compares the results byte-for-byte.

Usage:
    # Verify every 1000th image (default)
    uv run python dataprep/scripts/verify_imagenet_cache.py --root /path/to/imagenet --split train --cache-dir /path/to/cache

    # Verify every 100th image
    uv run python dataprep/scripts/verify_imagenet_cache.py --root /path/to/imagenet --split train --cache-dir /path/to/cache --stride 100

    # Verify all images (slow)
    uv run python dataprep/scripts/verify_imagenet_cache.py --root /path/to/imagenet --split train --cache-dir /path/to/cache --stride 1
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np


def verify_cache(
    root: str,
    split: str,
    cache_dir: str,
    stride: int,
    fmt: str,
):
    import torch
    from torchvision.datasets import ImageNet
    from torchvision.io import decode_image, encode_jpeg
    from torchvision.transforms.functional import center_crop, resize
    from slipstream.cache import OptimizedCache

    os.environ.setdefault("OMP_NUM_THREADS", "1")

    print(f"Source: {root} ({split})")
    print(f"Cache:  {cache_dir}")
    print(f"Format: {fmt}")
    print(f"Stride: every {stride}th image")
    print()

    # Load source dataset (for file paths and labels)
    print("Loading source dataset...")
    source = ImageNet(root, split=split)
    print(f"  {len(source)} images")

    # Load cache
    print("Loading cache...")
    cache = OptimizedCache.load(cache_dir)
    print(f"  {cache.num_samples} samples")

    assert len(source) == cache.num_samples, (
        f"Sample count mismatch: source={len(source)}, cache={cache.num_samples}"
    )

    # Verify
    indices = list(range(0, len(source), stride))
    num_checks = len(indices)
    print(f"\nVerifying {num_checks} samples...")

    mismatches = []
    t0 = time.time()

    for count, idx in enumerate(indices):
        # Get source image
        fullpath, label = source.imgs[idx]
        relpath = os.path.join(
            Path(fullpath).parent.name,
            Path(fullpath).name,
        )

        # Re-process source image (same as ImageNet1k_s256_l512.__getitem__)
        with open(fullpath, 'rb') as f:
            raw_bytes = f.read()

        img = decode_image(
            torch.frombuffer(bytearray(raw_bytes), dtype=torch.uint8),
            mode="RGB",
        )
        img = resize(img, 256, antialias=True)
        _, h, w = img.shape
        crop_h, crop_w = min(h, 512), min(w, 512)
        if crop_h < h or crop_w < w:
            img = center_crop(img, [crop_h, crop_w])

        # Get cached sample
        batch = cache.load_batch(np.array([idx], dtype=np.int64), fields=['image', 'label', 'index', 'path'])

        cached_size = batch['image']['sizes'][0]
        cached_bytes = bytes(batch['image']['data'][0][:cached_size])
        cached_label = batch['label']['data'][0]
        cached_index = batch['index']['data'][0]
        cached_path = batch['path']['data'][0]

        # Compare all fields
        errors = []
        if cached_index != idx:
            errors.append(f"index: expected {idx}, got {cached_index}")
        if cached_label != label:
            errors.append(f"label: expected {label}, got {cached_label}")
        if cached_path != relpath:
            errors.append(f"path: expected {relpath}, got {cached_path}")

        if fmt == "jpeg":
            expected_bytes = encode_jpeg(img, quality=100).numpy().tobytes()
            if cached_bytes != expected_bytes:
                errors.append(
                    f"image bytes differ: expected {len(expected_bytes)}, "
                    f"got {len(cached_bytes)}"
                )
        else:
            # YUV420: replicate the exact cache build pipeline.
            # The builder receives JPEG bytes from ImageNet1k_s256_l512,
            # then decodes with PIL (decode_image_to_rgb) and converts
            # to YUV420. We do the same here for exact byte match.
            from slipstream.cache import decode_image_to_rgb, rgb_to_yuv420

            jpeg_bytes = encode_jpeg(img, quality=100).numpy().tobytes()
            rgb = decode_image_to_rgb(jpeg_bytes)
            expected_yuv, expected_h, expected_w = rgb_to_yuv420(rgb)

            cached_h = int(batch['image']['heights'][0])
            cached_w = int(batch['image']['widths'][0])

            if (expected_h, expected_w) != (cached_h, cached_w):
                errors.append(
                    f"dimensions: expected {expected_h}x{expected_w}, "
                    f"got {cached_h}x{cached_w}"
                )
            elif expected_yuv != cached_bytes:
                errors.append(
                    f"YUV bytes differ: expected {len(expected_yuv)}, "
                    f"got {len(cached_bytes)}"
                )

        if errors:
            mismatches.append((idx, relpath, errors))
            print(f"  FAIL [{idx}] {relpath}: {'; '.join(errors)}")

        if (count + 1) % 100 == 0 or count + 1 == num_checks:
            elapsed = time.time() - t0
            rate = (count + 1) / elapsed
            print(
                f"  Checked {count + 1}/{num_checks} "
                f"({rate:.0f}/s, {len(mismatches)} failures)",
                end="\r",
            )

    elapsed = time.time() - t0
    print()
    print()

    if mismatches:
        print(f"FAILED: {len(mismatches)}/{num_checks} samples mismatched")
        for idx, path, errors in mismatches[:10]:
            print(f"  [{idx}] {path}: {'; '.join(errors)}")
        if len(mismatches) > 10:
            print(f"  ... and {len(mismatches) - 10} more")
        return False
    else:
        print(f"PASSED: {num_checks}/{num_checks} samples verified in {elapsed:.1f}s")
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Verify ImageNet slipstream cache integrity")
    parser.add_argument("--root", type=str, required=True,
                        help="ImageNet root directory")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "val"],
                        help="Dataset split (default: train)")
    parser.add_argument("--cache-dir", type=str, required=True,
                        help="Path to the slipstream cache directory")
    parser.add_argument("--stride", type=int, default=1000,
                        help="Check every Nth image (default: 1000)")
    parser.add_argument("--fmt", type=str, default="jpeg",
                        choices=["jpeg", "yuv420"],
                        help="Cache format (default: jpeg)")
    args = parser.parse_args()

    success = verify_cache(
        root=args.root,
        split=args.split,
        cache_dir=args.cache_dir,
        stride=args.stride,
        fmt=args.fmt,
    )
    exit(0 if success else 1)


if __name__ == "__main__":
    main()
