"""Test ImageNet-1K val preparation with parallel cache build.

Usage:
    uv run python scripts/test_imagenet_prep.py
"""
import os
import time
from pathlib import Path

import numpy as np

IMAGENET_ROOT = "/Users/gaa019/Datasets/imagenet1k/rawdata"
NUM_WORKERS = max(os.cpu_count() - 1, 1)
OUTPUT_DIR = Path("/tmp/slipstream-imagenet-val-test")


def main():
    from slipstream.prep import ImageNet1k_s256_l512
    from slipstream.cache import OptimizedCache

    # Clean previous test
    import shutil
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    # 1. Create prep dataset
    print("=" * 60)
    print("Step 1: Create ImageNet1k_s256_l512 dataset")
    print("=" * 60)
    dataset = ImageNet1k_s256_l512(IMAGENET_ROOT, split="val")
    print(dataset)
    print()

    # Quick sanity: check a single sample
    sample = dataset[0]
    print(f"Sample 0: image={len(sample['image'])} bytes, "
          f"label={sample['label']}, path={sample['path']}")
    assert isinstance(sample['image'], bytes)
    assert sample['image'][:2] == b'\xff\xd8', "Not a JPEG"
    print()

    # 2. Build cache with parallel workers
    print("=" * 60)
    print(f"Step 2: Build cache ({NUM_WORKERS} workers, {len(dataset):,} samples)")
    print("=" * 60)
    t0 = time.time()
    cache = OptimizedCache.build(
        dataset,
        output_dir=OUTPUT_DIR,
        num_workers=NUM_WORKERS,
    )
    elapsed = time.time() - t0
    print(f"\nBuild completed in {elapsed:.1f}s "
          f"({len(dataset) / elapsed:.0f} samples/sec)")
    print()

    # 3. Verify cache
    print("=" * 60)
    print("Step 3: Verify cache contents")
    print("=" * 60)
    print(f"Cache dir: {cache.cache_dir}")
    print(f"Samples: {cache.num_samples:,}")
    print(f"Fields: {list(cache.field_types.keys())}")

    # Spot-check: load a batch and verify
    indices = np.array([0, 100, 1000, 10000, 49999], dtype=np.int64)
    batch = cache.load_batch(indices, fields=['image', 'label', 'path'])

    print(f"\nSpot-check batch (indices {indices.tolist()}):")
    for i, idx in enumerate(indices):
        img_size = batch['image']['sizes'][i]
        label = batch['label']['data'][i]
        print(f"  [{idx:5d}] image={img_size:6d} bytes, label={label}")

    # Verify JPEG headers
    for i in range(len(indices)):
        img_data = batch['image']['data'][i]
        assert img_data[0] == 0xFF and img_data[1] == 0xD8, \
            f"Sample {indices[i]}: not a valid JPEG"

    # Verify labels match source
    for i, idx in enumerate(indices):
        src_sample = dataset[idx]
        assert batch['label']['data'][i] == src_sample['label'], \
            f"Label mismatch at {idx}"

    print("\nAll checks passed!")

    # 4. Cache file sizes
    print("\n" + "=" * 60)
    print("Step 4: Cache file sizes")
    print("=" * 60)
    cache_dir = OUTPUT_DIR / ".slipstream"
    total = 0
    for f in sorted(cache_dir.iterdir()):
        if f.is_file():
            size = f.stat().st_size
            total += size
            print(f"  {f.name:30s} {size / 1e6:8.1f} MB")
    print(f"  {'TOTAL':30s} {total / 1e6:8.1f} MB")


if __name__ == "__main__":
    main()
