# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # ImageNet-1K Validation Set — Slipstream Cache Preparation
#
# Build a slipstream cache from raw ImageNet validation images with
# **s256_l512** preprocessing (resize short edge to 256, center-crop
# long edge to 512, JPEG quality 100).
#
# Two cache formats are built:
# 1. **JPEG** — standard format, smaller on disk
# 2. **YUV420** — native YUV planes, faster decode for some decoders
#
# ## Prerequisites
# - Raw ImageNet val at `IMAGENET_ROOT/val/` with 1000 class subdirectories
# - slipstream on the `feature/parallel-cache-build` branch

# %% [markdown]
# ## Configuration

# %%
import os
from pathlib import Path

IMAGENET_ROOT = os.environ.get(
    "IMAGENET_ROOT",
    "/Users/gaa019/Datasets/imagenet1k/rawdata",
)
SPLIT = "val"
NUM_WORKERS = max(os.cpu_count() - 1, 1)

# Output directories (set to None to use default cache_path)
JPEG_OUTPUT_DIR = Path(os.environ.get(
    "JPEG_OUTPUT_DIR",
    os.path.expanduser("~/.slipstream/imagenet1k-s256_l512-jpeg-val"),
))
YUV_OUTPUT_DIR = Path(os.environ.get(
    "YUV_OUTPUT_DIR",
    os.path.expanduser("~/.slipstream/imagenet1k-s256_l512-yuv420-val"),
))

print(f"ImageNet root: {IMAGENET_ROOT}")
print(f"Split: {SPLIT}")
print(f"Workers: {NUM_WORKERS}")
print(f"JPEG output: {JPEG_OUTPUT_DIR}")
print(f"YUV output: {YUV_OUTPUT_DIR}")

# %% [markdown]
# ## Step 1: Create the prep dataset

# %%
from slipstream.prep import ImageNet1k_s256_l512

dataset = ImageNet1k_s256_l512(IMAGENET_ROOT, split=SPLIT)
print(dataset)

# %% [markdown]
# ### Quick sanity check — inspect a single sample

# %%
sample = dataset[0]
print(f"Fields: {list(sample.keys())}")
print(f"Image: {len(sample['image']):,} bytes (JPEG)")
print(f"Label: {sample['label']}")
print(f"Index: {sample['index']}")
print(f"Path: {sample['path']}")

# Verify it's valid JPEG
assert sample['image'][:2] == b'\xff\xd8', "Not a JPEG!"
print("\nJPEG header OK ✓")

# %%
# Visualize the preprocessed image
from PIL import Image
import io

img = Image.open(io.BytesIO(sample['image']))
print(f"Dimensions: {img.size[0]}x{img.size[1]}")
img

# %% [markdown]
# ## Step 2: Build JPEG cache (parallel)

# %%
import time
from slipstream.cache import OptimizedCache

print(f"Building JPEG cache with {NUM_WORKERS} workers...")
t0 = time.time()
jpeg_cache = OptimizedCache.build(
    dataset,
    output_dir=JPEG_OUTPUT_DIR,
    num_workers=NUM_WORKERS,
    image_format="jpeg",
)
elapsed = time.time() - t0
print(f"\nDone: {elapsed:.1f}s ({len(dataset)/elapsed:.0f} samples/sec)")

# %%
# Cache summary
print(f"Samples: {jpeg_cache.num_samples:,}")
print(f"Fields: {list(jpeg_cache.field_types.keys())}")
print(f"\nFile sizes:")
for f in sorted(jpeg_cache.cache_dir.iterdir()):
    if f.is_file():
        print(f"  {f.name:30s} {f.stat().st_size/1e6:8.1f} MB")

# %% [markdown]
# ## Step 3: Build YUV420 cache (parallel)

# %%
print(f"Building YUV420 cache with {NUM_WORKERS} workers...")
t0 = time.time()
yuv_cache = OptimizedCache.build(
    dataset,
    output_dir=YUV_OUTPUT_DIR,
    num_workers=NUM_WORKERS,
    image_format="yuv420",
)
elapsed = time.time() - t0
print(f"\nDone: {elapsed:.1f}s ({len(dataset)/elapsed:.0f} samples/sec)")

# %%
# Cache summary
print(f"Samples: {yuv_cache.num_samples:,}")
print(f"Fields: {list(yuv_cache.field_types.keys())}")
print(f"\nFile sizes:")
for f in sorted(yuv_cache.cache_dir.iterdir()):
    if f.is_file():
        print(f"  {f.name:30s} {f.stat().st_size/1e6:8.1f} MB")

# %% [markdown]
# ## Step 4: Verify both caches

# %%
import numpy as np

indices = np.array([0, 100, 1000, 10000, 49999], dtype=np.int64)

jpeg_batch = jpeg_cache.load_batch(indices, fields=['image', 'label'])
yuv_batch = yuv_cache.load_batch(indices, fields=['image', 'label'])

print("JPEG cache spot-check:")
for i, idx in enumerate(indices):
    size = jpeg_batch['image']['sizes'][i]
    label = jpeg_batch['label']['data'][i]
    print(f"  [{idx:5d}] image={size:6d} bytes, label={label}")

print("\nYUV420 cache spot-check:")
for i, idx in enumerate(indices):
    size = yuv_batch['image']['sizes'][i]
    label = yuv_batch['label']['data'][i]
    print(f"  [{idx:5d}] image={size:6d} bytes, label={label}")

# Labels should match between formats
np.testing.assert_array_equal(
    jpeg_batch['label']['data'],
    yuv_batch['label']['data'],
)
print("\nLabels match across formats ✓")

# %% [markdown]
# ## Step 5: Build indexes

# %%
from slipstream.cache import write_index

# Build label index for both caches
for name, cache in [("JPEG", jpeg_cache), ("YUV420", yuv_cache)]:
    write_index(cache.cache_dir.parent, field="label")
    print(f"{name}: label index built")

# %% [markdown]
# ## Step 6: (Optional) Sync to remote S3 cache
#
# Uncomment to upload the caches to S3 for lab-wide sharing.

# %%
# from slipstream.s3_sync import upload_s3_cache
#
# REMOTE_CACHE = "s3://visionlab-datasets/slipstream-cache/imagenet1k/"
#
# upload_s3_cache(
#     JPEG_OUTPUT_DIR,
#     REMOTE_CACHE,
# )
# print(f"JPEG cache uploaded to {REMOTE_CACHE}")

# %% [markdown]
# ## Summary
#
# | Format | Samples | Size | Build Time |
# |--------|---------|------|------------|
# | JPEG   | 50,000  | ~3.8 GB | ~34s (13 workers) |
# | YUV420 | 50,000  | ~4.8 GB | ~40s (13 workers) |
#
# Both caches are ready for use with `SlipstreamLoader`:
#
# ```python
# from slipstream import SlipstreamDataset, SlipstreamLoader
# from slipstream.pipelines import supervised_val
#
# dataset = SlipstreamDataset(local_dir=str(JPEG_OUTPUT_DIR))
# loader = SlipstreamLoader(dataset, batch_size=256, pipelines=supervised_val(224))
# ```
