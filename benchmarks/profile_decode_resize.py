#!/usr/bin/env python3
"""Profile decode vs resize time breakdown.

Runs the profiled decode+crop path and reports how much time is spent
in JPEG decoding vs stb/OpenCV resize.

Usage:
    uv run python benchmarks/profile_decode_resize.py
"""

import time
import numpy as np
from slipstream import SlipstreamDataset
from slipstream.cache import OptimizedCache
from slipstream.decoders.numba_decoder import NumbaBatchDecoder

DATASET_PATH = "s3://visionlab-datasets/imagenet1k/pre-processed/s256-l512-jpgbytes-q100-streaming/val/"
BATCH_SIZE = 256
NUM_BATCHES = 50  # ~12,800 images

def main():
    print("Loading dataset...")
    dataset = SlipstreamDataset(
        remote_dir=DATASET_PATH,
        decode_images=False,
    )
    print(f"Dataset: {len(dataset):,} samples")

    cache_path = dataset.cache_path
    print(f"Cache path: {cache_path}")

    print("Building/loading optimized cache...")
    if OptimizedCache.exists(cache_path):
        cache = OptimizedCache.load(cache_path)
    else:
        cache = OptimizedCache.build(dataset, cache_path)
    print(f"Cache: {len(cache):,} samples\n")

    # Create decoder in profiled mode
    decoder = NumbaBatchDecoder(crop_mode="profiled")
    print(f"Decoder: {decoder}")
    print(f"OpenCV available: {decoder.has_opencv()}\n")

    num_samples = len(cache)
    indices = np.arange(num_samples, dtype=np.int64)

    # Warmup (1 batch for JIT)
    print("Warming up JIT...")
    batch_data = cache.load_batch(indices[:BATCH_SIZE], fields=["image"])
    decoder.decode_batch_center_crop(
        batch_data["image"]["data"],
        batch_data["image"]["sizes"],
        batch_data["image"]["heights"],
        batch_data["image"]["widths"],
        crop_size=224,
    )

    # Reset counters after warmup
    decoder.reset_profile_stats()

    # Profile CenterCrop
    print("Profiling CenterCrop (224x224)...")
    start = time.perf_counter()
    total_samples = 0
    for i in range(NUM_BATCHES):
        batch_start = i * BATCH_SIZE
        batch_end = min(batch_start + BATCH_SIZE, num_samples)
        batch_indices = indices[batch_start:batch_end]

        batch_data = cache.load_batch(batch_indices, fields=["image"])
        decoder.decode_batch_center_crop(
            batch_data["image"]["data"],
            batch_data["image"]["sizes"],
            batch_data["image"]["heights"],
            batch_data["image"]["widths"],
            crop_size=224,
        )
        total_samples += len(batch_indices)
    elapsed = time.perf_counter() - start

    stats = decoder.get_profile_stats()
    print(f"\n{'='*60}")
    print(f"PROFILING RESULTS - CenterCrop")
    print(f"{'='*60}")
    print(f"Total samples:  {total_samples:,}")
    print(f"Total time:     {elapsed:.2f}s ({total_samples/elapsed:,.0f} samples/sec)")
    print(f"")
    print(f"Decode time:    {stats['decode_ms']:.0f}ms total ({stats['decode_ms']/stats['decode_count']:.2f}ms/image)")
    print(f"Resize time:    {stats['resize_ms']:.0f}ms total ({stats['resize_ms']/stats['resize_count']:.2f}ms/image)")
    print(f"Total C++ time: {stats['decode_ms'] + stats['resize_ms']:.0f}ms")
    print(f"")
    decode_pct = stats['decode_ms'] / (stats['decode_ms'] + stats['resize_ms']) * 100
    resize_pct = stats['resize_ms'] / (stats['decode_ms'] + stats['resize_ms']) * 100
    print(f"Decode:  {decode_pct:.1f}% of C++ time")
    print(f"Resize:  {resize_pct:.1f}% of C++ time")
    print(f"Count:   {stats['decode_count']:,} decodes, {stats['resize_count']:,} resizes")

    # Reset and profile RRC
    decoder.reset_profile_stats()
    print(f"\nProfiling RandomResizedCrop (224x224)...")
    start = time.perf_counter()
    total_samples = 0
    for i in range(NUM_BATCHES):
        batch_start = i * BATCH_SIZE
        batch_end = min(batch_start + BATCH_SIZE, num_samples)
        batch_indices = indices[batch_start:batch_end]

        batch_data = cache.load_batch(batch_indices, fields=["image"])
        decoder.decode_batch_random_crop(
            batch_data["image"]["data"],
            batch_data["image"]["sizes"],
            batch_data["image"]["heights"],
            batch_data["image"]["widths"],
            target_size=224,
        )
        total_samples += len(batch_indices)
    elapsed = time.perf_counter() - start

    stats = decoder.get_profile_stats()
    print(f"\n{'='*60}")
    print(f"PROFILING RESULTS - RandomResizedCrop")
    print(f"{'='*60}")
    print(f"Total samples:  {total_samples:,}")
    print(f"Total time:     {elapsed:.2f}s ({total_samples/elapsed:,.0f} samples/sec)")
    print(f"")
    print(f"Decode time:    {stats['decode_ms']:.0f}ms total ({stats['decode_ms']/stats['decode_count']:.2f}ms/image)")
    print(f"Resize time:    {stats['resize_ms']:.0f}ms total ({stats['resize_ms']/stats['resize_count']:.2f}ms/image)")
    print(f"Total C++ time: {stats['decode_ms'] + stats['resize_ms']:.0f}ms")
    print(f"")
    decode_pct = stats['decode_ms'] / (stats['decode_ms'] + stats['resize_ms']) * 100
    resize_pct = stats['resize_ms'] / (stats['decode_ms'] + stats['resize_ms']) * 100
    print(f"Decode:  {decode_pct:.1f}% of C++ time")
    print(f"Resize:  {resize_pct:.1f}% of C++ time")
    print(f"Count:   {stats['decode_count']:,} decodes, {stats['resize_count']:,} resizes")


if __name__ == "__main__":
    main()
