#!/usr/bin/env python3
"""Debug benchmark to isolate multi-crop slowdown.

Tests the decoder directly (no pipeline/tensor overhead) to determine
whether the slowdown is in the Numba decode+crop or in the pipeline's
hwc_to_chw + .clone() overhead.

Usage:
    uv run python benchmarks/benchmark_multicrop_debug.py
    uv run python benchmarks/benchmark_multicrop_debug.py --num-threads 12
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch


DEFAULT_DATASET = "s3://visionlab-datasets/imagenet1k/pre-processed/s256-l512-jpgbytes-q100-streaming/val/"


def main():
    parser = argparse.ArgumentParser(description="Debug multi-crop performance")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-threads", type=int, default=12)
    parser.add_argument("--num-iters", type=int, default=50)
    args = parser.parse_args()

    from slipstream import SlipstreamDataset
    from slipstream.cache import OptimizedCache
    from slipstream.decoders.numba_decoder import NumbaBatchDecoder

    print("Loading dataset...")
    ds = SlipstreamDataset(remote_dir=args.dataset, decode_images=False)
    cache = OptimizedCache.load(ds.cache_path)
    storage = cache.fields["image"]

    bs = args.batch_size
    N = args.num_iters
    indices = np.arange(bs, dtype=np.int64)
    batch = storage.load_batch(indices)
    data = batch["data"]
    sizes = batch["sizes"]
    heights = batch["heights"]
    widths = batch["widths"]

    decoder = NumbaBatchDecoder(num_threads=args.num_threads)
    print(f"Decoder: {decoder}")
    print(f"Batch size: {bs}, Iterations: {N}")
    print()

    # Warmup
    print("Warming up...")
    decoder.decode_batch_random_crop(data, sizes, heights, widths, target_size=224)
    decoder.decode_batch_multi_crop(data, sizes, heights, widths, num_crops=2, target_size=224)
    decoder.hwc_to_chw(
        decoder.decode_batch_random_crop(data, sizes, heights, widths, target_size=224)
    )
    print()

    # Test 1: Single RRC (decoder only)
    t0 = time.perf_counter()
    for _ in range(N):
        decoder.decode_batch_random_crop(data, sizes, heights, widths, target_size=224)
    single_time = (time.perf_counter() - t0) / N
    print(f"1. Single RRC (decoder only):     {bs / single_time:>10,.0f} img/s  ({single_time * 1000:.1f}ms/batch)")

    # Test 2: Multi-crop 2x (decoder only, no transpose/clone)
    t0 = time.perf_counter()
    for _ in range(N):
        decoder.decode_batch_multi_crop(data, sizes, heights, widths, num_crops=2, target_size=224)
    multi_time = (time.perf_counter() - t0) / N
    print(f"2. Multi-crop 2x (decoder only):  {bs / multi_time:>10,.0f} img/s  ({multi_time * 1000:.1f}ms/batch)")

    # Test 3: Single RRC + hwc_to_chw (no clone)
    t0 = time.perf_counter()
    for _ in range(N):
        hwc = decoder.decode_batch_random_crop(data, sizes, heights, widths, target_size=224)
        decoder.hwc_to_chw(hwc)
    transpose_time = (time.perf_counter() - t0) / N
    print(f"3. Single RRC + hwc_to_chw:       {bs / transpose_time:>10,.0f} img/s  ({transpose_time * 1000:.1f}ms/batch)")

    # Test 4: Single RRC + hwc_to_chw + torch tensor (no clone)
    t0 = time.perf_counter()
    for _ in range(N):
        hwc = decoder.decode_batch_random_crop(data, sizes, heights, widths, target_size=224)
        chw = decoder.hwc_to_chw(hwc)
        torch.from_numpy(chw)
    tensor_time = (time.perf_counter() - t0) / N
    print(f"4. Single RRC + transpose + tensor:{bs / tensor_time:>9,.0f} img/s  ({tensor_time * 1000:.1f}ms/batch)")

    # Test 5: Full multi-crop pipeline (decode + transpose + clone, matches MultiCropRandomResizedCrop)
    t0 = time.perf_counter()
    for _ in range(N):
        crops = decoder.decode_batch_multi_crop(data, sizes, heights, widths, num_crops=2, target_size=224)
        results = []
        for crop in crops:
            chw = decoder.hwc_to_chw(crop)
            results.append(torch.from_numpy(chw).clone())
    full_time = (time.perf_counter() - t0) / N
    print(f"5. Full multi-crop pipeline:      {bs / full_time:>10,.0f} img/s  ({full_time * 1000:.1f}ms/batch)")

    # Test 6: Multi-crop decode + transpose only (no clone)
    t0 = time.perf_counter()
    for _ in range(N):
        crops = decoder.decode_batch_multi_crop(data, sizes, heights, widths, num_crops=2, target_size=224)
        results = []
        for crop in crops:
            chw = decoder.hwc_to_chw(crop)
            results.append(torch.from_numpy(chw))
    no_clone_time = (time.perf_counter() - t0) / N
    print(f"6. Multi-crop + transpose (no clone):{bs / no_clone_time:>7,.0f} img/s  ({no_clone_time * 1000:.1f}ms/batch)")

    # Test 7: multi_hwc_to_chw (separate buffers, no clone needed)
    t0 = time.perf_counter()
    for _ in range(N):
        crops = decoder.decode_batch_multi_crop(data, sizes, heights, widths, num_crops=2, target_size=224)
        chw_crops = decoder.multi_hwc_to_chw(crops)
        results = [torch.from_numpy(chw) for chw in chw_crops]
    multi_chw_time = (time.perf_counter() - t0) / N
    print(f"7. multi_hwc_to_chw (new, no clone): {bs / multi_chw_time:>10,.0f} img/s  ({multi_chw_time * 1000:.1f}ms/batch)")

    print()
    print("Analysis:")
    print(f"  Decode overhead (multi vs single):  {multi_time / single_time:.2f}x")
    print(f"  Transpose overhead:                 +{(transpose_time - single_time) * 1000:.1f}ms")
    print(f"  Clone overhead (test 5 vs 6):       +{(full_time - no_clone_time) * 1000:.1f}ms")
    print(f"  Full pipeline vs decoder-only:      {full_time / multi_time:.2f}x")
    print(f"  New pipeline (7) vs old (5):        {full_time / multi_chw_time:.2f}x speedup")


if __name__ == "__main__":
    main()
