#!/usr/bin/env python3
"""Benchmark raw I/O performance.

Compares:
- OptimizedCache.load_batch (direct memory-mapped access)
- SlipstreamLoader with no pipelines (raw bytes)
- StreamingDataLoader (LitData baseline)

Usage:
    uv run python benchmarks/benchmark_raw_io.py
    uv run python benchmarks/benchmark_raw_io.py --batch-size 512 --epochs 5
    uv run python benchmarks/benchmark_raw_io.py --output results/raw_io.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from benchmarks.utils import (
    BenchmarkResult,
    format_results_table,
    get_machine_info,
    run_benchmark,
    save_results,
)


# Default dataset path
DEFAULT_DATASET = "s3://visionlab-datasets/imagenet1k/pre-processed/s256-l512-jpgbytes-q100-streaming/val/"


def benchmark_cache_direct(
    cache,
    batch_size: int,
    num_epochs: int,
    num_warmup: int,
    field: str = "image",
) -> BenchmarkResult:
    """Benchmark OptimizedCache.load_batch directly."""
    num_samples = len(cache)
    num_batches = (num_samples + batch_size - 1) // batch_size
    indices = np.arange(num_samples, dtype=np.int64)

    def run_epoch():
        total_samples = 0
        for i in tqdm(range(num_batches), leave=False):
            batch_start = i * batch_size
            batch_end = min(batch_start + batch_size, num_samples)
            batch_indices = indices[batch_start:batch_end]
            batch_data = cache.load_batch(batch_indices, fields=[field])
            total_samples += len(batch_indices)
        return total_samples

    # Warmup
    print(f"\nOptimizedCache.load_batch ({field}):")
    print(f"  Warmup ({num_warmup} epoch(s)):")
    warmup_results = []
    for i in range(num_warmup):
        import time
        start = time.perf_counter()
        total = run_epoch()
        elapsed = time.perf_counter() - start
        rate = total / elapsed
        warmup_results.append({"samples_per_sec": rate, "elapsed_sec": elapsed, "total_samples": total})
        print(f"    Warmup {i + 1}: {rate:,.0f} samples/sec ({elapsed:.2f}s)")

    # Timed epochs
    epoch_results = []
    for epoch in range(num_epochs):
        import time
        start = time.perf_counter()
        total = run_epoch()
        elapsed = time.perf_counter() - start
        rate = total / elapsed
        epoch_results.append({"samples_per_sec": rate, "elapsed_sec": elapsed, "total_samples": total})
        print(f"  Epoch {epoch + 1}: {rate:,.0f} samples/sec ({elapsed:.2f}s)")

    avg_rate = np.mean([r["samples_per_sec"] for r in epoch_results])
    print(f"  Average: {avg_rate:,.0f} samples/sec")

    return BenchmarkResult(
        name=f"OptimizedCache.load_batch ({field})",
        samples_per_sec=avg_rate,
        total_samples=epoch_results[0]["total_samples"],
        elapsed_sec=sum(r["elapsed_sec"] for r in epoch_results),
        num_epochs=num_epochs,
        warmup_epochs=num_warmup,
        per_epoch_results=warmup_results + epoch_results,
        metadata={"field": field, "batch_size": batch_size},
    )


def benchmark_slipstream_raw(
    dataset,
    batch_size: int,
    num_epochs: int,
    num_warmup: int,
) -> BenchmarkResult:
    """Benchmark SlipstreamLoader with no pipelines (raw bytes)."""
    from slipstream import SlipstreamLoader

    loader = SlipstreamLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        # No pipelines = raw data
    )

    def count_samples(batch):
        img_data = batch.get("image")
        if isinstance(img_data, dict):
            return len(img_data["data"])
        return batch_size

    result = run_benchmark(
        name="SlipstreamLoader (raw, no pipelines)",
        iterator_fn=lambda: loader,
        count_fn=count_samples,
        num_epochs=num_epochs,
        num_warmup=num_warmup,
        metadata={"batch_size": batch_size},
    )

    loader.shutdown()
    return result


def benchmark_streaming_dataloader(
    dataset,
    batch_size: int,
    num_workers: int,
    num_epochs: int,
    num_warmup: int,
) -> BenchmarkResult:
    """Benchmark StreamingDataLoader (LitData baseline)."""
    from litdata import StreamingDataLoader

    from slipstream import list_collate_fn

    loader = StreamingDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        collate_fn=list_collate_fn,
    )

    def count_samples(batch):
        if isinstance(batch, dict):
            img_data = batch.get("image")
            if isinstance(img_data, list):
                return len(img_data)
        return batch_size

    return run_benchmark(
        name=f"StreamingDataLoader ({num_workers} workers)",
        iterator_fn=lambda: loader,
        count_fn=count_samples,
        num_epochs=num_epochs,
        num_warmup=num_warmup,
        metadata={"batch_size": batch_size, "num_workers": num_workers},
    )


def main():
    parser = argparse.ArgumentParser(description="Benchmark raw I/O performance")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET, help="Dataset path (S3 or local)")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of timed epochs")
    parser.add_argument("--warmup", type=int, default=1, help="Number of warmup epochs")
    parser.add_argument("--num-workers", type=int, default=8, help="Workers for StreamingDataLoader")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    # Print machine info
    machine_info = get_machine_info()
    print(machine_info)

    # Load dataset
    print("\nLoading dataset...")
    from slipstream import SlipstreamDataset
    from slipstream.cache import OptimizedCache

    dataset = SlipstreamDataset(
        remote_dir=args.dataset,
        decode_images=False,
    )
    print(f"Dataset: {len(dataset):,} samples")

    # Build/load optimized cache
    print("\nBuilding/loading optimized cache...")
    cache_path = dataset.cache_path
    if OptimizedCache.exists(cache_path):
        cache = OptimizedCache.load(cache_path)
    else:
        cache = OptimizedCache.build(dataset, cache_path)
    print(f"Cache: {len(cache):,} samples, fields: {list(cache.fields.keys())}")

    results = []

    # 1. OptimizedCache direct (image field)
    result = benchmark_cache_direct(
        cache, args.batch_size, args.epochs, args.warmup, field="image"
    )
    results.append(result)

    # 2. OptimizedCache direct (label field)
    result = benchmark_cache_direct(
        cache, args.batch_size, args.epochs, args.warmup, field="label"
    )
    results.append(result)

    # 3. SlipstreamLoader raw
    result = benchmark_slipstream_raw(
        dataset, args.batch_size, args.epochs, args.warmup
    )
    results.append(result)

    # 4. StreamingDataLoader baseline
    result = benchmark_streaming_dataloader(
        dataset, args.batch_size, args.num_workers, args.epochs, args.warmup
    )
    results.append(result)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(format_results_table(results))

    # Calculate speedup
    slipstream_rate = results[2].samples_per_sec
    streaming_rate = results[3].samples_per_sec
    if streaming_rate > 0:
        speedup = slipstream_rate / streaming_rate
        print(f"\nSlipstreamLoader is {speedup:.1f}x faster than StreamingDataLoader")

    # Save results
    if args.output:
        save_results(results, machine_info, args.output, "raw_io")
    else:
        # Default output path
        hostname = machine_info.hostname.replace(".", "_")
        output_path = Path(__file__).parent / "results" / f"raw_io_{hostname}.json"
        save_results(results, machine_info, output_path, "raw_io")


if __name__ == "__main__":
    main()
