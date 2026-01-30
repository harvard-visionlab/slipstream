#!/usr/bin/env python3
"""Benchmark SlipstreamLoader fed from an FFCV .beton file.

Tests the full FFCV reader → OptimizedCache → SlipstreamLoader pipeline
with RandomResizedCrop, verifying performance matches the LitData path.

Usage:
    uv run python benchmarks/benchmark_ffcv_loader.py
    uv run python benchmarks/benchmark_ffcv_loader.py --batch-size 256 --epochs 3
    uv run python benchmarks/benchmark_ffcv_loader.py --ffcv-path /local/path/to/file.ffcv
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from benchmarks.utils import (
    BenchmarkResult,
    format_results_table,
    get_drive_info,
    get_machine_info,
    save_results,
)


# Default FFCV dataset path
DEFAULT_FFCV_PATH = "s3://visionlab-datasets/imagenet1k/pre-processed/s256-l512-jpgbytes-q100-ffcv/imagenet1k-s256-l512-jpg-q100-cs100-val-7ac6386e.ffcv"


def benchmark_ffcv_loader(
    reader,
    batch_size: int,
    num_epochs: int,
    num_warmup: int,
    num_threads: int = 0,
    target_size: int = 224,
    use_threading: bool = True,
) -> BenchmarkResult:
    """Benchmark SlipstreamLoader with FFCV reader + RRC pipeline."""
    from slipstream import SlipstreamLoader
    from slipstream.pipelines import RandomResizedCrop

    mode = "threaded" if use_threading else "simple"
    name = f"FFCV → SlipstreamLoader (RRC, {mode})"

    pipelines = {
        "image": [
            RandomResizedCrop(target_size, num_threads=num_threads),
        ],
    }

    loader = SlipstreamLoader(
        reader,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pipelines=pipelines,
        exclude_fields=["path"],
        use_threading=use_threading,
    )

    def run_epoch():
        total_samples = 0
        for batch in tqdm(loader, leave=False):
            img = batch["image"]
            total_samples += img.shape[0]
        return total_samples

    # Warmup
    print(f"\n{name}:")
    print(f"  Warmup ({num_warmup} epoch(s)):")
    warmup_results = []
    for i in range(num_warmup):
        start = time.perf_counter()
        total = run_epoch()
        elapsed = time.perf_counter() - start
        rate = total / elapsed
        warmup_results.append({"samples_per_sec": rate, "elapsed_sec": elapsed, "total_samples": total})
        print(f"    Warmup {i + 1}: {rate:,.0f} samples/sec ({elapsed:.2f}s)")

    # Timed epochs
    epoch_results = []
    for epoch in range(num_epochs):
        start = time.perf_counter()
        total = run_epoch()
        elapsed = time.perf_counter() - start
        rate = total / elapsed
        epoch_results.append({"samples_per_sec": rate, "elapsed_sec": elapsed, "total_samples": total})
        print(f"  Epoch {epoch + 1}: {rate:,.0f} samples/sec ({elapsed:.2f}s)")

    avg_rate = np.mean([r["samples_per_sec"] for r in epoch_results])
    print(f"  Average: {avg_rate:,.0f} samples/sec")

    loader.shutdown()

    return BenchmarkResult(
        name=name,
        samples_per_sec=avg_rate,
        total_samples=epoch_results[0]["total_samples"],
        elapsed_sec=sum(r["elapsed_sec"] for r in epoch_results),
        num_epochs=num_epochs,
        warmup_epochs=num_warmup,
        per_epoch_results=warmup_results + epoch_results,
        metadata={
            "pipeline_type": "train",
            "target_size": target_size,
            "batch_size": batch_size,
            "num_threads": num_threads,
            "use_threading": use_threading,
            "source": "ffcv",
        },
    )


def main():
    parser = argparse.ArgumentParser(description="Benchmark SlipstreamLoader with FFCV reader")
    parser.add_argument("--ffcv-path", type=str, default=DEFAULT_FFCV_PATH, help="FFCV file path (S3 or local)")
    parser.add_argument("--cache-dir", type=str, default=None, help="Override cache directory")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of timed epochs")
    parser.add_argument("--warmup", type=int, default=1, help="Number of warmup epochs")
    parser.add_argument("--num-threads", type=int, default=0, help="NumbaBatchDecoder threads (0=auto)")
    parser.add_argument("--target-size", type=int, default=224, help="Target crop size")
    parser.add_argument("--machine-name", type=str, default=None, help="Machine name for results")
    parser.add_argument("--save", action="store_true", help="Save results to JSON file")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path (implies --save)")
    args = parser.parse_args()

    # Print machine info
    machine_info = get_machine_info(args.machine_name)
    print(machine_info)

    # Load FFCV reader
    print("\nLoading FFCV file...")
    from slipstream.readers import FFCVFileReader

    reader = FFCVFileReader(
        args.ffcv_path,
        cache_dir=args.cache_dir,
    )
    print(f"Dataset: {len(reader):,} samples")
    print(f"Fields: {reader.field_types}")

    # Report cache drive info
    cache_path = reader.cache_path
    print(f"Cache path: {cache_path}")
    cache_drive = get_drive_info(cache_path)
    print(f"Cache drive: {cache_drive['type']} (device: {cache_drive['device']})")

    results = []

    # Benchmark RRC with simple mode
    result = benchmark_ffcv_loader(
        reader, args.batch_size, args.epochs, args.warmup,
        num_threads=args.num_threads,
        target_size=args.target_size,
        use_threading=False,
    )
    results.append(result)

    # Benchmark RRC with threaded mode
    result = benchmark_ffcv_loader(
        reader, args.batch_size, args.epochs, args.warmup,
        num_threads=args.num_threads,
        target_size=args.target_size,
        use_threading=True,
    )
    results.append(result)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(format_results_table(results))

    print("\nReference targets (LitData path, RRC):")
    print("  RRC (simple):   ~12,947 samples/sec")
    print("  RRC (threaded): ~13,498 samples/sec")

    # Save results
    if args.output:
        save_results(results, machine_info, args.output, "ffcv_loader")
    elif args.save:
        name = machine_info.machine_name.replace(".", "_").replace(" ", "_")
        output_path = Path(__file__).parent / "results" / f"ffcv_loader_{name}.json"
        save_results(results, machine_info, output_path, "ffcv_loader")


if __name__ == "__main__":
    main()
