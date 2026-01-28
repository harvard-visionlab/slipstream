#!/usr/bin/env python3
"""Benchmark SlipstreamLoader full pipeline performance.

Tests complete training/validation pipelines with:
- Async prefetching
- Decode + crop transforms
- Normalization
- Device placement

Usage:
    uv run python benchmarks/benchmark_loader.py
    uv run python benchmarks/benchmark_loader.py --batch-size 512 --epochs 5
    uv run python benchmarks/benchmark_loader.py --device cuda --output results/loader.json
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from benchmarks.utils import (
    BenchmarkResult,
    format_results_table,
    get_drive_info,
    get_machine_info,
    save_results,
)


# Default dataset path
DEFAULT_DATASET = "s3://visionlab-datasets/imagenet1k/pre-processed/s256-l512-jpgbytes-q100-streaming/val/"


def benchmark_loader(
    dataset,
    batch_size: int,
    num_epochs: int,
    num_warmup: int,
    pipeline_type: str,  # "train", "val", "raw"
    device: str = "cpu",
    num_workers: int = 8,
    target_size: int = 224,
) -> BenchmarkResult:
    """Benchmark SlipstreamLoader with specified pipeline."""
    from slipstream import (
        SlipstreamLoader,
        RandomResizedCrop,
        CenterCrop,
        Normalize,
    )

    # Build pipeline based on type
    if pipeline_type == "train":
        pipelines = {
            "image": [
                RandomResizedCrop(target_size, scale=(0.08, 1.0), device=device, num_workers=num_workers),
                Normalize(),
            ],
        }
        name = f"SlipstreamLoader (train, RRC {target_size}, {device})"
    elif pipeline_type == "val":
        pipelines = {
            "image": [
                CenterCrop(target_size, device=device, num_workers=num_workers),
                Normalize(),
            ],
        }
        name = f"SlipstreamLoader (val, CenterCrop {target_size}, {device})"
    else:  # raw
        pipelines = None
        name = "SlipstreamLoader (raw, no pipeline)"

    loader = SlipstreamLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pipelines=pipelines,
        exclude_fields=["path"],
    )

    def run_epoch():
        total_samples = 0
        for batch in tqdm(loader, leave=False):
            if pipeline_type == "raw":
                # Raw mode - count from image data dict
                img_data = batch.get("image")
                if isinstance(img_data, dict):
                    total_samples += len(img_data["data"])
                else:
                    total_samples += batch_size
            else:
                # Pipeline mode - count from tensor shape
                total_samples += batch["image"].shape[0]
                # Ensure tensor is materialized
                _ = batch["image"].sum()
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
            "pipeline_type": pipeline_type,
            "device": device,
            "target_size": target_size,
            "batch_size": batch_size,
            "num_workers": num_workers,
        },
    )


def main():
    parser = argparse.ArgumentParser(description="Benchmark SlipstreamLoader full pipeline")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET, help="Dataset path (S3 or local)")
    parser.add_argument("--cache-dir", type=str, default=None, help="Override cache directory (e.g., /path/to/fast/nvme)")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of timed epochs")
    parser.add_argument("--warmup", type=int, default=1, help="Number of warmup epochs")
    parser.add_argument("--num-workers", type=int, default=8, help="CPU decoder workers")
    parser.add_argument("--target-size", type=int, default=224, help="Target crop size")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    parser.add_argument("--machine-name", type=str, default=None, help="Machine name for results (e.g., 'nolan-25')")
    parser.add_argument("--save", action="store_true", help="Save results to JSON file")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path (implies --save)")
    args = parser.parse_args()

    # Print machine info
    machine_info = get_machine_info(args.machine_name)
    print(machine_info)

    # Validate device
    device = args.device
    if device.startswith("cuda") and not machine_info.cuda_available:
        print(f"\nWarning: CUDA requested but not available, falling back to CPU")
        device = "cpu"

    # Load dataset
    print("\nLoading dataset...")
    from slipstream import SlipstreamDataset

    dataset = SlipstreamDataset(
        remote_dir=args.dataset,
        cache_dir=args.cache_dir,
        decode_images=False,
    )
    print(f"Dataset: {len(dataset):,} samples")

    # Report cache drive info
    cache_path = dataset.cache_path
    print(f"Cache path: {cache_path}")
    cache_drive = get_drive_info(cache_path)
    print(f"Cache drive: {cache_drive['type']} (device: {cache_drive['device']})")

    results = []

    # 1. Raw I/O (no pipeline)
    result = benchmark_loader(
        dataset, args.batch_size, args.epochs, args.warmup,
        pipeline_type="raw", device=device, num_workers=args.num_workers,
        target_size=args.target_size,
    )
    results.append(result)

    # 2. Training pipeline (RandomResizedCrop)
    result = benchmark_loader(
        dataset, args.batch_size, args.epochs, args.warmup,
        pipeline_type="train", device=device, num_workers=args.num_workers,
        target_size=args.target_size,
    )
    results.append(result)

    # 3. Validation pipeline (CenterCrop)
    result = benchmark_loader(
        dataset, args.batch_size, args.epochs, args.warmup,
        pipeline_type="val", device=device, num_workers=args.num_workers,
        target_size=args.target_size,
    )
    results.append(result)

    # If CPU was used but CUDA is available, also test CUDA
    if device == "cpu" and machine_info.cuda_available:
        print("\n--- Also benchmarking with CUDA ---")

        result = benchmark_loader(
            dataset, args.batch_size, args.epochs, args.warmup,
            pipeline_type="train", device="cuda", num_workers=args.num_workers,
            target_size=args.target_size,
        )
        results.append(result)

        result = benchmark_loader(
            dataset, args.batch_size, args.epochs, args.warmup,
            pipeline_type="val", device="cuda", num_workers=args.num_workers,
            target_size=args.target_size,
        )
        results.append(result)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(format_results_table(results))

    # Reference targets
    print("\nReference targets (from litdata-mmap on Linux):")
    print("  Raw I/O: ~480k+ samples/sec")
    print("  CPU Decode + RRC: ~5.7k samples/sec")
    print("  GPU Decode + RRC: ~10-11k samples/sec")

    # Save results (only if --save or --output specified)
    if args.output:
        save_results(results, machine_info, args.output, "loader")
    elif args.save:
        name = machine_info.machine_name.replace(".", "_").replace(" ", "_")
        output_path = Path(__file__).parent / "results" / f"loader_{name}.json"
        save_results(results, machine_info, output_path, "loader")


if __name__ == "__main__":
    main()
