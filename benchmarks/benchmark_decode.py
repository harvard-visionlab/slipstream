#!/usr/bin/env python3
"""Benchmark decode performance.

Compares:
- CPU decode (TurboJPEG) - no crop
- CPU decode + RandomResizedCrop
- CPU decode + CenterCrop
- GPU decode (if available)

Usage:
    uv run python benchmarks/benchmark_decode.py
    uv run python benchmarks/benchmark_decode.py --batch-size 512 --epochs 5
    uv run python benchmarks/benchmark_decode.py --output results/decode.json
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


def benchmark_decode(
    cache,
    decoder,
    batch_size: int,
    num_epochs: int,
    num_warmup: int,
    with_crop: str | None = None,  # None, "rrc", "center"
    target_size: int = 224,
) -> BenchmarkResult:
    """Benchmark decode throughput."""
    num_samples = len(cache)
    num_batches = (num_samples + batch_size - 1) // batch_size
    indices = np.arange(num_samples, dtype=np.int64)

    crop_str = ""
    if with_crop == "rrc":
        crop_str = " + RandomResizedCrop"
    elif with_crop == "center":
        crop_str = " + CenterCrop"

    name = f"{type(decoder).__name__}{crop_str}"

    def run_epoch():
        total_samples = 0
        for i in tqdm(range(num_batches), leave=False):
            batch_start = i * batch_size
            batch_end = min(batch_start + batch_size, num_samples)
            batch_indices = indices[batch_start:batch_end]

            # Load raw data
            batch_data = cache.load_batch(batch_indices, fields=["image"])
            data = batch_data["image"]["data"]
            sizes = batch_data["image"]["sizes"]
            heights = batch_data["image"]["heights"]
            widths = batch_data["image"]["widths"]

            # Decode
            if with_crop == "rrc":
                images = decoder.decode_batch_random_crop(
                    data, sizes, heights, widths,
                    target_size=target_size,
                    scale=(0.08, 1.0),
                )
            elif with_crop == "center":
                images = decoder.decode_batch_center_crop(
                    data, sizes, heights, widths,
                    crop_size=target_size,
                )
            else:
                images = decoder.decode_batch(data, sizes)

            total_samples += len(batch_indices)
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

    return BenchmarkResult(
        name=name,
        samples_per_sec=avg_rate,
        total_samples=epoch_results[0]["total_samples"],
        elapsed_sec=sum(r["elapsed_sec"] for r in epoch_results),
        num_epochs=num_epochs,
        warmup_epochs=num_warmup,
        per_epoch_results=warmup_results + epoch_results,
        metadata={
            "decoder": type(decoder).__name__,
            "with_crop": with_crop,
            "target_size": target_size,
            "batch_size": batch_size,
        },
    )


def main():
    parser = argparse.ArgumentParser(description="Benchmark decode performance")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET, help="Dataset path (S3 or local)")
    parser.add_argument("--cache-dir", type=str, default=None, help="Override cache directory (e.g., /path/to/fast/nvme)")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of timed epochs")
    parser.add_argument("--warmup", type=int, default=1, help="Number of warmup epochs")
    parser.add_argument("--num-workers", type=int, default=8, help="CPU decoder workers")
    parser.add_argument("--target-size", type=int, default=224, help="Target crop size")
    parser.add_argument("--machine-name", type=str, default=None, help="Machine name for results (e.g., 'nolan-25')")
    parser.add_argument("--save", action="store_true", help="Save results to JSON file")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path (implies --save)")
    parser.add_argument("--skip-gpu", action="store_true", help="Skip GPU benchmarks")
    args = parser.parse_args()

    # Print machine info
    machine_info = get_machine_info(args.machine_name)
    print(machine_info)

    # Load dataset
    print("\nLoading dataset...")
    from slipstream import SlipstreamDataset
    from slipstream.cache import OptimizedCache

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

    # Build/load optimized cache
    print("\nBuilding/loading optimized cache...")
    if OptimizedCache.exists(cache_path):
        cache = OptimizedCache.load(cache_path)
    else:
        cache = OptimizedCache.build(dataset, cache_path)
    print(f"Cache: {len(cache):,} samples")

    results = []

    # CPU decoder benchmarks
    from slipstream.decoders import CPUDecoder

    cpu_decoder = CPUDecoder(num_workers=args.num_workers)
    print(f"\nCPU Decoder: {cpu_decoder}")

    # 1. CPU decode only
    result = benchmark_decode(
        cache, cpu_decoder, args.batch_size, args.epochs, args.warmup,
        with_crop=None,
    )
    results.append(result)

    # 2. CPU decode + RandomResizedCrop
    result = benchmark_decode(
        cache, cpu_decoder, args.batch_size, args.epochs, args.warmup,
        with_crop="rrc", target_size=args.target_size,
    )
    results.append(result)

    # 3. CPU decode + CenterCrop
    result = benchmark_decode(
        cache, cpu_decoder, args.batch_size, args.epochs, args.warmup,
        with_crop="center", target_size=args.target_size,
    )
    results.append(result)

    cpu_decoder.shutdown()

    # GPU decoder benchmarks (if available)
    if not args.skip_gpu and machine_info.cuda_available:
        from slipstream.decoders import check_gpu_decoder_available, GPUDecoder, GPUDecoderFallback

        if check_gpu_decoder_available():
            gpu_decoder = GPUDecoder(device=0, max_batch_size=args.batch_size)
            print(f"\nGPU Decoder: {gpu_decoder}")

            # 4. GPU decode only
            result = benchmark_decode(
                cache, gpu_decoder, args.batch_size, args.epochs, args.warmup,
                with_crop=None,
            )
            results.append(result)

            # 5. GPU decode + RandomResizedCrop
            result = benchmark_decode(
                cache, gpu_decoder, args.batch_size, args.epochs, args.warmup,
                with_crop="rrc", target_size=args.target_size,
            )
            results.append(result)

            # 6. GPU decode + CenterCrop
            result = benchmark_decode(
                cache, gpu_decoder, args.batch_size, args.epochs, args.warmup,
                with_crop="center", target_size=args.target_size,
            )
            results.append(result)

            gpu_decoder.shutdown()
        else:
            print("\nnvImageCodec not available, using fallback GPU decoder...")
            if torch.cuda.is_available():
                gpu_fallback = GPUDecoderFallback(device=0, num_workers=args.num_workers)
                print(f"GPU Fallback Decoder: {gpu_fallback}")

                # 4. GPU fallback decode + RRC
                result = benchmark_decode(
                    cache, gpu_fallback, args.batch_size, args.epochs, args.warmup,
                    with_crop="rrc", target_size=args.target_size,
                )
                results.append(result)

                gpu_fallback.shutdown()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(format_results_table(results))

    # Reference targets
    print("\nReference targets (from litdata-mmap on Linux):")
    print("  CPU Decode + RRC: ~5.7k samples/sec")
    print("  GPU Decode + RRC: ~10-11k samples/sec")

    # Save results (only if --save or --output specified)
    if args.output:
        save_results(results, machine_info, args.output, "decode")
    elif args.save:
        name = machine_info.machine_name.replace(".", "_").replace(" ", "_")
        output_path = Path(__file__).parent / "results" / f"decode_{name}.json"
        save_results(results, machine_info, output_path, "decode")


if __name__ == "__main__":
    main()
