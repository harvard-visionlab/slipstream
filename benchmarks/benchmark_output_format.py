#!/usr/bin/env python3
"""Benchmark decoder output format performance.

Tests different decoder output configurations to find the fastest path
from JPEG bytes to GPU-ready float tensors:

1. Baseline (to_tensor=True, permute=True): decoder -> CHW torch tensor
2. Numpy CHW (to_tensor=False, permute=True): decoder -> CHW numpy -> ToTorchImage
3. Numpy HWC (to_tensor=False, permute=False): decoder -> HWC numpy -> ToTorchImage

Each configuration is tested with:
- CPU-only pipeline (no ToTorchImage, just decode)
- GPU pipeline (decode -> ToTorchImage(cuda))

Usage:
    uv run python benchmarks/benchmark_output_format.py
    uv run python benchmarks/benchmark_output_format.py --device cuda
    uv run python benchmarks/benchmark_output_format.py --device cpu
    uv run python benchmarks/benchmark_output_format.py --image-format all
    uv run python benchmarks/benchmark_output_format.py --output results/output_format.json
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


def benchmark_decode_only(
    dataset,
    batch_size: int,
    num_epochs: int,
    num_warmup: int,
    to_tensor: bool,
    permute: bool,
    num_threads: int = 0,
    target_size: int = 224,
    image_format: str = "jpeg",
) -> BenchmarkResult:
    """Benchmark decoder only (no ToTorchImage)."""
    from slipstream import SlipstreamLoader
    from slipstream.decoders import DecodeRandomResizedCrop

    name_parts = ["Decode RRC"]
    if to_tensor:
        name_parts.append("tensor")
    else:
        name_parts.append("numpy")
    if permute:
        name_parts.append("CHW")
    else:
        name_parts.append("HWC")
    if image_format != "jpeg":
        name_parts.append(image_format)
    name = " ".join(name_parts)

    pipelines = {
        "image": [
            DecodeRandomResizedCrop(
                target_size,
                num_threads=num_threads,
                to_tensor=to_tensor,
                permute=permute,
            ),
        ],
    }

    loader = SlipstreamLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pipelines=pipelines,
        exclude_fields=["path"],
        use_threading=True,
        image_format=image_format,
    )

    def run_epoch():
        total_samples = 0
        for batch in tqdm(loader, leave=False):
            img = batch["image"]
            # Just access shape to confirm data exists (no reduction ops)
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
            "pipeline_type": "decode_only",
            "to_tensor": to_tensor,
            "permute": permute,
            "target_size": target_size,
            "batch_size": batch_size,
            "num_threads": num_threads,
            "image_format": image_format,
        },
    )


def benchmark_full_pipeline(
    dataset,
    batch_size: int,
    num_epochs: int,
    num_warmup: int,
    to_tensor: bool,
    permute: bool,
    device: str,
    num_threads: int = 0,
    target_size: int = 224,
    image_format: str = "jpeg",
) -> BenchmarkResult:
    """Benchmark full pipeline: decoder -> ToTorchImage."""
    from slipstream import SlipstreamLoader
    from slipstream.decoders import DecodeRandomResizedCrop
    from slipstream.transforms import ToTorchImage

    name_parts = ["RRC"]
    if to_tensor:
        name_parts.append("tensor")
    else:
        name_parts.append("numpy")
    if permute:
        name_parts.append("CHW")
    else:
        name_parts.append("HWC")
    name_parts.append(f"-> {device}")
    if image_format != "jpeg":
        name_parts.append(image_format)
    name = " ".join(name_parts)

    pipelines = {
        "image": [
            DecodeRandomResizedCrop(
                target_size,
                num_threads=num_threads,
                to_tensor=to_tensor,
                permute=permute,
            ),
            ToTorchImage(device=device, dtype=torch.float32),
        ],
    }

    loader = SlipstreamLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pipelines=pipelines,
        exclude_fields=["path"],
        use_threading=True,
        image_format=image_format,
    )

    use_cuda_sync = device.startswith("cuda")

    def run_epoch():
        total_samples = 0
        for batch in tqdm(loader, leave=False):
            img = batch["image"]
            total_samples += img.shape[0]
        # Sync at end of epoch for accurate GPU timing
        if use_cuda_sync:
            torch.cuda.synchronize()
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
            "pipeline_type": "full_pipeline",
            "to_tensor": to_tensor,
            "permute": permute,
            "device": device,
            "target_size": target_size,
            "batch_size": batch_size,
            "num_threads": num_threads,
            "image_format": image_format,
        },
    )


def main():
    parser = argparse.ArgumentParser(description="Benchmark decoder output format performance")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET, help="Dataset path (S3 or local)")
    parser.add_argument("--cache-dir", type=str, default=None, help="Override cache directory")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of timed epochs")
    parser.add_argument("--warmup", type=int, default=1, help="Number of warmup epochs")
    parser.add_argument("--num-threads", type=int, default=0, help="NumbaBatchDecoder threads (0=auto)")
    parser.add_argument("--target-size", type=int, default=224, help="Target crop size")
    parser.add_argument("--machine-name", type=str, default=None, help="Machine name for results")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"],
                        help="Target device: auto (cuda if available), cpu, or cuda")
    parser.add_argument("--image-format", type=str, default="jpeg",
                        choices=["jpeg", "yuv420", "all"],
                        help="Image format: jpeg, yuv420, or all (run both)")
    parser.add_argument("--skip-decode-only", action="store_true",
                        help="Skip decode-only benchmarks")
    parser.add_argument("--save", action="store_true", help="Save results to JSON file")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path (implies --save)")
    args = parser.parse_args()

    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # Print machine info
    machine_info = get_machine_info(args.machine_name)
    print(machine_info)

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
    print(f"Target device: {device}")

    results = []

    # Determine which formats to benchmark
    if args.image_format == "all":
        formats = ["jpeg", "yuv420"]
    else:
        formats = [args.image_format]

    # Configuration matrix: (to_tensor, permute)
    configs = [
        (True, True),    # Baseline: tensor CHW (current default)
        (False, True),   # Numpy CHW
        (False, False),  # Numpy HWC
    ]

    for image_format in formats:
        if len(formats) > 1:
            print(f"\n{'=' * 60}")
            print(f"FORMAT: {image_format.upper()}")
            print(f"{'=' * 60}")

        # Decode-only benchmarks (CPU, no ToTorchImage)
        if not args.skip_decode_only:
            print(f"\n--- Decode Only (no device transfer) ---")
            for to_tensor, permute in configs:
                result = benchmark_decode_only(
                    dataset,
                    args.batch_size,
                    args.epochs,
                    args.warmup,
                    to_tensor=to_tensor,
                    permute=permute,
                    num_threads=args.num_threads,
                    target_size=args.target_size,
                    image_format=image_format,
                )
                results.append(result)

        # Full pipeline benchmarks (with ToTorchImage)
        print(f"\n--- Full Pipeline (decode -> ToTorchImage({device})) ---")
        for to_tensor, permute in configs:
            result = benchmark_full_pipeline(
                dataset,
                args.batch_size,
                args.epochs,
                args.warmup,
                to_tensor=to_tensor,
                permute=permute,
                device=device,
                num_threads=args.num_threads,
                target_size=args.target_size,
                image_format=image_format,
            )
            results.append(result)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(format_results_table(results))

    print("\nConfiguration key:")
    print("  tensor CHW: to_tensor=True, permute=True (current default)")
    print("  numpy CHW:  to_tensor=False, permute=True (Numba permute, torch.from_numpy)")
    print("  numpy HWC:  to_tensor=False, permute=False (no Numba permute, GPU permute)")

    # Save results
    if args.output:
        save_results(results, machine_info, args.output, "output_format")
    elif args.save:
        name = machine_info.machine_name.replace(".", "_").replace(" ", "_")
        output_path = Path(__file__).parent / "results" / f"output_format_{name}.json"
        save_results(results, machine_info, output_path, "output_format")


if __name__ == "__main__":
    main()
