#!/usr/bin/env python3
"""Benchmark SlipstreamLoader full pipeline performance.

Tests complete training/validation pipelines with:
- Raw I/O (no pipeline)
- RandomResizedCrop (training)
- CenterCrop (validation)
- 2x RandomResizedCrop (multi-crop SSL)

Usage:
    uv run python benchmarks/benchmark_loader.py
    uv run python benchmarks/benchmark_loader.py --batch-size 256 --epochs 3
    uv run python benchmarks/benchmark_loader.py --image-format yuv420
    uv run python benchmarks/benchmark_loader.py --image-format all
    uv run python benchmarks/benchmark_loader.py --output results/loader.json
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


# Default dataset path
DEFAULT_DATASET = "s3://visionlab-datasets/imagenet1k/pre-processed/s256-l512-jpgbytes-q100-streaming/val/"


def benchmark_loader(
    dataset,
    batch_size: int,
    num_epochs: int,
    num_warmup: int,
    pipeline_type: str,  # "train", "val", "raw", "multi-crop"
    num_threads: int = 0,
    target_size: int = 224,
    use_threading: bool = True,
    image_format: str = "jpeg",
) -> BenchmarkResult:
    """Benchmark SlipstreamLoader with specified pipeline."""
    from slipstream import SlipstreamLoader
    from slipstream.pipelines import (
        CenterCrop, RandomResizedCrop, MultiCropRandomResizedCrop,
    )

    mode = "threaded" if use_threading else "simple"
    fmt_label = f", {image_format}" if image_format != "jpeg" else ""

    if pipeline_type == "train":
        pipelines = {
            "image": [
                RandomResizedCrop(target_size, num_threads=num_threads),
            ],
        }
        name = f"SlipstreamLoader (RRC, {mode}{fmt_label})"
    elif pipeline_type == "val":
        pipelines = {
            "image": [
                CenterCrop(target_size, num_threads=num_threads),
            ],
        }
        name = f"SlipstreamLoader (CenterCrop, {mode}{fmt_label})"
    elif pipeline_type == "multi-crop":
        pipelines = {
            "image": [
                MultiCropRandomResizedCrop(
                    num_crops=2, size=target_size, num_threads=num_threads,
                ),
            ],
        }
        name = f"SlipstreamLoader (2x RRC fused multi-crop, {mode}{fmt_label})"
    else:  # raw
        pipelines = None
        name = f"SlipstreamLoader (raw, {mode}{fmt_label})"

    loader = SlipstreamLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pipelines=pipelines,
        exclude_fields=["path"],
        use_threading=use_threading,
        image_format=image_format,
    )

    def run_epoch():
        total_samples = 0
        for batch in tqdm(loader, leave=False):
            if pipeline_type == "raw":
                img_data = batch.get("image")
                if isinstance(img_data, dict):
                    total_samples += len(img_data["data"])
                else:
                    total_samples += batch_size
            elif pipeline_type == "multi-crop":
                views = batch["image"]
                total_samples += views[0].shape[0]
            else:
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
            "pipeline_type": pipeline_type,
            "target_size": target_size,
            "batch_size": batch_size,
            "num_threads": num_threads,
            "use_threading": use_threading,
            "image_format": image_format,
        },
    )


def main():
    parser = argparse.ArgumentParser(description="Benchmark SlipstreamLoader full pipeline")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET, help="Dataset path (S3 or local)")
    parser.add_argument("--cache-dir", type=str, default=None, help="Override cache directory")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of timed epochs")
    parser.add_argument("--warmup", type=int, default=1, help="Number of warmup epochs")
    parser.add_argument("--num-threads", type=int, default=0, help="NumbaBatchDecoder threads (0=auto)")
    parser.add_argument("--target-size", type=int, default=224, help="Target crop size")
    parser.add_argument("--machine-name", type=str, default=None, help="Machine name for results")
    parser.add_argument("--image-format", type=str, default="jpeg",
                        choices=["jpeg", "yuv420", "all"],
                        help="Image format: jpeg, yuv420, or all (run both)")
    parser.add_argument("--skip-multi-crop", action="store_true", help="Skip multi-crop benchmark")
    parser.add_argument("--multi-crop", action="store_true", help="Only run multi-crop benchmark")
    parser.add_argument("--save", action="store_true", help="Save results to JSON file")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path (implies --save)")
    args = parser.parse_args()

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

    results = []

    # Determine which formats to benchmark
    if args.image_format == "all":
        formats = ["jpeg", "yuv420"]
    else:
        formats = [args.image_format]

    for image_format in formats:
        if len(formats) > 1:
            print(f"\n{'=' * 60}")
            print(f"FORMAT: {image_format.upper()}")
            print(f"{'=' * 60}")

        # Benchmark each pipeline type (skip if --multi-crop)
        if not args.multi_crop:
            for pipeline_type in ["raw", "train", "val"]:
                result = benchmark_loader(
                    dataset, args.batch_size, args.epochs, args.warmup,
                    pipeline_type=pipeline_type, num_threads=args.num_threads,
                    target_size=args.target_size, use_threading=False,
                    image_format=image_format,
                )
                results.append(result)

                result = benchmark_loader(
                    dataset, args.batch_size, args.epochs, args.warmup,
                    pipeline_type=pipeline_type, num_threads=args.num_threads,
                    target_size=args.target_size, use_threading=True,
                    image_format=image_format,
                )
                results.append(result)

        # Multi-crop SSL benchmark
        if args.multi_crop or not args.skip_multi_crop:
            result = benchmark_loader(
                dataset, args.batch_size, args.epochs, args.warmup,
                pipeline_type="multi-crop", num_threads=args.num_threads,
                target_size=args.target_size, use_threading=False,
                image_format=image_format,
            )
            results.append(result)

            result = benchmark_loader(
                dataset, args.batch_size, args.epochs, args.warmup,
                pipeline_type="multi-crop", num_threads=args.num_threads,
                target_size=args.target_size, use_threading=True,
                image_format=image_format,
            )
            results.append(result)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(format_results_table(results))

    print("\nReference targets (direct NumbaBatchDecoder, no tensor conversion):")
    print("  Raw I/O: ~775k-939k samples/sec")
    print("  Decode + RRC: ~13,851 samples/sec")
    print("  Decode + CenterCrop: ~15,749 samples/sec")
    print("  Multi-crop (2x RRC fused): ~10,500 samples/sec (decode once, crop twice)")

    # Save results
    if args.output:
        save_results(results, machine_info, args.output, "loader")
    elif args.save:
        name = machine_info.machine_name.replace(".", "_").replace(" ", "_")
        output_path = Path(__file__).parent / "results" / f"loader_{name}.json"
        save_results(results, machine_info, output_path, "loader")


if __name__ == "__main__":
    main()
