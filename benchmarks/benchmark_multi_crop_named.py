#!/usr/bin/env python3
"""Benchmark DecodeMultiRandomResizedCrop vs DecodeUniformMultiRandomResizedCrop.

Compares:
1. DecodeUniformMultiRandomResizedCrop (legacy): 2 crops @ 224, same scale/ratio
2. DecodeMultiRandomResizedCrop: 2 crops @ 224, same params (direct comparison)
3. DecodeMultiRandomResizedCrop: 1 global @ 224 + 1 local @ 96
4. DecodeMultiRandomResizedCrop: 2 local @ 96
5. DecodeMultiRandomResizedCrop: 2 global @ 224 + 4 local @ 96 (DINO-style)

All benchmarks are decode+crop only (no transforms/normalization).

Usage:
    uv run python benchmarks/benchmark_multi_crop_named.py
    uv run python benchmarks/benchmark_multi_crop_named.py --batch-size 256 --epochs 3
    uv run python benchmarks/benchmark_multi_crop_named.py --image-format yuv420
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from tqdm import tqdm

from benchmarks.utils import (
    BenchmarkResult,
    format_results_table,
    get_drive_info,
    get_machine_info,
)


DEFAULT_DATASET = "s3://visionlab-datasets/imagenet1k/pre-processed/s256-l512-jpgbytes-q100-streaming/val/"


def benchmark_loader(
    dataset,
    batch_size: int,
    num_epochs: int,
    num_warmup: int,
    name: str,
    pipelines: dict,
    count_fn,
    image_format: str = "jpeg",
) -> BenchmarkResult:
    """Run a loader benchmark."""
    from slipstream import SlipstreamLoader

    loader = SlipstreamLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pipelines=pipelines,
        exclude_fields=["path"],
        use_threading=False,
        image_format=image_format,
    )

    def run_epoch():
        total = 0
        for batch in tqdm(loader, leave=False):
            total += count_fn(batch)
        return total

    print(f"\n{name}:")
    # Warmup
    print(f"  Warmup ({num_warmup} epoch(s)):")
    for i in range(num_warmup):
        start = time.perf_counter()
        total = run_epoch()
        elapsed = time.perf_counter() - start
        rate = total / elapsed
        print(f"    Warmup {i + 1}: {rate:,.0f} samples/sec ({elapsed:.2f}s)")

    # Timed
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
        per_epoch_results=epoch_results,
        metadata={"batch_size": batch_size, "image_format": image_format},
    )


def main():
    parser = argparse.ArgumentParser(description="Benchmark multi-crop variants")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--num-threads", type=int, default=12, help="NumbaBatchDecoder threads")
    parser.add_argument("--image-format", type=str, default="jpeg",
                        choices=["jpeg", "yuv420"])
    args = parser.parse_args()

    machine_info = get_machine_info()
    print(machine_info)

    print("\nLoading dataset...")
    from slipstream import SlipstreamDataset
    dataset = SlipstreamDataset(
        remote_dir=args.dataset,
        cache_dir=args.cache_dir,
        decode_images=False,
    )
    print(f"Dataset: {len(dataset):,} samples")

    cache_path = dataset.cache_path
    print(f"Cache path: {cache_path}")
    cache_drive = get_drive_info(cache_path)
    print(f"Cache drive: {cache_drive['type']} (device: {cache_drive['device']})")

    fmt = args.image_format
    results = []

    # --- Benchmark 1: Original DecodeUniformMultiRandomResizedCrop (2 crops @ 224) ---
    from slipstream.decoders import DecodeUniformMultiRandomResizedCrop
    pipelines_orig = {
        "image": [
            DecodeUniformMultiRandomResizedCrop(num_crops=2, size=224, num_threads=args.num_threads),
        ],
    }
    results.append(benchmark_loader(
        dataset, args.batch_size, args.epochs, args.warmup,
        name=f"DecodeUniformMultiRandomResizedCrop (2x224, {fmt})",
        pipelines=pipelines_orig,
        count_fn=lambda b: b["image"][0].shape[0],
        image_format=fmt,
    ))

    # --- Benchmark 2: New DecodeMultiRandomResizedCrop (2 crops @ 224, matching params) ---
    from slipstream.decoders import DecodeMultiRandomResizedCrop
    pipelines_new_2 = {
        "image": [
            DecodeDecodeMultiRandomResizedCrop({
                "view_0": dict(size=224),
                "view_1": dict(size=224),
            }, num_threads=args.num_threads),
        ],
    }
    results.append(benchmark_loader(
        dataset, args.batch_size, args.epochs, args.warmup,
        name=f"DecodeMultiRandomResizedCrop (2x224, {fmt})",
        pipelines=pipelines_new_2,
        count_fn=lambda b: b["view_0"].shape[0],
        image_format=fmt,
    ))

    # --- Benchmark 3: DecodeMultiRandomResizedCrop (1 global + 1 local) ---
    pipelines_1g1l = {
        "image": [
            DecodeMultiRandomResizedCrop({
                "global_0": dict(size=224, scale=(0.4, 1.0)),
                "local_0":  dict(size=96,  scale=(0.05, 0.4)),
            }, num_threads=args.num_threads),
        ],
    }
    results.append(benchmark_loader(
        dataset, args.batch_size, args.epochs, args.warmup,
        name=f"DecodeMultiRandomResizedCrop (1x224+1x96, {fmt})",
        pipelines=pipelines_1g1l,
        count_fn=lambda b: b["global_0"].shape[0],
        image_format=fmt,
    ))

    # --- Benchmark 4: DecodeMultiRandomResizedCrop (2 local) ---
    pipelines_2l = {
        "image": [
            DecodeMultiRandomResizedCrop({
                "local_0": dict(size=96, scale=(0.05, 0.4)),
                "local_1": dict(size=96, scale=(0.05, 0.4)),
            }, num_threads=args.num_threads),
        ],
    }
    results.append(benchmark_loader(
        dataset, args.batch_size, args.epochs, args.warmup,
        name=f"DecodeMultiRandomResizedCrop (2x96, {fmt})",
        pipelines=pipelines_2l,
        count_fn=lambda b: b["local_0"].shape[0],
        image_format=fmt,
    ))

    # --- Benchmark 5: DecodeMultiRandomResizedCrop (2 global + 4 local, DINO-style) ---
    pipelines_dino = {
        "image": [
            DecodeMultiRandomResizedCrop({
                "global_0": dict(size=224, scale=(0.4, 1.0)),
                "global_1": dict(size=224, scale=(0.4, 1.0)),
                "local_0":  dict(size=96,  scale=(0.05, 0.4)),
                "local_1":  dict(size=96,  scale=(0.05, 0.4)),
                "local_2":  dict(size=96,  scale=(0.05, 0.4)),
                "local_3":  dict(size=96,  scale=(0.05, 0.4)),
            }, num_threads=args.num_threads),
        ],
    }
    results.append(benchmark_loader(
        dataset, args.batch_size, args.epochs, args.warmup,
        name=f"DecodeMultiRandomResizedCrop (2x224+4x96 DINO, {fmt})",
        pipelines=pipelines_dino,
        count_fn=lambda b: b["global_0"].shape[0],
        image_format=fmt,
    ))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(format_results_table(results))


if __name__ == "__main__":
    main()
