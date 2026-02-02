"""Benchmark slipstream GPU batch transforms vs torchvision v2 equivalents.

Measures three modes:
1. Per-sample slipstream — loop over [C,H,W] tensors
2. Per-sample torchvision — loop over [C,H,W] tensors
3. Per-batch slipstream — apply to [B,C,H,W] batch (TV can't do per-image randomization)

Usage:
    uv run python benchmarks/benchmark_transforms.py
    uv run python benchmarks/benchmark_transforms.py --transform RandomHorizontalFlip
    uv run python benchmarks/benchmark_transforms.py --batch-size 256 --num-samples 10000
    uv run python benchmarks/benchmark_transforms.py --device cuda
    uv run python benchmarks/benchmark_transforms.py --list
    uv run python benchmarks/benchmark_transforms.py --skip-torchvision
"""

from __future__ import annotations

import argparse
import time

import torch
from torchvision.transforms import v2

from slipstream.transforms import (
    CircularMask,
    FixedOpticalDistortion,
    Normalize,
    RandomBrightness,
    RandomColorJitter,
    RandomColorJitterYIQ,
    RandomContrast,
    RandomGaussianBlur,
    RandomGrayscale,
    RandomHorizontalFlip,
    RandomPatchShuffle,
    RandomRotate,
    RandomRotateObject,
    RandomSolarization,
    RandomZoom,
    SRGBToLMS,
    ToGrayscale,
)

from benchmarks.utils import get_machine_info

# ImageNet stats
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def build_transform_configs(device: str) -> dict:
    """Build {name: (ss_transform, tv_transform_or_None, input_dtype, input_range)}."""
    return {
        "Normalize": (
            Normalize(MEAN, STD, device=device),
            v2.Normalize(MEAN, STD),
            torch.float32, (0, 1),
        ),
        "ToGrayscale": (
            ToGrayscale(num_output_channels=3),
            v2.Grayscale(3),
            torch.float32, (0, 1),
        ),
        "RandomGrayscale": (
            RandomGrayscale(p=0.5, device=device),
            v2.RandomGrayscale(p=0.5),
            torch.float32, (0, 1),
        ),
        "RandomHorizontalFlip": (
            RandomHorizontalFlip(p=0.5, device=device),
            v2.RandomHorizontalFlip(p=0.5),
            torch.float32, (0, 1),
        ),
        "RandomRotate": (
            RandomRotate(p=0.5, max_deg=45, device=device),
            v2.RandomRotation(degrees=45),
            torch.float32, (0, 1),
        ),
        "RandomZoom": (
            RandomZoom(p=0.5, zoom=(0.5, 1.0), device=device),
            v2.RandomAffine(degrees=0, scale=(0.5, 1.0)),
            torch.float32, (0, 1),
        ),
        "RandomBrightness": (
            RandomBrightness(p=0.5, scale_range=(0.6, 1.4), device=device),
            v2.ColorJitter(brightness=(0.6, 1.4)),
            torch.float32, (0, 1),
        ),
        "RandomContrast": (
            RandomContrast(p=0.5, scale_range=(0.6, 1.4), device=device),
            v2.ColorJitter(contrast=(0.6, 1.4)),
            torch.float32, (0, 1),
        ),
        "RandomGaussianBlur": (
            RandomGaussianBlur(p=0.5, kernel_size=7,
                               sigma_range=(0.1, 2.0), device=device),
            v2.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0)),
            torch.float32, (0, 1),
        ),
        "RandomSolarization": (
            RandomSolarization(p=0.5, threshold=0.5, device=device),
            v2.RandomSolarize(threshold=0.5, p=0.5),
            torch.float32, (0, 1),
        ),
        "RandomColorJitter(HSV)": (
            RandomColorJitter(p=0.5, hue=0.1, saturation=0.3,
                              value=0.3, contrast=0.3, device=device),
            v2.ColorJitter(brightness=0.3, contrast=0.3,
                           saturation=0.3, hue=0.1),
            torch.float32, (0, 1),
        ),
        "RandomColorJitterYIQ": (
            RandomColorJitterYIQ(p=0.5, hue=20, saturation=0.3,
                                 value=0.3, brightness=0.3, contrast=0.3, device=device),
            None,
            torch.float32, (0, 1),
        ),
        # Slipstream-only
        "RandomPatchShuffle": (
            RandomPatchShuffle(sizes=0.25, p=0.5, img_size=224, device=device),
            None,
            torch.float32, (0, 1),
        ),
        "CircularMask": (
            CircularMask(output_size=224, device=device),
            None,
            torch.float32, (0, 1),
        ),
        "FixedOpticalDistortion": (
            FixedOpticalDistortion(output_size=(
                224, 224), distortion=-0.5, device=device),
            None,
            torch.float32, (0, 1),
        ),
        "RandomRotateObject": (
            RandomRotateObject(p=0.5, max_deg=30,
                               scale=(1.0, 1.5), device=device),
            None,
            torch.float32, (0, 1),
        ),
        "SRGBToLMS": (
            SRGBToLMS(),
            None,
            torch.float32, (0, 1),
        ),
    }


def make_input(batch_size: int, dtype: torch.dtype, device: str) -> torch.Tensor:
    """Create synthetic input batch [B, 3, 224, 224]."""
    return torch.rand(batch_size, 3, 224, 224, dtype=dtype, device=device)


def benchmark_per_sample_ss(transform, data: torch.Tensor, num_samples: int, device: str) -> float:
    """Per-sample slipstream: loop over images, unsqueeze/squeeze."""
    use_cuda = device.startswith("cuda")

    # Warmup
    for i in range(min(10, data.shape[0])):
        out = transform(data[i:i+1])
        _ = out.shape[0]
    if use_cuda:
        torch.cuda.synchronize()

    start = time.perf_counter()
    count = 0
    while count < num_samples:
        for i in range(data.shape[0]):
            out = transform(data[i:i+1])
            _ = out.shape[0]
            count += 1
            if count >= num_samples:
                break
    if use_cuda:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return count / elapsed


def benchmark_per_sample_tv(transform, data: torch.Tensor, num_samples: int, device: str) -> float:
    """Per-sample torchvision: loop over [C,H,W] images."""
    use_cuda = device.startswith("cuda")

    # Warmup
    for i in range(min(10, data.shape[0])):
        out = transform(data[i])
        _ = out.shape[0]
    if use_cuda:
        torch.cuda.synchronize()

    start = time.perf_counter()
    count = 0
    while count < num_samples:
        for i in range(data.shape[0]):
            out = transform(data[i])
            _ = out.shape[0]
            count += 1
            if count >= num_samples:
                break
    if use_cuda:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return count / elapsed


def benchmark_per_batch_ss(transform, data: torch.Tensor, num_samples: int, device: str) -> float:
    """Per-batch slipstream: apply to full [B,C,H,W] batch."""
    use_cuda = device.startswith("cuda")
    batch_size = data.shape[0]

    # Warmup
    for _ in range(10):
        out = transform(data)
        _ = out.shape[0]
    if use_cuda:
        torch.cuda.synchronize()

    start = time.perf_counter()
    count = 0
    while count < num_samples:
        out = transform(data)
        _ = out.shape[0]
        count += batch_size
    if use_cuda:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return count / elapsed


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark slipstream transforms vs torchvision v2")
    parser.add_argument("--transform", type=str, default=None,
                        help="Run only this transform")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size (default: 64)")
    parser.add_argument("--num-samples", type=int, default=10000,
                        help="Number of samples per benchmark")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device (cpu or cuda)")
    parser.add_argument("--list", action="store_true",
                        help="List available transforms")
    parser.add_argument("--skip-torchvision", action="store_true",
                        help="Skip torchvision benchmarks")
    args = parser.parse_args()

    configs = build_transform_configs(args.device)

    if args.list:
        print("Available transforms:")
        for name, (_, tv, _, _) in configs.items():
            tv_str = "yes" if tv is not None else "no"
            print(f"  {name:<30s} TV equivalent: {tv_str}")
        return

    # Select transforms
    if args.transform:
        if args.transform not in configs:
            print(f"Unknown transform: {args.transform}")
            print(f"Available: {', '.join(configs.keys())}")
            return
        configs = {args.transform: configs[args.transform]}

    # Machine info
    info = get_machine_info()
    print(info)
    print(
        f"\nBenchmark config: batch_size={args.batch_size}, num_samples={args.num_samples}, device={args.device}")
    print()

    # Results
    results = []

    for name, (ss_transform, tv_transform, dtype, _) in configs.items():
        print(f"--- {name} ---")
        data = make_input(args.batch_size, dtype, args.device)

        # Per-batch slipstream
        try:
            batch_ss = benchmark_per_batch_ss(
                ss_transform, data, args.num_samples, args.device)
        except Exception as e:
            print(f"  Per-batch SS error: {e}")
            batch_ss = 0.0
        print(f"  Per-batch (SS):  {batch_ss:>10,.0f} samples/sec")

        # Per-sample slipstream
        try:
            sample_ss = benchmark_per_sample_ss(
                ss_transform, data, args.num_samples, args.device)
        except Exception as e:
            print(f"  Per-sample SS error: {e}")
            sample_ss = 0.0
        print(f"  Per-sample (SS): {sample_ss:>10,.0f} samples/sec")

        # Per-sample torchvision
        sample_tv = 0.0
        if tv_transform is not None and not args.skip_torchvision:
            try:
                sample_tv = benchmark_per_sample_tv(
                    tv_transform, data, args.num_samples, args.device)
            except Exception as e:
                print(f"  Per-sample TV error: {e}")
                sample_tv = 0.0
            print(f"  Per-sample (TV): {sample_tv:>10,.0f} samples/sec")

        batch_speedup = batch_ss / sample_ss if sample_ss > 0 else 0
        ss_vs_tv = sample_ss / sample_tv if sample_tv > 0 else 0
        print(f"  Batch speedup:   {batch_speedup:>10.2f}x")
        if sample_tv > 0:
            print(f"  SS vs TV:        {ss_vs_tv:>10.2f}x")
        print()

        results.append({
            "name": name,
            "sample_ss": sample_ss,
            "sample_tv": sample_tv,
            "batch_ss": batch_ss,
            "batch_speedup": batch_speedup,
            "ss_vs_tv": ss_vs_tv,
        })

    # Summary table
    print("\n" + "=" * 115)
    print("SUMMARY")
    print("=" * 115)
    header = (
        f"| {'Transform':<25s} | {'Per-Sample (SS)':>15s} | {'Per-Sample (TV)':>15s} "
        f"| {'SS vs TV':>10s} | {'Per-Batch (SS)':>15s} | {'Batch Speedup':>13s} |"
    )
    sep = (
        f"|{'-'*27}|{'-'*17}|{'-'*17}"
        f"|{'-'*12}|{'-'*17}|{'-'*15}|"
    )
    print(header)
    print(sep)
    for r in results:
        tv_str = f"{r['sample_tv']:>12,.0f}" if r['sample_tv'] > 0 else "           N/A"
        vs_str = f"{r['ss_vs_tv']:>7.2f}x" if r['ss_vs_tv'] > 0 else "       N/A"
        print(
            f"| {r['name']:<25s} | {r['sample_ss']:>12,.0f}    | {tv_str}    "
            f"| {vs_str}    | {r['batch_ss']:>12,.0f}    | {r['batch_speedup']:>10.2f}x   |"
        )
    print(sep)


if __name__ == "__main__":
    main()
