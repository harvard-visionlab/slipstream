#!/usr/bin/env python3
"""Benchmark YUV420P vs JPEG decode throughput via Numba prange pipeline.

Compares YUV420NumbaBatchDecoder (YUV420→RGB color conversion only)
vs NumbaBatchDecoder (full JPEG Huffman+IDCT decode).

Three modes per format:
  - Decode-only (full resolution to buffer)
  - Decode + CenterCrop(224)
  - Decode + RandomResizedCrop(224)

Requires YUV420 cache files (image_yuv420.bin, image_yuv420.meta.npy).
Generate them with:
    uv run python experiments/format_comparison/convert_yuv420.py

Usage:
    uv run python benchmarks/benchmark_yuv420_decode.py
    uv run python benchmarks/benchmark_yuv420_decode.py --batch-size 256 --epochs 2 --warmup 1
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

DEFAULT_DATASET = "s3://visionlab-datasets/imagenet1k/pre-processed/s256-l512-jpgbytes-q100-streaming/val/"

VARIABLE_METADATA_DTYPE = np.dtype([
    ('data_ptr', '<u8'),
    ('data_size', '<u8'),
    ('height', '<u4'),
    ('width', '<u4'),
])


class FormatData:
    """Mmap-backed image data for a single format."""

    def __init__(self, cache_dir: Path, image_field: str, suffix: str = ""):
        tag = f"_{suffix}" if suffix else ""
        data_path = cache_dir / f"{image_field}{tag}.bin"
        meta_path = cache_dir / f"{image_field}{tag}.meta.npy"

        if not data_path.exists():
            raise FileNotFoundError(f"Not found: {data_path}")

        self.data_mmap = np.memmap(data_path, dtype=np.uint8, mode='r')
        self.metadata = np.load(meta_path, mmap_mode='r')
        self.num_samples = len(self.metadata)

    def load_batch(self, indices: np.ndarray, max_size: int | None = None):
        batch_size = len(indices)
        sizes = np.array([int(self.metadata[i]['data_size']) for i in indices], dtype=np.uint64)
        heights = np.array([int(self.metadata[i]['height']) for i in indices], dtype=np.uint32)
        widths = np.array([int(self.metadata[i]['width']) for i in indices], dtype=np.uint32)

        if max_size is None:
            max_size = int(np.max(sizes))

        data = np.zeros((batch_size, max_size), dtype=np.uint8)
        for j, idx in enumerate(indices):
            ptr = int(self.metadata[idx]['data_ptr'])
            sz = int(sizes[j])
            data[j, :sz] = self.data_mmap[ptr:ptr + sz]

        return data, sizes, heights, widths


def run_benchmark(
    label: str,
    fmt_data: FormatData,
    decoder,
    batch_size: int,
    num_warmup: int,
    num_epochs: int,
    mode: str = "decode",
    target_size: int = 224,
) -> float:
    num_samples = fmt_data.num_samples
    num_batches = (num_samples + batch_size - 1) // batch_size
    indices = np.arange(num_samples, dtype=np.int64)

    decode_buffer = None
    if mode == "decode":
        first_data, first_sizes, first_h, first_w = fmt_data.load_batch(indices[:batch_size])
        max_h = int(np.max(first_h) * 1.2)
        max_w = int(np.max(first_w) * 1.2)
        decode_buffer = np.zeros((batch_size, max_h, max_w, 3), dtype=np.uint8)

    def run_epoch():
        nonlocal decode_buffer
        total = 0
        for i in tqdm(range(num_batches), leave=False):
            start = i * batch_size
            end = min(start + batch_size, num_samples)
            batch_idx = indices[start:end]
            data, sizes, heights, widths = fmt_data.load_batch(batch_idx)

            if mode == "rrc":
                decoder.decode_batch_random_crop(
                    data, sizes, heights, widths,
                    target_size=target_size, scale=(0.08, 1.0),
                )
            elif mode == "center":
                decoder.decode_batch_center_crop(
                    data, sizes, heights, widths,
                    crop_size=target_size,
                )
            else:
                max_h = int(np.max(heights))
                max_w = int(np.max(widths))
                if decode_buffer.shape[1] < max_h or decode_buffer.shape[2] < max_w:
                    decode_buffer = np.zeros((batch_size, max_h, max_w, 3), dtype=np.uint8)
                decoder.decode_batch_to_buffer(data, sizes, heights, widths, decode_buffer)

            total += len(batch_idx)
        return total

    print(f"\n{label}:")

    for i in range(num_warmup):
        t0 = time.perf_counter()
        total = run_epoch()
        elapsed = time.perf_counter() - t0
        print(f"  Warmup {i+1}: {total/elapsed:,.0f} img/s ({elapsed:.1f}s)")

    rates = []
    for i in range(num_epochs):
        t0 = time.perf_counter()
        total = run_epoch()
        elapsed = time.perf_counter() - t0
        rate = total / elapsed
        rates.append(rate)
        print(f"  Epoch {i+1}: {rate:,.0f} img/s ({elapsed:.1f}s)")

    avg = np.mean(rates)
    print(f"  Average: {avg:,.0f} img/s")
    return avg


def main():
    parser = argparse.ArgumentParser(description="Benchmark YUV420 vs JPEG decode (Numba prange)")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--target-size", type=int, default=224)
    parser.add_argument("--cache-dir", type=str, default=None)
    args = parser.parse_args()

    from slipstream import SlipstreamDataset
    from slipstream.cache import OptimizedCache

    print("Loading dataset...")
    dataset = SlipstreamDataset(remote_dir=args.dataset, cache_dir=args.cache_dir, decode_images=False)
    cache_path = dataset.cache_path

    if OptimizedCache.exists(cache_path):
        cache = OptimizedCache.load(cache_path)
    else:
        cache = OptimizedCache.build(dataset)

    cache_dir = cache.cache_dir
    print(f"Cache: {cache_dir}")
    print(f"Samples: {cache.num_samples:,}")

    # Load format data
    print("\nLoading format data...")
    jpeg_data = FormatData(cache_dir, "image")
    print(f"  JPEG:   {jpeg_data.num_samples:,} samples")

    try:
        yuv_data = FormatData(cache_dir, "image", suffix="yuv420")
        print(f"  YUV420: {yuv_data.num_samples:,} samples")
    except FileNotFoundError:
        print("  YUV420: NOT FOUND — run converter first:")
        print("    uv run python experiments/format_comparison/convert_yuv420.py")
        return

    # Report storage comparison
    jpeg_bytes = sum(int(jpeg_data.metadata[i]['data_size']) for i in range(jpeg_data.num_samples))
    yuv_bytes = sum(int(yuv_data.metadata[i]['data_size']) for i in range(yuv_data.num_samples))
    print(f"\n  Storage: JPEG={jpeg_bytes/1e9:.2f} GB, YUV420={yuv_bytes/1e9:.2f} GB "
          f"({yuv_bytes/jpeg_bytes:.2f}x JPEG)")

    # Create decoders
    from slipstream.decoders.numba_decoder import NumbaBatchDecoder
    from slipstream.decoders.yuv420_decoder import YUV420NumbaBatchDecoder

    jpeg_decoder = NumbaBatchDecoder(num_threads=args.num_workers)
    yuv_decoder = YUV420NumbaBatchDecoder(num_threads=args.num_workers)
    print(f"\nJPEG decoder:   {jpeg_decoder}")
    print(f"YUV420 decoder: {yuv_decoder}")

    # Run benchmarks
    results = {}
    modes = [
        ("decode", "Decode Only"),
        ("center", "+ CenterCrop(224)"),
        ("rrc", "+ RRC(224)"),
    ]

    for mode, mode_label in modes:
        print(f"\n{'='*60}")
        print(f"  {mode_label}")
        print(f"{'='*60}")

        jpeg_rate = run_benchmark(
            f"JPEG NumbaBatchDecoder {mode_label}",
            jpeg_data, jpeg_decoder, args.batch_size,
            args.warmup, args.epochs, mode=mode, target_size=args.target_size,
        )

        yuv_rate = run_benchmark(
            f"YUV420 YUV420NumbaBatchDecoder {mode_label}",
            yuv_data, yuv_decoder, args.batch_size,
            args.warmup, args.epochs, mode=mode, target_size=args.target_size,
        )

        results[mode] = {'jpeg': jpeg_rate, 'yuv420': yuv_rate}

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY: YUV420 vs JPEG (Numba prange pipeline)")
    print(f"{'='*60}")
    print(f"Storage: JPEG={jpeg_bytes/1e9:.2f} GB, YUV420={yuv_bytes/1e9:.2f} GB "
          f"({yuv_bytes/jpeg_bytes:.2f}x)")
    print(f"\n{'Mode':<20} {'JPEG':>10} {'YUV420':>10} {'YUV/JPEG':>10}")
    print("-" * 52)
    for mode, mode_label in modes:
        r = results[mode]
        ratio = r['yuv420'] / r['jpeg'] if r['jpeg'] > 0 else 0
        print(f"{mode_label:<20} {r['jpeg']:>8,.0f}/s {r['yuv420']:>8,.0f}/s {ratio:>9.2f}x")

    print(f"\nTarget: ≥1.5x throughput at ≤2x storage")
    winner = all(results[m]['yuv420'] / results[m]['jpeg'] >= 1.5 for m in results)
    storage_ok = (yuv_bytes / jpeg_bytes) <= 2.0
    if winner and storage_ok:
        print("→ YUV420 PASSES — worth integrating")
    else:
        print("→ YUV420 does not meet threshold")

    jpeg_decoder.shutdown()
    yuv_decoder.shutdown()


if __name__ == "__main__":
    main()
