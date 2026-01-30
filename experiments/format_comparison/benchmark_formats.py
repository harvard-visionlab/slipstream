#!/usr/bin/env python3
"""Benchmark decode throughput for alternative image formats.

Measures per-image decode latency and batch throughput for each format,
comparing against the JPEG/TurboJPEG baseline.

All benchmarks iterate the FULL dataset per epoch (50k images) with:
- 1 warmup epoch (cold → warm page cache)
- 2 timed epochs (warm, mmap page cache populated)

Three levels of measurement:
1. Single-image decode latency — fair per-image format comparison (full dataset)
2. Batch decode throughput — batched Python-loop throughput (full dataset)
3. Raw RGB memcpy via Numba — absolute ceiling (no decode, full dataset)

Usage:
    uv sync --group experiment
    uv run python experiments/format_comparison/benchmark_formats.py
    uv run python experiments/format_comparison/benchmark_formats.py --batch-size 256
    uv run python experiments/format_comparison/benchmark_formats.py --formats jpeg qoi raw
"""

from __future__ import annotations

import argparse
import struct
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm
from turbojpeg import TurboJPEG

LITDATA_VAL_PATH = "s3://visionlab-datasets/imagenet1k/pre-processed/s256-l512-jpgbytes-q100-streaming/val/"

VARIABLE_METADATA_DTYPE = np.dtype([
    ('data_ptr', '<u8'),
    ('data_size', '<u8'),
    ('height', '<u4'),
    ('width', '<u4'),
])

_turbo = TurboJPEG()


# =============================================================================
# Decoders (single image)
# =============================================================================

def decode_jpeg(data: bytes) -> np.ndarray:
    """Decode JPEG bytes to RGB [H, W, 3]."""
    return _turbo.decode(data, pixel_format=0)


def decode_qoi(data: bytes) -> np.ndarray:
    """Decode QOI bytes to RGB [H, W, 3]."""
    import qoi
    return qoi.decode(data)


def decode_raw(data: bytes) -> np.ndarray:
    """Decode raw RGB bytes (4-byte header + flat RGB)."""
    h, w = struct.unpack('<HH', data[:4])
    return np.frombuffer(data, dtype=np.uint8, offset=4).reshape(h, w, 3)


DECODERS = {
    'jpeg': decode_jpeg,
    'qoi': decode_qoi,
    'raw': decode_raw,
}


# =============================================================================
# Helpers
# =============================================================================

def _get_format_paths(cache_dir: Path, image_field: str, fmt: str):
    """Return (data_path, meta_path) for a format."""
    suffix = f"_{fmt}" if fmt != "jpeg" else ""
    data_path = cache_dir / f"{image_field}{suffix}.bin"
    meta_path = cache_dir / f"{image_field}{suffix}.meta.npy"
    return data_path, meta_path


def _run_epochs(run_fn, num_warmup: int, num_timed: int, label: str, verbose: bool,
                 total_samples: int = 0):
    """Run warmup + timed epochs, return per-epoch results.

    run_fn(pbar) should iterate the full dataset, calling pbar.update(n)
    for progress, and return total samples processed.
    """
    # Warmup
    for i in range(num_warmup):
        desc = f"    Warmup {i+1}"
        pbar = tqdm(total=total_samples, desc=desc, leave=False) if verbose and total_samples else None
        t0 = time.perf_counter()
        total = run_fn(pbar)
        elapsed = time.perf_counter() - t0
        if pbar:
            pbar.close()
        if verbose:
            print(f"    Warmup {i+1}: {total/elapsed:,.0f} img/s ({elapsed:.1f}s)")

    # Timed
    epoch_results = []
    for i in range(num_timed):
        desc = f"    Epoch {i+1}"
        pbar = tqdm(total=total_samples, desc=desc, leave=False) if verbose and total_samples else None
        t0 = time.perf_counter()
        total = run_fn(pbar)
        elapsed = time.perf_counter() - t0
        if pbar:
            pbar.close()
        rate = total / elapsed
        epoch_results.append({'samples_per_sec': rate, 'elapsed_sec': elapsed, 'total': total})
        if verbose:
            print(f"    Epoch {i+1}: {rate:,.0f} img/s ({elapsed:.1f}s)")

    avg_rate = np.mean([r['samples_per_sec'] for r in epoch_results])
    if verbose:
        print(f"    Average: {avg_rate:,.0f} img/s")
    return epoch_results, avg_rate


# =============================================================================
# Benchmarks
# =============================================================================

def benchmark_single_image_decode(
    cache_dir: Path,
    image_field: str,
    formats: list[str],
    num_warmup: int = 1,
    num_timed: int = 2,
    verbose: bool = True,
) -> dict[str, dict]:
    """Benchmark single-image decode latency for each format.

    Iterates the FULL dataset each epoch (single-threaded Python decode).
    1 warmup epoch + 2 timed epochs.
    """
    if verbose:
        print(f"\n{'='*60}")
        print("SINGLE-IMAGE DECODE (full dataset, per-image Python decode)")
        print(f"{'='*60}")
        print(f"({num_warmup} warmup + {num_timed} timed epochs)\n")

    meta_path = cache_dir / f"{image_field}.meta.npy"
    metadata = np.load(meta_path, mmap_mode='r')
    total_samples = len(metadata)

    results = {}

    for fmt in formats:
        data_path, fmt_meta_path = _get_format_paths(cache_dir, image_field, fmt)
        if not data_path.exists():
            if verbose:
                print(f"  {fmt.upper()}: skipped (not converted)")
            continue

        decoder = DECODERS[fmt]
        data_mmap = np.memmap(data_path, dtype=np.uint8, mode='r')
        fmt_meta = np.load(fmt_meta_path, mmap_mode='r')

        def run_epoch(pbar=None, d=decoder, dm=data_mmap, fm=fmt_meta, n=total_samples):
            for idx in range(n):
                ptr = int(fm[idx]['data_ptr'])
                size = int(fm[idx]['data_size'])
                _ = d(bytes(dm[ptr:ptr + size]))
                if pbar:
                    pbar.update(1)
            return n

        if verbose:
            print(f"  {fmt.upper()} ({total_samples:,} images):")

        epoch_results, avg_rate = _run_epochs(
            run_epoch, num_warmup, num_timed, fmt.upper(), verbose,
            total_samples=total_samples,
        )

        per_image_us = 1e6 / avg_rate
        results[fmt] = {
            'per_image_us': per_image_us,
            'throughput': avg_rate,
            'epochs': epoch_results,
        }

    # Comparison
    if verbose and 'jpeg' in results:
        jpeg_us = results['jpeg']['per_image_us']
        print(f"\n  {'Format':<8} {'Latency':>10} {'Throughput':>12} {'vs JPEG':>10}")
        print(f"  {'-'*42}")
        for fmt, info in results.items():
            ratio = jpeg_us / info['per_image_us']
            print(f"  {fmt.upper():<8} {info['per_image_us']:>8.1f} us  "
                  f"{info['throughput']:>8.0f}/s    {ratio:>6.2f}x")

    return results


def benchmark_batch_decode(
    cache_dir: Path,
    image_field: str,
    formats: list[str],
    batch_size: int = 256,
    num_warmup: int = 1,
    num_timed: int = 2,
    verbose: bool = True,
) -> dict[str, dict]:
    """Benchmark batch decode throughput (Python loop, no Numba).

    Iterates the FULL dataset each epoch in batches.
    1 warmup epoch + 2 timed epochs.
    """
    if verbose:
        print(f"\n{'='*60}")
        print("BATCH DECODE THROUGHPUT (full dataset, batched Python loop)")
        print(f"{'='*60}")
        print(f"(batch_size={batch_size}, {num_warmup} warmup + {num_timed} timed epochs)\n")

    meta_path = cache_dir / f"{image_field}.meta.npy"
    metadata = np.load(meta_path, mmap_mode='r')
    total_samples = len(metadata)
    all_indices = np.arange(total_samples, dtype=np.int64)

    results = {}

    for fmt in formats:
        data_path, fmt_meta_path = _get_format_paths(cache_dir, image_field, fmt)
        if not data_path.exists():
            if verbose:
                print(f"  {fmt.upper()}: skipped (not converted)")
            continue

        decoder = DECODERS[fmt]
        data_mmap = np.memmap(data_path, dtype=np.uint8, mode='r')
        fmt_meta = np.load(fmt_meta_path, mmap_mode='r')

        def run_epoch(
            pbar=None, d=decoder, dm=data_mmap, fm=fmt_meta,
            indices=all_indices, bs=batch_size,
        ):
            # Shuffle each epoch for realistic access pattern
            rng = np.random.RandomState(None)
            shuffled = rng.permutation(indices)
            total = 0
            for start in range(0, len(shuffled), bs):
                batch_idx = shuffled[start:start + bs]
                for idx in batch_idx:
                    ptr = int(fm[idx]['data_ptr'])
                    size = int(fm[idx]['data_size'])
                    _ = d(bytes(dm[ptr:ptr + size]))
                total += len(batch_idx)
                if pbar:
                    pbar.update(len(batch_idx))
            return total

        if verbose:
            print(f"  {fmt.upper()} ({total_samples:,} images, {total_samples // batch_size} batches):")

        epoch_results, avg_rate = _run_epochs(
            run_epoch, num_warmup, num_timed, fmt.upper(), verbose,
            total_samples=total_samples,
        )

        results[fmt] = {
            'throughput': avg_rate,
            'epochs': epoch_results,
        }

    # Comparison
    if verbose and 'jpeg' in results:
        jpeg_tp = results['jpeg']['throughput']
        print(f"\n  {'Format':<8} {'Throughput':>12} {'vs JPEG':>10}")
        print(f"  {'-'*32}")
        for fmt, info in results.items():
            ratio = info['throughput'] / jpeg_tp
            print(f"  {fmt.upper():<8} {info['throughput']:>8.0f}/s    {ratio:>6.2f}x")

    return results


def benchmark_numba_raw_ceiling(
    cache_dir: Path,
    image_field: str,
    batch_size: int = 256,
    num_warmup: int = 1,
    num_timed: int = 2,
    verbose: bool = True,
) -> dict | None:
    """Benchmark raw RGB batch load via Numba (no decode — pure memcpy ceiling).

    Iterates the FULL dataset each epoch using Numba prange batch loader.
    1 warmup epoch + 2 timed epochs.
    """
    raw_data_path = cache_dir / f"{image_field}_raw.bin"
    raw_meta_path = cache_dir / f"{image_field}_raw.meta.npy"

    if not raw_data_path.exists():
        if verbose:
            print("\n  Raw RGB ceiling: skipped (not converted)")
        return None

    if verbose:
        print(f"\n{'='*60}")
        print("RAW RGB CEILING (Numba batch memcpy, no decode)")
        print(f"{'='*60}")
        print(f"(batch_size={batch_size}, {num_warmup} warmup + {num_timed} timed, full dataset)\n")

    from numba import njit, prange

    data_mmap = np.memmap(raw_data_path, dtype=np.uint8, mode='r')
    metadata = np.load(raw_meta_path, mmap_mode='r')
    data_array = np.asarray(data_mmap)
    metadata_c = np.ascontiguousarray(metadata)

    total_samples = len(metadata)
    max_size = int(np.max(metadata['data_size']))
    all_indices = np.arange(total_samples, dtype=np.int64)

    @njit(nogil=True, parallel=True, cache=False)
    def load_batch(indices, meta, data, dest, sizes):
        n = len(indices)
        for i in prange(n):
            sid = indices[i]
            ptr = meta[sid]['data_ptr']
            sz = meta[sid]['data_size']
            dest[i, :sz] = data[ptr:ptr + sz]
            sizes[i] = sz

    dest = np.zeros((batch_size, max_size), dtype=np.uint8)
    sizes = np.zeros(batch_size, dtype=np.uint64)

    # JIT warmup (compile only)
    warmup_idx = all_indices[:batch_size].copy()
    load_batch(warmup_idx, metadata_c, data_array, dest, sizes)

    def run_epoch(pbar=None):
        rng = np.random.RandomState(None)
        shuffled = rng.permutation(all_indices)
        total = 0
        for start in range(0, len(shuffled), batch_size):
            batch_idx = shuffled[start:start + batch_size]
            if len(batch_idx) < batch_size:
                # Pad last batch
                padded = np.zeros(batch_size, dtype=np.int64)
                padded[:len(batch_idx)] = batch_idx
                load_batch(padded, metadata_c, data_array, dest, sizes)
            else:
                load_batch(batch_idx, metadata_c, data_array, dest, sizes)
            total += len(batch_idx)
            if pbar:
                pbar.update(len(batch_idx))
        return total

    if verbose:
        print(f"  Raw RGB Numba ({total_samples:,} images, {total_samples // batch_size} batches):")

    epoch_results, avg_rate = _run_epochs(
        run_epoch, num_warmup, num_timed, "RAW NUMBA", verbose,
        total_samples=total_samples,
    )

    if verbose:
        print(f"  (This is the ceiling — no decode, just memcpy from mmap)")

    return {
        'throughput': avg_rate,
        'epochs': epoch_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark image format decode throughput")
    parser.add_argument("--dataset", type=str, default=LITDATA_VAL_PATH)
    parser.add_argument("--formats", nargs="+",
                        default=["jpeg", "qoi", "raw"])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=1, help="Number of warmup epochs")
    parser.add_argument("--epochs", type=int, default=2, help="Number of timed epochs")
    parser.add_argument("--image-field", type=str, default="image")
    parser.add_argument("--skip-single", action="store_true")
    parser.add_argument("--skip-batch", action="store_true")
    parser.add_argument("--skip-ceiling", action="store_true")
    args = parser.parse_args()

    # Load dataset to get cache dir
    from slipstream import SlipstreamDataset, OptimizedCache

    print("Loading dataset...")
    dataset = SlipstreamDataset(remote_dir=args.dataset, decode_images=False)

    if OptimizedCache.exists(dataset.cache_path):
        cache = OptimizedCache.load(dataset.cache_path)
    else:
        cache = OptimizedCache.build(dataset)

    cache_dir = cache.cache_dir
    print(f"Cache: {cache_dir}")
    print(f"Samples: {cache.num_samples:,}")

    # Check which formats are available
    available = []
    for fmt in args.formats:
        data_path, _ = _get_format_paths(cache_dir, args.image_field, fmt)
        if data_path.exists():
            available.append(fmt)
        else:
            print(f"  {fmt.upper()}: not found (run convert_formats.py first)")
    print(f"Available formats: {[f.upper() for f in available]}")

    all_results = {}

    if not args.skip_single:
        all_results['single'] = benchmark_single_image_decode(
            cache_dir, args.image_field, available,
            num_warmup=args.warmup, num_timed=args.epochs,
        )

    if not args.skip_batch:
        all_results['batch'] = benchmark_batch_decode(
            cache_dir, args.image_field, available,
            batch_size=args.batch_size,
            num_warmup=args.warmup, num_timed=args.epochs,
        )

    if not args.skip_ceiling:
        ceiling = benchmark_numba_raw_ceiling(
            cache_dir, args.image_field,
            batch_size=args.batch_size,
            num_warmup=args.warmup, num_timed=args.epochs,
        )
        if ceiling:
            all_results['ceiling'] = ceiling

    # Final summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    if 'single' in all_results and 'jpeg' in all_results['single']:
        jpeg_us = all_results['single']['jpeg']['per_image_us']
        print(f"\nSingle-image decode (lower is better):")
        for fmt, info in all_results['single'].items():
            speedup = jpeg_us / info['per_image_us']
            bar = "█" * int(speedup * 10)
            print(f"  {fmt.upper():<8} {info['per_image_us']:>7.1f} us  {speedup:>5.2f}x  {bar}")

    if 'batch' in all_results and 'jpeg' in all_results['batch']:
        jpeg_tp = all_results['batch']['jpeg']['throughput']
        print(f"\nBatch decode throughput (higher is better):")
        for fmt, info in all_results['batch'].items():
            speedup = info['throughput'] / jpeg_tp
            bar = "█" * int(speedup * 10)
            print(f"  {fmt.upper():<8} {info['throughput']:>7.0f}/s  {speedup:>5.2f}x  {bar}")

    if 'ceiling' in all_results:
        print(f"\nRaw RGB ceiling (Numba memcpy, no decode):")
        print(f"  {all_results['ceiling']['throughput']:>7.0f}/s")

    print(f"\nConclusion guidance:")
    print(f"  - If raw ceiling ≈ JPEG throughput → decode is NOT the bottleneck")
    print(f"  - If QOI/JXL >> JPEG in single-image → format wins at C level too")
    print(f"  - Consider storage cost: check convert_formats.py output for sizes")


if __name__ == "__main__":
    main()
