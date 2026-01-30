#!/usr/bin/env python3
"""Convert a slip cache's JPEG images to alternative formats for benchmarking.

Creates sibling files alongside the original cache:
    .slipstream/
    ├── image.bin           # original JPEG
    ├── image.meta.npy      # original metadata
    ├── image_qoi.bin       # QOI transcoded
    ├── image_qoi.meta.npy  # QOI metadata
    ├── image_jxl.bin       # JPEG XL transcoded
    ├── image_jxl.meta.npy  # JXL metadata
    ├── image_raw.bin       # Raw RGB (decoded, no compression)
    ├── image_raw.meta.npy  # Raw metadata (fixed size per image)
    └── ...

Usage:
    uv sync --group experiment
    uv run python experiments/format_comparison/convert_formats.py
    uv run python experiments/format_comparison/convert_formats.py --formats qoi jxl raw
    uv run python experiments/format_comparison/convert_formats.py --max-samples 1000
"""

from __future__ import annotations

import argparse
import struct
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm
from turbojpeg import TurboJPEG

# Lazy imports for optional format libraries
_turbo = TurboJPEG()

LITDATA_VAL_PATH = "s3://visionlab-datasets/imagenet1k/pre-processed/s256-l512-jpgbytes-q100-streaming/val/"

# Metadata dtype matches slipstream cache format
VARIABLE_METADATA_DTYPE = np.dtype([
    ('data_ptr', '<u8'),
    ('data_size', '<u8'),
    ('height', '<u4'),
    ('width', '<u4'),
])


def decode_jpeg(jpeg_bytes: bytes) -> np.ndarray:
    """Decode JPEG to RGB numpy array [H, W, 3]."""
    return _turbo.decode(jpeg_bytes, pixel_format=0)  # TJPF_RGB


def encode_qoi(rgb: np.ndarray) -> bytes:
    """Encode RGB array to QOI format."""
    import qoi
    return qoi.encode(rgb)


def encode_raw(rgb: np.ndarray) -> bytes:
    """Encode RGB array as raw bytes with a minimal 8-byte header (height, width)."""
    h, w = rgb.shape[:2]
    header = struct.pack('<HH', h, w)  # 4 bytes: uint16 height, uint16 width
    return header + rgb.tobytes()


ENCODERS = {
    'qoi': encode_qoi,
    'raw': encode_raw,
}


def convert_cache(
    cache_dir: Path,
    formats: list[str],
    image_field: str = 'image',
    max_samples: int | None = None,
    verbose: bool = True,
) -> dict[str, dict]:
    """Convert JPEG images in a slip cache to alternative formats.

    Args:
        cache_dir: Path to the .slipstream directory
        formats: List of formats to convert to ('qoi', 'jxl', 'raw')
        image_field: Name of the image field
        max_samples: Limit conversion to first N samples (for testing)
        verbose: Print progress

    Returns:
        Dict of format -> {total_bytes, num_samples, avg_bytes, elapsed_sec}
    """
    # Load original JPEG data
    data_path = cache_dir / f"{image_field}.bin"
    meta_path = cache_dir / f"{image_field}.meta.npy"

    if not data_path.exists():
        raise FileNotFoundError(f"No image data at {data_path}")

    data_mmap = np.memmap(data_path, dtype=np.uint8, mode='r')
    metadata = np.load(meta_path, mmap_mode='r')
    num_samples = len(metadata)

    if max_samples is not None:
        num_samples = min(num_samples, max_samples)

    if verbose:
        jpeg_total = sum(int(metadata[i]['data_size']) for i in range(num_samples))
        print(f"Source: {num_samples:,} JPEG images, {jpeg_total / 1e9:.2f} GB")

    results = {}

    for fmt in formats:
        if fmt not in ENCODERS:
            print(f"  Skipping unknown format: {fmt}")
            continue

        encoder = ENCODERS[fmt]
        out_data_path = cache_dir / f"{image_field}_{fmt}.bin"
        out_meta_path = cache_dir / f"{image_field}_{fmt}.meta.npy"

        if out_data_path.exists() and out_meta_path.exists():
            # Already converted — load stats
            existing_meta = np.load(out_meta_path, mmap_mode='r')
            existing_n = len(existing_meta)
            if existing_n == num_samples:
                total_bytes = sum(int(existing_meta[i]['data_size']) for i in range(existing_n))
                if verbose:
                    print(f"  {fmt.upper()}: already exists — {total_bytes / 1e9:.2f} GB "
                          f"({existing_n:,} samples)")
                results[fmt] = {
                    'total_bytes': total_bytes,
                    'num_samples': existing_n,
                    'avg_bytes': total_bytes / existing_n,
                    'elapsed_sec': 0.0,
                    'skipped': True,
                }
                continue

        if verbose:
            print(f"\n  Converting to {fmt.upper()}...")

        out_metadata = np.zeros(num_samples, dtype=VARIABLE_METADATA_DTYPE)
        current_ptr = 0
        total_bytes = 0
        max_size = 0

        t0 = time.perf_counter()

        with open(out_data_path, 'wb') as f:
            iterator = range(num_samples)
            if verbose:
                iterator = tqdm(iterator, desc=f"    {fmt.upper()}")

            for i in iterator:
                # Read JPEG from mmap
                ptr = int(metadata[i]['data_ptr'])
                size = int(metadata[i]['data_size'])
                jpeg_bytes = bytes(data_mmap[ptr:ptr + size])

                # Decode JPEG → RGB
                rgb = decode_jpeg(jpeg_bytes)
                h, w = rgb.shape[:2]

                # Encode to target format
                encoded = encoder(rgb)
                enc_size = len(encoded)

                # Write
                f.write(encoded)

                # Record metadata
                out_metadata[i]['data_ptr'] = current_ptr
                out_metadata[i]['data_size'] = enc_size
                out_metadata[i]['height'] = h
                out_metadata[i]['width'] = w

                current_ptr += enc_size
                total_bytes += enc_size
                max_size = max(max_size, enc_size)

        elapsed = time.perf_counter() - t0

        # Save metadata
        np.save(out_meta_path, out_metadata)

        results[fmt] = {
            'total_bytes': total_bytes,
            'num_samples': num_samples,
            'avg_bytes': total_bytes / num_samples,
            'max_size': max_size,
            'elapsed_sec': elapsed,
            'skipped': False,
        }

        if verbose:
            print(f"    Size: {total_bytes / 1e9:.2f} GB "
                  f"(avg {total_bytes / num_samples / 1024:.1f} KB/image, "
                  f"max {max_size / 1024:.1f} KB)")
            print(f"    Time: {elapsed:.1f}s "
                  f"({num_samples / elapsed:.0f} images/sec)")

    # Print comparison table
    if verbose and results:
        jpeg_total = sum(int(metadata[i]['data_size']) for i in range(num_samples))
        jpeg_avg = jpeg_total / num_samples

        print(f"\n{'Format':<10} {'Total':>10} {'Avg/img':>10} {'vs JPEG':>10}")
        print("-" * 42)
        print(f"{'JPEG':<10} {jpeg_total / 1e9:>9.2f}G {jpeg_avg / 1024:>8.1f}KB {'1.00x':>10}")

        for fmt, info in results.items():
            ratio = info['total_bytes'] / jpeg_total
            print(f"{fmt.upper():<10} {info['total_bytes'] / 1e9:>9.2f}G "
                  f"{info['avg_bytes'] / 1024:>8.1f}KB "
                  f"{ratio:>9.2f}x")

    return results


def main():
    parser = argparse.ArgumentParser(description="Convert slip cache images to alternative formats")
    parser.add_argument("--dataset", type=str, default=LITDATA_VAL_PATH,
                        help="Dataset path (S3 or local)")
    parser.add_argument("--formats", nargs="+", default=["qoi", "raw"],
                        help="Formats to convert to")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit to first N samples")
    parser.add_argument("--image-field", type=str, default="image",
                        help="Image field name")
    args = parser.parse_args()

    # Load dataset to ensure cache exists
    from slipstream import SlipstreamDataset, OptimizedCache

    print("Loading dataset...")
    dataset = SlipstreamDataset(
        remote_dir=args.dataset,
        decode_images=False,
    )

    # Ensure slip cache exists
    if OptimizedCache.exists(dataset.cache_path):
        cache = OptimizedCache.load(dataset.cache_path)
    else:
        cache = OptimizedCache.build(dataset)

    print(f"Cache: {cache.cache_dir}")
    print(f"Samples: {cache.num_samples:,}")

    results = convert_cache(
        cache.cache_dir,
        formats=args.formats,
        image_field=args.image_field,
        max_samples=args.max_samples,
    )

    return results


if __name__ == "__main__":
    main()
