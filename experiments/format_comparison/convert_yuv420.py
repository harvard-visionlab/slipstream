#!/usr/bin/env python3
"""Convert a slip cache's JPEG images to raw YUV420P format for benchmarking.

YUV420P (I420) layout per image:
  Y plane: H * W bytes
  U plane: (H/2) * (W/2) bytes
  V plane: (H/2) * (W/2) bytes
  Total: H * W * 3/2 bytes (50% of RGB, ~1.95x JPEG for this dataset)

Images with odd dimensions are padded to even (required for 4:2:0 subsampling).

Creates sibling files in the slip cache:
    .slipstream/
    ├── image.bin           # original JPEG
    ├── image.meta.npy      # original metadata
    ├── image_yuv420.bin    # YUV420P transcoded
    └── image_yuv420.meta.npy  # YUV420 metadata

Usage:
    uv run python experiments/format_comparison/convert_yuv420.py
    uv run python experiments/format_comparison/convert_yuv420.py --max-samples 1000
"""

from __future__ import annotations

import argparse
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


def rgb_to_yuv420p(rgb: np.ndarray) -> tuple[bytes, int, int]:
    """Convert RGB [H, W, 3] uint8 to YUV420P bytes.

    Returns (yuv_bytes, padded_height, padded_width).
    Dimensions are padded to even if needed.
    """
    h, w = rgb.shape[:2]

    # Pad to even dimensions if needed
    pad_h = h + (h % 2)
    pad_w = w + (w % 2)
    if pad_h != h or pad_w != w:
        padded = np.zeros((pad_h, pad_w, 3), dtype=np.uint8)
        padded[:h, :w, :] = rgb
        # Replicate edge pixels for padding
        if pad_h > h:
            padded[h, :w, :] = rgb[h-1, :, :]
        if pad_w > w:
            padded[:h, w, :] = rgb[:, w-1, :]
        if pad_h > h and pad_w > w:
            padded[h, w, :] = rgb[h-1, w-1, :]
        rgb = padded

    # BT.601 RGB→YUV conversion (matches our C decoder)
    r = rgb[:, :, 0].astype(np.float32)
    g = rgb[:, :, 1].astype(np.float32)
    b = rgb[:, :, 2].astype(np.float32)

    y = (0.299 * r + 0.587 * g + 0.114 * b)
    u = (-0.168736 * r - 0.331264 * g + 0.5 * b + 128.0)
    v = (0.5 * r - 0.418688 * g - 0.081312 * b + 128.0)

    y = np.clip(y, 0, 255).astype(np.uint8)
    u = np.clip(u, 0, 255).astype(np.uint8)
    v = np.clip(v, 0, 255).astype(np.uint8)

    # Subsample chroma (average 2x2 blocks)
    u_sub = u.reshape(pad_h // 2, 2, pad_w // 2, 2).mean(axis=(1, 3))
    v_sub = v.reshape(pad_h // 2, 2, pad_w // 2, 2).mean(axis=(1, 3))
    u_sub = np.clip(u_sub, 0, 255).astype(np.uint8)
    v_sub = np.clip(v_sub, 0, 255).astype(np.uint8)

    # Pack: Y + U + V contiguous
    yuv_bytes = y.tobytes() + u_sub.tobytes() + v_sub.tobytes()
    return yuv_bytes, pad_h, pad_w


def convert_cache_yuv420(
    cache_dir: Path,
    image_field: str = 'image',
    max_samples: int | None = None,
    verbose: bool = True,
) -> dict:
    """Convert JPEG images in a slip cache to YUV420P format."""
    data_path = cache_dir / f"{image_field}.bin"
    meta_path = cache_dir / f"{image_field}.meta.npy"

    if not data_path.exists():
        raise FileNotFoundError(f"No image data at {data_path}")

    data_mmap = np.memmap(data_path, dtype=np.uint8, mode='r')
    metadata = np.load(meta_path, mmap_mode='r')
    num_samples = len(metadata)

    if max_samples is not None:
        num_samples = min(num_samples, max_samples)

    out_data_path = cache_dir / f"{image_field}_yuv420.bin"
    out_meta_path = cache_dir / f"{image_field}_yuv420.meta.npy"

    # Check if already exists
    if out_data_path.exists() and out_meta_path.exists():
        existing_meta = np.load(out_meta_path, mmap_mode='r')
        if len(existing_meta) == num_samples:
            total_bytes = sum(int(existing_meta[i]['data_size']) for i in range(num_samples))
            if verbose:
                print(f"YUV420: already exists — {total_bytes / 1e9:.2f} GB ({num_samples:,} samples)")
            return {'total_bytes': total_bytes, 'num_samples': num_samples, 'skipped': True}

    if verbose:
        jpeg_total = sum(int(metadata[i]['data_size']) for i in range(num_samples))
        print(f"Source: {num_samples:,} JPEG images, {jpeg_total / 1e9:.2f} GB")
        print(f"Converting to YUV420P...")

    out_metadata = np.zeros(num_samples, dtype=VARIABLE_METADATA_DTYPE)
    current_ptr = 0
    total_bytes = 0

    t0 = time.perf_counter()

    with open(out_data_path, 'wb') as f:
        iterator = range(num_samples)
        if verbose:
            iterator = tqdm(iterator, desc="  YUV420")

        for i in iterator:
            ptr = int(metadata[i]['data_ptr'])
            size = int(metadata[i]['data_size'])
            jpeg_bytes = bytes(data_mmap[ptr:ptr + size])

            # Decode JPEG → RGB
            rgb = _turbo.decode(jpeg_bytes, pixel_format=0)  # TJPF_RGB

            # Convert to YUV420P
            yuv_bytes, pad_h, pad_w = rgb_to_yuv420p(rgb)
            enc_size = len(yuv_bytes)

            f.write(yuv_bytes)

            out_metadata[i]['data_ptr'] = current_ptr
            out_metadata[i]['data_size'] = enc_size
            out_metadata[i]['height'] = pad_h
            out_metadata[i]['width'] = pad_w

            current_ptr += enc_size
            total_bytes += enc_size

    elapsed = time.perf_counter() - t0

    np.save(out_meta_path, out_metadata)

    if verbose:
        jpeg_total = sum(int(metadata[i]['data_size']) for i in range(num_samples))
        ratio = total_bytes / jpeg_total
        print(f"  Size: {total_bytes / 1e9:.2f} GB ({ratio:.2f}x JPEG)")
        print(f"  Time: {elapsed:.1f}s ({num_samples / elapsed:.0f} images/sec)")

    return {
        'total_bytes': total_bytes,
        'num_samples': num_samples,
        'elapsed_sec': elapsed,
        'skipped': False,
    }


def main():
    parser = argparse.ArgumentParser(description="Convert slip cache JPEG → YUV420P")
    parser.add_argument("--dataset", type=str, default=LITDATA_VAL_PATH)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--image-field", type=str, default="image")
    args = parser.parse_args()

    from slipstream import SlipstreamDataset
    from slipstream.cache import OptimizedCache

    print("Loading dataset...")
    dataset = SlipstreamDataset(remote_dir=args.dataset, decode_images=False)

    if OptimizedCache.exists(dataset.cache_path):
        cache = OptimizedCache.load(dataset.cache_path)
    else:
        cache = OptimizedCache.build(dataset)

    print(f"Cache: {cache.cache_dir}")
    print(f"Samples: {cache.num_samples:,}")

    convert_cache_yuv420(
        cache.cache_dir,
        image_field=args.image_field,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
