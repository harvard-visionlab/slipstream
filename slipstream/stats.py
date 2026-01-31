"""Dataset statistics computation.

Computes per-channel RGB normalization stats (mean, std) from a slip cache.

For JPEG images, uses PyTurboJPEG directly (accurate DCT, matches PIL/torchvision
exactly) rather than the C extension's FASTDCT path. Stats is a one-time operation
where accuracy matters more than peak throughput.

Usage:
    from slipstream import compute_normalization_stats

    cache = OptimizedCache.load(cache_dir)
    stats = compute_normalization_stats(cache)
    print(stats)
    # {'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225)}
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
from tqdm import tqdm

from slipstream.cache import OptimizedCache

__all__ = ["compute_normalization_stats"]


def _decode_jpeg_accurate(jpeg_bytes: bytes, turbo: Any) -> np.ndarray:
    """Decode JPEG using PyTurboJPEG with accurate DCT (matches PIL exactly)."""
    return turbo.decode(jpeg_bytes, pixel_format=0)  # TJPF_RGB


def compute_normalization_stats(
    cache: OptimizedCache,
    image_field: str = "image",
    image_format: str = "jpeg",
    num_samples: int | None = None,
    batch_size: int = 256,
    num_threads: int = 0,
    verbose: bool = True,
) -> dict[str, tuple[float, float, float]]:
    """Compute per-channel RGB normalization statistics from a slip cache.

    For JPEG, uses PyTurboJPEG with accurate DCT (matches PIL/torchvision
    exactly). For YUV420, uses the YUV420NumbaBatchDecoder. Accumulation
    uses vectorized numpy sum/sum_sq (stable for bounded [0,255] values).

    Args:
        cache: An OptimizedCache instance with an image field.
        image_field: Name of the image field in the cache.
        image_format: ``"jpeg"`` or ``"yuv420"`` — selects which cache
            storage and decoder to use.
        num_samples: Number of samples to use. ``None`` = all samples.
        batch_size: Decode batch size.
        num_threads: Decoder threads. 0 = auto. For JPEG, controls the
            ThreadPoolExecutor size for parallel TurboJPEG decoding.
        verbose: Print progress.

    Returns:
        Dict with ``'mean'`` and ``'std'`` keys, each a tuple of 3 floats
        (R, G, B) in [0, 1] range.
    """
    from slipstream.cache import ImageBytesStorage, load_yuv420_cache

    # Resolve storage
    if image_format == "yuv420":
        storage = load_yuv420_cache(cache.cache_dir, image_field)
        if storage is None:
            raise FileNotFoundError(
                f"No YUV420 cache found for field '{image_field}' in {cache.cache_dir}. "
                "Build it first with SlipstreamLoader(dataset, image_format='yuv420')."
            )
    else:
        storage = cache.fields.get(image_field)
        if storage is None or not isinstance(storage, ImageBytesStorage):
            raise KeyError(
                f"No image field '{image_field}' in cache. "
                f"Available fields: {list(cache.fields.keys())}"
            )

    total = min(num_samples, cache.num_samples) if num_samples is not None else cache.num_samples
    indices = np.arange(total, dtype=np.int64)

    # Accumulate sum and sum_sq per channel (float64 for precision)
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_sum_sq = np.zeros(3, dtype=np.float64)
    pixel_count = np.int64(0)

    num_batches = (total + batch_size - 1) // batch_size
    iterator = range(num_batches)
    if verbose:
        iterator = tqdm(iterator, desc="Computing stats", total=num_batches)

    if image_format == "jpeg":
        # Use PyTurboJPEG with accurate DCT (matches PIL/torchvision exactly)
        from turbojpeg import TurboJPEG

        if num_threads < 1:
            from slipstream.decoders.numba_decoder import _available_cpus
            num_threads = _available_cpus()

        # Thread-local TurboJPEG handles (each thread gets its own)
        import threading
        _thread_local = threading.local()

        def _get_turbo() -> TurboJPEG:
            if not hasattr(_thread_local, 'turbo'):
                _thread_local.turbo = TurboJPEG()
            return _thread_local.turbo

        def _decode_one(jpeg_bytes: bytes) -> np.ndarray:
            return _get_turbo().decode(jpeg_bytes, pixel_format=0)

        with ThreadPoolExecutor(max_workers=num_threads) as pool:
            for batch_idx in iterator:
                start = batch_idx * batch_size
                end = min(start + batch_size, total)
                batch_indices = indices[start:end]

                batch_data = storage.load_batch(batch_indices)
                actual_batch = len(batch_indices)

                # Extract JPEG bytes for each image
                jpeg_list = []
                for i in range(actual_batch):
                    size = int(batch_data['sizes'][i])
                    jpeg_list.append(bytes(batch_data['data'][i, :size]))

                # Parallel decode with accurate DCT
                images = list(pool.map(_decode_one, jpeg_list))

                # Vectorized accumulation
                for img in images:
                    pixels = img.reshape(-1, 3).astype(np.float64) / 255.0
                    channel_sum += pixels.sum(axis=0)
                    channel_sum_sq += (pixels * pixels).sum(axis=0)
                    pixel_count += pixels.shape[0]

    else:
        # YUV420: use the Numba decoder (no DCT involved, exact conversion)
        from slipstream.decoders.yuv420_decoder import YUV420NumbaBatchDecoder
        decoder = YUV420NumbaBatchDecoder(num_threads=num_threads)

        for batch_idx in iterator:
            start = batch_idx * batch_size
            end = min(start + batch_size, total)
            batch_indices = indices[start:end]

            batch_data = storage.load_batch(batch_indices)
            images = decoder.decode_batch(
                batch_data['data'], batch_data['sizes'],
                batch_data['heights'], batch_data['widths'],
            )

            for img in images:
                pixels = img.reshape(-1, 3).astype(np.float64) / 255.0
                channel_sum += pixels.sum(axis=0)
                channel_sum_sq += (pixels * pixels).sum(axis=0)
                pixel_count += pixels.shape[0]

        decoder.shutdown()

    if pixel_count < 1:
        raise ValueError("No pixels found in dataset")

    mean = channel_sum / pixel_count
    variance = (channel_sum_sq / pixel_count) - (mean * mean)
    std = np.sqrt(np.maximum(variance, 0.0))

    result = {
        'mean': (float(mean[0]), float(mean[1]), float(mean[2])),
        'std': (float(std[0]), float(std[1]), float(std[2])),
    }

    if verbose:
        print(f"Normalization stats ({total:,} samples, {pixel_count:,} pixels):")
        print(f"  mean: ({mean[0]:.6f}, {mean[1]:.6f}, {mean[2]:.6f})")
        print(f"  std:  ({std[0]:.6f}, {std[1]:.6f}, {std[2]:.6f})")

    return result
