"""Numba-optimized JPEG decoder with crop-during-decode support.

Based on litdata-mmap's numba_crop_decoder.py pattern:
1. ThreadPoolExecutor for parallel decode (PyTurboJPEG releases GIL)
2. Numba JIT for vectorized crop parameter generation
3. TurboJPEG scaled decode → numpy crop → PyTorch resize

Usage:
    decoder = NumbaBatchDecoder(num_threads=12)

    # RandomResizedCrop during decode (for training)
    images = decoder.decode_batch_random_crop(
        data, sizes, heights, widths,
        target_size=224,
    )

    # CenterCrop during decode (for validation)
    images = decoder.decode_batch_center_crop(
        data, sizes, heights, widths,
        crop_size=224,
    )
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from numba import njit, prange
from numpy.typing import NDArray

if TYPE_CHECKING:
    pass

__all__ = [
    "NumbaBatchDecoder",
    "check_numba_decoder_available",
]

# Thread-local TurboJPEG handles
_thread_local = threading.local()


def _get_turbo_jpeg():
    """Get thread-local TurboJPEG instance."""
    if not hasattr(_thread_local, "jpeg"):
        from turbojpeg import TurboJPEG
        _thread_local.jpeg = TurboJPEG()
    return _thread_local.jpeg


def check_numba_decoder_available() -> bool:
    """Check if the Numba batch decoder is available."""
    try:
        from turbojpeg import TurboJPEG
        TurboJPEG()
        return True
    except Exception:
        return False


# =============================================================================
# Numba JIT-compiled crop parameter generation
# =============================================================================

@njit(cache=True, fastmath=True)
def _generate_random_crop_params_batch(
    widths: NDArray[np.int32],
    heights: NDArray[np.int32],
    scale_min: float,
    scale_max: float,
    log_ratio_min: float,
    log_ratio_max: float,
    seed: int,
) -> NDArray[np.int32]:
    """Generate random crop parameters for a batch using Numba.

    Returns:
        Array of shape [B, 4] with (x, y, crop_w, crop_h) for each image
    """
    np.random.seed(seed)
    batch_size = len(widths)
    params = np.zeros((batch_size, 4), dtype=np.int32)

    for i in range(batch_size):
        w = widths[i]
        h = heights[i]
        area = w * h

        # Try up to 10 times to find valid crop
        found = False
        for _ in range(10):
            target_area = area * np.random.uniform(scale_min, scale_max)
            aspect_ratio = np.exp(np.random.uniform(log_ratio_min, log_ratio_max))

            crop_w = int(np.sqrt(target_area * aspect_ratio) + 0.5)
            crop_h = int(np.sqrt(target_area / aspect_ratio) + 0.5)

            if 0 < crop_w <= w and 0 < crop_h <= h:
                crop_x = np.random.randint(0, w - crop_w + 1)
                crop_y = np.random.randint(0, h - crop_h + 1)

                params[i, 0] = crop_x
                params[i, 1] = crop_y
                params[i, 2] = crop_w
                params[i, 3] = crop_h
                found = True
                break

        if not found:
            # Fallback: center crop to min dimension
            crop_size = min(w, h)
            crop_x = (w - crop_size) // 2
            crop_y = (h - crop_size) // 2
            params[i, 0] = crop_x
            params[i, 1] = crop_y
            params[i, 2] = crop_size
            params[i, 3] = crop_size

    return params


@njit(cache=True, parallel=True)
def _generate_center_crop_params_batch(
    widths: NDArray[np.int32],
    heights: NDArray[np.int32],
    crop_size: int,
) -> NDArray[np.int32]:
    """Generate center crop parameters for a batch using Numba parallel.

    Returns:
        Array of shape [B, 4] with (x, y, crop_w, crop_h) for each image
    """
    batch_size = len(widths)
    params = np.zeros((batch_size, 4), dtype=np.int32)

    for i in prange(batch_size):
        w = widths[i]
        h = heights[i]

        # Center crop to min(crop_size, min_dim)
        actual_crop = min(crop_size, w, h)
        crop_x = (w - actual_crop) // 2
        crop_y = (h - actual_crop) // 2

        params[i, 0] = crop_x
        params[i, 1] = crop_y
        params[i, 2] = actual_crop
        params[i, 3] = actual_crop

    return params


# =============================================================================
# Decode helper functions
# =============================================================================

def _decode_one_full(jpeg_bytes: bytes) -> NDArray[np.uint8]:
    """Decode single JPEG to full size."""
    from turbojpeg import TJPF_RGB
    jpeg = _get_turbo_jpeg()
    return jpeg.decode(jpeg_bytes, pixel_format=TJPF_RGB)


def _decode_one_with_crop(
    jpeg_bytes: bytes,
    crop_x: int,
    crop_y: int,
    crop_w: int,
    crop_h: int,
) -> NDArray[np.uint8]:
    """Decode single JPEG and crop in numpy (fast)."""
    from turbojpeg import TJPF_RGB
    jpeg = _get_turbo_jpeg()
    full_image = jpeg.decode(jpeg_bytes, pixel_format=TJPF_RGB)
    return full_image[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w].copy()


# =============================================================================
# Main decoder class
# =============================================================================

class NumbaBatchDecoder:
    """High-performance JPEG decoder using ThreadPoolExecutor + Numba.

    Based on litdata-mmap's numba_crop_decoder.py:
    - ThreadPoolExecutor for parallel decode (TurboJPEG releases GIL)
    - Numba JIT for vectorized crop parameter generation
    - Decode full → numpy crop → PyTorch resize

    Example:
        decoder = NumbaBatchDecoder(num_threads=8)

        # RandomResizedCrop to 224x224
        result = decoder.decode_batch_random_crop(
            jpeg_data, sizes, heights, widths,
            target_size=224,
        )
    """

    def __init__(self, num_threads: int = 0) -> None:
        """Initialize the decoder.

        Args:
            num_threads: Number of parallel decode threads. 0 = auto (cpu_count)
        """
        from multiprocessing import cpu_count
        if num_threads < 1:
            num_threads = cpu_count()
        self.num_threads = num_threads
        self._executor: ThreadPoolExecutor | None = None
        self._seed_counter = 0

    def _ensure_executor(self) -> None:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=self.num_threads)

    def shutdown(self) -> None:
        """Shutdown the thread pool."""
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None

    def decode_batch(
        self,
        data: NDArray[np.uint8],
        sizes: NDArray,
        heights: NDArray | None = None,
        widths: NDArray | None = None,
        destination: NDArray[np.uint8] | None = None,
    ) -> list[NDArray[np.uint8]]:
        """Decode batch of JPEGs to full size.

        Args:
            data: Padded JPEG data [B, max_size] uint8
            sizes: Actual JPEG sizes [B]
            heights: Image heights [B] (unused, for API compatibility)
            widths: Image widths [B] (unused, for API compatibility)
            destination: Unused (for API compatibility)

        Returns:
            List of decoded RGB images [H, W, 3]
        """
        self._ensure_executor()
        assert self._executor is not None

        batch_size = len(sizes)

        # Extract JPEG bytes
        jpeg_bytes_list = [
            bytes(data[i, :int(sizes[i])])
            for i in range(batch_size)
        ]

        # Parallel decode
        results = list(self._executor.map(_decode_one_full, jpeg_bytes_list))
        return results

    def decode_batch_center_crop(
        self,
        data: NDArray[np.uint8],
        sizes: NDArray,
        heights: NDArray,
        widths: NDArray,
        crop_size: int = 224,
        destination: NDArray[np.uint8] | None = None,
    ) -> NDArray[np.uint8]:
        """Decode batch with CenterCrop.

        Args:
            data: Padded JPEG data [B, max_size] uint8
            sizes: Actual JPEG sizes [B]
            heights: Image heights [B]
            widths: Image widths [B]
            crop_size: Size of center crop
            destination: Unused (for API compatibility)

        Returns:
            Array [B, crop_size, crop_size, 3] uint8
        """
        self._ensure_executor()
        assert self._executor is not None

        batch_size = len(sizes)

        # Vectorized crop param generation (Numba parallel)
        crop_params = _generate_center_crop_params_batch(
            widths.astype(np.int32),
            heights.astype(np.int32),
            crop_size,
        )

        # Prepare decode args
        decode_args = []
        for i in range(batch_size):
            jpeg_bytes = bytes(data[i, :int(sizes[i])])
            crop_x, crop_y, crop_w, crop_h = crop_params[i]
            decode_args.append((
                jpeg_bytes,
                int(crop_x),
                int(crop_y),
                int(crop_w),
                int(crop_h),
            ))

        # Parallel decode with crop
        results = list(self._executor.map(
            lambda args: _decode_one_with_crop(*args),
            decode_args,
        ))

        # Stack into array (all same size for center crop)
        return np.stack(results, axis=0)

    def decode_batch_random_crop(
        self,
        data: NDArray[np.uint8],
        sizes: NDArray,
        heights: NDArray,
        widths: NDArray,
        target_size: int = 224,
        scale: tuple[float, float] = (0.08, 1.0),
        ratio: tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
        destination: NDArray[np.uint8] | None = None,
    ) -> torch.Tensor:
        """Decode batch with RandomResizedCrop.

        Args:
            data: Padded JPEG data [B, max_size] uint8
            sizes: Actual JPEG sizes [B]
            heights: Image heights [B]
            widths: Image widths [B]
            target_size: Final output size (square)
            scale: Scale range for random area
            ratio: Aspect ratio range
            destination: Unused (for API compatibility)

        Returns:
            Tensor [B, 3, target_size, target_size] uint8
        """
        import math
        self._ensure_executor()
        assert self._executor is not None

        batch_size = len(sizes)

        # Vectorized crop param generation (Numba JIT)
        log_ratio_min = math.log(ratio[0])
        log_ratio_max = math.log(ratio[1])
        self._seed_counter += 1

        crop_params = _generate_random_crop_params_batch(
            widths.astype(np.int32),
            heights.astype(np.int32),
            scale[0],
            scale[1],
            log_ratio_min,
            log_ratio_max,
            self._seed_counter,
        )

        # Prepare decode args
        decode_args = []
        for i in range(batch_size):
            jpeg_bytes = bytes(data[i, :int(sizes[i])])
            crop_x, crop_y, crop_w, crop_h = crop_params[i]
            decode_args.append((
                jpeg_bytes,
                int(crop_x),
                int(crop_y),
                int(crop_w),
                int(crop_h),
            ))

        # Parallel decode with crop
        cropped_images = list(self._executor.map(
            lambda args: _decode_one_with_crop(*args),
            decode_args,
        ))

        # Resize to exact target size using batched PyTorch
        # First convert to tensors and find max dimensions
        max_h = max(img.shape[0] for img in cropped_images)
        max_w = max(img.shape[1] for img in cropped_images)

        # Pad to uniform size for batched resize
        padded = torch.zeros((batch_size, 3, max_h, max_w), dtype=torch.float32)
        for i, img in enumerate(cropped_images):
            h, w = img.shape[:2]
            t = torch.from_numpy(img).permute(2, 0, 1).float()
            padded[i, :, :h, :w] = t

        # Single batched resize (much faster than per-image)
        resized = F.interpolate(
            padded,
            size=(target_size, target_size),
            mode='bilinear',
            align_corners=False,
        )

        return resized.to(torch.uint8)

    def __repr__(self) -> str:
        return f"NumbaBatchDecoder(num_threads={self.num_threads})"
