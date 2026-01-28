"""CPU JPEG decoder using PyTurboJPEG + ThreadPoolExecutor.

This module provides high-performance parallel JPEG decoding for CPU environments.
Key optimizations:

1. **Thread-local TurboJPEG handles**: Avoids lock contention between workers
2. **DCT-space cropping**: Lossless crop before decompression (~2x faster)
3. **ThreadPoolExecutor parallelism**: GIL released during libturbojpeg calls

Usage:
    from slipstream.decoders import CPUDecoder

    decoder = CPUDecoder(num_workers=8)

    # Simple batch decode
    images = decoder.decode_batch(jpeg_data, sizes)

    # With RandomResizedCrop during decode (training acceleration)
    images = decoder.decode_batch_random_crop(
        jpeg_data, sizes, heights, widths,
        target_size=224,
        scale=(0.08, 1.0),
    )

Performance:
    - Simple decode: ~3-5x improvement over single-threaded
    - With crop-during-decode: ~5-7x improvement (crop in DCT space)
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import numpy as np
import torch

from slipstream.utils.crop import (
    CropParams,
    generate_center_crop_params,
    generate_random_crop_params,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

# TurboJPEG pixel format constants
TJPF_RGB = 0

# Thread-local storage for TurboJPEG instances
_thread_local = threading.local()


def _get_thread_local_jpeg():
    """Get or create thread-local TurboJPEG instance.

    Each thread gets its own TurboJPEG handle to avoid lock contention.
    Handles are created lazily on first use.

    Returns:
        Thread-local TurboJPEG handle
    """
    if not hasattr(_thread_local, "jpeg"):
        from turbojpeg import TurboJPEG
        _thread_local.jpeg = TurboJPEG()
    return _thread_local.jpeg


def check_turbojpeg_available() -> bool:
    """Check if TurboJPEG is available.

    Returns:
        True if TurboJPEG can be initialized
    """
    try:
        from turbojpeg import TurboJPEG
        TurboJPEG()
        return True
    except Exception:
        return False


class CPUDecoder:
    """Parallel JPEG decoder using PyTurboJPEG + ThreadPoolExecutor.

    This decoder achieves high performance through:
    1. Thread-local TurboJPEG handles (no lock contention)
    2. Optional crop in DCT space (lossless, ~2x faster than post-decode crop)
    3. ThreadPoolExecutor parallelism (GIL released during libturbojpeg calls)

    Attributes:
        num_workers: Number of parallel decode workers

    Example:
        decoder = CPUDecoder(num_workers=8)

        # Decode batch of JPEGs
        images = decoder.decode_batch(data_array, sizes_array)

        # With RandomResizedCrop during decode (for training)
        images = decoder.decode_batch_random_crop(
            data_array, sizes_array, heights, widths,
            target_size=224,
            scale=(0.08, 1.0),
        )
    """

    def __init__(self, num_workers: int = 8) -> None:
        """Initialize the batch decoder.

        Args:
            num_workers: Number of parallel decode workers

        Raises:
            RuntimeError: If TurboJPEG is not available
        """
        if not check_turbojpeg_available():
            raise RuntimeError(
                "TurboJPEG not available. Install libturbojpeg:\n"
                "  macOS: brew install libjpeg-turbo\n"
                "  Ubuntu: apt-get install libturbojpeg0-dev"
            )

        self.num_workers = num_workers
        self._executor: ThreadPoolExecutor | None = None

    def _ensure_executor(self) -> ThreadPoolExecutor:
        """Ensure executor is initialized."""
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=self.num_workers)
        return self._executor

    def _decode_one(self, jpeg_bytes: bytes) -> NDArray[np.uint8]:
        """Decode a single JPEG using thread-local TurboJPEG handle.

        Args:
            jpeg_bytes: Raw JPEG bytes

        Returns:
            Decoded RGB image as numpy array [H, W, 3]
        """
        from turbojpeg import TJPF_RGB
        jpeg = _get_thread_local_jpeg()
        return jpeg.decode(jpeg_bytes, pixel_format=TJPF_RGB)

    def _decode_one_with_crop(
        self,
        jpeg_bytes: bytes,
        crop: CropParams,
    ) -> NDArray[np.uint8]:
        """Decode JPEG with crop in DCT space (lossless, fast).

        DCT-space cropping happens before decompression, making it:
        - Lossless (no quality degradation)
        - Faster (less data to decompress)

        Args:
            jpeg_bytes: Raw JPEG bytes
            crop: Crop parameters (x, y, width, height)

        Returns:
            Cropped and decoded RGB image [H, W, 3]
        """
        from turbojpeg import TJPF_RGB
        jpeg = _get_thread_local_jpeg()

        try:
            # Step 1: Lossless crop in DCT space
            # TurboJPEG adjusts coordinates to MCU boundaries automatically
            cropped_jpeg = jpeg.crop(
                jpeg_bytes,
                crop.x,
                crop.y,
                crop.width,
                crop.height,
            )

            # Step 2: Decode the cropped JPEG
            return jpeg.decode(cropped_jpeg, pixel_format=TJPF_RGB)

        except Exception:
            # Fallback: decode full image then crop
            # This happens if crop params are invalid for DCT-space crop
            full_image = jpeg.decode(jpeg_bytes, pixel_format=TJPF_RGB)
            return full_image[crop.y:crop.y + crop.height, crop.x:crop.x + crop.width]

    def _get_jpeg_dimensions(self, jpeg_bytes: bytes) -> tuple[int, int]:
        """Get JPEG dimensions without full decode.

        Args:
            jpeg_bytes: Raw JPEG bytes

        Returns:
            (width, height) tuple
        """
        jpeg = _get_thread_local_jpeg()
        width, height, _, _ = jpeg.decode_header(jpeg_bytes)
        return width, height

    def decode_batch(
        self,
        data: NDArray[np.uint8],
        sizes: NDArray[np.uint64 | np.uint32 | np.int64],
    ) -> list[NDArray[np.uint8]]:
        """Decode batch of JPEGs in parallel.

        Args:
            data: Raw JPEG data array [B, max_size] where each row is padded
            sizes: Actual size of each JPEG [B]

        Returns:
            List of decoded RGB images, each [H, W, 3]
        """
        executor = self._ensure_executor()
        batch_size = len(sizes)

        # Extract JPEG bytes from padded array
        jpeg_bytes_list = [bytes(data[i, :int(sizes[i])]) for i in range(batch_size)]

        # Parallel decode
        return list(executor.map(self._decode_one, jpeg_bytes_list))

    def decode_batch_to_tensor(
        self,
        data: NDArray[np.uint8],
        sizes: NDArray[np.uint64 | np.uint32 | np.int64],
    ) -> list[torch.Tensor]:
        """Decode batch of JPEGs to torch tensors in CHW format.

        Args:
            data: Raw JPEG data array [B, max_size]
            sizes: Actual size of each JPEG [B]

        Returns:
            List of decoded RGB tensors, each [C, H, W] uint8
        """
        images_hwc = self.decode_batch(data, sizes)
        return [torch.from_numpy(img).permute(2, 0, 1) for img in images_hwc]

    def decode_batch_random_crop(
        self,
        data: NDArray[np.uint8],
        sizes: NDArray[np.uint64 | np.uint32 | np.int64],
        heights: NDArray[np.uint32] | None = None,
        widths: NDArray[np.uint32] | None = None,
        target_size: int = 224,
        scale: tuple[float, float] = (0.08, 1.0),
        ratio: tuple[float, float] = (3 / 4, 4 / 3),
    ) -> list[NDArray[np.uint8]]:
        """Decode batch with RandomResizedCrop during decode.

        This is ~2x faster than decode + crop because:
        1. Crop happens in DCT space (before decompression)
        2. Less data to decompress after crop

        Note: The returned images may need final resize to target_size
        since DCT-space crop is constrained to MCU boundaries.

        Args:
            data: Raw JPEG data [B, max_size]
            sizes: Actual size of each JPEG [B]
            heights: Pre-stored heights [B] (optional, parsed from header if None)
            widths: Pre-stored widths [B] (optional, parsed from header if None)
            target_size: Final desired size (for reference, actual resize done later)
            scale: Range of size relative to original
            ratio: Range of aspect ratios

        Returns:
            List of cropped RGB images, each [H, W, 3]
            Note: Sizes may vary slightly due to MCU alignment
        """
        executor = self._ensure_executor()
        batch_size = len(sizes)

        # Extract JPEG bytes
        jpeg_bytes_list = [bytes(data[i, :int(sizes[i])]) for i in range(batch_size)]

        # Pre-compute dimensions if provided
        dims_provided = heights is not None and widths is not None

        def decode_with_crop(args: tuple[int, bytes]) -> NDArray[np.uint8]:
            i, jpeg_bytes = args

            # Get dimensions (from metadata or header)
            if dims_provided:
                w, h = int(widths[i]), int(heights[i])
            else:
                w, h = self._get_jpeg_dimensions(jpeg_bytes)

            # Generate random crop params using unified utility
            crop = generate_random_crop_params(w, h, scale=scale, ratio=ratio)

            # Decode with DCT-space crop
            return self._decode_one_with_crop(jpeg_bytes, crop)

        # Parallel decode with crop
        args_list = list(enumerate(jpeg_bytes_list))
        return list(executor.map(decode_with_crop, args_list))

    def decode_batch_center_crop(
        self,
        data: NDArray[np.uint8],
        sizes: NDArray[np.uint64 | np.uint32 | np.int64],
        heights: NDArray[np.uint32] | None = None,
        widths: NDArray[np.uint32] | None = None,
        crop_size: int = 224,
    ) -> list[NDArray[np.uint8]]:
        """Decode batch with center crop during decode.

        For validation/inference where center crop is needed.

        Args:
            data: Raw JPEG data [B, max_size]
            sizes: Actual size of each JPEG [B]
            heights: Pre-stored heights [B] (optional)
            widths: Pre-stored widths [B] (optional)
            crop_size: Size of center crop (square)

        Returns:
            List of center-cropped RGB images
        """
        executor = self._ensure_executor()
        batch_size = len(sizes)

        jpeg_bytes_list = [bytes(data[i, :int(sizes[i])]) for i in range(batch_size)]
        dims_provided = heights is not None and widths is not None

        def decode_with_center_crop(args: tuple[int, bytes]) -> NDArray[np.uint8]:
            i, jpeg_bytes = args

            if dims_provided:
                w, h = int(widths[i]), int(heights[i])
            else:
                w, h = self._get_jpeg_dimensions(jpeg_bytes)

            # Generate center crop params using unified utility
            crop = generate_center_crop_params(w, h, crop_size=crop_size)

            return self._decode_one_with_crop(jpeg_bytes, crop)

        args_list = list(enumerate(jpeg_bytes_list))
        return list(executor.map(decode_with_center_crop, args_list))

    def shutdown(self) -> None:
        """Shutdown the executor."""
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None

    def __del__(self) -> None:
        """Cleanup on deletion."""
        self.shutdown()

    def __repr__(self) -> str:
        return f"CPUDecoder(num_workers={self.num_workers})"


# Convenience alias
TurboJPEGBatchDecoder = CPUDecoder

__all__ = [
    "CPUDecoder",
    "TurboJPEGBatchDecoder",
    "check_turbojpeg_available",
]
