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

    def _decode_one_scaled(
        self,
        jpeg_bytes: bytes,
        scale_factor: tuple[int, int] = (1, 2),
    ) -> NDArray[np.uint8]:
        """Decode JPEG at reduced resolution (faster).

        TurboJPEG can decode directly to 1/2, 1/4, 1/8 size during
        decompression, which is much faster than full decode + resize.

        Args:
            jpeg_bytes: Raw JPEG bytes
            scale_factor: (numerator, denominator) e.g., (1, 2) for half size

        Returns:
            Scaled decoded RGB image [H, W, 3]
        """
        from turbojpeg import TJPF_RGB
        jpeg = _get_thread_local_jpeg()
        return jpeg.decode(jpeg_bytes, pixel_format=TJPF_RGB, scaling_factor=scale_factor)

    def _decode_one_crop_and_scale(
        self,
        jpeg_bytes: bytes,
        crop: CropParams,
        scale_factor: tuple[int, int] = (1, 2),
    ) -> NDArray[np.uint8]:
        """Decode JPEG with DCT crop AND scaled decode (fastest).

        Combines two optimizations:
        1. DCT-space crop (less data to decompress)
        2. Scaled decode (decode at reduced resolution)

        Args:
            jpeg_bytes: Raw JPEG bytes
            crop: Crop parameters (x, y, width, height)
            scale_factor: (numerator, denominator) for output scaling

        Returns:
            Cropped and scaled RGB image [H, W, 3]
        """
        from turbojpeg import TJPF_RGB
        jpeg = _get_thread_local_jpeg()

        try:
            # Step 1: Lossless crop in DCT space
            cropped_jpeg = jpeg.crop(
                jpeg_bytes,
                crop.x,
                crop.y,
                crop.width,
                crop.height,
            )

            # Step 2: Decode at reduced resolution
            return jpeg.decode(cropped_jpeg, pixel_format=TJPF_RGB, scaling_factor=scale_factor)

        except Exception:
            # Fallback: decode full at scale, then crop
            full_image = jpeg.decode(jpeg_bytes, pixel_format=TJPF_RGB, scaling_factor=scale_factor)
            # Adjust crop coords for scaled image
            scale = scale_factor[0] / scale_factor[1]
            sx, sy = int(crop.x * scale), int(crop.y * scale)
            sw, sh = int(crop.width * scale), int(crop.height * scale)
            return full_image[sy:sy + sh, sx:sx + sw]

    @staticmethod
    def get_best_scale_factor(
        source_size: int,
        target_size: int,
    ) -> tuple[int, int]:
        """Get the best TurboJPEG scale factor for target size.

        Returns a scale factor that produces an image at least as large
        as target_size, minimizing decode work while avoiding upscaling.

        Args:
            source_size: Source dimension (width or height)
            target_size: Target dimension

        Returns:
            (numerator, denominator) scale factor
        """
        # TurboJPEG supports these scale factors (ordered smallest to largest)
        scale_factors = [
            (1, 8),   # 12.5%
            (1, 4),   # 25%
            (3, 8),   # 37.5%
            (1, 2),   # 50%
            (5, 8),   # 62.5%
            (3, 4),   # 75%
            (7, 8),   # 87.5%
            (1, 1),   # 100%
        ]

        # Find the smallest scale that still produces >= target_size
        for num, denom in scale_factors:
            scaled_size = (source_size * num) // denom
            if scaled_size >= target_size:
                return (num, denom)

        # If no scale works (target > source), use 1:1
        return (1, 1)

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
        """Decode batch with RandomResizedCrop (decode full, then crop).

        Fast approach: decode full image then numpy slice. This is faster
        than DCT-space cropping for typical ImageNet-sized images (~256-512px)
        because the DCT crop overhead exceeds decode savings.

        Args:
            data: Raw JPEG data [B, max_size]
            sizes: Actual size of each JPEG [B]
            heights: Pre-stored heights [B] (optional)
            widths: Pre-stored widths [B] (optional)
            target_size: Final desired size (for reference)
            scale: Range of size relative to original
            ratio: Range of aspect ratios

        Returns:
            List of cropped RGB images, each [H, W, 3]
        """
        executor = self._ensure_executor()
        batch_size = len(sizes)

        jpeg_bytes_list = [bytes(data[i, :int(sizes[i])]) for i in range(batch_size)]
        dims_provided = heights is not None and widths is not None

        def decode_then_crop(args: tuple[int, bytes]) -> NDArray[np.uint8]:
            i, jpeg_bytes = args

            # Decode full image first
            img = self._decode_one(jpeg_bytes)
            h, w = img.shape[:2]

            # Generate random crop params
            crop = generate_random_crop_params(w, h, scale=scale, ratio=ratio)

            # Numpy slicing then copy (need copy since we return different sizes)
            return img[crop.y:crop.y + crop.height, crop.x:crop.x + crop.width].copy()

        args_list = list(enumerate(jpeg_bytes_list))
        return list(executor.map(decode_then_crop, args_list))

    def decode_batch_random_crop_dct(
        self,
        data: NDArray[np.uint8],
        sizes: NDArray[np.uint64 | np.uint32 | np.int64],
        heights: NDArray[np.uint32] | None = None,
        widths: NDArray[np.uint32] | None = None,
        target_size: int = 224,
        scale: tuple[float, float] = (0.08, 1.0),
        ratio: tuple[float, float] = (3 / 4, 4 / 3),
    ) -> list[NDArray[np.uint8]]:
        """Decode batch with RandomResizedCrop using DCT-space cropping.

        DCT-space cropping crops in the frequency domain before decompression.
        This can be faster for very large images (>1024px) but has overhead
        that makes it slower for typical ImageNet-sized images.

        Note: The returned images may need final resize to target_size
        since DCT-space crop is constrained to MCU boundaries (8x8 or 16x16).

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
        """Decode batch with center crop (decode full, then crop).

        Fast approach: decode full image then numpy slice. This is faster
        than DCT-space cropping for typical ImageNet-sized images (~256-512px).

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

        def decode_then_crop(args: tuple[int, bytes]) -> NDArray[np.uint8]:
            i, jpeg_bytes = args

            # Decode full image first
            img = self._decode_one(jpeg_bytes)
            h, w = img.shape[:2]

            # Calculate center crop coordinates
            top = (h - crop_size) // 2
            left = (w - crop_size) // 2

            # Numpy slicing then copy
            return img[top:top + crop_size, left:left + crop_size].copy()

        args_list = list(enumerate(jpeg_bytes_list))
        return list(executor.map(decode_then_crop, args_list))

    def decode_batch_center_crop_dct(
        self,
        data: NDArray[np.uint8],
        sizes: NDArray[np.uint64 | np.uint32 | np.int64],
        heights: NDArray[np.uint32] | None = None,
        widths: NDArray[np.uint32] | None = None,
        crop_size: int = 224,
    ) -> list[NDArray[np.uint8]]:
        """Decode batch with center crop using DCT-space cropping.

        DCT-space cropping crops in the frequency domain before decompression.
        This can be faster for very large images (>1024px) but has overhead
        that makes it slower for typical ImageNet-sized images.

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

    def decode_batch_random_crop_to_tensor(
        self,
        data: NDArray[np.uint8],
        sizes: NDArray[np.uint64 | np.uint32 | np.int64],
        heights: NDArray[np.uint32] | None = None,
        widths: NDArray[np.uint32] | None = None,
        target_size: int = 224,
        scale: tuple[float, float] = (0.08, 1.0),
        ratio: tuple[float, float] = (3 / 4, 4 / 3),
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decode batch with RandomResizedCrop, returning a pre-sized tensor.

        Optimized version that:
        1. Uses DCT-space crop to reduce data
        2. Uses scaled decode when beneficial
        3. Writes directly to pre-allocated output tensor

        Args:
            data: Raw JPEG data [B, max_size]
            sizes: Actual size of each JPEG [B]
            heights: Pre-stored heights [B] (optional)
            widths: Pre-stored widths [B] (optional)
            target_size: Final output size (square)
            scale: Range of crop area relative to original
            ratio: Range of aspect ratios
            output: Optional pre-allocated tensor [B, 3, target_size, target_size]

        Returns:
            Tensor [B, 3, target_size, target_size] uint8
        """
        from PIL import Image

        executor = self._ensure_executor()
        batch_size = len(sizes)

        # Pre-allocate output if not provided
        if output is None:
            output = torch.zeros(
                (batch_size, 3, target_size, target_size),
                dtype=torch.uint8,
            )

        jpeg_bytes_list = [bytes(data[i, :int(sizes[i])]) for i in range(batch_size)]
        dims_provided = heights is not None and widths is not None

        def decode_crop_resize(args: tuple[int, bytes]) -> tuple[int, NDArray[np.uint8]]:
            i, jpeg_bytes = args

            if dims_provided:
                w, h = int(widths[i]), int(heights[i])
            else:
                w, h = self._get_jpeg_dimensions(jpeg_bytes)

            # Generate random crop params
            crop = generate_random_crop_params(w, h, scale=scale, ratio=ratio)

            # Determine if scaled decode is beneficial
            # Use scaled decode if crop is significantly larger than target
            min_crop_dim = min(crop.width, crop.height)
            scale_factor = self.get_best_scale_factor(min_crop_dim, target_size)

            # Decode with crop and optional scaling
            if scale_factor != (1, 1):
                img = self._decode_one_crop_and_scale(jpeg_bytes, crop, scale_factor)
            else:
                img = self._decode_one_with_crop(jpeg_bytes, crop)

            # Resize to target size using PIL
            if img.shape[0] != target_size or img.shape[1] != target_size:
                pil_img = Image.fromarray(img)
                pil_img = pil_img.resize((target_size, target_size), Image.BILINEAR)
                img = np.array(pil_img)

            return i, img

        # Parallel decode + resize
        args_list = list(enumerate(jpeg_bytes_list))
        results = list(executor.map(decode_crop_resize, args_list))

        # Copy results to output tensor
        for i, img in results:
            output[i] = torch.from_numpy(img).permute(2, 0, 1)

        return output

    def decode_batch_center_crop_to_tensor(
        self,
        data: NDArray[np.uint8],
        sizes: NDArray[np.uint64 | np.uint32 | np.int64],
        heights: NDArray[np.uint32] | None = None,
        widths: NDArray[np.uint32] | None = None,
        crop_size: int = 224,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decode batch with center crop, returning a pre-sized tensor.

        Optimized version that:
        1. Uses DCT-space crop
        2. Uses scaled decode when beneficial
        3. Writes directly to pre-allocated output tensor

        Args:
            data: Raw JPEG data [B, max_size]
            sizes: Actual size of each JPEG [B]
            heights: Pre-stored heights [B] (optional)
            widths: Pre-stored widths [B] (optional)
            crop_size: Final output size (square)
            output: Optional pre-allocated tensor [B, 3, crop_size, crop_size]

        Returns:
            Tensor [B, 3, crop_size, crop_size] uint8
        """
        from PIL import Image

        executor = self._ensure_executor()
        batch_size = len(sizes)

        if output is None:
            output = torch.zeros(
                (batch_size, 3, crop_size, crop_size),
                dtype=torch.uint8,
            )

        jpeg_bytes_list = [bytes(data[i, :int(sizes[i])]) for i in range(batch_size)]
        dims_provided = heights is not None and widths is not None

        def decode_crop_resize(args: tuple[int, bytes]) -> tuple[int, NDArray[np.uint8]]:
            i, jpeg_bytes = args

            if dims_provided:
                w, h = int(widths[i]), int(heights[i])
            else:
                w, h = self._get_jpeg_dimensions(jpeg_bytes)

            # Generate center crop params
            crop = generate_center_crop_params(w, h, crop_size=crop_size)

            # Determine if scaled decode is beneficial
            min_crop_dim = min(crop.width, crop.height)
            scale_factor = self.get_best_scale_factor(min_crop_dim, crop_size)

            # Decode with crop and optional scaling
            if scale_factor != (1, 1):
                img = self._decode_one_crop_and_scale(jpeg_bytes, crop, scale_factor)
            else:
                img = self._decode_one_with_crop(jpeg_bytes, crop)

            # Resize to target size using PIL
            if img.shape[0] != crop_size or img.shape[1] != crop_size:
                pil_img = Image.fromarray(img)
                pil_img = pil_img.resize((crop_size, crop_size), Image.BILINEAR)
                img = np.array(pil_img)

            return i, img

        args_list = list(enumerate(jpeg_bytes_list))
        results = list(executor.map(decode_crop_resize, args_list))

        for i, img in results:
            output[i] = torch.from_numpy(img).permute(2, 0, 1)

        return output

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
