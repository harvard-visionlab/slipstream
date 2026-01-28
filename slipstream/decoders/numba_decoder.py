"""Numba-optimized JPEG decoder with parallel decode using prange.

Based on litdata-mmap's libffcv_lite/__init__.py pattern:
1. Numba prange for true parallelism (compiled to native code, no GIL)
2. ctypes function pointers callable from Numba
3. Pre-allocated output buffers for zero-copy decode

This replaces the ThreadPoolExecutor approach for ~1.8x better performance.

Usage:
    decoder = NumbaBatchDecoder(num_threads=12)

    # Decode only (returns list of numpy arrays)
    images = decoder.decode_batch(data, sizes, heights, widths)

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

import ctypes
from ctypes import CDLL, POINTER, c_bool, c_int, c_int64, c_uint32, c_uint64, c_void_p
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from numba import njit, prange, set_num_threads
from numpy.typing import NDArray

__all__ = [
    "NumbaBatchDecoder",
    "check_numba_decoder_available",
    "load_library",
    "Compiler",
]

# =============================================================================
# Library loading and ctypes setup
# =============================================================================

_lib: CDLL | None = None
_ctypes_imdecode_simple: Any = None
_ctypes_resize_crop: Any = None


def _find_library() -> Path:
    """Find the compiled libslipstream library."""
    # Look in libslipstream directory relative to this file
    module_dir = Path(__file__).parent.parent.parent / "libslipstream"

    # Look for .so or .dylib files matching our pattern
    for so_file in module_dir.glob("_libslipstream*.so"):
        return so_file
    for so_file in module_dir.glob("_libslipstream*.dylib"):
        return so_file

    raise RuntimeError(
        "libslipstream library not found. Build it with:\n"
        "  uv run python libslipstream/setup.py build_ext --inplace"
    )


def load_library() -> CDLL:
    """Load the libslipstream C++ library."""
    global _lib, _ctypes_imdecode_simple, _ctypes_resize_crop
    if _lib is not None:
        return _lib

    lib_path = _find_library()
    _lib = CDLL(str(lib_path))

    # Set up function signatures
    # jpeg_header(input, size, out_width, out_height) -> int
    _lib.jpeg_header.argtypes = [c_void_p, c_uint64, POINTER(c_uint32), POINTER(c_uint32)]
    _lib.jpeg_header.restype = c_int

    # imdecode_simple(input, size, output, height, width) -> int
    _lib.imdecode_simple.argtypes = [
        c_void_p, c_uint64, c_void_p, c_uint32, c_uint32
    ]
    _lib.imdecode_simple.restype = c_int

    # resize_crop(source_p, source_h, source_w, crop_y, crop_x, crop_h, crop_w,
    #             dest_p, target_h, target_w) -> int
    _lib.resize_crop.argtypes = [
        c_int64, c_int64, c_int64,  # source_p, source_h, source_w
        c_int64, c_int64, c_int64, c_int64,  # crop_y, crop_x, crop_h, crop_w
        c_int64, c_int64, c_int64,  # dest_p, target_h, target_w
    ]
    _lib.resize_crop.restype = c_int

    # resize_simple(source_p, source_h, source_w, dest_p, target_h, target_w) -> int
    _lib.resize_simple.argtypes = [
        c_int64, c_int64, c_int64,  # source_p, source_h, source_w
        c_int64, c_int64, c_int64,  # dest_p, target_h, target_w
    ]
    _lib.resize_simple.restype = c_int

    # Cache the ctypes functions for Numba
    _ctypes_imdecode_simple = _lib.imdecode_simple
    _ctypes_resize_crop = _lib.resize_crop

    return _lib


def check_numba_decoder_available() -> bool:
    """Check if the Numba batch decoder is available."""
    try:
        load_library()
        return True
    except Exception:
        return False


# =============================================================================
# Numba wrappers for ctypes functions
# =============================================================================

def imdecode_simple_numba(
    source: np.ndarray,
    dst: np.ndarray,
    height: int,
    width: int,
) -> int:
    """Numba-compatible imdecode_simple wrapper.

    This function is designed to be called from within Numba prange loops.
    It accesses the ctypes function via the module-level variable.
    """
    global _ctypes_imdecode_simple
    if _ctypes_imdecode_simple is None:
        load_library()
    return _ctypes_imdecode_simple(
        source.ctypes.data, source.size,
        dst.ctypes.data,
        height, width,
    )


def resize_crop_numba(
    source: np.ndarray,
    source_h: int,
    source_w: int,
    crop_y: int,
    crop_x: int,
    crop_h: int,
    crop_w: int,
    dest: np.ndarray,
    target_h: int,
    target_w: int,
) -> int:
    """Numba-compatible resize_crop wrapper."""
    global _ctypes_resize_crop
    if _ctypes_resize_crop is None:
        load_library()
    return _ctypes_resize_crop(
        source.ctypes.data, source_h, source_w,
        crop_y, crop_x, crop_h, crop_w,
        dest.ctypes.data, target_h, target_w,
    )


# =============================================================================
# Compiler class matching FFCV/litdata-mmap pattern
# =============================================================================

class Compiler:
    """JIT compiler for decode functions, matching FFCV's pattern."""

    is_enabled: bool = True
    num_threads: int = 1

    @classmethod
    def set_enabled(cls, b: bool) -> None:
        cls.is_enabled = b

    @classmethod
    def set_num_threads(cls, n: int) -> None:
        if n < 1:
            n = cpu_count()
        cls.num_threads = n
        set_num_threads(n)

    @classmethod
    def compile(cls, code: Any, signature: Any = None) -> Any:
        """Compile a function with Numba."""
        parallel = False
        if hasattr(code, 'is_parallel'):
            parallel = code.is_parallel and cls.num_threads > 1

        if cls.is_enabled:
            return njit(signature, fastmath=True, nogil=True,
                        error_model='numpy', parallel=parallel)(code)
        return code

    @classmethod
    def get_iterator(cls) -> Any:
        """Get parallel or sequential iterator."""
        if cls.num_threads > 1:
            return prange
        return range


# Initialize compiler with default thread count
# Note: set_num_threads must be called before any parallel jit compilation
try:
    Compiler.set_num_threads(cpu_count())
except ValueError:
    # Already set, ignore
    pass


# =============================================================================
# Compiled decode functions
# =============================================================================

def _create_decode_function() -> Any:
    """Create the Numba-compiled decode function.

    This follows FFCV/litdata-mmap's pattern exactly:
    1. Get the imdecode wrapper
    2. Compile it with Numba
    3. Use prange for parallelism
    """
    # Compile the imdecode wrapper for use in Numba
    imdecode_c = Compiler.compile(imdecode_simple_numba)
    my_range = Compiler.get_iterator()

    def decode_batch(
        jpeg_data: np.ndarray,  # [B, max_size] uint8
        sizes: np.ndarray,  # [B] uint64
        heights: np.ndarray,  # [B] uint32
        widths: np.ndarray,  # [B] uint32
        destination: np.ndarray,  # [B, max_h, max_w, 3] uint8
    ) -> np.ndarray:
        """Decode batch of JPEGs in parallel using prange."""
        batch_size = len(sizes)
        for i in my_range(batch_size):
            size = int(sizes[i])
            h = int(heights[i])
            w = int(widths[i])

            # Get source and destination slices
            source = jpeg_data[i, :size]
            dst = destination[i, :h, :w, :]

            # Call the compiled imdecode
            imdecode_c(source, dst, h, w)

        return destination[:batch_size]

    decode_batch.is_parallel = True
    return Compiler.compile(decode_batch)


def _create_decode_with_crop_function() -> Any:
    """Create decode + crop + resize function.

    Decodes to temp buffer, then crops and resizes to target size.
    """
    imdecode_c = Compiler.compile(imdecode_simple_numba)
    resize_crop_c = Compiler.compile(resize_crop_numba)
    my_range = Compiler.get_iterator()

    def decode_crop_batch(
        jpeg_data: np.ndarray,  # [B, max_size] uint8
        sizes: np.ndarray,  # [B] uint64
        heights: np.ndarray,  # [B] uint32
        widths: np.ndarray,  # [B] uint32
        crop_params: np.ndarray,  # [B, 4] int32 (crop_x, crop_y, crop_w, crop_h)
        temp_buffer: np.ndarray,  # [B, max_h, max_w, 3] uint8
        destination: np.ndarray,  # [B, target_h, target_w, 3] uint8
        target_h: int,
        target_w: int,
    ) -> np.ndarray:
        """Decode + crop + resize batch in parallel using prange."""
        batch_size = len(sizes)
        for i in my_range(batch_size):
            size = int(sizes[i])
            h = int(heights[i])
            w = int(widths[i])

            # Get source slice
            source = jpeg_data[i, :size]
            temp = temp_buffer[i, :h, :w, :]

            # Decode to temp buffer
            imdecode_c(source, temp, h, w)

            # Get crop parameters
            crop_x = int(crop_params[i, 0])
            crop_y = int(crop_params[i, 1])
            crop_w = int(crop_params[i, 2])
            crop_h = int(crop_params[i, 3])

            # Crop and resize to destination
            dest = destination[i, :, :, :]
            resize_crop_c(
                temp, h, w,
                crop_y, crop_x, crop_h, crop_w,
                dest, target_h, target_w,
            )

        return destination[:batch_size]

    decode_crop_batch.is_parallel = True
    return Compiler.compile(decode_crop_batch)


# Lazy-loaded compiled functions
_decode_batch_compiled: Any = None
_decode_crop_batch_compiled: Any = None


def _get_decode_batch() -> Any:
    """Get or create the compiled decode function."""
    global _decode_batch_compiled
    if _decode_batch_compiled is None:
        # Ensure library is loaded first
        load_library()
        _decode_batch_compiled = _create_decode_function()
    return _decode_batch_compiled


def _get_decode_crop_batch() -> Any:
    """Get or create the compiled decode + crop function."""
    global _decode_crop_batch_compiled
    if _decode_crop_batch_compiled is None:
        load_library()
        _decode_crop_batch_compiled = _create_decode_with_crop_function()
    return _decode_crop_batch_compiled


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
# Main decoder class
# =============================================================================

class NumbaBatchDecoder:
    """High-performance batch JPEG decoder using Numba prange + C extension.

    Uses Numba prange for batch parallelism. The decode loop is JIT-compiled
    to native code with nogil=True, allowing true parallel execution.

    This matches FFCV's approach for maximum performance (~1.8x faster than
    ThreadPoolExecutor).

    Example:
        decoder = NumbaBatchDecoder(num_threads=8)

        # Decode only (returns list of variable-sized images)
        images = decoder.decode_batch(data, sizes, heights, widths)

        # Pre-allocate destination for fixed-size output
        destination = np.zeros((batch_size, max_h, max_w, 3), dtype=np.uint8)
        decoder.decode_batch_to_buffer(data, sizes, heights, widths, destination)

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
        if num_threads < 1:
            num_threads = cpu_count()
        self.num_threads = num_threads
        Compiler.set_num_threads(num_threads)

        # Ensure library and compiled function are ready
        load_library()
        self._decode_fn = _get_decode_batch()
        self._decode_crop_fn = _get_decode_crop_batch()

        # Seed counter for random crops
        self._seed_counter = 0

        # Reusable buffers (allocated on first use)
        self._temp_buffer: np.ndarray | None = None
        self._dest_buffer: np.ndarray | None = None

    def _ensure_temp_buffer(self, batch_size: int, max_h: int, max_w: int) -> np.ndarray:
        """Get or allocate temp buffer for decode."""
        if (self._temp_buffer is None or
            self._temp_buffer.shape[0] < batch_size or
            self._temp_buffer.shape[1] < max_h or
            self._temp_buffer.shape[2] < max_w):
            self._temp_buffer = np.zeros((batch_size, max_h, max_w, 3), dtype=np.uint8)
        return self._temp_buffer

    def _ensure_dest_buffer(self, batch_size: int, target_h: int, target_w: int) -> np.ndarray:
        """Get or allocate destination buffer for crop+resize."""
        if (self._dest_buffer is None or
            self._dest_buffer.shape[0] < batch_size or
            self._dest_buffer.shape[1] != target_h or
            self._dest_buffer.shape[2] != target_w):
            self._dest_buffer = np.zeros((batch_size, target_h, target_w, 3), dtype=np.uint8)
        return self._dest_buffer

    def decode_batch_to_buffer(
        self,
        jpeg_data: NDArray[np.uint8],
        sizes: NDArray,
        heights: NDArray,
        widths: NDArray,
        destination: NDArray[np.uint8],
    ) -> NDArray[np.uint8]:
        """Decode batch of JPEGs in parallel to pre-allocated buffer.

        Args:
            jpeg_data: Padded JPEG data [B, max_size] uint8
            sizes: Actual JPEG sizes [B] uint64
            heights: Image heights [B] uint32
            widths: Image widths [B] uint32
            destination: Pre-allocated output [B, max_h, max_w, 3] uint8

        Returns:
            The destination array (for convenience)
        """
        # Ensure correct dtypes
        sizes = np.ascontiguousarray(sizes, dtype=np.uint64)
        heights = np.ascontiguousarray(heights, dtype=np.uint32)
        widths = np.ascontiguousarray(widths, dtype=np.uint32)

        return self._decode_fn(jpeg_data, sizes, heights, widths, destination)

    def decode_batch(
        self,
        data: NDArray[np.uint8],
        sizes: NDArray,
        heights: NDArray | None = None,
        widths: NDArray | None = None,
        destination: NDArray[np.uint8] | None = None,
    ) -> list[NDArray[np.uint8]]:
        """Decode batch of JPEGs to list of numpy arrays.

        Args:
            data: Padded JPEG data [B, max_size] uint8
            sizes: Actual JPEG sizes [B]
            heights: Image heights [B] (required for Numba path)
            widths: Image widths [B] (required for Numba path)
            destination: Optional pre-allocated buffer

        Returns:
            List of decoded RGB images [H, W, 3]
        """
        if heights is None or widths is None:
            raise ValueError("heights and widths are required for NumbaBatchDecoder")

        batch_size = len(sizes)
        max_h = int(np.max(heights))
        max_w = int(np.max(widths))

        # Use provided buffer or allocate temp
        if destination is None:
            destination = self._ensure_temp_buffer(batch_size, max_h, max_w)

        # Decode in parallel
        self.decode_batch_to_buffer(data, sizes, heights, widths, destination)

        # Extract individual images (variable sizes)
        results = []
        for i in range(batch_size):
            h, w = int(heights[i]), int(widths[i])
            results.append(destination[i, :h, :w, :].copy())

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
            crop_size: Size of center crop (and final output)
            destination: Unused (for API compatibility)

        Returns:
            Array [B, crop_size, crop_size, 3] uint8
        """
        batch_size = len(sizes)
        max_h = int(np.max(heights))
        max_w = int(np.max(widths))

        # Ensure dtypes
        sizes = np.ascontiguousarray(sizes, dtype=np.uint64)
        heights_arr = np.ascontiguousarray(heights, dtype=np.uint32)
        widths_arr = np.ascontiguousarray(widths, dtype=np.uint32)

        # Generate crop parameters
        crop_params = _generate_center_crop_params_batch(
            widths.astype(np.int32),
            heights.astype(np.int32),
            crop_size,
        )

        # Allocate buffers
        temp_buffer = self._ensure_temp_buffer(batch_size, max_h, max_w)
        dest_buffer = self._ensure_dest_buffer(batch_size, crop_size, crop_size)

        # Decode + crop + resize
        self._decode_crop_fn(
            data, sizes, heights_arr, widths_arr,
            crop_params, temp_buffer, dest_buffer,
            crop_size, crop_size,
        )

        # Return copy of relevant portion
        return dest_buffer[:batch_size].copy()

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

        batch_size = len(sizes)
        max_h = int(np.max(heights))
        max_w = int(np.max(widths))

        # Ensure dtypes
        sizes = np.ascontiguousarray(sizes, dtype=np.uint64)
        heights_arr = np.ascontiguousarray(heights, dtype=np.uint32)
        widths_arr = np.ascontiguousarray(widths, dtype=np.uint32)

        # Generate random crop parameters
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

        # Allocate buffers
        temp_buffer = self._ensure_temp_buffer(batch_size, max_h, max_w)
        dest_buffer = self._ensure_dest_buffer(batch_size, target_size, target_size)

        # Decode + crop + resize
        self._decode_crop_fn(
            data, sizes, heights_arr, widths_arr,
            crop_params, temp_buffer, dest_buffer,
            target_size, target_size,
        )

        # Convert to tensor [B, C, H, W]
        result = torch.from_numpy(dest_buffer[:batch_size].copy())
        result = result.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]

        return result

    def shutdown(self) -> None:
        """Release resources (no-op for this implementation)."""
        self._temp_buffer = None
        self._dest_buffer = None

    def __repr__(self) -> str:
        return f"NumbaBatchDecoder(num_threads={self.num_threads})"
