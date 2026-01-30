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

from ctypes import CDLL, POINTER, c_int, c_int64, c_uint32, c_uint64, c_void_p
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any

import numpy as np
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


def _create_decode_multi_crop_function() -> Any:
    """Create decode-once + multi-crop function.

    Decodes each image once to temp buffer, then applies N different
    crop+resize operations from the same decoded data. This avoids
    redundant JPEG decodes for multi-crop SSL.
    """
    imdecode_c = Compiler.compile(imdecode_simple_numba)
    resize_crop_c = Compiler.compile(resize_crop_numba)
    my_range = Compiler.get_iterator()

    def decode_multi_crop_batch(
        jpeg_data: np.ndarray,      # [B, max_size] uint8
        sizes: np.ndarray,          # [B] uint64
        heights: np.ndarray,        # [B] uint32
        widths: np.ndarray,         # [B] uint32
        all_crop_params: np.ndarray, # [num_crops, B, 4] int32
        temp_buffer: np.ndarray,    # [B, max_h, max_w, 3] uint8
        destinations: np.ndarray,   # [num_crops, B, target_h, target_w, 3] uint8
        target_h: int,
        target_w: int,
        num_crops: int,
    ) -> np.ndarray:
        """Decode once + multi-crop+resize in parallel using prange."""
        batch_size = len(sizes)
        for i in my_range(batch_size):
            size = int(sizes[i])
            h = int(heights[i])
            w = int(widths[i])

            # Decode JPEG to temp buffer (ONCE per image)
            source = jpeg_data[i, :size]
            temp = temp_buffer[i, :h, :w, :]
            imdecode_c(source, temp, h, w)

            # Apply each crop from the same decoded image
            for c in range(num_crops):
                crop_x = int(all_crop_params[c, i, 0])
                crop_y = int(all_crop_params[c, i, 1])
                crop_w = int(all_crop_params[c, i, 2])
                crop_h = int(all_crop_params[c, i, 3])

                dest = destinations[c, i, :, :, :]
                resize_crop_c(
                    temp, h, w,
                    crop_y, crop_x, crop_h, crop_w,
                    dest, target_h, target_w,
                )

        return destinations

    decode_multi_crop_batch.is_parallel = True
    return Compiler.compile(decode_multi_crop_batch)


@njit(parallel=True, cache=True, fastmath=True)
def _transpose_hwc_to_chw(
    src: np.ndarray,   # [B, H, W, 3] uint8
    dst: np.ndarray,   # [B, 3, H, W] uint8
    batch_size: int,
) -> None:
    """Transpose HWC→CHW per-image in parallel.

    Each image (~150KB at 224x224) fits in L1/L2 cache, making this
    much faster than a bulk transpose of the entire batch.
    """
    H = src.shape[1]
    W = src.shape[2]
    for i in prange(batch_size):
        for h in range(H):
            for w in range(W):
                dst[i, 0, h, w] = src[i, h, w, 0]
                dst[i, 1, h, w] = src[i, h, w, 1]
                dst[i, 2, h, w] = src[i, h, w, 2]


# Lazy-loaded compiled functions
_decode_batch_compiled: Any = None
_decode_crop_batch_compiled: Any = None
_decode_multi_crop_batch_compiled: Any = None


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


def _get_decode_multi_crop_batch() -> Any:
    """Get or create the compiled decode-once + multi-crop function."""
    global _decode_multi_crop_batch_compiled
    if _decode_multi_crop_batch_compiled is None:
        load_library()
        _decode_multi_crop_batch_compiled = _create_decode_multi_crop_function()
    return _decode_multi_crop_batch_compiled


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
        self._decode_multi_crop_fn = _get_decode_multi_crop_batch()

        # Seed counter for random crops
        self._seed_counter = 0

        # Reusable buffers (allocated on first use)
        self._temp_buffer: np.ndarray | None = None
        self._dest_buffer: np.ndarray | None = None
        self._chw_buffer: np.ndarray | None = None
        self._multi_crop_buffer: np.ndarray | None = None
        self._multi_chw_buffer: np.ndarray | None = None

    def _ensure_temp_buffer(self, batch_size: int, max_h: int, max_w: int) -> np.ndarray:
        """Get or allocate temp buffer for decode."""
        if (self._temp_buffer is None or
            self._temp_buffer.shape[0] < batch_size or
            self._temp_buffer.shape[1] < max_h or
            self._temp_buffer.shape[2] < max_w):
            self._temp_buffer = np.zeros((batch_size, max_h, max_w, 3), dtype=np.uint8)
        return self._temp_buffer

    def _ensure_chw_buffer(self, batch_size: int, target_h: int, target_w: int) -> np.ndarray:
        """Get or allocate CHW output buffer."""
        if (self._chw_buffer is None or
            self._chw_buffer.shape[0] < batch_size or
            self._chw_buffer.shape[2] != target_h or
            self._chw_buffer.shape[3] != target_w):
            self._chw_buffer = np.zeros((batch_size, 3, target_h, target_w), dtype=np.uint8)
        return self._chw_buffer

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

        # Ensure dtypes (avoid copies if already correct dtype)
        sizes_u64 = sizes if sizes.dtype == np.uint64 else np.ascontiguousarray(sizes, dtype=np.uint64)
        heights_u32 = heights if heights.dtype == np.uint32 else np.ascontiguousarray(heights, dtype=np.uint32)
        widths_u32 = widths if widths.dtype == np.uint32 else np.ascontiguousarray(widths, dtype=np.uint32)
        heights_i32 = heights if heights.dtype == np.int32 else np.ascontiguousarray(heights, dtype=np.int32)
        widths_i32 = widths if widths.dtype == np.int32 else np.ascontiguousarray(widths, dtype=np.int32)

        # Generate crop parameters
        crop_params = _generate_center_crop_params_batch(
            widths_i32, heights_i32, crop_size,
        )

        # Allocate buffers
        temp_buffer = self._ensure_temp_buffer(batch_size, max_h, max_w)
        dest_buffer = self._ensure_dest_buffer(batch_size, crop_size, crop_size)

        # Decode + crop + resize
        self._decode_crop_fn(
            data, sizes_u64, heights_u32, widths_u32,
            crop_params, temp_buffer, dest_buffer,
            crop_size, crop_size,
        )

        # Return view (no copy — caller should not hold reference across batches)
        return dest_buffer[:batch_size]

    def decode_batch_resize_crop(
        self,
        data: NDArray[np.uint8],
        sizes: NDArray,
        heights: NDArray,
        widths: NDArray,
        resize_size: int = 256,
        crop_size: int = 224,
        destination: NDArray[np.uint8] | None = None,
    ) -> NDArray[np.uint8]:
        """Decode batch with resize shortest edge + center crop.

        Standard ImageNet validation transform:
        1. Decode JPEG
        2. Resize so shortest edge = resize_size
        3. Center crop to crop_size x crop_size

        This is implemented as a single fused operation using resize_crop:
        we compute the crop region that corresponds to resizing the shortest
        edge to resize_size and then taking a center crop of crop_size.

        Args:
            data: Padded JPEG data [B, max_size] uint8
            sizes: Actual JPEG sizes [B]
            heights: Image heights [B]
            widths: Image widths [B]
            resize_size: Target size for shortest edge (default 256)
            crop_size: Final crop size (default 224)
            destination: Unused (for API compatibility)

        Returns:
            Array [B, crop_size, crop_size, 3] uint8
        """
        batch_size = len(sizes)
        max_h = int(np.max(heights))
        max_w = int(np.max(widths))

        # Ensure dtypes
        sizes_u64 = sizes if sizes.dtype == np.uint64 else np.ascontiguousarray(sizes, dtype=np.uint64)
        heights_u32 = heights if heights.dtype == np.uint32 else np.ascontiguousarray(heights, dtype=np.uint32)
        widths_u32 = widths if widths.dtype == np.uint32 else np.ascontiguousarray(widths, dtype=np.uint32)

        # Compute crop params: resize shortest edge then center crop
        # Instead of actually resizing first, we compute the equivalent crop
        # in the original image space.
        crop_params = np.zeros((batch_size, 4), dtype=np.int32)
        for i in range(batch_size):
            h = int(heights[i])
            w = int(widths[i])

            # After resizing shortest edge to resize_size, what's the scale?
            if h < w:
                scale = resize_size / h
                new_h = resize_size
                new_w = int(w * scale + 0.5)
            else:
                scale = resize_size / w
                new_w = resize_size
                new_h = int(h * scale + 0.5)

            # Center crop of crop_size in the resized image
            # Map back to original image coordinates
            crop_h_resized = min(crop_size, new_h)
            crop_w_resized = min(crop_size, new_w)
            start_y_resized = (new_h - crop_h_resized) // 2
            start_x_resized = (new_w - crop_w_resized) // 2

            # Map to original coordinates
            crop_x = int(start_x_resized / scale + 0.5)
            crop_y = int(start_y_resized / scale + 0.5)
            crop_w_orig = int(crop_w_resized / scale + 0.5)
            crop_h_orig = int(crop_h_resized / scale + 0.5)

            # Clamp to image bounds
            crop_x = max(0, min(crop_x, w - 1))
            crop_y = max(0, min(crop_y, h - 1))
            crop_w_orig = max(1, min(crop_w_orig, w - crop_x))
            crop_h_orig = max(1, min(crop_h_orig, h - crop_y))

            crop_params[i, 0] = crop_x
            crop_params[i, 1] = crop_y
            crop_params[i, 2] = crop_w_orig
            crop_params[i, 3] = crop_h_orig

        # Allocate buffers
        temp_buffer = self._ensure_temp_buffer(batch_size, max_h, max_w)
        dest_buffer = self._ensure_dest_buffer(batch_size, crop_size, crop_size)

        # Decode + crop + resize
        self._decode_crop_fn(
            data, sizes_u64, heights_u32, widths_u32,
            crop_params, temp_buffer, dest_buffer,
            crop_size, crop_size,
        )

        return dest_buffer[:batch_size]

    def decode_batch_random_crop(
        self,
        data: NDArray[np.uint8],
        sizes: NDArray,
        heights: NDArray,
        widths: NDArray,
        target_size: int = 224,
        scale: tuple[float, float] = (0.08, 1.0),
        ratio: tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
        seed: int | None = None,
        destination: NDArray[np.uint8] | None = None,
    ) -> NDArray[np.uint8]:
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
            Array [B, target_size, target_size, 3] uint8
        """
        import math

        batch_size = len(sizes)
        max_h = int(np.max(heights))
        max_w = int(np.max(widths))

        # Ensure dtypes (avoid copies if already correct dtype)
        sizes_u64 = sizes if sizes.dtype == np.uint64 else np.ascontiguousarray(sizes, dtype=np.uint64)
        heights_u32 = heights if heights.dtype == np.uint32 else np.ascontiguousarray(heights, dtype=np.uint32)
        widths_u32 = widths if widths.dtype == np.uint32 else np.ascontiguousarray(widths, dtype=np.uint32)
        heights_i32 = heights if heights.dtype == np.int32 else np.ascontiguousarray(heights, dtype=np.int32)
        widths_i32 = widths if widths.dtype == np.int32 else np.ascontiguousarray(widths, dtype=np.int32)

        # Generate random crop parameters
        log_ratio_min = math.log(ratio[0])
        log_ratio_max = math.log(ratio[1])

        if seed is not None:
            # Reproducible: use provided seed + auto-incrementing counter
            self._seed_counter += 1
            batch_seed = seed + self._seed_counter
        else:
            # Non-reproducible: use auto-incrementing counter (original behavior)
            self._seed_counter += 1
            batch_seed = self._seed_counter

        crop_params = _generate_random_crop_params_batch(
            widths_i32, heights_i32,
            scale[0], scale[1],
            log_ratio_min, log_ratio_max,
            batch_seed,
        )

        # Allocate buffers
        temp_buffer = self._ensure_temp_buffer(batch_size, max_h, max_w)
        dest_buffer = self._ensure_dest_buffer(batch_size, target_size, target_size)

        # Decode + crop + resize
        self._decode_crop_fn(
            data, sizes_u64, heights_u32, widths_u32,
            crop_params, temp_buffer, dest_buffer,
            target_size, target_size,
        )

        # Return view (no copy — caller should not hold reference across batches)
        # Tensor conversion + permute to [B, C, H, W] should happen in the pipeline
        return dest_buffer[:batch_size]

    def decode_batch_multi_crop(
        self,
        data: NDArray[np.uint8],
        sizes: NDArray,
        heights: NDArray,
        widths: NDArray,
        num_crops: int = 2,
        target_size: int = 224,
        scale: tuple[float, float] = (0.08, 1.0),
        ratio: tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
        seeds: list[int | None] | None = None,
    ) -> list[NDArray[np.uint8]]:
        """Decode batch once, then apply N random crops from decoded data.

        Much faster than N separate decode_batch_random_crop calls because
        JPEG decode (~80-92% of time) happens only once per image.

        Args:
            data: Padded JPEG data [B, max_size] uint8
            sizes: Actual JPEG sizes [B]
            heights: Image heights [B]
            widths: Image widths [B]
            num_crops: Number of random crops per image
            target_size: Final output size (square)
            scale: Scale range for random area
            ratio: Aspect ratio range
            seeds: Per-crop seeds for reproducibility. None = auto.

        Returns:
            List of num_crops arrays, each [B, target_size, target_size, 3] uint8
        """
        import math

        batch_size = len(sizes)
        max_h = int(np.max(heights))
        max_w = int(np.max(widths))

        # Ensure dtypes
        sizes_u64 = sizes if sizes.dtype == np.uint64 else np.ascontiguousarray(sizes, dtype=np.uint64)
        heights_u32 = heights if heights.dtype == np.uint32 else np.ascontiguousarray(heights, dtype=np.uint32)
        widths_u32 = widths if widths.dtype == np.uint32 else np.ascontiguousarray(widths, dtype=np.uint32)
        heights_i32 = heights if heights.dtype == np.int32 else np.ascontiguousarray(heights, dtype=np.int32)
        widths_i32 = widths if widths.dtype == np.int32 else np.ascontiguousarray(widths, dtype=np.int32)

        log_ratio_min = math.log(ratio[0])
        log_ratio_max = math.log(ratio[1])

        # Generate crop params for each crop view
        all_crop_params = np.zeros((num_crops, batch_size, 4), dtype=np.int32)
        for c in range(num_crops):
            if seeds is not None and seeds[c] is not None:
                self._seed_counter += 1
                batch_seed = seeds[c] + self._seed_counter
            else:
                self._seed_counter += 1
                batch_seed = self._seed_counter

            all_crop_params[c] = _generate_random_crop_params_batch(
                widths_i32, heights_i32,
                scale[0], scale[1],
                log_ratio_min, log_ratio_max,
                batch_seed,
            )

        # Allocate buffers
        temp_buffer = self._ensure_temp_buffer(batch_size, max_h, max_w)

        # Multi-crop destination: [num_crops, B, target_size, target_size, 3]
        if (self._multi_crop_buffer is None or
            self._multi_crop_buffer.shape[0] < num_crops or
            self._multi_crop_buffer.shape[1] < batch_size or
            self._multi_crop_buffer.shape[2] != target_size or
            self._multi_crop_buffer.shape[3] != target_size):
            self._multi_crop_buffer = np.zeros(
                (num_crops, batch_size, target_size, target_size, 3), dtype=np.uint8)

        # Decode once + multi-crop
        self._decode_multi_crop_fn(
            data, sizes_u64, heights_u32, widths_u32,
            all_crop_params, temp_buffer, self._multi_crop_buffer,
            target_size, target_size, num_crops,
        )

        # Return list of views, one per crop
        return [self._multi_crop_buffer[c, :batch_size] for c in range(num_crops)]

    def hwc_to_chw(
        self,
        hwc: NDArray[np.uint8],
        batch_size: int | None = None,
    ) -> NDArray[np.uint8]:
        """Transpose [B, H, W, 3] → [B, 3, H, W] using parallel per-image copy.

        Uses Numba prange so each image (~150KB) is transposed in-cache.
        Much faster than torch's .permute().contiguous() on the full batch.

        Args:
            hwc: Input array [B, H, W, 3] uint8
            batch_size: Actual batch size (if hwc is a larger pre-allocated buffer)

        Returns:
            Array [B, 3, H, W] uint8 (view of pre-allocated buffer)
        """
        if batch_size is None:
            batch_size = hwc.shape[0]
        H, W = hwc.shape[1], hwc.shape[2]
        chw_buffer = self._ensure_chw_buffer(batch_size, H, W)
        _transpose_hwc_to_chw(hwc, chw_buffer, batch_size)
        return chw_buffer[:batch_size]

    def shutdown(self) -> None:
        """Release resources (no-op for this implementation)."""
        self._temp_buffer = None
        self._dest_buffer = None
        self._chw_buffer = None
        self._multi_crop_buffer = None
        self._multi_chw_buffer = None

    def __repr__(self) -> str:
        return f"NumbaBatchDecoder(num_threads={self.num_threads})"
