"""FFCV-style data loader with contiguous metadata table for O(1) sample access.

This module provides high-performance data loading by converting LitData streaming
datasets into a memory-mapped format optimized for training. The key insight from
the FFCV paper is that a contiguous fixed-size metadata table enables direct
pointer access without any indirection in the hot path.

File Format (V2):
    Header:      [version: u64][num_samples: u64][max_sample_size: u64] (24 bytes)
    Metadata:    [(data_ptr: u64, data_size: u64, height: u32, width: u32), ...] per sample
    Data region: Raw sample bytes (JPEG, etc.) concatenated

Key Performance Features:
    - O(1) sample access via direct pointer lookup
    - Numba JIT-compiled parallel batch loading (releases GIL)
    - Memory-mapped file with OS page cache (warm epochs are zero-copy)
    - Pre-stored JPEG dimensions (eliminates header parsing at decode time)
    - Optional MADV_HUGEPAGE hint for transparent huge pages

Usage:
    from slipstream.backends import FFCVStyleDataset, PrefetchingDataLoader

    # Build from LitData cache (first time) or load existing
    dataset = FFCVStyleDataset(cache_dir="/path/to/litdata/cache")

    # High-performance iteration with prefetching
    loader = PrefetchingDataLoader(dataset, batch_size=256, shuffle=True)
    for batch in loader:
        images = batch['data']      # [B, max_size] uint8 JPEG bytes
        sizes = batch['sizes']      # [B] actual JPEG sizes
        heights = batch['heights']  # [B] pre-stored image heights
        widths = batch['widths']    # [B] pre-stored image widths
"""

from __future__ import annotations

import ctypes
import json
import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numba import njit, prange

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _resolve_cache_dir(cache_dir: Any) -> Path:
    """Resolve cache_dir to a Path, handling Dir objects from LitData.

    Args:
        cache_dir: Path-like, string, or LitData Dir object

    Returns:
        Resolved Path object

    Raises:
        TypeError: If cache_dir type is not supported
    """
    # Handle LitData Dir objects (have .path attribute)
    if hasattr(cache_dir, "path"):
        return Path(cache_dir.path)

    # Handle string or Path
    if isinstance(cache_dir, (str, Path)):
        return Path(cache_dir)

    raise TypeError(
        f"cache_dir must be a string, Path, or LitData Dir object, "
        f"got {type(cache_dir).__name__}"
    )

# =============================================================================
# File Format Constants
# =============================================================================

# V2 metadata format: stores JPEG dimensions for zero-overhead decode
METADATA_DTYPE_V2 = np.dtype([
    ('data_ptr', '<u8'),   # 64-bit pointer into data region
    ('data_size', '<u8'),  # 64-bit size of sample data
    ('height', '<u4'),     # 32-bit image height
    ('width', '<u4'),      # 32-bit image width
])

# V1 metadata format (legacy, without dimensions)
METADATA_DTYPE_V1 = np.dtype([
    ('data_ptr', '<u8'),   # 64-bit pointer into data region
    ('data_size', '<u8'),  # 64-bit size
])

# Current version uses V2 format
METADATA_DTYPE = METADATA_DTYPE_V2

# Header layout: [version: u64][num_samples: u64][max_sample_size: u64]
HEADER_SIZE = 24  # 3 * 8 bytes
FILE_VERSION = 2

# Default FFCV-style file name
FFCV_STYLE_FILENAME = ".ffcv_style.bin"

# =============================================================================
# LitData Format Constants
# =============================================================================

# LitData chunk item header size (bytes before actual sample data)
# This is the serialization overhead added by LitData per sample.
# Structure: 4 bytes field count + variable field metadata
# For single-field (image only) samples, this is typically 16 bytes.
DEFAULT_LITDATA_HEADER_SIZE = 16


# =============================================================================
# madvise Constants and Helpers
# =============================================================================

MADV_NORMAL = 0
MADV_SEQUENTIAL = 2
MADV_WILLNEED = 3
MADV_HUGEPAGE = 14  # Linux transparent huge pages


def _apply_madvise(mmap_array: np.ndarray, advice: int) -> bool:
    """Apply madvise hint to a numpy memmap array.

    madvise is a Linux system call that advises the kernel about expected
    memory access patterns. This can significantly improve performance
    for large memory-mapped files.

    Args:
        mmap_array: A numpy array backed by mmap
        advice: madvise constant (e.g., MADV_HUGEPAGE, MADV_SEQUENTIAL)

    Returns:
        True if successful, False otherwise
    """
    try:
        libc = ctypes.CDLL(None)
        ptr = mmap_array.ctypes.data
        size = mmap_array.nbytes
        result = libc.madvise(
            ctypes.c_void_p(ptr),
            ctypes.c_size_t(size),
            ctypes.c_int(advice),
        )
        return result == 0
    except Exception:
        return False


# =============================================================================
# JPEG Parsing Utilities
# =============================================================================

def find_jpeg_end(data: np.ndarray, start: int, max_len: int) -> int:
    """Find actual JPEG size by locating FFD9 end marker.

    LitData may pad JPEG data to fixed sizes. This function finds the
    actual JPEG end by searching for the EOI (End Of Image) marker.

    Args:
        data: The chunk data array (uint8)
        start: Start offset of JPEG data
        max_len: Maximum length to search (padded size)

    Returns:
        Actual JPEG size in bytes (including FFD9 marker)
    """
    jpeg_bytes = bytes(data[start:start + max_len])
    eoi_index = jpeg_bytes.find(b'\xff\xd9')
    if eoi_index != -1:
        return eoi_index + 2  # Include the 2-byte EOI marker
    return max_len


def read_jpeg_dimensions(jpeg_data: np.ndarray | bytes) -> tuple[int, int]:
    """Read JPEG dimensions from header without full decode.

    Parses JPEG markers to find SOF (Start of Frame) which contains
    the image dimensions. Much faster than a full decode.

    Args:
        jpeg_data: Raw JPEG bytes (numpy array or bytes)

    Returns:
        (width, height) tuple, or (0, 0) if parsing fails
    """
    if isinstance(jpeg_data, np.ndarray):
        data = bytes(jpeg_data)
    else:
        data = jpeg_data

    n = len(data)
    if n < 2 or data[0] != 0xFF or data[1] != 0xD8:
        return (0, 0)  # Not a valid JPEG

    i = 2
    while i < n - 1:
        if data[i] != 0xFF:
            i += 1
            continue

        marker = data[i + 1]

        # Skip padding bytes (0xFF followed by 0xFF)
        if marker == 0xFF:
            i += 1
            continue

        # Skip standalone markers (no length field)
        if marker in (0x00, 0x01, 0xD0, 0xD1, 0xD2, 0xD3,
                      0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9):
            i += 2
            continue

        # SOF markers (Start of Frame): 0xC0-0xCF except 0xC4, 0xC8, 0xCC
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            # SOF layout: marker(2) + length(2) + precision(1) + height(2) + width(2)
            if i + 9 < n:
                height = (data[i + 5] << 8) | data[i + 6]
                width = (data[i + 7] << 8) | data[i + 8]
                return (width, height)
            return (0, 0)

        # Skip other segments using their length field
        if i + 3 < n:
            seg_len = (data[i + 2] << 8) | data[i + 3]
            i += 2 + seg_len
        else:
            break

    return (0, 0)


# =============================================================================
# Numba JIT-Compiled Batch Loaders
# =============================================================================

@njit(nogil=True, parallel=True, cache=True, fastmath=True, error_model='numpy')
def _load_batch_parallel(
    batch_indices: np.ndarray,
    metadata: np.ndarray,
    data_region: np.ndarray,
    destination: np.ndarray,
    sizes: np.ndarray,
) -> None:
    """Load a batch using FFCV-style direct pointer access (parallel).

    The ENTIRE loop is JIT-compiled with prange for true parallelism
    (no Python GIL). Each sample access is O(1) - just metadata[sample_id].

    Args:
        batch_indices: Global sample indices to load [B]
        metadata: Contiguous metadata table [N] with structured dtype
        data_region: Contiguous data region (all sample bytes)
        destination: Pre-allocated output buffer [B, max_sample_size]
        sizes: Output array to store actual size of each sample [B]
    """
    batch_size = len(batch_indices)

    for i in prange(batch_size):
        sample_id = batch_indices[i]

        # Direct O(1) access - no indirection!
        data_ptr = metadata[sample_id]['data_ptr']
        data_size = metadata[sample_id]['data_size']

        # Copy data to destination
        destination[i, :data_size] = data_region[data_ptr:data_ptr + data_size]
        sizes[i] = data_size


@njit(nogil=True, cache=True, fastmath=True, error_model='numpy')
def _load_batch_sequential(
    batch_indices: np.ndarray,
    metadata: np.ndarray,
    data_region: np.ndarray,
    destination: np.ndarray,
    sizes: np.ndarray,
) -> None:
    """Load a batch sequentially (for comparison/debugging)."""
    batch_size = len(batch_indices)

    for i in range(batch_size):
        sample_id = batch_indices[i]
        data_ptr = metadata[sample_id]['data_ptr']
        data_size = metadata[sample_id]['data_size']
        destination[i, :data_size] = data_region[data_ptr:data_ptr + data_size]
        sizes[i] = data_size


# =============================================================================
# FFCVStyleDataset
# =============================================================================

class FFCVStyleDataset:
    """High-performance dataset using FFCV-style contiguous metadata table.

    This class converts LitData streaming datasets into a memory-mapped format
    optimized for training. The key optimization is a contiguous metadata table
    that enables O(1) sample access without any indirection.

    On first load (or when force_rebuild=True):
        1. Reads LitData index.json and all chunk files
        2. Extracts JPEG data and dimensions from each sample
        3. Writes: [header][metadata_table][data_region] to .ffcv_style.bin

    On subsequent loads:
        1. Memory-maps the .ffcv_style.bin file
        2. Metadata table at fixed offset (HEADER_SIZE)
        3. Data region at HEADER_SIZE + num_samples * METADATA_DTYPE.itemsize

    Attributes:
        cache_dir: Path to LitData cache directory
        num_samples: Total number of samples
        max_sample_size: Maximum sample size in bytes
        parallel: Whether to use parallel batch loading
    """

    def __init__(
        self,
        cache_dir: str | Path | Any,  # Any to accept LitData Dir objects
        max_sample_size: int | None = None,
        parallel: bool = True,
        force_rebuild: bool = False,
        use_hugepages: bool = False,
        litdata_header_size: int = DEFAULT_LITDATA_HEADER_SIZE,
        verbose: bool = True,
    ) -> None:
        """Initialize the FFCV-style dataset.

        Args:
            cache_dir: Path to local LitData cache directory. Can be a string,
                      Path, or LitData Dir object (will extract .path attribute).
            max_sample_size: Maximum sample size in bytes (auto-detected if None)
            parallel: Use parallel Numba batch loading (recommended)
            force_rebuild: Force rebuilding even if .ffcv_style.bin exists
            use_hugepages: Apply MADV_HUGEPAGE hint (Linux only)
            litdata_header_size: Size of LitData per-item header in bytes.
                                 Default is 16 bytes for single-field samples.
            verbose: Print progress messages during build/load
        """
        self.cache_dir = _resolve_cache_dir(cache_dir)
        self.parallel = parallel
        self.use_hugepages = use_hugepages
        self.litdata_header_size = litdata_header_size
        self.verbose = verbose

        ffcv_path = self.cache_dir / FFCV_STYLE_FILENAME

        if not force_rebuild and ffcv_path.exists():
            self._load_ffcv_style()
        else:
            self._build_ffcv_style(max_sample_size)

        # Pre-allocate buffers for batch loading
        self._batch_buffer: np.ndarray | None = None
        self._sizes_buffer: np.ndarray | None = None
        self._current_batch_size = 0

    def _log(self, msg: str) -> None:
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(msg)

    def _build_ffcv_style(self, max_sample_size: int | None) -> None:
        """Build FFCV-style file from LitData chunks.

        Uses a two-pass streaming approach to avoid OOM:
        1. First pass: scan chunks to get sample sizes AND JPEG dimensions
        2. Second pass: write header, metadata (with dims), then stream sample data

        This is necessary because we need sample sizes to compute data pointers
        for the metadata table, which must be written before the data region.
        """
        index_path = self.cache_dir / "index.json"
        if not index_path.exists():
            raise FileNotFoundError(
                f"LitData index not found: {index_path}\n"
                "Ensure the cache_dir points to a valid LitData cache."
            )

        with open(index_path) as f:
            index = json.load(f)

        chunks_data = index.get("chunks", [])
        if not chunks_data:
            raise ValueError("Invalid LitData index: missing 'chunks' key")

        chunk_files = [c["filename"] for c in chunks_data]
        chunk_sizes = [c["chunk_size"] for c in chunks_data]
        total_samples = sum(chunk_sizes)

        self._log(
            f"Building FFCV-style V2 file: {total_samples:,} samples "
            f"from {len(chunk_files)} chunks"
        )

        # PASS 1: Scan chunks to get actual JPEG sizes and dimensions
        self._log("Pass 1: Scanning for JPEG sizes and dimensions...")
        sample_sizes = np.zeros(total_samples, dtype=np.uint64)
        sample_heights = np.zeros(total_samples, dtype=np.uint32)
        sample_widths = np.zeros(total_samples, dtype=np.uint32)
        detected_max_size = 0
        sample_idx = 0

        for chunk_idx, chunk_file in enumerate(chunk_files):
            chunk_path = self.cache_dir / chunk_file
            if not chunk_path.exists():
                raise FileNotFoundError(f"Chunk not found: {chunk_path}")

            # Memory-map chunk (no RAM copy)
            chunk_data = np.memmap(chunk_path, dtype=np.uint8, mode='r')
            num_items = int(np.frombuffer(chunk_data[:4], dtype=np.uint32)[0])

            for local_idx in range(num_items):
                # Read item offset from chunk header
                offset_pos = 4 + local_idx * 4
                item_begin = int(np.frombuffer(
                    chunk_data[offset_pos:offset_pos + 4], dtype=np.uint32
                )[0])
                item_end = int(np.frombuffer(
                    chunk_data[offset_pos + 4:offset_pos + 8], dtype=np.uint32
                )[0])

                # JPEG data starts after LitData header
                jpeg_start = item_begin + self.litdata_header_size
                padded_size = item_end - item_begin - self.litdata_header_size

                # Find actual JPEG size by scanning for FFD9 end marker
                actual_size = find_jpeg_end(chunk_data, jpeg_start, padded_size)

                # Extract JPEG dimensions from header
                jpeg_bytes = np.array(chunk_data[jpeg_start:jpeg_start + actual_size])
                width, height = read_jpeg_dimensions(jpeg_bytes)

                sample_sizes[sample_idx] = actual_size
                sample_heights[sample_idx] = height
                sample_widths[sample_idx] = width
                detected_max_size = max(detected_max_size, actual_size)
                sample_idx += 1

            del chunk_data  # Release mmap

            if (chunk_idx + 1) % 10 == 0 or chunk_idx == 0:
                self._log(
                    f"  Scanned {chunk_idx + 1}/{len(chunk_files)} chunks "
                    f"({sample_idx:,} samples)"
                )

        # Use provided max_sample_size or detected + 20% buffer
        if max_sample_size is None:
            max_sample_size = int(detected_max_size * 1.2)

        self._log(f"Max sample size: {max_sample_size:,} bytes")
        self._log(
            f"Image dimensions: {sample_heights.min()}-{sample_heights.max()} x "
            f"{sample_widths.min()}-{sample_widths.max()}"
        )

        # Build metadata table with data pointers and dimensions
        metadata = np.zeros(total_samples, dtype=METADATA_DTYPE)
        current_ptr = 0
        for i in range(total_samples):
            metadata[i]['data_ptr'] = current_ptr
            metadata[i]['data_size'] = sample_sizes[i]
            metadata[i]['height'] = sample_heights[i]
            metadata[i]['width'] = sample_widths[i]
            current_ptr += sample_sizes[i]

        total_data_size = current_ptr
        self._log(f"Total data size: {total_data_size / 1e9:.2f} GB")

        # PASS 2: Write file (stream data to avoid OOM)
        ffcv_path = self.cache_dir / FFCV_STYLE_FILENAME
        self._log(f"Pass 2: Writing FFCV-style V2 file: {ffcv_path}")

        with open(ffcv_path, 'wb') as f:
            # Header: [version, num_samples, max_sample_size]
            header = np.array([FILE_VERSION, total_samples, max_sample_size], dtype=np.uint64)
            header.tofile(f)

            # Metadata table (contiguous, includes dimensions)
            metadata.tofile(f)

            # Data region: stream from chunks
            sample_idx = 0
            for chunk_idx, chunk_file in enumerate(chunk_files):
                chunk_path = self.cache_dir / chunk_file
                chunk_data = np.memmap(chunk_path, dtype=np.uint8, mode='r')
                num_items = int(np.frombuffer(chunk_data[:4], dtype=np.uint32)[0])

                for local_idx in range(num_items):
                    offset_pos = 4 + local_idx * 4
                    item_begin = int(np.frombuffer(
                        chunk_data[offset_pos:offset_pos + 4], dtype=np.uint32
                    )[0])

                    jpeg_start = item_begin + self.litdata_header_size
                    actual_size = int(sample_sizes[sample_idx])

                    # Write only actual JPEG bytes (no padding)
                    f.write(bytes(chunk_data[jpeg_start:jpeg_start + actual_size]))
                    sample_idx += 1

                del chunk_data

                if (chunk_idx + 1) % 40 == 0:
                    self._log(f"  Wrote {chunk_idx + 1}/{len(chunk_files)} chunks")

        file_size = os.path.getsize(ffcv_path)
        self._log(f"FFCV-style V2 file created: {file_size / 1e9:.2f} GB")

        # Load the file we just created
        self._load_ffcv_style()

    def _load_ffcv_style(self) -> None:
        """Load (memory-map) the FFCV-style file.

        Supports both V1 (no dimensions) and V2 (with dimensions) formats.
        V1 files will trigger a warning recommending rebuild.
        """
        ffcv_path = self.cache_dir / FFCV_STYLE_FILENAME

        # Memory-map entire file
        self._file_mmap = np.memmap(ffcv_path, dtype=np.uint8, mode='r')

        # Read first 8 bytes to check version
        first_u64 = int(np.frombuffer(self._file_mmap[:8], dtype=np.uint64)[0])

        if first_u64 == FILE_VERSION:
            # V2 format: [version][num_samples][max_sample_size]
            header = np.frombuffer(self._file_mmap[:HEADER_SIZE], dtype=np.uint64)
            self._file_version = int(header[0])
            self.num_samples = int(header[1])
            self.max_sample_size = int(header[2])
            metadata_dtype = METADATA_DTYPE_V2
            header_size = HEADER_SIZE
        else:
            # V1 format (legacy): first u64 is num_samples (large number)
            # NOTE: We no longer auto-delete V1 files. User must explicitly rebuild.
            warnings.warn(
                f"Detected V1 format FFCV-style file at {ffcv_path}. "
                "V1 format lacks pre-stored image dimensions, which reduces "
                "decode performance. To upgrade to V2, delete the file and "
                "reinitialize with force_rebuild=True:\n"
                f"  rm {ffcv_path}\n"
                f"  dataset = FFCVStyleDataset('{self.cache_dir}', force_rebuild=True)",
                UserWarning,
                stacklevel=2,
            )
            # Load V1 anyway (for backwards compatibility)
            header = np.frombuffer(self._file_mmap[:16], dtype=np.uint64)
            self._file_version = 1
            self.num_samples = int(header[0])
            self.max_sample_size = int(header[1])
            metadata_dtype = METADATA_DTYPE_V1
            header_size = 16  # V1 header is only 16 bytes

        # Metadata table starts right after header
        metadata_size = self.num_samples * metadata_dtype.itemsize
        metadata_end = header_size + metadata_size

        # View metadata as structured array
        self.metadata = np.frombuffer(
            self._file_mmap[header_size:metadata_end],
            dtype=metadata_dtype,
        )

        # Data region starts after metadata
        self.data_region_offset = metadata_end
        self.data_region = self._file_mmap[metadata_end:]

        # Pre-convert to contiguous arrays for Numba
        self._metadata_array = np.ascontiguousarray(self.metadata)
        self._data_array = np.asarray(self.data_region)

        # Extract height/width arrays for fast access (V2 only)
        if self._file_version >= 2:
            self._heights = np.ascontiguousarray(self.metadata['height'])
            self._widths = np.ascontiguousarray(self.metadata['width'])
        else:
            # V1: heights/widths must be parsed from JPEG headers at decode time
            self._heights = None
            self._widths = None

        # Apply madvise hints if requested
        if self.use_hugepages:
            if _apply_madvise(self._file_mmap, MADV_HUGEPAGE):
                self._log("Applied MADV_HUGEPAGE hint for transparent huge pages")
            else:
                self._log("Warning: MADV_HUGEPAGE failed (may not be supported)")

        self._log(
            f"Loaded FFCV-style V{self._file_version} file: "
            f"{self.num_samples:,} samples, max_size={self.max_sample_size:,}, "
            f"data_region={len(self.data_region) / 1e9:.2f} GB"
        )

    def _ensure_buffers(self, batch_size: int) -> None:
        """Ensure pre-allocated buffers are large enough."""
        if self._current_batch_size < batch_size:
            self._batch_buffer = np.zeros(
                (batch_size, self.max_sample_size), dtype=np.uint8
            )
            self._sizes_buffer = np.zeros(batch_size, dtype=np.uint64)
            self._current_batch_size = batch_size

    def load_batch_raw(
        self, indices: NDArray[np.int64]
    ) -> tuple[NDArray[np.uint8], NDArray[np.uint64]]:
        """Load a batch of raw samples using FFCV-style direct pointer access.

        This is the hot path - Numba JIT-compiled with nogil=True for
        true parallelism.

        Args:
            indices: Array of global sample indices

        Returns:
            (data, sizes) tuple:
            - data: [batch_size, max_sample_size] uint8 array
            - sizes: [batch_size] actual size of each sample
        """
        batch_size = len(indices)
        self._ensure_buffers(batch_size)

        dest = self._batch_buffer[:batch_size]
        sizes = self._sizes_buffer[:batch_size]

        # Ensure indices are int64 for Numba
        if indices.dtype != np.int64:
            indices = indices.astype(np.int64)

        # JIT-compiled batch load
        if self.parallel:
            _load_batch_parallel(
                indices,
                self._metadata_array,
                self._data_array,
                dest,
                sizes,
            )
        else:
            _load_batch_sequential(
                indices,
                self._metadata_array,
                self._data_array,
                dest,
                sizes,
            )

        return dest, sizes

    def load_batch_with_dims(
        self, indices: NDArray[np.int64]
    ) -> tuple[NDArray[np.uint8], NDArray[np.uint64], NDArray[np.uint32], NDArray[np.uint32]]:
        """Load a batch with pre-stored JPEG dimensions.

        This is the key V2 optimization - dimensions are stored in metadata
        during file creation, eliminating JPEG header parsing at decode time.

        Args:
            indices: Array of global sample indices

        Returns:
            (data, sizes, heights, widths) tuple:
            - data: [batch_size, max_sample_size] uint8 JPEG bytes
            - sizes: [batch_size] actual JPEG sizes
            - heights: [batch_size] image heights
            - widths: [batch_size] image widths

        Raises:
            RuntimeError: If file is V1 format (no stored dimensions)
        """
        if self._heights is None or self._widths is None:
            raise RuntimeError(
                "load_batch_with_dims() requires V2 format with pre-stored dimensions. "
                "Rebuild the file with force_rebuild=True to upgrade from V1."
            )

        data, sizes = self.load_batch_raw(indices)

        # Fast O(1) lookup of pre-stored dimensions
        heights = self._heights[indices]
        widths = self._widths[indices]

        return data, sizes, heights, widths

    def get_sample_dims(self, idx: int) -> tuple[int, int]:
        """Get pre-stored dimensions for a single sample.

        Args:
            idx: Sample index

        Returns:
            (height, width) tuple

        Raises:
            RuntimeError: If file is V1 format
        """
        if self._heights is None or self._widths is None:
            raise RuntimeError(
                "get_sample_dims() requires V2 format. "
                "Rebuild with force_rebuild=True."
            )
        return int(self._heights[idx]), int(self._widths[idx])

    def __len__(self) -> int:
        return self.num_samples

    def create_loader(
        self,
        batch_size: int = 256,
        shuffle: bool = False,
        drop_last: bool = False,
    ) -> FFCVStyleDataLoader:
        """Create a simple iterator over batches."""
        return FFCVStyleDataLoader(self, batch_size, shuffle, drop_last)

    def __repr__(self) -> str:
        return (
            f"FFCVStyleDataset(\n"
            f"    cache_dir='{self.cache_dir}',\n"
            f"    num_samples={self.num_samples:,},\n"
            f"    max_sample_size={self.max_sample_size:,},\n"
            f"    version={self._file_version},\n"
            f"    parallel={self.parallel},\n"
            f")"
        )


# =============================================================================
# Data Loaders
# =============================================================================

class FFCVStyleDataLoader:
    """Simple batch iterator for FFCVStyleDataset."""

    def __init__(
        self,
        dataset: FFCVStyleDataset,
        batch_size: int = 256,
        shuffle: bool = False,
        drop_last: bool = False,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):
        indices = np.arange(len(self.dataset), dtype=np.int64)
        if self.shuffle:
            np.random.shuffle(indices)

        num_batches = len(indices) // self.batch_size
        if not self.drop_last and len(indices) % self.batch_size != 0:
            num_batches += 1

        for i in range(num_batches):
            start = i * self.batch_size
            end = min(start + self.batch_size, len(indices))
            batch_indices = indices[start:end]

            data, sizes = self.dataset.load_batch_raw(batch_indices)
            yield {
                'data': data[:len(batch_indices)],
                'sizes': sizes[:len(batch_indices)],
                'indices': batch_indices,
            }

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


class PrefetchingDataLoader:
    """Data loader with background prefetching using threading.

    This mimics FFCV's EpochIterator approach:
    - Background thread runs batch loading with Numba (releases GIL)
    - Main thread consumes batches
    - Pre-allocated memory banks avoid per-batch allocation

    V2 enhancement: Also returns pre-stored JPEG dimensions for zero-overhead decode.

    Attributes:
        dataset: The FFCVStyleDataset to load from
        batch_size: Number of samples per batch
        shuffle: Shuffle indices at the start of each epoch
        drop_last: Drop incomplete final batch
        batches_ahead: Number of batches to prefetch
    """

    def __init__(
        self,
        dataset: FFCVStyleDataset,
        batch_size: int = 256,
        shuffle: bool = False,
        drop_last: bool = False,
        batches_ahead: int = 3,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.batches_ahead = batches_ahead

        # Pre-allocate memory banks (batches_ahead + 2 slots like FFCV)
        num_slots = batches_ahead + 2
        self._data_banks = [
            np.zeros((batch_size, dataset.max_sample_size), dtype=np.uint8)
            for _ in range(num_slots)
        ]
        self._size_banks = [
            np.zeros(batch_size, dtype=np.uint64)
            for _ in range(num_slots)
        ]

        # Dimension banks for V2 zero-overhead decode
        if dataset._heights is not None:
            self._height_banks = [
                np.zeros(batch_size, dtype=np.uint32)
                for _ in range(num_slots)
            ]
            self._width_banks = [
                np.zeros(batch_size, dtype=np.uint32)
                for _ in range(num_slots)
            ]
        else:
            self._height_banks = None
            self._width_banks = None

    def __iter__(self):
        import queue
        import threading

        indices = np.arange(len(self.dataset), dtype=np.int64)
        if self.shuffle:
            np.random.shuffle(indices)

        num_batches = len(indices) // self.batch_size
        if not self.drop_last and len(indices) % self.batch_size != 0:
            num_batches += 1

        output_queue: queue.Queue = queue.Queue(maxsize=self.batches_ahead)
        stop_event = threading.Event()
        num_slots = len(self._data_banks)

        def epoch_worker():
            """Background thread - mimics FFCV's EpochIterator.run()"""
            current_slot = 0

            for batch_idx in range(num_batches):
                if stop_event.is_set():
                    break

                start = batch_idx * self.batch_size
                end = min(start + self.batch_size, len(indices))
                batch_indices = indices[start:end]
                actual_batch_size = len(batch_indices)

                # Get pre-allocated buffers for this slot
                dest = self._data_banks[current_slot]
                sizes = self._size_banks[current_slot]

                # Load directly into pre-allocated buffer (no copy!)
                # This runs with nogil=True, so GIL is released
                if self.dataset.parallel:
                    _load_batch_parallel(
                        batch_indices,
                        self.dataset._metadata_array,
                        self.dataset._data_array,
                        dest,
                        sizes,
                    )
                else:
                    _load_batch_sequential(
                        batch_indices,
                        self.dataset._metadata_array,
                        self.dataset._data_array,
                        dest,
                        sizes,
                    )

                # Copy pre-stored dimensions (fast O(1) lookup)
                if self._height_banks is not None:
                    heights = self._height_banks[current_slot]
                    widths = self._width_banks[current_slot]
                    heights[:actual_batch_size] = self.dataset._heights[batch_indices]
                    widths[:actual_batch_size] = self.dataset._widths[batch_indices]

                # Put (slot, batch_size, batch_indices) in queue
                output_queue.put((current_slot, actual_batch_size, batch_indices))
                current_slot = (current_slot + 1) % num_slots

            # Signal end
            output_queue.put(None)

        # Start background thread
        worker = threading.Thread(target=epoch_worker, daemon=True)
        worker.start()

        try:
            while True:
                result = output_queue.get()
                if result is None:
                    break
                slot, actual_size, batch_indices = result

                batch = {
                    'data': self._data_banks[slot][:actual_size],
                    'sizes': self._size_banks[slot][:actual_size],
                    'indices': batch_indices,
                }

                if self._height_banks is not None:
                    batch['heights'] = self._height_banks[slot][:actual_size]
                    batch['widths'] = self._width_banks[slot][:actual_size]

                yield batch
        finally:
            stop_event.set()
            worker.join(timeout=1.0)

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __repr__(self) -> str:
        return (
            f"PrefetchingDataLoader(\n"
            f"    dataset={self.dataset!r},\n"
            f"    batch_size={self.batch_size},\n"
            f"    shuffle={self.shuffle},\n"
            f"    batches_ahead={self.batches_ahead},\n"
            f")"
        )


__all__ = [
    "FFCVStyleDataset",
    "FFCVStyleDataLoader",
    "PrefetchingDataLoader",
    "METADATA_DTYPE",
    "METADATA_DTYPE_V1",
    "METADATA_DTYPE_V2",
    "read_jpeg_dimensions",
    "find_jpeg_end",
]
