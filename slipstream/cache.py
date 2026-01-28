"""Optimized cache for high-performance data loading.

This module provides OptimizedCache, which converts any SlipstreamDataset into
a memory-mapped format optimized for training. The key insight is that different
field types benefit from different storage strategies:

- ImageBytes: Contiguous data with metadata table for O(1) access
- Numeric (int, float): Simple numpy arrays with mmap
- Strings: Concatenated bytes with offset table

All storage uses memory-mapping for zero-copy reads after the first epoch
(OS page cache handles warm epochs).

For LitData-backed datasets, building uses a fast path that reads directly
from chunk files using mmap, avoiding the slow `dataset[i]` iteration.

Usage:
    # Build from dataset (automatic in SlipstreamLoader)
    cache = OptimizedCache.build(dataset, output_dir)

    # Load existing cache
    cache = OptimizedCache.load(cache_dir)

    # Load a batch
    batch = cache.load_batch(indices, fields=['image', 'label'])
"""

from __future__ import annotations

import hashlib
import json
import random
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numba import njit, prange
from tqdm import tqdm

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from slipstream.dataset import SlipstreamDataset


# =============================================================================
# Constants
# =============================================================================

CACHE_SUBDIR = ".slipstream"
MANIFEST_FILE = "manifest.json"
CACHE_VERSION = 1

# Metadata dtype for variable-size fields (images, bytes, strings)
VARIABLE_METADATA_DTYPE = np.dtype([
    ('data_ptr', '<u8'),   # 64-bit pointer into data region
    ('data_size', '<u8'),  # 64-bit size of data
    ('height', '<u4'),     # 32-bit height (for images, 0 otherwise)
    ('width', '<u4'),      # 32-bit width (for images, 0 otherwise)
])


# =============================================================================
# Numba JIT Batch Loaders
# =============================================================================

@njit(nogil=True, parallel=True, cache=True, fastmath=True, error_model='numpy')
def _load_variable_batch_parallel(
    batch_indices: np.ndarray,
    metadata: np.ndarray,
    data_region: np.ndarray,
    destination: np.ndarray,
    sizes: np.ndarray,
) -> None:
    """Load a batch of variable-size data using parallel O(1) access."""
    batch_size = len(batch_indices)

    for i in prange(batch_size):
        sample_id = batch_indices[i]
        data_ptr = metadata[sample_id]['data_ptr']
        data_size = metadata[sample_id]['data_size']
        destination[i, :data_size] = data_region[data_ptr:data_ptr + data_size]
        sizes[i] = data_size


@njit(nogil=True, cache=True, fastmath=True, error_model='numpy')
def _load_variable_batch_sequential(
    batch_indices: np.ndarray,
    metadata: np.ndarray,
    data_region: np.ndarray,
    destination: np.ndarray,
    sizes: np.ndarray,
) -> None:
    """Load a batch of variable-size data sequentially."""
    batch_size = len(batch_indices)

    for i in range(batch_size):
        sample_id = batch_indices[i]
        data_ptr = metadata[sample_id]['data_ptr']
        data_size = metadata[sample_id]['data_size']
        destination[i, :data_size] = data_region[data_ptr:data_ptr + data_size]
        sizes[i] = data_size


# =============================================================================
# JPEG Utilities
# =============================================================================

def find_jpeg_end(data: bytes | np.ndarray, max_len: int) -> int:
    """Find actual JPEG size by locating FFD9 end marker."""
    if isinstance(data, np.ndarray):
        data = bytes(data[:max_len])
    else:
        data = data[:max_len]
    eoi_index = data.find(b'\xff\xd9')
    if eoi_index != -1:
        return eoi_index + 2
    return max_len


def read_jpeg_dimensions(jpeg_data: bytes | np.ndarray) -> tuple[int, int]:
    """Read JPEG dimensions from header without full decode.

    Returns:
        (width, height) tuple, or (0, 0) if parsing fails
    """
    if isinstance(jpeg_data, np.ndarray):
        jpeg_data = bytes(jpeg_data)

    n = len(jpeg_data)
    if n < 2 or jpeg_data[0] != 0xFF or jpeg_data[1] != 0xD8:
        return (0, 0)

    i = 2
    while i < n - 1:
        if jpeg_data[i] != 0xFF:
            i += 1
            continue

        marker = jpeg_data[i + 1]

        if marker == 0xFF:
            i += 1
            continue

        if marker in (0x00, 0x01, 0xD0, 0xD1, 0xD2, 0xD3,
                      0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9):
            i += 2
            continue

        # SOF markers contain dimensions
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            if i + 9 < n:
                height = (jpeg_data[i + 5] << 8) | jpeg_data[i + 6]
                width = (jpeg_data[i + 7] << 8) | jpeg_data[i + 8]
                return (width, height)
            return (0, 0)

        if i + 3 < n:
            seg_len = (jpeg_data[i + 2] << 8) | jpeg_data[i + 3]
            i += 2 + seg_len
        else:
            break

    return (0, 0)


# =============================================================================
# LitData Chunk Reading Utilities
# =============================================================================

def _deserialize_int(data: bytes) -> int:
    """Deserialize LitData IntegerSerializer format."""
    return int(np.frombuffer(data, dtype=np.int64)[0])


def _deserialize_str(data: bytes) -> str:
    """Deserialize LitData StringSerializer format."""
    return data.decode('utf-8')


def _deserialize_float(data: bytes) -> float:
    """Deserialize LitData FloatSerializer format."""
    return float(np.frombuffer(data, dtype=np.float64)[0])


# Map LitData data_format strings to deserializers
LITDATA_DESERIALIZERS = {
    'int': _deserialize_int,
    'str': _deserialize_str,
    'float': _deserialize_float,
    # 'jpeg' and 'bytes' are handled specially (raw bytes)
}


def _read_all_fields_from_litdata_chunks(
    litdata_cache_dir: Path,
    field_types: dict[str, str],
    dataset: SlipstreamDataset,
    verbose: bool = True,
    num_download_workers: int = 8,
) -> dict[str, list]:
    """Read ALL fields directly from LitData chunks (full fast path).

    LitData chunk format per item:
        [size_header: uint32 * N][field_0_data][field_1_data]...[field_N-1_data]

    We read the size header to find field offsets, then extract each field
    without going through dataset[i].

    Uses parallel downloads: a ThreadPoolExecutor downloads multiple chunks
    simultaneously while the main thread reads completed chunks in order.

    Args:
        litdata_cache_dir: Path to LitData cache
        field_types: Dict mapping field names to types (from dataset.field_types)
        dataset: The SlipstreamDataset (used to trigger chunk downloads)
        verbose: Show progress
        num_download_workers: Number of parallel download threads (default: 8)

    Returns:
        Dict mapping field names to lists of values
    """
    index_path = litdata_cache_dir / "index.json"
    with open(index_path) as f:
        index = json.load(f)

    # Get data format (serializer types) from config
    config = index.get("config", {})
    data_format = config.get("data_format", [])

    if not data_format:
        raise ValueError("LitData index missing 'config.data_format'")

    chunks_data = index.get("chunks", [])
    chunk_files = [c["filename"] for c in chunks_data]
    chunk_sizes = [c["chunk_size"] for c in chunks_data]

    num_fields = len(data_format)
    size_header_bytes = num_fields * 4

    # Map field names to indices based on order
    # LitData stores fields in the order they appear in the sample dict
    field_names = list(field_types.keys())
    if len(field_names) != num_fields:
        raise ValueError(
            f"Field count mismatch: dataset has {len(field_names)} fields, "
            f"but LitData config has {num_fields}"
        )

    if verbose:
        print(f"  Reading {num_fields} fields from {len(chunk_files)} chunks")
        print(f"  Data format: {data_format}")
        print(f"  Using {num_download_workers} parallel download workers")

    # Initialize storage
    samples_by_field: dict[str, list] = {k: [] for k in field_names}

    # Special handling for image fields (need dimensions)
    image_metadata: dict[str, dict] = {}
    for i, (name, fmt) in enumerate(zip(field_names, data_format)):
        if fmt in ('jpeg', 'bytes') or field_types[name] == 'ImageBytes':
            image_metadata[name] = {
                'index': i,
                'sizes': [],
                'heights': [],
                'widths': [],
            }

    # Calculate first sample index for each chunk (for triggering downloads)
    chunk_first_sample = [0]
    for size in chunk_sizes[:-1]:
        chunk_first_sample.append(chunk_first_sample[-1] + size)

    # Ensure cache exists and set on_demand_bytes=False for chunk caching
    if dataset.cache is None:
        _ = dataset[0]
    original_on_demand = dataset.on_demand_bytes
    dataset.on_demand_bytes = False

    num_chunks = len(chunk_files)

    # Access LitData's internal downloader for direct chunk downloads
    # This bypasses dataset[i] which may have internal locks
    litdata_config = None
    try:
        # Access the underlying LitData StreamingDataset's reader config
        inner_dataset = dataset._dataset
        if hasattr(inner_dataset, '_reader') and inner_dataset._reader is not None:
            reader = inner_dataset._reader
            if hasattr(reader, 'config'):
                litdata_config = reader.config
            elif hasattr(reader, '_config') and reader._config is not None:
                litdata_config = reader._config
    except Exception:
        pass  # Fall back to dataset[i] approach

    if verbose and litdata_config is not None:
        print("  Using direct LitData chunk downloads (parallel-safe)")
    elif verbose:
        print("  Using dataset[i] fallback for downloads (may be slower)")

    def download_chunk(chunk_idx: int) -> int:
        """Download a single chunk if not already cached. Returns chunk_idx."""
        chunk_path = litdata_cache_dir / chunk_files[chunk_idx]
        if not chunk_path.exists():
            if litdata_config is not None:
                # Use LitData's direct chunk download (no locks, parallel-safe)
                litdata_config.download_chunk_from_index(chunk_idx)
            else:
                # Fallback: trigger download via sample access
                sample_idx = chunk_first_sample[chunk_idx]
                _ = dataset[sample_idx]
        return chunk_idx

    if verbose:
        pbar = tqdm(total=num_chunks, desc="  Processing chunks")
    else:
        pbar = None

    try:
        with ThreadPoolExecutor(max_workers=num_download_workers) as executor:
            # Submit all chunks for download
            # futures maps future -> chunk_idx for tracking
            futures = {
                executor.submit(download_chunk, i): i
                for i in range(num_chunks)
            }

            # Track which chunks are ready (downloaded)
            ready_chunks: set[int] = set()
            next_chunk_to_process = 0

            # Process completed downloads as they finish
            for future in as_completed(futures):
                chunk_idx = futures[future]
                try:
                    future.result()  # Raises if download failed
                    ready_chunks.add(chunk_idx)
                except Exception as e:
                    raise RuntimeError(f"Failed to download chunk {chunk_idx}: {e}") from e

                # Process all consecutive ready chunks starting from next_chunk_to_process
                while next_chunk_to_process in ready_chunks:
                    chunk_file = chunk_files[next_chunk_to_process]
                    chunk_path = litdata_cache_dir / chunk_file

                    if not chunk_path.exists():
                        raise FileNotFoundError(
                            f"Chunk {chunk_file} not found at {chunk_path} after download. "
                            f"Check max_cache_size setting."
                        )

                    chunk_data = np.memmap(chunk_path, dtype=np.uint8, mode='r')
                    num_items = int(np.frombuffer(chunk_data[:4], dtype=np.uint32)[0])

                    for local_idx in range(num_items):
                        # Get item bounds from offset table
                        offset_pos = 4 + local_idx * 4
                        item_begin = int(np.frombuffer(
                            chunk_data[offset_pos:offset_pos + 4], dtype=np.uint32
                        )[0])

                        # Read size header
                        size_header = np.frombuffer(
                            chunk_data[item_begin:item_begin + size_header_bytes],
                            dtype=np.uint32
                        )

                        # Calculate field offsets
                        field_offsets = [size_header_bytes]
                        for size in size_header[:-1]:
                            field_offsets.append(field_offsets[-1] + size)

                        # Extract each field
                        for field_idx, (name, fmt) in enumerate(zip(field_names, data_format)):
                            field_start = item_begin + field_offsets[field_idx]
                            field_size = int(size_header[field_idx])
                            field_bytes = chunk_data[field_start:field_start + field_size]

                            if fmt == 'jpeg' or field_types[name] == 'ImageBytes':
                                # For images, store raw bytes and parse dimensions
                                actual_size = find_jpeg_end(field_bytes, field_size)
                                w, h = read_jpeg_dimensions(field_bytes[:actual_size])
                                samples_by_field[name].append(bytes(field_bytes[:actual_size]))
                                image_metadata[name]['sizes'].append(actual_size)
                                image_metadata[name]['heights'].append(h)
                                image_metadata[name]['widths'].append(w)
                            elif fmt in LITDATA_DESERIALIZERS:
                                value = LITDATA_DESERIALIZERS[fmt](bytes(field_bytes))
                                samples_by_field[name].append(value)
                            else:
                                # Unknown format, store raw bytes
                                samples_by_field[name].append(bytes(field_bytes))

                    del chunk_data
                    ready_chunks.discard(next_chunk_to_process)

                    if pbar:
                        pbar.update(1)

                    next_chunk_to_process += 1

    finally:
        if pbar:
            pbar.close()
        # Restore original on_demand_bytes setting
        dataset.on_demand_bytes = original_on_demand

    # Attach image metadata to the result
    for name, meta in image_metadata.items():
        samples_by_field[f"__{name}_sizes"] = meta['sizes']
        samples_by_field[f"__{name}_heights"] = meta['heights']
        samples_by_field[f"__{name}_widths"] = meta['widths']

    return samples_by_field


# =============================================================================
# Field Storage Classes
# =============================================================================

class FieldStorage(ABC):
    """Base class for field storage."""

    field_name: str
    num_samples: int

    @abstractmethod
    def load_batch(
        self,
        indices: NDArray[np.int64],
        parallel: bool = True,
    ) -> dict[str, Any]:
        """Load a batch of samples.

        Returns:
            Dict with 'data' and possibly 'sizes', 'heights', 'widths'
        """
        ...

    @classmethod
    @abstractmethod
    def build(
        cls,
        field_name: str,
        samples: list[Any],
        output_dir: Path,
        field_type: str,
        **kwargs: Any,
    ) -> FieldStorage:
        """Build storage from samples."""
        ...

    @classmethod
    @abstractmethod
    def load(cls, field_name: str, cache_dir: Path, metadata: dict) -> FieldStorage:
        """Load existing storage."""
        ...


class ImageBytesStorage(FieldStorage):
    """Storage for variable-size image bytes with O(1) access.

    Files:
        <field>.bin  - Contiguous image data
        <field>.meta.npy - Metadata table (ptr, size, height, width)
    """

    def __init__(
        self,
        field_name: str,
        data_mmap: np.ndarray,
        metadata: np.ndarray,
        max_size: int,
    ) -> None:
        self.field_name = field_name
        self._data_mmap = data_mmap
        self._metadata = np.ascontiguousarray(metadata)
        self._data_array = np.asarray(data_mmap)
        self.max_size = max_size
        self.num_samples = len(metadata)

        # Extract dimension arrays for fast access
        self._heights = np.ascontiguousarray(metadata['height'])
        self._widths = np.ascontiguousarray(metadata['width'])

        # Pre-allocated buffers
        self._batch_buffer: np.ndarray | None = None
        self._sizes_buffer: np.ndarray | None = None
        self._current_batch_size = 0

    def _ensure_buffers(self, batch_size: int) -> None:
        """Ensure pre-allocated buffers are large enough."""
        if self._current_batch_size < batch_size:
            self._batch_buffer = np.zeros((batch_size, self.max_size), dtype=np.uint8)
            self._sizes_buffer = np.zeros(batch_size, dtype=np.uint64)
            self._current_batch_size = batch_size

    def load_batch(
        self,
        indices: NDArray[np.int64],
        parallel: bool = True,
    ) -> dict[str, Any]:
        """Load a batch of image bytes."""
        batch_size = len(indices)
        self._ensure_buffers(batch_size)

        dest = self._batch_buffer[:batch_size]
        sizes = self._sizes_buffer[:batch_size]

        if indices.dtype != np.int64:
            indices = indices.astype(np.int64)

        if parallel:
            _load_variable_batch_parallel(
                indices, self._metadata, self._data_array, dest, sizes
            )
        else:
            _load_variable_batch_sequential(
                indices, self._metadata, self._data_array, dest, sizes
            )

        return {
            'data': dest,
            'sizes': sizes,
            'heights': self._heights[indices],
            'widths': self._widths[indices],
        }

    def load_batch_into(
        self,
        indices: NDArray[np.int64],
        dest: np.ndarray,
        sizes: np.ndarray,
        heights: np.ndarray,
        widths: np.ndarray,
        parallel: bool = True,
    ) -> int:
        """Load a batch directly into provided buffers (zero-copy).

        This is the fast path used by SlipstreamLoader to avoid intermediate copies.
        The JIT function writes directly into the destination buffers.

        Args:
            indices: Sample indices to load
            dest: Pre-allocated destination buffer [batch_size, max_size]
            sizes: Pre-allocated sizes buffer [batch_size]
            heights: Pre-allocated heights buffer [batch_size]
            widths: Pre-allocated widths buffer [batch_size]
            parallel: Use parallel loading

        Returns:
            Number of samples loaded (actual batch size)
        """
        batch_size = len(indices)

        if indices.dtype != np.int64:
            indices = indices.astype(np.int64)

        # JIT writes directly into provided buffers - no intermediate copy!
        if parallel:
            _load_variable_batch_parallel(
                indices, self._metadata, self._data_array,
                dest[:batch_size], sizes[:batch_size]
            )
        else:
            _load_variable_batch_sequential(
                indices, self._metadata, self._data_array,
                dest[:batch_size], sizes[:batch_size]
            )

        # Copy pre-stored dimensions (these are small arrays, fast)
        heights[:batch_size] = self._heights[indices]
        widths[:batch_size] = self._widths[indices]

        return batch_size

    @classmethod
    def build(
        cls,
        field_name: str,
        samples: list[bytes],
        output_dir: Path,
        field_type: str,
        sizes: list[int] | None = None,
        heights: list[int] | None = None,
        widths: list[int] | None = None,
    ) -> ImageBytesStorage:
        """Build image storage from samples.

        Args:
            field_name: Name of the field
            samples: List of image bytes
            output_dir: Directory to write storage files
            field_type: Type string (for compatibility)
            sizes: Pre-computed sizes (optional, from fast path)
            heights: Pre-computed heights (optional)
            widths: Pre-computed widths (optional)
        """
        num_samples = len(samples)

        # Use pre-computed metadata if available, otherwise compute
        if sizes is not None and heights is not None and widths is not None:
            sizes_arr = np.array(sizes, dtype=np.uint64)
            heights_arr = np.array(heights, dtype=np.uint32)
            widths_arr = np.array(widths, dtype=np.uint32)
            max_size = int(np.max(sizes_arr) * 1.2)
        else:
            sizes_arr = np.zeros(num_samples, dtype=np.uint64)
            heights_arr = np.zeros(num_samples, dtype=np.uint32)
            widths_arr = np.zeros(num_samples, dtype=np.uint32)
            max_size = 0

            for i, data in enumerate(samples):
                if isinstance(data, np.ndarray):
                    data = bytes(data)
                actual_size = find_jpeg_end(data, len(data))
                sizes_arr[i] = actual_size
                max_size = max(max_size, actual_size)
                w, h = read_jpeg_dimensions(data[:actual_size])
                heights_arr[i] = h
                widths_arr[i] = w

            max_size = int(max_size * 1.2)

        # Build metadata table
        metadata = np.zeros(num_samples, dtype=VARIABLE_METADATA_DTYPE)
        current_ptr = 0
        for i in range(num_samples):
            metadata[i]['data_ptr'] = current_ptr
            metadata[i]['data_size'] = sizes_arr[i]
            metadata[i]['height'] = heights_arr[i]
            metadata[i]['width'] = widths_arr[i]
            current_ptr += sizes_arr[i]

        # Write data file
        data_path = output_dir / f"{field_name}.bin"
        with open(data_path, 'wb') as f:
            for i, data in enumerate(samples):
                if isinstance(data, np.ndarray):
                    data = bytes(data)
                actual_size = int(sizes_arr[i])
                f.write(data[:actual_size])

        # Write metadata
        meta_path = output_dir / f"{field_name}.meta.npy"
        np.save(meta_path, metadata)

        # Load and return
        data_mmap = np.memmap(data_path, dtype=np.uint8, mode='r')
        return cls(field_name, data_mmap, metadata, max_size)

    @classmethod
    def load(cls, field_name: str, cache_dir: Path, metadata: dict) -> ImageBytesStorage:
        """Load existing image storage."""
        data_path = cache_dir / f"{field_name}.bin"
        meta_path = cache_dir / f"{field_name}.meta.npy"

        data_mmap = np.memmap(data_path, dtype=np.uint8, mode='r')
        meta_array = np.load(meta_path, mmap_mode='r')

        return cls(field_name, data_mmap, meta_array, metadata['max_size'])


class NumpyStorage(FieldStorage):
    """Storage for fixed-size numeric arrays using mmap.

    Files:
        <field>.npy - Numpy array (memory-mapped)
    """

    def __init__(self, field_name: str, data: np.ndarray) -> None:
        self.field_name = field_name
        self._data = data
        self.num_samples = len(data)
        self.dtype = data.dtype

    def load_batch(
        self,
        indices: NDArray[np.int64],
        parallel: bool = True,
    ) -> dict[str, Any]:
        """Load a batch of values."""
        return {'data': self._data[indices]}

    @classmethod
    def build(
        cls,
        field_name: str,
        samples: list[Any],
        output_dir: Path,
        field_type: str,
        **kwargs: Any,
    ) -> NumpyStorage:
        """Build numpy storage from samples."""
        arr = np.array(samples)
        path = output_dir / f"{field_name}.npy"
        np.save(path, arr)
        data = np.load(path, mmap_mode='r')
        return cls(field_name, data)

    @classmethod
    def load(cls, field_name: str, cache_dir: Path, metadata: dict) -> NumpyStorage:
        """Load existing numpy storage."""
        path = cache_dir / f"{field_name}.npy"
        data = np.load(path, mmap_mode='r')
        return cls(field_name, data)


class StringStorage(FieldStorage):
    """Storage for variable-length strings using mmap.

    Files:
        <field>.bin - Concatenated UTF-8 bytes
        <field>.offsets.npy - Offset and length for each string
    """

    def __init__(
        self,
        field_name: str,
        data_mmap: np.ndarray,
        offsets: np.ndarray,
    ) -> None:
        self.field_name = field_name
        self._data_mmap = data_mmap
        self._offsets = offsets
        self.num_samples = len(offsets)

    def load_batch(
        self,
        indices: NDArray[np.int64],
        parallel: bool = True,
    ) -> dict[str, Any]:
        """Load a batch of strings."""
        strings = []
        for idx in indices:
            offset, length = self._offsets[idx]
            raw_bytes = bytes(self._data_mmap[offset:offset + length])
            strings.append(raw_bytes.decode('utf-8'))
        return {'data': strings}

    @classmethod
    def build(
        cls,
        field_name: str,
        samples: list[str],
        output_dir: Path,
        field_type: str,
        **kwargs: Any,
    ) -> StringStorage:
        """Build string storage from samples."""
        num_samples = len(samples)
        offsets = np.zeros((num_samples, 2), dtype=np.uint64)
        current_offset = 0

        data_path = output_dir / f"{field_name}.bin"
        with open(data_path, 'wb') as f:
            for i, s in enumerate(samples):
                encoded = s.encode('utf-8')
                offsets[i, 0] = current_offset
                offsets[i, 1] = len(encoded)
                f.write(encoded)
                current_offset += len(encoded)

        offsets_path = output_dir / f"{field_name}.offsets.npy"
        np.save(offsets_path, offsets)

        data_mmap = np.memmap(data_path, dtype=np.uint8, mode='r')
        return cls(field_name, data_mmap, offsets)

    @classmethod
    def load(cls, field_name: str, cache_dir: Path, metadata: dict) -> StringStorage:
        """Load existing string storage."""
        data_path = cache_dir / f"{field_name}.bin"
        offsets_path = cache_dir / f"{field_name}.offsets.npy"

        data_mmap = np.memmap(data_path, dtype=np.uint8, mode='r')
        offsets = np.load(offsets_path, mmap_mode='r')
        return cls(field_name, data_mmap, offsets)


# =============================================================================
# Storage Factory
# =============================================================================

def get_storage_class(field_type: str) -> type[FieldStorage]:
    """Get the appropriate storage class for a field type."""
    if field_type == "ImageBytes":
        return ImageBytesStorage
    elif field_type == "str":
        return StringStorage
    elif field_type in ("bytes",):
        return ImageBytesStorage
    else:
        return NumpyStorage


def _has_litdata_cache(cache_dir: Path) -> bool:
    """Check if directory contains LitData cache structure."""
    index_path = cache_dir / "index.json"
    if not index_path.exists():
        return False
    # Also check for config.data_format which we need for fast path
    with open(index_path) as f:
        index = json.load(f)
    return "config" in index and "data_format" in index.get("config", {})


# =============================================================================
# OptimizedCache
# =============================================================================

class OptimizedCache:
    """Optimized cache for high-performance data loading.

    Converts any SlipstreamDataset into a memory-mapped format with O(1) batch
    access. Different field types use appropriate storage strategies.

    For LitData-backed datasets, building uses a fast path that reads directly
    from chunk files, avoiding slow dataset iteration.

    Attributes:
        cache_dir: Path to the .slipstream cache directory
        fields: Dict mapping field names to FieldStorage instances
        num_samples: Total number of samples
        field_types: Dict mapping field names to type strings
    """

    def __init__(
        self,
        cache_dir: Path,
        fields: dict[str, FieldStorage],
        field_types: dict[str, str],
        num_samples: int,
    ) -> None:
        self.cache_dir = cache_dir
        self.fields = fields
        self.field_types = field_types
        self.num_samples = num_samples

    @classmethod
    def exists(cls, parent_dir: Path) -> bool:
        """Check if an optimized cache exists in the given directory."""
        cache_dir = parent_dir / CACHE_SUBDIR
        manifest_path = cache_dir / MANIFEST_FILE
        return manifest_path.exists()

    @classmethod
    def build(
        cls,
        dataset: SlipstreamDataset,
        output_dir: Path | None = None,
        verbose: bool = True,
    ) -> OptimizedCache:
        """Build optimized cache from a SlipstreamDataset.

        Uses fast path for LitData-backed datasets (reads all fields directly
        from chunks without calling dataset[i]).

        Args:
            dataset: Any SlipstreamDataset (or compatible iterable)
            output_dir: Where to store cache (defaults to dataset.cache_path)
            verbose: Show progress

        Returns:
            Loaded OptimizedCache instance
        """
        if output_dir is None:
            output_dir = dataset.cache_path
            if output_dir is None:
                raise ValueError(
                    "Cannot determine cache directory. "
                    "Specify output_dir explicitly."
                )

        cache_dir = Path(output_dir) / CACHE_SUBDIR
        cache_dir.mkdir(parents=True, exist_ok=True)

        if verbose:
            print(f"Building optimized cache: {cache_dir}")

        # Get field types from dataset
        field_types = {
            k: (v if isinstance(v, str) else v.__name__)
            for k, v in dataset.field_types.items()
        }

        if verbose:
            print(f"Fields: {field_types}")

        # Check for LitData fast path
        litdata_cache_dir = Path(output_dir)
        use_fast_path = _has_litdata_cache(litdata_cache_dir)

        num_samples = len(dataset)

        if use_fast_path:
            if verbose:
                print("Using fast path (reading all fields from LitData chunks)")

            # Read ALL fields directly from chunks (triggers downloads as needed)
            samples_by_field = _read_all_fields_from_litdata_chunks(
                litdata_cache_dir,
                field_types,
                dataset=dataset,
                verbose=verbose,
            )
        else:
            if verbose:
                print("Using iteration path (no LitData chunk format detected)")

            # Fall back to dataset iteration
            samples_by_field = {k: [] for k in field_types}

            if verbose:
                iterator = tqdm(range(num_samples), desc="Reading samples")
            else:
                iterator = range(num_samples)

            for i in iterator:
                sample = dataset[i]
                for field_name in field_types:
                    samples_by_field[field_name].append(sample[field_name])

        # Build storage for each field
        fields: dict[str, FieldStorage] = {}
        field_metadata: dict[str, dict] = {}

        for field_name, field_type in field_types.items():
            if verbose:
                print(f"Building {field_name} ({field_type})...")

            storage_cls = get_storage_class(field_type)

            # Pass pre-computed metadata for image fields if available
            extra_kwargs: dict[str, Any] = {}
            if field_type == "ImageBytes" and use_fast_path:
                extra_kwargs['sizes'] = samples_by_field.get(f"__{field_name}_sizes")
                extra_kwargs['heights'] = samples_by_field.get(f"__{field_name}_heights")
                extra_kwargs['widths'] = samples_by_field.get(f"__{field_name}_widths")

            storage = storage_cls.build(
                field_name,
                samples_by_field[field_name],
                cache_dir,
                field_type,
                **extra_kwargs,
            )
            fields[field_name] = storage

            meta: dict[str, Any] = {
                'type': field_type,
                'num_samples': storage.num_samples,
            }
            if isinstance(storage, ImageBytesStorage):
                meta['max_size'] = storage.max_size
            field_metadata[field_name] = meta

        # Write manifest
        manifest = {
            'version': CACHE_VERSION,
            'num_samples': num_samples,
            'fields': field_metadata,
        }
        manifest_path = cache_dir / MANIFEST_FILE
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

        if verbose:
            print(f"Cache built: {num_samples:,} samples, {len(fields)} fields")

        result = cls(cache_dir, fields, field_types, num_samples)

        # Run sanity check
        if verbose:
            print("Running sanity check...")
            result.verify(dataset, num_checks=10, verbose=verbose)

        return result

    @classmethod
    def load(cls, parent_dir: Path, verbose: bool = True) -> OptimizedCache:
        """Load existing optimized cache."""
        cache_dir = Path(parent_dir) / CACHE_SUBDIR
        manifest_path = cache_dir / MANIFEST_FILE

        if not manifest_path.exists():
            raise FileNotFoundError(
                f"No optimized cache found at {cache_dir}. "
                "Use OptimizedCache.build() to create one."
            )

        with open(manifest_path) as f:
            manifest = json.load(f)

        num_samples = manifest['num_samples']
        field_metadata = manifest['fields']

        fields: dict[str, FieldStorage] = {}
        field_types: dict[str, str] = {}

        for field_name, meta in field_metadata.items():
            field_type = meta['type']
            field_types[field_name] = field_type

            storage_cls = get_storage_class(field_type)
            storage = storage_cls.load(field_name, cache_dir, meta)
            fields[field_name] = storage

        if verbose:
            print(f"Loaded cache: {num_samples:,} samples, {len(fields)} fields")

        return cls(cache_dir, fields, field_types, num_samples)

    def verify(
        self,
        dataset: SlipstreamDataset,
        num_checks: int = 100,
        verbose: bool = True,
    ) -> bool:
        """Verify cache matches source dataset.

        Checks first, last, and random samples to ensure data integrity.

        Args:
            dataset: Source dataset to verify against
            num_checks: Number of random samples to check
            verbose: Print results

        Returns:
            True if all checks pass

        Raises:
            ValueError: If any check fails
        """
        n = len(dataset)
        if n != self.num_samples:
            raise ValueError(
                f"Sample count mismatch: dataset has {n}, cache has {self.num_samples}"
            )

        # Select indices to check: first, last, and random
        check_indices = [0, n - 1]
        if num_checks > 2:
            random.seed(42)  # Reproducible
            random_indices = random.sample(range(1, n - 1), min(num_checks - 2, n - 2))
            check_indices.extend(random_indices)

        errors = []

        for idx in check_indices:
            # Get from dataset
            sample = dataset[idx]

            # Get from cache
            cache_batch = self.load_batch(np.array([idx], dtype=np.int64))

            # Compare each field
            for field_name, field_type in self.field_types.items():
                dataset_value = sample[field_name]
                cache_value = cache_batch[field_name]['data']

                if field_type == "ImageBytes":
                    # Compare image bytes (hash for efficiency)
                    if isinstance(dataset_value, np.ndarray):
                        dataset_bytes = bytes(dataset_value)
                    else:
                        dataset_bytes = dataset_value

                    cache_size = int(cache_batch[field_name]['sizes'][0])
                    cache_bytes = bytes(cache_value[0, :cache_size])

                    # Find actual JPEG end in dataset bytes
                    dataset_size = find_jpeg_end(dataset_bytes, len(dataset_bytes))
                    dataset_bytes = dataset_bytes[:dataset_size]

                    if hashlib.md5(dataset_bytes).hexdigest() != \
                       hashlib.md5(cache_bytes).hexdigest():
                        errors.append(
                            f"Image mismatch at index {idx}, field '{field_name}'"
                        )
                elif field_type == "str":
                    if dataset_value != cache_value[0]:
                        errors.append(
                            f"String mismatch at index {idx}, field '{field_name}': "
                            f"'{dataset_value}' vs '{cache_value[0]}'"
                        )
                else:
                    # Numeric comparison
                    if isinstance(cache_value, np.ndarray):
                        cache_scalar = cache_value[0]
                    else:
                        cache_scalar = cache_value

                    if dataset_value != cache_scalar:
                        errors.append(
                            f"Value mismatch at index {idx}, field '{field_name}': "
                            f"{dataset_value} vs {cache_scalar}"
                        )

        if errors:
            error_msg = "\n".join(errors[:10])  # Show first 10 errors
            if len(errors) > 10:
                error_msg += f"\n... and {len(errors) - 10} more errors"
            raise ValueError(f"Cache verification failed:\n{error_msg}")

        if verbose:
            print(f"  Verified {len(check_indices)} samples: all match")

        return True

    def load_batch(
        self,
        indices: NDArray[np.int64],
        fields: list[str] | None = None,
        parallel: bool = True,
    ) -> dict[str, dict[str, Any]]:
        """Load a batch of samples."""
        if fields is None:
            fields = list(self.fields.keys())

        result = {}
        for field_name in fields:
            if field_name not in self.fields:
                raise KeyError(f"Unknown field: {field_name}")
            result[field_name] = self.fields[field_name].load_batch(indices, parallel)

        return result

    def get_field_type(self, field_name: str) -> str:
        """Get the type of a field."""
        return self.field_types[field_name]

    def get_image_dims(self, field_name: str, idx: int) -> tuple[int, int]:
        """Get pre-stored dimensions for an image field."""
        storage = self.fields[field_name]
        if not isinstance(storage, ImageBytesStorage):
            raise TypeError(f"Field {field_name} is not an image field")
        return int(storage._heights[idx]), int(storage._widths[idx])

    def __len__(self) -> int:
        return self.num_samples

    def __repr__(self) -> str:
        fields_str = ", ".join(f"{k}: {v}" for k, v in self.field_types.items())
        return (
            f"OptimizedCache(\n"
            f"    cache_dir='{self.cache_dir}',\n"
            f"    num_samples={self.num_samples:,},\n"
            f"    fields={{{fields_str}}},\n"
            f")"
        )


__all__ = [
    "OptimizedCache",
    "FieldStorage",
    "ImageBytesStorage",
    "NumpyStorage",
    "StringStorage",
    "CACHE_SUBDIR",
]
