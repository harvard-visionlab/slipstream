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

# Protocol for any dataset/reader that can be used with OptimizedCache.build()
# Must have: cache_path, field_types, __len__, __getitem__
# Optional: read_all_fields() for bulk fast path
DatasetLike = Any  # Using Any since Protocol requires runtime_checkable for isinstance


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
# Image Format Detection
# =============================================================================

def detect_image_format(data: bytes | np.ndarray) -> str:
    """Detect image format from header bytes.

    Args:
        data: Image bytes (at least first 8 bytes needed for reliable detection)

    Returns:
        Format string: "jpeg", "png", or "other"
    """
    if isinstance(data, np.ndarray):
        data = bytes(data[:8])
    elif len(data) > 8:
        data = data[:8]

    if len(data) < 2:
        return "other"

    # JPEG: starts with FF D8
    if data[0] == 0xFF and data[1] == 0xD8:
        return "jpeg"

    # PNG: starts with 89 50 4E 47 0D 0A 1A 0A
    if len(data) >= 8 and data[:8] == b'\x89PNG\r\n\x1a\n':
        return "png"

    return "other"


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


def find_image_end(data: bytes | np.ndarray, max_len: int) -> int:
    """Find actual image size, handling JPEG trailing garbage.

    JPEG files may have trailing garbage bytes after the FFD9 end marker.
    PNG and other formats use the full provided length.

    Args:
        data: Image bytes
        max_len: Maximum length to consider

    Returns:
        Actual image size in bytes
    """
    if isinstance(data, np.ndarray):
        data_bytes = bytes(data[:max_len])
    else:
        data_bytes = data[:max_len]

    # JPEG: find FFD9 end marker
    if len(data_bytes) >= 2 and data_bytes[0] == 0xFF and data_bytes[1] == 0xD8:
        return find_jpeg_end(data_bytes, max_len)

    # PNG, GIF, BMP, WebP, etc.: use full length
    return len(data_bytes)


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


def read_png_dimensions(png_data: bytes | np.ndarray) -> tuple[int, int]:
    """Read PNG dimensions from IHDR chunk without full decode.

    Returns:
        (width, height) tuple, or (0, 0) if parsing fails
    """
    if isinstance(png_data, np.ndarray):
        png_data = bytes(png_data)

    # PNG signature: 89 50 4E 47 0D 0A 1A 0A
    if len(png_data) < 24:
        return (0, 0)

    if png_data[:8] != b'\x89PNG\r\n\x1a\n':
        return (0, 0)

    # IHDR chunk starts at byte 8
    # Format: length (4) + type (4) + width (4) + height (4) + ...
    # Width is at bytes 16-19, height at 20-23
    width = int.from_bytes(png_data[16:20], 'big')
    height = int.from_bytes(png_data[20:24], 'big')

    return (width, height)


def read_image_dimensions(image_data: bytes | np.ndarray) -> tuple[int, int]:
    """Read image dimensions from header without full decode.

    Supports JPEG, PNG. Falls back to PIL for other formats.

    Returns:
        (width, height) tuple, or (0, 0) if parsing fails
    """
    import io
    from PIL import Image

    if isinstance(image_data, np.ndarray):
        image_data = bytes(image_data)

    if len(image_data) < 8:
        return (0, 0)

    # Try JPEG (starts with FF D8)
    if image_data[0] == 0xFF and image_data[1] == 0xD8:
        dims = read_jpeg_dimensions(image_data)
        if dims != (0, 0):
            return dims

    # Try PNG (starts with 89 50 4E 47)
    if image_data[:4] == b'\x89PNG':
        dims = read_png_dimensions(image_data)
        if dims != (0, 0):
            return dims

    # Fall back to PIL for other formats (GIF, BMP, WebP, etc.)
    try:
        with Image.open(io.BytesIO(image_data)) as img:
            return img.size  # (width, height)
    except Exception:
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


def _extract_image_bytes(data: bytes | dict | np.ndarray) -> bytes:
    """Extract raw image bytes from various formats.

    Handles:
    - Raw bytes: returned as-is
    - numpy array: converted to bytes
    - HuggingFace image dict: {'bytes': ..., 'path': ...}
    """
    if isinstance(data, dict):
        # HuggingFace image dict format
        if 'bytes' in data and data['bytes']:
            return data['bytes'] if isinstance(data['bytes'], bytes) else bytes(data['bytes'])
        elif 'path' in data and data['path']:
            with open(data['path'], 'rb') as f:
                return f.read()
        else:
            raise ValueError(f"Invalid HuggingFace image dict: {data}")
    elif isinstance(data, np.ndarray):
        return bytes(data)
    elif isinstance(data, bytes):
        return data
    else:
        raise TypeError(f"Unsupported image data type: {type(data)}")


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
                                actual_size = find_image_end(field_bytes, field_size)
                                w, h = read_image_dimensions(field_bytes[:actual_size])
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
        samples: list[bytes | dict],
        output_dir: Path,
        field_type: str,
        sizes: list[int] | None = None,
        heights: list[int] | None = None,
        widths: list[int] | None = None,
    ) -> tuple[ImageBytesStorage, str]:
        """Build image storage from samples.

        Non-JPEG images (PNG, BMP, etc.) are automatically converted to YUV420
        format during cache building, providing ~2x faster decode than JPEG.

        Args:
            field_name: Name of the field
            samples: List of image bytes or HuggingFace image dicts
            output_dir: Directory to write storage files
            field_type: Type string (for compatibility)
            sizes: Pre-computed sizes (optional, from fast path)
            heights: Pre-computed heights (optional)
            widths: Pre-computed widths (optional)

        Returns:
            Tuple of (ImageBytesStorage, image_format) where image_format is
            "jpeg" or "yuv420".
        """
        num_samples = len(samples)

        # Extract bytes from various formats (raw bytes, numpy, HF dicts)
        # We do this once upfront to avoid repeated extraction
        extracted_samples: list[bytes] = []
        for data in samples:
            extracted_samples.append(_extract_image_bytes(data))

        # Detect image format from first sample
        detected_format = detect_image_format(extracted_samples[0]) if extracted_samples else "jpeg"
        use_yuv420_conversion = detected_format != "jpeg"

        if use_yuv420_conversion:
            # Non-JPEG path: decode → YUV420 → store
            return cls._build_yuv420_from_samples(
                field_name, extracted_samples, output_dir
            )

        # JPEG path: store raw bytes
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

            for i, data in enumerate(extracted_samples):
                actual_size = find_image_end(data, len(data))
                sizes_arr[i] = actual_size
                max_size = max(max_size, actual_size)
                w, h = read_image_dimensions(data[:actual_size])
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
            for i, data in enumerate(extracted_samples):
                actual_size = int(sizes_arr[i])
                f.write(data[:actual_size])

        # Write metadata
        meta_path = output_dir / f"{field_name}.meta.npy"
        np.save(meta_path, metadata)

        # Load and return
        data_mmap = np.memmap(data_path, dtype=np.uint8, mode='r')
        return cls(field_name, data_mmap, metadata, max_size), "jpeg"

    @classmethod
    def _build_yuv420_from_samples(
        cls,
        field_name: str,
        extracted_samples: list[bytes],
        output_dir: Path,
    ) -> tuple[ImageBytesStorage, str]:
        """Build YUV420 storage from non-JPEG image samples.

        Decodes each image with PIL, converts to YUV420P format, and stores.
        This is used for PNG, BMP, GIF, WebP, and other non-JPEG formats.

        Args:
            field_name: Name of the field
            extracted_samples: List of raw image bytes
            output_dir: Directory to write storage files

        Returns:
            Tuple of (ImageBytesStorage, "yuv420")
        """
        num_samples = len(extracted_samples)
        metadata = np.zeros(num_samples, dtype=VARIABLE_METADATA_DTYPE)
        current_ptr = 0
        max_size = 0

        data_path = output_dir / f"{field_name}.bin"

        with open(data_path, 'wb') as f:
            for i, img_bytes in enumerate(extracted_samples):
                # Decode image to RGB using PIL
                rgb = decode_image_to_rgb(img_bytes)

                # Convert RGB → YUV420P
                yuv_bytes, pad_h, pad_w = rgb_to_yuv420(rgb)
                enc_size = len(yuv_bytes)

                f.write(yuv_bytes)

                metadata[i]['data_ptr'] = current_ptr
                metadata[i]['data_size'] = enc_size
                metadata[i]['height'] = pad_h
                metadata[i]['width'] = pad_w

                current_ptr += enc_size
                max_size = max(max_size, enc_size)

        max_size = int(max_size * 1.2)

        # Write metadata
        meta_path = output_dir / f"{field_name}.meta.npy"
        np.save(meta_path, metadata)

        # Load and return
        data_mmap = np.memmap(data_path, dtype=np.uint8, mode='r')
        return cls(field_name, data_mmap, metadata, max_size), "yuv420"

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
    if field_type in ("ImageBytes", "HFImageDict"):
        return ImageBytesStorage
    elif field_type == "str":
        return StringStorage
    elif field_type in ("bytes",):
        return ImageBytesStorage
    else:
        return NumpyStorage


def _has_litdata_binary_cache(cache_dir: Path) -> bool:
    """Check if directory contains LitData binary chunk cache structure.

    Returns True only for native LitData binary chunks, not Parquet-based
    datasets (e.g., HuggingFace datasets use ParquetLoader).
    """
    index_path = cache_dir / "index.json"
    if not index_path.exists():
        return False

    with open(index_path) as f:
        index = json.load(f)

    config = index.get("config", {})
    if "data_format" not in config:
        return False

    # Parquet-based datasets (HuggingFace, etc.) use ParquetLoader
    # These have a different chunk format and need iteration path
    item_loader = config.get("item_loader", "")
    if item_loader == "ParquetLoader":
        return False

    # Check that chunk files are .bin files (native LitData format)
    chunks = index.get("chunks", [])
    if chunks:
        first_chunk = chunks[0].get("filename", "")
        if first_chunk.endswith(".parquet"):
            return False

    return True


def _is_parquet_dataset(cache_dir: Path) -> bool:
    """Check if dataset uses Parquet-based storage (e.g., HuggingFace)."""
    index_path = cache_dir / "index.json"
    if not index_path.exists():
        return False

    with open(index_path) as f:
        index = json.load(f)

    config = index.get("config", {})
    item_loader = config.get("item_loader", "")

    return item_loader == "ParquetLoader"


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
        field_metadata: dict[str, dict] | None = None,
    ) -> None:
        self.cache_dir = cache_dir
        self.fields = fields
        self.field_types = field_types
        self.num_samples = num_samples
        self._field_metadata = field_metadata or {}
        self._indexes: dict[str, dict] = {}

    @classmethod
    def exists(cls, parent_dir: Path) -> bool:
        """Check if an optimized cache exists in the given directory."""
        cache_dir = parent_dir / CACHE_SUBDIR
        manifest_path = cache_dir / MANIFEST_FILE
        return manifest_path.exists()

    @classmethod
    def build(
        cls,
        dataset: Any,
        output_dir: Path | None = None,
        verbose: bool = True,
    ) -> OptimizedCache:
        """Build optimized cache from a dataset or reader.

        Uses fast path when available: reader.read_all_fields() for readers,
        or direct LitData chunk reading for LitData-backed datasets.

        Args:
            dataset: Any object with cache_path, field_types, __len__, __getitem__
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

        # Check for reader fast path (read_all_fields protocol)
        reader_fast_path = hasattr(dataset, 'read_all_fields') and callable(dataset.read_all_fields)

        # Check for LitData fast path (binary chunks only, not Parquet)
        litdata_cache_dir = Path(output_dir)
        use_litdata_fast_path = _has_litdata_binary_cache(litdata_cache_dir)

        num_samples = len(dataset)

        use_fast_path = False

        if reader_fast_path:
            if verbose:
                print("Using reader fast path (read_all_fields)")

            samples_by_field = dataset.read_all_fields()
            if samples_by_field is not None:
                use_fast_path = True
            else:
                reader_fast_path = False

        if not use_fast_path and use_litdata_fast_path:
            use_fast_path = True
            if verbose:
                print("Using fast path (reading all fields from LitData chunks)")

            # Read ALL fields directly from chunks (triggers downloads as needed)
            samples_by_field = _read_all_fields_from_litdata_chunks(
                litdata_cache_dir,
                field_types,
                dataset=dataset,
                verbose=verbose,
            )

        if not use_fast_path:
            if verbose:
                if _is_parquet_dataset(litdata_cache_dir):
                    print("Using iteration path (Parquet/HuggingFace dataset)")
                else:
                    print("Using iteration path (no fast path available)")

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
            if field_type in ("ImageBytes", "HFImageDict") and use_fast_path:
                extra_kwargs['sizes'] = samples_by_field.get(f"__{field_name}_sizes")
                extra_kwargs['heights'] = samples_by_field.get(f"__{field_name}_heights")
                extra_kwargs['widths'] = samples_by_field.get(f"__{field_name}_widths")

            build_result = storage_cls.build(
                field_name,
                samples_by_field[field_name],
                cache_dir,
                field_type,
                **extra_kwargs,
            )

            # ImageBytesStorage.build() returns (storage, image_format) tuple
            if isinstance(build_result, tuple):
                storage, image_format = build_result
            else:
                storage = build_result
                image_format = None

            fields[field_name] = storage

            meta: dict[str, Any] = {
                'type': field_type,
                'num_samples': storage.num_samples,
            }
            if isinstance(storage, ImageBytesStorage):
                meta['max_size'] = storage.max_size
                if image_format is not None:
                    meta['image_format'] = image_format
                    if verbose and image_format == "yuv420":
                        print(f"  Non-JPEG images detected → converted to YUV420 format")
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

        result = cls(cache_dir, fields, field_types, num_samples, field_metadata)

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

        instance = cls(cache_dir, fields, field_types, num_samples, field_metadata)
        instance._discover_indexes()
        if verbose and instance._indexes:
            print(f"  Loaded indexes: {list(instance._indexes.keys())}")
        return instance

    def verify(
        self,
        dataset: Any,
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

                if field_type in ("ImageBytes", "HFImageDict"):
                    # Check if this field was converted to YUV420 (non-JPEG source)
                    stored_format = self.get_image_format(field_name)

                    if stored_format == "yuv420":
                        # YUV420-converted images: verify dimensions match
                        # We can't compare bytes directly since format changed
                        dataset_bytes = _extract_image_bytes(dataset_value)
                        dataset_w, dataset_h = read_image_dimensions(dataset_bytes)

                        cache_h = int(cache_batch[field_name]['heights'][0])
                        cache_w = int(cache_batch[field_name]['widths'][0])

                        # Cache dimensions may be padded to even values
                        padded_h = dataset_h + (dataset_h % 2)
                        padded_w = dataset_w + (dataset_w % 2)

                        if cache_h != padded_h or cache_w != padded_w:
                            errors.append(
                                f"Image dimension mismatch at index {idx}, "
                                f"field '{field_name}': expected {padded_h}x{padded_w}, "
                                f"got {cache_h}x{cache_w}"
                            )
                    else:
                        # JPEG path: compare image bytes (hash for efficiency)
                        # Extract bytes from various formats (raw bytes, numpy, HF dicts)
                        dataset_bytes = _extract_image_bytes(dataset_value)

                        cache_size = int(cache_batch[field_name]['sizes'][0])
                        cache_bytes = bytes(cache_value[0, :cache_size])

                        # Find actual JPEG end in dataset bytes
                        dataset_size = find_image_end(dataset_bytes, len(dataset_bytes))
                        dataset_bytes = dataset_bytes[:dataset_size]

                        if hashlib.md5(dataset_bytes).hexdigest() != \
                           hashlib.md5(cache_bytes).hexdigest():
                            errors.append(
                                f"Image mismatch at index {idx}, field '{field_name}'"
                            )
                elif field_type == "bytes":
                    # Variable-length bytes field (same storage as images)
                    if isinstance(dataset_value, np.ndarray):
                        dataset_bytes = bytes(dataset_value)
                    else:
                        dataset_bytes = dataset_value

                    cache_size = int(cache_batch[field_name]['sizes'][0])
                    cache_bytes = bytes(cache_value[0, :cache_size])

                    if dataset_bytes != cache_bytes:
                        errors.append(
                            f"Bytes mismatch at index {idx}, field '{field_name}'"
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

    def get_image_format(self, field_name: str) -> str:
        """Get the stored image format for a field.

        Args:
            field_name: Name of the image field

        Returns:
            Format string: "jpeg" or "yuv420"
        """
        if field_name not in self._field_metadata:
            return "jpeg"  # Default for backwards compatibility
        return self._field_metadata[field_name].get('image_format', 'jpeg')

    def get_image_dims(self, field_name: str, idx: int) -> tuple[int, int]:
        """Get pre-stored dimensions for an image field."""
        storage = self.fields[field_name]
        if not isinstance(storage, ImageBytesStorage):
            raise TypeError(f"Field {field_name} is not an image field")
        return int(storage._heights[idx]), int(storage._widths[idx])

    def _discover_indexes(self) -> None:
        """Scan cache directory for index files and load them."""
        for path in sorted(self.cache_dir.glob("*_index.npy")):
            field_name = path.stem.removesuffix("_index")
            self._indexes[field_name] = np.load(path, allow_pickle=True).item()

    def get_index(self, field_name: str) -> dict:
        """Get the field index mapping unique values to sample indices.

        Args:
            field_name: Name of the indexed field (e.g. 'label')

        Returns:
            Dict mapping field values to numpy arrays of sample indices.

        Raises:
            KeyError: If no index exists for this field.
        """
        if field_name not in self._indexes:
            available = list(self._indexes.keys())
            raise KeyError(
                f"No index found for field '{field_name}'. "
                f"Available indexes: {available}. "
                f"Use write_index(cache, fields=['{field_name}']) to build one."
            )
        return self._indexes[field_name]

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


# =============================================================================
# Index Utilities
# =============================================================================

def _resolve_cache_dir(source: Any) -> Path:
    """Resolve the .slipstream cache directory from a source object."""
    if isinstance(source, OptimizedCache):
        return source.cache_dir
    # Dataset or reader with cache_path attribute
    if hasattr(source, 'cache_path'):
        cache_dir = Path(source.cache_path) / CACHE_SUBDIR
        if cache_dir.exists():
            return cache_dir
    raise ValueError(
        "Cannot resolve cache directory from source. "
        "Pass an OptimizedCache instance or an object with cache_path."
    )


def write_index(
    source: Any,
    fields: list[str],
    verbose: bool = True,
) -> None:
    """Build field indexes for an optimized cache.

    An index maps each unique field value to the sample indices that have
    that value. Indexes are saved as ``{field}_index.npy`` inside the cache
    directory and are auto-discovered on ``OptimizedCache.load()``.

    Args:
        source: An ``OptimizedCache`` instance, or any object with a
            ``cache_path`` attribute (e.g. ``SlipstreamDataset``, reader).
        fields: List of field names to index (e.g. ``['label']``).
        verbose: Print progress information.

    Example::

        from slipstream.cache import OptimizedCache, write_index

        cache = OptimizedCache.load(cache_dir)
        write_index(cache, fields=['label'])

        # Reload to pick up new indexes
        cache = OptimizedCache.load(cache_dir)
        label_idx = cache.get_index('label')
        print(f"Samples with label 0: {len(label_idx[0])}")
    """
    cache_dir = _resolve_cache_dir(source)

    # Load manifest to validate fields
    manifest_path = cache_dir / MANIFEST_FILE
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No manifest found at {cache_dir}. Build the cache first."
        )
    with open(manifest_path) as f:
        manifest = json.load(f)

    field_metadata = manifest['fields']

    for field_name in fields:
        if field_name not in field_metadata:
            available = list(field_metadata.keys())
            raise KeyError(
                f"Field '{field_name}' not found in cache. "
                f"Available fields: {available}"
            )

        field_type = field_metadata[field_name]['type']

        if field_type == 'ImageBytes':
            raise ValueError(
                f"Cannot index image field '{field_name}'. "
                "Indexing is only supported for numeric and string fields."
            )

        if verbose:
            print(f"Building index for '{field_name}' ({field_type})...")

        if field_type == 'str':
            # String fields: read via StringStorage
            storage = StringStorage.load(field_name, cache_dir, field_metadata[field_name])
            all_indices = np.arange(storage.num_samples, dtype=np.int64)
            # Read all strings
            values = []
            for idx in range(storage.num_samples):
                offset, length = storage._offsets[idx]
                raw = bytes(storage._data_mmap[offset:offset + length])
                values.append(raw.decode('utf-8'))

            # Build index
            index: dict[Any, np.ndarray] = {}
            for i, val in enumerate(values):
                if val not in index:
                    index[val] = []
                index[val].append(i)
            index = {k: np.array(v, dtype=np.int64) for k, v in index.items()}
        else:
            # Numeric fields: fast numpy path
            data = np.load(cache_dir / f"{field_name}.npy", mmap_mode='r')
            unique_vals = np.unique(data)
            index = {}
            for val in unique_vals:
                index[val.item()] = np.where(data == val)[0].astype(np.int64)

        # Save index
        index_path = cache_dir / f"{field_name}_index.npy"
        np.save(index_path, index, allow_pickle=True)

        if verbose:
            print(f"  {len(index)} unique values → {index_path.name}")

    # If source is an OptimizedCache, update its in-memory indexes
    if isinstance(source, OptimizedCache):
        source._discover_indexes()


# =============================================================================
# YUV420 Cache Utilities
# =============================================================================

def rgb_to_yuv420(rgb: np.ndarray) -> tuple[bytes, int, int]:
    """Convert RGB array to YUV420P bytes.

    Args:
        rgb: RGB image array, shape (H, W, 3), dtype uint8

    Returns:
        Tuple of (yuv_bytes, padded_height, padded_width).
        Dimensions are padded to even values for YUV420 subsampling.
    """
    h, w = rgb.shape[:2]
    pad_h = h + (h % 2)
    pad_w = w + (w % 2)

    # Pad to even dimensions if necessary
    if pad_h != h or pad_w != w:
        padded = np.zeros((pad_h, pad_w, 3), dtype=np.uint8)
        padded[:h, :w, :] = rgb
        if pad_h > h:
            padded[h, :w, :] = rgb[h - 1, :, :]
        if pad_w > w:
            padded[:h, w, :] = rgb[:, w - 1, :]
        if pad_h > h and pad_w > w:
            padded[h, w, :] = rgb[h - 1, w - 1, :]
        rgb = padded

    r = rgb[:, :, 0].astype(np.float32)
    g = rgb[:, :, 1].astype(np.float32)
    b = rgb[:, :, 2].astype(np.float32)

    # BT.601 conversion
    y = np.clip(0.299 * r + 0.587 * g + 0.114 * b, 0, 255).astype(np.uint8)
    u = np.clip(-0.168736 * r - 0.331264 * g + 0.5 * b + 128.0, 0, 255).astype(np.uint8)
    v = np.clip(0.5 * r - 0.418688 * g - 0.081312 * b + 128.0, 0, 255).astype(np.uint8)

    # Subsample U/V (4:2:0)
    u_sub = u.reshape(pad_h // 2, 2, pad_w // 2, 2).mean(axis=(1, 3))
    v_sub = v.reshape(pad_h // 2, 2, pad_w // 2, 2).mean(axis=(1, 3))
    u_sub = np.clip(u_sub, 0, 255).astype(np.uint8)
    v_sub = np.clip(v_sub, 0, 255).astype(np.uint8)

    yuv_bytes = y.tobytes() + u_sub.tobytes() + v_sub.tobytes()
    return yuv_bytes, pad_h, pad_w


def decode_image_to_rgb(image_bytes: bytes) -> np.ndarray:
    """Decode image bytes to RGB array using PIL.

    Handles any format PIL supports: PNG, BMP, GIF, WebP, etc.

    Args:
        image_bytes: Raw image file bytes

    Returns:
        RGB array, shape (H, W, 3), dtype uint8
    """
    import io
    from PIL import Image

    img = Image.open(io.BytesIO(image_bytes))
    # Convert to RGB (handles grayscale, RGBA, palette modes, etc.)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return np.array(img, dtype=np.uint8)


def load_yuv420_cache(
    cache_dir: Path,
    image_field: str = "image",
) -> ImageBytesStorage | None:
    """Load existing YUV420 image storage if it exists.

    Args:
        cache_dir: The .slipstream cache directory.
        image_field: Name of the image field.

    Returns:
        ImageBytesStorage for YUV420 data, or None if not built yet.
    """
    yuv_field = f"{image_field}_yuv420"
    data_path = cache_dir / f"{yuv_field}.bin"
    meta_path = cache_dir / f"{yuv_field}.meta.npy"

    if not data_path.exists() or not meta_path.exists():
        return None

    data_mmap = np.memmap(data_path, dtype=np.uint8, mode='r')
    meta_array = np.load(meta_path, mmap_mode='r')
    max_size = int(np.max(meta_array['data_size']) * 1.2)

    return ImageBytesStorage(yuv_field, data_mmap, meta_array, max_size)


def build_yuv420_cache(
    cache_dir: Path,
    image_field: str = "image",
    batch_size: int = 256,
    verbose: bool = True,
) -> ImageBytesStorage:
    """Build YUV420 cache from existing JPEG cache.

    Reads JPEG images from the slip cache, decodes them, converts RGB to
    YUV420P, and writes the result as a sibling storage.

    Args:
        cache_dir: The .slipstream cache directory.
        image_field: Name of the image field.
        batch_size: Processing batch size (for progress reporting).
        verbose: Print progress.

    Returns:
        ImageBytesStorage for the YUV420 data.
    """
    from turbojpeg import TurboJPEG

    data_path = cache_dir / f"{image_field}.bin"
    meta_path = cache_dir / f"{image_field}.meta.npy"

    if not data_path.exists():
        raise FileNotFoundError(f"No image data at {data_path}")

    data_mmap = np.memmap(data_path, dtype=np.uint8, mode='r')
    metadata = np.load(meta_path, mmap_mode='r')
    num_samples = len(metadata)

    yuv_field = f"{image_field}_yuv420"
    out_data_path = cache_dir / f"{yuv_field}.bin"
    out_meta_path = cache_dir / f"{yuv_field}.meta.npy"

    # Check if already exists with correct sample count
    if out_data_path.exists() and out_meta_path.exists():
        existing_meta = np.load(out_meta_path, mmap_mode='r')
        if len(existing_meta) == num_samples:
            if verbose:
                print(f"YUV420 cache already exists ({num_samples:,} samples)")
            return load_yuv420_cache(cache_dir, image_field)  # type: ignore[return-value]

    turbo = TurboJPEG()

    if verbose:
        print(f"Building YUV420 cache: {num_samples:,} samples")

    out_metadata = np.zeros(num_samples, dtype=VARIABLE_METADATA_DTYPE)
    current_ptr = 0

    iterator = range(num_samples)
    if verbose:
        iterator = tqdm(iterator, desc="  JPEG → YUV420")

    with open(out_data_path, 'wb') as f:
        for i in iterator:
            ptr = int(metadata[i]['data_ptr'])
            size = int(metadata[i]['data_size'])
            jpeg_bytes = bytes(data_mmap[ptr:ptr + size])

            # Decode JPEG → RGB
            rgb = turbo.decode(jpeg_bytes, pixel_format=0)  # TJPF_RGB

            # Convert RGB → YUV420P
            h, w = rgb.shape[:2]
            pad_h = h + (h % 2)
            pad_w = w + (w % 2)
            if pad_h != h or pad_w != w:
                padded = np.zeros((pad_h, pad_w, 3), dtype=np.uint8)
                padded[:h, :w, :] = rgb
                if pad_h > h:
                    padded[h, :w, :] = rgb[h - 1, :, :]
                if pad_w > w:
                    padded[:h, w, :] = rgb[:, w - 1, :]
                if pad_h > h and pad_w > w:
                    padded[h, w, :] = rgb[h - 1, w - 1, :]
                rgb = padded

            r = rgb[:, :, 0].astype(np.float32)
            g = rgb[:, :, 1].astype(np.float32)
            b = rgb[:, :, 2].astype(np.float32)

            y = np.clip(0.299 * r + 0.587 * g + 0.114 * b, 0, 255).astype(np.uint8)
            u = np.clip(-0.168736 * r - 0.331264 * g + 0.5 * b + 128.0, 0, 255).astype(np.uint8)
            v = np.clip(0.5 * r - 0.418688 * g - 0.081312 * b + 128.0, 0, 255).astype(np.uint8)

            u_sub = u.reshape(pad_h // 2, 2, pad_w // 2, 2).mean(axis=(1, 3))
            v_sub = v.reshape(pad_h // 2, 2, pad_w // 2, 2).mean(axis=(1, 3))
            u_sub = np.clip(u_sub, 0, 255).astype(np.uint8)
            v_sub = np.clip(v_sub, 0, 255).astype(np.uint8)

            yuv_bytes = y.tobytes() + u_sub.tobytes() + v_sub.tobytes()
            enc_size = len(yuv_bytes)

            f.write(yuv_bytes)

            out_metadata[i]['data_ptr'] = current_ptr
            out_metadata[i]['data_size'] = enc_size
            out_metadata[i]['height'] = pad_h
            out_metadata[i]['width'] = pad_w

            current_ptr += enc_size

    np.save(out_meta_path, out_metadata)

    if verbose:
        total_bytes = current_ptr
        jpeg_total = sum(int(metadata[i]['data_size']) for i in range(num_samples))
        ratio = total_bytes / jpeg_total if jpeg_total > 0 else 0
        print(f"  Size: {total_bytes / 1e9:.2f} GB ({ratio:.2f}x JPEG)")

    return load_yuv420_cache(cache_dir, image_field)  # type: ignore[return-value]


__all__ = [
    "OptimizedCache",
    "FieldStorage",
    "ImageBytesStorage",
    "NumpyStorage",
    "StringStorage",
    "CACHE_SUBDIR",
    "write_index",
    "build_yuv420_cache",
    "load_yuv420_cache",
    "detect_image_format",
    "rgb_to_yuv420",
    "decode_image_to_rgb",
]
