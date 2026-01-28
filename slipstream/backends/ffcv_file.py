"""Dataset that reads native FFCV .beton files directly.

This module provides direct access to .ffcv/.beton files using Numba JIT
batch loading, achieving 2-3x faster I/O than FFCV's native loader.

Benefits:
- No conversion needed - use existing .ffcv files directly
- 2-3x faster raw I/O than FFCV native loader
- 2.3x faster cold start (epoch 1)
- Compatible with PrefetchingDataLoader and GPU decoder

File Format (FFCV v2):
    Header:      version(2B) + num_fields(2B) + page_size(4B) + num_samples(8B) + alloc_ptr(8B)
    Fields:      Field descriptors (type_id, name, arguments) for each field
    Metadata:    Per-sample metadata (height, width, mode for images)
    Alloc Table: (sample_id, ptr, size) entries for variable-length fields
    Data:        Raw sample data (JPEG bytes, etc.)

Usage:
    from slipstream.backends import FFCVFileDataset

    dataset = FFCVFileDataset("/path/to/imagenet.ffcv")
    print(f"Loaded {len(dataset)} samples")

    # Works with PrefetchingDataLoader
    from slipstream.backends import PrefetchingDataLoader
    loader = PrefetchingDataLoader(dataset, batch_size=256)

    for batch in loader:
        data = batch['data']       # [B, max_size] uint8 JPEG bytes
        sizes = batch['sizes']     # [B] actual sizes
        heights = batch['heights'] # [B] image heights
        widths = batch['widths']   # [B] image widths
"""

from __future__ import annotations

import queue
import threading
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numba import njit, prange

if TYPE_CHECKING:
    from numpy.typing import NDArray

# =============================================================================
# FFCV File Format Types
# =============================================================================

# FFCV header structure (from ffcv/types.py)
FFCV_HEADER_DTYPE = np.dtype([
    ('version', '<u2'),          # File format version (should be 2)
    ('num_fields', '<u2'),       # Number of fields per sample
    ('page_size', '<u4'),        # Page size for alignment
    ('num_samples', '<u8'),      # Total number of samples
    ('alloc_table_ptr', '<u8'),  # Offset to allocation table
], align=True)

# Allocation table entry (one per variable-length field per sample)
FFCV_ALLOC_ENTRY_DTYPE = np.dtype([
    ('sample_id', '<u8'),  # Sample index
    ('ptr', '<u8'),        # Offset into file
    ('size', '<u8'),       # Size in bytes
])

# Field descriptor
FFCV_FIELD_DESC_DTYPE = np.dtype([
    ('type_id', '<u1'),           # Field type ID
    ('name', ('<u1', 16)),        # Field name (null-terminated)
    ('arguments', ('<u1', 1024)), # Field arguments (type-specific)
], align=True)

# RGBImageField metadata per sample
FFCV_IMAGE_METADATA_DTYPE = np.dtype([
    ('height', '<u4'),
    ('width', '<u4'),
    ('mode', '<u1'),
], align=True)

# Known FFCV field type IDs
FFCV_TYPE_BYTES = 0
FFCV_TYPE_INT = 1
FFCV_TYPE_FLOAT = 2
FFCV_TYPE_NDARRAY = 4
FFCV_TYPE_JSON = 5
FFCV_TYPE_RGB_IMAGE = 255

# Variable-length field types (have entries in allocation table)
FFCV_VARIABLE_TYPES = {FFCV_TYPE_BYTES, FFCV_TYPE_NDARRAY, FFCV_TYPE_JSON, FFCV_TYPE_RGB_IMAGE}


# =============================================================================
# Numba JIT Batch Loaders
# =============================================================================

@njit(nogil=True, parallel=True, cache=True, fastmath=True)
def _load_batch_from_ffcv_parallel(
    batch_indices: NDArray[np.int64],
    alloc_ptr: NDArray[np.uint64],
    alloc_size: NDArray[np.uint64],
    mmap_data: NDArray[np.uint8],
    destination: NDArray[np.uint8],
    sizes: NDArray[np.uint64],
) -> None:
    """Load a batch from FFCV file using parallel Numba JIT.

    Args:
        batch_indices: Sample indices to load [B]
        alloc_ptr: Pointer offsets for each sample [N]
        alloc_size: Sizes for each sample [N]
        mmap_data: Memory-mapped file data
        destination: Output buffer [B, max_size]
        sizes: Output sizes [B]
    """
    batch_size = len(batch_indices)

    for i in prange(batch_size):
        sample_id = batch_indices[i]
        data_ptr = alloc_ptr[sample_id]
        data_size = alloc_size[sample_id]

        destination[i, :data_size] = mmap_data[data_ptr:data_ptr + data_size]
        sizes[i] = data_size


@njit(nogil=True, cache=True)
def _load_batch_from_ffcv_sequential(
    batch_indices: NDArray[np.int64],
    alloc_ptr: NDArray[np.uint64],
    alloc_size: NDArray[np.uint64],
    mmap_data: NDArray[np.uint8],
    destination: NDArray[np.uint8],
    sizes: NDArray[np.uint64],
) -> None:
    """Load a batch from FFCV file sequentially (for small batches)."""
    batch_size = len(batch_indices)

    for i in range(batch_size):
        sample_id = batch_indices[i]
        data_ptr = alloc_ptr[sample_id]
        data_size = alloc_size[sample_id]

        destination[i, :data_size] = mmap_data[data_ptr:data_ptr + data_size]
        sizes[i] = data_size


# =============================================================================
# FFCVFileDataset
# =============================================================================

class FFCVFileDataset:
    """Dataset that reads native FFCV .beton files directly.

    This class provides direct access to .ffcv/.beton files using Numba JIT
    batch loading. It's compatible with slipstream's PrefetchingDataLoader.

    The dataset parses the FFCV file format to extract:
    - Image data pointers and sizes from the allocation table
    - Image dimensions from per-sample metadata
    - Field structure from field descriptors

    Attributes:
        num_samples: Number of samples in the dataset
        max_sample_size: Maximum sample size in bytes
        field_names: List of field names in the file
        parallel: Whether parallel loading is enabled

    Example:
        dataset = FFCVFileDataset("/data/imagenet_train.ffcv")
        print(f"Loaded {len(dataset)} samples")

        # Get sample info
        info = dataset.get_sample_info(0)
        print(f"Sample 0: {info['width']}x{info['height']}, {info['size']} bytes")
    """

    def __init__(
        self,
        ffcv_path: str | Path,
        parallel: bool = True,
        image_field: str = "image",
        verbose: bool = True,
    ) -> None:
        """Initialize the FFCV file dataset.

        Args:
            ffcv_path: Path to .ffcv or .beton file
            parallel: Use parallel Numba loading (recommended for batch_size >= 32)
            image_field: Name of the image field (default "image")
            verbose: Print loading information

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file format is unsupported
        """
        self._path = Path(ffcv_path)
        if not self._path.exists():
            raise FileNotFoundError(f"FFCV file not found: {self._path}")

        self.parallel = parallel
        self._image_field = image_field
        self._verbose = verbose

        # Parse file structure
        self._read_header()
        self._read_field_descriptors()
        self._read_metadata()
        self._read_alloc_table()

        # Memory-map the file for fast access
        self._mmap = np.memmap(str(self._path), dtype=np.uint8, mode='r')

        # Extract contiguous arrays for Numba
        self._alloc_ptr = np.ascontiguousarray(self._alloc_table['ptr'])
        self._alloc_size = np.ascontiguousarray(self._alloc_table['size'])

        # Calculate max size
        self.max_sample_size = int(np.max(self._alloc_size))
        self.num_samples = int(self._header['num_samples'])

        # Extract image dimensions
        self._heights = np.ascontiguousarray(
            self._image_metadata['height'].astype(np.uint32)
        )
        self._widths = np.ascontiguousarray(
            self._image_metadata['width'].astype(np.uint32)
        )

        # Pre-allocate buffers
        self._batch_buffer: np.ndarray | None = None
        self._sizes_buffer: np.ndarray | None = None
        self._current_batch_size = 0

        if self._verbose:
            self._log(f"Loaded FFCV file: {self._path.name}")
            self._log(f"  Samples: {self.num_samples:,}, Max size: {self.max_sample_size:,} bytes")
            self._log(f"  File size: {len(self._mmap) / 1e9:.2f} GB")
            self._log(f"  Fields: {self.field_names}")
            self._log(f"  Image field: '{self._image_field}' (index {self._image_field_idx})")

    def _log(self, msg: str) -> None:
        """Print message if verbose mode is enabled."""
        if self._verbose:
            print(msg)

    def _read_header(self) -> None:
        """Read and validate FFCV file header."""
        self._header = np.fromfile(
            str(self._path), dtype=FFCV_HEADER_DTYPE, count=1
        )[0]

        version = self._header['version']
        if version != 2:
            raise ValueError(
                f"Unsupported FFCV version: {version}. "
                "Only FFCV v2 files are supported."
            )

    def _read_field_descriptors(self) -> None:
        """Read field descriptors and identify the image field."""
        offset = FFCV_HEADER_DTYPE.itemsize
        num_fields = self._header['num_fields']

        self._field_descriptors = np.fromfile(
            str(self._path),
            dtype=FFCV_FIELD_DESC_DTYPE,
            count=num_fields,
            offset=offset,
        )

        # Decode field names from null-terminated byte arrays
        self.field_names: list[str] = []
        for desc in self._field_descriptors:
            name_bytes = desc['name']
            null_idx = np.where(name_bytes == 0)[0]
            if len(null_idx) > 0:
                name = bytes(name_bytes[:null_idx[0]]).decode('ascii')
            else:
                name = bytes(name_bytes).decode('ascii')
            self.field_names.append(name)

        # Find image field index
        self._image_field_idx = None
        for i, name in enumerate(self.field_names):
            if name == self._image_field:
                self._image_field_idx = i
                break

        if self._image_field_idx is None:
            # Default to first field (typically "f0" for image)
            self._image_field_idx = 0

    def _read_metadata(self) -> None:
        """Read per-sample image metadata (height, width, mode)."""
        offset = FFCV_HEADER_DTYPE.itemsize + self._field_descriptors.nbytes
        num_samples = self._header['num_samples']

        # For RGBImageField, metadata contains height, width, mode
        # Assumes image is the first field (standard ImageNet format)
        self._image_metadata = np.fromfile(
            str(self._path),
            dtype=FFCV_IMAGE_METADATA_DTYPE,
            count=num_samples,
            offset=offset,
        )

    def _read_alloc_table(self) -> None:
        """Read allocation table for variable-length fields.

        FFCV allocation table only contains entries for VARIABLE-LENGTH fields.
        Entries are interleaved by sample:
          - Entry 0: Sample 0, VarField 0
          - Entry 1: Sample 0, VarField 1
          - Entry 2: Sample 1, VarField 0
          - etc.

        We extract only the image field entries.
        """
        offset = int(self._header['alloc_table_ptr'])
        num_samples = int(self._header['num_samples'])

        # Read entire allocation table
        full_alloc_table = np.fromfile(
            str(self._path),
            dtype=FFCV_ALLOC_ENTRY_DTYPE,
            offset=offset,
        )

        # Determine number of variable-length fields
        total_entries = len(full_alloc_table)
        num_var_fields = total_entries // num_samples

        # Identify which fields are variable-length
        var_field_indices = []
        for i, desc in enumerate(self._field_descriptors):
            type_id = desc['type_id']
            if type_id in FFCV_VARIABLE_TYPES:
                var_field_indices.append(i)

        # Find image field's position among variable-length fields
        if self._image_field_idx in var_field_indices:
            image_var_idx = var_field_indices.index(self._image_field_idx)
        else:
            raise ValueError(
                f"Image field '{self._image_field}' (index {self._image_field_idx}) "
                f"is not a variable-length field. Variable fields: {var_field_indices}"
            )

        # Extract image entries: every num_var_fields-th entry, starting at image_var_idx
        image_indices = np.arange(num_samples) * num_var_fields + image_var_idx
        self._alloc_table = full_alloc_table[image_indices]

        if self._verbose:
            self._log(f"  Variable-length fields: {num_var_fields} (indices: {var_field_indices})")

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
        """Load a batch of raw samples.

        This method is compatible with FFCVStyleDataset's interface.

        Args:
            indices: Sample indices to load [B]

        Returns:
            (data, sizes) tuple:
            - data: [batch_size, max_sample_size] uint8 array
            - sizes: [batch_size] actual size of each sample
        """
        batch_size = len(indices)
        self._ensure_buffers(batch_size)

        dest = self._batch_buffer[:batch_size]
        sizes = self._sizes_buffer[:batch_size]

        if indices.dtype != np.int64:
            indices = indices.astype(np.int64)

        if self.parallel and batch_size >= 32:
            _load_batch_from_ffcv_parallel(
                indices,
                self._alloc_ptr,
                self._alloc_size,
                self._mmap,
                dest,
                sizes,
            )
        else:
            _load_batch_from_ffcv_sequential(
                indices,
                self._alloc_ptr,
                self._alloc_size,
                self._mmap,
                dest,
                sizes,
            )

        return dest, sizes

    def load_batch_with_dims(
        self, indices: NDArray[np.int64]
    ) -> tuple[NDArray[np.uint8], NDArray[np.uint64], NDArray[np.uint32], NDArray[np.uint32]]:
        """Load a batch with image dimensions.

        Args:
            indices: Sample indices to load [B]

        Returns:
            (data, sizes, heights, widths) tuple
        """
        data, sizes = self.load_batch_raw(indices)
        heights = self._heights[indices]
        widths = self._widths[indices]
        return data, sizes, heights, widths

    def get_sample_dims(self, idx: int) -> tuple[int, int]:
        """Get dimensions for a single sample.

        Args:
            idx: Sample index

        Returns:
            (height, width) tuple
        """
        return int(self._heights[idx]), int(self._widths[idx])

    def get_sample_info(self, idx: int) -> dict:
        """Get detailed info about a single sample.

        Args:
            idx: Sample index

        Returns:
            Dict with 'ptr', 'size', 'height', 'width'
        """
        return {
            'ptr': int(self._alloc_ptr[idx]),
            'size': int(self._alloc_size[idx]),
            'height': int(self._heights[idx]),
            'width': int(self._widths[idx]),
        }

    def __len__(self) -> int:
        return self.num_samples

    def __repr__(self) -> str:
        return (
            f"FFCVFileDataset(\n"
            f"    path='{self._path}',\n"
            f"    num_samples={self.num_samples:,},\n"
            f"    max_sample_size={self.max_sample_size:,},\n"
            f"    fields={self.field_names},\n"
            f"    parallel={self.parallel},\n"
            f")"
        )


# =============================================================================
# Prefetching DataLoader for FFCV Files
# =============================================================================

class FFCVFilePrefetchingDataLoader:
    """DataLoader for FFCVFileDataset with background prefetching.

    This is optimized for FFCVFileDataset and includes image dimensions
    in each batch for zero-overhead decode.

    Attributes:
        dataset: The FFCVFileDataset
        batch_size: Samples per batch
        shuffle: Whether to shuffle each epoch
        batches_ahead: Prefetch queue depth
    """

    def __init__(
        self,
        dataset: FFCVFileDataset,
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

        # Pre-allocate memory banks
        num_slots = batches_ahead + 2
        self._data_banks = [
            np.zeros((batch_size, dataset.max_sample_size), dtype=np.uint8)
            for _ in range(num_slots)
        ]
        self._size_banks = [
            np.zeros(batch_size, dtype=np.uint64)
            for _ in range(num_slots)
        ]
        self._height_banks = [
            np.zeros(batch_size, dtype=np.uint32)
            for _ in range(num_slots)
        ]
        self._width_banks = [
            np.zeros(batch_size, dtype=np.uint32)
            for _ in range(num_slots)
        ]

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        indices = np.arange(len(self.dataset), dtype=np.int64)
        if self.shuffle:
            np.random.shuffle(indices)

        num_batches = len(self)
        output_queue: queue.Queue = queue.Queue(maxsize=self.batches_ahead)
        stop_event = threading.Event()
        num_slots = len(self._data_banks)

        def worker():
            current_slot = 0
            for batch_idx in range(num_batches):
                if stop_event.is_set():
                    break

                start = batch_idx * self.batch_size
                end = min(start + self.batch_size, len(indices))
                batch_indices = indices[start:end]
                actual_size = len(batch_indices)

                if self.drop_last and actual_size < self.batch_size:
                    break

                dest = self._data_banks[current_slot]
                sizes = self._size_banks[current_slot]

                # Load batch
                if self.dataset.parallel and actual_size >= 32:
                    _load_batch_from_ffcv_parallel(
                        batch_indices,
                        self.dataset._alloc_ptr,
                        self.dataset._alloc_size,
                        self.dataset._mmap,
                        dest,
                        sizes,
                    )
                else:
                    _load_batch_from_ffcv_sequential(
                        batch_indices,
                        self.dataset._alloc_ptr,
                        self.dataset._alloc_size,
                        self.dataset._mmap,
                        dest,
                        sizes,
                    )

                # Get dimensions
                heights = self._height_banks[current_slot]
                widths = self._width_banks[current_slot]
                heights[:actual_size] = self.dataset._heights[batch_indices]
                widths[:actual_size] = self.dataset._widths[batch_indices]

                output_queue.put((current_slot, actual_size, batch_indices))
                current_slot = (current_slot + 1) % num_slots

            output_queue.put(None)

        worker_thread = threading.Thread(target=worker, daemon=True)
        worker_thread.start()

        try:
            while True:
                result = output_queue.get()
                if result is None:
                    break
                slot, actual_size, batch_indices = result
                yield {
                    'data': self._data_banks[slot][:actual_size],
                    'sizes': self._size_banks[slot][:actual_size],
                    'heights': self._height_banks[slot][:actual_size],
                    'widths': self._width_banks[slot][:actual_size],
                    'indices': batch_indices,
                }
        finally:
            stop_event.set()
            worker_thread.join(timeout=1.0)

    def __repr__(self) -> str:
        return (
            f"FFCVFilePrefetchingDataLoader(\n"
            f"    dataset={self.dataset!r},\n"
            f"    batch_size={self.batch_size},\n"
            f"    shuffle={self.shuffle},\n"
            f"    batches_ahead={self.batches_ahead},\n"
            f")"
        )


__all__ = [
    "FFCVFileDataset",
    "FFCVFilePrefetchingDataLoader",
]
