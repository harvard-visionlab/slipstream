"""SlipstreamLoader: High-level API for training with streaming datasets.

This module provides the main training interface that combines:
- OptimizedCache for O(1) memory-mapped sample access
- Async batch prefetching with pre-allocated buffers
- Composable pipelines for decode/crop/normalize

Usage:
    from slipstream import SlipstreamDataset, SlipstreamLoader
    from slipstream.pipelines import RandomResizedCrop, Normalize

    # Create dataset
    dataset = SlipstreamDataset(
        remote_dir="s3://bucket/imagenet/train",
        decode_images=False,
    )

    # Create loader with pipelines
    loader = SlipstreamLoader(
        dataset,
        batch_size=256,
        pipelines={
            'image': [
                RandomResizedCrop(224, device='cuda'),
                Normalize(),
            ],
        },
    )

    for batch in loader:
        images = batch['image']  # [B, 3, 224, 224] normalized tensor
        labels = batch['label']  # [B] tensor
        # Training...

    # Raw I/O benchmark (no pipelines)
    raw_loader = SlipstreamLoader(dataset, batch_size=256)
    for batch in raw_loader:
        raw_bytes = batch['image']  # dict with 'data', 'sizes', etc.

Performance:
    - Warm epochs: 480k+ images/sec raw I/O (memory-mapped, OS page cache)
    - With GPU decode + RRC: ~10k images/sec
    - With CPU decode + RRC: ~5.7k images/sec
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from slipstream.cache import OptimizedCache

if TYPE_CHECKING:
    from slipstream.dataset import SlipstreamDataset
    from slipstream.pipelines import BatchTransform


class SlipstreamLoader:
    """High-level data loader for training with streaming datasets.

    SlipstreamLoader provides FFCV-like performance by combining:
    1. OptimizedCache for O(1) memory-mapped sample access
    2. Async prefetching with pre-allocated buffers
    3. Composable pipelines for decode/crop/normalize

    The loader handles efficient I/O and batching. Processing (decode, crop,
    normalize) is handled by pipelines, which can be customized per field.

    On first use, the loader automatically builds an optimized cache
    from the dataset. This cache is stored in a .slipstream subdirectory
    and reused in subsequent runs.

    Attributes:
        dataset: The source SlipstreamDataset
        cache: The OptimizedCache for fast batch loading
        batch_size: Number of samples per batch
        pipelines: Dict mapping field names to transform pipelines

    Example:
        from slipstream.pipelines import RandomResizedCrop, Normalize

        # Training with pipelines
        loader = SlipstreamLoader(
            dataset,
            batch_size=256,
            pipelines={
                'image': [RandomResizedCrop(224, device='cuda'), Normalize()],
            },
        )

        for batch in loader:
            images = batch['image']  # [256, 3, 224, 224] on GPU
            labels = batch['label']  # [256] on GPU
            loss = model(images, labels)

        # Raw I/O (no pipelines, for benchmarking)
        raw_loader = SlipstreamLoader(dataset, batch_size=256)
    """

    def __init__(
        self,
        dataset: SlipstreamDataset,
        batch_size: int = 256,
        shuffle: bool = True,
        drop_last: bool = True,
        batches_ahead: int = 3,
        pipelines: dict[str, Sequence[BatchTransform] | BatchTransform | Callable] | None = None,
        device: int | str = 'cpu',
        image_field: str = "image",
        exclude_fields: list[str] | None = None,
        force_rebuild: bool = False,
        verbose: bool = True,
        use_threading: bool = True,
    ) -> None:
        """Initialize SlipstreamLoader.

        Args:
            dataset: SlipstreamDataset to load from
            batch_size: Number of samples per batch
            shuffle: Shuffle indices each epoch
            drop_last: Drop incomplete final batch
            batches_ahead: Number of batches to prefetch
            pipelines: Dict mapping field names to transform pipelines.
                Each pipeline can be a list of transforms, a single transform,
                or a callable. If None, raw data is returned.
            device: Device for non-pipelined fields (labels, indices)
            image_field: Name of the image field in the dataset
            exclude_fields: List of field names to exclude from loading
            force_rebuild: Force rebuilding the optimized cache
            verbose: Print progress messages
            use_threading: Use background thread for prefetching (default True).
                Set to False for debugging to isolate threading overhead.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.batches_ahead = batches_ahead
        self.image_field = image_field
        self.exclude_fields = set(exclude_fields or [])
        self.verbose = verbose
        self.use_threading = use_threading

        # Parse device for non-pipelined fields
        if isinstance(device, str):
            self._device_str = device if device == 'cpu' or ':' in device else f'{device}:0'
        else:
            self._device_str = f'cuda:{device}'

        # Store pipelines
        self.pipelines: dict[str, list[Any]] = {}
        if pipelines:
            for field_name, pipeline in pipelines.items():
                if isinstance(pipeline, (list, tuple)):
                    self.pipelines[field_name] = list(pipeline)
                else:
                    self.pipelines[field_name] = [pipeline]

        # Build or load optimized cache
        cache_dir = dataset.cache_path
        if cache_dir is None:
            raise ValueError(
                "Cannot determine cache directory from dataset. "
                "Ensure the dataset has a valid cache_dir."
            )

        if force_rebuild or not OptimizedCache.exists(cache_dir):
            if verbose:
                print("Building optimized cache (this only happens once)...")
            self.cache = OptimizedCache.build(dataset, cache_dir, verbose=verbose)
        else:
            self.cache = OptimizedCache.load(cache_dir, verbose=verbose)

        # Determine which fields to load
        self._fields_to_load = [
            f for f in self.cache.fields.keys()
            if f not in self.exclude_fields
        ]

        # Pre-allocate memory banks for prefetching (only for image field)
        self._setup_prefetch_banks()

    def _setup_prefetch_banks(self) -> None:
        """Set up pre-allocated memory banks for async prefetching."""
        num_slots = self.batches_ahead + 2

        # Get image storage to know max size
        if self.image_field in self.cache.fields:
            image_storage = self.cache.fields[self.image_field]
            max_size = image_storage.max_size

            self._data_banks = [
                np.zeros((self.batch_size, max_size), dtype=np.uint8)
                for _ in range(num_slots)
            ]
            self._size_banks = [
                np.zeros(self.batch_size, dtype=np.uint64)
                for _ in range(num_slots)
            ]
            self._height_banks = [
                np.zeros(self.batch_size, dtype=np.uint32)
                for _ in range(num_slots)
            ]
            self._width_banks = [
                np.zeros(self.batch_size, dtype=np.uint32)
                for _ in range(num_slots)
            ]
        else:
            self._data_banks = None

    def _apply_pipeline(self, field_name: str, data: Any) -> Any:
        """Apply pipeline transforms to field data."""
        if field_name not in self.pipelines:
            return data

        result = data
        for transform in self.pipelines[field_name]:
            result = transform(result)

        return result

    def __iter__(self):
        """Iterate over batches with async prefetching.

        Uses zero-copy loading: JIT functions write directly into pre-allocated
        buffers, and only slot indices are passed through the queue.
        """
        if self.use_threading:
            yield from self._iter_threaded()
        else:
            yield from self._iter_simple()

    def _iter_simple(self):
        """Simple iteration without threading (for debugging/profiling)."""
        indices = np.arange(len(self.cache), dtype=np.int64)
        if self.shuffle:
            np.random.shuffle(indices)

        num_batches = len(indices) // self.batch_size
        if not self.drop_last and len(indices) % self.batch_size != 0:
            num_batches += 1

        image_storage = self.cache.fields.get(self.image_field)
        has_image_field = image_storage is not None and self._data_banks is not None

        for batch_idx in range(num_batches):
            start = batch_idx * self.batch_size
            end = min(start + self.batch_size, len(indices))
            batch_indices = indices[start:end]
            actual_size = len(batch_indices)

            # Build output batch
            batch = {
                'indices': torch.from_numpy(batch_indices).to(self._device_str),
            }

            # Load and add image data
            if has_image_field:
                image_storage.load_batch_into(
                    batch_indices,
                    self._data_banks[0],
                    self._size_banks[0],
                    self._height_banks[0],
                    self._width_banks[0],
                    parallel=True,
                )
                image_data = {
                    'data': self._data_banks[0][:actual_size],
                    'sizes': self._size_banks[0][:actual_size],
                    'heights': self._height_banks[0][:actual_size],
                    'widths': self._width_banks[0][:actual_size],
                }

                if self.image_field in self.pipelines:
                    batch[self.image_field] = self._apply_pipeline(
                        self.image_field, image_data
                    )
                else:
                    batch[self.image_field] = image_data

            # Load other fields
            for field_name in self._fields_to_load:
                if field_name == self.image_field:
                    continue
                field_data = self.cache.fields[field_name].load_batch(batch_indices)['data']
                if field_name in self.pipelines:
                    batch[field_name] = self._apply_pipeline(field_name, field_data)
                elif isinstance(field_data, np.ndarray):
                    batch[field_name] = torch.from_numpy(field_data).to(self._device_str)
                else:
                    batch[field_name] = field_data

            yield batch

    def _iter_threaded(self):
        """Threaded iteration with async prefetching."""
        indices = np.arange(len(self.cache), dtype=np.int64)
        if self.shuffle:
            np.random.shuffle(indices)

        num_batches = len(indices) // self.batch_size
        if not self.drop_last and len(indices) % self.batch_size != 0:
            num_batches += 1

        # Queue passes only (slot, batch_size, batch_indices, other_fields)
        # Image data is accessed directly from pre-allocated banks using slot
        output_queue: queue.Queue = queue.Queue(maxsize=self.batches_ahead)
        stop_event = threading.Event()
        num_slots = len(self._data_banks) if self._data_banks else 1

        # Get the image field storage for direct access
        image_storage = self.cache.fields.get(self.image_field)
        has_image_field = image_storage is not None and self._data_banks is not None

        def prefetch_worker():
            """Background thread for async batch loading.

            Mimics FFCV's EpochIterator: JIT runs with nogil=True, releasing GIL.
            """
            current_slot = 0

            for batch_idx in range(num_batches):
                if stop_event.is_set():
                    break

                start = batch_idx * self.batch_size
                end = min(start + self.batch_size, len(indices))
                batch_indices = indices[start:end]
                actual_batch_size = len(batch_indices)

                # Load image data directly into pre-allocated buffers (ZERO-COPY!)
                if has_image_field:
                    image_storage.load_batch_into(
                        batch_indices,
                        self._data_banks[current_slot],
                        self._size_banks[current_slot],
                        self._height_banks[current_slot],
                        self._width_banks[current_slot],
                        parallel=True,
                    )

                # Load other fields (labels are fast - simple array indexing)
                other_fields = {}
                for field_name in self._fields_to_load:
                    if field_name == self.image_field:
                        continue
                    field_data = self.cache.fields[field_name].load_batch(batch_indices)
                    other_fields[field_name] = field_data['data']

                # Only pass slot index and metadata - not the actual data!
                output_queue.put((
                    current_slot,
                    actual_batch_size,
                    batch_indices,
                    other_fields,
                ))
                current_slot = (current_slot + 1) % num_slots

            output_queue.put(None)

        worker = threading.Thread(target=prefetch_worker, daemon=True)
        worker.start()

        try:
            while True:
                result = output_queue.get()
                if result is None:
                    break

                slot, actual_size, batch_indices, other_fields = result

                # Build output batch
                batch = {
                    'indices': torch.from_numpy(batch_indices).to(self._device_str),
                }

                # Access image data from pre-allocated banks using slot index
                if has_image_field:
                    image_data = {
                        'data': self._data_banks[slot][:actual_size],
                        'sizes': self._size_banks[slot][:actual_size],
                        'heights': self._height_banks[slot][:actual_size],
                        'widths': self._width_banks[slot][:actual_size],
                    }

                    if self.image_field in self.pipelines:
                        batch[self.image_field] = self._apply_pipeline(
                            self.image_field, image_data
                        )
                    else:
                        # No pipeline - return raw data dict
                        batch[self.image_field] = image_data

                # Add other fields
                for field_name, field_data in other_fields.items():
                    if field_name in self.pipelines:
                        batch[field_name] = self._apply_pipeline(field_name, field_data)
                    elif isinstance(field_data, np.ndarray):
                        batch[field_name] = torch.from_numpy(field_data).to(self._device_str)
                    else:
                        # Strings or other types - keep as-is
                        batch[field_name] = field_data

                yield batch
        finally:
            stop_event.set()
            worker.join(timeout=1.0)

    def __len__(self) -> int:
        """Return number of batches per epoch."""
        if self.drop_last:
            return len(self.cache) // self.batch_size
        return (len(self.cache) + self.batch_size - 1) // self.batch_size

    def shutdown(self) -> None:
        """Release resources."""
        for field_name, pipeline in self.pipelines.items():
            for transform in pipeline:
                if hasattr(transform, 'shutdown'):
                    transform.shutdown()

    def __del__(self) -> None:
        """Cleanup on deletion."""
        self.shutdown()

    def __repr__(self) -> str:
        pipeline_strs = []
        for field, pipeline in self.pipelines.items():
            transforms = [type(t).__name__ for t in pipeline]
            pipeline_strs.append(f"'{field}': [{', '.join(transforms)}]")

        pipelines_str = "{" + ", ".join(pipeline_strs) + "}" if pipeline_strs else "{}"

        return (
            f"SlipstreamLoader(\n"
            f"    num_samples={len(self.cache):,},\n"
            f"    batch_size={self.batch_size},\n"
            f"    pipelines={pipelines_str},\n"
            f"    device='{self._device_str}',\n"
            f"    fields={self._fields_to_load},\n"
            f"    excluded={list(self.exclude_fields)},\n"
            f")"
        )


__all__ = [
    "SlipstreamLoader",
]
