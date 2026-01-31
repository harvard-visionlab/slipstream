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

import math
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
        dataset: Any,
        batch_size: int = 256,
        shuffle: bool = True,
        seed: int | None = None,
        distributed: bool = False,
        indices: Sequence[int] | np.ndarray | None = None,
        drop_last: bool = True,
        batches_ahead: int = 3,
        pipelines: dict[str, Sequence[BatchTransform] | BatchTransform | Callable] | None = None,
        device: int | str = 'cpu',
        image_field: str = "image",
        exclude_fields: list[str] | None = None,
        image_format: str = "jpeg",
        force_rebuild: bool = False,
        presync_s3: bool = False,
        presync_s3_workers: int = 32,
        presync_s3_endpoint_url: str | None = None,
        verbose: bool = True,
        use_threading: bool = True,
    ) -> None:
        """Initialize SlipstreamLoader.

        Args:
            dataset: SlipstreamDataset to load from
            batch_size: Number of samples per batch
            shuffle: Shuffle indices each epoch
            seed: Random seed for deterministic shuffle. If None, shuffle is
                non-deterministic. When set, epoch N uses seed (seed + N).
            distributed: Enable distributed training partitioning. Requires
                torch.distributed to be initialized. Each rank gets a
                disjoint strided subset of the shuffled indices.
            indices: Subset of dataset sample indices to use. If None, all
                samples are used. Useful for debugging, few-shot experiments,
                or custom sampling strategies.
            drop_last: Drop incomplete final batch
            batches_ahead: Number of batches to prefetch
            pipelines: Dict mapping field names to transform pipelines.
                Each pipeline can be a list of transforms, a single transform,
                or a callable. If None, raw data is returned.
            device: Device for non-pipelined fields (labels, indices)
            image_field: Name of the image field in the dataset
            image_format: Image storage format to use. "jpeg" (default) uses
                the standard JPEG cache. "yuv420" uses raw YUV420P storage
                for ~1.7-1.9x faster decode (built on demand from JPEG cache).
            exclude_fields: List of field names to exclude from loading
            force_rebuild: Force rebuilding the optimized cache
            presync_s3: If True, use s5cmd to sync the dataset's S3 remote
                directory to local disk before building the optimized cache.
                Much faster than LitData's built-in download for large datasets.
                Requires s5cmd to be installed.
            presync_s3_workers: Number of parallel s5cmd workers for presync.
            presync_s3_endpoint_url: S3-compatible endpoint URL for presync
                (e.g. for Wasabi).
            verbose: Print progress messages
            use_threading: Use background thread for prefetching (default True).
                Set to False for debugging to isolate threading overhead.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.indices = np.asarray(indices, dtype=np.int64) if indices is not None else None
        self.drop_last = drop_last
        self._epoch = 0

        # Distributed setup
        if distributed:
            import torch.distributed as dist
            if not dist.is_initialized():
                raise RuntimeError(
                    "torch.distributed must be initialized before creating "
                    "a distributed SlipstreamLoader"
                )
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1
        self.distributed = distributed
        self.batches_ahead = batches_ahead
        self.image_field = image_field
        self.image_format = image_format
        self.exclude_fields = set(exclude_fields or [])
        self.verbose = verbose
        self.use_threading = use_threading

        # Parse device for non-pipelined fields
        if isinstance(device, str):
            self._device_str = device if device == 'cpu' or ':' in device else f'{device}:0'
        else:
            self._device_str = f'cuda:{device}'

        # Store pipelines, detecting multi-pipeline fields
        self.pipelines: dict[str, list[Any]] = {}
        self._multi_pipeline_fields: set[str] = set()
        if pipelines:
            for field_name, pipeline in pipelines.items():
                if (isinstance(pipeline, (list, tuple))
                    and len(pipeline) > 0
                    and isinstance(pipeline[0], (list, tuple))):
                    # Multi-pipeline mode: list of sub-pipelines
                    self.pipelines[field_name] = [list(p) for p in pipeline]
                    self._multi_pipeline_fields.add(field_name)
                elif isinstance(pipeline, (list, tuple)):
                    self.pipelines[field_name] = list(pipeline)
                else:
                    self.pipelines[field_name] = [pipeline]

        # Pre-sync S3 data to local disk if requested
        if presync_s3:
            remote = getattr(dataset, 'remote_dir', None)
            if remote is not None:
                from slipstream.s3_sync import sync_s3_dataset
                sync_s3_dataset(
                    remote,
                    cache_dir=dataset.cache_path,
                    endpoint_url=presync_s3_endpoint_url,
                    numworkers=presync_s3_workers,
                    verbose=verbose,
                )
            elif verbose:
                print("presync_s3=True but dataset has no remote_dir, skipping sync")

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

        # Build/load alternative image format if requested
        if image_format == "yuv420":
            from slipstream.cache import build_yuv420_cache, load_yuv420_cache

            yuv_storage = load_yuv420_cache(self.cache.cache_dir, self.image_field)
            if yuv_storage is None:
                if verbose:
                    print("Building YUV420 cache (one-time conversion from JPEG)...")
                yuv_storage = build_yuv420_cache(
                    self.cache.cache_dir, self.image_field, verbose=verbose,
                )
            elif verbose:
                print(f"Loaded YUV420 cache ({yuv_storage.num_samples:,} samples)")
            self._image_storage = yuv_storage
        else:
            self._image_storage = self.cache.fields.get(self.image_field)

        # Configure pipelines for the selected image format
        if image_format != "jpeg":
            for field_name, pipeline in self.pipelines.items():
                if field_name != self.image_field:
                    continue
                if field_name in self._multi_pipeline_fields:
                    for sub_pipeline in pipeline:
                        for transform in sub_pipeline:
                            if hasattr(transform, 'set_image_format'):
                                transform.set_image_format(image_format)
                else:
                    for transform in pipeline:
                        if hasattr(transform, 'set_image_format'):
                            transform.set_image_format(image_format)

        # Determine which fields to load
        self._fields_to_load = [
            f for f in self.cache.fields.keys()
            if f not in self.exclude_fields
        ]

        # Pre-allocate memory banks for prefetching (only for image field)
        self._setup_prefetch_banks()

    def _generate_indices(self, epoch: int) -> np.ndarray:
        """Generate sample indices for an epoch.

        When shuffle is enabled, indices are shuffled using a deterministic
        RNG if seed is set, otherwise non-deterministic. When distributed,
        indices are padded to be evenly divisible by world_size, then each
        rank takes a strided subset (matching PyTorch DistributedSampler).

        Args:
            epoch: Current epoch number, used with seed for deterministic ordering.

        Returns:
            Array of sample indices for this rank to process.
        """
        if self.indices is not None:
            indices = self.indices.copy()
            n = len(indices)
        else:
            n = len(self.cache)
            indices = np.arange(n, dtype=np.int64)

        if self.shuffle:
            rng_seed = (self.seed + epoch) if self.seed is not None else None
            rng = np.random.default_rng(rng_seed)
            rng.shuffle(indices)

        if self.distributed:
            total = math.ceil(n / self.world_size) * self.world_size
            if total > n:
                indices = np.concatenate([indices, indices[:total - n]])
            indices = indices[self.rank::self.world_size]

        return indices

    def set_epoch(self, epoch: int) -> None:
        """Set epoch for deterministic shuffle ordering.

        In distributed training, call this before each epoch to ensure
        different shuffle orderings across epochs while keeping all
        ranks synchronized.
        """
        self._epoch = epoch

    def _setup_prefetch_banks(self) -> None:
        """Set up pre-allocated memory banks for async prefetching."""
        num_slots = self.batches_ahead + 2

        # Get image storage to know max size
        image_storage = self._image_storage
        if image_storage is not None:
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

    def _apply_single_pipeline(self, pipeline: list[Any], data: Any) -> Any:
        """Apply a single pipeline (list of transforms) to data."""
        result = data
        for transform in pipeline:
            result = transform(result)
        return result

    def _apply_pipeline(self, field_name: str, data: Any) -> Any:
        """Apply pipeline transforms to field data.

        For multi-pipeline fields, returns a list of results (one per sub-pipeline).
        """
        if field_name not in self.pipelines:
            return data

        if field_name in self._multi_pipeline_fields:
            return [
                self._apply_single_pipeline(sub_pipeline, data)
                for sub_pipeline in self.pipelines[field_name]
            ]

        return self._apply_single_pipeline(self.pipelines[field_name], data)

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
        indices = self._generate_indices(self._epoch)
        self._epoch += 1

        num_batches = len(indices) // self.batch_size
        if not self.drop_last and len(indices) % self.batch_size != 0:
            num_batches += 1

        image_storage = self._image_storage
        has_image_field = image_storage is not None and self._data_banks is not None

        for batch_idx in range(num_batches):
            start = batch_idx * self.batch_size
            end = min(start + self.batch_size, len(indices))
            batch_indices = indices[start:end]
            actual_size = len(batch_indices)

            # Build output batch
            batch = {
                '_indices': torch.from_numpy(batch_indices).to(self._device_str),
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
        indices = self._generate_indices(self._epoch)
        self._epoch += 1

        num_batches = len(indices) // self.batch_size
        if not self.drop_last and len(indices) % self.batch_size != 0:
            num_batches += 1

        # Queue passes only (slot, batch_size, batch_indices, other_fields)
        # Image data is accessed directly from pre-allocated banks using slot
        output_queue: queue.Queue = queue.Queue(maxsize=self.batches_ahead)
        stop_event = threading.Event()
        num_slots = len(self._data_banks) if self._data_banks else 1

        # Get the image field storage for direct access
        image_storage = self._image_storage
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
                # Use parallel=False here because Numba's workqueue threading
                # layer is not thread-safe — the main thread may concurrently
                # run NumbaBatchDecoder (also parallel=True). Sequential mmap
                # reads are fast enough that this doesn't bottleneck.
                if has_image_field:
                    image_storage.load_batch_into(
                        batch_indices,
                        self._data_banks[current_slot],
                        self._size_banks[current_slot],
                        self._height_banks[current_slot],
                        self._width_banks[current_slot],
                        parallel=False,
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
                    '_indices': torch.from_numpy(batch_indices).to(self._device_str),
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
        total = len(self.indices) if self.indices is not None else len(self.cache)
        if self.distributed:
            per_rank = math.ceil(total / self.world_size)
        else:
            per_rank = total
        if self.drop_last:
            return per_rank // self.batch_size
        return (per_rank + self.batch_size - 1) // self.batch_size

    def shutdown(self) -> None:
        """Release resources."""
        for field_name, pipeline in self.pipelines.items():
            if field_name in self._multi_pipeline_fields:
                for sub_pipeline in pipeline:
                    for transform in sub_pipeline:
                        if hasattr(transform, 'shutdown'):
                            transform.shutdown()
            else:
                for transform in pipeline:
                    if hasattr(transform, 'shutdown'):
                        transform.shutdown()

    def __del__(self) -> None:
        """Cleanup on deletion."""
        self.shutdown()

    def __repr__(self) -> str:
        pipeline_strs = []
        for field, pipeline in self.pipelines.items():
            if field in self._multi_pipeline_fields:
                sub_strs = []
                for sub in pipeline:
                    transforms = [type(t).__name__ for t in sub]
                    sub_strs.append(f"[{', '.join(transforms)}]")
                pipeline_strs.append(f"'{field}': [{', '.join(sub_strs)}]")
            else:
                transforms = [type(t).__name__ for t in pipeline]
                pipeline_strs.append(f"'{field}': [{', '.join(transforms)}]")

        pipelines_str = "{" + ", ".join(pipeline_strs) + "}" if pipeline_strs else "{}"

        indices_str = (
            f"    indices=subset ({len(self.indices):,} of {len(self.cache):,} total),\n"
            if self.indices is not None else ""
        )
        seed_str = f"    seed={self.seed},\n" if self.seed is not None else ""
        dist_str = (
            f"    distributed=True (rank={self.rank}, world_size={self.world_size}),\n"
            if self.distributed else ""
        )

        return (
            f"SlipstreamLoader(\n"
            f"    num_samples={len(self.indices) if self.indices is not None else len(self.cache):,},\n"
            f"    batch_size={self.batch_size},\n"
            f"    shuffle={self.shuffle},\n"
            f"{indices_str}"
            f"{seed_str}"
            f"{dist_str}"
            f"    pipelines={pipelines_str},\n"
            f"    device='{self._device_str}',\n"
            f"    fields={self._fields_to_load},\n"
            f"    excluded={list(self.exclude_fields)},\n"
            f")"
        )


__all__ = [
    "SlipstreamLoader",
]
