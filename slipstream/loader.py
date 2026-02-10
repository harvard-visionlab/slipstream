"""SlipstreamLoader: High-level API for training with streaming datasets.

This module provides the main training interface that combines:
- OptimizedCache for O(1) memory-mapped sample access
- Async batch prefetching with pre-allocated buffers
- Composable pipelines for decode/crop/normalize

Usage:
    from slipstream import SlipstreamDataset, SlipstreamLoader
    from slipstream.decoders import RandomResizedCrop
    from slipstream.transforms import Normalize

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
    from slipstream.decoders import BatchTransform


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
        from slipstream.decoders import RandomResizedCrop
        from slipstream.transforms import Normalize

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
        image_field: str | None = None,
        image_format: str = "jpeg",
        exclude_fields: list[str] | None = None,
        force_rebuild: bool = False,
        presync_s3: bool = False,
        presync_s3_workers: int = 32,
        presync_s3_endpoint_url: str | None = None,
        remote_cache: str | None = None,
        remote_cache_endpoint_url: str | None = None,
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
            image_field: Primary image field name for prefetch optimization.
                Auto-detected if None.
            image_format: Image format to use ("jpeg" or "yuv420"). Default "jpeg".
                Auto-adjusted if cache stores images in a different format.
            exclude_fields: List of field names to exclude from loading
            force_rebuild: Force rebuilding the optimized cache
            presync_s3: If True, use s5cmd to sync the dataset's S3 remote
                directory to local disk before building the optimized cache.
                Much faster than LitData's built-in download for large datasets.
                Requires s5cmd to be installed.
            presync_s3_workers: Number of parallel s5cmd workers for presync.
            presync_s3_endpoint_url: S3-compatible endpoint URL for presync
                (e.g. for Wasabi).
            remote_cache: S3 base path for cache discovery and sharing. When set,
                the loader will check for a pre-built cache at
                ``{remote_cache}/slipcache-{dataset_hash}/`` and download it if
                found. If not found, the cache is built locally and uploaded
                to S3 for future use. This enables sharing caches across machines.
                Example: ``"s3://my-bucket/slipstream-caches/"``.
            remote_cache_endpoint_url: S3-compatible endpoint URL for remote_cache
                (e.g., for Wasabi, MinIO).
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

        # Remote cache settings (stored for potential re-sync)
        self._remote_cache = remote_cache
        self._remote_cache_endpoint_url = remote_cache_endpoint_url
        self._remote_cache_full: str | None = None  # Set later when hash is computed

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

        # Remote cache discovery and download
        cache_downloaded = False
        remote_cache_full: str | None = None
        if remote_cache is not None:
            from slipstream.s3_sync import (
                download_s3_cache,
                s3_path_exists,
                upload_s3_cache,
            )

            dataset_hash = dataset.dataset_hash
            remote_cache_full = f"{remote_cache.rstrip('/')}/slipcache-{dataset_hash}"
            self._remote_cache_full = remote_cache_full
            remote_manifest = f"{remote_cache_full}/.slipstream/manifest.json"

            if s3_path_exists(remote_manifest, endpoint_url=remote_cache_endpoint_url):
                if verbose:
                    print(f"Found remote cache: {remote_cache_full}")

                # Download if local cache doesn't exist or force_rebuild
                if force_rebuild or not OptimizedCache.exists(cache_dir):
                    success = download_s3_cache(
                        remote_cache_full,
                        cache_dir,
                        endpoint_url=remote_cache_endpoint_url,
                        verbose=verbose,
                    )
                    if success:
                        cache_downloaded = True
                    elif verbose:
                        print("  Download failed, will build locally")
                else:
                    if verbose:
                        print("  Local cache exists, skipping download")
            elif verbose:
                print(f"Remote cache not found, will build and upload: {remote_cache_full}")

        # Determine whether to build or load
        needs_build = (force_rebuild or not OptimizedCache.exists(cache_dir)) and not cache_downloaded

        if needs_build:
            if verbose:
                print("Building optimized cache (this only happens once)...")
            self.cache = OptimizedCache.build(dataset, cache_dir, verbose=verbose)

            # Upload to remote if requested and we just built it
            if remote_cache is not None:
                from slipstream.s3_sync import upload_s3_cache
                try:
                    upload_s3_cache(
                        cache_dir,
                        remote_cache_full,
                        endpoint_url=remote_cache_endpoint_url,
                        verbose=verbose,
                    )
                except Exception as e:
                    if verbose:
                        print(f"  Warning: Failed to upload cache to S3: {e}")
                        print("  Continuing with local cache")
        else:
            self.cache = OptimizedCache.load(cache_dir, verbose=verbose)

        # Bidirectional sync: ensure local and remote have same derived files
        # (indexes, stats, YUV420 cache, etc.)
        if remote_cache is not None:
            from slipstream.s3_sync import sync_s3_cache
            try:
                downloaded, uploaded = sync_s3_cache(
                    cache_dir,
                    remote_cache_full,
                    endpoint_url=remote_cache_endpoint_url,
                    verbose=verbose,
                )
                # If files were downloaded, reload indexes to pick them up
                if downloaded > 0:
                    self.cache._discover_indexes()
                    if verbose and self.cache._indexes:
                        print(f"  Loaded indexes: {list(self.cache._indexes.keys())}")
            except Exception as e:
                if verbose:
                    print(f"  Warning: Bidirectional sync failed: {e}")

        # Auto-detect image fields from cache field types
        # TODO: Future enhancement - generalize prefetch banks to handle multiple
        # image fields if the need arises. Currently only the primary image field
        # gets pre-allocated memory banks for zero-copy async loading.
        self._image_fields: set[str] = set()
        for field_name, field_type in self.cache.field_types.items():
            if field_type in ("ImageBytes", "HFImageDict"):
                self._image_fields.add(field_name)

        # Auto-select primary image field for prefetch optimization
        if self.image_field is None and self._image_fields:
            self.image_field = next(iter(self._image_fields))
        elif self.image_field not in self._image_fields and self._image_fields:
            old_field = self.image_field
            self.image_field = next(iter(self._image_fields))
            if verbose:
                print(f"  Auto-detected image field: '{self.image_field}' (specified '{old_field}' not found)")

        # Check if cache stores non-JPEG images as YUV420 (auto-converted during build)
        stored_format = self.cache.get_image_format(self.image_field) if self._image_fields else "jpeg"
        if stored_format == "yuv420" and image_format == "jpeg":
            # Non-JPEG images were converted to YUV420 during cache build
            # Override user's image_format to use the stored format
            if verbose:
                print(f"Cache stores images as YUV420 (non-JPEG source), using YUV420 decoder")
            image_format = "yuv420"
            self.image_format = "yuv420"

        # Build/load alternative image format if requested
        if image_format == "yuv420" and stored_format != "yuv420":
            # User requested YUV420 but cache stores JPEG - need sibling cache
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

        # Configure pipelines for image fields based on the effective format
        # Use self.image_format (which accounts for sibling YUV420 cache)
        # rather than cache.get_image_format() (which only knows the main cache)
        for field_name, pipeline in self.pipelines.items():
            if field_name not in self._image_fields:
                continue
            field_format = self.image_format
            if field_format == "jpeg":
                continue  # JPEG is the default, no configuration needed

            if field_name in self._multi_pipeline_fields:
                for sub_pipeline in pipeline:
                    for transform in sub_pipeline:
                        if hasattr(transform, 'set_image_format'):
                            transform.set_image_format(field_format)
            else:
                for transform in pipeline:
                    if hasattr(transform, 'set_image_format'):
                        transform.set_image_format(field_format)

        # Determine which fields to load
        self._fields_to_load = [
            f for f in self.cache.fields.keys()
            if f not in self.exclude_fields
        ]

        # Pre-allocate memory banks for prefetching (only for image field)
        self._setup_prefetch_banks()

        # Track worker thread + stop event for cleanup between iterations
        self._worker_thread: threading.Thread | None = None
        self._stop_event: threading.Event | None = None

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

        Also resets seed counters on all decoders used by pipelines so that
        augmentations are reproducible from any epoch (e.g., checkpoint resume).
        """
        self._epoch = epoch

        # Reset seed counters on decoders to epoch * batches_per_epoch
        # so augmentations resume deterministically from this epoch.
        batches_per_epoch = len(self)
        target_counter = epoch * batches_per_epoch

        seen_decoders: set[int] = set()
        for field_name, pipeline in self.pipelines.items():
            transforms = []
            if field_name in self._multi_pipeline_fields:
                for sub_pipeline in pipeline:
                    transforms.extend(sub_pipeline)
            else:
                transforms.extend(pipeline)
            for transform in transforms:
                decoder = getattr(transform, '_decoder', None)
                if decoder is not None and id(decoder) not in seen_decoders:
                    seen_decoders.add(id(decoder))
                    decoder._seed_counter = target_counter

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
                    pipeline_result = self._apply_pipeline(
                        self.image_field, image_data
                    )
                    if isinstance(pipeline_result, dict):
                        batch.update(pipeline_result)
                    else:
                        batch[self.image_field] = pipeline_result
                else:
                    batch[self.image_field] = image_data

            # Load other fields
            for field_name in self._fields_to_load:
                if field_name == self.image_field:
                    continue

                field_result = self.cache.fields[field_name].load_batch(batch_indices)

                # Image-type fields need full dict (data, sizes, heights, widths)
                if field_name in self._image_fields:
                    field_data = field_result  # Full dict
                else:
                    field_data = field_result['data']  # Just the data

                if field_name in self.pipelines:
                    batch[field_name] = self._apply_pipeline(field_name, field_data)
                elif isinstance(field_data, np.ndarray):
                    batch[field_name] = torch.from_numpy(field_data).to(self._device_str)
                else:
                    batch[field_name] = field_data

            yield batch

    def _stop_worker(self) -> None:
        """Stop any running prefetch worker and wait for it to finish."""
        if self._stop_event is not None:
            self._stop_event.set()
        if self._worker_thread is not None and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5.0)
        self._worker_thread = None
        self._stop_event = None

    def _iter_threaded(self):
        """Threaded iteration with async prefetching."""
        # Ensure any previous worker is fully stopped before we start a new
        # one — otherwise two workers write to the same prefetch banks.
        self._stop_worker()

        indices = self._generate_indices(self._epoch)
        self._epoch += 1

        num_batches = len(indices) // self.batch_size
        if not self.drop_last and len(indices) % self.batch_size != 0:
            num_batches += 1

        # Queue passes only (slot, batch_size, batch_indices, other_fields)
        # Image data is accessed directly from pre-allocated banks using slot
        output_queue: queue.Queue = queue.Queue(maxsize=self.batches_ahead)
        stop_event = threading.Event()
        self._stop_event = stop_event
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

                # Load other fields (labels are fast - simple array indexing).
                # Use parallel=False for ALL loads in the worker thread —
                # Numba's workqueue threading layer is not reentrant, so the
                # main thread's NumbaBatchDecoder (parallel=True) would crash
                # if the worker also runs parallel Numba.
                other_fields = {}
                for field_name in self._fields_to_load:
                    if field_name == self.image_field:
                        continue
                    field_result = self.cache.fields[field_name].load_batch(
                        batch_indices, parallel=False
                    )
                    # Image-type fields need full dict (data, sizes, heights, widths)
                    if field_name in self._image_fields:
                        other_fields[field_name] = field_result  # Full dict
                    else:
                        other_fields[field_name] = field_result['data']  # Just the data

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
        self._worker_thread = worker

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
                        pipeline_result = self._apply_pipeline(
                            self.image_field, image_data
                        )
                        if isinstance(pipeline_result, dict):
                            batch.update(pipeline_result)
                        else:
                            batch[self.image_field] = pipeline_result
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
            # Drain the queue so the worker isn't blocked on put()
            while not output_queue.empty():
                try:
                    output_queue.get_nowait()
                except queue.Empty:
                    break
            worker.join(timeout=5.0)
            self._worker_thread = None

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
        self._stop_worker()
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

    def sync_remote_cache(self) -> tuple[int, int]:
        """Manually sync local cache with remote S3 cache.

        Call this after adding indexes, stats, or other derived files to ensure
        they are uploaded to the remote cache and available on other machines.

        Returns:
            Tuple of (downloaded_count, uploaded_count) indicating files transferred.
            Returns (0, 0) if remote_cache was not configured.

        Raises:
            RuntimeError: If s5cmd is not installed
        """
        if self._remote_cache is None or self._remote_cache_full is None:
            if self.verbose:
                print("No remote_cache configured, skipping sync")
            return (0, 0)

        from slipstream.s3_sync import sync_s3_cache

        return sync_s3_cache(
            self.cache.cache_dir.parent,  # .slipstream is inside cache_dir
            self._remote_cache_full,
            endpoint_url=self._remote_cache_endpoint_url,
            verbose=self.verbose,
        )

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
