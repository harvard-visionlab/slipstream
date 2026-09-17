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
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from slipstream.cache import MANIFEST_FILE, OptimizedCache

if TYPE_CHECKING:
    from slipstream.dataset import SlipstreamDataset
    from slipstream.decoders import BatchTransform


# =============================================================================
# Cache warmup helpers
# =============================================================================

_PAGE = 4096
# Merge record ranges separated by less than this many bytes: reading a few
# spare pages is cheaper than an extra seek, and adjacent records in a
# subset are usually page-neighbours anyway.
_COALESCE_GAP = 64 * 1024


def _coalesce_ranges(
    starts: np.ndarray, ends: np.ndarray, gap: int = _COALESCE_GAP
) -> list[tuple[int, int]]:
    """Sort ``(start, end)`` byte ranges by offset and merge touching/near ones.

    Empty ranges are dropped. Ranges whose start lies within ``gap`` bytes of
    the running end are merged into it.
    """
    if len(starts) == 0:
        return []
    order = np.argsort(starts, kind="stable")
    starts = starts[order]
    ends = ends[order]
    out: list[tuple[int, int]] = []
    cur_s: int | None = None
    cur_e = 0
    for s, e in zip(starts.tolist(), ends.tolist()):
        if e <= s:
            continue
        if cur_s is None:
            cur_s, cur_e = s, e
        elif s <= cur_e + gap:
            cur_e = max(cur_e, e)
        else:
            out.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    if cur_s is not None:
        out.append((cur_s, cur_e))
    return out


def _madvise_willneed(fpath: Path, ranges: list[tuple[int, int]] | None) -> None:
    """Best-effort MADV_WILLNEED (+ MADV_SEQUENTIAL for whole files) on ``fpath``.

    Kicks off kernel readahead for the pages the follow-up read will touch.
    Silently a no-op where madvise is unavailable.
    """
    import ctypes

    MADV_SEQUENTIAL, MADV_WILLNEED = 2, 3
    try:
        libc = ctypes.CDLL(None)
        mm = np.memmap(fpath, dtype=np.uint8, mode="r")
        base = mm.ctypes.data
        if ranges is None:
            spans = [(0, mm.nbytes)]
            libc.madvise(
                ctypes.c_void_p(base), ctypes.c_size_t(mm.nbytes),
                ctypes.c_int(MADV_SEQUENTIAL),
            )
        else:
            spans = [
                (s - (s % _PAGE), min(e, mm.nbytes)) for s, e in ranges
            ]
        for s, e in spans:
            if e > s:
                libc.madvise(
                    ctypes.c_void_p(base + s), ctypes.c_size_t(e - s),
                    ctypes.c_int(MADV_WILLNEED),
                )
        del mm
    except Exception:
        pass


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
        after_batch_transforms: list[Callable] | None = None,
        window: int | tuple[int, int] | None = None,
        sample_data: dict[str, Any] | None = None,
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
            image_field: Primary variable-size bytes field that gets the
                zero-copy prefetch banks. May be an image field (ImageBytes /
                HFImageDict) or a raw ``bytes`` field (e.g. a video container).
                Auto-detected if None: first image field, else first ``bytes``
                field. Its batch value (without a pipeline) is a dict
                ``{data, sizes, heights, widths}`` of views into the current
                bank slot. Every other bytes-backed field is returned as an
                owned ``{data, sizes}`` dict (plus heights/widths for images).
            image_format: Image format to use ("jpeg" or "yuv420"). Default "jpeg".
                Auto-adjusted if cache stores images in a different format.
                Ignored when the primary field is a raw ``bytes`` field.
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
            after_batch_transforms: Optional list of callables applied to the
                whole batch dict after per-field pipelines complete. Each
                callable receives ``batch: dict`` and returns a (possibly
                modified) ``batch: dict``. Use for batch-level ops that
                consume multiple fields (e.g. Mixup / CutMix on image+label).
                With ``window`` set, they see the folded ``[B, T, ...]`` batch.
            window: Sequence windows ``(T, stride)`` (or ``T`` for stride 1).
                ``indices`` are then *anchors*: for each anchor ``a`` the loader
                reads records ``a, a+stride, ..., a+(T-1)*stride`` and returns
                every field with a leading ``[B, T, ...]``. Shuffle, distributed
                sharding and drop_last work on anchors; ``batch_size`` counts
                windows. Per-sample augmentation parameters (crop, flip, color,
                ...) are drawn once per window and shared by its T frames, so a
                seeded run is reproducible pixel for pixel. If ``indices`` is
                None, every record ``a`` with ``a+(T-1)*stride < len(cache)`` is
                an anchor. ``batch['_indices']`` is ``[B, T]`` record indices and
                ``batch['_anchors']`` the ``[B]`` anchors.
            sample_data: Per-sample side arrays aligned with ``indices`` (which
                is then required), e.g. ``{"t0": window_start_seconds}``. They
                travel with their sample through shuffling and sharding: each
                batch carries ``batch[name]`` (a ``[B]`` tensor / list) and the
                primary field's pipeline receives them under
                ``batch_data['sample_data']``. This is how a record index can
                be repeated with different per-sample parameters.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.indices = np.asarray(indices, dtype=np.int64) if indices is not None else None
        self.drop_last = drop_last
        self._epoch = 0

        # Window (sequence) sampling: T records per sample, `stride` apart
        if window is None:
            T, stride = 1, 1
        elif isinstance(window, int):
            T, stride = window, 1
        else:
            T, stride = int(window[0]), int(window[1])
        if T < 1 or stride < 1:
            raise ValueError(f"window must be (T >= 1, stride >= 1), got {window!r}")
        self.window_size = T
        self.window_stride = stride

        # Per-sample side data aligned with `indices` (shuffled / sharded together with them)
        self.sample_data: dict[str, np.ndarray] = {}
        if sample_data:
            if self.indices is None:
                raise ValueError("sample_data requires indices (the arrays are aligned with it)")
            for k, v in sample_data.items():
                arr = np.asarray(v)
                if len(arr) != len(self.indices):
                    raise ValueError(f"sample_data[{k!r}] has {len(arr)} entries, indices has {len(self.indices)}")
                self.sample_data[k] = arr

        # Distributed setup — auto-detect if torch.distributed is initialized
        if not distributed:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
                distributed = True
                import warnings
                warnings.warn(
                    "torch.distributed is initialized but distributed=False was passed. "
                    "Auto-enabling distributed mode for correct DDP sharding.",
                    stacklevel=2,
                )
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
        self.after_batch_transforms: list[Callable] = list(after_batch_transforms or [])

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
            remote_manifest = f"{remote_cache_full}/{MANIFEST_FILE}"

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
                        # Verify downloaded cache integrity
                        is_valid, problems = OptimizedCache.check_integrity(cache_dir)
                        if not is_valid:
                            if verbose:
                                print(f"  Downloaded cache is incomplete: {problems}")
                            OptimizedCache._wipe_cache(cache_dir, "; ".join(problems))
                            cache_downloaded = False
                    elif verbose:
                        print("  Download failed, will build locally")
                else:
                    if verbose:
                        print("  Local cache exists, skipping download")
            elif verbose:
                print(f"Remote cache not found, will build and upload: {remote_cache_full}")

        # Check integrity of existing local cache
        needs_build = force_rebuild and not cache_downloaded
        if not needs_build and not cache_downloaded:
            if OptimizedCache.exists(cache_dir):
                is_valid, problems = OptimizedCache.check_integrity(cache_dir)
                if not is_valid:
                    if verbose:
                        print(f"Cache integrity check failed: {problems}")
                    OptimizedCache._wipe_cache(cache_dir, "; ".join(problems))
                    needs_build = True
                else:
                    needs_build = False
            else:
                needs_build = True

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
            try:
                self.cache = OptimizedCache.load(cache_dir, verbose=verbose)
            except Exception as e:
                # Safety net: if load fails despite integrity check passing
                # (e.g., race condition, corrupt mmap), wipe and rebuild
                if verbose:
                    print(f"Cache load failed ({e}), rebuilding...")
                OptimizedCache._wipe_cache(cache_dir, str(e))
                self.cache = OptimizedCache.build(dataset, cache_dir, verbose=verbose)

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
        #
        # Two kinds of variable-size byte fields share ImageBytesStorage:
        #   - image fields (ImageBytes / HFImageDict): get JPEG/YUV420 format
        #     handling and can be decoded by pipelines
        #   - raw `bytes` fields (e.g. video MP4, np.save blobs): stored and
        #     returned as-is, never decoded or format-converted
        # Both are "bank-eligible": any of them may be the primary field that
        # gets the slot-rotated zero-copy prefetch banks. Image fields win
        # auto-detection; declaration order breaks ties.
        self._image_fields: set[str] = set()
        self._bytes_fields: set[str] = set()
        for field_name, field_type in self.cache.field_types.items():
            if field_type in ("ImageBytes", "HFImageDict"):
                self._image_fields.add(field_name)
            elif field_type == "bytes":
                self._bytes_fields.add(field_name)
        bank_eligible = [
            f for f in self.cache.field_types if f in self._image_fields
        ] + [
            f for f in self.cache.field_types if f in self._bytes_fields
        ]

        # Auto-select primary field for prefetch optimization
        if self.image_field is None and bank_eligible:
            self.image_field = bank_eligible[0]
        elif self.image_field not in bank_eligible and bank_eligible:
            old_field = self.image_field
            self.image_field = bank_eligible[0]
            if verbose:
                print(f"  Auto-detected image field: '{self.image_field}' (specified '{old_field}' not found)")
        elif self.image_field is not None and not bank_eligible:
            # Named field is not variable-size bytes (or does not exist):
            # nothing to prefetch into banks.
            self.image_field = None

        # Format handling (JPEG vs YUV420) only applies to true image fields.
        # A raw `bytes` primary (manifest image_format e.g. "bytes"/"mp4")
        # must never trigger YUV420 detection or conversion.
        primary_is_image = self.image_field in self._image_fields

        # Check if cache stores non-JPEG images as YUV420 (auto-converted during build)
        stored_format = self.cache.get_image_format(self.image_field) if primary_is_image else "jpeg"
        if primary_is_image and stored_format == "yuv420" and image_format == "jpeg":
            # Non-JPEG images were converted to YUV420 during cache build
            # Override user's image_format to use the stored format
            if verbose:
                print(f"Cache stores images as YUV420 (non-JPEG source), using YUV420 decoder")
            image_format = "yuv420"
            self.image_format = "yuv420"

        # Build/load alternative image format if requested
        if primary_is_image and image_format == "yuv420" and stored_format != "yuv420":
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

        # Async decode stage: the primary field's pipeline may start with an object exposing
        # submit()/collect() (e.g. DecodeVideoWindow). The prefetch thread then calls submit()
        # as soon as a batch's bytes are in the bank and the main thread collect()s, so
        # `batches_ahead * batch_size` decodes are in flight instead of one batch's worth.
        self._async_stage = None
        self._async_rest: list[Any] = []
        prim = self.pipelines.get(self.image_field) if self.image_field is not None else None
        if prim and self.image_field not in self._multi_pipeline_fields:
            first = prim[0]
            if hasattr(first, 'submit') and hasattr(first, 'collect'):
                self._async_stage = first
                self._async_rest = list(prim[1:])
                if hasattr(first, 'set_batches_ahead'):
                    first.set_batches_ahead(self.batches_ahead)   # its output ring must outlive our prefetch depth

        # Determine which fields to load
        self._fields_to_load = [
            f for f in self.cache.fields.keys()
            if f not in self.exclude_fields
        ]

        # Windows: anchors must leave room for the whole window, and every
        # random augmentation must draw its parameters once per window.
        if self.window_size > 1:
            span = (self.window_size - 1) * self.window_stride
            if self.indices is not None:
                bad = self.indices[(self.indices < 0) | (self.indices + span >= len(self.cache))]
                if len(bad):
                    raise ValueError(
                        f"{len(bad)} anchor(s) leave no room for a window of {self.window_size} "
                        f"records at stride {self.window_stride} (e.g. anchor {int(bad[0])}, "
                        f"cache has {len(self.cache)} records)"
                    )
            elif len(self.cache) <= span:
                raise ValueError(
                    f"cache has {len(self.cache)} records, too few for a window of "
                    f"{self.window_size} at stride {self.window_stride}"
                )
        self._propagate_seed_repeat(self.window_size)

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
        n = self._num_anchors()
        pos = np.arange(n, dtype=np.int64)          # positions into self.indices (or the anchor range)

        if self.shuffle:
            rng_seed = (self.seed + epoch) if self.seed is not None else None
            rng = np.random.default_rng(rng_seed)
            rng.shuffle(pos)

        if self.distributed:
            total = math.ceil(n / self.world_size) * self.world_size
            if total > n:
                pos = np.concatenate([pos, pos[:total - n]])
            pos = pos[self.rank::self.world_size]

        self._epoch_positions = pos                  # lets the iterators pick up sample_data per batch
        return self.indices[pos] if self.indices is not None else pos

    def _batch_sample_data(self, start: int, end: int) -> dict[str, np.ndarray]:
        """sample_data rows for the batch occupying positions [start, end) of this epoch's order."""
        if not self.sample_data:
            return {}
        pos = self._epoch_positions[start:end]
        return {k: v[pos] for k, v in self.sample_data.items()}

    def _num_anchors(self) -> int:
        """Number of samples (anchors when windowed) before sharding."""
        if self.indices is not None:
            return len(self.indices)
        return len(self.cache) - (self.window_size - 1) * self.window_stride

    def _expand_window(self, anchors: np.ndarray) -> np.ndarray:
        """Anchors [B] -> record indices [B*T] (anchor-major: window i is rows i*T .. i*T+T-1)."""
        if self.window_size == 1:
            return anchors
        offsets = np.arange(self.window_size, dtype=np.int64) * self.window_stride
        return (anchors[:, None] + offsets[None, :]).reshape(-1)

    @staticmethod
    def _walk_transforms(obj: Any):
        """Yield obj and everything it wraps: pipeline lists, `.transforms`, `._decoder`, `._cpu_decoder`."""
        if obj is None:
            return
        if isinstance(obj, (list, tuple)):
            for o in obj:
                yield from SlipstreamLoader._walk_transforms(o)
            return
        yield obj
        for attr in ('transforms', '_decoder', '_cpu_decoder'):
            if attr == 'transforms' and getattr(obj, 'owns_transforms', False):
                continue                      # e.g. DecodeVideoWindow manages its inner transforms itself
            inner = getattr(obj, attr, None)
            if inner is not None and inner is not obj:
                yield from SlipstreamLoader._walk_transforms(inner)

    def _propagate_seed_repeat(self, T: int) -> None:
        """Tell every decoder / augmentation in the pipelines to share random params across T frames."""
        for obj in self._walk_transforms(list(self.pipelines.values())):
            if hasattr(obj, 'seed_repeat'):
                try:
                    obj.seed_repeat = T
                except AttributeError:
                    pass

    def _fold_window(self, batch: dict[str, Any], num_windows: int) -> dict[str, Any]:
        """Reshape every per-record value [B*T, ...] into [B, T, ...] (lists -> nested lists)."""
        T = self.window_size
        keep = {'_indices', '_anchors'}

        def fold(v: Any) -> Any:
            if isinstance(v, torch.Tensor):
                return v.reshape(num_windows, T, *v.shape[1:]) if v.ndim >= 1 and v.shape[0] == num_windows * T else v
            if isinstance(v, np.ndarray):
                return v.reshape(num_windows, T, *v.shape[1:]) if v.ndim >= 1 and v.shape[0] == num_windows * T else v
            if isinstance(v, dict):
                return {k: fold(x) for k, x in v.items()}
            if isinstance(v, (list, tuple)):
                if len(v) == num_windows * T and not any(isinstance(x, (torch.Tensor, np.ndarray)) and getattr(x, 'ndim', 0) >= 3 for x in v[:1]):
                    out = [list(v[i * T:(i + 1) * T]) for i in range(num_windows)]
                    return out if isinstance(v, list) else tuple(out)
                return type(v)(fold(x) for x in v)      # e.g. multi-pipeline outputs, list of decoded frames
            return v

        return {k: (v if k in keep else fold(v)) for k, v in batch.items()}

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

        seen: set[int] = set()
        for obj in self._walk_transforms(list(self.pipelines.values())):
            if hasattr(obj, '_seed_counter') and id(obj) not in seen:
                seen.add(id(obj))
                obj._seed_counter = target_counter

    def _setup_prefetch_banks(self) -> None:
        """Set up pre-allocated memory banks for async prefetching."""
        num_slots = self.batches_ahead + 2

        # Get image storage to know max size
        image_storage = self._image_storage
        if image_storage is not None:
            max_size = image_storage.max_size
            rows = self.batch_size * self.window_size      # one row per record of the expanded batch

            self._data_banks = [
                np.zeros((rows, max_size), dtype=np.uint8)
                for _ in range(num_slots)
            ]
            self._size_banks = [
                np.zeros(rows, dtype=np.uint64)
                for _ in range(num_slots)
            ]
            self._height_banks = [
                np.zeros(rows, dtype=np.uint32)
                for _ in range(num_slots)
            ]
            self._width_banks = [
                np.zeros(rows, dtype=np.uint32)
                for _ in range(num_slots)
            ]
        else:
            self._data_banks = None

    def _load_secondary_bytes_field(
        self,
        field_name: str,
        batch_indices: np.ndarray,
        parallel: bool,
    ) -> dict[str, np.ndarray]:
        """Load a non-primary variable-size bytes field as an owned copy.

        ``ImageBytesStorage.load_batch`` returns views into a single scratch
        buffer that is reused by the next call. For the primary field the
        loader avoids this via slot-rotated banks; every other bytes-backed
        field (secondary image fields, raw ``bytes`` fields such as np.save
        blobs or video containers) gets copied here so the batch dict stays
        valid after the prefetch worker moves on.

        Returns:
            ``{'data': uint8 [B, max_size_in_batch], 'sizes': uint64 [B]}``
            plus ``'heights'``/``'widths'`` for image-typed fields. Row ``i``
            holds ``data[i, :sizes[i]]``.
        """
        storage = self.cache.fields[field_name]
        result = storage.load_batch(batch_indices, parallel=parallel)
        n = len(batch_indices)
        sizes = np.array(result['sizes'][:n], dtype=np.uint64)  # copy
        width = int(sizes.max()) if n > 0 else 0
        out: dict[str, np.ndarray] = {
            'data': np.array(result['data'][:n, :width], dtype=np.uint8),  # copy
            'sizes': sizes,
        }
        if field_name in self._image_fields:
            out['heights'] = np.array(result['heights'][:n], dtype=np.uint32)
            out['widths'] = np.array(result['widths'][:n], dtype=np.uint32)
        return out

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
        if self.verbose and not getattr(self, '_residency_checked', False):
            self._residency_checked = True
            frac = self.page_cache_residency(max_records=512)
            if frac is not None and frac < 0.5:
                print(
                    f"  Note: only {frac * 100:.0f}% of this loader's records are in the OS page cache; "
                    f"on a network mount cold reads serialize badly. Consider loader.warmup_cache() first."
                )
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
            anchors = indices[start:end]
            batch_indices = self._expand_window(anchors)     # records to read ([B*T] when windowed)
            actual_size = len(batch_indices)

            # Build output batch
            batch = self._index_fields(anchors, batch_indices)
            sdata = self._batch_sample_data(start, end)
            self._add_sample_data(batch, sdata)

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
                image_data = self._primary_dict(0, actual_size, batch_indices, sdata)

                if self.image_field in self.pipelines:
                    pipeline_result = self._apply_pipeline(
                        self.image_field, image_data
                    )
                    if isinstance(pipeline_result, dict):
                        batch.update(pipeline_result)
                    else:
                        batch[self.image_field] = pipeline_result
                else:
                    batch[self.image_field] = self._raw_view(image_data)

            # Load other fields
            for field_name in self._fields_to_load:
                if field_name == self.image_field:
                    continue

                if field_name in self._image_fields or field_name in self._bytes_fields:
                    # Variable-size bytes: dict with data + sizes (owned copy)
                    field_data = self._load_secondary_bytes_field(
                        field_name, batch_indices, parallel=True
                    )
                else:
                    field_result = self.cache.fields[field_name].load_batch(batch_indices)
                    field_data = field_result['data']  # Just the data

                if field_name in self.pipelines:
                    batch[field_name] = self._apply_pipeline(field_name, field_data)
                elif isinstance(field_data, np.ndarray):
                    batch[field_name] = torch.from_numpy(field_data).to(self._device_str)
                else:
                    batch[field_name] = field_data

            if self.window_size > 1:
                batch = self._fold_window(batch, len(anchors))
            for transform in self.after_batch_transforms:
                batch = transform(batch)

            yield batch

    def _index_fields(self, anchors: np.ndarray, batch_indices: np.ndarray) -> dict[str, Any]:
        """The bookkeeping entries of a batch dict: '_indices' ([B] or [B, T] record indices), '_anchors' when windowed."""
        if self.window_size == 1:
            return {'_indices': torch.from_numpy(batch_indices).to(self._device_str)}
        rec = torch.from_numpy(batch_indices.reshape(len(anchors), self.window_size)).to(self._device_str)
        return {'_indices': rec, '_anchors': torch.from_numpy(anchors).to(self._device_str)}

    def _primary_dict(self, slot: int, n: int, batch_indices: np.ndarray, sdata: dict[str, np.ndarray]) -> dict[str, Any]:
        """The primary field's pipeline input: bank views plus bookkeeping the stages may use."""
        d = {
            'data': self._data_banks[slot][:n],
            'sizes': self._size_banks[slot][:n],
            'heights': self._height_banks[slot][:n],
            'widths': self._width_banks[slot][:n],
            'indices': batch_indices,
            'field': self.image_field,
        }
        if sdata:
            d['sample_data'] = sdata
        return d

    @staticmethod
    def _raw_view(d: dict[str, Any]) -> dict[str, Any]:
        """The user-facing raw bytes dict: data/sizes/heights/widths only (bookkeeping keys are for stages)."""
        return {k: d[k] for k in ('data', 'sizes', 'heights', 'widths')}

    def _add_sample_data(self, batch: dict[str, Any], sdata: dict[str, np.ndarray]) -> None:
        for k, v in sdata.items():
            batch[k] = torch.from_numpy(v).to(self._device_str) if v.dtype.kind in 'biuf' else v.tolist()

    def _stop_worker(self) -> None:
        """Stop any running prefetch worker and wait for it to finish."""
        # getattr: shutdown()/__del__ may run on a loader whose __init__ raised early
        ev = getattr(self, '_stop_event', None)
        if ev is not None:
            ev.set()
        th = getattr(self, '_worker_thread', None)
        if th is not None and th.is_alive():
            th.join(timeout=5.0)
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
            Any exception is handed to the consumer through the queue so the
            main thread never blocks on a dead worker.
            """
            try:
                _prefetch_loop()
            except BaseException as exc:          # noqa: BLE001 - re-raised on the main thread
                output_queue.put(exc)

        def _prefetch_loop():
            current_slot = 0

            for batch_idx in range(num_batches):
                if stop_event.is_set():
                    break

                start = batch_idx * self.batch_size
                end = min(start + self.batch_size, len(indices))
                anchors = indices[start:end]
                batch_indices = self._expand_window(anchors)     # records to read ([B*T] when windowed)
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

                # Async stage: start decoding this batch now; the main thread collects it later.
                pending = None
                if has_image_field and self._async_stage is not None:
                    pending = self._async_stage.submit(
                        self._primary_dict(current_slot, actual_batch_size, batch_indices,
                                           self._batch_sample_data(start, end)))

                # Load other fields (labels are fast - simple array indexing).
                # Use parallel=False for ALL loads in the worker thread —
                # Numba's workqueue threading layer is not reentrant, so the
                # main thread's NumbaBatchDecoder (parallel=True) would crash
                # if the worker also runs parallel Numba.
                other_fields = {}
                for field_name in self._fields_to_load:
                    if field_name == self.image_field:
                        continue
                    if field_name in self._image_fields or field_name in self._bytes_fields:
                        # Variable-size bytes: the storage's load_batch hands
                        # back a view into ONE shared scratch buffer that the
                        # next batch overwrites, while the main thread may
                        # still be consuming this batch (worker runs up to
                        # batches_ahead batches ahead). Copy out the payload.
                        other_fields[field_name] = self._load_secondary_bytes_field(
                            field_name, batch_indices, parallel=False
                        )
                    else:
                        field_result = self.cache.fields[field_name].load_batch(
                            batch_indices, parallel=False
                        )
                        other_fields[field_name] = field_result['data']  # Just the data

                # Only pass slot index and metadata - not the actual data!
                output_queue.put((
                    current_slot,
                    actual_batch_size,
                    anchors,
                    batch_indices,
                    other_fields,
                    (start, end),
                    pending,
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
                if isinstance(result, BaseException):
                    raise result

                slot, actual_size, anchors, batch_indices, other_fields, (start, end), pending = result

                # Build output batch
                batch = self._index_fields(anchors, batch_indices)
                sdata = self._batch_sample_data(start, end)
                self._add_sample_data(batch, sdata)

                # Access image data from pre-allocated banks using slot index
                if has_image_field:
                    image_data = self._primary_dict(slot, actual_size, batch_indices, sdata)

                    if pending is not None:
                        pipeline_result = self._async_stage.collect(pending)
                        for transform in self._async_rest:
                            pipeline_result = transform(pipeline_result)
                        if isinstance(pipeline_result, dict):
                            batch.update(pipeline_result)
                        else:
                            batch[self.image_field] = pipeline_result
                    elif self.image_field in self.pipelines:
                        pipeline_result = self._apply_pipeline(
                            self.image_field, image_data
                        )
                        if isinstance(pipeline_result, dict):
                            batch.update(pipeline_result)
                        else:
                            batch[self.image_field] = pipeline_result
                    else:
                        # No pipeline - return raw data dict
                        batch[self.image_field] = self._raw_view(image_data)

                # Add other fields
                for field_name, field_data in other_fields.items():
                    if field_name in self.pipelines:
                        batch[field_name] = self._apply_pipeline(field_name, field_data)
                    elif isinstance(field_data, np.ndarray):
                        batch[field_name] = torch.from_numpy(field_data).to(self._device_str)
                    else:
                        # Strings or other types - keep as-is
                        batch[field_name] = field_data

                if self.window_size > 1:
                    batch = self._fold_window(batch, len(anchors))
                for transform in self.after_batch_transforms:
                    batch = transform(batch)

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
        """Return number of batches per epoch (batches of anchors when windowed)."""
        total = self._num_anchors()
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
        for field_name, pipeline in getattr(self, 'pipelines', {}).items():
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
            self.cache.cache_dir,
            self._remote_cache_full,
            endpoint_url=self._remote_cache_endpoint_url,
            verbose=self.verbose,
        )

    def warmup_cache(
        self,
        verbose: bool = True,
        indices: Sequence[int] | np.ndarray | None = None,
    ) -> dict:
        """Pre-read cache files to populate OS page cache. No decoding, no pipeline execution.

        This makes the first epoch fast by avoiding on-demand page faults during training.

        When a subset is in play (``indices`` here, or the loader's own
        ``indices``), only the byte ranges of the selected records are read
        for variable-size fields (``.bin`` of image / ``bytes`` / ``str``
        fields): record ranges are sorted by offset and coalesced, so a
        contiguous subset becomes a single sequential read. Fixed-size
        ``.npy`` fields and metadata tables are read whole (they are tiny per
        record). With no subset, every ``*.bin`` / ``*.npy`` in the cache dir
        is read sequentially.

        Args:
            verbose: Show tqdm progress bar with throughput.
            indices: Records to warm. ``None`` (default) uses the loader's
                ``indices`` (the full subset, not this rank's shard); if the
                loader has none, the whole cache is read. Pass
                ``np.arange(len(loader.cache))`` to force a full read on a
                subset loader. With ``window`` set these are anchors and every
                record of every window is warmed.

        Returns:
            dict with: elapsed_sec, total_bytes, throughput_mb_s, cache_dir,
            num_files, subset (bool), num_records, num_ranges
        """
        import sys
        import time

        from tqdm.auto import tqdm

        cache_dir = Path(self.cache.cache_dir)

        if indices is None:
            indices = self.indices
        if indices is not None:
            indices = np.asarray(indices, dtype=np.int64)
            if self.window_size > 1:                     # anchors -> every record of every window
                indices = self._expand_window(indices)
            indices = np.unique(indices)
        elif self.window_size > 1:
            indices = np.arange(len(self.cache), dtype=np.int64)   # windows over all anchors cover the whole cache

        if indices is None:
            # Full read: every data file in the cache dir
            data_files = sorted(cache_dir.glob("*.bin")) + sorted(cache_dir.glob("*.npy"))
            plan = [(f, None) for f in data_files]
            num_records = len(self.cache)
        else:
            plan = self._warmup_plan(indices)
            num_records = len(indices)

        total_bytes = sum(
            f.stat().st_size if ranges is None else int(sum(e - s for s, e in ranges))
            for f, ranges in plan
        )
        num_ranges = sum(1 if ranges is None else len(ranges) for _, ranges in plan)

        if not plan:
            return dict(
                elapsed_sec=0,
                total_bytes=0,
                throughput_mb_s=0,
                cache_dir=str(cache_dir),
                num_files=0,
                subset=indices is not None,
                num_records=num_records,
                num_ranges=0,
            )

        # Phase 1: madvise hints (best-effort, Linux/macOS)
        for fpath, ranges in plan:
            _madvise_willneed(fpath, ranges)

        # Phase 2: read to fault pages into cache
        CHUNK = 16 * 1024 * 1024  # 16 MB
        t0 = time.time()

        pbar = tqdm(
            total=total_bytes,
            unit="B",
            unit_scale=True,
            desc="Cache warmup",
            disable=not verbose,
            leave=True,
            file=sys.stdout,
        )
        for fpath, ranges in plan:
            with open(fpath, "rb") as f:
                if ranges is None:
                    while True:
                        chunk = f.read(CHUNK)
                        if not chunk:
                            break
                        pbar.update(len(chunk))
                else:
                    for start, end in ranges:
                        f.seek(start)
                        remaining = end - start
                        while remaining > 0:
                            chunk = f.read(min(CHUNK, remaining))
                            if not chunk:
                                break
                            remaining -= len(chunk)
                            pbar.update(len(chunk))
        pbar.close()

        elapsed = time.time() - t0
        return dict(
            elapsed_sec=elapsed,
            total_bytes=total_bytes,
            throughput_mb_s=(total_bytes / (1024**2)) / elapsed if elapsed > 0 else 0,
            cache_dir=str(cache_dir),
            num_files=len(plan),
            subset=indices is not None,
            num_records=num_records,
            num_ranges=num_ranges,
        )

    def _warmup_plan(
        self, indices: np.ndarray
    ) -> list[tuple[Path, list[tuple[int, int]] | None]]:
        """Build the per-file read plan for a subset warmup.

        Returns ``[(path, ranges)]`` where ``ranges`` is a sorted, coalesced
        list of ``(start, end)`` byte offsets, or ``None`` for whole-file
        reads. Only fields the loader will actually load are included, plus
        the sibling YUV420 store when that is the active primary storage.
        """
        from slipstream.cache import ImageBytesStorage, StringStorage

        cache_dir = Path(self.cache.cache_dir)
        storages: dict[str, Any] = {
            name: self.cache.fields[name] for name in self._fields_to_load
        }
        primary = self._image_storage
        if primary is not None and primary is not self.cache.fields.get(self.image_field):
            storages[primary.field_name] = primary  # sibling YUV420 cache

        plan: list[tuple[Path, list[tuple[int, int]] | None]] = []
        for name, storage in storages.items():
            if isinstance(storage, ImageBytesStorage):
                meta = storage._metadata
                starts = meta['data_ptr'][indices].astype(np.int64)
                ends = starts + meta['data_size'][indices].astype(np.int64)
                plan.append((cache_dir / f"{name}.bin", _coalesce_ranges(starts, ends)))
                plan.append((cache_dir / f"{name}.meta.npy", None))
            elif isinstance(storage, StringStorage):
                offs = np.asarray(storage._offsets)[indices]
                starts = offs[:, 0].astype(np.int64)
                ends = starts + offs[:, 1].astype(np.int64)
                plan.append((cache_dir / f"{name}.bin", _coalesce_ranges(starts, ends)))
                plan.append((cache_dir / f"{name}.offsets.npy", None))
            else:
                plan.append((cache_dir / f"{name}.npy", None))

        return [(p, r) for p, r in plan if p.exists()]

    def page_cache_residency(self, indices: Sequence[int] | np.ndarray | None = None, max_records: int = 4096) -> float | None:
        """Fraction of the primary field's bytes for ``indices`` that are resident in the OS page cache.

        Uses ``mincore`` on the store's mmap (Linux / macOS); returns None where unavailable. Samples
        at most ``max_records`` records evenly. ``indices`` default to the loader's own (anchors are
        expanded when windowed). Cheap: no data is read.
        """
        import ctypes
        import sys

        storage = self._image_storage
        if storage is None or not hasattr(storage, '_metadata'):
            return None
        if indices is None:
            indices = self.indices if self.indices is not None else np.arange(self._num_anchors(), dtype=np.int64)
        indices = np.asarray(indices, dtype=np.int64)
        if self.window_size > 1:
            indices = self._expand_window(indices)
        if len(indices) == 0:
            return None
        if len(indices) > max_records:
            indices = indices[np.linspace(0, len(indices) - 1, max_records).astype(np.int64)]
        meta = storage._metadata
        starts = meta['data_ptr'][indices].astype(np.int64)
        ends = starts + meta['data_size'][indices].astype(np.int64)
        mm = storage._data_mmap
        try:
            libc = ctypes.CDLL(None)
            page = 4096 if sys.platform != 'darwin' else 16384
            base = mm.ctypes.data
            vec_t = ctypes.c_ubyte if sys.platform != 'darwin' else ctypes.c_char
            resident = total = 0
            for s0, e0 in zip(starts.tolist(), ends.tolist()):
                a = s0 - (s0 % page)
                n_pages = (e0 - a + page - 1) // page
                vec = (vec_t * n_pages)()
                if libc.mincore(ctypes.c_void_p(base + a), ctypes.c_size_t(n_pages * page), vec) != 0:
                    return None
                resident += sum(1 for v in vec if (v if isinstance(v, int) else ord(v)) & 1)
                total += n_pages
            return resident / total if total else None
        except Exception:
            return None

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
        window_str = (
            f"    window=(T={self.window_size}, stride={self.window_stride}),\n"
            if self.window_size > 1 else ""
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
            f"{window_str}"
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
