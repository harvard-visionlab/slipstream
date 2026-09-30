# Changelog

All notable changes to slipstream are documented here. Versions follow
[Semantic Versioning](https://semver.org/); the version lives in
`slipstream/version.py`.

## [0.10.0] - 2026-09-30

### Added

- `SlipstreamLoader(on_invalid_cache="rebuild" | "raise")`. `"rebuild"` (default) keeps the old
  behaviour. `"raise"` raises `CacheIntegrityError` (new, `slipstream.CacheIntegrityError`) listing the
  problems, and never deletes or rebuilds: use it for shared read-only caches. It is automatic when the
  dataset reads the cache directly (`SlipstreamDataset(input_dir=<cache>)`). Before, a failed check
  wiped the only copy and then tried to rebuild it from that same, deleted, directory.
- Content hashes: builds record `"file_sha256": {fname: sha256 hex}` next to `file_sizes` (same files:
  every field storage file, not manifest.json or derived index files).
  `OptimizedCache.check_integrity(cache_dir, deep=True, workers=None)` also hashes (in parallel), and
  reports `no file_sha256 in manifest` on older caches. The `(ok, problems)` return is unchanged.
  `OptimizedCache.add_hashes(cache_dir)` and `slipstream hash <cache_dir>` add hashes to an existing
  cache (run on a trusted copy; refuses a copy that fails the cheap check).

### Fixed

- A directory with slipcache data files but no manifest.json (e.g. purged by scratch cleanup) is no
  longer rebuilt in place by the loader, or handed to another reader by `SlipstreamDataset`: both
  raise `CacheIntegrityError`. Use `force_rebuild=True` to rebuild on purpose.
- manifest.json is written last and atomically: builds fsync the data files, then write a temp file,
  fsync it and rename it. `download_s3_cache` removes a stale local manifest, copies the data, then
  fetches the manifest last. `upload_s3_cache` uploads the data before the manifest. An interrupted
  build or copy no longer looks complete.

## [0.9.6] - 2026-09-30

### Fixed

- The wheel is tagged per interpreter and platform (e.g. `cp311-cp311-macosx_15_0_arm64`), not
  `py3-none-any`, and contains only this interpreter's `_libslipstream<EXT_SUFFIX>`. Before, uv
  cached one "pure" wheel per git commit and installed, for example, a cpython-312 decoder into a
  3.10 venv (without running the build hook, so the 0.9.3 `--force` never applied). Stale builds for
  other Pythons in a shared checkout were also packed into it (via the old `artifacts` glob).
- The sdist no longer bundles `.devcontainer/cache` (a local uv cache), notebooks or build
  leftovers: 164 MB → 0.8 MB.

## [0.9.5] - 2026-09-30

### Fixed

- `use_threading=True` (the default) was 10-20x slower than `use_threading=False`: 3.9k vs 38k
  img/s (jpeg) and 2.3k vs 45k (yuv420) on machina. The prefetch thread reads records with the
  serial cache kernel. Under numba 0.67 (in the lock since 0.4.5; BENCHMARKS.md was measured on
  0.63) its slice assignment `dest[i, :n] = src[p:p + n]` compiles to a copy of ~1.8 GB/s, 20x
  slower than a plain loop. The cache read kernels, and the dev-only FFCV loaders' copies, now
  use explicit byte loops: serial reads went from 22k to 447k img/s (jpeg) and from 13k to 305k
  (yuv420) on an M4 Pro. Output bytes are identical. Regression tests in
  `tests/test_cache_copy_kernels.py`.
  machina release gate against v0.9.4: threaded jpeg 3.9k → 41.4k img/s, yuv420 2.3k → 44.0k. The full
  GPU pipeline (ToTorchImage + Normalize bf16 + flip) went from 3.9k → 24.2k (jpeg) and 2.3k → 28.2k (yuv420).

### Added

- Release gate: `benchmarks/loader_ablation.py` (steps A-G, from bare decode to a GPU training pipeline)
  and `benchmarks/release_gate.py`. It runs on machina before every tag, refuses to start on a busy
  machine, saves `benchmarks/results/gate/<host>/v<version>.json`, appends to BENCHMARKS.md, and fails
  on a >10% drop vs the previous version or when threaded prefetch is below 75% of simple.

## [0.9.4] - 2026-09-30

### Fixed

- The decoder loader now prefers the libslipstream build for the running interpreter
  (`_libslipstream<EXT_SUFFIX>`). Before, it took the first `_libslipstream*.so` in directory
  order, so a stale build for another Python version left in the same checkout could be loaded,
  along with the older libturbojpeg it linked. It still falls back to another build, with a
  `RuntimeWarning`. `slipstream status` flags a decoder built for another Python.

## [0.9.3] - 2026-09-30

### Fixed

- The build hook now always recompiles and relinks libslipstream (`build_ext --force`). Before,
  setuptools skipped the build when an up-to-date `.so` was already in the source tree (e.g. uv's
  cached git checkout of a rev), so the decoder kept the rpath and libturbojpeg of the first
  build and ignored a changed `TURBOJPEG_ROOT`, even after `uv cache clean` + `--reinstall-package`.
  `setup.py` also prints which libturbojpeg it links.

## [0.9.2] - 2026-09-30

### Changed

- `import slipstream` is lazy (PEP 562): every public name, and submodules such as
  `slipstream.decoders`, loads on first access. The import no longer pulls in torch, numba,
  litdata or torchvision: it now takes ~10 ms, down from 7.9 s cold on FASRC with 0.9.0 (3.3 s with
  only torchvision made lazy) and 1.2 s locally. `from slipstream import X`, `slipstream.X`,
  `from slipstream import *` and type checkers (via a `TYPE_CHECKING` block) work as before.
  `decode_image` imports torchvision on first call. `slipstream.cli` no longer loads numba
  (`MANIFEST_FILE` now lives in `slipstream.utils.cache_dir`; `slipstream.cache` re-exports it).
  No API change.
- The build hook now fails the install when the libslipstream C++ extension can't be built,
  instead of warning and producing a slipstream whose decode pipelines fail at the first batch.
  `SLIPSTREAM_SKIP_EXT=1` installs without it on purpose (e.g. CLI-only machines).

### Fixed

- `libslipstream/setup.py` also searches `$TURBOJPEG_ROOT`, `$CONDA_PREFIX` and `~/.local`
  (`include`, `lib` and `lib64`, with rpaths). A cmake install of libjpeg-turbo into ~/.local
  (lib64) or a conda-forge install was not found before.

### Added

- `slipstream status` reports whether the decoder is built and loadable, and lists a missing
  decoder under Problems. There is a new `decoder` key in `--json`, and `slipstream.cli.check_decoder()`.

## [0.9.1] - 2026-09-29

### Fixed

- Decoders with `seed=None` are now non-reproducible, as documented: their per-sample draws come
  from fresh OS entropy. Before, `seed=None` was keyed like a fixed seed, so every unseeded run got
  the same crops, positions and embed placements. Seeded streams are unchanged from 0.9.0.

### Documentation

- `FFCVFilePrefetchingDataLoader`, `FFCVStyleDataLoader` and `PrefetchingDataLoader` are marked
  development / benchmarking only. `SlipstreamLoader` is the supported training loader.

## [0.9.0] - 2026-09-29

### Changed (seeded streams differ from 0.8.0)

- Every seeded stream is now a pure function of `(seed, rank, epoch)` (`slipstream.seeds`). The
  loader reseeds all decoders, seeded transforms (`BatchAugment`s, `RandomApply`) and
  `after_batch_transforms` (`Mixup`, `CutMixClutter`, `SideBySideSearchPair`) at the start of every
  epoch and in `set_epoch`. That includes objects nested in `MultiCropPipeline` / `Compose` /
  `RandomApply` / wrappers and `DecodeVideoWindow`'s inner transforms.
  - Fixes: seeded transforms were seeded once at construction and never reset, so a run resumed
    at epoch N replayed epoch 0's flips / jitter / rotations / mixup.
  - Fixes: decoders that advance their counter more than once per batch (e.g.
    `DecodeUniformMultiRandomResizedCrop`) resumed from the wrong counter.
  - Resume no longer depends on `set_epoch`'s counter arithmetic or on what else ran in between.
- The global `torch.distributed` rank is part of every augmentation key, so DDP ranks no longer
  draw identical crop boxes / flip masks at the same batch position. The epoch shuffle still
  excludes the rank (one shared permutation, strided per rank).
- Pipeline presets derive per-view / per-transform seeds with `derive_seed(base, offset, view)`
  (hashed) instead of `base + offset + view`. The sum made base `b` view `k+1` identical to base
  `b+1` view `k`.
- Decoder `_seed_counter` now counts batches within the current epoch; the epoch and rank live in
  `_seed_key`.

### Added

- `slipstream.derive_seed(base, *parts)`: 63-bit hashed seed for configs (use instead of
  adding offsets to a base seed).
- `slipstream.seeds`: `derive_seed`, `sample_seeds`, `stream_seed`, `epoch_rng`, `reseed`.
  Objects used outside a loader behave as rank 0, epoch 0.
- `seed=None` keyword on the random-crop methods of `GPUDecoder`, `GPUDecoderFallback` and
  `CPUDecoder` (`decode_batch_random_crop`, `_dct`, `_to_tensor`), which previously could not be
  seeded at all. Seeded crops are keyed per sample like the Numba decoders (reset by the loader's
  reseed), and GPU and CPU give the same boxes for the same seed. `generate_batch_random_crop_params`
  takes per-sample `seeds`. Unseeded behaviour is unchanged.

## [0.8.0] - 2026-09-29

### Changed (seeded streams differ from 0.7.x)

- Seeds are now hashed instead of added. Every seeded per-sample draw in the decoders (RRC /
  direct RRC crops, multi-crop views, resize-short-crop-long sizes and positions, embed placement,
  `DecodeVideoWindow` t0) uses `SeedSequence([seed, counter, *stream])` through
  `slipstream.decoders._seeds.sample_seeds`, and the loader's epoch shuffle uses
  `SeedSequence([seed, epoch])`. Previously the per-sample seed was `seed + B*counter + i` and the
  shuffle seed was `seed + epoch`. That made seed `s+1` replay seed `s` shifted by one sample (or
  one epoch), gave SSL view `k+1` of sample `i` the draws of view `k` of sample `i+1`, and made
  some size draws reuse the position seed of the same or the next batch. Same seed + same
  version is still bit-reproducible, `set_epoch` resume is unchanged, and crops that share a seed
  are still yoked. **A given seed now produces different crops and orders than 0.7.x.**
- Private Numba crop kernels (`_generate_*_params_batch`, `_embed_batch_rgba`) take a per-sample
  `seeds` array instead of a scalar `seed`.

## [0.7.1] - 2026-09-29

### Fixed

- `affine_transform` (so `RandomRotate` and the other affine augments) crashed on bf16 images
  (`expected scalar type BFloat16 but found Float`): the fp32 sampling grid and the image dtype
  differed. The image is now sampled in fp32 whenever its dtype differs from the grid's and cast
  back; fp32 / CPU fp16 results are unchanged.
- `DecodeMultiResizeCropEmbed` (and anything else calling `decode_batch_resize_short_crop_long`)
  with `image_format="yuv420"`: `YUV420NumbaBatchDecoder` now has that method (same crop geometry
  and output as the JPEG decoder).
- `SlipstreamLoader.set_epoch` now reaches wrappers that hold their decoder as `_inner`
  (`DecodeMultiResizeCropEmbed`) and resets `_embed_seed_counter` too, so a run resumed at epoch
  N reproduces the fresh run's crops and embed placements. The same walk propagates `seed_repeat`
  to the inner crop decoder, so windowed loaders now share its crop params across a window's T
  frames as documented for 0.7.0.

## [0.7.0] - 2026-09-16

### Added

- `SlipstreamLoader(window=(T, stride))`: sequence windows. `indices` are anchors; for each
  anchor `a` the loader reads records `a, a+stride, ..., a+(T-1)*stride` and returns every
  field folded to `[B, T, ...]` (tensors and arrays reshape, string lists nest, raw bytes
  dicts fold `data`/`sizes`). `batch['_indices']` is `[B, T]`, `batch['_anchors']` is `[B]`.
  Shuffle, distributed sharding and `drop_last` operate on anchors; `batch_size` counts
  windows; `len(loader)` counts anchor batches. Prefetch banks hold `batch_size * T` rows.
  `warmup_cache()` warms every record of every window. `window=None` / `(1, 1)` is the
  previous behaviour exactly.
- Window-consistent augmentation: every decoder and `BatchAugment` transform that draws
  per-sample random parameters carries `seed_repeat` (set by the loader from `T`). Crop
  params (`NumbaBatchDecoder`, `YUV420NumbaBatchDecoder`, GPU decoder, multi-crop wrappers)
  and per-sample draws in flip / rotate / zoom / rotate-object / color jitter (HSV, YIQ) /
  grayscale / blur / solarization / patch shuffle / brightness / contrast / erasing / embed
  are drawn once per window and shared by its T frames, so a seeded run reproduces the
  same anchor order and the same pixels. `slipstream.decoders._window` holds the helpers.
- Fixed-shape array field types `"<dtype>[d0,d1,...]"` (e.g. `"float32[7]"`, `"int16[2,3]"`)
  in writer, storage, parallel-build merge and `verify()`; stored as one `(N, d0, ...)`
  `.npy`, returned as `[B, d0, ...]` (or `[B, T, d0, ...]`) tensors. Readers that report
  `np.ndarray` values have the type inferred from the first sample.
- `benchmarks/bench_window_loader.py`: windows/s and frames/s for a frame store.
- `DecodeVideoWindow` (`slipstream.decoders.video`): time-based video window decoding with
  torchcodec for a raw `bytes` video field. Per record it decodes `T` frames at `rate_hz`
  from `t0` and returns `{field: [B, T, 3, H, W] uint8, field_t_sec: [B, T] true frame
  times, field_t0: [B], field_rec: [B]}`. `t0` is random-in-clip (seeded per sample with
  the decoders' `_seed_counter` contract, so `set_epoch` resumes it) or given per sample
  through `t0_key` and the loader's new `sample_data`. The last `end_margin_frames` of a
  clip are never requested; CUDA decodes retry once on the CPU. Persistent decoder thread
  pool, `num_ffmpeg_threads=1`, `seek_mode="exact"`, optional decoder-side `resize`, and
  inner `transforms` applied to the flat frames with `seed_repeat = T`. The stage is
  asynchronous (`submit()` / `collect()`): the loader's prefetch thread submits a batch's
  decodes as soon as its bytes are loaded and the main thread collects them, so
  `batches_ahead * batch_size` decodes are in flight; workers write straight into the
  `[B, T, 3, H, W]` output. `device` may be a list of CUDA devices (threads pinned per
  device, results gathered on `output_device`).
- `SlipstreamLoader`: async stage protocol. If the primary field's pipeline starts with an
  object exposing `submit`/`collect`, decoding is pipelined across `batches_ahead` batches
  (and `set_batches_ahead` is called so the stage can size its buffers).
- Video stage tuning, measured on a 32-core/64-thread host at T=120, 15 Hz, resize 224:
  default `num_workers = os.cpu_count()` (64 workers 124 windows/s vs 48 workers 111),
  one ffmpeg thread per decoder, frames kept in native HWC memory with a CHW view (a
  per-window transpose was 15x a memcpy), output buffers recycled through a ring
  (`reuse_output=True`), and a one-time warning when torch's intra-op threads would
  oversubscribe the cores (`torch.set_num_threads(1)` per process: +19 % in one process, 16x
  across N processes that each host a stage).
- `SlipstreamLoader`: an exception in the prefetch thread (or in an async stage's `submit`)
  is re-raised on the main thread instead of hanging the iterator.
- `warmup_cache(touch=True)`: after the `read()` pass, the primary field's records are also
  faulted through the loader's own mmap (one load per page) so they are mapped into this
  process; milliseconds when cached. Measured on a CIFS mount (`cache=strict`): the client
  drops cached pages when the last holder closes the file, so every process starts cold
  and the first pass costs about half the warm rate; later epochs in the same process are
  warm regardless. Stage stores on node-local disk for training. `loader.page_cache_residency()`
  reports how much of an epoch's bytes are in the page cache; on a network mount cold reads
  serialize (2.4 windows/s cold vs 90+ warm), so `warmup_cache()` first.
- `RandomResizedCropBatch`: per-image random resized crop on decoded tensors
  (uint8 or float), `seed_repeat`-aware, replayable.
- `SlipstreamLoader(sample_data={name: array})`: per-sample side arrays aligned with
  `indices`, shuffled and sharded together with their sample; each batch carries
  `batch[name]` and the primary field's pipeline receives them as
  `batch_data['sample_data']`. Lets a record index be repeated with different parameters.
- The primary field's pipeline dict now also carries `indices` and `field` (the raw
  `{data, sizes, heights, widths}` returned without a pipeline is unchanged).
  `set_epoch` resets the seed counter of every decoder or stage reachable from the pipelines.

### Fixed

- `SlipstreamLoader.shutdown()` / `__del__` no longer raise when `__init__` failed before
  the prefetch worker state existed.

## [0.6.0] - 2026-09-12

### Added

- `SlipstreamLoader.warmup_cache(indices=None)`: indices-aware page-cache
  warmup. Defaults to the loader's own `indices`; for variable-size fields
  (image / `bytes` / `str` `.bin` files) only the byte ranges of the selected
  records are read, sorted by offset and coalesced into sequential runs.
  Fixed-size `.npy` fields and metadata tables are read whole. With no subset
  the previous full-file behaviour is unchanged. The stats dict gains
  `subset`, `num_records`, `num_ranges`.
- Raw `bytes` fields (video containers, `np.save` blobs, …) are bank-eligible:
  `image_field` may name one, and it is auto-selected as primary when the cache
  has no image field. A `bytes` primary gets the slot-rotated zero-copy
  prefetch banks and never enters JPEG/YUV420 format handling (any manifest
  `image_format`, e.g. `"mp4"`, is passed through untouched).

### Fixed

- Non-primary bytes-backed fields (secondary image fields and raw `bytes`
  fields) are now returned as an owned `{data, sizes}` dict (plus
  `heights`/`widths` for image types) with `data` trimmed to the batch's
  largest record. Previously the loader handed out a view into the storage's
  single shared scratch buffer, which the prefetch worker overwrote while the
  main thread was still consuming the batch, and dropped the per-sample sizes
  for `bytes` fields entirely.
- `image_format="yuv420"` on a cache whose primary field is not an image no
  longer attempts to build a sibling YUV420 store.

## [0.5.0] - 2026-09-11

- BREAKING: `slipstream sync` / `slipstream datasets` removed; slipstream CLI is
  plumbing-only. Lab-dataset status/sync moved to `visionlab-datasets`.
