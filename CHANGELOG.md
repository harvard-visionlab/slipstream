# Changelog

All notable changes to slipstream are documented here. Versions follow
[Semantic Versioning](https://semver.org/); the version lives in
`slipstream/version.py`.

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
  inner `transforms` applied to the flat frames with `seed_repeat = T`.
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
