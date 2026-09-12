# Changelog

All notable changes to slipstream are documented here. Versions follow
[Semantic Versioning](https://semver.org/); the version lives in
`slipstream/version.py`.

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
