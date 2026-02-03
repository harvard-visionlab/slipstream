# PROGRESS.md - Implementation History

This file tracks completed work and historical implementation details. For current project guidance, see [CLAUDE.md](CLAUDE.md).

---

## Completed Phases

### Phase 1: Core Infrastructure ✅

1. ✅ Set up pyproject.toml with hatchling
2. ✅ Create basic package structure
3. ✅ Set up dev environment (uv, jupyterlab, nbstripout)
4. ✅ Port StreamingDatasetVisionlab → SlipstreamDataset
    - Intuitive API: `remote_dir`, `cache_dir`, `local_dir`
    - Automatic field type detection
    - Pipeline support for per-field transforms
    - `decode_images` and `to_pil` options
    - `SLIPSTREAM_CACHE_DIR` env var support
    - Falls back to LitData's default caching (`~/.lightning/`)
    - Cluster symlink setup (`ensure_lightning_symlink_on_cluster`)
5. ✅ Create `01_dataset_basics.ipynb` tutorial notebook

### Phase 2: Decoder Infrastructure ✅

1. ✅ Create libslipstream C++ extension
    - TurboJPEG decode with thread-local handles
    - stb_image_resize2 for crop + resize (no OpenCV needed)
    - Linux and macOS build support
2. ✅ Port OptimizedCache (V2 metadata format)
3. ✅ Port NumbaBatchDecoder (Numba prange + libslipstream)
    - `decode_batch_to_buffer()` - 17,366 samples/sec
    - `decode_batch_center_crop()` - 15,749 samples/sec (98.5% of FFCV)
    - `decode_batch_random_crop()` - 13,851 samples/sec (104.5% of FFCV)
4. ✅ Port CPUDecoder (TurboJPEG + ThreadPoolExecutor fallback)
5. ✅ Port GPUDecoder (nvImageCodec) — optional, CPU path is faster
6. ✅ Create decode benchmarks

### Phase 3: Loader Integration ✅

1. ✅ SlipstreamLoader with NumbaBatchDecoder pipelines
2. ✅ Multi-crop SSL support (list-of-pipelines per field)
3. ✅ Pipeline stages: DecodeOnly, CenterCrop, RandomResizedCrop, ResizeCrop
4. ✅ Seed support for reproducible multi-crop
5. ✅ Deterministic seeded shuffle (`seed=`, epoch-varying via `seed + epoch`)
6. ✅ Distributed training support (`distributed=True`, strided partitioning, `set_epoch()`)
7. ✅ Subset filtering (`indices=` param, matches FFCV's `indices` parameter)
8. ✅ Field index utility for class-based subsetting
9. ✅ Visual verification of loader outputs
10. ✅ FFCVFileReader (.beton/.ffcv reader, no FFCV dependency)
    - Reader protocol: `cache_path`, `field_types`, `__len__`, `__getitem__`, `read_all_fields()`
    - Reads all FFCV field types: RGBImageField, IntField, FloatField, JSONField, BytesField
    - S3 download via s5cmd (fast, parallel) with fsspec fallback
    - Benchmarked: 14,914 samples/sec RRC (matches LitData path)
11. ✅ YUV420 image format support (`image_format="yuv420"`)
    - 2.04x RRC, 2.05x CenterCrop, 1.65x multi-crop vs JPEG loader

### Phase 4: Cache & Format Enhancements ✅

1. ✅ Dataset stats: `compute_normalization_stats()`
2. ✅ YUV output mode: `DecodeYUVFullRes`, `DecodeYUVPlanes`
3. ✅ Fast S3→local sync: `sync_s3_dataset()` via `presync_s3=True`

### Phase 5a: Port fastaugs ✅

1. ✅ Port fastaugs GPU batch augmentations (direct port from lrm-ssl)
2. ✅ Optimized for speed: cached affine grids, fused matrix ops
3. ✅ Bug fixes: YIQ Yiq2Rgb matrix, contrast offset, device handling
4. ✅ Implement DirectRandomResizedCrop (analytic, no rejection sampling)

### Phase 5b: Standard Pipelines ✅

1. ✅ Refactor: `pipelines.py` → `decoders/stages.py` + `pipelines/` package
2. ✅ Renamed fused decode+crop stages (CenterCrop → DecodeCenterCrop, etc.)
3. ✅ Decoder output format flexibility (`to_tensor`, `permute` flags)
4. ✅ GPU optimization: numpy HWC → cuda is 2.6x faster than tensor CHW → cuda

### Phase 5c: Multi-Crop API Enhancements ✅

1. ✅ Named field emission (`batch["global_0"]` instead of list)
2. ✅ Per-crop parameter control (size, scale, ratio, seed per crop)
3. ✅ Per-sample seeding (FFCV-SSL compatible)
4. ✅ Yoked crops (same center, different scale)

### Phase 5d: Pipeline Presets ✅

Ported pipeline presets matching lrm-ssl yaml configs:

1. ✅ `supervised_train` / `supervised_val` - standard ImageNet training/validation
2. ✅ `simclr` / `simclr_symmetric` - SimCLR two-view SSL (symmetric augmentations)
3. ✅ `simclr_standard` - SimCLR matching ssl_standard.yaml (asymmetric: view1 has blur, view2 has solarization)
4. ✅ `ipcl` - IPCL 5-crop SSL matching ssl_ipcl5_standard.yaml
5. ✅ `lejepa` - L-JEPA 2 global + 4 local matching ssl_global2_local4_ratio1.yaml
6. ✅ `multicrop` - Flexible global+local crops supporting all ssl_globalN_localM configs

---

## Benchmark Results Summary

All CPU decode targets met or exceeded. See [BENCHMARKS.md](BENCHMARKS.md) for full results.

- **Raw I/O**: 939k img/s (2.3x FFCV)
- **CPU RRC**: 13,851 img/s (104.5% of FFCV); YUV420: 27,955 img/s (2.04x JPEG)
- **H100 YUV420 RRC**: 44,987 img/s (2.69x JPEG)
- **GPU batch augmentations**: 7.6x–53.7x speedup over per-sample
- **GPU-optimal path**: numpy HWC → cuda: 11,680 samples/sec (2.6x faster than tensor CHW)

---

## Image Format Experiments (Completed)

See [BENCHMARKS.md](BENCHMARKS.md#image-format-experiments-completed) for full details.

**Summary**: JPEG XL and QOI eliminated (slower or marginal). Raw YUV420 is the optimal format: 1.68-2.83x JPEG throughput at 1.73x storage, zero new dependencies. Integrated as `image_format="yuv420"`.

---

## Reference Implementation Locations

These were the source files ported to slipstream:

### visionlab/datasets
```
/Users/gaa019/Documents/GitHub/visionlab/datasets/datasets/streaming_dataset.py
```

### litdata-mmap
```
/Users/gaa019/Documents/GitHub/litdata-mmap
```

| litdata-mmap File                       | slipstream Target            |
| --------------------------------------- | ---------------------------- |
| `src/litdata_mmap/ffcv_style_loader.py` | `slipstream/loader.py`       |
| `src/litdata_mmap/gpu_decoder.py`       | `slipstream/decoders/gpu.py` |
| `src/litdata_mmap/turbo_decoder.py`     | `slipstream/decoders/cpu.py` |
| `src/litdata_mmap/ffcv_file_dataset.py` | `slipstream/ffcv_reader.py`  |
| `src/litdata_mmap/optimized_dataset.py` | `slipstream/dataset.py`      |

### fastaugs
```
/Users/gaa019/Documents/GitHub/lrm-ssl/lrm_ssl/datasets/dataloaders/fastaugs
```

### Pipeline configs
```
/Users/gaa019/Documents/GitHub/lrm-ssl/lrm_ssl/datasets/pipelines/configs/
```
