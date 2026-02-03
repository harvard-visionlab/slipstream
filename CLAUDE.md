# CLAUDE.md - Agent Instructions for slipstream

## Project Overview

**slipstream** is a high-performance data loading library for PyTorch vision workloads. It provides FFCV-like performance without the FFCV dependency hassle (ffcv and ffcv-ssl are no longer actively maintained), using modern dependencies and a more versatile architecture (e.g., converting a streaming dataset to an ffcv-like dataset on the fly).

**Namespace**: `visionlab.slipstream`

```python
from slipstream import SlipstreamDataset, SlipstreamLoader
```

---

## Design Decisions (Confirmed)

These decisions were confirmed during project planning and should be followed:

### Build System & Environment

- **Build system**: `hatchling` (not setuptools) - simpler config, better namespace package support
- **Python version**: `>=3.10`, development on **3.11** (pinned in `.python-version`)
- **Package structure**: `slipstream/` folder directly (not `src/visionlab/slipstream/`)

### Architecture

- **Two-layer loader architecture**:
    1. `PrefetchingDataLoader` - Raw I/O layer (480k img/s target)
        - Pre-allocated memory banks (zero-copy)
        - Background thread with `nogil=True` Numba
        - Returns: `{data, sizes, heights, widths, indices}`
    2. `SlipstreamLoader` - Training-ready layer
        - Integrates PrefetchingDataLoader
        - Adds GPU/CPU decoders
        - Adds RandomResizedCrop, CenterCrop, normalization
        - Returns: decoded GPU tensors `[B, C, H, W]`
        - Deterministic seeded shuffle (`seed=` param, epoch-varying via `seed + epoch`)
        - Distributed training support (`distributed=True`, auto-detects rank/world_size)
        - `set_epoch(n)` for manual epoch control (standard PyTorch distributed pattern)
        - Subset filtering (`indices=` param, for debugging/few-shot/custom sampling)

- **Dataset sources**: Start with LitData variant only, expand later
    - Future: `SlipstreamDataset.from_litdata()`, `.from_imagefolder()`, `.from_huggingface()`

- **Sample handling**: Work with any sample dict; if sample is a tuple, ask user for field names to return a dict

### Decoders

- **Primary decoder**: NumbaBatchDecoder (Numba prange + libslipstream C extension + TurboJPEG + stb_image_resize2)
    - Matches or exceeds FFCV performance for all operations
    - System dependency: `libturbojpeg`
    - No OpenCV required — stb_image_resize2 is sufficient
- **GPU decoder**: nvImageCodec exists but is slower than CPU path for this dataset size (~10k vs ~17k). Keep as optional for future larger-resolution datasets where GPU decode may win.

### File Formats

- **V2 metadata format**: Use exact same format as litdata-mmap for compatibility
- **First-epoch generation**: Create optimized files during first epoch from any source dataset
- **Dataset versioning**: Content-addressable hash for 1-to-1 mapping to source dataset
    ```python
    # Hash of: source_path + dataset_version + slipstream_format_version
    dataset_id = hashlib.sha256(f"{source_path}:{dataset.version}:slipstream-v1").hexdigest()[:8]
    # Filename: imagenet1k-val-{dataset_id}.slipstream
    ```
- **FFCV .beton support**: Yes, read existing .ffcv files directly via `FFCVFileDataset`
    - Location: `/Users/gaa019/Documents/GitHub/litdata-mmap/src/litdata_mmap/ffcv_file_dataset.py`

### RandomResizedCrop

- **Standard**: Torchvision-compatible (10 attempts, rejection sampling)
- **Optimized**: `DirectRandomResizedCrop` (or `AnalyticRandomResizedCrop`) - no loop needed
    - Choose random ratio from range
    - Choose long edge length from valid range
    - Short edge determined by ratio
    - Sample top-left coordinates from valid range directly

### Cluster & Lab Infrastructure

- **Include** `ensure_lightning_symlink_on_cluster()` for SLURM environments
- **Default cache_dir**: Points to shared lab cache directory (lightning symlink)

### Testing Strategy

- **Do NOT port** litdata-mmap tests (too sprawling)
- **Fresh test approach**:
    - Always run 3 full epochs (1 cold, 2 warm) to verify mmap reduces I/O bottleneck
    - Report: batch shape, device, images/s per epoch
    - Separate tests for:
        - I/O only (raw data pass through)
        - Minimal decoding (decode + center-crop for uniform sizes)

### Batch Augmentations (fastaugs)

- **Location**: `/Users/gaa019/Documents/GitHub/lrm-ssl/lrm_ssl/datasets/dataloaders/fastaugs`
- **Approach**: Direct port first, mark TODOs for cleanup/standardization/bug fixes
- **Features**: 27 GPU-accelerated transforms, per-image randomization, parameter replay for SSL

### Pipelines (for reference)

- **Location**: `/Users/gaa019/Documents/GitHub/lrm-ssl/lrm_ssl/datasets/pipelines`
- **Pattern**: OmegaConf/Hydra configuration-driven, two-stage (decode + batch)

---

## Reference Implementations

### visionlab/datasets (StreamingDatasetVisionlab)

```
/Users/gaa019/Documents/GitHub/visionlab/datasets/datasets/streaming_dataset.py
```

Provides:

- LitData StreamingDataset wrapper with pipelines
- Automatic image decoding and field type detection
- Cluster symlink setup for shared credentials
- AWS S3 or S3-compatible (e.g., Wasabi) storage options

### litdata-mmap (Core Loader)

```
/Users/gaa019/Documents/GitHub/litdata-mmap
```

| litdata-mmap File                       | slipstream Target            | Purpose                                  |
| --------------------------------------- | ---------------------------- | ---------------------------------------- |
| `src/litdata_mmap/ffcv_style_loader.py` | `slipstream/loader.py`       | PrefetchingDataLoader + FFCVStyleDataset |
| `src/litdata_mmap/gpu_decoder.py`       | `slipstream/decoders/gpu.py` | nvImageCodec GPU decoder                 |
| `src/litdata_mmap/turbo_decoder.py`     | `slipstream/decoders/cpu.py` | TurboJPEG CPU decoder                    |
| `src/litdata_mmap/ffcv_file_dataset.py` | `slipstream/ffcv_reader.py`  | Native .ffcv/.beton file reader          |
| `src/litdata_mmap/optimized_dataset.py` | `slipstream/dataset.py`      | High-level dataset wrapper               |

### Benchmark Datasets

```python
FFCV_VAL_PATH = "s3://visionlab-datasets/imagenet1k/pre-processed/s256-l512-jpgbytes-q100-ffcv/imagenet1k-s256-l512-jpg-q100-cs100-val-7ac6386e.ffcv"
LITDATA_VAL_PATH = "s3://visionlab-datasets/imagenet1k/pre-processed/s256-l512-jpgbytes-q100-streaming/val/"
```

Benchmark environments: macOS laptop (CPU), GPU workstation, cluster

---

## Performance Summary

All CPU decode targets met or exceeded. See [BENCHMARKS.md](BENCHMARKS.md) for full results.

- **Raw I/O**: 939k img/s (2.3x FFCV)
- **CPU RRC**: 13,851 img/s (104.5% of FFCV); YUV420: 27,955 img/s (2.04x JPEG)
- **H100 YUV420 RRC**: 44,987 img/s (2.69x JPEG) — loader is not the training bottleneck
- **GPU batch augmentations**: 7.6x–53.7x speedup over per-sample

### Build libslipstream

```bash
# Linux (cluster)
uv run python libslipstream/setup.py build_ext --inplace

# macOS
uv run python libslipstream/setup.py build_ext --inplace
```

Requires system libturbojpeg:

- Linux: `/usr/libjpeg-turbo/include` and `/usr/libjpeg-turbo/lib64`
- macOS: `brew install libjpeg-turbo`

---

## Project Structure

```
slipstream/
├── .python-version             # Pinned to 3.11
├── CLAUDE.md                   # This file
├── pyproject.toml              # Project config (hatchling)
├── README.md                   # User documentation (TODO)
├── libslipstream/              # C++ extension for fast decode
│   ├── __init__.py             # ✅ Package marker
│   ├── setup.py                # ✅ Build script (setuptools)
│   ├── libslipstream.cpp       # ✅ TurboJPEG + stb_image_resize2
│   ├── stb_image_resize2.h     # ✅ Header-only resize library
│   └── _libslipstream*.so      # ✅ Compiled extension (platform-specific)
├── slipstream/
│   ├── __init__.py             # ✅ Package exports
│   ├── dataset.py              # ✅ SlipstreamDataset (LitData wrapper)
│   ├── cache.py                # ✅ OptimizedCache (slip cache format)
│   ├── loader.py               # ✅ SlipstreamLoader (multi-crop support)
│   ├── readers/                # ✅ Dataset format adapters
│   │   ├── __init__.py         # ✅ Reader exports
│   │   └── ffcv.py             # ✅ FFCVFileReader (.ffcv/.beton, S3 download)
│   ├── backends/               # ✅ Low-level dataset backends
│   │   ├── __init__.py         # ✅ Backend exports
│   │   ├── ffcv_file.py        # ✅ FFCVFileDataset (Numba JIT batch loading)
│   │   └── ffcv_style.py       # ✅ FFCVStyleDataset (mmap batch loading)
│   ├── decoders/
│   │   ├── __init__.py         # ✅ Decoder + stage exports
│   │   ├── base.py             # ✅ BatchTransform ABC (loader stage protocol)
│   │   ├── cpu.py              # ✅ CPUDecoder (TurboJPEG + ThreadPool)
│   │   ├── gpu.py              # ✅ GPUDecoder (nvImageCodec)
│   │   ├── numba_decoder.py    # ✅ NumbaBatchDecoder (prange + libslipstream)
│   │   ├── yuv420_decoder.py   # ✅ YUV420NumbaBatchDecoder (2x JPEG throughput)
│   │   ├── decode.py           # ✅ Pure decode: DecodeOnly, DecodeYUVFullRes, DecodeYUVPlanes
│   │   ├── crop.py             # ✅ Fused decode+crop: CenterCrop, RandomResizedCrop, DirectRRC, ResizeCrop
│   │   ├── multicrop.py        # ✅ Multi-crop: MultiCropRRC, MultiRRC, MultiCropPipeline
│   │   └── utils.py            # ✅ estimate_rejection_fallback_rate
│   ├── pipelines/              # ✅ Pipeline presets (sequences of transforms)
│   │   ├── __init__.py         # ✅ Re-exports decoders+transforms (backward compat) + all presets
│   │   ├── _common.py          # ✅ Shared seed helpers + offset constants
│   │   ├── supervised.py       # ✅ supervised_train, supervised_val, make_train/val_pipeline
│   │   ├── simclr.py           # ✅ SimCLR two-view SSL
│   │   ├── byol.py             # ✅ BYOL two-view SSL (asymmetric augmentation)
│   │   └── multicrop_preset.py # ✅ Multi-crop SSL (DINO, iBOT)
│   └── transforms/             # ✅ GPU batch augmentations (fastaugs port) + pipeline transforms
│       ├── __init__.py         # ✅ All transform exports + IMAGENET_MEAN/STD
│       ├── base.py             # ✅ BatchAugment, Compose, RandomApply, MultiSample
│       ├── normalization.py    # ✅ Normalize, NormalizeLGN, IMAGENET_MEAN/STD
│       ├── conversion.py       # ✅ ToTorchImage, ToDevice, ToFloat, ToFloatDiv, etc.
│       ├── functional.py       # ✅ Functional implementations
│       └── ...                 # ✅ geometric, effects, color_jitter, grayscale, etc.
├── tests/
│   ├── __init__.py             # ✅ Created
│   ├── test_loader.py          # ⬜ 3-epoch tests (cold + warm)
│   ├── test_decoders.py        # ⬜
│   └── test_dataset.py         # ⬜
├── benchmarks/
│   ├── __init__.py             # ✅ Created
│   ├── utils.py                # ✅ Benchmark utilities
│   ├── benchmark_decode.py     # ✅ Decode benchmarks
│   ├── benchmark_loader.py     # ✅ Loader benchmarks (raw, RRC, CenterCrop, multi-crop)
│   └── benchmark_ffcv_loader.py # ✅ FFCV reader → SlipstreamLoader benchmark
└── notebooks/
    ├── 00_environment_test.ipynb  # ✅ Environment verification
    ├── 01_dataset_basics.ipynb    # ✅ SlipstreamDataset tutorial
    ├── 02_field_indexes.ipynb     # ✅ Field indexes & class-based subsetting
    └── 03_visual_verification.ipynb # ✅ Visual verification of loader outputs
```

---

## Implementation Plan

### Phase 1: Core Infrastructure ✅ COMPLETE

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

### Phase 2: Decoder Infrastructure ✅ COMPLETE

1. ✅ Create libslipstream C++ extension
    - TurboJPEG decode with thread-local handles
    - stb_image_resize2 for crop + resize (no OpenCV needed)
    - Linux and macOS build support
2. ✅ Port OptimizedCache (V2 metadata format)
3. ✅ Port NumbaBatchDecoder (Numba prange + libslipstream)
    - `decode_batch_to_buffer()` - 17,366 samples/sec ✅
    - `decode_batch_center_crop()` - 15,749 samples/sec ✅ (98.5% of FFCV)
    - `decode_batch_random_crop()` - 13,851 samples/sec ✅ (104.5% of FFCV)
4. ✅ Port CPUDecoder (TurboJPEG + ThreadPoolExecutor fallback)
5. ✅ Port GPUDecoder (nvImageCodec) — optional, CPU path is faster
6. ✅ Create decode benchmarks

### Phase 3: Loader Integration ✅ COMPLETE

1. ✅ SlipstreamLoader with NumbaBatchDecoder pipelines
2. ✅ Multi-crop SSL support (list-of-pipelines per field)
3. ✅ Pipeline stages: DecodeOnly, CenterCrop, RandomResizedCrop, ResizeCrop
4. ✅ Seed support for reproducible multi-crop
5. ✅ Deterministic seeded shuffle (`seed=`, epoch-varying via `seed + epoch`)
6. ✅ Distributed training support (`distributed=True`, strided partitioning, `set_epoch()`)
   6b. ✅ Subset filtering (`indices=` param, matches FFCV's `indices` parameter)
   6c. ✅ Field index utility: `write_index(cache, fields=['label'])` builds `{field}_index.npy` in cache dir (unique value → sample indices mapping). Auto-discovered on `OptimizedCache.load()`, accessed via `cache.get_index('label')`. Works for any numeric or string field. Enables class-based subsetting — e.g. get indices for 10 ImageNet classes and pass to `SlipstreamLoader(ds, indices=...)` for Imagenette
7. ✅ Visual verification of loader outputs (view decoded images, multi-crop views, seed reproducibility)
8. ✅ FFCVFileReader (.beton/.ffcv reader, no FFCV dependency)
    - Reader protocol: `cache_path`, `field_types`, `__len__`, `__getitem__`, `read_all_fields()`
    - Reads all FFCV field types: RGBImageField, IntField, FloatField, JSONField, BytesField
    - Correct FFCV metadata parsing (compound dtype with `align=True`, matching ffcv source)
    - Alloc table sorted by `sample_id` (FFCV writes in page order, not sample order)
    - S3 download via s5cmd (fast, parallel) with fsspec fallback
    - `read_all_fields()` fast path for bulk slip cache building
    - `OptimizedCache.build()` extended with generic `read_all_fields()` hook
    - Benchmarked: 14,914 samples/sec RRC (matches LitData path)
9. ✅ YUV420 image format support (`image_format="yuv420"`)
    - `SlipstreamLoader(dataset, image_format="yuv420")` — opt-in, default remains JPEG
    - YUV420 cache built on demand from JPEG cache (one-time ~2 min for 50k images)
    - `build_yuv420_cache()` / `load_yuv420_cache()` in `slipstream/cache.py`
    - `YUV420NumbaBatchDecoder` with full API parity: `decode_batch`, `decode_batch_resize_crop`, `decode_batch_multi_crop`, `hwc_to_chw`
    - `set_image_format()` on all pipeline classes (auto-swaps decoder)
    - Benchmarked: 2.04x RRC, 2.05x CenterCrop, 1.65x multi-crop vs JPEG loader

### Phase 4: Cache & Format Enhancements

1. ✅ **Dataset stats**: `compute_normalization_stats()` computes per-channel RGB mean/std from a slip cache. JPEG path uses PyTurboJPEG (accurate DCT, matches PIL/torchvision exactly). YUV420 path uses YUV420NumbaBatchDecoder.
2. ✅ **YUV output mode**: `DecodeYUVFullRes` (nearest-neighbor U/V upsample → [H,W,3] YUV) and `DecodeYUVPlanes` (raw Y/U/V planes) pipelines. C kernels: `yuv420p_to_yuv_fullres`, `yuv420p_extract_planes`.
3. ✅ **Fast S3→local sync**: `sync_s3_dataset()` utility using s5cmd. Integrated into `SlipstreamLoader` via `presync_s3=True`.
4. ⬜ **YUV crop pipelines**: `CenterCropYUV`, `RandomResizedCropYUV` — crop+resize while keeping YUV colorspace (currently YUV output is full-res only, no batching of variable-size images).

### Phase 5: Augmentations & Pipelines ✅ BENCHMARKS COMPLETE

1. ✅ Decode benchmarks (benchmark_decode.py)
2. ✅ Multi-epoch benchmarks with warmup (benchmark_loader.py — covers cold/warm epoch testing)
3. ✅ End-to-end loader benchmarks (benchmark_loader.py)
4. ✅ FFCV reader loader benchmark (benchmark_ffcv_loader.py)
5. ✅ FFCV baseline comparison (all targets met or exceeded — see Benchmark Results above)

#### Phase 5a: Port fastaugs ✅ COMPLETE

1. ✅ Port fastaugs GPU batch augmentations (direct port from lrm-ssl)
2. ✅ Optimized for speed: cached affine grids, fused matrix ops, eliminated index_select/copy overhead
3. ✅ Bug fixes: YIQ Yiq2Rgb matrix (exact inverse), contrast offset (DALI-compatible), device handling
4. ✅ Implement DirectRandomResizedCrop (analytic, no rejection sampling)

#### Phase 5b: Standard Pipelines & Demos

1. ✅ Create standard pipeline presets (training, validation, SSL multi-crop) for demos and API validation
2. ✅ Refactor: `pipelines.py` → `decoders/stages.py` (fused decode+crop stages) + `pipelines/` package (preset configurations)
3. ✅ **Renamed** fused decode+crop stages to clarify they are decoders, not pure transforms:
    - `CenterCrop` → `DecodeCenterCrop`
    - `RandomResizedCrop` → `DecodeRandomResizedCrop`
    - `DirectRandomResizedCrop` → `DecodeDirectRandomResizedCrop`
    - `ResizeCrop` → `DecodeResizeCrop`
    - `MultiRandomResizedCrop` → `DecodeMultiRandomResizedCrop` (preferred API: per-crop params, named outputs)
    - `MultiCropRandomResizedCrop` → `DecodeUniformMultiRandomResizedCrop` (legacy: uniform params, list output)
    - These are fused JPEG-decode + crop/resize operations (backed by NumbaBatchDecoder), not standalone transforms
    - Old names kept as deprecated aliases for backward compatibility
4. ✅ Eliminated duplicate `Normalize`/`ToDevice`/`Compose` from decoders — pipeline presets now use `transforms.ToTorchImage` + `transforms.Normalize` exclusively. `BatchTransform` ABC kept in `decoders/base.py` as loader stage protocol.
5. ⬜ **TODO**: Resolve decoder output format for optimal GPU pipeline:
    - Currently decode stages return CHW **uint8 torch tensors** (`torch.from_numpy(chw)` after internal `hwc_to_chw`)
    - Fastest path would be: decode → **HWC uint8 numpy** → `ToTorchImage(device, dtype=float16)` (fused device+permute+cast+divide on GPU) → augmentations → `Normalize(mean, std)` last
    - Current path: decode → CHW uint8 torch → `ToTorchImage` (permutes BCHW→BCHW no-op since already CHW, but still does device+cast+divide)
    - Consider: will users ever want raw numpy arrays from decoders? (e.g., for non-PyTorch workflows, visualization)
    - `ToTorchImage` currently assumes HWC input — would need adjustment if decoders already return CHW
5. ⬜ Notebook demonstrating pipeline presets and common training workflows

#### Phase 5c: Multi-Crop API Enhancements ✅ CORE COMPLETE

1. ✅ **Named field emission**: `MultiRandomResizedCrop` emits named top-level batch keys (e.g., `batch["global_0"]`) instead of a list. `MultiCropPipeline` applies per-crop transform chains.
2. ✅ **Per-crop parameter control**: All params (size, scale, ratio, seed) accept per-crop settings via dict API. Enables:
    - Different sized crops (e.g., 2 global 224px + 4 local 96px)
    - Different scale ranges per crop (e.g., global `(0.4, 1.0)` + local `(0.05, 0.4)`)
    - Per-crop seeds for reproducibility
    - Single JPEG decode per image with `dest_stride_w` support in libslipstream for varied-size output buffers
3. ✅ Notebook demonstrating multi-crop configurations (`notebooks/08_multi_crop_named.ipynb`)
4. ✅ Per-sample seeding for crop param generation (FFCV-SSL compatible seed formula, per-sample RNG independence, `set_epoch()` resets decoder seed counters for checkpoint resume)
5. ✅ Yoked crops (same seed → same center point, different scale → zoomed-in / zoomed-out pair)

#### Phase 5d: Multi-Crop Yoked Crop Visualizations

1. Add a visualization notebook (notebooks/09\_...) to visualize the crop parameters by plotting the crop params (subplot x,y axis limitis reflect the full image resolution, blue_crop reflects the large crop, red crop reflects the yoked smaller crop, show for a few different parameter settings, including non-yoked)

### Phase 6: Additional Dataset Sources

1. ⬜ `SlipstreamDataset.from_imagefolder()` (torchvision ImageFolder)
2. ⬜ `SlipstreamDataset.from_huggingface()` (HuggingFace datasets)
3. Note: Both can also be wrapped via LitData StreamingDataset, so direct support may not be needed — evaluate whether the LitData path is sufficient or if native adapters offer meaningful benefits (e.g., skipping the streaming conversion step).

### Phase 7: End-to-End Correctness Verification

These tests verify that the slip cache format faithfully represents the source data, with zero errors across all samples. They require a **separate test environment** that installs both slipstream and the source format's native reader (e.g., ffcv-ssl for .beton files, litdata for streaming datasets).

1. ⬜ **FFCV → slip cache correctness**: Per-sample comparison of FFCVFileReader output vs ffcv-ssl's native `Reader`/`Loader`. For every sample in the dataset, verify:
    - Image bytes are identical (JPEG bitstream match)
    - Labels, indices match exactly
    - Path/metadata fields match
    - Image dimensions (height, width) match
    - Requires: separate repo/env with both `slipstream` and `ffcv-ssl` installed
    - Test dataset: `imagenet1k-s256-l512-jpg-q100-cs100-val-7ac6386e.ffcv` (50k samples)

2. ⬜ **LitData StreamingDataset → slip cache correctness**: Per-sample comparison of `SlipstreamDataset[i]` vs the slip cache built from it. For every sample:
    - Image bytes match (accounting for JPEG EOI marker trimming)
    - All non-image fields match exactly
    - Dimensions from V2 metadata match JPEG header parse
    - Test dataset: `s3://visionlab-datasets/imagenet1k/.../val/` (50k samples)

3. ⬜ **Round-trip decode verification**: After building slip cache from any source, verify that decoded images (via NumbaBatchDecoder) produce correct pixel values by comparing against PIL/TurboJPEG reference decode of the original source bytes.

4. ⬜ **YUV420 round-trip verification**: Verify pixel differences from chroma subsampling are within expected bounds (mean ~5, max <200) across all samples.

### Phase 8: Documentation

1. ⬜ README with usage examples
2. ⬜ API documentation
3. ⬜ Performance guide (benchmarking section with cold/warm results across machines)

### Appendix: Image Format Experiments (completed)

See [BENCHMARKS.md](BENCHMARKS.md#image-format-experiments-completed) for full details. Summary: JPEG XL and QOI eliminated (slower or marginal). Raw YUV420 is the optimal format: 1.68-2.83x JPEG throughput at 1.73x storage, zero new dependencies. Integrated as `image_format="yuv420"`.

---

## Important Notes for Claude

### DO NOT run benchmarks directly

The benchmark scripts output progress bars that consume excessive tokens. Instead:

1. Prepare code changes
2. Ask the user to run benchmarks
3. User will paste the results back

Example: "Please run `uv run python benchmarks/benchmark_decode.py --numba-only --epochs 1 --warmup 1 --batch-size 256 --skip-streaming` and paste the results."

### DO NOT add materialization ops to CPU benchmarks

**Never** add `.sum()`, `.item()`, `.numpy()`, `.tolist()`, or similar reduction/conversion
ops to benchmark loops for CPU pipelines. On CPU, all operations are synchronous — data is
fully materialized the moment the decode/transform function returns. Adding `.sum()` on a
`[256, 3, 224, 224]` uint8 tensor reads 36.75 MB per batch, which can cost as much as the
decode itself (we measured 41% of total time from `.sum()` alone).

- **CPU pipelines**: No materialization op needed. `img.shape[0]` is sufficient to confirm
  the tensor exists.
- **GPU pipelines**: Use `torch.cuda.synchronize()` for accurate timing, NOT `.sum()`.
  `.sum()` both synchronizes AND does expensive work — it conflates the two.
- **Benchmark rule**: Only measure the operations that would happen in real training.
  In training, the tensor goes into `model.forward()` — it doesn't get `.sum()`'d.

---

## Development Commands

```bash
# Install dependencies
uv sync --group dev

# Install with GPU support (Linux only)
uv sync --group dev --group gpu

# Run tests
uv run pytest tests/ -v

# Run type checking
uv run mypy slipstream/

# Run linting
uv run ruff check slipstream/ tests/

# Launch JupyterLab
uv run jupyter lab
```

---

## Key Technical Details

### Loader Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   SlipstreamLoader API                       │
│              (Training-ready interface)                      │
├─────────────────────────────────────────────────────────────┤
│  Decoders                                                   │
│  ├── CPU: NumbaBatchDecoder (17k decode, 14-16k +crop)       │
│  └── GPU: nvImageCodec (~10k, optional)                      │
├─────────────────────────────────────────────────────────────┤
│  PrefetchingDataLoader                                      │
│  ├── Pre-allocated memory banks (zero-copy)                 │
│  ├── Background thread prefetching                          │
│  └── Returns: {data, sizes, heights, widths, indices}       │
├─────────────────────────────────────────────────────────────┤
│  FFCVStyleDataset / FFCVFileDataset                         │
│  ├── Memory-mapped files (mmap)                             │
│  ├── Numba JIT batch loading (@njit nogil=True parallel=True)│
│  ├── V2 metadata: pre-stored JPEG dimensions                │
│  └── OS page cache for warm epochs (480k+ img/s)            │
└─────────────────────────────────────────────────────────────┘
```

### Why This Works

1. **Cold Start (Epoch 1)**: LitData downloads chunks in parallel
2. **Warm Epochs (2+)**: mmap + OS page cache = zero-copy reads
3. **CPU Decode**: NumbaBatchDecoder with prange + libslipstream C extension matches FFCV
4. **No Python GIL**: Numba JIT with `nogil=True` for true parallelism
5. **Zero-copy returns**: Pre-allocated buffers returned as views, no `.copy()` overhead
6. **V2 Metadata**: Pre-stored JPEG dimensions eliminate header parsing overhead

---

## Dependencies

### Required

- `numpy`, `torch`, `torchvision`, `tqdm`
- `numba` - JIT-compiled batch loading
- `litdata` - Streaming dataset infrastructure
- `PyTurboJPEG` - Fast JPEG decoding (requires system `libturbojpeg`)
- `boto3`, `fsspec[s3]` - S3 support

### Optional (GPU)

- `nvidia-nvimgcodec-cu12` - GPU JPEG decoding
- `cvcuda-cu12` - GPU image processing

### System Dependencies

```bash
# macOS
brew install libjpeg-turbo

# Ubuntu
apt-get install libturbojpeg0-dev
```

---

## Usage Example (Target API)

```python
from slipstream import SlipstreamDataset, SlipstreamLoader

# Create dataset (LitData-backed)
dataset = SlipstreamDataset(
    input_dir="s3://bucket/dataset/",
    cache_dir="/local/cache",
    decode_images=False,  # Raw bytes for loader
)

# Create high-performance loader
loader = SlipstreamLoader(
    dataset,
    batch_size=256,
    num_workers=8,
    device="cuda",  # or None for auto-detect
    crop_size=224,
    crop_mode="random",  # or "center" for validation
)

for batch in loader:
    images = batch['image']  # [B, C, H, W] GPU tensor
    labels = batch['label']  # [B] tensor
    # Training...

# For ~2x faster decode, use YUV420 format (builds cache on first use):
loader = SlipstreamLoader(
    dataset,
    batch_size=256,
    image_format="yuv420",  # 1.73x storage, ~2x decode throughput
    pipelines={'image': [RandomResizedCrop(224)]},
)
```

---

## Commit Messages

Use conventional commits:

- `feat: description` for new features
- `fix: description` for bug fixes
- `test: description` for tests
- `docs: description` for documentation
- `refactor: description` for refactoring
