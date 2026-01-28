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

| litdata-mmap File                       | slipstream Target                 | Purpose                                      |
| --------------------------------------- | --------------------------------- | -------------------------------------------- |
| `src/litdata_mmap/ffcv_style_loader.py` | `slipstream/loader.py`            | PrefetchingDataLoader + FFCVStyleDataset     |
| `src/litdata_mmap/gpu_decoder.py`       | `slipstream/decoders/gpu.py`      | nvImageCodec GPU decoder                     |
| `src/litdata_mmap/turbo_decoder.py`     | `slipstream/decoders/cpu.py`      | TurboJPEG CPU decoder                        |
| `src/litdata_mmap/ffcv_file_dataset.py` | `slipstream/ffcv_reader.py`       | Native .ffcv/.beton file reader              |
| `src/litdata_mmap/optimized_dataset.py` | `slipstream/dataset.py`           | High-level dataset wrapper                   |

### Benchmark Datasets

```python
FFCV_VAL_PATH = "s3://visionlab-datasets/imagenet1k/pre-processed/s256-l512-jpgbytes-q100-ffcv/imagenet1k-s256-l512-jpg-q100-cs100-val-7ac6386e.ffcv"
LITDATA_VAL_PATH = "s3://visionlab-datasets/imagenet1k/pre-processed/s256-l512-jpgbytes-q100-streaming/val/"
```

Benchmark environments: macOS laptop (CPU), GPU workstation, cluster

---

## Performance Targets

| Metric           | FFCV        | Slipstream         | Status |
| ---------------- | ----------- | ------------------ | ------ |
| Raw I/O          | ~413k img/s | **939k img/s**     | ✅ 2.3x faster |
| CPU Decode Only  | ~17k        | **17,366 img/s**   | ✅ Target met |
| CPU + CenterCrop | ~15,840     | **15,749 img/s**   | ✅ 99.4% of FFCV |
| CPU + RRC        | ~13,250     | **13,851 img/s**   | ✅ 104.5% of FFCV |
| GPU Decode Only  | -           | ~10k img/s         | ✅ (CPU path preferred) |
| Cold Start       | baseline    | -                  | ⬜ Not measured |

All CPU decode targets met or exceeded. No OpenCV dependency required — stb_image_resize2 matches OpenCV's cv::resize(INTER_AREA) performance for this workload.

---

## Benchmark Results (Measured 2026-01-28)

All benchmarks on **machina** (GPU workstation: AMD Threadripper PRO 3975WX, 64 cores, NVIDIA RTX A6000), ImageNet-1k val (50k samples), batch_size=256, num_workers=12.

### Raw I/O (no decode)

| System | Samples/sec | vs FFCV | Status |
|--------|-------------|---------|--------|
| **Slipstream OptimizedCache** | **938,674** | **2.27x** | ✅ Exceeds target |
| **Slipstream Loader (simple)** | **774,801** | **1.87x** | ✅ Exceeds target |
| Slipstream Loader (threaded) | 393,245 | 0.95x | Threading overhead |
| FFCV | 413,413 | baseline | Reference |
| litdata-mmap PageAligned | 365,553 | 0.88x | |
| litdata-mmap FFCVStyle | 355,827 | 0.86x | |

**Target was 480k+ img/s → Achieved 775k-939k img/s (1.6-2x target)**

### CPU Decode + Transforms (NumbaBatchDecoder)

| System | Decode Only | + CenterCrop | + RRC | Status |
|--------|-------------|--------------|-------|--------|
| **Slipstream NumbaBatchDecoder** | **17,366** | **15,749** | **13,851** | ✅ All targets met |
| FFCV Reference | - | 15,840 | 13,250 | Target |
| litdata-mmap CPU (Numba) | 17,425 | 1,308 | 2,039 | Reference |

**Analysis:**
- ✅ **Decode-only matches target**: 17,366 vs 17,425 (litdata-mmap) — essentially identical
- ✅ **CenterCrop at 99.4% of FFCV**: 15,749 vs 15,840
- ✅ **RRC exceeds FFCV by 4.5%**: 13,851 vs 13,250
- ✅ **7-10x faster than litdata-mmap's crop path**
- No OpenCV required — stb_image_resize2 is sufficient (resize is only 8-20% of per-image time)

### GPU Decode + Transforms (nvImageCodec)

| System | Decode Only | + CenterCrop | + RRC | Status |
|--------|-------------|--------------|-------|--------|
| **Slipstream GPUDecoder** | TBD | TBD | TBD | ⬜ Not started |
| FFCV Reference | - | - | - | No GPU decode |
| litdata-mmap nvImageCodec | 10,000 | 5,250 | 4,880 | Reference |

**Note:** GPU decode is slower than optimized CPU decode for this dataset size. CPU path is preferred.

---

## Performance Optimization History

The gap between Slipstream and FFCV for crop operations was closed by eliminating unnecessary buffer copies.

### Key Finding: The Bottleneck Was Memory Copies, Not Resize

Profiling revealed that JPEG decode is 80-92% of per-image C++ time, with resize only 8-20%. The performance gap was caused by unnecessary `.copy()` calls on large numpy buffers (37-200MB per batch) when returning results. FFCV returns views into pre-allocated buffers with zero copies.

Approaches tried and rejected:
- **TurboJPEG scaled decode** (`decode_crop_resize`): No benefit for 256-512px source images
- **JPEG-domain crop** (`tjTransform`): 30% slower due to per-image malloc/free in prange
- **OpenCV cv::resize(INTER_AREA)**: <1% difference vs stb_image_resize2

The fix: return `dest_buffer[:batch_size]` (view) instead of `dest_buffer[:batch_size].copy()`.

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
│   ├── cache.py                # ✅ OptimizedCache (V2 metadata format)
│   ├── loader.py               # ⬜ PrefetchingDataLoader + SlipstreamLoader
│   ├── ffcv_reader.py          # ⬜ Native .ffcv file reader
│   ├── decoders/
│   │   ├── __init__.py         # ✅ Decoder exports
│   │   ├── cpu.py              # ✅ CPUDecoder (TurboJPEG + ThreadPool)
│   │   ├── gpu.py              # ✅ GPUDecoder (nvImageCodec)
│   │   └── numba_decoder.py    # ✅ NumbaBatchDecoder (prange + libslipstream)
│   └── transforms/             # ⬜ fastaugs port (TODO: cleanup)
│       ├── __init__.py
│       ├── functional.py
│       └── transforms.py
├── tests/
│   ├── __init__.py             # ✅ Created
│   ├── test_loader.py          # ⬜ 3-epoch tests (cold + warm)
│   ├── test_decoders.py        # ⬜
│   └── test_dataset.py         # ⬜
├── benchmarks/
│   ├── __init__.py             # ✅ Created
│   ├── utils.py                # ✅ Benchmark utilities
│   ├── benchmark_decode.py     # ✅ Decode benchmarks
│   └── benchmark_loader.py     # ⬜ End-to-end loader benchmarks
└── notebooks/
    ├── 00_environment_test.ipynb  # ✅ Environment verification
    └── 01_dataset_basics.ipynb    # ✅ SlipstreamDataset tutorial
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

### Phase 3: Loader Integration

1. ⬜ Port PrefetchingDataLoader (raw I/O layer with prefetch)
2. ⬜ Create SlipstreamLoader (training-ready layer)
3. ⬜ Port FFCVFileDataset (.beton reader)

### Phase 4: Augmentations

1. ⬜ Port fastaugs (direct port)
2. ⬜ TODO markers for cleanup/standardization
3. ⬜ Implement DirectRandomResizedCrop (analytic, no rejection sampling)

### Phase 5: Testing & Benchmarks

1. ✅ Decode benchmarks (benchmark_decode.py)
2. ⬜ 3-epoch test framework
3. ⬜ End-to-end loader benchmarks
4. ⬜ Comparison vs FFCV baseline

### Phase 6: Documentation

1. ⬜ README with usage examples
2. ⬜ API documentation
3. ⬜ Performance guide

---

## Important Notes for Claude

### DO NOT run benchmarks directly
The benchmark scripts output progress bars that consume excessive tokens. Instead:
1. Prepare code changes
2. Ask the user to run benchmarks
3. User will paste the results back

Example: "Please run `uv run python benchmarks/benchmark_decode.py --numba-only --epochs 1 --warmup 1 --batch-size 256 --skip-streaming` and paste the results."

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
```

---

## Commit Messages

Use conventional commits:

- `feat: description` for new features
- `fix: description` for bug fixes
- `test: description` for tests
- `docs: description` for documentation
- `refactor: description` for refactoring
