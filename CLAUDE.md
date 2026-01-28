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

- **Device selection**: Optional `device` argument, auto-detect (CUDA if available) when `None`
- **TurboJPEG**: Strictly required for the Numba decoding path (system dependency: `libturbojpeg`)
- **GPU decoder**: nvImageCodec for fused decode + RandomResizedCrop

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

| Metric           | FFCV        | SlipstreamLoader | Notes                    |
| ---------------- | ----------- | ---------------- | ------------------------ |
| Raw I/O          | ~350k img/s | **480k+ img/s**  | +37% faster              |
| GPU Decode-Only  | ~11k img/s  | ~10k img/s       | Equivalent               |
| GPU Decode + RRC | ~10k img/s  | **10.1k img/s**  | Equivalent               |
| CPU Decode + RRC | -           | ~5.7k img/s      | TurboJPEG fallback       |
| Cold Start       | baseline    | **20% faster**   | Parallel chunk downloads |

---

## Benchmark Results (Measured)

All benchmarks on **machina** (GPU workstation), ImageNet-1k val (50k samples), batch_size=256.

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

### CPU Decode + Transforms

| System | Decode Only | + CenterCrop | + RRC | Status |
|--------|-------------|--------------|-------|--------|
| **Slipstream NumbaBatchDecoder** | TBD | TBD | TBD | 🔄 In progress |
| FFCV Reference | ~11k | ~11k | ~15.7k | Target |
| litdata-mmap NumbaCropDecoder | ~11k | ~11k | ~10k | Reference |

**Target: Match FFCV's ~15k samples/sec for decode + RRC**

### GPU Decode + Transforms

| System | Decode Only | + RRC | Status |
|--------|-------------|-------|--------|
| **Slipstream GPUDecoder** | TBD | TBD | ⬜ Not started |
| FFCV Reference | ~11k | ~10k | Target |
| litdata-mmap nvImageCodec | ~10k | ~10.1k | Reference |

---

## Project Structure

```
slipstream/
├── .python-version             # Pinned to 3.11
├── CLAUDE.md                   # This file
├── pyproject.toml              # Project config (hatchling)
├── README.md                   # User documentation (TODO)
├── slipstream/
│   ├── __init__.py             # ✅ Package exports
│   ├── dataset.py              # ✅ SlipstreamDataset (LitData wrapper)
│   ├── loader.py               # ⬜ PrefetchingDataLoader + SlipstreamLoader
│   ├── ffcv_reader.py          # ⬜ Native .ffcv file reader
│   ├── decoders/
│   │   ├── __init__.py         # ✅ Created
│   │   ├── gpu.py              # ⬜ nvImageCodec decoder
│   │   └── cpu.py              # ⬜ TurboJPEG decoder
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
│   └── benchmark_loader.py     # ⬜
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

### Phase 2: Loader Infrastructure

1. ⬜ Port FFCVStyleDataset (V2 metadata format)
2. ⬜ Port PrefetchingDataLoader (raw I/O layer)
3. ⬜ Port CPU decoder (TurboJPEG)
4. ⬜ Port GPU decoder (nvImageCodec)
5. ⬜ Create SlipstreamLoader (training-ready layer)
6. ⬜ Port FFCVFileDataset (.beton reader)

### Phase 3: Augmentations

1. ⬜ Port fastaugs (direct port)
2. ⬜ TODO markers for cleanup/standardization
3. ⬜ Implement DirectRandomResizedCrop

### Phase 4: Testing & Benchmarks

1. ⬜ 3-epoch test framework
2. ⬜ I/O-only benchmarks
3. ⬜ Decode + crop benchmarks
4. ⬜ Comparison vs FFCV baseline

### Phase 5: Documentation

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
│  ├── GPU: nvImageCodec (decode + RRC fused, 10.1k img/s)    │
│  └── CPU: TurboJPEG (5.7k img/s)                            │
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
3. **GPU Decode**: nvImageCodec performs decode + RandomResizedCrop in one fused op
4. **No Python GIL**: Numba JIT with `nogil=True` for true parallelism
5. **V2 Metadata**: Pre-stored JPEG dimensions eliminate header parsing overhead

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
