# CLAUDE.md - Agent Instructions for slipstream

## Project Overview

**slipstream** is a high-performance data loading library for PyTorch vision workloads. It provides FFCV-like performance without the FFCV dependency hassle (ffcv and ffcv-ssl are no longer actively maintained), using modern dependencies and a more versatile architecture (e.g., converting a streaming dataset to an ffcv-like dataset on the fly).

**Namespace**: `visionlab.slipstream`

```python
from visionlab.slipstream import SlipstreamDataset, SlipstreamLoader
```

## Reference Implementation

The core `SlipstreamDataset` implementaton is based on the prototype work in our **visionlab.datasets** repo:

```
'/Users/gaa019/Documents/GitHub/visionlab/datasets'
```

The core `SlipstreamLoader` implementation is based on the prototype work in our **litdata-mmap** repo:

```
/Users/gaa019/Documents/GitHub/litdata-mmap
```

### Key Files to Port

| litdata-mmap File                       | slipstream Target                 | Purpose                                      |
| --------------------------------------- | --------------------------------- | -------------------------------------------- |
| `src/litdata_mmap/ffcv_style_loader.py` | `slipstream/slipstream_loader.py` | **SlipstreamLoader** - main high-perf loader |
| `src/litdata_mmap/gpu_decoder.py`       | `slipstream/decoders/gpu.py`      | nvImageCodec GPU decoder                     |
| `src/litdata_mmap/cpu_decoder.py`       | `slipstream/decoders/cpu.py`      | TurboJPEG CPU decoder                        |
| `src/litdata_mmap/optimized_dataset.py` | `slipstream/dataset.py`           | FFCVStyleDataset base                        |

### Performance Benchmarks (from litdata-mmap)

For this "slipstream" repo, we need to ensure that we reproduce these metrics/benchmark results, otherwise we've lost something in translation.

WE want to benchmark all steps of the pipleline, from Raw I/O only, to Raw I/O + decode (both CenterCrop and RandomResizedCrop). For RandomResizedCrop we want to implement both a torchvision-compatible version (current), and one with a slightly smarter sampling algorithm for choosing params (needs design + implementation + benchmarking)

| Metric           | FFCV        | SlipstreamLoader | Notes                    |
| ---------------- | ----------- | ---------------- | ------------------------ |
| Raw I/O          | ~350k img/s | **480k+ img/s**  | +37% faster              |
| GPU Decode-Only  | ~11k img/s  | ~10k img/s       | Equivalent               |
| GPU Decode + RRC | ~10k img/s  | **10.1k img/s**  | Equivalent               |
| Cold Start       | baseline    | **20% faster**   | Parallel chunk downloads |

### Decoder Recommendations

| Environment      | Decoder                 | Performance                              |
| ---------------- | ----------------------- | ---------------------------------------- |
| **GPU Training** | nvImageCodec            | 10.1k img/s (decode + RandomResizedCrop) |
| CPU-Only         | TurboJPEG               | 5.7k img/s (decode + RandomResizedCrop)  |
| Multi-GPU        | nvImageCodec per device | Scales linearly                          |

**Note**: nvImageCodec is optimal for GPU workflows because it performs decode + RandomResizedCrop in a single fused operation, avoiding CPU↔GPU memory transfers.

---

## StreamingDataset Integration

Include `StreamingDatasetVisionlab` from visionlab/datasets as the base streaming dataset, but rename as `SlipstreamDataset` and we'll make enhancements as needed (e.g., to support other dataset formats, e.g., torchvision.ImageFolder or huggingface datasets; basically any dataset that emits samples). It's possible the right way to do this is to have separate datasets (`SlipstreamLitDataset`, `SlipstreamImageFolder`, `SlipstreamHFDataset`, etc.) and then create a universal interface with `SlipstreamDataset`.

The prototype `FFCVStyleDataset` needs to be merged with `visionab.datasets.StreamingDataset` to form our new `SlipstreamDataset`, and even better if we can make this work with any dataset.

```
/Users/gaa019/Documents/GitHub/visionlab/datasets/datasets/streaming_dataset.py
```

This provides:

- LitData StreamingDataset wrapper with pipelines
- Automatic image decoding and field type detection
- Cluster symlink setup for shared credentials
- AWS S3 or S3-compatible (e.g., Wasabi) storage options

---

## Project Structure

```
slipstream/
├── CLAUDE.md                   # This file
├── pyproject.toml              # Project config
├── README.md                   # User documentation
├── slipstream/                 # Maps to visionlab.slipstream
│   ├── __init__.py
│   ├── slipstream_loader.py    # SlipstreamLoader (from ffcv_style_loader.py)
│   ├── dataset.py              # FFCVStyleDataset base
│   ├── streaming.py            # StreamingDatasetVisionlab
│   └── decoders/
│       ├── __init__.py
│       ├── gpu.py              # nvImageCodec decoder
│       └── cpu.py              # TurboJPEG decoder
├── tests/
│   ├── test_slipstream_loader.py
│   ├── test_decoders.py
│   └── test_streaming.py
└── benchmarks/
    └── benchmark_loader.py
```

---

## pyproject.toml Setup

Follow the pattern from litdata-mmap, adapted for visionlab namespace:

```toml
[project]
name = "visionlab-slipstream"
version = "0.1.0"
description = "High-performance data loading for PyTorch vision workloads"
readme = "README.md"
requires-python = ">=3.10,<3.11"

dependencies = [
    # Core (no version pinning like ffcv required)
    "numpy",
    "torch",
    "torchvision",
    "tqdm",
    # Numba for JIT-compiled batch loading
    "numba",
    # LitData (use main branch to match visionlab-datasets)
    "litdata@git+https://github.com/Lightning-AI/litdata.git@main",
    # PyTurboJPEG for fast JPEG decoding
    "PyTurboJPEG@git+https://github.com/lilohuang/PyTurboJPEG.git",
    # AWS/S3 support
    "boto3",
    "fsspec[s3]"
]

[dependency-groups]
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "pytest-xdist",
    "mypy>=1.0.0",
    "ruff>=0.1.0",
    "ipykernel",
    "pandas",
    "matplotlib",
    "seaborn",
]

# GPU acceleration dependencies (nvImageCodec + CV-CUDA)
# Install with: uv sync --group gpu
gpu = [
    "nvidia-nvimgcodec-cu12>=0.2.0",
    "cvcuda-cu12>=0.5.0",
]

# Route torch to CUDA (Linux) or CPU (others)
[tool.uv.sources]
torch = [
    { index = "pytorch-cu121", marker = "sys_platform == 'linux' and platform_machine == 'x86_64'" },
    { index = "pytorch-cpu",   marker = "sys_platform != 'linux' or platform_machine != 'x86_64'" },
]
torchvision = [
    { index = "pytorch-cu121", marker = "sys_platform == 'linux' and platform_machine == 'x86_64'" },
    { index = "pytorch-cpu",   marker = "sys_platform != 'linux' or platform_machine != 'x86_64'" },
]

[[tool.uv.index]]
name = "pytorch-cu121"
url  = "https://download.pytorch.org/whl/cu121"
explicit = true

[[tool.uv.index]]
name = "pytorch-cpu"
url  = "https://download.pytorch.org/whl/cpu"
explicit = true

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.metadata]
allow-direct-references = true

[tool.hatch.build.targets.wheel]
packages = ["slipstream"]

# Note: For visionlab.slipstream namespace, may need to adjust package structure
# Option 1: Use slipstream/ folder directly (simpler)
# Option 2: Use src/visionlab/slipstream/ with proper namespace packages

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short"

[tool.mypy]
python_version = "3.10"
strict = true
ignore_missing_imports = true

[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP"]
```

### Key Differences from litdata-mmap

- **No FFCV dependency** - We replace FFCV with our own implementation
- **No scipy pinning** - Not needed without ffcv
- **GPU decoder optional** - Install with `uv sync --group gpu`
- **visionlab namespace** - Imports as `visionlab.slipstream`

---

## Implementation Plan

### Phase 1: Core Infrastructure

1. Set up pyproject.toml with visionlab namespace
2. Create basic package structure
3. Port StreamingDatasetVisionlab from visionlab/datasets

### Phase 2: FastLoader

1. Port FFCVStyleDataset from litdata-mmap
2. Port PrefetchingDataLoader
3. Port decoders (GPU/CPU)
4. Create FastLoader API wrapper

### Phase 3: Testing & Benchmarks

1. Unit tests for all components
2. Benchmark suite comparing to FFCV
3. Integration tests with real datasets

### Phase 4: Documentation

1. README with usage examples
2. API documentation
3. Performance guide

---

## Development Workflow

### Code Quality

- **Type hints**: All functions must have type annotations
- **Tests**: Every feature needs tests
- **Run checks before committing**:
    ```bash
    uv run pytest tests/ -v
    uv run mypy slipstream/
    uv run ruff check slipstream/ tests/
    ```

### Commit Messages

Use conventional commits:

- `feat: description` for new features
- `fix: description` for bug fixes
- `test: description` for tests
- `docs: description` for documentation

---

## Key Technical Details

### FastLoader Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     FastLoader API                          │
├─────────────────────────────────────────────────────────────┤
│  PrefetchingDataLoader                                      │
│  ├── Pre-allocated memory banks (zero-copy)                 │
│  ├── Async I/O with ThreadPoolExecutor                      │
│  └── Overlapped fetch/decode                                │
├─────────────────────────────────────────────────────────────┤
│  FFCVStyleDataset                                           │
│  ├── Memory-mapped chunk files                              │
│  ├── Numba JIT batch loading (@njit parallel=True)          │
│  └── OS page cache for warm epochs                          │
├─────────────────────────────────────────────────────────────┤
│  Decoders                                                   │
│  ├── GPU: nvImageCodec (decode + RRC fused)                 │
│  └── CPU: TurboJPEG (fallback)                              │
└─────────────────────────────────────────────────────────────┘
```

### Why This Works

1. **Cold Start (Epoch 1)**: LitData downloads chunks in parallel
2. **Warm Epochs (2+)**: mmap + OS page cache = zero-copy reads
3. **GPU Decode**: nvImageCodec performs decode + RandomResizedCrop in one fused op
4. **No Python GIL**: Numba JIT with `nogil=True` for true parallelism

---

## Dependencies NOT Included

These are explicitly NOT dependencies (the whole point of slipstream):

- ❌ `ffcv` - Replaced by our implementation
- ❌ `opencv-python` (old version) - Not needed
- ❌ `numba` (old version) - Use latest
- ❌ `numpy` (old version) - Use latest

---

## Usage Example (Target API)

```python

from visionlab.slipstream import StreamingDataset, FastStreamingDataLoader

# TODO:
# modify streaing dataset to convert s3_dir and cache_dir to: input_dir=Dir(path=cache_dir, url=input_dir)
dataset = StreamingDataset(
    s3_dir="s3://...",
    cache_dir="..",
    decode_images=False,   # convenience can automatically decode and convert to_pil when working in notebook
    to_pil=False,
    sample_pipelines={     # for training, use optimized decoders
        "image": [RandomResizeCropDecoder], # or CenterCropDecoder, etc.
    }
)

# Simple usage
loader = FastStreamingDataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=num_workers > 0,
    batch_pipelines={
        "image": ... # batch level piplines using "fastaugs" library
    }
)

for batch in loader:
    images = batch['image']  # [B, C, H, W] GPU tensor
    labels = batch['label']  # [B] tensor
    # Training...

```
