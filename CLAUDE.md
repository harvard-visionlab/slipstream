# CLAUDE.md - Agent Instructions for slipstream

## Project Overview

**slipstream** is a high-performance data loading library for PyTorch vision workloads. It provides FFCV-like performance without the FFCV dependency hassle, using modern dependencies and a more versatile architecture.

**Namespace**: `visionlab.slipstream`

```python
from slipstream import SlipstreamDataset, SlipstreamLoader
```

For implementation history and completed work, see [PROGRESS.md](PROGRESS.md).

---

## Quick Reference

### Performance
- **Raw I/O**: 939k img/s (2.3x FFCV)
- **CPU RRC**: 13,851 img/s (104.5% of FFCV)
- **YUV420 RRC**: 27,955 img/s (2.04x JPEG)

### Build libslipstream
```bash
uv run python libslipstream/setup.py build_ext --inplace
```

Requires system libturbojpeg:
- Linux: `/usr/libjpeg-turbo/include` and `/usr/libjpeg-turbo/lib64`
- macOS: `brew install libjpeg-turbo`

### Development Commands
```bash
uv sync --group dev              # Install dependencies
uv sync --group dev --group gpu  # With GPU support (Linux)
uv run pytest tests/ -v          # Run tests
uv run jupyter lab               # Launch JupyterLab
```

---

## Project Structure

```
slipstream/
├── slipstream/
│   ├── __init__.py             # Package exports
│   ├── dataset.py              # SlipstreamDataset (composition, auto-detects source)
│   ├── cache.py                # OptimizedCache (slip cache format)
│   ├── loader.py               # SlipstreamLoader (multi-crop support)
│   ├── readers/                # Dataset format adapters (reader protocol)
│   │   ├── streaming.py        # StreamingReader (LitData wrapper)
│   │   ├── ffcv.py             # FFCVFileReader (.ffcv/.beton)
│   │   └── imagefolder.py      # SlipstreamImageFolder (torchvision-style)
│   ├── backends/               # Low-level dataset backends
│   │   ├── ffcv_file.py        # FFCVFileDataset
│   │   └── ffcv_style.py       # FFCVStyleDataset
│   ├── decoders/               # Decode stages
│   │   ├── numba_decoder.py    # NumbaBatchDecoder (primary)
│   │   ├── yuv420_decoder.py   # YUV420NumbaBatchDecoder
│   │   ├── crop.py             # Fused decode+crop stages
│   │   └── multicrop.py        # Multi-crop decode stages
│   ├── pipelines/              # Pipeline presets
│   │   ├── supervised.py       # supervised_train, supervised_val
│   │   ├── simclr.py           # simclr, simclr_symmetric, simclr_standard
│   │   ├── ipcl.py             # ipcl (5-crop SSL)
│   │   ├── lejepa.py           # lejepa (2 global + 4 local)
│   │   └── multicrop_preset.py # multicrop (flexible global+local)
│   ├── utils/
│   │   ├── cache_dir.py        # Unified cache directory utilities
│   │   └── image_header.py     # Fast JPEG/PNG dimension parsing
│   └── transforms/             # GPU batch augmentations (fastaugs port)
├── libslipstream/              # C++ extension (TurboJPEG + stb_image_resize2)
├── benchmarks/                 # Benchmark scripts
└── notebooks/                  # Tutorial notebooks
```

---

## Pipeline Presets

Factory functions returning pipeline dicts for `SlipstreamLoader(pipelines=...)`:

| Preset | Description | Source yaml |
|--------|-------------|-------------|
| `supervised_train(size=224)` | Standard training (RRC + augmentations) | - |
| `supervised_val(size=224)` | Standard validation (center crop) | - |
| `simclr()` / `simclr_symmetric()` | SimCLR two-view (symmetric) | - |
| `simclr_standard()` | SimCLR two-view (asymmetric) | ssl_standard.yaml |
| `ipcl(num_crops=5)` | IPCL 5-crop SSL | ssl_ipcl5_standard.yaml |
| `lejepa()` | L-JEPA 2 global + 4 local | ssl_global2_local4_ratio1.yaml |
| `multicrop(global_crops, local_crops)` | Flexible multi-crop | ssl_globalN_localM_ratio1.yaml |

### multicrop configurations
```python
multicrop(global_crops=2, local_crops=0)  # ssl_global2_local0
multicrop(global_crops=2, local_crops=4)  # ssl_global2_local4 (default)
multicrop(global_crops=2, local_crops=6)  # ssl_global2_local6
multicrop(global_crops=3, local_crops=4)  # ssl_global3_local4
multicrop(global_crops=6, local_crops=0)  # ssl_global6_local0
```

### Common parameters
All presets accept: `size`, `seed`, `device`, `dtype`, `normalize`

---

## Current Work

### Remaining Tasks

1. ✅ **YUV crop pipelines**: `DecodeYUVCenterCrop`, `DecodeYUVRandomResizedCrop`, `DecodeYUVResizeCrop` — crop+resize while keeping YUV colorspace
2. ✅ **HuggingFace support**: `hf://` URIs work via LitData integration
3. ✅ **ImageFolder reader**: `SlipstreamImageFolder` for torchvision-style directories + S3 tar archives
4. ✅ **FFCV reader fixes**: Data pointer bug, image end trimming, text field auto-decode (bytes→str)
5. ✅ **Composition refactor**: `SlipstreamDataset` wraps pluggable readers (StreamingReader, ImageFolder, FFCV)
6. ✅ **End-to-end verification tests**: Comprehensive verification test suite added. Includes: FFCV reader byte-for-byte verification against native ffcv-ssl (devcontainer required), cache round-trip tests (JPEG byte-identical, PNG→YUV420 ±2 tolerance), decode correctness tests (vs PIL ±5 tolerance, BT.601 YUV coefficients), and functional model accuracy tests (ResNet50 cross-format). Also added dimension validation in cache verify() to catch silent parse failures.
   - ✅ **Notebook cleanup**: Fixed visualization code in notebooks 03, 04, 06, 08 to handle numpy HWC output from decoders. Fixed interpolation mode mismatch in transform verification (slipstream uses bilinear, torchvision defaults to nearest).
7. ⬜ **Documentation**: README, API docs, performance guide
8. ✅ **Remove `transform` parameter**: Removed global `transform` in favor of `pipelines` for consistency
9. ✅ **Remote cache storage**: `remote_cache` parameter for S3-based cache sharing. Auto-discovers, downloads, and uploads caches using hash-based paths for consistency
10. ✅ **Unified cache directory**: All readers use `~/.slipstream/` as default cache base. Configurable via `SLIPSTREAM_CACHE_DIR` env var for easy cluster deployment (symlink to shared storage)

---

## Key Design Decisions

### Architecture
- **Composition pattern**: `SlipstreamDataset` wraps a pluggable `_reader` (StreamingReader, SlipstreamImageFolder, or FFCVFileReader). Source-agnostic processing (decode, transform, pipelines) lives in the Dataset.
- **Auto-detection**: `_create_reader()` dispatches: FFCV (.ffcv/.beton) → ImageFolder (tar/class dirs) → StreamingReader (default)
- **Two-layer loader**: PrefetchingDataLoader (raw I/O) + SlipstreamLoader (decode + transforms)
- **Primary decoder**: NumbaBatchDecoder (Numba prange + libslipstream C extension)
- **GPU decoder**: nvImageCodec (optional, CPU path is faster for most workloads)

### Decoders output flexibility
```python
# GPU-optimal path (2.6x faster):
DecodeRandomResizedCrop(size=224, to_tensor=False, permute=False),
ToTorchImage(device='cuda', dtype=torch.float32),
```

### Pipeline conventions (matching lrm-ssl yaml)
- **Seed offsets**: crop=1234, hflip=1111, jitter=2222, gray=3333, solar=4444, blur=5555
- **Global crops**: scale=(0.30, 1.0), size=224
- **Local crops**: scale=(0.05, 0.30), size=98 (0.4375 × global_size)
- **Symmetric augmentations**: All crops get identical transforms (solarization on all)

---

## Important Notes for Claude

### DO NOT run benchmarks directly
Benchmark scripts output progress bars that consume excessive tokens. Instead:
1. Prepare code changes
2. Ask the user to run benchmarks
3. User will paste the results back

### DO NOT add materialization ops to CPU benchmarks
Never add `.sum()`, `.item()`, etc. to benchmark loops. On CPU, operations are synchronous.
- **CPU**: `img.shape[0]` is sufficient
- **GPU**: Use `torch.cuda.synchronize()`, NOT `.sum()`

### DO NOT use synthetic data for testing
Always use real datasets for verification tests. Synthetic data tests can pass while real data fails due to edge cases in actual image files (corrupted headers, unusual dimensions, encoding variations, etc.).

**Real datasets for testing:**
- **FFCV**: `s3://visionlab-datasets/imagenet1k/pre-processed/s256-l512-jpgbytes-q100-ffcv/imagenet1k-s256-l512-jpg-q100-cs100-val-7ac6386e.ffcv`
- **LitData**: `s3://visionlab-datasets/imagenet1k/pre-processed/s256-l512-jpgbytes-q100-streaming/val/`
- **ImageFolder**: `s3://visionlab-datasets/imagenet1k-raw/val.tar.gz`

Tests requiring S3 access should be marked with `@pytest.mark.s3` so they can be skipped when credentials are unavailable.

---

## Usage Example

```python
from slipstream import SlipstreamDataset, SlipstreamLoader
from slipstream.pipelines import supervised_train, lejepa

# Supervised training (S3 streaming)
dataset = SlipstreamDataset(input_dir="s3://bucket/dataset/", decode_images=False)
loader = SlipstreamLoader(dataset, batch_size=256, pipelines=supervised_train(size=224))

# FFCV file (local or S3 - auto-detected by .ffcv/.beton extension)
dataset = SlipstreamDataset("s3://bucket/imagenet-val.ffcv")
dataset = SlipstreamDataset(local_dir="/path/to/imagenet-val.ffcv")

# ImageFolder (local or S3 tar archive - auto-detected)
dataset = SlipstreamDataset(local_dir="/path/to/imagenet/val")
dataset = SlipstreamDataset(remote_dir="s3://bucket/imagenet/val.tar.gz")

# HuggingFace dataset (hf:// URI)
dataset = SlipstreamDataset(input_dir="hf://datasets/cifar10/data", decode_images=True)
sample = dataset[0]  # {'img': PIL.Image, 'label': 0}

# SSL multi-crop (L-JEPA style)
loader = SlipstreamLoader(dataset, batch_size=256, pipelines=lejepa(seed=42, device='cuda'))
for batch in loader:
    global_views = [batch['global_0'], batch['global_1']]  # 224×224
    local_views = [batch[f'local_{i}'] for i in range(4)]  # 98×98

# Remote cache: auto-download from S3 if available, upload after build
loader = SlipstreamLoader(
    dataset,
    batch_size=256,
    remote_cache="s3://my-bucket/slipstream-caches/",  # Auto-discovers by hash
    pipelines=supervised_train(224),
)

# After adding indexes or stats, sync manually
from slipstream import write_index
write_index(loader.cache, fields=['label'])
loader.sync_remote_cache()  # Uploads new files to S3
```

---

## Commit Messages

Use conventional commits:
- `feat:` new features
- `fix:` bug fixes
- `refactor:` code restructuring
- `docs:` documentation
- `test:` tests
