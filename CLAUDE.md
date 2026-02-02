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

## Performance Targets

| Metric           | FFCV        | Slipstream (JPEG) | Slipstream (YUV420) | Status                       |
| ---------------- | ----------- | ----------------- | ------------------- | ---------------------------- |
| Raw I/O          | ~413k img/s | **939k img/s**    | 640k img/s          | ✅ 2.3x faster (JPEG)        |
| CPU Decode Only  | ~17k        | **17,366 img/s**  | —                   | ✅ Target met                |
| CPU + CenterCrop | ~15,840     | **15,749 img/s**  | **32,864 img/s**    | ✅ YUV420: 2.05x JPEG        |
| CPU + RRC        | ~13,250     | **13,851 img/s**  | **27,955 img/s**    | ✅ YUV420: 2.04x JPEG        |
| CPU + 2x Multi   | —           | **11,482 img/s**  | **18,294 img/s**    | ✅ YUV420: 1.59x JPEG        |
| GPU Decode Only  | -           | ~10k img/s        | —                   | ✅ (CPU path preferred)      |

Above: M4 Pro laptop (14 cores). On FASRC H100 node (EPYC 9454, 23 threads), YUV420 peaks at **2.69x RRC (44,987 img/s), 2.83x CenterCrop (54,460 img/s)** — see H100 benchmark section below.

All CPU decode targets met or exceeded. YUV420 format provides **2-2.8x throughput** at 1.73x storage cost (advantage scales with available cores and how CPU-constrained JPEG decode is). No OpenCV dependency required — stb_image_resize2 matches OpenCV's cv::resize(INTER_AREA) performance for this workload.

---

## Benchmark Results (Measured 2026-01-28)

All benchmarks on **machina** (GPU workstation: AMD Threadripper PRO 3975WX, 64 cores, NVIDIA RTX A6000), ImageNet-1k val (50k samples), batch_size=256, num_workers=12.

### Raw I/O (no decode)

| System                         | Samples/sec | vs FFCV   | Status             |
| ------------------------------ | ----------- | --------- | ------------------ |
| **Slipstream OptimizedCache**  | **938,674** | **2.27x** | ✅ Exceeds target  |
| **Slipstream Loader (simple)** | **774,801** | **1.87x** | ✅ Exceeds target  |
| Slipstream Loader (threaded)   | 393,245     | 0.95x     | Threading overhead |
| FFCV                           | 413,413     | baseline  | Reference          |
| litdata-mmap PageAligned       | 365,553     | 0.88x     |                    |
| litdata-mmap FFCVStyle         | 355,827     | 0.86x     |                    |

**Target was 480k+ img/s → Achieved 775k-939k img/s (1.6-2x target)**

### CPU Decode + Transforms (NumbaBatchDecoder)

| System                           | Decode Only | + CenterCrop | + RRC      | Status             |
| -------------------------------- | ----------- | ------------ | ---------- | ------------------ |
| **Slipstream NumbaBatchDecoder** | **17,366**  | **15,749**   | **13,851** | ✅ All targets met |
| FFCV Reference                   | -           | 15,840       | 13,250     | Target             |
| litdata-mmap CPU (Numba)         | 17,425      | 1,308        | 2,039      | Reference          |

**Analysis:**

- ✅ **Decode-only matches target**: 17,366 vs 17,425 (litdata-mmap) — essentially identical
- ✅ **CenterCrop at 99.4% of FFCV**: 15,749 vs 15,840
- ✅ **RRC exceeds FFCV by 4.5%**: 13,851 vs 13,250
- ✅ **7-10x faster than litdata-mmap's crop path**
- No OpenCV required — stb_image_resize2 is sufficient (resize is only 8-20% of per-image time)

### SlipstreamLoader End-to-End (measured 2026-01-29)

| Pipeline                               | Samples/sec | vs Direct Decoder        | Status             |
| -------------------------------------- | ----------- | ------------------------ | ------------------ |
| **Raw I/O (simple)**                   | **958,270** | 102% of OptimizedCache   | ✅                 |
| **RRC (simple)**                       | **12,947**  | 93% of direct            | ✅                 |
| **RRC (threaded)**                     | **13,498**  | 97% of direct            | ✅                 |
| **CenterCrop (simple)**                | **15,449**  | 98% of direct            | ✅                 |
| **CenterCrop (threaded)**              | **14,768**  | 94% of direct            | ✅                 |
| **2x RRC fused multi-crop (simple)**   | **10,653**  | 48% faster than naive 2x | ✅                 |
| **2x RRC fused multi-crop (threaded)** | **10,453**  | 46% faster than naive 2x | ✅                 |
| Raw I/O (threaded)                     | 76,409      | —                        | Threading overhead |

**Analysis:**

- ✅ Loader overhead is <7% for decode+crop pipelines
- ✅ Fused multi-crop (decode-once, crop-N-times) gives ~48% speedup over naive 2x decode
- Simple mode is preferred — threading adds overhead without benefit for this workload
- Threaded raw I/O is slow due to `parallel=False` constraint (Numba workqueue not thread-safe for concurrent access)

### SlipstreamLoader End-to-End — YUV420, M4 Pro laptop (measured 2026-01-30)

| Pipeline                                        | Samples/sec | vs JPEG Loader | Status |
| ----------------------------------------------- | ----------- | -------------- | ------ |
| **Raw I/O (simple, yuv420)**                    | **639,785** | 0.73x          | ✅     |
| **RRC (simple, yuv420)**                        | **27,955**  | **2.04x**      | ✅     |
| **RRC (threaded, yuv420)**                      | **28,541**  | **1.95x**      | ✅     |
| **CenterCrop (simple, yuv420)**                 | **32,864**  | **2.05x**      | ✅     |
| **CenterCrop (threaded, yuv420)**               | **33,361**  | **2.12x**      | ✅     |
| **2x RRC fused multi-crop (simple, yuv420)**    | **18,294**  | **1.59x**      | ✅     |
| **2x RRC fused multi-crop (threaded, yuv420)**  | **18,418**  | **1.61x**      | ✅     |
| Raw I/O (threaded, yuv420)                      | 47,076      | —              | Threading overhead |

**Analysis:**

- ✅ **2x speedup** for all decode+crop pipelines vs JPEG
- ✅ Multi-crop speedup is lower (1.6x vs 2.0x) because crop+resize cost is fixed regardless of format — with YUV420's cheaper decode, the second crop is a larger relative cost
- ✅ Raw I/O is slower (640k vs 882k) because YUV420 images are 1.73x larger (more bytes to mmap-read per batch)
- ✅ No regressions in JPEG path — all JPEG benchmarks within normal run-to-run variance

### SlipstreamLoader End-to-End — Machina (measured 2026-01-31)

GPU workstation: AMD Threadripper PRO 3975WX, 64 cores, NVIDIA RTX A6000. `--num-threads 12`. ImageNet-1k val (50k samples), batch_size=256. Best of 3 epochs (warm).

| Pipeline                               | JPEG       | YUV420     | YUV/JPEG   |
| -------------------------------------- | ---------- | ---------- | ---------- |
| **Raw I/O (simple)**                   | 940,903    | 644,181    | 0.68x      |
| **RRC (simple)**                       | **12,920** | **19,173** | **1.48x**  |
| **RRC (threaded)**                     | **12,660** | **18,401** | **1.45x**  |
| **CenterCrop (simple)**               | **14,483** | **21,806** | **1.51x**  |
| **CenterCrop (threaded)**             | **14,151** | **21,790** | **1.54x**  |
| **2x RRC fused multi-crop (simple)**  | **10,186** | **13,660** | **1.34x**  |
| **2x RRC fused multi-crop (threaded)**| **10,016** | **13,634** | **1.36x**  |

**Analysis:**

- ✅ YUV420 speedup is 1.3-1.5x on machina — lower than M4 Pro (2.0x) and cluster (2.4-2.9x) because Threadripper's AVX2 makes JPEG decode relatively faster, narrowing the gap
- ✅ JPEG numbers match prior baselines (12,947 RRC from 2026-01-29) — no regression from stride fix
- ✅ Simple mode slightly preferred over threaded on this machine

### SlipstreamLoader End-to-End — FASRC Cluster (measured 2026-01-30)

Cluster node: Intel Xeon Platinum 8358 @ 2.60GHz, 16 allocated cores, 192 GB RAM, A100-SXM4-40GB MIG 3g.20gb. Partition: `gpu_test`. `--num-threads 12`. ImageNet-1k val (50k samples), batch_size=256.

These results are from `gpu_test` partition with 16 allocated cores. See H100 section below for production training hardware.

| Pipeline                               | JPEG       | YUV420     | YUV/JPEG   |
| -------------------------------------- | ---------- | ---------- | ---------- |
| **Raw I/O (simple)**                   | 1,139,398  | 639,071    | 0.56x      |
| **RRC (simple)**                       | **7,709**  | **18,544** | **2.41x**  |
| **RRC (threaded)**                     | **7,786**  | **19,551** | **2.51x**  |
| **CenterCrop (simple)**                | **8,422**  | **23,241** | **2.76x**  |
| **CenterCrop (threaded)**              | **8,541**  | **25,006** | **2.93x**  |
| **2x RRC fused multi-crop (simple)**   | **6,538**  | **12,696** | **1.94x**  |
| **2x RRC fused multi-crop (threaded)** | **6,606**  | **13,239** | **2.00x**  |

**Analysis:**

- ✅ **2.4-2.9x YUV420 speedup** — the strongest of all tested machines
- ✅ YUV420 advantage scales with how CPU-constrained JPEG decode is. With 16 cores on Xeon (no AVX-512 turbo for TurboJPEG), JPEG decode is the dominant bottleneck. YUV420 bypasses it entirely.
- ✅ Multi-crop shows **2.0x** speedup (vs 1.6x on M4 Pro) — JPEG decode was so dominant here that the second crop cost is relatively small
- ✅ YUV420 RRC at 18.5k img/s means the data loader is no longer the bottleneck for most single-GPU training workloads
- Threaded mode is slightly faster than simple on the cluster (opposite of M4 Pro) — likely due to NFS latency benefiting from async prefetch

### SlipstreamLoader End-to-End — FASRC H100 Node (measured 2026-01-31)

H100 node: AMD EPYC 9454 48-Core @ 2.75GHz, 24 allocated cores (1 GPU), 250 GB RAM, NVIDIA H100 80GB HBM3. ImageNet-1k val (50k samples), batch_size=256.

#### Default threads (23 = available - 1)

| Pipeline                               | JPEG          | YUV420         | YUV/JPEG   |
| -------------------------------------- | ------------- | -------------- | ---------- |
| **Raw I/O (simple)**                   | 1,930,867     | 1,201,282      | 0.62x      |
| **RRC (simple)**                       | **16,715**    | **44,987**     | **2.69x**  |
| **RRC (threaded)**                     | **13,865**    | **30,293**     | **2.18x**  |
| **CenterCrop (simple)**                | **19,254**    | **54,460**     | **2.83x**  |
| **CenterCrop (threaded)**              | **18,490**    | **31,846**     | **1.72x**  |
| **2x RRC fused multi-crop (simple)**   | **15,328**    | **28,475**     | **1.86x**  |
| **2x RRC fused multi-crop (threaded)** | **13,158**    | **33,332**     | **2.53x**  |

#### 12 threads (for cross-machine comparison)

| Pipeline                               | JPEG          | YUV420         | YUV/JPEG   |
| -------------------------------------- | ------------- | -------------- | ---------- |
| **Raw I/O (simple)**                   | 1,913,122     | 1,221,489      | 0.64x      |
| **RRC (simple)**                       | **9,887**     | **25,112**     | **2.54x**  |
| **RRC (threaded)**                     | **9,846**     | **24,723**     | **2.51x**  |
| **CenterCrop (simple)**                | **10,635**    | **30,702**     | **2.89x**  |
| **CenterCrop (threaded)**              | **9,831**     | **30,634**     | **3.12x**  |
| **2x RRC fused multi-crop (simple)**   | **8,555**     | **18,027**     | **2.11x**  |
| **2x RRC fused multi-crop (threaded)** | **8,505**     | **18,462**     | **2.17x**  |

**Analysis:**

- ✅ **2.7-2.8x YUV420 speedup at 23 threads** — best single-crop results of any machine
- ✅ **YUV420 CenterCrop at 54,460 img/s** — approaching raw I/O speed of threaded mode (68k)
- ✅ **YUV420 RRC at 44,987 img/s** — data loader is definitively not the training bottleneck
- ✅ JPEG scales well 12→23 threads (9.9k→16.7k RRC), but YUV420 scales even better (25k→45k) — 1.8x from 1.9x more threads
- ✅ Simple mode preferred for YUV420 — threading overhead hurts when decode is already fast
- ✅ At 23 threads with YUV420, a single CPU can feed an H100 at 45k img/s RRC — well beyond what the GPU training loop consumes

### GPU Decode + Transforms (nvImageCodec)

| System                    | Decode Only | + CenterCrop | + RRC | Status         |
| ------------------------- | ----------- | ------------ | ----- | -------------- |
| **Slipstream GPUDecoder** | TBD         | TBD          | TBD   | ⬜ Not started |
| FFCV Reference            | -           | -            | -     | No GPU decode  |
| litdata-mmap nvImageCodec | 10,000      | 5,250        | 4,880 | Reference      |

**Note:** GPU decode is slower than optimized CPU decode for this dataset size. CPU path is preferred.

### Batch Augmentation Transforms (measured 2026-02-01)

Ported from lrm-ssl fastaugs, optimized for slipstream v1. 17 GPU-accelerated batch transforms with per-image randomization and parameter replay for SSL.

#### CPU — M4 Pro laptop (batch_size=256, num_samples=10000)

| Transform              | Per-Sample (SS) | Per-Sample (TV) | SS vs TV | Per-Batch (SS) | Batch Speedup |
| ---------------------- | --------------- | --------------- | -------- | -------------- | ------------- |
| Normalize              | 11,981          | 11,838          | 1.01x    | 94,002         | 7.85x         |
| ToGrayscale            | 7,911           | 7,858           | 1.01x    | 80,787         | 10.21x        |
| RandomGrayscale        | 14,582          | 13,975          | 1.04x    | 34,474         | 2.36x         |
| RandomHorizontalFlip   | 36,964          | 38,109          | 0.97x    | 35,618         | 0.96x         |
| RandomRotate           | 1,951           | 1,344           | **1.45x** | 3,961         | 2.03x         |
| RandomZoom             | 2,010           | 1,382           | **1.45x** | 4,002         | 1.99x         |
| RandomBrightness       | 10,428          | 10,094          | 1.03x    | 40,177         | 3.85x         |
| RandomContrast         | 3,211           | 3,216           | 1.00x    | 16,113         | 5.02x         |
| RandomGaussianBlur     | 2,638           | 1,608           | **1.64x** | 3,698         | 1.40x         |
| RandomSolarization     | 10,878          | 10,832          | 1.00x    | 15,201         | 1.40x         |
| RandomColorJitter(HSV) | 297             | 252             | 1.18x    | 1,339          | 4.50x         |
| RandomColorJitterYIQ   | 5,623           | N/A             | N/A      | 10,092         | 1.79x         |
| RandomPatchShuffle     | 3,750           | N/A             | N/A      | 9,597          | 2.56x         |
| CircularMask           | 20,218          | N/A             | N/A      | 47,432         | 2.35x         |
| FixedOpticalDistortion | 3,436           | N/A             | N/A      | 16,388         | 4.77x         |
| RandomRotateObject     | 1,658           | N/A             | N/A      | 3,919          | 2.36x         |
| SRGBToLMS              | 1,768           | N/A             | N/A      | 2,400          | 1.36x         |

#### GPU — Machina (RTX A6000, batch_size=256, num_samples=10000)

| Transform              | Per-Sample (SS) | Per-Sample (TV) | SS vs TV | Per-Batch (SS) | Batch Speedup |
| ---------------------- | --------------- | --------------- | -------- | -------------- | ------------- |
| Normalize              | 39,375          | 13,404          | **2.94x** | 298,047       | 7.57x         |
| ToGrayscale            | 27,178          | 20,911          | **1.30x** | 442,677       | 16.29x        |
| RandomGrayscale        | 10,830          | 22,047          | 0.49x    | 253,319        | **23.39x**    |
| RandomHorizontalFlip   | 15,044          | 38,536          | 0.39x    | 339,994        | **22.60x**    |
| RandomRotate           | 1,942           | 3,230           | 0.60x    | 55,689         | **28.68x**    |
| RandomZoom             | 2,367           | 3,103           | 0.76x    | 58,529         | **24.73x**    |
| RandomBrightness       | 7,527           | 10,365          | 0.73x    | 275,207        | **36.56x**    |
| RandomContrast         | 4,771           | 6,398           | 0.75x    | 126,615        | **26.54x**    |
| RandomGaussianBlur     | 7,297           | 5,524           | **1.32x** | 126,081       | **17.28x**    |
| RandomSolarization     | 10,589          | 25,182          | 0.42x    | 159,949        | **15.11x**    |
| RandomColorJitter(HSV) | 1,233           | 1,303           | 0.95x    | 17,516         | **14.21x**    |
| RandomColorJitterYIQ   | 3,168           | N/A             | N/A      | 132,340        | **41.77x**    |
| RandomPatchShuffle     | 4,297           | N/A             | N/A      | 102,271        | **23.80x**    |
| CircularMask           | 43,850          | N/A             | N/A      | 594,394        | 13.56x        |
| FixedOpticalDistortion | 34,057          | N/A             | N/A      | 666,311        | 19.56x        |
| RandomRotateObject     | 969             | N/A             | N/A      | 52,067         | **53.72x**    |
| SRGBToLMS              | 5,650           | N/A             | N/A      | 53,773         | 9.52x         |

**Analysis:**

- ✅ **GPU batch speedups range from 7.6x to 53.7x** — validates the batch augmentation design
- ✅ **CPU per-sample matches or exceeds torchvision** for all transforms (0.97x–1.64x)
- ✅ **GPU per-sample trails torchvision** for some transforms (0.39x–0.76x) due to per-image randomization overhead that doesn't amortize for single images — this is expected and irrelevant since batch mode is the production path
- ✅ **RandomColorJitterYIQ at 132k batch on GPU** — single 3x3 matmul approach is 7.5x faster than HSV's 4-step sequential path (17.5k)
- ✅ **YIQ bugs fixed**: exact Yiq2Rgb inverse (was 0.3% roundtrip error), DALI-compatible contrast offset
- ✅ Key optimizations: cached affine grids + bmm (no F.affine_grid), eliminated index_select/index_copy for point-wise transforms, fused SRGBToLMS matrix, direct tensor indexing for PatchShuffle (no grid_sample)

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
│   │   ├── __init__.py         # ✅ Decoder exports
│   │   ├── cpu.py              # ✅ CPUDecoder (TurboJPEG + ThreadPool)
│   │   ├── gpu.py              # ✅ GPUDecoder (nvImageCodec)
│   │   ├── numba_decoder.py    # ✅ NumbaBatchDecoder (prange + libslipstream)
│   │   └── yuv420_decoder.py   # ✅ YUV420NumbaBatchDecoder (2x JPEG throughput)
│   └── transforms/             # ✅ fastaugs port (optimized v1)
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
4. ⬜ Implement DirectRandomResizedCrop (analytic, no rejection sampling)

#### Phase 5b: Standard Pipelines & Demos

1. ⬜ Create standard pipeline presets (training, validation, SSL multi-crop) for demos and API validation
2. ⬜ Notebook demonstrating pipeline presets and common training workflows

#### Phase 5c: Multi-Crop API Enhancements

1. ⬜ **Named field emission**: `MultiCropRandomResizedCrop` should optionally emit named fields (e.g., `image_0`, `image_1`) instead of a list `[crop0, crop1]` — less ambiguous downstream
2. ⬜ **Per-crop parameter control**: All params (size, scale, ratio, seed) should optionally accept a list of per-crop settings. Enables:
    - Different sized crops (e.g., 2 global 224px + 4 local 96px)
    - Yoked crops (same seed → same center point, different scale → zoomed-in / zoomed-out pair)
    - Different scale ranges per crop (e.g., global `(0.4, 1.0)` + local `(0.05, 0.4)`)
3. ⬜ Notebook demonstrating multi-crop configurations (named fields, yoked crops, mixed sizes)

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

✅ Alternative image storage formats: **investigated and benchmarked** (see `experiments/format_comparison/`).
- **JPEG XL: eliminated.** Tested lossless (Modular), lossy (VarDCT d=1.0), and fast lossy (d=2.0, effort=1, decodingspeed=4). All 2.5-19x *slower* than TurboJPEG. JXL is not competitive for this workload.
- **QOI: eliminated.** Python single-threaded showed 1.22x faster decode, but this was misleading (measured wrapper overhead, not raw decode). C implementation through Numba prange pipeline (tested 2026-01-30 on machina, `experiment/qoi-decode` branch, deleted): **only 1.05-1.07x vs JPEG**. TurboJPEG has SIMD (NEON/AVX2) that scalar QOI can't match, QOI decode is inherently sequential per-pixel, and with 12 prange threads both formats saturate memory write bandwidth. At 1.05x speedup with 2.06x storage cost, QOI is not viable.
- **Raw RGB: 35x faster** (no decode, just memcpy). Confirms decode is 95% of per-image time. 3.9x storage cost makes this impractical at ImageNet scale. Could be considered for tiny subsets (e.g., Imagenette/IN-100) for fast iteration, but likely not worthwhile given the modest decode bottleneck.
- **Conclusion (round 1):** Alternative *codecs* cannot beat SIMD-accelerated TurboJPEG. The bottleneck is Huffman decode + IDCT, not the codec algorithm's complexity. But raw RGB (72k img/s vs 15k JPEG on machina) proves the compute gap is real — the question is whether we can eliminate the JPEG bitstream parsing while keeping storage practical.

✅ Alternative storage formats — round 2: **bypass JPEG decode entirely** (tested 2026-01-30, `experiment/yuv420-decode` branch).
- **Insight**: JPEG's bottleneck is entropy decode (Huffman) + IDCT, not resize/crop. Raw RGB eliminates this but costs 3.9x storage. Round 2 targeted the middle ground.
- **Experiment A — Raw YUV420: WINNER.** Store decoded images as raw planar YUV 4:2:0 (Y: H×W, U: H/2×W/2, V: H/2×W/2). "Decode" is just a fixed-point BT.601 color conversion (no bitstream parsing). Benchmarked on machina (50k ImageNet val, batch_size=256, num_workers=12):

    | Mode             | JPEG       | YUV420     | YUV/JPEG |
    |------------------|------------|------------|----------|
    | Decode Only      | 14,322/s   | 27,370/s   | **1.91x** |
    | + CenterCrop(224)| 13,005/s   | 23,257/s   | **1.79x** |
    | + RRC(224)       | 12,398/s   | 20,828/s   | **1.68x** |

    Storage: 6.54 GB (1.73x JPEG). No new dependencies. C kernel: `yuv420p_to_rgb_buffer()` in libslipstream.cpp. Python: `YUV420NumbaBatchDecoder` in `slipstream/decoders/yuv420_decoder.py`. Converter: `experiments/format_comparison/convert_yuv420.py`.

- **Experiment B — LZ4+YUV420: eliminated.** Per-image `lz4.block.decompress()` in Python reintroduces the serial bottleneck. Storage improved to 5.63 GB (1.49x JPEG), but throughput collapsed to **1.06-1.10x JPEG** — nearly all YUV420 advantage lost. Would require C-level LZ4 decompression inside the prange loop to be viable, adding `liblz4` dependency for marginal storage savings (1.49x vs 1.73x JPEG). Not worth the complexity.

- **Conclusion (round 2):** Raw YUV420 is the optimal format for decode-bound workloads. 1.68-1.91x JPEG throughput at 1.73x storage with zero new dependencies. ✅ **Integrated** as `SlipstreamLoader(dataset, image_format="yuv420")` — see Phase 3 item 9.

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
