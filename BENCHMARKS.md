# Benchmark Results & Performance History

## Performance Targets

| Metric           | FFCV        | Slipstream (JPEG) | Slipstream (YUV420) | Status                       |
| ---------------- | ----------- | ----------------- | ------------------- | ---------------------------- |
| Raw I/O          | ~413k img/s | **939k img/s**    | 640k img/s          | ✅ 2.3x faster (JPEG)        |
| CPU Decode Only  | ~17k        | **17,366 img/s**  | —                   | ✅ Target met                |
| CPU + CenterCrop | ~15,840     | **15,749 img/s**  | **32,864 img/s**    | ✅ YUV420: 2.05x JPEG        |
| CPU + RRC        | ~13,250     | **13,851 img/s**  | **27,955 img/s**    | ✅ YUV420: 2.04x JPEG        |
| CPU + 2x Multi   | —           | **11,482 img/s**  | **18,294 img/s**    | ✅ YUV420: 1.59x JPEG        |
| GPU Decode Only  | -           | ~10k img/s        | —                   | ✅ (CPU path preferred)      |

Above: M4 Pro laptop (14 cores). On FASRC H100 node (EPYC 9454, 23 threads), YUV420 peaks at **2.69x RRC (44,987 img/s), 2.83x CenterCrop (54,460 img/s)**.

All CPU decode targets met or exceeded. YUV420 format provides **2-2.8x throughput** at 1.73x storage cost.

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

---

## Image Format Experiments (completed)

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
