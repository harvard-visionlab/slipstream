# Format Comparison Experiment

Can alternative image formats decode faster than TurboJPEG for training workloads?

## Results (2026-01-30, macOS ARM64)

ImageNet-1k val (50k images, s256-l512, JPEG q100). Full-dataset epochs, 1 warmup + 2 timed.

### Single-Image Decode (Python, single-threaded)

| Format | Latency | Throughput | vs JPEG | File Size vs JPEG |
|--------|---------|------------|---------|-------------------|
| JPEG (TurboJPEG) | 490 µs | 2,039/s | 1.00x | 1.00x (3.79 GB) |
| QOI | 401 µs | 2,492/s | **1.22x** | 2.06x (7.80 GB) |
| JXL lossless | 9,259 µs | 108/s | 0.05x | — |
| JXL lossy (VarDCT) | 1,323 µs | 756/s | 0.37x | — |
| JXL fast (d=2, e=1) | 1,178 µs | 849/s | 0.42x | — |
| Raw RGB | 14 µs | 72,123/s | **35x** | 3.89x (~14.7 GB) |

### Raw RGB Ceiling (Numba batch memcpy, no decode)

341,906 img/s — this is the absolute upper bound with zero decode.

### Conclusions

- **JPEG XL: eliminated.** Even lossy VarDCT mode with max decodingspeed is 2.5x *slower*
  than TurboJPEG. Tested lossless (Modular), lossy (VarDCT d=1.0), and fast lossy (d=2.0,
  effort=1, decodingspeed=4). All slower. JXL files removed from cache.

- **QOI: promising but modest.** 1.22x faster decode in Python. The `qoi` package wraps
  the C reference implementation, so the ratio likely holds at the C level. With a Numba
  prange + C integration (like our JPEG path), this could potentially reach 1.3-1.5x if
  QOI's simpler algorithm benefits more from SIMD. Worth revisiting if we need more decode
  headroom. Storage cost: 2x JPEG.

- **Raw RGB: ceiling confirmed.** 35x faster than JPEG decode (Python), 342k/s via Numba.
  Confirms decode is genuinely the bottleneck (~95% of per-image time). But 3.9x storage
  makes this impractical for most use cases.

- **TurboJPEG is near-optimal for 256px images.** At this resolution, JPEG decode is so
  fast (~490µs Python, ~58µs Numba/C) that no practical lossless format can compete. The
  only path to significantly faster loading is avoiding decode entirely (raw storage or
  pre-cropped caches).

## Quick Start

```bash
# Install experimental dependency
uv sync --group dev --group experiment

# Step 1: Convert cache to QOI and raw RGB
uv run python experiments/format_comparison/convert_formats.py

# Step 2: Benchmark decode throughput
uv run python experiments/format_comparison/benchmark_formats.py
```

## Files

- `convert_formats.py` — transcodes JPEG → QOI/raw, reports file sizes
- `benchmark_formats.py` — measures decode throughput (full-dataset epochs)
