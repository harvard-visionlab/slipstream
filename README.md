# slipstream

Slipstream: Frictionless streaming and mmap-accelerated PyTorch dataloading

## Overview

Slipstream provides FFCV-like performance for PyTorch vision workloads without the FFCV dependency hassle. It combines:

- **LitData's streaming** for fast cold starts (parallel chunk downloads from S3)
- **Memory-mapped cache** for fast warm epochs (OS page cache, zero-copy reads)
- **Composable pipelines** for decode/crop/normalize transforms

## Installation

```bash
uv pip install -e .
```

## Quick Start

```python
from slipstream import SlipstreamDataset, SlipstreamLoader
from slipstream import RandomResizedCrop, Normalize

# Create dataset
dataset = SlipstreamDataset(
    remote_dir="s3://bucket/dataset/train/",
    decode_images=False,  # Let loader handle decoding
)

# Create high-performance loader with pipelines
loader = SlipstreamLoader(
    dataset,
    batch_size=256,
    pipelines={
        'image': [
            RandomResizedCrop(224, device='cuda'),
            Normalize(),
        ],
    },
)

for batch in loader:
    images = batch['image']  # [B, 3, 224, 224] normalized tensor
    labels = batch['label']  # [B] tensor
    # Training...
```

---

## Development Status

### Phase 1: Core Infrastructure ✅

- [x] `SlipstreamDataset` - Wrapper for LitData StreamingDataset
- [x] `OptimizedCache` - Memory-mapped cache with O(1) batch loading
- [x] Numba JIT batch loaders with `nogil=True` for true parallelism
- [x] Fast path for building cache directly from LitData chunks

### Phase 2: SlipstreamLoader 🚧 **IN PROGRESS**

- [x] Basic loader with async prefetching
- [x] Composable pipeline system (`RandomResizedCrop`, `CenterCrop`, `Normalize`, `Decoder`)
- [x] CPU decoder (TurboJPEG) and GPU decoder (nvImageCodec)
- [ ] **Performance issue: 18x overhead vs direct cache access** (see below)

### Phase 3: Testing & Benchmarks 🚧 **IN PROGRESS**

- [x] Benchmark scripts with machine info collection
- [ ] Match reference performance (480k+ samples/sec raw I/O)
- [ ] Multi-machine benchmark tracking

### Phase 4: Documentation

- [ ] API documentation
- [ ] Performance tuning guide

---

## Known Issues

### SlipstreamLoader Performance Gap

**Status**: Under investigation

Current benchmarks show a significant performance gap:

| Benchmark | Samples/sec | Notes |
|-----------|-------------|-------|
| OptimizedCache.load_batch (direct) | ~950k | Reference target |
| SlipstreamLoader (raw, no pipelines) | ~53k | **18x slower** |
| StreamingDataLoader (8 workers) | ~2.8k | Baseline |

**Root cause identified**: The loader has an unnecessary data copy:

```python
# Current (slow): JIT writes to storage buffer, then copy to loader buffer
batch_data = image_storage.load_batch(batch_indices)
dest[:actual_batch_size] = batch_data['data'][:actual_batch_size]  # EXTRA COPY

# litdata-mmap (fast): JIT writes directly to loader buffer
_load_batch_ffcv_style(batch_indices, metadata, data_region, dest, sizes)
```

**Fix planned**: Modify loader to call JIT functions directly with pre-allocated buffers, bypassing the intermediate copy.

---

## Performance Targets

Reference numbers from litdata-mmap (Linux server with NVMe):

| Metric | Target | Notes |
|--------|--------|-------|
| Raw I/O | 480k+ samples/sec | Memory-mapped with OS page cache |
| CPU Decode + RRC | ~5.7k samples/sec | TurboJPEG |
| GPU Decode + RRC | ~10-11k samples/sec | nvImageCodec |
| vs StreamingDataLoader | 50-100x faster | After warmup epoch |

---

## Benchmarks

Run benchmarks with machine info tracking:

```bash
# Individual benchmarks
uv run python benchmarks/benchmark_raw_io.py
uv run python benchmarks/benchmark_decode.py
uv run python benchmarks/benchmark_loader.py

# All benchmarks
uv run python benchmarks/run_all.py

# With custom cache directory (for testing different drives)
uv run python benchmarks/benchmark_raw_io.py --cache-dir /path/to/fast/nvme
```

Results are saved to `benchmarks/results/` as JSON files by hostname.

---

## Future Enhancements

### Faster S3 chunk downloads with s5cmd

Currently `OptimizedCache.build()` downloads LitData chunks via LitData's internal mechanisms, which can be slow (~2 min for 160 chunks). A potential optimization is to use [s5cmd](https://github.com/peak/s5cmd) for bulk parallel downloads before building the cache.

```bash
# s5cmd can download many files in parallel with high throughput
s5cmd cp "s3://bucket/dataset/chunks/*" /local/cache/chunks/

# Sync only downloads missing/changed files
s5cmd sync --size-only 's3://bucket/dataset/chunks/*' /local/cache/chunks/
```

Benefits:
- s5cmd is **32x faster than s3cmd** and **12x faster than aws-cli**
- `sync` only downloads missing files, making subsequent runs fast
- Chunks are fully local before processing, avoiding cache eviction issues

See: [s5cmd GitHub](https://github.com/peak/s5cmd) | [AWS Blog: Parallelizing S3 Workloads with s5cmd](https://aws.amazon.com/blogs/opensource/parallelizing-s3-workloads-s5cmd/)
