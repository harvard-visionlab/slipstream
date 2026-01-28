# Slipstream Benchmark Results

This file tracks benchmark results across different machines.

## Reference Targets (from litdata-mmap on Linux)

| Metric | Target | Notes |
|--------|--------|-------|
| Raw I/O | 480k+ samples/sec | Memory-mapped with OS page cache |
| CPU Decode + RRC | ~5.7k samples/sec | TurboJPEG |
| GPU Decode + RRC | ~10-11k samples/sec | nvImageCodec |
| vs StreamingDataLoader | 50-100x faster | After warmup epoch |

---

## How to Run Benchmarks

```bash
# Raw I/O benchmark
uv run python benchmarks/benchmark_raw_io.py

# Decode benchmark
uv run python benchmarks/benchmark_decode.py

# Full loader pipeline benchmark
uv run python benchmarks/benchmark_loader.py

# With custom options
uv run python benchmarks/benchmark_loader.py --batch-size 512 --device cuda
```

Results are automatically saved to `benchmarks/results/` as JSON files named by hostname.

---

## Results by Machine

### Template

To add results for a new machine, copy this template and fill in the values:

```markdown
### [hostname] - [description]

**Machine Info:**
- Platform: [OS and version]
- CPU: [model]
- Cores: [physical] physical, [logical] logical
- RAM: [X] GB
- GPU: [model or "None"]
- Drive: [SSD/NVMe/HDD]
- Date: [YYYY-MM-DD]

**Raw I/O:**
| Benchmark | Samples/sec |
|-----------|-------------|
| OptimizedCache (image) | X |
| SlipstreamLoader (raw) | X |
| StreamingDataLoader | X |
| **Speedup** | **Xx** |

**Decode:**
| Benchmark | Samples/sec |
|-----------|-------------|
| CPU Decode (no crop) | X |
| CPU Decode + RRC | X |
| CPU Decode + CenterCrop | X |
| GPU Decode + RRC | X |

**Full Pipeline:**
| Benchmark | Samples/sec |
|-----------|-------------|
| SlipstreamLoader (train, cpu) | X |
| SlipstreamLoader (val, cpu) | X |
| SlipstreamLoader (train, cuda) | X |
| SlipstreamLoader (val, cuda) | X |
```

---

<!-- Add machine results below this line -->

