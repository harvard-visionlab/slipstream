# slipstream

Fast, frictionless PyTorch dataloading for vision. [FFCV](https://github.com/libffcv/ffcv)-level performance, zero hassle.

> **Beta software** — API may change. Use at your own risk.

## Why Slipstream?

- **[FFCV](https://github.com/facebookresearch/FFCV-SSL) speeds** without FFCV's installation pain (no custom compilers, no CUDA build, no outdated dependencies - yet anyway)
- **Any source**: LitData, HuggingFace, ImageFolder, FFCV files — all work seamlessly
- **Local or remote source**: "s3://bucket/imagenet/train/" or "/mnt/data/imagenet/train/" both work
- **One-time cache build**, then blazing fast epochs via memory-mapped I/O
- **Fast transforms**: batch transforms on CUDA tensors with per-sample random params
- **Built-in SSL pipelines**: SimCLR, IPCL, L-JEPA, flexible multi-crop
- **Remote cache sharing**: Build once, share via S3 across your team

## Performance Comparable to FFCV (Random Resized Crop)

| Benchmark (warm cache)   | Device | FFCV          | Slipstream    | Speedup   |
| ------------------------ | ------ | ------------- | ------------- | --------- |
| Raw I/O (JPEG Bytes)     | CPU    | 413k imgs/s   | 939k imgs/s   | **2.3x**  |
| RRC 224px (JPEG Decoded) | CPU    | 13,250 imgs/s | 13,851 imgs/s | **1.05x** |

## Support for YUV420 format for greater speed (vs. jpeg format)

| Benchmark (warm cache)     | Device | Slipstream JPEG  | Slipstream YUV420 | Speedup  |
| -------------------------- | ------ | ---------------- | ----------------- | -------- |
| RRC 224px                  | H100   | 16,715 imgs/s    | 44,987 imgs/s     | **2.7x** |
| Multi-RRC (2 views, 224px) | H100   | 15,328 x2 imgs/s | 28,475 x2 imgs/s  | **1.9x** |

_Full benchmarks: [BENCHMARKS.md](BENCHMARKS.md)_

## Installation

```bash
uv add git+https://github.com/harvard-visionlab/slipstream

# Required: libturbojpeg
# apt install libturbojpeg  # Ubuntu/Debian

# Optional: S3 remote cache support
uv tool install s5cmd
```

## Quick Start

```python
from slipstream import SlipstreamDataset, SlipstreamLoader
from slipstream.pipelines import supervised_train

# Any source, local or remote: litdata, image folder, HuggingFace, FFCV
dataset = SlipstreamDataset("s3://bucket/imagenet/train/")

# One line: auto-cache + decode + augment
loader = SlipstreamLoader(
    dataset,
    batch_size=256,
    pipelines=supervised_train(224),
)

for batch in loader:
    images, labels = batch['image'], batch['label']
    # images: [B, 3, 224, 224] normalized GPU tensor
```

First epoch builds the cache (if not already present locally). Subsequent epochs run at full speed.

## More Examples

See **[Advanced Usage](docs/ADVANCED.md)** for:

- Remote cache sharing (S3)
- SSL multi-crop pipelines (SimCLR, L-JEPA, IPCL)
- YUV420 format (2x faster decode)
- Different dataset sources
- Cluster deployment

## Requirements

- Python 3.10+
- PyTorch 2.0+
- libturbojpeg (`brew install libjpeg-turbo` or system package)

## Development

```bash
git clone https://github.com/harvard-visionlab/slipstream
cd slipstream
uv sync --group dev

# Build C extension
uv run python libslipstream/setup.py build_ext --inplace

# Run tests
uv run pytest tests/ -v
```

## License

MIT
