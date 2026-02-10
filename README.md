# slipstream

Fast, frictionless PyTorch dataloading for vision. FFCV-level performance, zero hassle.

> **Beta software** — API may change. Use at your own risk.

## Why Slipstream?

- **FFCV speeds** without FFCV's installation pain (no custom compilers, no CUDA build)
- **Any source**: S3, HuggingFace, ImageFolder, FFCV files — all work seamlessly
- **One-time cache build**, then blazing fast epochs via memory-mapped I/O
- **Built-in SSL pipelines**: SimCLR, IPCL, L-JEPA, flexible multi-crop
- **Remote cache sharing**: Build once, share via S3 across your team

## Performance

| Benchmark | Slipstream | vs FFCV |
|-----------|------------|---------|
| Raw I/O | 939k img/s | **2.3x faster** |
| CPU RRC (224px) | 13,851 img/s | **105%** |
| YUV420 RRC | 27,955 img/s | **2x JPEG** |

*Benchmarks on Linux server with NVMe storage. Your mileage may vary.*

## Installation

```bash
uv add git+https://github.com/harvard-visionlab/slipstream

# Required: libturbojpeg
brew install libjpeg-turbo  # macOS
# apt install libturbojpeg  # Ubuntu/Debian

# Optional: S3 remote cache support
uv tool install s5cmd
```

## Quick Start

```python
from slipstream import SlipstreamDataset, SlipstreamLoader
from slipstream.pipelines import supervised_train

# Any source: S3, local, HuggingFace, FFCV
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

First epoch builds the cache. Subsequent epochs run at full speed.

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
