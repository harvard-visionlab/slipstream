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

## Am I set up? (`slipstream status`)

Installing slipstream adds a `slipstream` command (also `python -m slipstream`):

```bash
uv run slipstream status     # cache dir + permissions, S3 credentials/access, lab datasets
uv run slipstream datasets   # list the lab's registered datasets (needs visionlab-datasets)
uv run slipstream sync imagenet100 --fmt yuv420          # fetch train+val caches from S3
uv run slipstream sync imagenet1k --split val --fmt all  # both formats, val only
uv run slipstream sync s3://bucket/slipstream-cache/imagenet10/imagenet10-s256_l512-jpeg-val
```

`status` reports:

- **Cache directory**: the resolved path, where the setting came from (`SLIPSTREAM_CACHE_DIR`,
  the platform default from `visionlab-datasets`, or `~/.slipstream`), owner/mode, whether
  you can read and write it, and free disk space.
- **S3 access**: `s5cmd` availability, which AWS credentials were found, your IAM identity,
  and whether you can list the lab cache bucket.
- **Lab datasets**: for every registered `(dataset, split, fmt)` in `visionlab-datasets`,
  whether the cache is present and intact locally (with its exact path) and readable on S3.
  Missing entries come with the exact `slipstream sync ...` command to fetch them.

Exit code is non-zero when something would block training (unreadable cache dir, no S3
access), so it can be used in setup scripts. Add `--json` for machine-readable output and
`--no-remote` to skip network checks.

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
