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

### Sequence windows (video frame stores)

For a cache whose records are consecutive frames, `window=(T, stride)` turns the loader
into a window sampler. `indices` are the anchor records; each sample is the `T` records
`a, a+stride, …, a+(T-1)*stride`, and every field comes back as `[B, T, ...]`:

```python
loader = SlipstreamLoader(
    frame_store, batch_size=16, shuffle=True, seed=0,
    indices=anchors,                 # records with room for a whole window inside their clip
    window=(40, 1),                  # 40 frames per sample
    pipelines={"image": [DecodeRandomResizedCrop(224, seed=0, to_tensor=True, permute=True)]},
)
for batch in loader:
    batch["image"]     # [B, 40, 3, 224, 224]: one crop per window, shared by its 40 frames
    batch["pose"]      # [B, 40, 7]  from a "float32[7]" field
    batch["_indices"]  # [B, 40] record indices, batch["_anchors"] the [B] anchors
```

Shuffle, distributed sharding and `drop_last` operate on anchors. Every decoder and
augmentation draws its random parameters once per window, so a seeded run reproduces the
same anchor order and the same pixels. Fixed-shape per-record arrays are declared with the
field type `"<dtype>[d0,d1,...]"`, e.g. `"float32[7]"`, and stored as one mmap'd `.npy`.

## Am I set up? (`slipstream status`)

Installing slipstream adds a `slipstream` command (also `python -m slipstream`):

```bash
uv run slipstream status     # cache dir + permissions, s5cmd, AWS credentials/identity,
                             # S3 bucket read, slipstream caches found on disk
uv run slipstream config     # alias
```

`status` reports the resolved cache directory (`SLIPSTREAM_CACHE_DIR` or `~/.slipstream`),
its owner/mode and whether you can read and write it, free disk space, `s5cmd` availability,
which AWS credentials were found, your IAM identity (account id masked), whether you can list
the lab cache bucket, and every slipstream cache present in the cache dir with its size.
Exit code is non-zero when something would block training. Add `--json` for machine-readable
output, `--no-remote` to skip network checks, `--no-color` to disable colors.

**Lab datasets** (which registered datasets are present locally, and fetching them from S3)
live in the separate `visionlab-datasets` package, whose CLI sets the per-platform cache dir
and builds on the helpers in `slipstream.cli`:

```bash
uv add git+https://github.com/harvard-visionlab/datasets.git
uv run visionlab-datasets status
uv run visionlab-datasets sync imagenet100 --fmt yuv420
```

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
