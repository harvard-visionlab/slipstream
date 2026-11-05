# Advanced Usage

## Remote Cache (S3 Sharing)

Share caches across your team. First user builds and uploads, others download automatically.

```python
from slipstream import SlipstreamDataset, SlipstreamLoader
from slipstream.pipelines import supervised_train

dataset = SlipstreamDataset("s3://data-bucket/imagenet/train/")

# Auto-discovers cache by content hash
# Downloads if exists, builds and uploads if not
loader = SlipstreamLoader(
    dataset,
    batch_size=256,
    remote_cache="s3://lab-bucket/slipstream-caches/",
    pipelines=supervised_train(224),
)
```

After adding indexes or derived files:

```python
from slipstream import write_index

write_index(loader.cache, fields=['label'])
loader.sync_remote_cache()  # Uploads new files to S3
```

## SSL Multi-Crop Pipelines

Built-in presets for self-supervised learning:

```python
from slipstream import SlipstreamDataset, SlipstreamLoader
from slipstream.pipelines import simclr, ipcl, lejepa, multicrop

dataset = SlipstreamDataset("s3://bucket/imagenet/train/")

# SimCLR: 2 augmented views
loader = SlipstreamLoader(dataset, batch_size=256, pipelines=simclr())
for batch in loader:
    view1, view2 = batch['view_0'], batch['view_1']  # [B, 3, 224, 224]

# IPCL: 5 crops
loader = SlipstreamLoader(dataset, batch_size=256, pipelines=ipcl())
for batch in loader:
    crops = [batch[f'crop_{i}'] for i in range(5)]

# L-JEPA: 2 global (224px) + 4 local (98px)
loader = SlipstreamLoader(dataset, batch_size=256, pipelines=lejepa())
for batch in loader:
    global_views = [batch['global_0'], batch['global_1']]  # 224x224
    local_views = [batch[f'local_{i}'] for i in range(4)]  # 98x98

# Custom multi-crop configuration
loader = SlipstreamLoader(
    dataset,
    batch_size=256,
    pipelines=multicrop(global_crops=2, local_crops=6),
)
```

All presets accept common parameters:

```python
supervised_train(
    size=224,           # Output resolution
    seed=42,            # Reproducibility
    device='cuda',      # Output device
    dtype=torch.float32,
    normalize=True,     # ImageNet normalization
)
```

## Augmentation Parameters in the Batch (`CollectParams`)

Some objectives need to know what was done to each view: equivariant SSL needs the jitter factors,
flips and rotations, and VICRegL needs the crop boxes. Add `CollectParams` as an
`after_batch_transform`, giving it the same `pipelines` dict:

```python
from slipstream.decoders import DecodeMultiRandomResizedCrop, MultiCropPipeline
from slipstream.transforms import (CollectParams, ToTorchImage, RandomHorizontalFlip,
                                   RandomColorJitterHSV, RandomGaussianBlur)

def view(s):
    return [ToTorchImage("cuda"), RandomHorizontalFlip(seed=s), RandomColorJitterHSV(p=0.8, seed=s + 1),
            RandomGaussianBlur(p=0.5, seed=s + 2)]

pipelines = {"image": [
    DecodeMultiRandomResizedCrop({"g0": dict(size=224, seed=1), "g1": dict(size=224, seed=2)}),
    MultiCropPipeline({"g0": view(10), "g1": view(20)}),
]}
loader = SlipstreamLoader(dataset, batch_size=256, pipelines=pipelines,
                          after_batch_transforms=[CollectParams(pipelines)])
for batch in loader:
    p = batch["_params"]["image"]["g0"]
    p["DecodeMultiRandomResizedCrop"]["box"]     # [B, 4] int32 (x, y, w, h), source-image pixels
    p["RandomHorizontalFlip"]["applied"]         # [B] bool
    p["RandomColorJitterHSV"]["h"]               # [B]; 0 where not applied (s, v, c: 1)
    p["RandomGaussianBlur"]["sigma"]             # [B]; 0 where not applied
```

- **Layout:** `batch[key][field][view][TransformName]`.
- **Views** are the names given by `MultiCropPipeline`, `NamedCopies` or a multi-crop decoder. Each view lists
  every random step that produced it, including shared steps such as a single decoder before `NamedCopies`.
- **Values:** every value is per sample. `applied` replaces `do` / `idx`, and samples that weren't augmented get
  the identity value. Other keys (`deg`, `xs`, `mat`, …) come back as the transform stores them.
- **What's left out:** deterministic steps (`ToTorchImage`, `Normalize`, `RGBToLab`, …). With a window, params
  are `[B, T]`. A skipped `RandomApply` gives only `applied` (all False).
- **Devices:** decoder boxes are CPU tensors. Pass `device="cuda"` to move everything.
- **yuv420 caches** store even dimensions, so `heights` / `widths` can be 1 px larger than the original (the padding
  is at the right / bottom edge).
- **Without the loader:** decoders expose the same boxes through `last_params()`. This covers
  `DecodeRandomResizedCrop`, `DecodeMultiRandomResizedCrop` and the resize-short-crop-long decoders.
- **Cost:** opt-in, a few tiny tensor ops per random step. Loaders without it are unchanged.

### Lab colour space (CMC)

`RGBToLab` converts a float sRGB batch in [0, 1] to CIE L\*a\*b\* (D65) on its device. The constants
match `skimage.color.rgb2lab`. Normalise with the CMC constants and split in the model:

```python
from slipstream.transforms import RGBToLab, Normalize, LAB_CMC_MEAN, LAB_CMC_STD
view = [ToTorchImage("cuda"), ..., RGBToLab(), Normalize(LAB_CMC_MEAN, LAB_CMC_STD)]
# model side: L, ab = x[:, :1], x[:, 1:]
```

## Paired Augmentation (`group=`)

`SlipstreamLoader(..., group=G)` makes every `G` consecutive samples of a batch share one draw of every
random step: crop, flip, colour jitter, grayscale, blur, solarize, rotation and so on. Sampling is
unchanged (shuffled, distinct images), so a group is G different images given the same augmentation.
Use it for objectives that compare images under identical augmentations, such as ce_ssl (Yerxa et al. 2024):

```python
loader = SlipstreamLoader(dataset, batch_size=256, pipelines=pipelines, group=2)
for batch in loader:
    for view in ("g0", "g1"):
        x = batch[view]
        a, b = x[0::2], x[1::2]       # pair k = rows 2k, 2k+1: different images, same draw in this view
```

- **Layout:** rows stay interleaved, so `[B, ...]` keeps its shape. Views are still drawn independently of
  each other.
- **Crops:** decoders share the per-sample draw (fractional centre, area fraction, aspect ratio), not the
  pixel box, so every crop fits its own image. Pairs of the same size get the same box. Otherwise the boxes
  match in relative terms, except where one image needs a retry or fallback, or a centre clamped at the edge.
- **Batch size:** `batch_size` (per rank under DDP) must be a multiple of `group`. Pairs never straddle
  ranks or batches. With `drop_last=False`, a final partial batch of odd size has one unpaired sample.
- **Windows:** with `window=(T, stride)`, groups of windows share a draw.
- **Not supported:** stages that manage their own transforms (`DecodeVideoWindow`) raise.
- **Check:** `CollectParams` shows the pairing. Every param is equal within a pair.

## YUV420 Format (2x Faster)

YUV420 stores images in a format optimized for JPEG decode. ~2x faster than RGB JPEG with 1.7x storage.

```python
# Force YUV420 during cache build
loader = SlipstreamLoader(
    dataset,
    batch_size=256,
    image_format="yuv420",  # Default: "jpeg"
    pipelines=supervised_train(224),
)
```

Best for:
- Training where decode is the bottleneck
- Large-scale experiments where speed > storage

## Dataset Sources

Slipstream auto-detects the source type:

```python
from slipstream import SlipstreamDataset

# S3 streaming (LitData format)
dataset = SlipstreamDataset("s3://bucket/litdata-dataset/")

# Local ImageFolder (torchvision-style)
dataset = SlipstreamDataset(local_dir="/data/imagenet/train")

# S3 tar archive (auto-extracts)
dataset = SlipstreamDataset(remote_dir="s3://bucket/imagenet/train.tar.gz")

# FFCV file (local or S3)
dataset = SlipstreamDataset("s3://bucket/imagenet.ffcv")
dataset = SlipstreamDataset(local_dir="/data/imagenet.ffcv")

# HuggingFace datasets
dataset = SlipstreamDataset(input_dir="hf://datasets/cifar10")
```

## Cluster Deployment

### Shared Cache Directory

All users can share caches via a common directory:

```bash
# Option 1: Symlink (recommended)
ln -s /mnt/fast-storage/slipstream-cache ~/.slipstream

# Option 2: Environment variable
export SLIPSTREAM_CACHE_DIR=/mnt/fast-storage/slipstream-cache
```

### Combined with Remote Cache

For maximum efficiency:

1. Set shared local cache: `~/.slipstream` → fast NVMe
2. Set remote cache: `remote_cache="s3://lab-bucket/caches/"`
3. First user builds locally, uploads to S3
4. Other users download from S3 to shared local cache
5. Everyone benefits from cached data

```python
# All lab members use the same remote_cache URL
loader = SlipstreamLoader(
    dataset,
    remote_cache="s3://lab-bucket/slipstream-caches/",
    pipelines=supervised_train(224),
)
```

## Named Copies (Multi-View from Single Decode)

`NamedCopies` duplicates a single decoded batch into a named dict, letting you
apply different transforms per view via `MultiCropPipeline` — without decoding
more than once.

```python
from slipstream import (
    SlipstreamLoader, DecodeResizeCrop,
    NamedCopies, MultiCropPipeline, ToTorchImage, Normalize,
    IMAGENET_MEAN, IMAGENET_STD,
)
from slipstream.transforms import RandomZoom, RandomHorizontalFlip

# Two views with different zoom levels
loader = SlipstreamLoader(
    dataset,
    batch_size=256,
    pipelines={'image': [
        DecodeResizeCrop(resize_size=256, crop_size=224),
        NamedCopies(['view1', 'view2']),
        MultiCropPipeline({
            'view1': [
                ToTorchImage(device='cuda'),
                RandomZoom(p=1.0, zoom=(1.0, 1.0), x=0.5, y=0.5, device='cuda'),
                Normalize(IMAGENET_MEAN, IMAGENET_STD, device='cuda'),
            ],
            'view2': [
                ToTorchImage(device='cuda'),
                RandomZoom(p=1.0, zoom=(0.5, 0.5), x=0.5, y=0.5, device='cuda'),
                Normalize(IMAGENET_MEAN, IMAGENET_STD, device='cuda'),
            ],
        }),
    ]},
)

for batch in loader:
    v1 = batch['view1']  # [B, 3, 224, 224] — full zoom
    v2 = batch['view2']  # [B, 3, 224, 224] — 50% zoom
```

Works with any single-output decoder (`DecodeCenterCrop`, `DecodeResizeCrop`,
`DecodeRandomResizedCrop`) and any number of named copies.
Use `DecodeMultiRandomResizedCrop` instead when you need different
*random crops* per view (e.g., global/local crops for DINO).

## Resize-Short-Crop-Long Decoder

`DecodeRandomResizeShortCropLong` resizes the short edge and crops the long
edge — preserving aspect ratio while producing a square output. Useful when
you want less spatial distortion than `DecodeRandomResizedCrop`.

```python
from slipstream import SlipstreamLoader, DecodeRandomResizeShortCropLong

# Fixed 96px output, random crop position
loader = SlipstreamLoader(
    dataset,
    batch_size=256,
    pipelines={'image': [
        DecodeRandomResizeShortCropLong(
            size=96,
            x_range=(0, 1), y_range=(0, 1),  # random crop along long edge
            seed=42,
        ),
    ]},
)

# Variable-size output (multi-scale training)
loader = SlipstreamLoader(
    dataset,
    batch_size=256,
    pipelines={'image': [
        DecodeRandomResizeShortCropLong(
            size=(64, 160),       # sample size uniformly from [64, 160]
            size_mode="per_batch", # all images in batch share same size
        ),
    ]},
)
```

## RandomEmbed (Canvas Embedding)

Embed smaller images onto a larger canvas with configurable backgrounds and
optional circular fade. Pairs naturally with the resize-short-crop decoder
for scale-jitter experiments.

```python
from slipstream import RandomEmbed

# Basic: center-embed on black canvas
embed = RandomEmbed(canvas_size=224)

# Random position with power-law noise background
embed = RandomEmbed(
    canvas_size=224,
    x_range=(0, 1), y_range=(0, 1),
    background="power_law", alpha_range=1.5,
    seed=42,
)

# Circular fade so the image blends into the background
embed = RandomEmbed(
    canvas_size=224,
    background="power_law", alpha_range=1.5,
    fade_radius=(0.35, 0.50),  # (inner, outer) as fraction of min(H, W)
    seed=42,
)

# Constant fill background
embed = RandomEmbed(
    canvas_size=224,
    background="constant", fill=[0.485, 0.456, 0.406],
)
```

Backgrounds: `"zeros"` (literal zeros), `"constant"` (constant fill via `fill=`), `"power_law"` (1/f^α noise).
`color_noise=True` (default) generates independent noise per channel;
`color_noise=False` produces grayscale noise.

Supports SSL replay — `embed(view1)` then `embed.apply_last(view2)` places
both views at the same position.

## Custom Pipelines

Build your own decode/transform pipelines:

```python
from slipstream import SlipstreamLoader
from slipstream.decoders import DecodeRandomResizedCrop, DecodeCenterCrop
from slipstream.transforms import ToTorchImage, RandomHorizontalFlip, Normalize

# Training pipeline
train_pipelines = {
    'image': [
        DecodeRandomResizedCrop(224),
        ToTorchImage(device='cuda'),
        RandomHorizontalFlip(),
        Normalize(),
    ],
}

# Validation pipeline
val_pipelines = {
    'image': [
        DecodeCenterCrop(224),
        ToTorchImage(device='cuda'),
        Normalize(),
    ],
}

train_loader = SlipstreamLoader(train_dataset, batch_size=256, pipelines=train_pipelines)
val_loader = SlipstreamLoader(val_dataset, batch_size=256, pipelines=val_pipelines)
```

## S3-Compatible Storage

For Wasabi, MinIO, or other S3-compatible services:

```python
loader = SlipstreamLoader(
    dataset,
    remote_cache="s3://my-bucket/caches/",
    remote_cache_endpoint_url="https://s3.wasabisys.com",
    pipelines=supervised_train(224),
)
```
