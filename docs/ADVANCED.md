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
