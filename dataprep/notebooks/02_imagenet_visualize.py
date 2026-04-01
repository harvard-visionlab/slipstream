# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # ImageNet-1K — Visualize Dataset and Loader
#
# Load pre-built ImageNet caches and visualize samples through the
# full SlipstreamLoader pipeline.

# %% [markdown]
# ## Configuration

# %%
import os
from pathlib import Path

# Cache locations (adjust if needed)
CACHE_BASE = Path(os.environ.get("SLIPSTREAM_CACHE_DIR", Path.home() / ".slipstream"))

JPEG_VAL = CACHE_BASE / "imagenet1k-s256_l512-jpeg-val"
JPEG_TRAIN = CACHE_BASE / "imagenet1k-s256_l512-jpeg-train"
YUV_VAL = CACHE_BASE / "imagenet1k-s256_l512-yuv420-val"
YUV_TRAIN = CACHE_BASE / "imagenet1k-s256_l512-yuv420-train"

for name, path in [("JPEG val", JPEG_VAL), ("JPEG train", JPEG_TRAIN),
                    ("YUV val", YUV_VAL), ("YUV train", YUV_TRAIN)]:
    status = "ok" if (path / "manifest.json").exists() else "not found"
    print(f"{name:12s}: {status}  ({path})")

# %% [markdown]
# ## Load a cache and inspect samples

# %%
from slipstream import SlipstreamDataset

dataset = SlipstreamDataset(local_dir=str(JPEG_VAL))
print(f"Samples: {len(dataset):,}")
print(f"Fields: {dataset.field_types}")

# %%
# View a single decoded sample
from PIL import Image
import io

sample = dataset[0]
img = Image.open(io.BytesIO(sample['image']))
print(f"Label: {sample['label']}, Path: {sample['path']}, Size: {img.size}")
img

# %% [markdown]
# ## Validation loader

# %%
from slipstream import SlipstreamLoader
from slipstream.decoders import DecodeCenterCrop
from slipstream import show_batch

loader = SlipstreamLoader(
    dataset,
    batch_size=16,
    pipelines={"image": [DecodeCenterCrop(224)]},
    exclude_fields=["path"],
)

batch = next(iter(loader))
print(f"Images: {batch['image'].shape}")
show_batch(batch['image'], batch['label'], n_cols=16)

# %% [markdown]
# ## Training loader (RandomResizedCrop)

# %%
from slipstream.decoders import DecodeRandomResizedCrop

train_loader = SlipstreamLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    pipelines={"image": [DecodeRandomResizedCrop(224)]},
    exclude_fields=["path"],
)

batch = next(iter(train_loader))
show_batch(batch['image'], batch['label'], n_cols=16)

# %%
loader.shutdown()
train_loader.shutdown()

# %% [markdown]
# ## Compare JPEG vs YUV420

# %%
if (YUV_VAL / "manifest.json").exists():
    yuv_dataset = SlipstreamDataset(local_dir=str(YUV_VAL))

    yuv_loader = SlipstreamLoader(
        yuv_dataset,
        batch_size=16,
        pipelines={"image": [DecodeCenterCrop(224)]},
        exclude_fields=["path"],
    )

    batch_yuv = next(iter(yuv_loader))
    print("YUV420 CenterCrop:")
    show_batch(batch_yuv['image'], batch_yuv['label'], n_cols=16)
    yuv_loader.shutdown()
else:
    print("YUV420 val cache not found — skip")
