"""IPCL multi-crop SSL pipeline preset."""

from __future__ import annotations

import torch

from slipstream.decoders.multicrop import DecodeMultiRandomResizedCrop, MultiCropPipeline
from slipstream.pipelines._common import CROP_OFFSET, _seed
from slipstream.transforms import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    Normalize,
    RandomColorJitterHSV,
    RandomGaussianBlur,
    RandomGrayscale,
    RandomHorizontalFlip,
    RandomSolarization,
    ToTorchImage,
)

# Seed offsets for each transform type (ensures independent randomness)
HFLIP_OFFSET = 1111
JITTER_OFFSET = 2222
GRAY_OFFSET = 3333
SOLAR_OFFSET = 4444
BLUR_OFFSET = 5555


def ipcl(
    num_crops: int = 5,
    size: int = 224,
    scale: tuple[float, float] = (0.20, 1.0),
    ratio: tuple[float, float] = (1.0, 1.0),
    seed: int | None = None,
    device: str | None = None,
    dtype: torch.dtype = torch.float16,
    normalize: bool = True,
) -> dict[str, list]:
    """IPCL multi-crop SSL pipeline.

    Multiple crops with identical augmentations.
    Uses DecodeMultiRandomResizedCrop for single-decode efficiency.

    Pipeline per crop:
        DecodeMultiRandomResizedCrop → ToTorchImage → RandomHorizontalFlip →
        RandomColorJitterHSV → RandomGrayscale → RandomSolarization →
        [Normalize] → RandomGaussianBlur

    Default crop params match ssl_ipcl5_standard.yaml:
        scale=(0.20, 1.0), ratio=(1.0, 1.0)

    Args:
        num_crops: Number of crops (default 5 for IPCL).
        size: Output crop size.
        scale: Crop area range relative to original.
        ratio: Aspect ratio range.
        seed: Base seed for reproducibility. None = non-reproducible.
        device: Device for GPU augmentations (None = CPU).
        dtype: Output tensor dtype.
        normalize: Whether to apply ImageNet normalization.

    Returns:
        Pipelines dict with DecodeMultiRandomResizedCrop + MultiCropPipeline.
        Output keys: 'crop_0', 'crop_1', ..., 'crop_{n-1}'.
    """
    dev = device or "cpu"

    # Build crop specs
    crop_specs = {}
    for i in range(num_crops):
        name = f'crop_{i}'
        crop_specs[name] = dict(
            size=size, scale=scale, ratio=ratio,
            seed=_seed(seed, CROP_OFFSET, i),
        )

    # Build per-crop augmentation pipelines (identical transforms, independent seeds)
    per_crop_pipes = {}
    for i in range(num_crops):
        name = f'crop_{i}'
        per_crop_pipes[name] = [
            ToTorchImage(device=dev, dtype=dtype),
            RandomHorizontalFlip(p=0.5, seed=_seed(seed, HFLIP_OFFSET, i), device=dev),
            RandomColorJitterHSV(
                p=0.8, hue=0.1, saturation=0.4, value=0.4, contrast=0.4,
                seed=_seed(seed, JITTER_OFFSET, i), device=dev,
            ),
            RandomGrayscale(p=0.2, seed=_seed(seed, GRAY_OFFSET, i), device=dev),
            RandomSolarization(p=0.2, threshold=0.5, seed=_seed(seed, SOLAR_OFFSET, i), device=dev),
            *([Normalize(IMAGENET_MEAN, IMAGENET_STD, device=dev, dtype=dtype)] if normalize else []),
            RandomGaussianBlur(
                p=0.2, kernel_size=(7, 7), sigma_range=(0.1, 2.0),
                seed=_seed(seed, BLUR_OFFSET, i), device=dev,
            ),
        ]

    return {
        'image': [
            DecodeMultiRandomResizedCrop(crop_specs),
            MultiCropPipeline(per_crop_pipes),
        ],
    }
