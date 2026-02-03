"""SimCLR two-view SSL pipeline presets.

Two variants:
- simclr_symmetric(): Both views have identical augmentation chains (true SimCLR)
- simclr_standard(): Asymmetric views matching lrm-ssl's ssl_standard.yaml
"""

from __future__ import annotations

import torch

from slipstream.decoders.multicrop import DecodeMultiRandomResizedCrop, MultiCropPipeline
from slipstream.pipelines._common import _seed
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

# Seed offsets matching ssl_standard.yaml
# View 1 offsets
CROP_OFFSET_V1 = 1234
HFLIP_OFFSET_V1 = 1111
JITTER_OFFSET_V1 = 2222
GRAY_OFFSET_V1 = 3333
BLUR_OFFSET_V1 = 4444

# View 2 offsets
CROP_OFFSET_V2 = 5678
HFLIP_OFFSET_V2 = 5555
JITTER_OFFSET_V2 = 6666
GRAY_OFFSET_V2 = 7777
SOLAR_OFFSET_V2 = 8888
BLUR_OFFSET_V2 = 9999


def simclr_symmetric(
    size: int = 224,
    scale: tuple[float, float] = (0.08, 1.0),
    ratio: tuple[float, float] = (0.75, 1.333),
    seed: int | None = None,
    device: str | None = None,
    dtype: torch.dtype = torch.float16,
    normalize: bool = True,
) -> dict[str, list]:
    """SimCLR two-view SSL pipeline with symmetric augmentations.

    Both views use identical augmentation chains with independent random seeds.
    This matches the original SimCLR paper design.

    Pipeline per view:
        DecodeRRC → ToTorchImage → RandomHorizontalFlip →
        RandomColorJitterHSV → RandomGrayscale → RandomGaussianBlur → [Normalize]

    Args:
        size: Output crop size.
        scale: Crop area range relative to original. Default (0.08, 1.0) matches torchvision.
        ratio: Aspect ratio range. Default (0.75, 1.333) matches torchvision.
        seed: Base seed for reproducibility. None = non-reproducible.
        device: Device for GPU augmentations (None = CPU).
        dtype: Output tensor dtype.
        normalize: Whether to apply ImageNet normalization.

    Returns:
        Pipelines dict with DecodeMultiRandomResizedCrop + MultiCropPipeline.
        Output keys: 'view_0', 'view_1'.
    """
    dev = device or "cpu"

    # Build crop specs for 2 views with different seed offsets
    crop_specs = {
        'view_0': dict(size=size, scale=scale, ratio=ratio, seed=_seed(seed, CROP_OFFSET_V1, 0)),
        'view_1': dict(size=size, scale=scale, ratio=ratio, seed=_seed(seed, CROP_OFFSET_V2, 0)),
    }

    # Build per-view augmentation pipelines (identical transforms, independent seeds)
    def _make_view_pipeline(hflip_offset: int, jitter_offset: int, gray_offset: int, blur_offset: int) -> list:
        return [
            ToTorchImage(device=dev, dtype=dtype),
            RandomHorizontalFlip(p=0.5, seed=_seed(
                seed, hflip_offset, 0), device=dev),
            RandomColorJitterHSV(
                p=0.8, hue=0.1, saturation=0.4, value=0.4, contrast=0.4,
                seed=_seed(seed, jitter_offset, 0), device=dev,
            ),
            RandomGrayscale(p=0.2, seed=_seed(
                seed, gray_offset, 0), device=dev),
            RandomGaussianBlur(
                p=0.5, kernel_size=(7, 7), sigma_range=(0.1, 2.0),
                seed=_seed(seed, blur_offset, 0), device=dev,
            ),
            *([Normalize(IMAGENET_MEAN, IMAGENET_STD,
              device=dev, dtype=dtype)] if normalize else []),
        ]

    per_view_pipes = {
        'view_0': _make_view_pipeline(HFLIP_OFFSET_V1, JITTER_OFFSET_V1, GRAY_OFFSET_V1, BLUR_OFFSET_V1),
        'view_1': _make_view_pipeline(HFLIP_OFFSET_V2, JITTER_OFFSET_V2, GRAY_OFFSET_V2, BLUR_OFFSET_V2),
    }

    return {
        'image': [
            DecodeMultiRandomResizedCrop(crop_specs),
            MultiCropPipeline(per_view_pipes),
        ],
    }


def simclr_standard(
    size: int = 224,
    scale: tuple[float, float] = (0.08, 1.0),
    ratio: tuple[float, float] = (0.75, 1.333),
    seed: int | None = None,
    device: str | None = None,
    dtype: torch.dtype = torch.float16,
    normalize: bool = True,
) -> dict[str, list]:
    """SimCLR two-view SSL pipeline matching lrm-ssl's ssl_standard.yaml.

    Asymmetric views:
    - View 1: HFlip → ColorJitter → Grayscale → Normalize → GaussianBlur(p=1.0)
    - View 2: HFlip → ColorJitter → Grayscale → Solarization(p=0.2) → Normalize

    This matches the exact augmentation scheme from:
        lrm-ssl/lrm_ssl/datasets/pipelines/configs/ssl_standard.yaml

    Args:
        size: Output crop size.
        scale: Crop area range relative to original. Default (0.08, 1.0) matches yaml.
        ratio: Aspect ratio range. Default (0.75, 1.333) matches yaml.
        seed: Base seed for reproducibility. None = non-reproducible.
        device: Device for GPU augmentations (None = CPU).
        dtype: Output tensor dtype.
        normalize: Whether to apply ImageNet normalization.

    Returns:
        Pipelines dict with DecodeMultiRandomResizedCrop + MultiCropPipeline.
        Output keys: 'view_0', 'view_1'.
    """
    dev = device or "cpu"

    # Build crop specs matching yaml seed offsets
    crop_specs = {
        'view_0': dict(size=size, scale=scale, ratio=ratio, seed=_seed(seed, CROP_OFFSET_V1, 0)),
        'view_1': dict(size=size, scale=scale, ratio=ratio, seed=_seed(seed, CROP_OFFSET_V2, 0)),
    }

    # View 1: HFlip → ColorJitter → Grayscale → Normalize → GaussianBlur(p=1.0)
    view_0_pipeline = [
        ToTorchImage(device=dev, dtype=dtype),
        RandomHorizontalFlip(p=0.5, seed=_seed(
            seed, HFLIP_OFFSET_V1, 0), device=dev),
        RandomColorJitterHSV(
            p=0.8, hue=0.1, saturation=0.4, value=0.4, contrast=0.4,
            seed=_seed(seed, JITTER_OFFSET_V1, 0), device=dev,
        ),
        RandomGrayscale(p=0.2, seed=_seed(
            seed, GRAY_OFFSET_V1, 0), device=dev),
        *([Normalize(IMAGENET_MEAN, IMAGENET_STD, device=dev, dtype=dtype)]
          if normalize else []),
        RandomGaussianBlur(
            p=1.0, kernel_size=(7, 7), sigma_range=(0.1, 2.0),
            seed=_seed(seed, BLUR_OFFSET_V1, 0), device=dev,
        ),
    ]

    # View 2: HFlip → ColorJitter → Grayscale → Solarization(p=0.2) → Normalize
    view_1_pipeline = [
        ToTorchImage(device=dev, dtype=dtype),
        RandomHorizontalFlip(p=0.5, seed=_seed(
            seed, HFLIP_OFFSET_V2, 0), device=dev),
        RandomColorJitterHSV(
            p=0.8, hue=0.1, saturation=0.4, value=0.4, contrast=0.4,
            seed=_seed(seed, JITTER_OFFSET_V2, 0), device=dev,
        ),
        RandomGrayscale(p=0.2, seed=_seed(
            seed, GRAY_OFFSET_V2, 0), device=dev),
        RandomSolarization(p=0.2, threshold=0.5, seed=_seed(
            seed, SOLAR_OFFSET_V2, 0), device=dev),
        *([Normalize(IMAGENET_MEAN, IMAGENET_STD, device=dev, dtype=dtype)]
          if normalize else []),
    ]

    per_view_pipes = {
        'view_0': view_0_pipeline,
        'view_1': view_1_pipeline,
    }

    return {
        'image': [
            DecodeMultiRandomResizedCrop(crop_specs),
            MultiCropPipeline(per_view_pipes),
        ],
    }


# Default export - symmetric version (true SimCLR)
simclr = simclr_symmetric
