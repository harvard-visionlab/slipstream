"""L-JEPA multi-crop SSL pipeline preset (2 global + 4 local crops).

Matches lrm-ssl's ssl_global2_local4_ratio1.yaml exactly.
"""

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

# Seed offsets matching ssl_global2_local4_ratio1.yaml
HFLIP_OFFSET = 1111
JITTER_OFFSET = 2222
GRAY_OFFSET = 3333
SOLAR_OFFSET = 4444
BLUR_OFFSET = 5555

# Crop parameters matching yaml
GLOBAL_SCALE = (0.30, 1.0)
LOCAL_SCALE = (0.05, 0.30)
RATIO = (1.0, 1.0)
LOCAL_SIZE_RATIO = 0.4375  # 98 for global_size=224


def lejepa(
    global_crops: int = 2,
    local_crops: int = 4,
    global_size: int = 224,
    local_size: int | None = None,
    global_scale: tuple[float, float] = GLOBAL_SCALE,
    local_scale: tuple[float, float] = LOCAL_SCALE,
    ratio: tuple[float, float] = RATIO,
    seed: int | None = None,
    device: str | None = None,
    dtype: torch.dtype = torch.float16,
    normalize: bool = True,
) -> dict[str, list]:
    """L-JEPA multi-crop SSL pipeline.

    Default configuration matches ssl_global2_local4_ratio1.yaml:
        - 2 global crops: size=224, scale=(0.30, 1.0), ratio=(1.0, 1.0)
        - 4 local crops: size=98, scale=(0.05, 0.30), ratio=(1.0, 1.0)
        - All crops get identical augmentations (symmetric)

    Pipeline per crop:
        DecodeMultiRandomResizedCrop → ToTorchImage → RandomHorizontalFlip →
        RandomColorJitterHSV → RandomGrayscale → RandomSolarization →
        [Normalize] → RandomGaussianBlur

    Args:
        global_crops: Number of large (global) crops.
        local_crops: Number of small (local) crops.
        global_size: Size of global crops.
        local_size: Size of local crops. Default: int(0.4375 * global_size).
        global_scale: Scale range for global crops.
        local_scale: Scale range for local crops.
        ratio: Aspect ratio range.
        seed: Base seed for reproducibility. None = non-reproducible.
        device: Device for GPU augmentations (None = CPU).
        dtype: Output tensor dtype.
        normalize: Whether to apply ImageNet normalization.

    Returns:
        Pipelines dict with DecodeMultiRandomResizedCrop + MultiCropPipeline.
        Output keys: 'global_0', 'global_1', 'local_0', 'local_1', 'local_2', 'local_3'.
    """
    dev = device or "cpu"

    # Compute local size if not specified (0.4375 * global_size = 98 for 224)
    if local_size is None:
        local_size = int(LOCAL_SIZE_RATIO * global_size)

    # Build crop specs matching yaml
    crop_specs = {}
    crop_id = 0

    # Global crops (image, image_1 in yaml)
    for i in range(global_crops):
        name = f"global_{i}"
        crop_specs[name] = dict(
            size=global_size,
            scale=global_scale,
            ratio=ratio,
            seed=_seed(seed, CROP_OFFSET, crop_id),
        )
        crop_id += 1

    # Local crops (image_2, image_3, image_4, image_5 in yaml)
    for i in range(local_crops):
        name = f"local_{i}"
        crop_specs[name] = dict(
            size=local_size,
            scale=local_scale,
            ratio=ratio,
            seed=_seed(seed, CROP_OFFSET, crop_id),
        )
        crop_id += 1

    # Build per-crop augmentation pipelines (identical for all crops)
    def _make_crop_pipeline(cid: int) -> list:
        """Build augmentation pipeline matching yaml."""
        return [
            ToTorchImage(device=dev, dtype=dtype),
            RandomHorizontalFlip(p=0.5, seed=_seed(seed, HFLIP_OFFSET, cid), device=dev),
            RandomColorJitterHSV(
                p=0.8, hue=0.1, saturation=0.4, value=0.4, contrast=0.4,
                seed=_seed(seed, JITTER_OFFSET, cid), device=dev,
            ),
            RandomGrayscale(p=0.2, seed=_seed(seed, GRAY_OFFSET, cid), device=dev),
            RandomSolarization(p=0.2, threshold=0.5, seed=_seed(seed, SOLAR_OFFSET, cid), device=dev),
            *([Normalize(IMAGENET_MEAN, IMAGENET_STD, device=dev, dtype=dtype)] if normalize else []),
            RandomGaussianBlur(
                p=0.2, kernel_size=(7, 7), sigma_range=(0.1, 2.0),
                seed=_seed(seed, BLUR_OFFSET, cid), device=dev,
            ),
        ]

    per_crop_pipes = {}
    crop_id = 0

    for i in range(global_crops):
        name = f"global_{i}"
        per_crop_pipes[name] = _make_crop_pipeline(crop_id)
        crop_id += 1

    for i in range(local_crops):
        name = f"local_{i}"
        per_crop_pipes[name] = _make_crop_pipeline(crop_id)
        crop_id += 1

    return {
        'image': [
            DecodeMultiRandomResizedCrop(crop_specs),
            MultiCropPipeline(per_crop_pipes),
        ],
    }
