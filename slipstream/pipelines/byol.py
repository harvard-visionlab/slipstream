"""BYOL two-view SSL pipeline preset with asymmetric augmentation."""

from __future__ import annotations

import torch

from slipstream.decoders.crop import DecodeRandomResizedCrop
from slipstream.pipelines._common import CROP_OFFSET, _gpu_augmentations, _seed
from slipstream.transforms import IMAGENET_MEAN, IMAGENET_STD, Normalize, ToTorchImage


def byol(
    size: int = 224,
    seed: int | None = None,
    device: str | None = None,
    normalize: bool = True,
) -> dict[str, list[list]]:
    """BYOL two-view SSL pipeline with asymmetric augmentation.

    View 0 (online): strong blur (p=1.0), no solarization.
    View 1 (target): weak blur (p=0.1), solarization (p=0.2).

    Args:
        size: Output crop size.
        seed: Base seed for reproducibility. None = non-reproducible.
        device: Device for GPU augmentations (None = CPU).
        normalize: Whether to append ImageNet normalization.

    Returns:
        Pipelines dict with multi-view format.
    """
    dev = device or "cpu"
    views = []
    for view_id in range(2):
        stages: list = [
            DecodeRandomResizedCrop(size, seed=_seed(seed, CROP_OFFSET, view_id)),
            ToTorchImage(device=dev, dtype=torch.float16),
        ]

        if view_id == 0:
            stages.extend(_gpu_augmentations(
                seed=seed, crop_id=view_id, device=device,
                solarization=False, blur_p=1.0,
            ))
        else:
            stages.extend(_gpu_augmentations(
                seed=seed, crop_id=view_id, device=device,
                solarization=True, solar_p=0.2, blur_p=0.1,
            ))

        if normalize:
            stages.append(Normalize(IMAGENET_MEAN, IMAGENET_STD, device=dev))
        views.append(stages)
    return {'image': views}
