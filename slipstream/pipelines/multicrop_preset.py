"""Multi-crop SSL pipeline preset (e.g., DINO, iBOT)."""

from __future__ import annotations

from typing import Any

from slipstream.decoders.multicrop import MultiCropPipeline, MultiRandomResizedCrop
from slipstream.pipelines._common import CROP_OFFSET, _gpu_augmentations, _seed
from slipstream.transforms import IMAGENET_MEAN, IMAGENET_STD, Normalize, ToTorchImage


def multicrop(
    global_crops: int = 2,
    local_crops: int = 4,
    global_size: int = 224,
    local_size: int = 96,
    global_scale: tuple[float, float] = (0.4, 1.0),
    local_scale: tuple[float, float] = (0.05, 0.4),
    seed: int | None = None,
    device: str | None = None,
    normalize: bool = True,
) -> dict[str, list]:
    """Multi-crop SSL pipeline (e.g., DINO, iBOT).

    Uses ``MultiRandomResizedCrop`` for single-decode multi-crop with per-crop
    parameters, and ``MultiCropPipeline`` for per-crop GPU augmentations.

    Args:
        global_crops: Number of large crops.
        local_crops: Number of small crops.
        global_size: Size of global crops.
        local_size: Size of local crops.
        global_scale: Scale range for global crops.
        local_scale: Scale range for local crops.
        seed: Base seed for reproducibility. None = non-reproducible.
        device: Device for GPU augmentations (None = CPU).
        normalize: Whether to append ImageNet normalization.

    Returns:
        Pipelines dict using MultiRandomResizedCrop + MultiCropPipeline.
    """
    dev = device or "cpu"

    # Build crop specs
    crop_specs: dict[str, dict[str, Any]] = {}
    crop_id = 0

    for i in range(global_crops):
        name = f"global_{i}"
        crop_specs[name] = dict(
            size=global_size,
            scale=global_scale,
            seed=_seed(seed, CROP_OFFSET, crop_id),
        )
        crop_id += 1

    for i in range(local_crops):
        name = f"local_{i}"
        crop_specs[name] = dict(
            size=local_size,
            scale=local_scale,
            seed=_seed(seed, CROP_OFFSET, crop_id),
        )
        crop_id += 1

    # Build per-crop augmentation pipelines
    per_crop_pipes: dict[str, list] = {}
    crop_id = 0

    for i in range(global_crops):
        name = f"global_{i}"
        stages: list = [ToTorchImage(device=dev)]
        stages.extend(_gpu_augmentations(
            seed=seed, crop_id=crop_id, device=device,
            solarization=(i >= 1),
        ))
        if normalize:
            stages.append(Normalize(IMAGENET_MEAN, IMAGENET_STD, device=dev))
        per_crop_pipes[name] = stages
        crop_id += 1

    for i in range(local_crops):
        name = f"local_{i}"
        stages = [ToTorchImage(device=dev)]
        stages.extend(_gpu_augmentations(
            seed=seed, crop_id=crop_id, device=device,
            solarization=False,
        ))
        if normalize:
            stages.append(Normalize(IMAGENET_MEAN, IMAGENET_STD, device=dev))
        per_crop_pipes[name] = stages
        crop_id += 1

    return {
        'image': [
            MultiRandomResizedCrop(crop_specs),
            MultiCropPipeline(per_crop_pipes),
        ],
    }
