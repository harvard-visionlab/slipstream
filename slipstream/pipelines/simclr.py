"""SimCLR two-view SSL pipeline preset."""

from __future__ import annotations

from slipstream.decoders.crop import DecodeRandomResizedCrop
from slipstream.pipelines._common import CROP_OFFSET, _gpu_augmentations, _seed
from slipstream.transforms import IMAGENET_MEAN, IMAGENET_STD, Normalize, ToTorchImage


def simclr(
    size: int = 224,
    seed: int | None = None,
    device: str | None = None,
    normalize: bool = True,
) -> dict[str, list[list]]:
    """SimCLR two-view SSL pipeline.

    Two views with independent crops and augmentations. View 1 includes
    solarization; view 0 does not (standard SimCLR asymmetry).

    Pipeline per view: DecodeRandomResizedCrop → ToTorchImage → HFlip → ColorJitter
    → Grayscale → GaussianBlur → [Solarization (view 1 only)] → Normalize.

    Args:
        size: Output crop size.
        seed: Base seed for reproducibility. None = non-reproducible.
        device: Device for GPU augmentations (None = CPU).
        normalize: Whether to append ImageNet normalization.

    Returns:
        Pipelines dict with multi-view format:
        ``{'image': [[view0_stages], [view1_stages]]}``.
    """
    dev = device or "cpu"
    views = []
    for view_id in range(2):
        stages: list = [
            DecodeRandomResizedCrop(size, seed=_seed(seed, CROP_OFFSET, view_id)),
            ToTorchImage(device=dev),
        ]
        stages.extend(_gpu_augmentations(
            seed=seed, crop_id=view_id, device=device,
            solarization=(view_id == 1),
        ))
        if normalize:
            stages.append(Normalize(IMAGENET_MEAN, IMAGENET_STD, device=dev))
        views.append(stages)
    return {'image': views}
