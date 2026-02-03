"""Supervised training and validation pipeline presets."""

from __future__ import annotations

from slipstream.decoders.base import BatchTransform
from slipstream.decoders.crop import CenterCrop, RandomResizedCrop, ResizeCrop
from slipstream.pipelines._common import CROP_OFFSET, _seed
from slipstream.transforms import (
    Compose,
    IMAGENET_MEAN,
    IMAGENET_STD,
    Normalize,
    ToTorchImage,
)


def supervised_train(
    size: int = 224,
    seed: int | None = None,
    device: str | None = None,
    normalize: bool = True,
) -> dict[str, list]:
    """Standard supervised training pipeline.

    Pipeline: RandomResizedCrop → ToTorchImage → [Normalize].

    Decode stages return CHW uint8 torch tensors. ``ToTorchImage`` handles
    device transfer, dtype cast, and [0,255]→[0,1] conversion.

    Args:
        size: Output crop size.
        seed: Seed for reproducible crops. None = non-reproducible.
        device: Target device (None = CPU).
        normalize: Whether to append ImageNet normalization.

    Returns:
        Pipelines dict for ``SlipstreamLoader(pipelines=...)``.
    """
    stages: list = [
        RandomResizedCrop(size, seed=_seed(seed, CROP_OFFSET)),
        ToTorchImage(device=device or "cpu"),
    ]
    if normalize:
        stages.append(Normalize(IMAGENET_MEAN, IMAGENET_STD, device=device or "cpu"))
    return {'image': stages}


def supervised_val(
    size: int = 224,
    device: str | None = None,
    normalize: bool = True,
) -> dict[str, list]:
    """Standard supervised validation pipeline.

    Pipeline: ResizeCrop(256→size) → ToTorchImage → [Normalize].

    Args:
        size: Output crop size.
        device: Target device (None = CPU).
        normalize: Whether to append ImageNet normalization.

    Returns:
        Pipelines dict for ``SlipstreamLoader(pipelines=...)``.
    """
    stages: list = [
        ResizeCrop(resize_size=256, crop_size=size),
        ToTorchImage(device=device or "cpu"),
    ]
    if normalize:
        stages.append(Normalize(IMAGENET_MEAN, IMAGENET_STD, device=device or "cpu"))
    return {'image': stages}


def make_train_pipeline(
    size: int = 224,
    scale: tuple[float, float] = (0.08, 1.0),
    ratio: tuple[float, float] = (3/4, 4/3),
    normalize: bool = True,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
    num_threads: int = 0,
    seed: int | None = None,
) -> Compose:
    """Create a standard ImageNet training pipeline.

    .. deprecated:: Use :func:`supervised_train` instead for the dict-based API.
    """
    transforms: list = [
        RandomResizedCrop(size, scale, ratio, num_threads=num_threads, seed=seed),
        ToTorchImage(device="cpu"),
    ]
    if normalize:
        transforms.append(Normalize(mean, std, device="cpu"))
    return Compose(transforms)


def make_val_pipeline(
    size: int = 224,
    normalize: bool = True,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
    num_threads: int = 0,
) -> Compose:
    """Create a standard ImageNet validation pipeline.

    .. deprecated:: Use :func:`supervised_val` instead for the dict-based API.
    """
    transforms: list = [
        CenterCrop(size, num_threads=num_threads),
        ToTorchImage(device="cpu"),
    ]
    if normalize:
        transforms.append(Normalize(mean, std, device="cpu"))
    return Compose(transforms)
