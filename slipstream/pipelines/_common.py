"""Shared utilities for pipeline presets."""

from __future__ import annotations

from slipstream.transforms import (
    RandomColorJitterHSV,
    RandomGaussianBlur,
    RandomGrayscale,
    RandomHorizontalFlip,
    RandomSolarization,
)

# Seed offset constants (same convention as lrm-ssl configs)
CROP_OFFSET = 1234
FLIP_OFFSET = 1111
COLOR_OFFSET = 2222
GRAY_OFFSET = 3333
SOLAR_OFFSET = 4444
BLUR_OFFSET = 5555


def _seed(base: int | None, offset: int, crop_id: int = 0) -> int | None:
    """Derive a deterministic seed, or None if base is None."""
    if base is None:
        return None
    return base + offset + crop_id


def _gpu_augmentations(
    seed: int | None,
    crop_id: int,
    device: str | None,
    solarization: bool = False,
    solar_p: float = 0.2,
    blur_p: float = 0.5,
    color_jitter_p: float = 0.8,
    grayscale_p: float = 0.2,
) -> list:
    """Build a standard SSL GPU augmentation chain.

    These operate on float [0,1] BCHW tensors (after ToTorchImage).
    Normalize should come *after* these augmentations.
    """
    augs: list = [
        RandomHorizontalFlip(p=0.5, seed=_seed(seed, FLIP_OFFSET, crop_id), device=device),
        RandomColorJitterHSV(
            p=color_jitter_p,
            hue=0.1, saturation=0.4, value=0.4, contrast=0.4,
            seed=_seed(seed, COLOR_OFFSET, crop_id), device=device,
        ),
        RandomGrayscale(p=grayscale_p, seed=_seed(seed, GRAY_OFFSET, crop_id), device=device),
        RandomGaussianBlur(p=blur_p, seed=_seed(seed, BLUR_OFFSET, crop_id), device=device),
    ]
    if solarization:
        augs.append(
            RandomSolarization(p=solar_p, seed=_seed(seed, SOLAR_OFFSET, crop_id), device=device)
        )
    return augs
