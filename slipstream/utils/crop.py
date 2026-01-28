"""Unified crop parameter generation for slipstream.

This module provides crop parameter generation functions used by both CPU and GPU
decoders. Having a single source of truth ensures consistent behavior across
all decoding paths.

Key concepts:
- MCU (Minimum Coded Unit): JPEG compression block, typically 8x8 or 16x16 pixels
- DCT-space cropping: Cropping at MCU boundaries enables lossless, faster decode
- RandomResizedCrop: Training augmentation that randomly crops and resizes

Usage:
    from slipstream.utils.crop import generate_random_crop_params, align_to_mcu

    # Generate random crop params for training
    crop = generate_random_crop_params(
        width=384, height=256,
        scale=(0.08, 1.0),
        ratio=(3/4, 4/3),
    )

    # Align coordinates to MCU boundaries for efficient decode
    x_aligned = align_to_mcu(x, mcu_size=8)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class CropParams:
    """Parameters for a crop operation.

    All coordinates are in pixels, relative to the original image.
    For DCT-space cropping, use align_to_mcu() on x and y coordinates.

    Attributes:
        x: Left coordinate of crop region
        y: Top coordinate of crop region
        width: Width of crop region
        height: Height of crop region
    """

    x: int
    y: int
    width: int
    height: int

    def __iter__(self):
        """Allow unpacking: x, y, w, h = crop_params."""
        return iter((self.x, self.y, self.width, self.height))

    def to_tuple(self) -> tuple[int, int, int, int]:
        """Return as (x, y, width, height) tuple."""
        return (self.x, self.y, self.width, self.height)

    def to_roi_array(self) -> np.ndarray:
        """Return as numpy array [x, y, w, h] for batch operations."""
        return np.array([self.x, self.y, self.width, self.height], dtype=np.int32)


def align_to_mcu(value: int, mcu_size: int = 8, round_up: bool = False) -> int:
    """Align a coordinate to MCU (Minimum Coded Unit) boundary.

    JPEG images are encoded in blocks (MCUs), typically 8x8 pixels for baseline
    JPEG or 16x16 for progressive. DCT-space cropping is only possible at
    MCU boundaries, making aligned crops lossless and faster.

    Args:
        value: Coordinate value to align (x, y, width, or height)
        mcu_size: Size of MCU block (8 for baseline, 16 for some progressive)
        round_up: If True, round up to next boundary; otherwise round down

    Returns:
        Aligned coordinate value

    Examples:
        >>> align_to_mcu(13, mcu_size=8)  # Round down
        8
        >>> align_to_mcu(13, mcu_size=8, round_up=True)  # Round up
        16
    """
    if round_up:
        return ((value + mcu_size - 1) // mcu_size) * mcu_size
    return (value // mcu_size) * mcu_size


def generate_random_crop_params(
    width: int,
    height: int,
    scale: tuple[float, float] = (0.08, 1.0),
    ratio: tuple[float, float] = (3 / 4, 4 / 3),
    mcu_align: bool = True,
    mcu_size: int = 8,
    max_attempts: int = 10,
    rng: np.random.Generator | None = None,
) -> CropParams:
    """Generate random crop parameters for RandomResizedCrop.

    This implements the same algorithm as torchvision.transforms.RandomResizedCrop
    for consistent behavior across training pipelines.

    The algorithm:
    1. Sample target area uniformly from [scale[0]*area, scale[1]*area]
    2. Sample aspect ratio log-uniformly from ratio range
    3. Compute crop dimensions from area and aspect ratio
    4. Sample random position if crop fits; retry up to max_attempts times
    5. Fallback to center crop if no valid random crop found

    Args:
        width: Original image width in pixels
        height: Original image height in pixels
        scale: Range of size relative to original image (min, max)
        ratio: Range of aspect ratios (width/height) as (min, max)
        mcu_align: If True, align crop coordinates to MCU boundaries
        mcu_size: MCU block size for alignment (8 for baseline JPEG)
        max_attempts: Maximum attempts to find valid random crop
        rng: Optional numpy random generator for reproducibility

    Returns:
        CropParams with (x, y, width, height) for the crop region

    Examples:
        >>> crop = generate_random_crop_params(384, 256, scale=(0.08, 1.0))
        >>> print(f"Crop: x={crop.x}, y={crop.y}, w={crop.width}, h={crop.height}")

        # For reproducible augmentation:
        >>> rng = np.random.default_rng(seed=42)
        >>> crop = generate_random_crop_params(384, 256, rng=rng)
    """
    if rng is None:
        rng = np.random.default_rng()

    area = width * height
    log_ratio = (math.log(ratio[0]), math.log(ratio[1]))

    for _ in range(max_attempts):
        # Sample target area and aspect ratio
        target_area = area * rng.uniform(scale[0], scale[1])
        aspect_ratio = math.exp(rng.uniform(log_ratio[0], log_ratio[1]))

        # Compute crop dimensions
        crop_w = int(round(math.sqrt(target_area * aspect_ratio)))
        crop_h = int(round(math.sqrt(target_area / aspect_ratio)))

        # Check if crop fits in image
        if 0 < crop_w <= width and 0 < crop_h <= height:
            # Sample random position
            crop_x = rng.integers(0, width - crop_w + 1)
            crop_y = rng.integers(0, height - crop_h + 1)

            if mcu_align:
                # Align to MCU boundaries (round down for position)
                crop_x_aligned = align_to_mcu(crop_x, mcu_size)
                crop_y_aligned = align_to_mcu(crop_y, mcu_size)

                # Adjust width/height to compensate for position change
                # We want to cover the original crop region after alignment
                width_adjustment = crop_x - crop_x_aligned
                height_adjustment = crop_y - crop_y_aligned
                crop_w_adjusted = crop_w + width_adjustment
                crop_h_adjusted = crop_h + height_adjustment

                # Round up dimensions to MCU boundary for complete coverage
                crop_w_aligned = align_to_mcu(crop_w_adjusted, mcu_size, round_up=True)
                crop_h_aligned = align_to_mcu(crop_h_adjusted, mcu_size, round_up=True)

                # Clamp to image bounds
                crop_w_aligned = min(crop_w_aligned, width - crop_x_aligned)
                crop_h_aligned = min(crop_h_aligned, height - crop_y_aligned)

                return CropParams(
                    x=crop_x_aligned,
                    y=crop_y_aligned,
                    width=crop_w_aligned,
                    height=crop_h_aligned,
                )
            else:
                return CropParams(x=crop_x, y=crop_y, width=crop_w, height=crop_h)

    # Fallback: center crop with square aspect ratio
    return generate_center_crop_params(
        width, height,
        crop_size=min(width, height),
        mcu_align=mcu_align,
        mcu_size=mcu_size,
    )


def generate_center_crop_params(
    width: int,
    height: int,
    crop_size: int | None = None,
    crop_fraction: float | None = None,
    mcu_align: bool = True,
    mcu_size: int = 8,
) -> CropParams:
    """Generate center crop parameters.

    Used for validation/inference where deterministic center cropping is needed.

    Args:
        width: Original image width in pixels
        height: Original image height in pixels
        crop_size: Desired crop size in pixels (square crop).
                   If None, uses crop_fraction of the smaller dimension.
        crop_fraction: Fraction of smaller dimension to use as crop size.
                       Only used if crop_size is None. Default is 1.0 (full).
        mcu_align: If True, align crop coordinates to MCU boundaries
        mcu_size: MCU block size for alignment

    Returns:
        CropParams with center crop coordinates

    Examples:
        >>> crop = generate_center_crop_params(384, 256, crop_size=224)
        >>> print(f"Center crop at ({crop.x}, {crop.y})")

        # Crop to 80% of smaller dimension:
        >>> crop = generate_center_crop_params(384, 256, crop_fraction=0.8)
    """
    if crop_size is None:
        if crop_fraction is None:
            crop_fraction = 1.0
        crop_size = int(min(width, height) * crop_fraction)

    # Clamp crop size to image dimensions
    crop_size = min(crop_size, width, height)

    # Calculate center position
    crop_x = (width - crop_size) // 2
    crop_y = (height - crop_size) // 2

    if mcu_align:
        # Align to MCU boundaries
        crop_x_aligned = align_to_mcu(crop_x, mcu_size)
        crop_y_aligned = align_to_mcu(crop_y, mcu_size)

        # Adjust size to maintain coverage of center region
        width_adjustment = crop_x - crop_x_aligned
        height_adjustment = crop_y - crop_y_aligned
        crop_w = crop_size + width_adjustment
        crop_h = crop_size + height_adjustment

        # Round up to MCU boundary
        crop_w = align_to_mcu(crop_w, mcu_size, round_up=True)
        crop_h = align_to_mcu(crop_h, mcu_size, round_up=True)

        # Clamp to image bounds
        crop_w = min(crop_w, width - crop_x_aligned)
        crop_h = min(crop_h, height - crop_y_aligned)

        return CropParams(
            x=crop_x_aligned,
            y=crop_y_aligned,
            width=crop_w,
            height=crop_h,
        )
    else:
        return CropParams(x=crop_x, y=crop_y, width=crop_size, height=crop_size)


def generate_batch_random_crop_params(
    widths: np.ndarray,
    heights: np.ndarray,
    scale: tuple[float, float] = (0.08, 1.0),
    ratio: tuple[float, float] = (3 / 4, 4 / 3),
    mcu_align: bool = True,
    mcu_size: int = 8,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate random crop parameters for a batch of images.

    Vectorized version for batch processing. Returns an array of shape [B, 4]
    with columns (x, y, width, height) for each image.

    Args:
        widths: Array of image widths [B]
        heights: Array of image heights [B]
        scale: Range of size relative to original image
        ratio: Range of aspect ratios
        mcu_align: If True, align to MCU boundaries
        mcu_size: MCU block size for alignment
        rng: Optional numpy random generator

    Returns:
        Array of shape [B, 4] with (x, y, width, height) for each image
    """
    if rng is None:
        rng = np.random.default_rng()

    batch_size = len(widths)
    rois = np.zeros((batch_size, 4), dtype=np.int32)

    for i in range(batch_size):
        crop = generate_random_crop_params(
            width=int(widths[i]),
            height=int(heights[i]),
            scale=scale,
            ratio=ratio,
            mcu_align=mcu_align,
            mcu_size=mcu_size,
            rng=rng,
        )
        rois[i] = crop.to_roi_array()

    return rois


def generate_batch_center_crop_params(
    widths: np.ndarray,
    heights: np.ndarray,
    crop_size: int,
    mcu_align: bool = True,
    mcu_size: int = 8,
) -> np.ndarray:
    """Generate center crop parameters for a batch of images.

    Args:
        widths: Array of image widths [B]
        heights: Array of image heights [B]
        crop_size: Desired crop size (square)
        mcu_align: If True, align to MCU boundaries
        mcu_size: MCU block size for alignment

    Returns:
        Array of shape [B, 4] with (x, y, width, height) for each image
    """
    batch_size = len(widths)
    rois = np.zeros((batch_size, 4), dtype=np.int32)

    for i in range(batch_size):
        crop = generate_center_crop_params(
            width=int(widths[i]),
            height=int(heights[i]),
            crop_size=crop_size,
            mcu_align=mcu_align,
            mcu_size=mcu_size,
        )
        rois[i] = crop.to_roi_array()

    return rois


__all__ = [
    "CropParams",
    "align_to_mcu",
    "generate_random_crop_params",
    "generate_center_crop_params",
    "generate_batch_random_crop_params",
    "generate_batch_center_crop_params",
]
