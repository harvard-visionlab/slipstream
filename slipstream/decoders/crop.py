"""Fused decode+crop stages.

Each class owns a NumbaBatchDecoder and accepts raw batch data
(JPEG bytes + metadata). The JPEG decode and crop/resize are fused
into a single operation for maximum throughput.

- DecodeCenterCrop: JPEG → center-crop → target_size
- DecodeRandomResizedCrop: JPEG → random crop → resize (torchvision-compatible)
- DecodeDirectRandomResizedCrop: JPEG → analytic random crop (no rejection sampling)
- DecodeResizeCrop: JPEG → resize shortest edge → center crop
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from slipstream.decoders.base import BatchTransform
from slipstream.decoders.numba_decoder import NumbaBatchDecoder


def _get_yuv420_decoder_class() -> type:
    from slipstream.decoders.yuv420_decoder import YUV420NumbaBatchDecoder
    return YUV420NumbaBatchDecoder


def _swap_yuv420_if_needed(decoder, image_format: str):
    """Replace decoder with YUV420 variant if needed. Returns new decoder."""
    if image_format == "yuv420" and not isinstance(decoder, _get_yuv420_decoder_class()):
        nt = decoder.num_threads
        decoder.shutdown()
        return _get_yuv420_decoder_class()(num_threads=nt)
    return decoder


class DecodeCenterCrop(BatchTransform):
    """Decode JPEG batch with center crop to fixed size.

    Args:
        size: Output size (square).
        num_threads: Parallel decode threads. 0 = auto.

    Returns:
        Tensor [B, 3, size, size] uint8.
    """

    def __init__(self, size: int = 224, num_threads: int = 0) -> None:
        self.size = size
        self._decoder = NumbaBatchDecoder(num_threads=num_threads)

    def set_image_format(self, image_format: str) -> None:
        self._decoder = _swap_yuv420_if_needed(self._decoder, image_format)

    def __call__(self, batch_data: dict[str, Any]) -> torch.Tensor:
        result = self._decoder.decode_batch_center_crop(
            batch_data['data'], batch_data['sizes'],
            batch_data['heights'], batch_data['widths'],
            crop_size=self.size,
        )
        chw = self._decoder.hwc_to_chw(result)
        return torch.from_numpy(chw)

    def shutdown(self) -> None:
        self._decoder.shutdown()

    def __repr__(self) -> str:
        return f"DecodeCenterCrop(size={self.size})"


class DecodeRandomResizedCrop(BatchTransform):
    """Decode JPEG batch with random resized crop (torchvision-compatible).

    Args:
        size: Output size (square).
        scale: Crop area range relative to original.
        ratio: Aspect ratio range.
        num_threads: Parallel decode threads. 0 = auto.
        seed: Seed for reproducible crops. None = non-reproducible.

    Returns:
        Tensor [B, 3, size, size] uint8.
    """

    def __init__(
        self,
        size: int = 224,
        scale: tuple[float, float] = (0.08, 1.0),
        ratio: tuple[float, float] = (3 / 4, 4 / 3),
        num_threads: int = 0,
        seed: int | None = None,
    ) -> None:
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.seed = seed
        self._decoder = NumbaBatchDecoder(num_threads=num_threads)

    def set_image_format(self, image_format: str) -> None:
        self._decoder = _swap_yuv420_if_needed(self._decoder, image_format)

    def __call__(self, batch_data: dict[str, Any]) -> torch.Tensor:
        result = self._decoder.decode_batch_random_crop(
            batch_data['data'], batch_data['sizes'],
            batch_data['heights'], batch_data['widths'],
            target_size=self.size, scale=self.scale, ratio=self.ratio,
            seed=self.seed,
        )
        chw = self._decoder.hwc_to_chw(result)
        return torch.from_numpy(chw)

    def shutdown(self) -> None:
        self._decoder.shutdown()

    def __repr__(self) -> str:
        return (
            f"DecodeRandomResizedCrop(size={self.size}, scale={self.scale}, "
            f"ratio=({self.ratio[0]:.4f}, {self.ratio[1]:.4f}), seed={self.seed})"
        )


class DecodeDirectRandomResizedCrop(BatchTransform):
    """Decode JPEG batch with analytic random resized crop (no rejection sampling).

    Unlike DecodeRandomResizedCrop which uses torchvision's 10-attempt rejection loop,
    this computes valid crop parameters analytically in a single pass by clamping
    the ratio range to values that guarantee a valid crop.

    The distribution differs slightly from torchvision's — rejection sampling
    biases toward certain regions of (scale, ratio) space. This method produces
    a uniform distribution over the valid parameter space.

    Args:
        size: Output size (square).
        scale: Crop area range relative to original.
        ratio: Aspect ratio range.
        num_threads: Parallel decode threads. 0 = auto.
        seed: Seed for reproducible crops. None = non-reproducible.

    Returns:
        Tensor [B, 3, size, size] uint8.
    """

    def __init__(
        self,
        size: int = 224,
        scale: tuple[float, float] = (0.08, 1.0),
        ratio: tuple[float, float] = (3 / 4, 4 / 3),
        num_threads: int = 0,
        seed: int | None = None,
    ) -> None:
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.seed = seed
        self._decoder = NumbaBatchDecoder(num_threads=num_threads)

    def set_image_format(self, image_format: str) -> None:
        self._decoder = _swap_yuv420_if_needed(self._decoder, image_format)

    def __call__(self, batch_data: dict[str, Any]) -> torch.Tensor:
        result = self._decoder.decode_batch_direct_random_crop(
            batch_data['data'], batch_data['sizes'],
            batch_data['heights'], batch_data['widths'],
            target_size=self.size, scale=self.scale, ratio=self.ratio,
            seed=self.seed,
        )
        chw = self._decoder.hwc_to_chw(result)
        return torch.from_numpy(chw)

    def shutdown(self) -> None:
        self._decoder.shutdown()

    def __repr__(self) -> str:
        return (
            f"DecodeDirectRandomResizedCrop(size={self.size}, scale={self.scale}, "
            f"ratio=({self.ratio[0]:.4f}, {self.ratio[1]:.4f}), seed={self.seed})"
        )


class DecodeResizeCrop(BatchTransform):
    """Decode JPEG batch with resize shortest edge + center crop.

    Standard ImageNet validation transform:
    1. Resize so shortest edge = resize_size
    2. Center crop to crop_size x crop_size

    Args:
        resize_size: Target size for shortest edge (default 256).
        crop_size: Final crop size (default 224).
        num_threads: Parallel decode threads. 0 = auto.

    Returns:
        Tensor [B, 3, crop_size, crop_size] uint8.
    """

    def __init__(
        self,
        resize_size: int = 256,
        crop_size: int = 224,
        num_threads: int = 0,
    ) -> None:
        self.resize_size = resize_size
        self.crop_size = crop_size
        self._decoder = NumbaBatchDecoder(num_threads=num_threads)

    def set_image_format(self, image_format: str) -> None:
        self._decoder = _swap_yuv420_if_needed(self._decoder, image_format)

    def __call__(self, batch_data: dict[str, Any]) -> torch.Tensor:
        result = self._decoder.decode_batch_resize_crop(
            batch_data['data'], batch_data['sizes'],
            batch_data['heights'], batch_data['widths'],
            resize_size=self.resize_size, crop_size=self.crop_size,
        )
        chw = self._decoder.hwc_to_chw(result)
        return torch.from_numpy(chw)

    def shutdown(self) -> None:
        self._decoder.shutdown()

    def __repr__(self) -> str:
        return f"DecodeResizeCrop(resize={self.resize_size}, crop={self.crop_size})"


# Backward-compatible aliases (deprecated)
CenterCrop = DecodeCenterCrop
RandomResizedCrop = DecodeRandomResizedCrop
DirectRandomResizedCrop = DecodeDirectRandomResizedCrop
ResizeCrop = DecodeResizeCrop
