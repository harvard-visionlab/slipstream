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
        to_tensor: If True, return torch.Tensor; if False, return numpy array.
        permute: If True, permute HWC→CHW. If False, keep HWC layout.

    Returns:
        If to_tensor=True (default): Tensor [B, 3, size, size] uint8.
        If to_tensor=False, permute=True: numpy [B, 3, size, size] uint8.
        If to_tensor=False, permute=False: numpy [B, size, size, 3] uint8.
    """

    def __init__(
        self,
        size: int = 224,
        num_threads: int = 0,
        to_tensor: bool = True,
        permute: bool = True,
    ) -> None:
        self.size = size
        self.to_tensor = to_tensor
        self.permute = permute
        self._decoder = NumbaBatchDecoder(num_threads=num_threads)

    def set_image_format(self, image_format: str) -> None:
        self._decoder = _swap_yuv420_if_needed(self._decoder, image_format)

    def __call__(self, batch_data: dict[str, Any]) -> torch.Tensor | np.ndarray:
        result = self._decoder.decode_batch_center_crop(
            batch_data['data'], batch_data['sizes'],
            batch_data['heights'], batch_data['widths'],
            crop_size=self.size,
        )
        if self.permute:
            result = self._decoder.hwc_to_chw(result)
        if self.to_tensor:
            return torch.from_numpy(result)
        return result

    def shutdown(self) -> None:
        self._decoder.shutdown()

    def __repr__(self) -> str:
        extra = ""
        if not self.to_tensor:
            extra += ", to_tensor=False"
        if not self.permute:
            extra += ", permute=False"
        return f"DecodeCenterCrop(size={self.size}{extra})"


class DecodeRandomResizedCrop(BatchTransform):
    """Decode JPEG batch with random resized crop (torchvision-compatible).

    Args:
        size: Output size (square).
        scale: Crop area range relative to original.
        ratio: Aspect ratio range.
        num_threads: Parallel decode threads. 0 = auto.
        seed: Seed for reproducible crops. None = non-reproducible.
        to_tensor: If True, return torch.Tensor; if False, return numpy array.
        permute: If True, permute HWC→CHW. If False, keep HWC layout.

    Returns:
        If to_tensor=True (default): Tensor [B, 3, size, size] uint8.
        If to_tensor=False, permute=True: numpy [B, 3, size, size] uint8.
        If to_tensor=False, permute=False: numpy [B, size, size, 3] uint8.
    """

    def __init__(
        self,
        size: int = 224,
        scale: tuple[float, float] = (0.08, 1.0),
        ratio: tuple[float, float] = (3 / 4, 4 / 3),
        num_threads: int = 0,
        seed: int | None = None,
        to_tensor: bool = True,
        permute: bool = True,
    ) -> None:
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.seed = seed
        self.to_tensor = to_tensor
        self.permute = permute
        self._decoder = NumbaBatchDecoder(num_threads=num_threads)

    def set_image_format(self, image_format: str) -> None:
        self._decoder = _swap_yuv420_if_needed(self._decoder, image_format)

    def __call__(self, batch_data: dict[str, Any]) -> torch.Tensor | np.ndarray:
        result = self._decoder.decode_batch_random_crop(
            batch_data['data'], batch_data['sizes'],
            batch_data['heights'], batch_data['widths'],
            target_size=self.size, scale=self.scale, ratio=self.ratio,
            seed=self.seed,
        )
        if self.permute:
            result = self._decoder.hwc_to_chw(result)
        if self.to_tensor:
            return torch.from_numpy(result)
        return result

    def shutdown(self) -> None:
        self._decoder.shutdown()

    def __repr__(self) -> str:
        extra = ""
        if not self.to_tensor:
            extra += ", to_tensor=False"
        if not self.permute:
            extra += ", permute=False"
        return (
            f"DecodeRandomResizedCrop(size={self.size}, scale={self.scale}, "
            f"ratio=({self.ratio[0]:.4f}, {self.ratio[1]:.4f}), seed={self.seed}{extra})"
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
        to_tensor: If True, return torch.Tensor; if False, return numpy array.
        permute: If True, permute HWC→CHW. If False, keep HWC layout.

    Returns:
        If to_tensor=True (default): Tensor [B, 3, size, size] uint8.
        If to_tensor=False, permute=True: numpy [B, 3, size, size] uint8.
        If to_tensor=False, permute=False: numpy [B, size, size, 3] uint8.
    """

    def __init__(
        self,
        size: int = 224,
        scale: tuple[float, float] = (0.08, 1.0),
        ratio: tuple[float, float] = (3 / 4, 4 / 3),
        num_threads: int = 0,
        seed: int | None = None,
        to_tensor: bool = True,
        permute: bool = True,
    ) -> None:
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.seed = seed
        self.to_tensor = to_tensor
        self.permute = permute
        self._decoder = NumbaBatchDecoder(num_threads=num_threads)

    def set_image_format(self, image_format: str) -> None:
        self._decoder = _swap_yuv420_if_needed(self._decoder, image_format)

    def __call__(self, batch_data: dict[str, Any]) -> torch.Tensor | np.ndarray:
        result = self._decoder.decode_batch_direct_random_crop(
            batch_data['data'], batch_data['sizes'],
            batch_data['heights'], batch_data['widths'],
            target_size=self.size, scale=self.scale, ratio=self.ratio,
            seed=self.seed,
        )
        if self.permute:
            result = self._decoder.hwc_to_chw(result)
        if self.to_tensor:
            return torch.from_numpy(result)
        return result

    def shutdown(self) -> None:
        self._decoder.shutdown()

    def __repr__(self) -> str:
        extra = ""
        if not self.to_tensor:
            extra += ", to_tensor=False"
        if not self.permute:
            extra += ", permute=False"
        return (
            f"DecodeDirectRandomResizedCrop(size={self.size}, scale={self.scale}, "
            f"ratio=({self.ratio[0]:.4f}, {self.ratio[1]:.4f}), seed={self.seed}{extra})"
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
        to_tensor: If True, return torch.Tensor; if False, return numpy array.
        permute: If True, permute HWC→CHW. If False, keep HWC layout.

    Returns:
        If to_tensor=True (default): Tensor [B, 3, crop_size, crop_size] uint8.
        If to_tensor=False, permute=True: numpy [B, 3, crop_size, crop_size] uint8.
        If to_tensor=False, permute=False: numpy [B, crop_size, crop_size, 3] uint8.
    """

    def __init__(
        self,
        resize_size: int = 256,
        crop_size: int = 224,
        num_threads: int = 0,
        to_tensor: bool = True,
        permute: bool = True,
    ) -> None:
        self.resize_size = resize_size
        self.crop_size = crop_size
        self.to_tensor = to_tensor
        self.permute = permute
        self._decoder = NumbaBatchDecoder(num_threads=num_threads)

    def set_image_format(self, image_format: str) -> None:
        self._decoder = _swap_yuv420_if_needed(self._decoder, image_format)

    def __call__(self, batch_data: dict[str, Any]) -> torch.Tensor | np.ndarray:
        result = self._decoder.decode_batch_resize_crop(
            batch_data['data'], batch_data['sizes'],
            batch_data['heights'], batch_data['widths'],
            resize_size=self.resize_size, crop_size=self.crop_size,
        )
        if self.permute:
            result = self._decoder.hwc_to_chw(result)
        if self.to_tensor:
            return torch.from_numpy(result)
        return result

    def shutdown(self) -> None:
        self._decoder.shutdown()

    def __repr__(self) -> str:
        extra = ""
        if not self.to_tensor:
            extra += ", to_tensor=False"
        if not self.permute:
            extra += ", permute=False"
        return f"DecodeResizeCrop(resize={self.resize_size}, crop={self.crop_size}{extra})"


class DecodeRRCFused(BatchTransform):
    """Experimental: Fused decode + device transfer + dtype conversion.

    Tests different strategies for optimal GPU pipeline performance.
    Bypasses ToTorchImage by handling device transfer and dtype conversion
    internally.

    Args:
        size: Output size (square).
        scale: Crop area range relative to original.
        ratio: Aspect ratio range.
        num_threads: Parallel decode threads. 0 = auto.
        seed: Seed for reproducible crops. None = non-reproducible.
        device: Target device ('cpu', 'cuda', or torch.device).
        dtype: Output dtype (default torch.float16).
        strategy: Transfer/permute strategy:
            - 'cpu_permute': NumPy transpose on CPU, transfer NCHW to GPU
            - 'gpu_permute': Transfer HWC to GPU, permute + contiguous on GPU
            - 'channels_last': Transfer HWC, use channels_last memory format (no permute)

    Returns:
        torch.Tensor [B, 3, size, size] float in [0, 1] (or BHWC for channels_last).
    """

    def __init__(
        self,
        size: int = 224,
        scale: tuple[float, float] = (0.08, 1.0),
        ratio: tuple[float, float] = (3 / 4, 4 / 3),
        num_threads: int = 0,
        seed: int | None = None,
        device: str = 'cuda',
        dtype: torch.dtype = torch.float16,
        strategy: str = 'cpu_permute',
    ) -> None:
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.seed = seed
        self.device = device
        self.dtype = dtype
        self.strategy = strategy
        self._decoder = NumbaBatchDecoder(num_threads=num_threads)
        self._pinned: torch.Tensor | None = None

    def set_image_format(self, image_format: str) -> None:
        self._decoder = _swap_yuv420_if_needed(self._decoder, image_format)

    def __call__(self, batch_data: dict[str, Any]) -> torch.Tensor:
        # Decode to numpy HWC uint8
        result = self._decoder.decode_batch_random_crop(
            batch_data['data'], batch_data['sizes'],
            batch_data['heights'], batch_data['widths'],
            target_size=self.size, scale=self.scale, ratio=self.ratio,
            seed=self.seed,
        )  # [B, H, W, 3] uint8 numpy

        B = result.shape[0]
        is_gpu = self.device != 'cpu' and self.device is not None and str(self.device) != 'cpu'

        if is_gpu:
            # Strategy A: CPU permute (NumPy transpose), then transfer NCHW
            if self.strategy == 'cpu_permute':
                result = result.transpose(0, 3, 1, 2)  # NumPy HWC→CHW (fast, returns view)
                # Make contiguous copy for pinned memory
                result = np.ascontiguousarray(result)
                shape = result.shape
                if self._pinned is None or self._pinned.shape != shape:
                    self._pinned = torch.empty(shape, dtype=torch.uint8, pin_memory=True)
                self._pinned.copy_(torch.from_numpy(result))
                t = self._pinned.to(self.device, non_blocking=True)
                return t.to(self.dtype).div_(255.0)

            # Strategy B: Transfer HWC, GPU permute + contiguous
            elif self.strategy == 'gpu_permute':
                shape = (B, self.size, self.size, 3)
                if self._pinned is None or self._pinned.shape[0] < B:
                    self._pinned = torch.empty(shape, dtype=torch.uint8, pin_memory=True)
                self._pinned[:B].copy_(torch.from_numpy(result))
                t = self._pinned[:B].to(self.device, non_blocking=True)
                t = t.permute(0, 3, 1, 2).contiguous()
                return t.to(self.dtype).div_(255.0)

            # Strategy C: Channels last (no permute, NHWC memory format)
            elif self.strategy == 'channels_last':
                shape = (B, self.size, self.size, 3)
                if self._pinned is None or self._pinned.shape[0] < B:
                    self._pinned = torch.empty(shape, dtype=torch.uint8, pin_memory=True)
                self._pinned[:B].copy_(torch.from_numpy(result))
                t = self._pinned[:B].to(self.device, non_blocking=True)
                # Reinterpret as NCHW but keep underlying NHWC memory layout
                t = t.permute(0, 3, 1, 2)
                return t.to(self.dtype, memory_format=torch.channels_last).div_(255.0)

            else:
                raise ValueError(f"Unknown strategy: {self.strategy}")

        # CPU path (for completeness)
        else:
            t = torch.from_numpy(result)
            if self.strategy != 'channels_last':
                t = t.permute(0, 3, 1, 2)
            return t.to(self.dtype).div_(255.0)

    def shutdown(self) -> None:
        self._decoder.shutdown()

    def __repr__(self) -> str:
        return (
            f"DecodeRRCFused(size={self.size}, scale={self.scale}, "
            f"ratio=({self.ratio[0]:.4f}, {self.ratio[1]:.4f}), seed={self.seed}, "
            f"device={self.device}, dtype={self.dtype}, strategy={self.strategy})"
        )


# Backward-compatible aliases (deprecated)
CenterCrop = DecodeCenterCrop
RandomResizedCrop = DecodeRandomResizedCrop
DirectRandomResizedCrop = DecodeDirectRandomResizedCrop
ResizeCrop = DecodeResizeCrop
