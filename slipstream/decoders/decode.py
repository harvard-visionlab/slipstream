"""Decode stages for YUV and RGB output.

Pure decode (variable size output):
- DecodeOnly: JPEG/YUV420 → full-size RGB (variable sizes)
- DecodeYUVFullRes: YUV420P → full-res YUV [H,W,3]
- DecodeYUVPlanes: YUV420P → raw Y/U/V planes

YUV-output crop stages (fixed size, keeps YUV colorspace):
- DecodeYUVCenterCrop: YUV420P → center crop → [B,H,W,3] YUV
- DecodeYUVRandomResizedCrop: YUV420P → random crop → [B,H,W,3] YUV
- DecodeYUVResizeCrop: YUV420P → resize + center crop → [B,H,W,3] YUV
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from slipstream.decoders.base import BatchTransform
from slipstream.decoders.numba_decoder import NumbaBatchDecoder


def _get_yuv420_decoder_class() -> type:
    """Lazy import to avoid loading Numba/ctypes unless needed."""
    from slipstream.decoders.yuv420_decoder import YUV420NumbaBatchDecoder
    return YUV420NumbaBatchDecoder


class DecodeOnly(BatchTransform):
    """Decode JPEG batch to full-size RGB images.

    Returns a list of numpy arrays since images have variable sizes.

    Args:
        num_threads: Parallel decode threads. 0 = auto (cpu_count).
    """

    def __init__(self, num_threads: int = 0) -> None:
        self._decoder = NumbaBatchDecoder(num_threads=num_threads)

    def set_image_format(self, image_format: str) -> None:
        if image_format == "yuv420" and not isinstance(self._decoder, _get_yuv420_decoder_class()):
            nt = self._decoder.num_threads
            self._decoder.shutdown()
            self._decoder = _get_yuv420_decoder_class()(num_threads=nt)

    def __call__(self, batch_data: dict[str, Any]) -> list[np.ndarray]:
        return self._decoder.decode_batch(
            batch_data['data'], batch_data['sizes'],
            batch_data['heights'], batch_data['widths'],
        )

    def shutdown(self) -> None:
        self._decoder.shutdown()

    def __repr__(self) -> str:
        return f"DecodeOnly(num_threads={self._decoder.num_threads})"


class DecodeYUVFullRes(BatchTransform):
    """Decode YUV420P batch to full-resolution YUV (nearest-neighbor chroma upsample).

    Returns a tensor [B, 3, H, W] uint8 where channels are (Y, U, V).
    Same shape as RGB output but in YUV colorspace. Only works with
    ``image_format="yuv420"``.

    Args:
        num_threads: Parallel decode threads. 0 = auto.
    """

    def __init__(self, num_threads: int = 0) -> None:
        self._decoder = _get_yuv420_decoder_class()(num_threads=num_threads)

    def __call__(self, batch_data: dict[str, Any]) -> list[np.ndarray]:
        return self._decoder.decode_batch_yuv_fullres(
            batch_data['data'], batch_data['sizes'],
            batch_data['heights'], batch_data['widths'],
        )

    def shutdown(self) -> None:
        self._decoder.shutdown()

    def __repr__(self) -> str:
        return f"DecodeYUVFullRes(num_threads={self._decoder.num_threads})"


class DecodeYUVPlanes(BatchTransform):
    """Extract raw YUV420P planes without conversion.

    Returns a list of ``(Y, U, V)`` tuples where:
    - Y: ``[H, W]`` uint8
    - U: ``[H/2, W/2]`` uint8
    - V: ``[H/2, W/2]`` uint8

    This is the fastest possible decode — just memcpy of planes.
    Only works with ``image_format="yuv420"``.

    Args:
        num_threads: Parallel decode threads. 0 = auto.
    """

    def __init__(self, num_threads: int = 0) -> None:
        self._decoder = _get_yuv420_decoder_class()(num_threads=num_threads)

    def __call__(self, batch_data: dict[str, Any]) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        return self._decoder.decode_batch_yuv_planes(
            batch_data['data'], batch_data['sizes'],
            batch_data['heights'], batch_data['widths'],
        )

    def shutdown(self) -> None:
        self._decoder.shutdown()

    def __repr__(self) -> str:
        return f"DecodeYUVPlanes(num_threads={self._decoder.num_threads})"


class DecodeYUVCenterCrop(BatchTransform):
    """Decode YUV420P batch with center crop, keeping YUV colorspace.

    Returns [B, size, size, 3] uint8 where channels are (Y, U, V).
    Only works with ``image_format="yuv420"``.

    Args:
        size: Output size (square).
        num_threads: Parallel decode threads. 0 = auto.
        to_tensor: If True, return torch.Tensor; if False, return numpy array.
        permute: If True, permute HWC→CHW. If False, keep HWC layout.
    """

    def __init__(
        self,
        size: int = 224,
        num_threads: int = 0,
        to_tensor: bool = False,
        permute: bool = False,
    ) -> None:
        self.size = size
        self.to_tensor = to_tensor
        self.permute = permute
        self._decoder = _get_yuv420_decoder_class()(num_threads=num_threads)

    def __call__(self, batch_data: dict[str, Any]) -> torch.Tensor | np.ndarray:
        result = self._decoder.decode_batch_yuv_center_crop(
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
        return f"DecodeYUVCenterCrop(size={self.size}{extra})"


class DecodeYUVRandomResizedCrop(BatchTransform):
    """Decode YUV420P batch with random resized crop, keeping YUV colorspace.

    Returns [B, size, size, 3] uint8 where channels are (Y, U, V).
    Only works with ``image_format="yuv420"``.

    Args:
        size: Output size (square).
        scale: Crop area range relative to original.
        ratio: Aspect ratio range.
        num_threads: Parallel decode threads. 0 = auto.
        seed: Seed for reproducible crops. None = non-reproducible.
        to_tensor: If True, return torch.Tensor; if False, return numpy array.
        permute: If True, permute HWC→CHW. If False, keep HWC layout.
    """

    def __init__(
        self,
        size: int = 224,
        scale: tuple[float, float] = (0.08, 1.0),
        ratio: tuple[float, float] = (3 / 4, 4 / 3),
        num_threads: int = 0,
        seed: int | None = None,
        to_tensor: bool = False,
        permute: bool = False,
    ) -> None:
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.seed = seed
        self.to_tensor = to_tensor
        self.permute = permute
        self._decoder = _get_yuv420_decoder_class()(num_threads=num_threads)

    def __call__(self, batch_data: dict[str, Any]) -> torch.Tensor | np.ndarray:
        result = self._decoder.decode_batch_yuv_random_crop(
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
            f"DecodeYUVRandomResizedCrop(size={self.size}, scale={self.scale}, "
            f"ratio=({self.ratio[0]:.4f}, {self.ratio[1]:.4f}), seed={self.seed}{extra})"
        )


class DecodeYUVResizeCrop(BatchTransform):
    """Decode YUV420P batch with resize shortest edge + center crop, keeping YUV colorspace.

    Standard ImageNet validation transform but outputs YUV:
    1. Resize so shortest edge = resize_size
    2. Center crop to crop_size x crop_size

    Returns [B, crop_size, crop_size, 3] uint8 where channels are (Y, U, V).
    Only works with ``image_format="yuv420"``.

    Args:
        resize_size: Target size for shortest edge (default 256).
        crop_size: Final crop size (default 224).
        num_threads: Parallel decode threads. 0 = auto.
        to_tensor: If True, return torch.Tensor; if False, return numpy array.
        permute: If True, permute HWC→CHW. If False, keep HWC layout.
    """

    def __init__(
        self,
        resize_size: int = 256,
        crop_size: int = 224,
        num_threads: int = 0,
        to_tensor: bool = False,
        permute: bool = False,
    ) -> None:
        self.resize_size = resize_size
        self.crop_size = crop_size
        self.to_tensor = to_tensor
        self.permute = permute
        self._decoder = _get_yuv420_decoder_class()(num_threads=num_threads)

    def __call__(self, batch_data: dict[str, Any]) -> torch.Tensor | np.ndarray:
        result = self._decoder.decode_batch_yuv_resize_crop(
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
        return f"DecodeYUVResizeCrop(resize={self.resize_size}, crop={self.crop_size}{extra})"
