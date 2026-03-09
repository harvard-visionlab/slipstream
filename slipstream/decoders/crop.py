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
        If to_tensor=False, permute=False (default): numpy [B, size, size, 3] uint8 (HWC).
        If to_tensor=False, permute=True: numpy [B, 3, size, size] uint8 (CHW).
        If to_tensor=True: Tensor with same layout as numpy output.

    Note:
        Default output (numpy HWC) is optimal for GPU pipelines when followed
        by ToTorchImage, which transfers contiguous HWC to GPU then permutes.
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
        If to_tensor=False, permute=False (default): numpy [B, size, size, 3] uint8 (HWC).
        If to_tensor=False, permute=True: numpy [B, 3, size, size] uint8 (CHW).
        If to_tensor=True: Tensor with same layout as numpy output.

    Note:
        Default output (numpy HWC) is optimal for GPU pipelines when followed
        by ToTorchImage, which transfers contiguous HWC to GPU then permutes.
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
        If to_tensor=False, permute=False (default): numpy [B, size, size, 3] uint8 (HWC).
        If to_tensor=False, permute=True: numpy [B, 3, size, size] uint8 (CHW).
        If to_tensor=True: Tensor with same layout as numpy output.

    Note:
        Default output (numpy HWC) is optimal for GPU pipelines when followed
        by ToTorchImage, which transfers contiguous HWC to GPU then permutes.
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
        If to_tensor=False, permute=False (default): numpy [B, crop_size, crop_size, 3] uint8 (HWC).
        If to_tensor=False, permute=True: numpy [B, 3, crop_size, crop_size] uint8 (CHW).
        If to_tensor=True: Tensor with same layout as numpy output.

    Note:
        Default output (numpy HWC) is optimal for GPU pipelines when followed
        by ToTorchImage, which transfers contiguous HWC to GPU then permutes.
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


class DecodeRandomResizeShortCropLong(BatchTransform):
    """Decode JPEG batch: resize short edge, crop long edge to square.

    Resize the shortest edge of each image to ``size`` (preserving aspect
    ratio), then crop the longest edge to ``size``, producing a square output.
    Crop position is controllable via ``x_range`` (horizontal) and ``y_range``
    (vertical).

    Args:
        size: Output size. If int, fixed. If tuple ``(min, max)``, a size is
            sampled uniformly from the inclusive range. See ``size_mode``.
        size_mode: ``"per_batch"`` samples one size shared by all images in
            the batch (output is a stacked array). ``"per_image"`` samples
            independently per image (output is a list of arrays). Ignored
            when ``size`` is a fixed int.
        x_range: Horizontal crop position in [0, 1]. Float or ``(min, max)``
            tuple for uniform random sampling. 0 = left, 0.5 = center,
            1 = right. Only active when image is wider than tall after resize.
        y_range: Vertical crop position in [0, 1]. Float or ``(min, max)``
            tuple. 0 = top, 0.5 = center, 1 = bottom. Only active when image
            is taller than wide after resize.
        num_threads: Parallel decode threads. 0 = auto.
        seed: Seed for reproducible crop positions and size sampling.
            None = non-reproducible.
        to_tensor: If True, return torch.Tensor; if False, return numpy array.
        permute: If True, permute HWC->CHW. If False, keep HWC layout.

    Returns:
        Fixed size or per_batch mode:
            numpy [B, size, size, 3] uint8 (HWC) or torch.Tensor.
        Per_image mode:
            list of numpy [size_i, size_i, 3] uint8 or torch.Tensor.

    Note:
        Default output (numpy HWC) is optimal for GPU pipelines when followed
        by ToTorchImage, which transfers contiguous HWC to GPU then permutes.

    Example::

        # Center crop (default) — equivalent to DecodeResizeCrop(224, 224)
        dec = DecodeRandomResizeShortCropLong(size=224)

        # Random horizontal jitter for landscape images
        dec = DecodeRandomResizeShortCropLong(size=224, x_range=(0.0, 1.0))

        # Random size per batch
        dec = DecodeRandomResizeShortCropLong(size=(192, 256), seed=42)

        # Random size per image (returns list)
        dec = DecodeRandomResizeShortCropLong(
            size=(96, 224), size_mode="per_image", seed=42
        )
    """

    def __init__(
        self,
        size: int | tuple[int, int] = 224,
        size_mode: str = "per_batch",
        x_range: float | tuple[float, float] = (0.5, 0.5),
        y_range: float | tuple[float, float] = (0.5, 0.5),
        num_threads: int = 0,
        seed: int | None = None,
        to_tensor: bool = False,
        permute: bool = False,
    ) -> None:
        if size_mode not in ("per_batch", "per_image"):
            raise ValueError(f"size_mode must be 'per_batch' or 'per_image', got '{size_mode}'")
        if isinstance(size, int):
            self.size_range = (size, size)
        else:
            self.size_range = tuple(size)
        self.size_mode = size_mode
        self.x_range = (x_range, x_range) if isinstance(x_range, (int, float)) else tuple(x_range)
        self.y_range = (y_range, y_range) if isinstance(y_range, (int, float)) else tuple(y_range)
        self.seed = seed
        self.to_tensor = to_tensor
        self.permute = permute
        self._decoder = NumbaBatchDecoder(num_threads=num_threads)
        self._last_params: dict = {}

    def set_image_format(self, image_format: str) -> None:
        self._decoder = _swap_yuv420_if_needed(self._decoder, image_format)

    def __call__(self, batch_data: dict[str, Any] | bytes | bytearray | memoryview) -> torch.Tensor | np.ndarray | list:
        if isinstance(batch_data, (bytes, bytearray, memoryview)):
            from slipstream.decoders.base import _bytes_to_batch_dict, _unwrap_single_result
            return _unwrap_single_result(self(_bytes_to_batch_dict(batch_data)))

        from slipstream.decoders.numba_decoder import (
            _compute_resize_short_crop_long_params,
        )

        batch_size = len(batch_data['sizes'])

        self._decoder._seed_counter += 1
        batch_offset = self._decoder._seed_counter

        # --- Sample target sizes ---
        if self.size_range[0] == self.size_range[1]:
            # Fixed size
            target_sizes = np.full(batch_size, self.size_range[0], dtype=np.int32)
        elif self.size_mode == "per_batch":
            # One random size for the whole batch
            if self.seed is not None:
                rng = np.random.RandomState(
                    (self.seed + batch_offset) % 2147483647
                )
            else:
                rng = np.random.RandomState(
                    (batch_offset * 7919) % 2147483647
                )
            s = int(rng.randint(self.size_range[0], self.size_range[1] + 1))
            target_sizes = np.full(batch_size, s, dtype=np.int32)
        else:
            # Per-image random sizes
            target_sizes = np.empty(batch_size, dtype=np.int32)
            for i in range(batch_size):
                if self.seed is not None:
                    rng_i = np.random.RandomState(
                        (self.seed + batch_size * batch_offset + i) % 2147483647
                    )
                else:
                    rng_i = np.random.RandomState(
                        (batch_size * batch_offset + i) % 2147483647
                    )
                target_sizes[i] = int(rng_i.randint(self.size_range[0], self.size_range[1] + 1))

        # --- Sample crop positions [B] ---
        if self.seed is not None:
            batch_seed = (self.seed + batch_size * batch_offset) % 2147483647
        else:
            batch_seed = (batch_size * batch_offset) % 2147483647

        x_pos = np.empty(batch_size, dtype=np.float64)
        y_pos = np.empty(batch_size, dtype=np.float64)
        for i in range(batch_size):
            rng_i = np.random.RandomState((batch_seed + i) % 2147483647)
            x_pos[i] = rng_i.uniform(self.x_range[0], self.x_range[1])
            y_pos[i] = rng_i.uniform(self.y_range[0], self.y_range[1])

        # --- Decode ---
        result = self._decoder.decode_batch_resize_short_crop_long(
            batch_data['data'], batch_data['sizes'],
            batch_data['heights'], batch_data['widths'],
            target_sizes=target_sizes,
            x_pos=x_pos,
            y_pos=y_pos,
        )

        # --- Store last_params for inspection ---
        crop_params = _compute_resize_short_crop_long_params(
            batch_data['heights'], batch_data['widths'],
            target_sizes, x_pos, y_pos,
        )
        self._last_params = {
            "target_sizes": target_sizes,
            "x_pos": x_pos,
            "y_pos": y_pos,
            "heights": np.asarray(batch_data['heights'], dtype=np.int32),
            "widths": np.asarray(batch_data['widths'], dtype=np.int32),
            "crop_params": crop_params,
        }

        # --- Post-process ---
        if isinstance(result, list):
            # Per-image variable sizes
            if self.permute:
                result = [np.ascontiguousarray(r.transpose(2, 0, 1)) for r in result]
            if self.to_tensor:
                result = [torch.from_numpy(r) for r in result]
            return result
        else:
            if self.permute:
                result = self._decoder.hwc_to_chw(result)
            if self.to_tensor:
                return torch.from_numpy(result)
            return result

    @property
    def last_params(self) -> dict:
        """Parameters from the most recent ``__call__``."""
        return self._last_params

    def shutdown(self) -> None:
        self._decoder.shutdown()

    def __repr__(self) -> str:
        extra = ""
        if self.size_range[0] != self.size_range[1]:
            extra += f", size_mode='{self.size_mode}'"
        if self.x_range != (0.5, 0.5):
            extra += f", x_range={self.x_range}"
        if self.y_range != (0.5, 0.5):
            extra += f", y_range={self.y_range}"
        if self.seed is not None:
            extra += f", seed={self.seed}"
        if not self.to_tensor:
            extra += ", to_tensor=False"
        if not self.permute:
            extra += ", permute=False"
        size_str = self.size_range[0] if self.size_range[0] == self.size_range[1] else self.size_range
        return f"DecodeRandomResizeShortCropLong(size={size_str}{extra})"


# Backward-compatible aliases (deprecated)
CenterCrop = DecodeCenterCrop
RandomResizedCrop = DecodeRandomResizedCrop
DirectRandomResizedCrop = DecodeDirectRandomResizedCrop
ResizeCrop = DecodeResizeCrop
RandomResizeShortCropLong = DecodeRandomResizeShortCropLong
