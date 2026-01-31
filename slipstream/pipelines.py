"""Batch pipelines for SlipstreamLoader.

Pipeline stages wrap NumbaBatchDecoder for high-performance fused decode+crop.

Key classes:
- DecodeOnly: JPEG → full-size RGB (variable sizes)
- CenterCrop: JPEG → center-crop → target_size
- RandomResizedCrop: JPEG → random crop → resize to target_size
- ResizeCrop: JPEG → resize shortest edge → center crop
- Normalize: ImageNet-style normalization
- ToDevice: Move tensor to device

Usage:
    from slipstream import SlipstreamLoader
    from slipstream.pipelines import RandomResizedCrop, Normalize

    # Single pipeline (standard training)
    loader = SlipstreamLoader(dataset, pipelines={
        'image': [RandomResizedCrop(224), Normalize()],
    })

    # Multi-crop SSL (two views with different random crops)
    loader = SlipstreamLoader(dataset, pipelines={
        'image': [
            [RandomResizedCrop(224), Normalize()],  # view 0
            [RandomResizedCrop(224), Normalize()],  # view 1
        ],
    })
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch

from slipstream.decoders.numba_decoder import NumbaBatchDecoder


def _get_yuv420_decoder_class() -> type:
    """Lazy import to avoid loading Numba/ctypes unless needed."""
    from slipstream.decoders.yuv420_decoder import YUV420NumbaBatchDecoder
    return YUV420NumbaBatchDecoder


# ImageNet defaults
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class BatchTransform(ABC):
    """Base class for batch transforms."""

    @abstractmethod
    def __call__(self, batch_data: Any) -> Any:
        ...

    def set_image_format(self, image_format: str) -> None:
        """Called by loader to configure decoder for cache format. No-op by default."""
        pass

    def shutdown(self) -> None:
        pass


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


class CenterCrop(BatchTransform):
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
        if image_format == "yuv420" and not isinstance(self._decoder, _get_yuv420_decoder_class()):
            nt = self._decoder.num_threads
            self._decoder.shutdown()
            self._decoder = _get_yuv420_decoder_class()(num_threads=nt)

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
        return f"CenterCrop(size={self.size})"


class RandomResizedCrop(BatchTransform):
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
        if image_format == "yuv420" and not isinstance(self._decoder, _get_yuv420_decoder_class()):
            nt = self._decoder.num_threads
            self._decoder.shutdown()
            self._decoder = _get_yuv420_decoder_class()(num_threads=nt)

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
            f"RandomResizedCrop(size={self.size}, scale={self.scale}, "
            f"ratio=({self.ratio[0]:.4f}, {self.ratio[1]:.4f}), seed={self.seed})"
        )


class MultiCropRandomResizedCrop(BatchTransform):
    """Decode-once + N random crops for SSL multi-crop.

    Decodes each JPEG once, then applies N different random crops from the
    same decoded image. Much faster than N separate RandomResizedCrop stages
    since JPEG decode (~80-92% of per-image time) happens only once.

    Args:
        num_crops: Number of random crop views per image.
        size: Output size (square).
        scale: Crop area range relative to original.
        ratio: Aspect ratio range.
        num_threads: Parallel decode threads. 0 = auto.
        seeds: Per-crop seeds for reproducibility. None = auto.

    Returns:
        List of num_crops tensors, each [B, 3, size, size] uint8.

    Example:
        # Two random crop views for SimCLR/BYOL
        multi_crop = MultiCropRandomResizedCrop(num_crops=2, size=224)
        views = multi_crop(batch_data)  # [tensor1, tensor2]
    """

    def __init__(
        self,
        num_crops: int = 2,
        size: int = 224,
        scale: tuple[float, float] = (0.08, 1.0),
        ratio: tuple[float, float] = (3 / 4, 4 / 3),
        num_threads: int = 0,
        seeds: list[int | None] | None = None,
    ) -> None:
        self.num_crops = num_crops
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.seeds = seeds
        self._decoder = NumbaBatchDecoder(num_threads=num_threads)

    def set_image_format(self, image_format: str) -> None:
        if image_format == "yuv420" and not isinstance(self._decoder, _get_yuv420_decoder_class()):
            nt = self._decoder.num_threads
            self._decoder.shutdown()
            self._decoder = _get_yuv420_decoder_class()(num_threads=nt)

    def __call__(self, batch_data: dict[str, Any]) -> list[torch.Tensor]:
        crops = self._decoder.decode_batch_multi_crop(
            batch_data['data'], batch_data['sizes'],
            batch_data['heights'], batch_data['widths'],
            num_crops=self.num_crops, target_size=self.size,
            scale=self.scale, ratio=self.ratio, seeds=self.seeds,
        )
        # multi_hwc_to_chw uses separate pre-allocated buffers per crop,
        # so no .clone() is needed (unlike hwc_to_chw which reuses one buffer).
        chw_crops = self._decoder.multi_hwc_to_chw(crops)
        return [torch.from_numpy(chw) for chw in chw_crops]

    def shutdown(self) -> None:
        self._decoder.shutdown()

    def __repr__(self) -> str:
        return (
            f"MultiCropRandomResizedCrop(num_crops={self.num_crops}, size={self.size}, "
            f"scale={self.scale}, ratio=({self.ratio[0]:.4f}, {self.ratio[1]:.4f}))"
        )


class ResizeCrop(BatchTransform):
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
        if image_format == "yuv420" and not isinstance(self._decoder, _get_yuv420_decoder_class()):
            nt = self._decoder.num_threads
            self._decoder.shutdown()
            self._decoder = _get_yuv420_decoder_class()(num_threads=nt)

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
        return f"ResizeCrop(resize={self.resize_size}, crop={self.crop_size})"


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


class Normalize(BatchTransform):
    """ImageNet-style normalization for batch tensors.

    Converts uint8 [0, 255] to float32 and applies (x - mean) / std.

    Args:
        mean: Per-channel mean (default: ImageNet).
        std: Per-channel std (default: ImageNet).
    """

    def __init__(
        self,
        mean: tuple[float, float, float] = IMAGENET_MEAN,
        std: tuple[float, float, float] = IMAGENET_STD,
    ) -> None:
        self.mean = mean
        self.std = std
        self._mean_tensor = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
        self._std_tensor = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
        self._device: str | None = None

    def _ensure_device(self, device: str) -> None:
        if self._device != device:
            self._mean_tensor = self._mean_tensor.to(device)
            self._std_tensor = self._std_tensor.to(device)
            self._device = device

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        self._ensure_device(str(images.device))
        if images.dtype == torch.uint8:
            images = images.float() / 255.0
        elif images.dtype != torch.float32:
            images = images.float()
        return (images - self._mean_tensor) / self._std_tensor

    def __repr__(self) -> str:
        return f"Normalize(mean={self.mean}, std={self.std})"


class ToDevice(BatchTransform):
    """Move tensor to specified device."""

    def __init__(self, device: str | int = 'cuda') -> None:
        if isinstance(device, int):
            self._device_str = f'cuda:{device}'
        else:
            self._device_str = device

    def __call__(self, data: torch.Tensor | np.ndarray) -> torch.Tensor:
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        return data.to(self._device_str)

    def __repr__(self) -> str:
        return f"ToDevice('{self._device_str}')"


class Compose(BatchTransform):
    """Compose multiple transforms into a single pipeline."""

    def __init__(self, transforms: list[BatchTransform]) -> None:
        self.transforms = transforms

    def __call__(self, data: Any) -> Any:
        for t in self.transforms:
            data = t(data)
        return data

    def shutdown(self) -> None:
        for t in self.transforms:
            if hasattr(t, 'shutdown'):
                t.shutdown()

    def __repr__(self) -> str:
        lines = [f"  {t}" for t in self.transforms]
        return "Compose([\n" + ",\n".join(lines) + "\n])"


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
    """Create a standard ImageNet training pipeline."""
    transforms: list[BatchTransform] = [
        RandomResizedCrop(size, scale, ratio, num_threads=num_threads, seed=seed),
    ]
    if normalize:
        transforms.append(Normalize(mean, std))
    return Compose(transforms)


def make_val_pipeline(
    size: int = 224,
    normalize: bool = True,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
    num_threads: int = 0,
) -> Compose:
    """Create a standard ImageNet validation pipeline."""
    transforms: list[BatchTransform] = [
        CenterCrop(size, num_threads=num_threads),
    ]
    if normalize:
        transforms.append(Normalize(mean, std))
    return Compose(transforms)


__all__ = [
    # Base
    "BatchTransform",
    "Compose",
    # Decode + crop transforms
    "DecodeOnly",
    "DecodeYUVFullRes",
    "DecodeYUVPlanes",
    "CenterCrop",
    "RandomResizedCrop",
    "MultiCropRandomResizedCrop",
    "ResizeCrop",
    # Post-processing
    "Normalize",
    "ToDevice",
    # Convenience
    "make_train_pipeline",
    "make_val_pipeline",
    # Constants
    "IMAGENET_MEAN",
    "IMAGENET_STD",
]
