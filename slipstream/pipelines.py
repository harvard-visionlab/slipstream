"""Batch pipelines for SlipstreamLoader.

This module provides composable batch transforms for high-performance training.
Pipelines are applied to batches of data loaded from OptimizedCache.

Key classes:
- Decoder: Auto-selects GPU/CPU decoder based on device
- RandomResizedCrop: Training crop with random scale/ratio
- CenterCrop: Validation center crop
- Normalize: ImageNet-style normalization

Usage:
    from slipstream import SlipstreamLoader
    from slipstream.pipelines import Decoder, RandomResizedCrop, Normalize

    loader = SlipstreamLoader(
        dataset,
        batch_size=256,
        pipelines={
            'image': [
                Decoder(device='cuda'),
                RandomResizedCrop(224, scale=(0.08, 1.0)),
                Normalize(),
            ],
        },
    )

    # No pipeline = raw bytes (for benchmarking)
    raw_loader = SlipstreamLoader(dataset, batch_size=256)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch

from slipstream.decoders import (
    CPUDecoder,
    GPUDecoder,
    GPUDecoderFallback,
    check_gpu_decoder_available,
)


# ImageNet defaults
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class BatchTransform(ABC):
    """Base class for batch transforms."""

    @abstractmethod
    def __call__(self, batch_data: dict[str, Any]) -> Any:
        """Apply transform to batch data.

        Args:
            batch_data: Dict with 'data', 'sizes', 'heights', 'widths' for images,
                       or just 'data' for other fields.

        Returns:
            Transformed data (tensor or other)
        """
        ...

    def shutdown(self) -> None:
        """Release any resources."""
        pass


class Decoder(BatchTransform):
    """JPEG batch decoder that auto-selects GPU or CPU based on device.

    This is a convenience wrapper that:
    - Uses GPUDecoder (nvImageCodec) if device='cuda' and available
    - Falls back to GPUDecoderFallback (CPU decode + GPU transfer) if nvImageCodec unavailable
    - Uses CPUDecoder (TurboJPEG) if device='cpu'

    Args:
        device: 'cuda', 'cuda:N', 'cpu', or int (GPU index)
        num_workers: Number of CPU workers (for CPU decoder or fallback)
        max_batch_size: Max batch size for GPU decoder buffer allocation

    Example:
        decoder = Decoder(device='cuda')  # Uses GPU if available
        decoder = Decoder(device='cpu')   # Uses CPU (TurboJPEG)
        decoder = Decoder(device=0)       # GPU 0
    """

    def __init__(
        self,
        device: str | int = 'cuda',
        num_workers: int = 8,
        max_batch_size: int = 256,
    ) -> None:
        self.num_workers = num_workers
        self.max_batch_size = max_batch_size

        # Parse device
        if isinstance(device, int):
            self._device_idx = device
            self._device_str = f'cuda:{device}'
            self._use_gpu = True
        elif device == 'cpu':
            self._device_idx = None
            self._device_str = 'cpu'
            self._use_gpu = False
        elif device.startswith('cuda'):
            if ':' in device:
                self._device_idx = int(device.split(':')[1])
            else:
                self._device_idx = 0
            self._device_str = f'cuda:{self._device_idx}'
            self._use_gpu = True
        else:
            raise ValueError(f"Invalid device: {device}. Use 'cuda', 'cuda:N', 'cpu', or int.")

        # Select decoder
        if self._use_gpu and check_gpu_decoder_available():
            self._decoder = GPUDecoder(
                device=self._device_idx,
                max_batch_size=max_batch_size,
            )
            self._decoder_type = 'gpu'
        elif self._use_gpu and torch.cuda.is_available():
            self._decoder = GPUDecoderFallback(
                device=self._device_idx,
                num_workers=num_workers,
            )
            self._decoder_type = 'gpu_fallback'
        else:
            self._decoder = CPUDecoder(num_workers=num_workers)
            self._decoder_type = 'cpu'

    @property
    def device(self) -> str:
        """Return device string."""
        return self._device_str

    def __call__(self, batch_data: dict[str, Any]) -> torch.Tensor:
        """Decode batch of JPEG bytes to tensor.

        Args:
            batch_data: Dict with 'data', 'sizes', 'heights', 'widths'

        Returns:
            Decoded images as tensor [B, C, H, W] (variable sizes per image)
        """
        data = batch_data['data']
        sizes = batch_data['sizes']

        images = self._decoder.decode_batch(data, sizes)

        # CPU decoder returns list of numpy arrays
        if isinstance(images, list):
            # Convert to list of tensors
            tensors = [torch.from_numpy(img).permute(2, 0, 1) for img in images]
            return tensors  # Variable sizes, can't stack

        return images

    def shutdown(self) -> None:
        """Release decoder resources."""
        if hasattr(self._decoder, 'shutdown'):
            self._decoder.shutdown()

    def __repr__(self) -> str:
        return f"Decoder(device='{self._device_str}', type='{self._decoder_type}')"


class RandomResizedCrop(BatchTransform):
    """Batch RandomResizedCrop with fused decode+crop.

    For GPU decoder, decode and crop are fused into one operation.
    For CPU decoder, uses TurboJPEG's DCT-space cropping for speed.

    Args:
        size: Output size (square)
        scale: Scale range for crop area
        ratio: Aspect ratio range
        device: Device for output tensors

    Example:
        rrc = RandomResizedCrop(224, scale=(0.08, 1.0), device='cuda')
    """

    def __init__(
        self,
        size: int = 224,
        scale: tuple[float, float] = (0.08, 1.0),
        ratio: tuple[float, float] = (3/4, 4/3),
        device: str | int = 'cuda',
        num_workers: int = 8,
        max_batch_size: int = 256,
    ) -> None:
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.num_workers = num_workers
        self.max_batch_size = max_batch_size

        # Parse device
        if isinstance(device, int):
            self._device_idx = device
            self._device_str = f'cuda:{device}'
            self._use_gpu = True
        elif device == 'cpu':
            self._device_idx = None
            self._device_str = 'cpu'
            self._use_gpu = False
        elif device.startswith('cuda'):
            if ':' in device:
                self._device_idx = int(device.split(':')[1])
            else:
                self._device_idx = 0
            self._device_str = f'cuda:{self._device_idx}'
            self._use_gpu = True
        else:
            raise ValueError(f"Invalid device: {device}")

        # Select decoder (with fused crop)
        if self._use_gpu and check_gpu_decoder_available():
            self._decoder = GPUDecoder(
                device=self._device_idx,
                max_batch_size=max_batch_size,
            )
            self._decoder_type = 'gpu'
        elif self._use_gpu and torch.cuda.is_available():
            self._decoder = GPUDecoderFallback(
                device=self._device_idx,
                num_workers=num_workers,
            )
            self._decoder_type = 'gpu_fallback'
        else:
            self._decoder = CPUDecoder(num_workers=num_workers)
            self._decoder_type = 'cpu'

    def __call__(self, batch_data: dict[str, Any]) -> torch.Tensor:
        """Decode and crop batch of JPEG bytes.

        Args:
            batch_data: Dict with 'data', 'sizes', 'heights', 'widths'

        Returns:
            Cropped images as tensor [B, 3, size, size]
        """
        data = batch_data['data']
        sizes = batch_data['sizes']
        heights = batch_data['heights']
        widths = batch_data['widths']

        if self._decoder_type in ('gpu', 'gpu_fallback'):
            images = self._decoder.decode_batch_random_crop(
                data, sizes, heights, widths,
                target_size=self.size,
                scale=self.scale,
                ratio=self.ratio,
            )
        else:
            # CPU decoder returns list
            images_list = self._decoder.decode_batch_random_crop(
                data, sizes, heights, widths,
                target_size=self.size,
                scale=self.scale,
                ratio=self.ratio,
            )
            images = self._stack_and_resize(images_list)

        return images

    def _stack_and_resize(self, images_list: list[np.ndarray]) -> torch.Tensor:
        """Stack CPU-decoded images, resizing if needed."""
        from torch.nn import functional as F

        batch_size = len(images_list)
        output = torch.zeros(
            (batch_size, 3, self.size, self.size),
            dtype=torch.uint8,
            device=self._device_str,
        )

        for i, img_hwc in enumerate(images_list):
            tensor = torch.from_numpy(img_hwc).permute(2, 0, 1)
            h, w = img_hwc.shape[:2]

            if h != self.size or w != self.size:
                tensor = F.interpolate(
                    tensor.unsqueeze(0).float(),
                    size=(self.size, self.size),
                    mode='bilinear',
                    align_corners=False,
                ).squeeze(0).to(torch.uint8)

            output[i] = tensor.to(self._device_str)

        return output

    def shutdown(self) -> None:
        if hasattr(self._decoder, 'shutdown'):
            self._decoder.shutdown()

    def __repr__(self) -> str:
        return (
            f"RandomResizedCrop(size={self.size}, scale={self.scale}, "
            f"ratio={self.ratio}, device='{self._device_str}')"
        )


class CenterCrop(BatchTransform):
    """Batch CenterCrop with fused decode+crop.

    Args:
        size: Output size (square)
        device: Device for output tensors

    Example:
        cc = CenterCrop(224, device='cuda')
    """

    def __init__(
        self,
        size: int = 224,
        device: str | int = 'cuda',
        num_workers: int = 8,
        max_batch_size: int = 256,
    ) -> None:
        self.size = size
        self.num_workers = num_workers
        self.max_batch_size = max_batch_size

        # Parse device
        if isinstance(device, int):
            self._device_idx = device
            self._device_str = f'cuda:{device}'
            self._use_gpu = True
        elif device == 'cpu':
            self._device_idx = None
            self._device_str = 'cpu'
            self._use_gpu = False
        elif device.startswith('cuda'):
            if ':' in device:
                self._device_idx = int(device.split(':')[1])
            else:
                self._device_idx = 0
            self._device_str = f'cuda:{self._device_idx}'
            self._use_gpu = True
        else:
            raise ValueError(f"Invalid device: {device}")

        # Select decoder
        if self._use_gpu and check_gpu_decoder_available():
            self._decoder = GPUDecoder(
                device=self._device_idx,
                max_batch_size=max_batch_size,
            )
            self._decoder_type = 'gpu'
        elif self._use_gpu and torch.cuda.is_available():
            self._decoder = GPUDecoderFallback(
                device=self._device_idx,
                num_workers=num_workers,
            )
            self._decoder_type = 'gpu_fallback'
        else:
            self._decoder = CPUDecoder(num_workers=num_workers)
            self._decoder_type = 'cpu'

    def __call__(self, batch_data: dict[str, Any]) -> torch.Tensor:
        """Decode and center-crop batch of JPEG bytes."""
        data = batch_data['data']
        sizes = batch_data['sizes']
        heights = batch_data['heights']
        widths = batch_data['widths']

        if self._decoder_type in ('gpu', 'gpu_fallback'):
            images = self._decoder.decode_batch_center_crop(
                data, sizes, heights, widths,
                crop_size=self.size,
            )
        else:
            images_list = self._decoder.decode_batch_center_crop(
                data, sizes, heights, widths,
                crop_size=self.size,
            )
            images = self._stack_and_resize(images_list)

        return images

    def _stack_and_resize(self, images_list: list[np.ndarray]) -> torch.Tensor:
        """Stack CPU-decoded images, resizing if needed."""
        from torch.nn import functional as F

        batch_size = len(images_list)
        output = torch.zeros(
            (batch_size, 3, self.size, self.size),
            dtype=torch.uint8,
            device=self._device_str,
        )

        for i, img_hwc in enumerate(images_list):
            tensor = torch.from_numpy(img_hwc).permute(2, 0, 1)
            h, w = img_hwc.shape[:2]

            if h != self.size or w != self.size:
                tensor = F.interpolate(
                    tensor.unsqueeze(0).float(),
                    size=(self.size, self.size),
                    mode='bilinear',
                    align_corners=False,
                ).squeeze(0).to(torch.uint8)

            output[i] = tensor.to(self._device_str)

        return output

    def shutdown(self) -> None:
        if hasattr(self._decoder, 'shutdown'):
            self._decoder.shutdown()

    def __repr__(self) -> str:
        return f"CenterCrop(size={self.size}, device='{self._device_str}')"


class ResizeCenterCrop(BatchTransform):
    """Resize shortest edge then center crop - standard validation transform.

    This is the standard ImageNet validation transform:
    1. Resize image so shortest edge = resize_size
    2. Center crop to crop_size x crop_size

    Args:
        resize_size: Target size for shortest edge (default 256)
        crop_size: Final crop size (default 224)
        device: Device for output tensors
        num_workers: CPU workers for decoding

    Example:
        rcc = ResizeCenterCrop(256, 224, device='cuda')
    """

    def __init__(
        self,
        resize_size: int = 256,
        crop_size: int = 224,
        device: str | int = 'cuda',
        num_workers: int = 8,
        max_batch_size: int = 256,
    ) -> None:
        self.resize_size = resize_size
        self.crop_size = crop_size
        self.num_workers = num_workers
        self.max_batch_size = max_batch_size

        # Parse device
        if isinstance(device, int):
            self._device_idx = device
            self._device_str = f'cuda:{device}'
            self._use_gpu = True
        elif device == 'cpu':
            self._device_idx = None
            self._device_str = 'cpu'
            self._use_gpu = False
        elif device.startswith('cuda'):
            if ':' in device:
                self._device_idx = int(device.split(':')[1])
            else:
                self._device_idx = 0
            self._device_str = f'cuda:{self._device_idx}'
            self._use_gpu = True
        else:
            raise ValueError(f"Invalid device: {device}")

        # Select decoder
        if self._use_gpu and check_gpu_decoder_available():
            self._decoder = GPUDecoder(
                device=self._device_idx,
                max_batch_size=max_batch_size,
            )
            self._decoder_type = 'gpu'
        elif self._use_gpu and torch.cuda.is_available():
            self._decoder = GPUDecoderFallback(
                device=self._device_idx,
                num_workers=num_workers,
            )
            self._decoder_type = 'gpu_fallback'
        else:
            self._decoder = CPUDecoder(num_workers=num_workers)
            self._decoder_type = 'cpu'

    def __call__(self, batch_data: dict[str, Any]) -> torch.Tensor:
        """Decode, resize shortest edge, then center crop."""
        from torch.nn import functional as F

        data = batch_data['data']
        sizes = batch_data['sizes']
        heights = batch_data['heights']
        widths = batch_data['widths']

        batch_size = len(sizes)

        # Decode full images first
        if self._decoder_type in ('gpu', 'gpu_fallback'):
            # GPU path: decode full, then resize+crop
            images = self._decoder.decode_batch(data, sizes)
            # images is tensor [B, C, H, W] with varying sizes - need per-image processing
            # For now, fall back to CPU-style per-image processing
            images_list = [images[i] for i in range(batch_size)]
        else:
            images_list = self._decoder.decode_batch(data, sizes)

        # Process each image: resize shortest edge, then center crop
        output = torch.zeros(
            (batch_size, 3, self.crop_size, self.crop_size),
            dtype=torch.float32,
            device=self._device_str,
        )

        for i, img in enumerate(images_list):
            if isinstance(img, np.ndarray):
                # CPU decoder returns HWC numpy
                tensor = torch.from_numpy(img).permute(2, 0, 1).float()
            else:
                # GPU decoder returns CHW tensor
                tensor = img.float()

            _, h, w = tensor.shape

            # Resize so shortest edge = resize_size
            if h < w:
                new_h = self.resize_size
                new_w = int(w * self.resize_size / h)
            else:
                new_w = self.resize_size
                new_h = int(h * self.resize_size / w)

            tensor = F.interpolate(
                tensor.unsqueeze(0),
                size=(new_h, new_w),
                mode='bilinear',
                align_corners=False,
            ).squeeze(0)

            # Center crop
            start_h = (new_h - self.crop_size) // 2
            start_w = (new_w - self.crop_size) // 2
            cropped = tensor[:, start_h:start_h + self.crop_size, start_w:start_w + self.crop_size]

            output[i] = cropped.to(self._device_str)

        return output.to(torch.uint8)

    def shutdown(self) -> None:
        if hasattr(self._decoder, 'shutdown'):
            self._decoder.shutdown()

    def __repr__(self) -> str:
        return f"ResizeCenterCrop(resize={self.resize_size}, crop={self.crop_size}, device='{self._device_str}')"


class Normalize(BatchTransform):
    """ImageNet-style normalization for batch tensors.

    Converts uint8 [0, 255] to float32 and applies (x - mean) / std.

    Args:
        mean: Per-channel mean (default: ImageNet)
        std: Per-channel std (default: ImageNet)
        device: Device for normalization tensors

    Example:
        norm = Normalize()  # ImageNet defaults
        norm = Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    """

    def __init__(
        self,
        mean: tuple[float, float, float] = IMAGENET_MEAN,
        std: tuple[float, float, float] = IMAGENET_STD,
        device: str | int = 'cuda',
    ) -> None:
        self.mean = mean
        self.std = std

        # Parse device
        if isinstance(device, int):
            self._device_str = f'cuda:{device}'
        elif device == 'cpu':
            self._device_str = 'cpu'
        elif device.startswith('cuda'):
            self._device_str = device if ':' in device else 'cuda:0'
        else:
            raise ValueError(f"Invalid device: {device}")

        # Pre-compute tensors
        self._mean_tensor = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
        self._std_tensor = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
        self._tensors_on_device = False

    def _ensure_device(self, device: str) -> None:
        """Move tensors to device if needed."""
        if not self._tensors_on_device or self._device_str != device:
            self._mean_tensor = self._mean_tensor.to(device)
            self._std_tensor = self._std_tensor.to(device)
            self._device_str = device
            self._tensors_on_device = True

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """Normalize batch of images.

        Args:
            images: Tensor [B, C, H, W] uint8 or float

        Returns:
            Normalized tensor [B, C, H, W] float32
        """
        self._ensure_device(str(images.device))

        # Convert to float and scale to [0, 1]
        if images.dtype == torch.uint8:
            images = images.float() / 255.0
        elif images.dtype != torch.float32:
            images = images.float()

        # Normalize
        images = (images - self._mean_tensor) / self._std_tensor

        return images

    def __repr__(self) -> str:
        return f"Normalize(mean={self.mean}, std={self.std})"


class ToDevice(BatchTransform):
    """Move tensor to specified device.

    Args:
        device: Target device

    Example:
        to_gpu = ToDevice('cuda:0')
    """

    def __init__(self, device: str | int = 'cuda') -> None:
        if isinstance(device, int):
            self._device_str = f'cuda:{device}'
        else:
            self._device_str = device

    def __call__(self, data: torch.Tensor | np.ndarray) -> torch.Tensor:
        """Move data to device."""
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        return data.to(self._device_str)

    def __repr__(self) -> str:
        return f"ToDevice('{self._device_str}')"


class Compose(BatchTransform):
    """Compose multiple transforms into a single pipeline.

    Args:
        transforms: List of transforms to apply in order

    Example:
        pipeline = Compose([
            RandomResizedCrop(224, device='cuda'),
            Normalize(),
        ])
    """

    def __init__(self, transforms: list[BatchTransform]) -> None:
        self.transforms = transforms

    def __call__(self, data: Any) -> Any:
        """Apply all transforms in sequence."""
        for t in self.transforms:
            data = t(data)
        return data

    def shutdown(self) -> None:
        """Shutdown all transforms."""
        for t in self.transforms:
            if hasattr(t, 'shutdown'):
                t.shutdown()

    def __repr__(self) -> str:
        lines = [f"  {t}" for t in self.transforms]
        return "Compose([\n" + ",\n".join(lines) + "\n])"


# Convenience aliases for common training/val pipelines
def make_train_pipeline(
    size: int = 224,
    scale: tuple[float, float] = (0.08, 1.0),
    ratio: tuple[float, float] = (3/4, 4/3),
    normalize: bool = True,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
    device: str | int = 'cuda',
    num_workers: int = 8,
) -> Compose:
    """Create a standard ImageNet training pipeline.

    Args:
        size: Output image size
        scale: RandomResizedCrop scale range
        ratio: RandomResizedCrop aspect ratio range
        normalize: Whether to apply ImageNet normalization
        mean: Normalization mean
        std: Normalization std
        device: Output device
        num_workers: CPU workers for decoding

    Returns:
        Compose pipeline for training
    """
    transforms = [
        RandomResizedCrop(size, scale, ratio, device, num_workers),
    ]
    if normalize:
        transforms.append(Normalize(mean, std, device))
    return Compose(transforms)


def make_val_pipeline(
    size: int = 224,
    normalize: bool = True,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
    device: str | int = 'cuda',
    num_workers: int = 8,
) -> Compose:
    """Create a standard ImageNet validation pipeline.

    Args:
        size: Output image size (center crop)
        normalize: Whether to apply ImageNet normalization
        mean: Normalization mean
        std: Normalization std
        device: Output device
        num_workers: CPU workers for decoding

    Returns:
        Compose pipeline for validation
    """
    transforms = [
        CenterCrop(size, device, num_workers),
    ]
    if normalize:
        transforms.append(Normalize(mean, std, device))
    return Compose(transforms)


__all__ = [
    # Base
    "BatchTransform",
    "Compose",
    # Transforms
    "Decoder",
    "RandomResizedCrop",
    "CenterCrop",
    "ResizeCenterCrop",
    "Normalize",
    "ToDevice",
    # Convenience
    "make_train_pipeline",
    "make_val_pipeline",
    # Constants
    "IMAGENET_MEAN",
    "IMAGENET_STD",
]
