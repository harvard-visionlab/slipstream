"""GPU JPEG decoder using nvImageCodec + CV-CUDA.

This module provides high-performance GPU-accelerated JPEG decoding using
NVIDIA's nvImageCodec library, with optional CV-CUDA resize integration.

Key features:
1. **Zero-copy GPU output**: Decoded images go directly to CUDA tensors
2. **ROI decode**: Partial decode for RandomResizedCrop (decode MCU superset)
3. **CV-CUDA resize**: GPU-accelerated resize after decode

Usage:
    from slipstream.decoders import GPUDecoder, check_gpu_decoder_available

    if check_gpu_decoder_available():
        decoder = GPUDecoder(device=0)

        # Full decode to GPU tensor
        images = decoder.decode_batch(data, sizes, heights, widths)

        # ROI decode + resize (for RandomResizedCrop)
        images = decoder.decode_batch_random_crop(
            data, sizes, heights, widths,
            target_size=224,
            scale=(0.08, 1.0),
            ratio=(3/4, 4/3),
        )

Performance targets:
    - GPU decode: 30-50k+ images/sec (vs 17k CPU decode)
    - Zero-copy: 0ms CPU→GPU transfer (vs ~1ms/batch)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from slipstream.utils.crop import (
    generate_batch_center_crop_params,
    generate_batch_random_crop_params,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Optional imports for GPU decoding
_NVIMGCODEC_AVAILABLE = False
_CVCUDA_AVAILABLE = False

try:
    import nvidia.nvimgcodec as nvimgcodec
    _NVIMGCODEC_AVAILABLE = True
except ImportError:
    nvimgcodec = None

try:
    import cvcuda
    _CVCUDA_AVAILABLE = True
except ImportError:
    cvcuda = None
    _CVCUDA_AVAILABLE = False


def check_gpu_decoder_available() -> bool:
    """Check if GPU decoder dependencies are available.

    Returns:
        True if nvImageCodec is available and CUDA is accessible
    """
    if not _NVIMGCODEC_AVAILABLE:
        return False
    if not torch.cuda.is_available():
        return False
    return True


def check_cvcuda_available() -> bool:
    """Check if CV-CUDA is available for GPU resize.

    Returns:
        True if CV-CUDA is available
    """
    return _CVCUDA_AVAILABLE


@dataclass
class ROIParams:
    """Region of Interest parameters for partial decode.

    Attributes:
        x: Left coordinate
        y: Top coordinate
        width: ROI width
        height: ROI height
    """

    x: int
    y: int
    width: int
    height: int


class GPUDecoder:
    """GPU JPEG batch decoder using nvImageCodec.

    This decoder achieves significantly higher throughput than CPU decoders by:
    1. Decoding directly on the GPU
    2. Zero-copy output to PyTorch CUDA tensors
    3. Optional ROI decode for RandomResizedCrop acceleration
    4. Optional CV-CUDA integration for fused decode+resize

    Attributes:
        device: CUDA device index
        use_cvcuda_resize: Whether to use CV-CUDA for resize
        max_batch_size: Maximum batch size for pre-allocated buffers

    Example:
        decoder = GPUDecoder(device=0)

        # Decode batch of JPEGs
        images = decoder.decode_batch(data, sizes, heights, widths)

        # With RandomResizedCrop during decode
        images = decoder.decode_batch_random_crop(
            data, sizes, heights, widths,
            target_size=224,
            scale=(0.08, 1.0),
        )
    """

    def __init__(
        self,
        device: int = 0,
        use_cvcuda_resize: bool = True,
        max_batch_size: int = 256,
    ) -> None:
        """Initialize the GPU batch decoder.

        Args:
            device: CUDA device index
            use_cvcuda_resize: Use CV-CUDA for GPU resize (if available)
            max_batch_size: Maximum batch size for pre-allocated buffers

        Raises:
            RuntimeError: If nvImageCodec is not available
        """
        if not check_gpu_decoder_available():
            raise RuntimeError(
                "GPU decoder not available. Install nvidia-nvimgcodec-cu12:\n"
                "  pip install nvidia-nvimgcodec-cu12\n"
                "Or use: uv sync --group gpu"
            )

        self.device = device
        self.use_cvcuda_resize = use_cvcuda_resize and check_cvcuda_available()
        self.max_batch_size = max_batch_size

        # Initialize nvImageCodec decoder
        self._decoder = nvimgcodec.Decoder(device_id=device)

        # CUDA stream for async operations
        self._stream = torch.cuda.Stream(device=device)

        # Pre-allocated pinned memory buffer for staging
        self._staging_buffer: torch.Tensor | None = None
        self._staging_buffer_size = 0

    def _pack_jpeg_bytes(
        self,
        data: NDArray[np.uint8],
        sizes: NDArray[np.int64 | np.uint64 | np.uint32],
    ) -> list[bytes]:
        """Pack JPEG bytes from padded array into list.

        Args:
            data: Padded JPEG data [B, max_size]
            sizes: Actual size of each JPEG [B]

        Returns:
            List of JPEG bytes
        """
        return [bytes(data[i, :int(sizes[i])]) for i in range(len(sizes))]

    def decode_batch(
        self,
        data: NDArray[np.uint8],
        sizes: NDArray[np.int64 | np.uint64 | np.uint32],
        heights: NDArray[np.uint32] | None = None,
        widths: NDArray[np.uint32] | None = None,
    ) -> torch.Tensor:
        """Decode batch of JPEGs to GPU tensor.

        Args:
            data: Padded JPEG data [B, max_size] uint8
            sizes: Actual JPEG sizes [B]
            heights: Pre-stored heights [B] (optional, for buffer allocation)
            widths: Pre-stored widths [B] (optional, for buffer allocation)

        Returns:
            Decoded images as CUDA tensor [B, C, H, W] uint8

        Note:
            If images have different sizes, the output will be padded to
            the maximum dimensions in the batch. Use decode_batch_random_crop
            for uniform output sizes.
        """
        batch_size = len(sizes)
        if batch_size == 0:
            return torch.empty(
                (0, 3, 0, 0), dtype=torch.uint8, device=f"cuda:{self.device}"
            )

        # Pack JPEG bytes
        jpeg_list = self._pack_jpeg_bytes(data, sizes)

        with torch.cuda.stream(self._stream):
            # nvImageCodec decode returns list of GPU images
            images = self._decoder.decode(jpeg_list)

            # Convert to torch tensors and find max dimensions
            tensors: list[torch.Tensor] = []
            max_h, max_w = 0, 0

            for img in images:
                # Get tensor view of GPU image data (nvimgcodec returns HWC)
                tensor = torch.as_tensor(img, device=f"cuda:{self.device}")
                if tensor.ndim == 3:  # HWC
                    h, w, _c = tensor.shape
                    tensor = tensor.permute(2, 0, 1)  # Convert to CHW
                else:
                    _c, h, w = tensor.shape

                max_h = max(max_h, h)
                max_w = max(max_w, w)
                tensors.append(tensor)

            # Stack into batch tensor (pad if needed)
            if all(t.shape == tensors[0].shape for t in tensors):
                output = torch.stack(tensors)
            else:
                output = torch.zeros(
                    (batch_size, 3, max_h, max_w),
                    dtype=torch.uint8,
                    device=f"cuda:{self.device}",
                )
                for i, tensor in enumerate(tensors):
                    c, h, w = tensor.shape
                    output[i, :c, :h, :w] = tensor

        self._stream.synchronize()
        return output

    def decode_batch_with_roi_native(
        self,
        data: NDArray[np.uint8],
        sizes: NDArray[np.int64 | np.uint64 | np.uint32],
        rois: NDArray[np.int32],
        target_size: tuple[int, int],
    ) -> torch.Tensor:
        """Decode batch with native ROI decode (decode only the crop region).

        Uses nvImageCodec's Region API to decode only the ROI, avoiding
        decoding pixels that will be discarded. This is significantly faster
        than full decode + crop for large images.

        Args:
            data: Padded JPEG data [B, max_size]
            sizes: JPEG sizes [B]
            rois: ROI parameters [B, 4] as (x, y, w, h)
            target_size: Final output size (height, width)

        Returns:
            Cropped and resized images [B, C, H, W] on GPU
        """
        batch_size = len(sizes)
        th, tw = target_size
        if batch_size == 0:
            return torch.empty(
                (0, 3, th, tw), dtype=torch.uint8, device=f"cuda:{self.device}"
            )

        # Create code streams with regions for ROI decode
        jpeg_list = self._pack_jpeg_bytes(data, sizes)

        # Create regions for each image
        decode_sources = []
        for i in range(batch_size):
            x, y, w, h = rois[i]
            # nvimgcodec.Region uses (start_y, start_x, end_y, end_x)
            region = nvimgcodec.Region(
                start_y=int(y),
                start_x=int(x),
                end_y=int(y + h),
                end_x=int(x + w),
            )
            # Create code stream from bytes
            code_stream = nvimgcodec.CodeStream(jpeg_list[i])
            # Get sub-stream for the region
            view = nvimgcodec.CodeStreamView(image_idx=0, region=region)
            sub_stream = code_stream.get_sub_code_stream(view)
            decode_sources.append(sub_stream)

        with torch.cuda.stream(self._stream):
            # Decode only the ROI regions
            images = self._decoder.decode(decode_sources)

            # Convert to tensors - all should be similar size now (the ROI size)
            cropped_tensors: list[torch.Tensor] = []
            for img in images:
                tensor = torch.as_tensor(img, device=f"cuda:{self.device}")
                if tensor.ndim == 3:  # HWC
                    tensor = tensor.permute(2, 0, 1)  # Convert to CHW
                cropped_tensors.append(tensor)

        self._stream.synchronize()

        # Resize all crops to target size
        return self._resize_batch(cropped_tensors, (th, tw))

    def decode_batch_with_roi(
        self,
        data: NDArray[np.uint8],
        sizes: NDArray[np.int64 | np.uint64 | np.uint32],
        heights: NDArray[np.uint32],
        widths: NDArray[np.uint32],
        rois: NDArray[np.int32],
        target_size: tuple[int, int],
        use_native_roi: bool = True,
    ) -> torch.Tensor:
        """Decode batch with ROI (Region of Interest) for each image.

        Uses native ROI decode when available, falls back to full decode + crop.

        Args:
            data: Padded JPEG data [B, max_size]
            sizes: JPEG sizes [B]
            heights: Image heights [B]
            widths: Image widths [B]
            rois: ROI parameters [B, 4] as (x, y, w, h)
            target_size: Final output size (height, width)
            use_native_roi: Try native ROI decode first (faster for large images)

        Returns:
            Cropped and resized images [B, C, H, W] on GPU
        """
        batch_size = len(sizes)
        th, tw = target_size
        if batch_size == 0:
            return torch.empty(
                (0, 3, th, tw), dtype=torch.uint8, device=f"cuda:{self.device}"
            )

        # Try native ROI decode first (decodes only the crop region)
        if use_native_roi:
            try:
                return self.decode_batch_with_roi_native(data, sizes, rois, target_size)
            except Exception as e:
                # Fall back to full decode + crop if native ROI fails
                import warnings
                warnings.warn(f"Native ROI decode failed, falling back to full decode: {e}")

        # Fallback: Full decode then crop
        full_images = self.decode_batch(data, sizes, heights, widths)

        # Crop each image on GPU (fast indexing)
        cropped_tensors: list[torch.Tensor] = []
        for i in range(batch_size):
            x, y, w, h = rois[i]
            cropped = full_images[i, :, y:y+h, x:x+w].contiguous()
            cropped_tensors.append(cropped)

        # Batch resize all crops at once
        return self._resize_batch(cropped_tensors, (th, tw))

    def _resize_single(
        self,
        tensor: torch.Tensor,
        target_size: tuple[int, int],
    ) -> torch.Tensor:
        """Resize a single CHW tensor to target size.

        Args:
            tensor: Input tensor [C, H, W]
            target_size: Target (height, width)

        Returns:
            Resized tensor [C, H, W]
        """
        from torch.nn import functional as fn

        th, tw = target_size
        t_float = tensor.unsqueeze(0).float()
        resized = fn.interpolate(
            t_float,
            size=(th, tw),
            mode="bilinear",
            align_corners=False,
        )
        return resized.squeeze(0).to(torch.uint8)

    def _resize_batch(
        self,
        tensors: list[torch.Tensor],
        target_size: tuple[int, int],
    ) -> torch.Tensor:
        """Resize batch of tensors to target size.

        Uses CV-CUDA if available for GPU-accelerated resize,
        otherwise falls back to torch resize.

        Args:
            tensors: List of CHW tensors (can have different sizes)
            target_size: Target (height, width)

        Returns:
            Resized tensors [B, C, H, W]
        """
        batch_size = len(tensors)
        th, tw = target_size

        if self.use_cvcuda_resize and cvcuda is not None:
            # CV-CUDA resize path
            # API: cvcuda.resize(src, shape, interp) -> returns new cvcuda.Tensor
            output = torch.zeros(
                (batch_size, 3, th, tw),
                dtype=torch.uint8,
                device=f"cuda:{self.device}",
            )

            for i, tensor in enumerate(tensors):
                # Convert CHW to HWC for CV-CUDA
                hwc = tensor.permute(1, 2, 0).contiguous()

                # Create cvcuda tensor wrapper
                src_tensor = cvcuda.as_tensor(hwc, "HWC")

                # Resize using CV-CUDA - returns new cvcuda.Tensor
                resized_cvcuda = cvcuda.resize(
                    src_tensor, (th, tw, 3), cvcuda.Interp.LINEAR
                )

                # Convert cvcuda.Tensor back to torch
                # Use DLPack protocol (standard inter-framework tensor exchange)
                dst_hwc = torch.from_dlpack(resized_cvcuda)
                output[i] = dst_hwc.permute(2, 0, 1)

            return output
        else:
            # PyTorch resize fallback
            from torch.nn import functional as fn

            output = torch.zeros(
                (batch_size, 3, th, tw),
                dtype=torch.uint8,
                device=f"cuda:{self.device}",
            )

            for i, tensor in enumerate(tensors):
                t_float = tensor.unsqueeze(0).float()
                resized = fn.interpolate(
                    t_float,
                    size=(th, tw),
                    mode="bilinear",
                    align_corners=False,
                )
                output[i] = resized.squeeze(0).to(torch.uint8)

            return output

    def decode_batch_random_crop(
        self,
        data: NDArray[np.uint8],
        sizes: NDArray[np.int64 | np.uint64 | np.uint32],
        heights: NDArray[np.uint32],
        widths: NDArray[np.uint32],
        target_size: int = 224,
        scale: tuple[float, float] = (0.08, 1.0),
        ratio: tuple[float, float] = (3 / 4, 4 / 3),
    ) -> torch.Tensor:
        """Decode batch with RandomResizedCrop during decode.

        This combines ROI decode with GPU resize for training acceleration.
        Crops are generated using the same algorithm as torchvision's
        RandomResizedCrop.

        Args:
            data: Padded JPEG data [B, max_size]
            sizes: JPEG sizes [B]
            heights: Image heights [B]
            widths: Image widths [B]
            target_size: Final crop size (square)
            scale: Scale range relative to original
            ratio: Aspect ratio range

        Returns:
            Cropped and resized images [B, 3, target_size, target_size] on GPU
        """
        batch_size = len(sizes)
        if batch_size == 0:
            return torch.empty(
                (0, 3, target_size, target_size),
                dtype=torch.uint8,
                device=f"cuda:{self.device}",
            )

        # Generate random crop params using unified utility
        rois = generate_batch_random_crop_params(
            widths, heights, scale=scale, ratio=ratio
        )

        return self.decode_batch_with_roi(
            data, sizes, heights, widths, rois, (target_size, target_size)
        )

    def decode_batch_center_crop(
        self,
        data: NDArray[np.uint8],
        sizes: NDArray[np.int64 | np.uint64 | np.uint32],
        heights: NDArray[np.uint32],
        widths: NDArray[np.uint32],
        crop_size: int = 224,
        resize_first: int | None = 256,
    ) -> torch.Tensor:
        """Decode batch with center crop (for validation).

        Args:
            data: Padded JPEG data [B, max_size]
            sizes: JPEG sizes [B]
            heights: Image heights [B]
            widths: Image widths [B]
            crop_size: Center crop size
            resize_first: Resize to this size before center crop (None to skip)

        Returns:
            Center-cropped images [B, 3, crop_size, crop_size] on GPU
        """
        batch_size = len(sizes)
        if batch_size == 0:
            return torch.empty(
                (0, 3, crop_size, crop_size),
                dtype=torch.uint8,
                device=f"cuda:{self.device}",
            )

        # Generate center crop params using unified utility
        rois = generate_batch_center_crop_params(widths, heights, crop_size=crop_size)

        return self.decode_batch_with_roi(
            data, sizes, heights, widths, rois, (crop_size, crop_size)
        )

    def shutdown(self) -> None:
        """Release resources."""
        self._decoder = None
        self._staging_buffer = None
        self._stream = None

    def __del__(self) -> None:
        """Cleanup on deletion."""
        self.shutdown()

    def __repr__(self) -> str:
        return (
            f"GPUDecoder("
            f"device={self.device}, "
            f"use_cvcuda_resize={self.use_cvcuda_resize})"
        )


class GPUDecoderFallback:
    """Fallback decoder when nvImageCodec is not available.

    This decoder uses the CPU TurboJPEG decoder and transfers to GPU,
    providing API compatibility without the GPU acceleration benefits.

    Use this for development/testing on machines without nvImageCodec.
    """

    def __init__(
        self,
        device: int = 0,
        num_workers: int = 8,
    ) -> None:
        """Initialize fallback decoder.

        Args:
            device: CUDA device index
            num_workers: Number of CPU decode workers
        """
        from slipstream.decoders.cpu import CPUDecoder

        self.device = device
        self._cpu_decoder = CPUDecoder(num_workers=num_workers)

    def decode_batch(
        self,
        data: NDArray[np.uint8],
        sizes: NDArray[np.int64 | np.uint64 | np.uint32],
        heights: NDArray[np.uint32] | None = None,
        widths: NDArray[np.uint32] | None = None,
    ) -> torch.Tensor:
        """Decode batch and transfer to GPU.

        Args:
            data: Padded JPEG data [B, max_size]
            sizes: JPEG sizes [B]
            heights: Ignored
            widths: Ignored

        Returns:
            Images as CUDA tensor [B, C, H, W]
        """
        images_hwc = self._cpu_decoder.decode_batch(data, sizes)

        if len(images_hwc) == 0:
            return torch.empty(
                (0, 3, 0, 0), dtype=torch.uint8, device=f"cuda:{self.device}"
            )

        # Find max dimensions
        max_h = max(img.shape[0] for img in images_hwc)
        max_w = max(img.shape[1] for img in images_hwc)

        # Convert to tensors
        batch_size = len(images_hwc)
        if all(img.shape[:2] == images_hwc[0].shape[:2] for img in images_hwc):
            # All same size - simple stack
            tensors = [torch.from_numpy(img).permute(2, 0, 1) for img in images_hwc]
            output = torch.stack(tensors).to(f"cuda:{self.device}")
        else:
            # Different sizes - pad
            output = torch.zeros(
                (batch_size, 3, max_h, max_w),
                dtype=torch.uint8,
                device=f"cuda:{self.device}",
            )
            for i, img in enumerate(images_hwc):
                h, w, _c = img.shape
                tensor = torch.from_numpy(img).permute(2, 0, 1)
                output[i, :, :h, :w] = tensor.to(f"cuda:{self.device}")

        return output

    def decode_batch_random_crop(
        self,
        data: NDArray[np.uint8],
        sizes: NDArray[np.int64 | np.uint64 | np.uint32],
        heights: NDArray[np.uint32],
        widths: NDArray[np.uint32],
        target_size: int = 224,
        scale: tuple[float, float] = (0.08, 1.0),
        ratio: tuple[float, float] = (3 / 4, 4 / 3),
    ) -> torch.Tensor:
        """Decode with random crop using CPU decoder + GPU resize.

        Args:
            data: Padded JPEG data [B, max_size]
            sizes: JPEG sizes [B]
            heights: Image heights [B]
            widths: Image widths [B]
            target_size: Final crop size
            scale: Scale range
            ratio: Aspect ratio range

        Returns:
            Cropped images [B, 3, target_size, target_size] on GPU
        """
        from torch.nn import functional as fn

        images_hwc = self._cpu_decoder.decode_batch_random_crop(
            data, sizes, heights, widths,
            target_size=target_size, scale=scale, ratio=ratio
        )

        if len(images_hwc) == 0:
            return torch.empty(
                (0, 3, target_size, target_size),
                dtype=torch.uint8,
                device=f"cuda:{self.device}",
            )

        # Resize and transfer to GPU
        batch_size = len(images_hwc)
        output = torch.zeros(
            (batch_size, 3, target_size, target_size),
            dtype=torch.uint8,
            device=f"cuda:{self.device}",
        )

        for i, img_hwc in enumerate(images_hwc):
            tensor = torch.from_numpy(img_hwc).permute(2, 0, 1).unsqueeze(0).float()
            resized = fn.interpolate(
                tensor, size=(target_size, target_size),
                mode="bilinear", align_corners=False
            )
            output[i] = resized.squeeze(0).to(torch.uint8).to(f"cuda:{self.device}")

        return output

    def decode_batch_center_crop(
        self,
        data: NDArray[np.uint8],
        sizes: NDArray[np.int64 | np.uint64 | np.uint32],
        heights: NDArray[np.uint32],
        widths: NDArray[np.uint32],
        crop_size: int = 224,
        resize_first: int | None = 256,
    ) -> torch.Tensor:
        """Decode with center crop using CPU decoder.

        Args:
            data: Padded JPEG data [B, max_size]
            sizes: JPEG sizes [B]
            heights: Image heights [B]
            widths: Image widths [B]
            crop_size: Center crop size
            resize_first: Ignored in fallback

        Returns:
            Center-cropped images on GPU
        """
        from torch.nn import functional as fn

        images_hwc = self._cpu_decoder.decode_batch_center_crop(
            data, sizes, heights, widths, crop_size=crop_size
        )

        if len(images_hwc) == 0:
            return torch.empty(
                (0, 3, crop_size, crop_size),
                dtype=torch.uint8,
                device=f"cuda:{self.device}",
            )

        # Resize and transfer to GPU
        batch_size = len(images_hwc)
        output = torch.zeros(
            (batch_size, 3, crop_size, crop_size),
            dtype=torch.uint8,
            device=f"cuda:{self.device}",
        )

        for i, img_hwc in enumerate(images_hwc):
            tensor = torch.from_numpy(img_hwc).permute(2, 0, 1).unsqueeze(0).float()
            resized = fn.interpolate(
                tensor, size=(crop_size, crop_size),
                mode="bilinear", align_corners=False
            )
            output[i] = resized.squeeze(0).to(torch.uint8).to(f"cuda:{self.device}")

        return output

    def shutdown(self) -> None:
        """Release resources."""
        self._cpu_decoder.shutdown()

    def __repr__(self) -> str:
        return f"GPUDecoderFallback(device={self.device})"


def get_decoder(
    device: int = 0,
    prefer_gpu: bool = True,
    fallback_workers: int = 8,
) -> GPUDecoder | GPUDecoderFallback:
    """Get the best available GPU decoder.

    Returns GPUDecoder if nvImageCodec is available,
    otherwise returns GPUDecoderFallback which uses
    CPU decode + GPU transfer.

    Args:
        device: CUDA device index
        prefer_gpu: If True, prefer GPU decoder when available
        fallback_workers: Number of CPU workers for fallback

    Returns:
        GPU decoder instance (real or fallback)
    """
    if prefer_gpu and check_gpu_decoder_available():
        return GPUDecoder(device=device)
    else:
        return GPUDecoderFallback(device=device, num_workers=fallback_workers)


# Convenience aliases
NvImageCodecBatchDecoder = GPUDecoder
NvImageCodecBatchDecoderFallback = GPUDecoderFallback

__all__ = [
    "GPUDecoder",
    "GPUDecoderFallback",
    "NvImageCodecBatchDecoder",
    "NvImageCodecBatchDecoderFallback",
    "ROIParams",
    "check_gpu_decoder_available",
    "check_cvcuda_available",
    "get_decoder",
]
