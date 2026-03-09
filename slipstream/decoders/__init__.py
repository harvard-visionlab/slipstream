"""Decoders for slipstream.

Low-level decoders:
- CPUDecoder: TurboJPEG for parallel JPEG decoding
- GPUDecoder: nvImageCodec for GPU JPEG decoding
- NumbaBatchDecoder: Numba JIT-compiled batch decoder (FFCV-style, fastest)
- YUV420NumbaBatchDecoder: YUV420P raw format decoder (~2x JPEG throughput)

Fused decode+crop stages (for SlipstreamLoader pipelines):
- DecodeOnly, DecodeYUVFullRes, DecodeYUVPlanes: pure decode
- DecodeCenterCrop, DecodeRandomResizedCrop, DecodeDirectRandomResizedCrop, DecodeResizeCrop, DecodeRandomResizeShortCropLong: decode → RGB + crop
- DecodeYUVCenterCrop, DecodeYUVRandomResizedCrop, DecodeYUVResizeCrop: decode → YUV + crop (keeps colorspace)
- DecodeMultiRandomResizedCrop, DecodeMultiRandomResizeShortCropLong, DecodeUniformMultiRandomResizedCrop, MultiCropPipeline: multi-crop

Usage:
    from slipstream.decoders import CPUDecoder, GPUDecoder, get_decoder
    from slipstream.decoders import DecodeRandomResizedCrop, DecodeCenterCrop
    from slipstream.decoders import DecodeYUVRandomResizedCrop  # keeps YUV colorspace
"""

from slipstream.decoders.base import BatchTransform

from slipstream.decoders.cpu import (
    CPUDecoder,
    TurboJPEGBatchDecoder,
    check_turbojpeg_available,
)
from slipstream.decoders.gpu import (
    GPUDecoder,
    GPUDecoderFallback,
    NvImageCodecBatchDecoder,
    NvImageCodecBatchDecoderFallback,
    ROIParams,
    check_cvcuda_available,
    check_gpu_decoder_available,
    get_decoder,
)

# Import Numba decoder (optional - requires building libslipstream)
try:
    from slipstream.decoders.numba_decoder import (
        NumbaBatchDecoder,
        check_numba_decoder_available,
    )
except ImportError:
    NumbaBatchDecoder = None  # type: ignore
    def check_numba_decoder_available() -> bool:
        return False

# Fused decode+crop stages
from slipstream.decoders.decode import (
    DecodeOnly,
    DecodeYUVFullRes,
    DecodeYUVPlanes,
    DecodeYUVCenterCrop,
    DecodeYUVRandomResizedCrop,
    DecodeYUVResizeCrop,
)
from slipstream.decoders.crop import (
    DecodeCenterCrop,
    DecodeDirectRandomResizedCrop,
    DecodeRandomResizedCrop,
    DecodeRandomResizeShortCropLong,
    DecodeResizeCrop,
    # Backward-compatible aliases
    CenterCrop,
    DirectRandomResizedCrop,
    RandomResizedCrop,
    RandomResizeShortCropLong,
    ResizeCrop,
)
from slipstream.decoders.multicrop import (
    DecodeMultiRandomResizedCrop,
    DecodeMultiRandomResizeShortCropLong,
    DecodeUniformMultiRandomResizedCrop,
    MultiCropPipeline,
    NamedCopies,
    # Backward-compatible aliases
    MultiCropRandomResizedCrop,
    MultiRandomResizedCrop,
    MultiRandomResizeShortCropLong,
)
from slipstream.decoders.utils import estimate_rejection_fallback_rate

__all__ = [
    # Base
    "BatchTransform",
    # CPU decoder
    "CPUDecoder",
    "TurboJPEGBatchDecoder",
    "check_turbojpeg_available",
    # GPU decoder
    "GPUDecoder",
    "GPUDecoderFallback",
    "NvImageCodecBatchDecoder",
    "NvImageCodecBatchDecoderFallback",
    "ROIParams",
    "check_gpu_decoder_available",
    "check_cvcuda_available",
    "get_decoder",
    # Numba decoder (FFCV-style)
    "NumbaBatchDecoder",
    "check_numba_decoder_available",
    # Pure decode stages
    "DecodeOnly",
    "DecodeYUVFullRes",
    "DecodeYUVPlanes",
    # YUV-output crop stages
    "DecodeYUVCenterCrop",
    "DecodeYUVRandomResizedCrop",
    "DecodeYUVResizeCrop",
    # Fused decode+crop stages (new names)
    "DecodeCenterCrop",
    "DecodeRandomResizedCrop",
    "DecodeDirectRandomResizedCrop",
    "DecodeResizeCrop",
    "DecodeRandomResizeShortCropLong",
    # Multi-crop stages (new names)
    "DecodeMultiRandomResizedCrop",
    "DecodeMultiRandomResizeShortCropLong",
    "DecodeUniformMultiRandomResizedCrop",
    "MultiCropPipeline",
    "NamedCopies",
    # Backward-compatible aliases (deprecated)
    "CenterCrop",
    "RandomResizedCrop",
    "DirectRandomResizedCrop",
    "ResizeCrop",
    "RandomResizeShortCropLong",
    "MultiCropRandomResizedCrop",
    "MultiRandomResizedCrop",
    "MultiRandomResizeShortCropLong",
    # Utilities
    "estimate_rejection_fallback_rate",
]
