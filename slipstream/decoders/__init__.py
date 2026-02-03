"""Decoders for slipstream.

Low-level decoders:
- CPUDecoder: TurboJPEG for parallel JPEG decoding
- GPUDecoder: nvImageCodec for GPU JPEG decoding
- NumbaBatchDecoder: Numba JIT-compiled batch decoder (FFCV-style, fastest)
- YUV420NumbaBatchDecoder: YUV420P raw format decoder (~2x JPEG throughput)

Fused decode+crop stages (for SlipstreamLoader pipelines):
- DecodeOnly, DecodeYUVFullRes, DecodeYUVPlanes: pure decode
- CenterCrop, RandomResizedCrop, DirectRandomResizedCrop, ResizeCrop: decode+crop
- MultiCropRandomResizedCrop, MultiRandomResizedCrop, MultiCropPipeline: multi-crop

Usage:
    from slipstream.decoders import CPUDecoder, GPUDecoder, get_decoder
    from slipstream.decoders import RandomResizedCrop, CenterCrop
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
)
from slipstream.decoders.crop import (
    DecodeCenterCrop,
    DecodeDirectRandomResizedCrop,
    DecodeRandomResizedCrop,
    DecodeResizeCrop,
    # Backward-compatible aliases
    CenterCrop,
    DirectRandomResizedCrop,
    RandomResizedCrop,
    ResizeCrop,
)
from slipstream.decoders.multicrop import (
    DecodeMultiRandomResizedCrop,
    DecodeUniformMultiRandomResizedCrop,
    MultiCropPipeline,
    # Backward-compatible aliases
    MultiCropRandomResizedCrop,
    MultiRandomResizedCrop,
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
    # Fused decode+crop stages (new names)
    "DecodeCenterCrop",
    "DecodeRandomResizedCrop",
    "DecodeDirectRandomResizedCrop",
    "DecodeResizeCrop",
    # Multi-crop stages (new names)
    "DecodeMultiRandomResizedCrop",
    "DecodeUniformMultiRandomResizedCrop",
    "MultiCropPipeline",
    # Backward-compatible aliases (deprecated)
    "CenterCrop",
    "RandomResizedCrop",
    "DirectRandomResizedCrop",
    "ResizeCrop",
    "MultiCropRandomResizedCrop",
    "MultiRandomResizedCrop",
    # Utilities
    "estimate_rejection_fallback_rate",
]
