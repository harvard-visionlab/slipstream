"""Decoders for slipstream.

GPU decoder: nvImageCodec for fused decode + RandomResizedCrop
CPU decoder: TurboJPEG for parallel JPEG decoding
Numba decoder: Numba JIT-compiled batch decoder (FFCV-style, fastest)

Usage:
    from slipstream.decoders import CPUDecoder, GPUDecoder, get_decoder

    # CPU decoding (always available)
    cpu_decoder = CPUDecoder(num_workers=8)
    images = cpu_decoder.decode_batch(data, sizes)

    # GPU decoding (requires nvImageCodec)
    if check_gpu_decoder_available():
        gpu_decoder = GPUDecoder(device=0)
        images = gpu_decoder.decode_batch(data, sizes, heights, widths)

    # Numba JIT decoder (fastest, FFCV-style)
    if check_numba_decoder_available():
        numba_decoder = NumbaBatchDecoder(num_threads=8)
        destination = np.zeros((batch_size, max_h, max_w, 3), dtype=np.uint8)
        numba_decoder.decode_batch(data, sizes, heights, widths, destination)

    # Auto-select best decoder
    decoder = get_decoder(device=0, prefer_gpu=True)
"""

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

__all__ = [
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
]
