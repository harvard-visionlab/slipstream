"""libslipstream - Fast parallel JPEG decoding with TurboJPEG.

This module provides the C++ extension for high-performance JPEG decoding.
Build with:
    cd libslipstream && python setup.py build_ext --inplace

Or via uv:
    uv run python libslipstream/setup.py build_ext --inplace

For OpenCV support (enables resize_crop):
    USE_OPENCV=1 uv run python libslipstream/setup.py build_ext --inplace

The extension provides:
- jpeg_header(): Get JPEG dimensions without full decode
- imdecode(): Decode JPEG with optional crop, flip, and scale
- imdecode_simple(): Fast path without transforms
- get_scale_factor(): Get optimal TurboJPEG scale factor
"""

__version__ = "0.1.0"
