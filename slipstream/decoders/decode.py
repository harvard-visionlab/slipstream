"""Pure decode stages (no crop/resize).

- DecodeOnly: JPEG → full-size RGB (variable sizes)
- DecodeYUVFullRes: YUV420P → full-res YUV [H,W,3]
- DecodeYUVPlanes: YUV420P → raw Y/U/V planes
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
