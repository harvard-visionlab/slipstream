"""Base class for decoder stages.

Provides the protocol expected by SlipstreamLoader: any callable with
optional ``set_image_format()`` and ``shutdown()`` methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


def _bytes_to_batch_dict(image_bytes: bytes | bytearray | memoryview) -> dict:
    """Wrap raw image bytes into a batch-of-1 dict for batch decoders.

    Args:
        image_bytes: Raw image file bytes (e.g. JPEG).

    Returns:
        Dict with keys ``data``, ``sizes``, ``heights``, ``widths`` —
        the same format that :class:`SlipstreamLoader` produces.
    """
    from slipstream.utils.image_header import read_image_dimensions

    raw = bytes(image_bytes)
    w, h = read_image_dimensions(raw)
    arr = np.frombuffer(raw, dtype=np.uint8)
    data = np.zeros((1, len(arr)), dtype=np.uint8)
    data[0, :len(arr)] = arr
    return {
        'data': data,
        'sizes': np.array([len(arr)], dtype=np.uint64),
        'heights': np.array([h], dtype=np.uint32),
        'widths': np.array([w], dtype=np.uint32),
    }


def _unwrap_single_result(result: Any) -> Any:
    """Remove the batch dimension from a batch-of-1 decoder result.

    Handles ndarray/Tensor ``[1, H, W, C]`` → ``[H, W, C]``, lists of
    length 1, and dicts (recursed).
    """
    import torch

    if isinstance(result, dict):
        return {k: _unwrap_single_result(v) for k, v in result.items()}
    if isinstance(result, list):
        if len(result) == 1:
            item = result[0]
            # If it's a 4-D array/tensor (batch leftover), squeeze
            if isinstance(item, np.ndarray) and item.ndim == 4:
                return item[0]
            if isinstance(item, torch.Tensor) and item.ndim == 4:
                return item[0]
            return item
        # List of batched arrays — squeeze each
        out = []
        for item in result:
            if isinstance(item, np.ndarray) and item.ndim == 4 and item.shape[0] == 1:
                out.append(item[0])
            elif isinstance(item, torch.Tensor) and item.ndim == 4 and item.shape[0] == 1:
                out.append(item[0])
            else:
                out.append(item)
        return out
    if isinstance(result, np.ndarray) and result.ndim == 4 and result.shape[0] == 1:
        return result[0]
    if isinstance(result, torch.Tensor) and result.ndim == 4 and result.shape[0] == 1:
        return result[0]
    return result


class BatchTransform(ABC):
    """Base class for decode stages and pipeline transforms.

    The loader calls stages as plain callables and uses ``hasattr`` checks
    for ``set_image_format`` and ``shutdown``, so any callable works.  This
    ABC is provided for documentation and type-checking convenience.
    """

    @abstractmethod
    def __call__(self, batch_data: Any) -> Any:
        ...

    def set_image_format(self, image_format: str) -> None:
        """Called by loader to configure decoder for cache format. No-op by default."""

    def shutdown(self) -> None:
        """Release resources (decoder threads, etc.). No-op by default."""
