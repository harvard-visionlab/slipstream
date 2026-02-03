"""Base class for decoder stages.

Provides the protocol expected by SlipstreamLoader: any callable with
optional ``set_image_format()`` and ``shutdown()`` methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


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
