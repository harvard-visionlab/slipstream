"""Dataset readers for different source formats.

Readers provide a uniform interface for accessing data from various formats
(FFCV .beton, LitData, ImageFolder, etc.) and converting to OptimizedCache.
"""

from slipstream.readers.ffcv import FFCVFileReader
from slipstream.readers.streaming import StreamingReader

__all__ = [
    "FFCVFileReader",
    "SlipstreamImageFolder",
    "StreamingReader",
    "open_imagefolder",
]


# SlipstreamImageFolder subclasses torchvision's ImageFolder; importing torchvision costs seconds
# (models, torch._dynamo), so the imagefolder module loads on first use (PEP 562).
_LAZY_IMAGEFOLDER = ("SlipstreamImageFolder", "open_imagefolder")


def __getattr__(name):
    if name in _LAZY_IMAGEFOLDER:
        from slipstream.readers import imagefolder
        return getattr(imagefolder, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
