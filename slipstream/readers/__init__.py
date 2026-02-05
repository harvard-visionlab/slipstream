"""Dataset readers for different source formats.

Readers provide a uniform interface for accessing data from various formats
(FFCV .beton, LitData, ImageFolder, etc.) and converting to OptimizedCache.
"""

from slipstream.readers.ffcv import FFCVFileReader
from slipstream.readers.imagefolder import SlipstreamImageFolder, open_imagefolder
from slipstream.readers.streaming import StreamingReader

__all__ = [
    "FFCVFileReader",
    "SlipstreamImageFolder",
    "StreamingReader",
    "open_imagefolder",
]
