"""Dataset readers for different source formats.

Readers provide a uniform interface for accessing data from various formats
(FFCV .beton, LitData, ImageFolder, etc.) and converting to OptimizedCache.
"""

from slipstream.readers.ffcv import FFCVFileReader

__all__ = [
    "FFCVFileReader",
]
