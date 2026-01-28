"""Backend implementations for slipstream.

Backends provide high-performance data access for different dataset formats:

- OptimizedCache: Memory-mapped multi-field storage with O(1) sample access
- FFCVFileDataset: Native .ffcv/.beton file reader (actual FFCV format)

The legacy FFCVStyleDataset has been replaced by OptimizedCache, which
supports all field types (not just images) and works with any dataset.
"""

# Native FFCV file support (.beton files)
from slipstream.backends.ffcv_file import (
    FFCVFileDataset,
    FFCVFilePrefetchingDataLoader,
)

# Legacy exports (deprecated, for backwards compatibility)
# These will be removed in a future version
from slipstream.backends.ffcv_style import (
    FFCVStyleDataLoader,
    FFCVStyleDataset,
    PrefetchingDataLoader,
)

__all__ = [
    # Native FFCV files
    "FFCVFileDataset",
    "FFCVFilePrefetchingDataLoader",
    # Legacy (deprecated)
    "FFCVStyleDataset",
    "FFCVStyleDataLoader",
    "PrefetchingDataLoader",
]
