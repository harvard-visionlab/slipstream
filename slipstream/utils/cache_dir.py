"""Cache directory utilities for slipstream.

Provides a unified cache location for all SlipCache data, regardless of
the source dataset type (LitData, FFCV, ImageFolder).

Environment variable:
    SLIPSTREAM_CACHE_DIR: Override the default cache directory.
                          Default: ~/.slipstream/

Usage on cluster:
    # Option 1: Environment variable
    export SLIPSTREAM_CACHE_DIR=/mnt/fast-storage/slipstream-cache

    # Option 2: Symlink
    ln -s /mnt/fast-storage/slipstream-cache ~/.slipstream
"""

import os
from pathlib import Path

# Environment variable name for cache directory override
CACHE_DIR_ENV_VAR = "SLIPSTREAM_CACHE_DIR"

# Default cache directory (in user's home)
DEFAULT_CACHE_DIR = Path.home() / ".slipstream"


def get_cache_base() -> Path:
    """Get the base directory for all SlipCache data.

    Returns the cache base directory, checking in order:
    1. SLIPSTREAM_CACHE_DIR environment variable (if set)
    2. ~/.slipstream/ (default)

    The actual cache for each dataset will be stored in a subdirectory
    named `slipcache-{dataset_hash}` to prevent collisions and enable
    cache invalidation when source data changes.

    Returns:
        Path to the cache base directory.

    Example:
        >>> get_cache_base()
        PosixPath('/home/user/.slipstream')

        >>> os.environ['SLIPSTREAM_CACHE_DIR'] = '/mnt/fast/cache'
        >>> get_cache_base()
        PosixPath('/mnt/fast/cache')
    """
    env_path = os.environ.get(CACHE_DIR_ENV_VAR)
    if env_path:
        return Path(env_path).expanduser().resolve()
    return DEFAULT_CACHE_DIR


def get_cache_path(dataset_hash: str) -> Path:
    """Get the full cache path for a dataset with the given hash.

    Args:
        dataset_hash: The dataset's content hash (typically 8 characters).

    Returns:
        Path like ~/.slipstream/slipcache-{hash}/
    """
    return get_cache_base() / f"slipcache-{dataset_hash}"


__all__ = ["get_cache_base", "get_cache_path", "CACHE_DIR_ENV_VAR", "DEFAULT_CACHE_DIR"]
