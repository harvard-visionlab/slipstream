"""Test tiers (see CLAUDE.md "Testing tiers").

Files in SLOW_FILES get the ``slow`` marker: the core tier is ``pytest -m "not slow"``.
"""

import pytest

# The 4 files that take ~214 s of the ~264 s full suite on machina (2026-11-03): format conversions
# over real 50k-image val sets and the cache-integrity build/verify matrix.
SLOW_FILES = {
    "test_imagefolder_to_slipcache.py",
    "test_litdata_to_slipcache.py",
    "test_ffcv_to_slipcache.py",
    "test_cache_integrity.py",
}


def pytest_collection_modifyitems(config, items):
    for item in items:
        if item.path.name in SLOW_FILES:
            item.add_marker(pytest.mark.slow)
