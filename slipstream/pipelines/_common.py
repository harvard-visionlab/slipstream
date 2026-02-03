"""Shared utilities for pipeline presets."""

from __future__ import annotations

# Seed offset constants (same convention as lrm-ssl configs)
CROP_OFFSET = 1234
FLIP_OFFSET = 1111
COLOR_OFFSET = 2222
GRAY_OFFSET = 3333
SOLAR_OFFSET = 4444
BLUR_OFFSET = 5555


def _seed(base: int | None, offset: int, crop_id: int = 0) -> int | None:
    """Derive a deterministic seed, or None if base is None."""
    if base is None:
        return None
    return base + offset + crop_id
