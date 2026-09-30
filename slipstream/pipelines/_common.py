"""Shared utilities for pipeline presets."""

from __future__ import annotations

from slipstream.seeds import derive_seed

# Seed stream labels (values from the lrm-ssl configs; hashed with the base seed by _seed)
CROP_OFFSET = 1234
FLIP_OFFSET = 1111
COLOR_OFFSET = 2222
GRAY_OFFSET = 3333
SOLAR_OFFSET = 4444
BLUR_OFFSET = 5555


def _seed(base: int | None, offset: int, crop_id: int = 0) -> int | None:
    """Derive a deterministic seed, or None if base is None.

    Hashed (``derive_seed(base, offset, crop_id)``), not ``base + offset + crop_id``:
    the sum made base ``b`` view ``k+1`` equal to base ``b+1`` view ``k``. The offsets
    are now just labels for independent streams.
    """
    if base is None:
        return None
    return derive_seed(base, offset, crop_id)
