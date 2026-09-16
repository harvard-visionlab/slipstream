"""Window (sequence) support for per-sample random augmentation parameters.

When ``SlipstreamLoader(window=(T, stride))`` is used, a batch of B anchors is
loaded as B*T consecutive records (anchor-major: sample ``i`` is window ``i // T``,
frame ``i % T``). Augmentations must apply the *same* random parameters to all T
frames of a window, so every decoder / transform that draws per-sample parameters
carries a ``seed_repeat`` attribute (set by the loader, default 1) and passes its
per-sample parameter arrays through :func:`repeat_params`.
"""

from __future__ import annotations

import numpy as np


def repeat_params(params, group: int):
    """Repeat the first row of every ``group`` consecutive rows across the group.

    ``params`` is a numpy array or torch tensor whose first dimension is the
    (expanded) batch. With ``group == 1`` it is returned unchanged.
    """
    if group is None or group <= 1:
        return params
    n = len(params)
    if isinstance(params, np.ndarray):
        return np.ascontiguousarray(np.repeat(params[::group], group, axis=0)[:n])
    # torch tensor
    return params[::group].repeat_interleave(group, dim=0)[:n]


def n_groups(n: int, group: int) -> int:
    """Number of windows in an expanded batch of ``n`` samples (ceil division)."""
    if group is None or group <= 1:
        return n
    return -(-n // group)


def expand_groups(x, group: int, n: int):
    """Expand per-group values ``[ng, ...]`` to per-sample values ``[n, ...]``.

    Each group value is repeated ``group`` times (consecutive samples), then the
    result is trimmed to ``n``. With ``group == 1`` ``x`` is returned unchanged.
    """
    if group is None or group <= 1:
        return x
    if isinstance(x, np.ndarray):
        return np.repeat(x, group, axis=0)[:n]
    return x.repeat_interleave(group, dim=0)[:n]
