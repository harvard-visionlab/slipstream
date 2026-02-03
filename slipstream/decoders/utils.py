"""Decoder utility functions."""

from __future__ import annotations

import numpy as np


def estimate_rejection_fallback_rate(
    widths: np.ndarray | list[int],
    heights: np.ndarray | list[int],
    scale: tuple[float, float] = (0.08, 1.0),
    ratio: tuple[float, float] = (3 / 4, 4 / 3),
    n_samples: int = 50000,
    seed: int = 42,
) -> dict[str, float]:
    """Estimate how often rejection-sampling RRC falls back to center crop.

    Generates ``n_samples`` crop parameters per unique (width, height) pair
    using the rejection-sampling method and counts how many hit the
    center-crop fallback. If the fallback rate exceeds ~5%, consider using
    ``DirectRandomResizedCrop`` which guarantees a valid crop analytically.

    Args:
        widths: Image widths (one per image, or representative set).
        heights: Image heights (one per image, or representative set).
        scale: Scale range for random area.
        ratio: Aspect ratio range.
        n_samples: Samples to generate per unique (w, h) pair.
        seed: RNG seed.

    Returns:
        Dict with keys:
        - ``"fallback_rate"``: fraction of samples that hit center-crop fallback
        - ``"fallback_count"``: number of fallback samples
        - ``"total_samples"``: total samples tested
        - ``"recommend_direct"``: True if fallback_rate > 5%
    """
    import math
    from slipstream.decoders.numba_decoder import _generate_random_crop_params_batch

    widths = np.asarray(widths, dtype=np.int32)
    heights = np.asarray(heights, dtype=np.int32)

    log_ratio_min = math.log(ratio[0])
    log_ratio_max = math.log(ratio[1])

    pairs = np.unique(np.column_stack([widths, heights]), axis=0)

    total_fallbacks = 0
    total_tested = 0

    for row in pairs:
        w, h = int(row[0]), int(row[1])
        ws = np.full(n_samples, w, dtype=np.int32)
        hs = np.full(n_samples, h, dtype=np.int32)

        params = _generate_random_crop_params_batch(
            ws, hs, scale[0], scale[1], log_ratio_min, log_ratio_max, seed,
        )

        min_dim = min(w, h)
        cx = (w - min_dim) // 2
        cy = (h - min_dim) // 2
        is_fallback = (
            (params[:, 2] == min_dim) & (params[:, 3] == min_dim) &
            (params[:, 0] == cx) & (params[:, 1] == cy)
        )
        total_fallbacks += int(is_fallback.sum())
        total_tested += n_samples

    rate = total_fallbacks / total_tested if total_tested > 0 else 0.0
    return {
        "fallback_rate": rate,
        "fallback_count": total_fallbacks,
        "total_samples": total_tested,
        "recommend_direct": rate > 0.05,
    }
