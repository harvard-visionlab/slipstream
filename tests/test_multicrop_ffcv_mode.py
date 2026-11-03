"""DecodeMultiRandomResizedCrop(crop_mode="ffcv") (0.14.0): lrm-ssl / FFCV crop geometry.

Reference: ``get_random_crop`` from lrm_ssl_torch/dataloaders/custom_decoders/resized_crop.py
(lrm-ssl@f25706c), copied verbatim below (second_seed branch dropped).
"""

import numpy as np
import pytest
from numba import njit

from slipstream.decoders.numba_decoder import _generate_ffcv_random_crop_params_batch


@njit(parallel=False, fastmath=True, inline='always')
def get_random_crop(height, width, scale, ratio, seed=None):
    area = height * width
    log_ratio = np.log(ratio)

    if seed is not None:
        np.random.seed(seed)

    for _ in range(10):
        target_area = area * np.random.uniform(scale[0], scale[1])
        aspect_ratio = np.exp(np.random.uniform(log_ratio[0], log_ratio[1]))

        w = int(round(np.sqrt(target_area * aspect_ratio)))
        h = int(round(np.sqrt(target_area / aspect_ratio)))

        if 0 < w <= width and 0 < h <= height:
            i = int(np.random.uniform(0, height - h + 1))
            j = int(np.random.uniform(0, width - w + 1))
            return i, j, h, w

    # Fallback to central crop if no valid crop is found
    in_ratio = float(width) / float(height)
    if in_ratio < min(ratio):
        w = width
        h = int(round(w / min(ratio)))
    elif in_ratio > max(ratio):
        h = height
        w = int(round(h * max(ratio)))
    else:
        w = width // 2
        h = height // 2

    i = (height - h) // 2
    j = (width - w) // 2
    return i, j, h, w


def _ref(widths, heights, scale, ratio, seeds):
    out = []
    for w, h, s in zip(widths, heights, seeds):
        i, j, ch, cw = get_random_crop(np.uint32(h), np.uint32(w), np.array(scale), np.array(ratio), s)
        out.append((j, i, cw, ch))
    return np.array(out, dtype=np.int32)


CASES = [((0.08, 1.0), (3 / 4, 4 / 3)), ((0.4, 1.0), (2.0, 2.0)), ((0.4, 1.0), (0.5, 0.5)),
         ((0.05, 0.3), (3 / 4, 4 / 3)), ((0.9, 1.0), (0.2, 5.0))]


@pytest.mark.parametrize("scale,ratio", CASES)
def test_matches_get_random_crop(scale, ratio):
    rng = np.random.default_rng(0)
    n = 2000
    widths = rng.integers(16, 640, n).astype(np.int32)
    heights = rng.integers(16, 640, n).astype(np.int32)
    seeds = rng.integers(0, 2**31 - 1, n).astype(np.int64)
    got = _generate_ffcv_random_crop_params_batch(widths, heights, *scale, *ratio, seeds)
    np.testing.assert_array_equal(got, _ref(widths, heights, scale, ratio, seeds))


def test_fallback_keeps_ratio():
    # reviewer's case: 100x100, ratio 2, scale [0.4, 1] -> ~16% of draws exhaust the 10 attempts
    n = 4000
    w = np.full(n, 100, np.int32)
    seeds = np.arange(n, dtype=np.int64)
    got = _generate_ffcv_random_crop_params_batch(w, w, 0.4, 1.0, 2.0, 2.0, seeds)
    np.testing.assert_array_equal(got, _ref(w, w, (0.4, 1.0), (2.0, 2.0), seeds))
    fb = (got == (0, 25, 100, 50)).all(1)
    assert 0.10 < fb.mean() < 0.22
    assert (got[:, 2] >= got[:, 3]).all() and not (got[:, 2] == got[:, 3]).any()   # never square


def test_position_uniform_not_edge_piled():
    # small crops in a big image: lrm draws a uniform top-left, so edge-touching crops are rare
    n = 4000
    w = np.full(n, 500, np.int32)
    got = _generate_ffcv_random_crop_params_batch(w, w, 0.05, 0.1, 1.0, 1.0, np.arange(n, dtype=np.int64))
    at_edge = (got[:, 0] == 0) | (got[:, 0] + got[:, 2] == 500)
    assert at_edge.mean() < 0.02


def test_validation():
    from slipstream.decoders.multicrop import DecodeMultiRandomResizedCrop
    with pytest.raises(ValueError, match="ffcv"):
        DecodeMultiRandomResizedCrop({"x": dict(size=64)}, crop_mode="bogus")
    assert "crop_mode='ffcv'" in repr(DecodeMultiRandomResizedCrop({"x": dict(size=64)}, crop_mode="ffcv"))
