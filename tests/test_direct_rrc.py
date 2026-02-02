"""Tests for DirectRandomResizedCrop and rejection fallback rate estimation."""

import math

import numpy as np
import pytest

from slipstream.decoders.numba_decoder import (
    _generate_direct_random_crop_params_batch,
    _generate_random_crop_params_batch,
)
from slipstream.pipelines import estimate_rejection_fallback_rate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gen_params(func, width, height, n, scale, ratio, seed=42):
    """Generate crop params for a single image size."""
    ws = np.full(n, width, dtype=np.int32)
    hs = np.full(n, height, dtype=np.int32)
    return func(
        ws, hs, scale[0], scale[1],
        math.log(ratio[0]), math.log(ratio[1]), seed,
    )


def _fallback_mask(params, w, h):
    """Boolean mask: True where center-crop fallback was used."""
    min_dim = min(w, h)
    cx, cy = (w - min_dim) // 2, (h - min_dim) // 2
    return (
        (params[:, 2] == min_dim) & (params[:, 3] == min_dim) &
        (params[:, 0] == cx) & (params[:, 1] == cy)
    )


# ---------------------------------------------------------------------------
# Tests: crop validity
# ---------------------------------------------------------------------------

class TestDirectRRCValidity:
    """All crops produced by DirectRRC must fit within the image."""

    @pytest.mark.parametrize("w,h", [(256, 256), (512, 256), (256, 512), (333, 187)])
    @pytest.mark.parametrize("scale", [(0.08, 1.0), (0.01, 0.1), (0.5, 1.0)])
    @pytest.mark.parametrize("ratio", [(3/4, 4/3), (0.5, 2.0)])
    def test_crops_fit_in_image(self, w, h, scale, ratio):
        params = _gen_params(
            _generate_direct_random_crop_params_batch,
            w, h, n=5000, scale=scale, ratio=ratio,
        )
        assert np.all(params[:, 0] >= 0), "crop_x < 0"
        assert np.all(params[:, 1] >= 0), "crop_y < 0"
        assert np.all(params[:, 2] >= 1), "crop_w < 1"
        assert np.all(params[:, 3] >= 1), "crop_h < 1"
        assert np.all(params[:, 0] + params[:, 2] <= w), "crop extends past right edge"
        assert np.all(params[:, 1] + params[:, 3] <= h), "crop extends past bottom edge"


class TestDirectRRCNoFallback:
    """DirectRRC should have zero fallbacks for typical image sizes."""

    @pytest.mark.parametrize("w,h", [(256, 256), (512, 256), (256, 512)])
    def test_near_zero_fallbacks_default_params(self, w, h):
        n = 10000
        params = _gen_params(
            _generate_direct_random_crop_params_batch,
            w, h, n=n, scale=(0.08, 1.0), ratio=(3/4, 4/3),
        )
        fb = _fallback_mask(params, w, h).sum()
        # Allow a few false positives: scale~1.0 + ratio~1.0 on square images
        # produces a full-image crop that matches the fallback signature.
        assert fb / n < 0.001, f"Got {fb}/{n} fallbacks on {w}x{h}"


# ---------------------------------------------------------------------------
# Tests: distribution sanity
# ---------------------------------------------------------------------------

class TestDirectRRCDistribution:
    """Basic sanity checks on the distribution of crop parameters."""

    def test_mean_scale_reasonable(self):
        """Mean scale should be roughly centered in the scale range."""
        w, h = 256, 256
        params = _gen_params(
            _generate_direct_random_crop_params_batch,
            w, h, n=50000, scale=(0.08, 1.0), ratio=(3/4, 4/3),
        )
        scales = (params[:, 2].astype(float) * params[:, 3].astype(float)) / (w * h)
        # Mean should be in a reasonable range (not degenerate)
        assert 0.2 < scales.mean() < 0.8

    def test_aspect_ratio_symmetric(self):
        """For square images with symmetric ratio range, mean ratio ~ 1.0."""
        w, h = 256, 256
        params = _gen_params(
            _generate_direct_random_crop_params_batch,
            w, h, n=50000, scale=(0.08, 1.0), ratio=(3/4, 4/3),
        )
        ratios = params[:, 2].astype(float) / params[:, 3].astype(float)
        # Log-uniform over symmetric range has geometric mean = 1.0
        assert 0.9 < ratios.mean() < 1.15


# ---------------------------------------------------------------------------
# Tests: estimate_rejection_fallback_rate
# ---------------------------------------------------------------------------

class TestFallbackRateEstimation:
    """Tests for the fallback rate estimation utility."""

    def test_low_fallback_default_params(self):
        """Default ImageNet params on 256x256 should have near-zero fallback."""
        result = estimate_rejection_fallback_rate(
            widths=[256], heights=[256], n_samples=10000,
        )
        assert result["fallback_rate"] < 0.01
        assert result["recommend_direct"] is False

    def test_high_fallback_extreme_params(self):
        """Extreme scale+ratio on small images should have high fallback."""
        # Very narrow scale range at high values + wide ratio → many rejections
        result = estimate_rejection_fallback_rate(
            widths=[64], heights=[64],
            scale=(0.9, 1.0), ratio=(0.2, 5.0),
            n_samples=10000,
        )
        # With these extreme params, rejection sampling struggles
        assert result["fallback_rate"] > 0.05
        assert result["recommend_direct"] is True

    def test_multiple_image_sizes(self):
        """Should handle multiple image sizes."""
        result = estimate_rejection_fallback_rate(
            widths=[256, 512, 333], heights=[256, 256, 187],
            n_samples=5000,
        )
        assert "fallback_rate" in result
        assert result["total_samples"] == 15000  # 3 unique pairs × 5000

    def test_return_keys(self):
        """Should return all expected keys."""
        result = estimate_rejection_fallback_rate(
            widths=[256], heights=[256], n_samples=100,
        )
        assert set(result.keys()) == {
            "fallback_rate", "fallback_count", "total_samples", "recommend_direct",
        }
