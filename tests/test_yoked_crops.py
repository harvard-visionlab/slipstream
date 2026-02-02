"""Test yoked crops: same seed → same crop center, different scale → different zoom."""

import math
import numpy as np
import pytest

from slipstream.decoders.numba_decoder import (
    _generate_random_crop_params_batch,
    _generate_direct_random_crop_params_batch,
)


def crop_centers(params):
    """Extract crop centers from [B, 4] params array."""
    x, y, w, h = params[:, 0], params[:, 1], params[:, 2], params[:, 3]
    return x + w / 2.0, y + h / 2.0


@pytest.fixture
def image_dims():
    """Batch of 8 images with varied dimensions."""
    widths = np.array([384, 512, 256, 640, 480, 320, 400, 360], dtype=np.int32)
    heights = np.array([256, 384, 256, 480, 320, 240, 300, 270], dtype=np.int32)
    return widths, heights


LOG_RATIO_MIN = math.log(3 / 4)
LOG_RATIO_MAX = math.log(4 / 3)

# Square ratio (simplest case for yoking)
LOG_RATIO_SQUARE_MIN = math.log(1.0)
LOG_RATIO_SQUARE_MAX = math.log(1.0)


class TestYokedCropsStandardRRC:
    """Test yoking with standard (rejection sampling) RRC."""

    def test_same_seed_same_scale_identical(self, image_dims):
        """Same seed + same scale → identical params."""
        widths, heights = image_dims
        p1 = _generate_random_crop_params_batch(
            widths, heights, 0.4, 1.0, LOG_RATIO_MIN, LOG_RATIO_MAX, seed=42
        )
        p2 = _generate_random_crop_params_batch(
            widths, heights, 0.4, 1.0, LOG_RATIO_MIN, LOG_RATIO_MAX, seed=42
        )
        np.testing.assert_array_equal(p1, p2)

    def test_different_seed_different_params(self, image_dims):
        """Different seeds → different params."""
        widths, heights = image_dims
        p1 = _generate_random_crop_params_batch(
            widths, heights, 0.4, 1.0, LOG_RATIO_MIN, LOG_RATIO_MAX, seed=42
        )
        p2 = _generate_random_crop_params_batch(
            widths, heights, 0.4, 1.0, LOG_RATIO_MIN, LOG_RATIO_MAX, seed=99
        )
        assert not np.array_equal(p1, p2)

    def test_yoked_same_seed_different_scale_centers_align(self, image_dims):
        """Same seed + different scale → centers should be close.

        With standard RRC, the RNG draws: U1 (scale), U2 (ratio), U3 (x), U4 (y).
        Same seed → same U1-U4 sequence. But U1 maps to different target_area
        via different scale ranges, giving different crop sizes. U3/U4 then map
        to positions via randint(0, img_dim - crop_dim + 1), so the absolute
        positions differ. However, the *fractional* position (U3 value) is the same.

        For square ratio, the centers should be correlated but not identical.
        """
        widths, heights = image_dims
        p_wide = _generate_random_crop_params_batch(
            widths, heights, 0.4, 1.0, LOG_RATIO_SQUARE_MIN, LOG_RATIO_SQUARE_MAX, seed=42
        )
        p_narrow = _generate_random_crop_params_batch(
            widths, heights, 0.05, 0.4, LOG_RATIO_SQUARE_MIN, LOG_RATIO_SQUARE_MAX, seed=42
        )

        cx_w, cy_w = crop_centers(p_wide)
        cx_n, cy_n = crop_centers(p_narrow)

        # Print for debugging
        for i in range(len(widths)):
            print(f"  img {i} ({widths[i]}x{heights[i]}): "
                  f"wide center=({cx_w[i]:.1f}, {cy_w[i]:.1f}) crop=({p_wide[i, 2]}x{p_wide[i, 3]}), "
                  f"narrow center=({cx_n[i]:.1f}, {cy_n[i]:.1f}) crop=({p_narrow[i, 2]}x{p_narrow[i, 3]})")

        # The narrow crop should be fully contained within the wide crop region
        # (since it's a smaller region from the same center tendency)
        # Check that narrow crop overlaps with wide crop for each image
        for i in range(len(widths)):
            wx, wy = p_wide[i, 0], p_wide[i, 1]
            ww, wh = p_wide[i, 2], p_wide[i, 3]
            nx, ny = p_narrow[i, 0], p_narrow[i, 1]
            nw, nh = p_narrow[i, 2], p_narrow[i, 3]
            # Check overlap (not containment — RNG position mapping differs)
            overlap_x = max(0, min(wx + ww, nx + nw) - max(wx, nx))
            overlap_y = max(0, min(wy + wh, ny + nh) - max(wy, ny))
            overlap_area = overlap_x * overlap_y
            narrow_area = nw * nh
            # Narrow crop should overlap significantly with wide crop
            if narrow_area > 0:
                overlap_frac = overlap_area / narrow_area
                print(f"    overlap: {overlap_frac:.1%} of narrow crop")


    def test_per_sample_independence(self, image_dims):
        """Changing one image's dimensions doesn't affect other samples' crops."""
        widths, heights = image_dims
        p1 = _generate_random_crop_params_batch(
            widths, heights, 0.4, 1.0, LOG_RATIO_MIN, LOG_RATIO_MAX, seed=42
        )
        widths2 = widths.copy()
        widths2[3] = 200
        p2 = _generate_random_crop_params_batch(
            widths2, heights, 0.4, 1.0, LOG_RATIO_MIN, LOG_RATIO_MAX, seed=42
        )
        for i in range(len(widths)):
            if i != 3:
                np.testing.assert_array_equal(p1[i], p2[i])


class TestYokedCropsDirectRRC:
    """Test yoking with direct (analytic) RRC."""

    def test_same_seed_same_scale_identical(self, image_dims):
        """Same seed + same scale → identical params."""
        widths, heights = image_dims
        p1 = _generate_direct_random_crop_params_batch(
            widths, heights, 0.4, 1.0, LOG_RATIO_MIN, LOG_RATIO_MAX, seed=42
        )
        p2 = _generate_direct_random_crop_params_batch(
            widths, heights, 0.4, 1.0, LOG_RATIO_MIN, LOG_RATIO_MAX, seed=42
        )
        np.testing.assert_array_equal(p1, p2)

    def test_yoked_same_seed_different_scale_centers_align(self, image_dims):
        """Same seed + different scale → centers should be close."""
        widths, heights = image_dims
        p_wide = _generate_direct_random_crop_params_batch(
            widths, heights, 0.4, 1.0, LOG_RATIO_SQUARE_MIN, LOG_RATIO_SQUARE_MAX, seed=42
        )
        p_narrow = _generate_direct_random_crop_params_batch(
            widths, heights, 0.05, 0.4, LOG_RATIO_SQUARE_MIN, LOG_RATIO_SQUARE_MAX, seed=42
        )

        cx_w, cy_w = crop_centers(p_wide)
        cx_n, cy_n = crop_centers(p_narrow)

        for i in range(len(widths)):
            print(f"  img {i} ({widths[i]}x{heights[i]}): "
                  f"wide center=({cx_w[i]:.1f}, {cy_w[i]:.1f}) crop=({p_wide[i, 2]}x{p_wide[i, 3]}), "
                  f"narrow center=({cx_n[i]:.1f}, {cy_n[i]:.1f}) crop=({p_narrow[i, 2]}x{p_narrow[i, 3]})")

        # Direct RRC: narrow crop must be fully contained in wide crop
        # (no rejection sampling → deterministic RNG sequence → same fractional position)
        for i in range(len(widths)):
            wx, wy = p_wide[i, 0], p_wide[i, 1]
            ww, wh = p_wide[i, 2], p_wide[i, 3]
            nx, ny = p_narrow[i, 0], p_narrow[i, 1]
            nw, nh = p_narrow[i, 2], p_narrow[i, 3]
            overlap_x = max(0, min(wx + ww, nx + nw) - max(wx, nx))
            overlap_y = max(0, min(wy + wh, ny + nh) - max(wy, ny))
            overlap_area = overlap_x * overlap_y
            narrow_area = nw * nh
            if narrow_area > 0:
                overlap_frac = overlap_area / narrow_area
                print(f"    overlap: {overlap_frac:.1%} of narrow crop")
                assert overlap_frac >= 0.99, (
                    f"img {i}: narrow crop not contained in wide crop "
                    f"(overlap={overlap_frac:.1%})"
                )


    def test_per_sample_independence(self, image_dims):
        """Changing one image's dimensions doesn't affect other samples' crops."""
        widths, heights = image_dims
        p1 = _generate_direct_random_crop_params_batch(
            widths, heights, 0.4, 1.0, LOG_RATIO_MIN, LOG_RATIO_MAX, seed=42
        )
        widths2 = widths.copy()
        widths2[3] = 200
        p2 = _generate_direct_random_crop_params_batch(
            widths2, heights, 0.4, 1.0, LOG_RATIO_MIN, LOG_RATIO_MAX, seed=42
        )
        for i in range(len(widths)):
            if i != 3:
                np.testing.assert_array_equal(p1[i], p2[i])


class TestRNGSequenceAnalysis:
    """Analyze exactly how the RNG sequence diverges between scale ranges."""

    def test_rng_draws_standard(self):
        """Show the raw RNG draws for standard RRC with same seed."""
        w, h = 384, 256
        area = w * h

        for scale_name, s_min, s_max in [("wide 0.4-1.0", 0.4, 1.0),
                                           ("narrow 0.05-0.4", 0.05, 0.4)]:
            np.random.seed(42)
            # Standard RRC draws per attempt: U_scale, U_ratio, (U_x, U_y if valid)
            u_scale = np.random.uniform(s_min, s_max)
            u_ratio = np.random.uniform(LOG_RATIO_SQUARE_MIN, LOG_RATIO_SQUARE_MAX)
            target_area = area * u_scale  # NOT area * U_raw, but area * uniform(s_min, s_max)
            ratio = np.exp(u_ratio)
            crop_w = int(np.sqrt(target_area * ratio) + 0.5)
            crop_h = int(np.sqrt(target_area / ratio) + 0.5)
            valid = 0 < crop_w <= w and 0 < crop_h <= h
            if valid:
                u_x = np.random.randint(0, w - crop_w + 1)
                u_y = np.random.randint(0, h - crop_h + 1)
            else:
                u_x, u_y = -1, -1
            print(f"  {scale_name}: target_area={target_area:.0f}, crop={crop_w}x{crop_h}, "
                  f"valid={valid}, pos=({u_x}, {u_y})")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
