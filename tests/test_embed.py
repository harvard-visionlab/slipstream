"""Tests for RandomEmbed transform."""

import pytest
import torch

from slipstream.transforms.embed import RandomEmbed


# ── helpers ──────────────────────────────────────────────────────────────

def _make_batch(B=4, C=3, H=96, W=96, value=1.0):
    """Return a constant-filled (B, C, H, W) tensor."""
    return torch.full((B, C, H, W), value, dtype=torch.float32)


def _make_single(C=3, H=96, W=96, value=1.0):
    """Return a constant-filled (C, H, W) tensor."""
    return torch.full((C, H, W), value, dtype=torch.float32)


# ── output shape ─────────────────────────────────────────────────────────

class TestOutputShape:
    def test_output_shape_4d(self):
        t = RandomEmbed(canvas_size=224)
        out = t(_make_batch(B=4, H=96, W=96))
        assert out.shape == (4, 3, 224, 224)

    def test_output_shape_3d(self):
        t = RandomEmbed(canvas_size=224)
        out = t(_make_single(H=96, W=96))
        assert out.shape == (3, 224, 224)

    def test_canvas_size_int(self):
        t = RandomEmbed(canvas_size=128)
        out = t(_make_batch(B=2, H=64, W=64))
        assert out.shape == (2, 3, 128, 128)

    def test_canvas_size_tuple(self):
        t = RandomEmbed(canvas_size=(200, 300))
        out = t(_make_batch(B=2, H=64, W=64))
        assert out.shape == (2, 3, 200, 300)

    def test_channels_inferred(self):
        t = RandomEmbed(canvas_size=128)
        out = t(_make_batch(B=2, C=1, H=64, W=64))
        assert out.shape == (2, 1, 128, 128)


# ── range mode positioning ───────────────────────────────────────────────

class TestRangeMode:
    def test_center_embed(self):
        img = _make_batch(B=1, H=96, W=96)
        t = RandomEmbed(canvas_size=224, x_range=0.5, y_range=0.5)
        out = t(img)
        # Center position: (224 - 96) / 2 = 64
        assert out[0, 0, 64, 64] == 1.0  # inside image
        assert out[0, 0, 0, 0] == 0.0    # outside image (zeros bg)

    def test_top_left_embed(self):
        img = _make_batch(B=1, H=96, W=96)
        t = RandomEmbed(canvas_size=224, x_range=0.0, y_range=0.0)
        out = t(img)
        assert out[0, 0, 0, 0] == 1.0    # image at top-left
        assert out[0, 0, 95, 95] == 1.0  # still inside
        assert out[0, 0, 96, 96] == 0.0  # outside

    def test_bottom_right_embed(self):
        img = _make_batch(B=1, H=96, W=96)
        t = RandomEmbed(canvas_size=224, x_range=1.0, y_range=1.0)
        out = t(img)
        # Bottom-right position: start at 224 - 96 = 128
        assert out[0, 0, 128, 128] == 1.0  # image starts here
        assert out[0, 0, 223, 223] == 1.0  # image fills to end
        assert out[0, 0, 127, 127] == 0.0  # before image

    def test_per_image_different_positions(self):
        img = _make_batch(B=8, H=64, W=64, value=1.0)
        t = RandomEmbed(canvas_size=224, x_range=(0, 1), y_range=(0, 1), seed=42)
        t(img)
        # Positions should differ across images
        xs = t.last_params()["xs"]
        assert not torch.all(xs == xs[0]), "All images got the same x position"

    def test_image_content_preserved(self):
        """Placed pixels should exactly match the input."""
        B = 2
        img = torch.randn(B, 3, 64, 64)
        t = RandomEmbed(canvas_size=128, x_range=0.0, y_range=0.0)
        out = t(img)
        # At top-left corner, the image region should be identical
        assert torch.equal(out[:, :, :64, :64], img)


# ── coords mode ──────────────────────────────────────────────────────────

class TestCoordsMode:
    def test_coords_basic(self):
        img = _make_batch(B=2, H=32, W=32, value=5.0)
        # Place at center (112, 112) → top-left = (96, 96)
        t = RandomEmbed(canvas_size=224, coords=[(112, 112)])
        out = t(img)
        assert out[0, 0, 96, 96] == 5.0
        assert out[0, 0, 0, 0] == 0.0

    def test_coords_clipping(self):
        """Coord near edge clips the image — should not crash."""
        img = _make_batch(B=1, H=64, W=64, value=1.0)
        t = RandomEmbed(canvas_size=128, coords=[(0, 0)])  # center at (0,0) → mostly off-screen
        out = t(img)
        assert out.shape == (1, 3, 128, 128)
        # Only top-left 32x32 quadrant of the image is visible
        assert out[0, 0, 0, 0] == 1.0

    def test_coords_multiple(self):
        """Multiple coords — each image samples one."""
        img = _make_batch(B=4, H=32, W=32)
        t = RandomEmbed(canvas_size=128, coords=[(32, 32), (96, 96)], seed=42)
        out = t(img)
        assert out.shape == (4, 3, 128, 128)


# ── coords_dict mode ────────────────────────────────────────────────────

class TestCoordsDictMode:
    def test_coords_dict_basic(self):
        img = _make_batch(B=2, H=64, W=64, value=3.0)
        t = RandomEmbed(
            canvas_size=224,
            coords_dict={(64, 64): [(112, 112)]},
        )
        out = t(img)
        # Center at (112, 112) → top-left = (80, 80)
        assert out[0, 0, 80, 80] == 3.0

    def test_coords_dict_missing_key(self):
        t = RandomEmbed(
            canvas_size=224,
            coords_dict={(32, 32): [(112, 112)]},
        )
        with pytest.raises(KeyError, match="No coords for image size"):
            t(_make_batch(B=1, H=64, W=64))


# ── validation ─────────────────────────────────────────────────────────

class TestValidation:
    def test_coords_and_coords_dict_error(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            RandomEmbed(
                canvas_size=224,
                coords=[(112, 112)],
                coords_dict={(64, 64): [(112, 112)]},
            )

    def test_invalid_background(self):
        with pytest.raises(ValueError, match="background must be one of"):
            RandomEmbed(canvas_size=224, background="invalid")

    def test_neuronal_no_longer_valid(self):
        with pytest.raises(ValueError, match="background must be one of"):
            RandomEmbed(canvas_size=224, background="neuronal")

    def test_diffusion_no_longer_valid(self):
        with pytest.raises(ValueError, match="background must be one of"):
            RandomEmbed(canvas_size=224, background="diffusion")

    def test_mean_background_requires_mean(self):
        with pytest.raises(ValueError, match="mean.*required"):
            RandomEmbed(canvas_size=224, background="mean")

    def test_std_without_mean_raises(self):
        with pytest.raises(ValueError, match="mean.*required"):
            RandomEmbed(canvas_size=224, std=[0.229, 0.224, 0.225])


# ── background modes ────────────────────────────────────────────────────

class TestBackgrounds:
    def test_background_zeros(self):
        img = _make_batch(B=1, H=64, W=64, value=1.0)
        t = RandomEmbed(canvas_size=128, x_range=0.0, y_range=0.0, background="zeros")
        out = t(img)
        # Outside image region should be zero
        assert out[0, 0, 64:, 64:].sum() == 0.0

    def test_background_mean_scalar(self):
        img = _make_batch(B=1, H=64, W=64, value=1.0)
        t = RandomEmbed(canvas_size=128, x_range=0.0, y_range=0.0,
                        background="mean", mean=0.5)
        out = t(img)
        assert torch.allclose(out[0, 0, 100, 100], torch.tensor(0.5))

    def test_background_mean_per_channel(self):
        img = _make_batch(B=1, C=3, H=64, W=64, value=1.0)
        means = [0.485, 0.456, 0.406]
        t = RandomEmbed(canvas_size=128, x_range=0.0, y_range=0.0,
                        background="mean", mean=means)
        out = t(img)
        for c, m in enumerate(means):
            assert torch.allclose(out[0, c, 100, 100], torch.tensor(m), atol=1e-6)

    def test_background_power_law(self):
        img = _make_batch(B=2, H=64, W=64, value=0.0)
        t = RandomEmbed(canvas_size=128, x_range=0.0, y_range=0.0,
                        background="power_law", alpha_range=2.0, seed=42)
        out = t(img)
        # Background region should be non-zero noise
        bg_region = out[:, :, 64:, 64:]
        assert bg_region.abs().sum() > 0
        # Values should be in [0, 1] (always generated in this range)
        assert bg_region.min() >= 0.0 - 1e-6
        assert bg_region.max() <= 1.0 + 1e-6

    def test_background_power_law_variable_alpha(self):
        img = _make_batch(B=4, H=64, W=64, value=0.0)
        t = RandomEmbed(canvas_size=128, x_range=0.0, y_range=0.0,
                        background="power_law", alpha_range=(0.5, 2.5), seed=42)
        out = t(img)
        assert out.shape == (4, 3, 128, 128)
        bg_region = out[:, :, 64:, 64:]
        assert bg_region.abs().sum() > 0

    def test_background_power_law_color_noise(self):
        """color_noise=True (default): channels should differ."""
        img = _make_batch(B=2, H=64, W=64, value=0.0)
        t = RandomEmbed(canvas_size=128, x_range=0.0, y_range=0.0,
                        background="power_law", alpha_range=2.0, seed=42,
                        color_noise=True)
        out = t(img)
        bg = out[:, :, 64:, 64:]
        # R, G, B channels should be different
        assert not torch.equal(bg[:, 0], bg[:, 1])

    def test_background_power_law_grayscale_noise(self):
        """color_noise=False: all channels should be identical."""
        img = _make_batch(B=2, H=64, W=64, value=0.0)
        t = RandomEmbed(canvas_size=128, x_range=0.0, y_range=0.0,
                        background="power_law", alpha_range=2.0, seed=42,
                        color_noise=False)
        out = t(img)
        bg = out[:, :, 64:, 64:]
        assert torch.equal(bg[:, 0], bg[:, 1])
        assert torch.equal(bg[:, 1], bg[:, 2])

    def test_color_noise_default_true(self):
        """Default color_noise should be True."""
        t = RandomEmbed(canvas_size=128, background="power_law")
        assert t.color_noise is True


# ── normalisation (mean/std) ─────────────────────────────────────────────

class TestNormalisation:
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    def test_normalization_zeros(self):
        """zeros + mean/std → (0 - mean) / std (black in normalised space)."""
        img = _make_batch(B=1, H=64, W=64, value=0.0)
        t = RandomEmbed(canvas_size=128, x_range=0.0, y_range=0.0,
                        background="zeros", mean=self.MEAN, std=self.STD)
        out = t(img)
        for c in range(3):
            expected = (0.0 - self.MEAN[c]) / self.STD[c]
            assert torch.allclose(out[0, c, 100, 100], torch.tensor(expected), atol=1e-5)

    def test_normalization_mean(self):
        """mean bg + std → (mean - mean) / std = 0."""
        img = _make_batch(B=1, H=64, W=64, value=0.0)
        t = RandomEmbed(canvas_size=128, x_range=0.0, y_range=0.0,
                        background="mean", mean=self.MEAN, std=self.STD)
        out = t(img)
        # Background region should be ~0 (normalised mean)
        bg_region = out[0, :, 100, 100]
        assert torch.allclose(bg_region, torch.zeros(3), atol=1e-5)

    def test_normalization_power_law(self):
        """power_law + mean/std → noise is normalised."""
        img = _make_batch(B=2, H=64, W=64, value=0.0)
        t = RandomEmbed(canvas_size=128, x_range=0.0, y_range=0.0,
                        background="power_law", alpha_range=1.5,
                        mean=self.MEAN, std=self.STD, seed=42)
        out = t(img)
        bg_region = out[:, :, 64:, 64:]
        # Normalised power_law: raw [0,1] → (raw - mean) / std
        # For ch0: (0 - 0.485) / 0.229 ≈ -2.12, (1 - 0.485) / 0.229 ≈ 2.25
        # Values should be in a reasonable normalised range
        assert bg_region.min() < 0.0  # some values below zero
        assert bg_region.max() > 0.0  # some values above zero

    def test_no_normalization_without_std(self):
        """Without std, backgrounds remain in [0, 1] space."""
        img = _make_batch(B=2, H=64, W=64, value=0.0)
        t = RandomEmbed(canvas_size=128, x_range=0.0, y_range=0.0,
                        background="power_law", alpha_range=1.5, seed=42)
        out = t(img)
        bg_region = out[:, :, 64:, 64:]
        assert bg_region.min() >= 0.0 - 1e-6
        assert bg_region.max() <= 1.0 + 1e-6


# ── seed reproducibility ────────────────────────────────────────────────

class TestSeed:
    def test_seed_reproducible(self):
        img = _make_batch(B=4, H=64, W=64)
        t1 = RandomEmbed(canvas_size=128, x_range=(0, 1), y_range=(0, 1), seed=42)
        t2 = RandomEmbed(canvas_size=128, x_range=(0, 1), y_range=(0, 1), seed=42)
        out1 = t1(img)
        out2 = t2(img)
        assert torch.equal(t1.last_params()["xs"], t2.last_params()["xs"])
        assert torch.equal(t1.last_params()["ys"], t2.last_params()["ys"])

    def test_different_seeds_differ(self):
        img = _make_batch(B=4, H=64, W=64)
        t1 = RandomEmbed(canvas_size=128, x_range=(0, 1), y_range=(0, 1), seed=42)
        t2 = RandomEmbed(canvas_size=128, x_range=(0, 1), y_range=(0, 1), seed=99)
        t1(img)
        t2(img)
        assert not torch.equal(t1.last_params()["xs"], t2.last_params()["xs"])


# ── SSL replay (last_params / apply_last) ────────────────────────────────

class TestReplay:
    def test_last_params_keys(self):
        t = RandomEmbed(canvas_size=128, x_range=(0, 1), y_range=(0, 1), seed=42)
        t(_make_batch(B=4, H=64, W=64))
        params = t.last_params()
        assert "xs" in params
        assert "ys" in params
        assert "img_h" in params
        assert "img_w" in params
        assert "x_frac" in params
        assert "y_frac" in params

    def test_last_params_keys_coords_mode(self):
        t = RandomEmbed(canvas_size=128, coords=[(64, 64)], seed=42)
        t(_make_batch(B=2, H=32, W=32))
        params = t.last_params()
        assert "xs" in params
        assert "ys" in params
        assert "x_frac" not in params  # not available in coords mode

    def test_apply_last_same_position(self):
        """apply_last on a different input uses the same stored positions."""
        t = RandomEmbed(canvas_size=128, x_range=(0, 1), y_range=(0, 1), seed=42)
        view1 = torch.randn(4, 3, 64, 64)
        view2 = torch.randn(4, 3, 64, 64)
        out1 = t(view1)          # calls before_call + apply_last
        out2 = t.apply_last(view2)  # reuses stored positions
        # Positions are the same — find image region and verify content differs
        # but location matches
        xs = t.last_params()["xs"]
        ys = t.last_params()["ys"]
        for i in range(4):
            x, y = xs[i].item(), ys[i].item()
            # Both outputs have image content at the same location
            region1 = out1[i, :, y:y + 64, x:x + 64]
            region2 = out2[i, :, y:y + 64, x:x + 64]
            # Content should come from different views
            assert torch.equal(region1, view1[i])
            assert torch.equal(region2, view2[i])


# ── circular fade ─────────────────────────────────────────────────────────

class TestFadeRadius:
    def test_fade_default_none(self):
        t = RandomEmbed(canvas_size=128)
        assert t.fade_radius is None

    def test_fade_center_opaque(self):
        """Pixels at image center (well inside inner) should equal the input."""
        img = _make_batch(B=1, C=3, H=96, W=96, value=0.7)
        t = RandomEmbed(canvas_size=224, x_range=0.5, y_range=0.5,
                        fade_radius=(0.40, 0.50))
        out = t(img)
        # Center of canvas = center of image
        cy, cx = 112, 112  # center of 224
        assert torch.allclose(out[0, :, cy, cx], torch.tensor(0.7), atol=1e-6)

    def test_fade_corner_transparent(self):
        """Pixels at image corners (beyond outer radius) should equal background."""
        img = _make_batch(B=1, C=3, H=96, W=96, value=0.7)
        t = RandomEmbed(canvas_size=224, x_range=0.5, y_range=0.5,
                        background="zeros", fade_radius=(0.40, 0.50))
        out = t(img)
        # Image placed at (64, 64). Corner of image at (64, 64) on canvas.
        # Distance from image center = sqrt(48^2 + 48^2) ≈ 67.9
        # outer = 0.50 * 96 = 48, so 67.9 >> 48 → fully transparent
        # That pixel should be background (zeros)
        assert torch.allclose(out[0, :, 64, 64], torch.tensor(0.0), atol=1e-4)

    def test_fade_creates_blend(self):
        """Pixels in the fade band differ from both pure image and background."""
        bg_val = 0.3
        img_val = 0.9
        img = _make_batch(B=1, C=3, H=96, W=96, value=img_val)
        t = RandomEmbed(canvas_size=224, x_range=0.5, y_range=0.5,
                        background="mean", mean=bg_val,
                        fade_radius=(0.30, 0.50))
        out = t(img)
        # Pick a pixel in the fade band: ~40 pixels from image center
        # inner = 0.30 * 96 = 28.8, outer = 0.50 * 96 = 48
        # 40 pixels from center is in the fade band
        cy, cx = 112, 112  # canvas center = image center
        pixel = out[0, 0, cy, cx + 40].item()
        assert bg_val < pixel < img_val, f"Expected blend, got {pixel}"

    def test_fade_validation_inner_ge_outer(self):
        with pytest.raises(ValueError, match="inner < outer"):
            RandomEmbed(canvas_size=128, fade_radius=(0.5, 0.3))

    def test_fade_validation_equal(self):
        with pytest.raises(ValueError, match="inner < outer"):
            RandomEmbed(canvas_size=128, fade_radius=(0.5, 0.5))

    def test_fade_repr(self):
        t = RandomEmbed(canvas_size=224, fade_radius=(0.45, 0.50))
        r = repr(t)
        assert "fade_radius=(0.45, 0.5)" in r

    def test_fade_repr_none_omitted(self):
        t = RandomEmbed(canvas_size=224)
        r = repr(t)
        assert "fade_radius" not in r

    def test_p_fade_default(self):
        t = RandomEmbed(canvas_size=128, fade_radius=(0.4, 0.5))
        assert t.p_fade == 1.0

    def test_p_fade_zero_no_fade(self):
        """p_fade=0 should never apply fade (identical to no fade_radius)."""
        img = _make_batch(B=4, C=3, H=96, W=96, value=0.7)
        t_no_fade = RandomEmbed(canvas_size=224, x_range=0.5, y_range=0.5,
                                background="zeros", seed=42)
        t_p0 = RandomEmbed(canvas_size=224, x_range=0.5, y_range=0.5,
                           background="zeros", fade_radius=(0.35, 0.50),
                           p_fade=0.0, seed=42)
        out_no_fade = t_no_fade(img)
        out_p0 = t_p0(img)
        assert torch.equal(out_no_fade, out_p0)

    def test_p_fade_one_always_fades(self):
        """p_fade=1 should fade every image."""
        img = _make_batch(B=4, C=3, H=96, W=96, value=0.7)
        t = RandomEmbed(canvas_size=224, x_range=0.5, y_range=0.5,
                        background="zeros", fade_radius=(0.35, 0.50),
                        p_fade=1.0, seed=42)
        out = t(img)
        params = t.last_params()
        assert params["do_fade"].all()

    def test_p_fade_half_mixed(self):
        """p_fade=0.5 with large batch should produce a mix of faded and unfaded."""
        img = _make_batch(B=32, C=3, H=96, W=96, value=0.7)
        t = RandomEmbed(canvas_size=224, x_range=0.5, y_range=0.5,
                        background="zeros", fade_radius=(0.35, 0.50),
                        p_fade=0.5, seed=42)
        t(img)
        do_fade = t.last_params()["do_fade"]
        n_faded = do_fade.sum().item()
        # With 32 images at p=0.5, expect some faded and some not
        assert 0 < n_faded < 32, f"Expected a mix, got {n_faded}/32 faded"

    def test_p_fade_repr(self):
        t = RandomEmbed(canvas_size=224, fade_radius=(0.4, 0.5), p_fade=0.5)
        r = repr(t)
        assert "p_fade=0.5" in r

    def test_p_fade_repr_default_omitted(self):
        t = RandomEmbed(canvas_size=224, fade_radius=(0.4, 0.5))
        r = repr(t)
        assert "p_fade" not in r


# ── edge cases ───────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_image_equals_canvas(self):
        img = _make_batch(B=1, H=128, W=128, value=1.0)
        t = RandomEmbed(canvas_size=128)
        out = t(img)
        assert torch.equal(out, img)

    def test_image_larger_than_canvas_range_mode(self):
        """When image is larger than canvas, slack is 0 → placed at origin, clipped."""
        img = _make_batch(B=1, H=256, W=256, value=1.0)
        t = RandomEmbed(canvas_size=128)
        out = t(img)
        # Output should be 128x128 filled with the top-left 128x128 of input
        assert out.shape == (1, 3, 128, 128)
        assert torch.equal(out, img[:, :, :128, :128])

    def test_rectangular_image_on_square_canvas(self):
        img = _make_batch(B=2, H=64, W=128, value=1.0)
        t = RandomEmbed(canvas_size=224, x_range=0.5, y_range=0.5)
        out = t(img)
        assert out.shape == (2, 3, 224, 224)

    def test_batch_size_1(self):
        img = _make_batch(B=1, H=64, W=64)
        t = RandomEmbed(canvas_size=128, x_range=(0, 1), y_range=(0, 1), seed=42)
        out = t(img)
        assert out.shape == (1, 3, 128, 128)


# ── repr ─────────────────────────────────────────────────────────────────

class TestRepr:
    def test_repr_range_mode(self):
        t = RandomEmbed(canvas_size=224, x_range=(0, 1), background="zeros", seed=42)
        r = repr(t)
        assert "RandomEmbed" in r
        assert "224" in r
        assert "zeros" in r
        assert "seed=42" in r

    def test_repr_coords_mode(self):
        t = RandomEmbed(canvas_size=224, coords=[(100, 100), (50, 50)])
        r = repr(t)
        assert "2 positions" in r

    def test_repr_power_law(self):
        t = RandomEmbed(canvas_size=224, background="power_law", alpha_range=(1, 2))
        r = repr(t)
        assert "power_law" in r
        assert "alpha_range" in r

    def test_repr_with_std(self):
        t = RandomEmbed(canvas_size=224, background="power_law",
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        r = repr(t)
        assert "std=" in r

    def test_repr_color_noise_false(self):
        t = RandomEmbed(canvas_size=224, background="power_law", color_noise=False)
        r = repr(t)
        assert "color_noise=False" in r

    def test_repr_color_noise_true_omitted(self):
        """Default color_noise=True should not appear in repr."""
        t = RandomEmbed(canvas_size=224, background="power_law")
        r = repr(t)
        assert "color_noise" not in r
