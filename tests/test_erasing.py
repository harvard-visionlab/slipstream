"""Tests for RandomErasing transform."""

import math
import pytest
import torch

from slipstream.transforms import RandomErasing


B, C, H, W = 8, 3, 32, 32


def make_batch_float():
    return torch.rand(B, C, H, W)


def make_batch_uint8():
    return torch.randint(0, 256, (B, C, H, W), dtype=torch.uint8)


def make_image_float():
    return torch.rand(C, H, W)


# ---------- Shape / dtype / device ----------

class TestShapePreservation:
    def test_batch_float(self):
        b = make_batch_float()
        out = RandomErasing(p=1.0)(b.clone())
        assert out.shape == b.shape
        assert out.dtype == b.dtype

    def test_batch_uint8_uniform(self):
        b = make_batch_uint8()
        out = RandomErasing(p=1.0, fill="uniform")(b.clone())
        assert out.shape == b.shape
        assert out.dtype == torch.uint8

    def test_single_3d(self):
        img = make_image_float()
        out = RandomErasing(p=1.0)(img.clone())
        assert out.shape == img.shape

    def test_each_mode(self):
        b = make_batch_float()
        for mode in ("zeros", "random_color_uniform", "random_color_pixel"):
            out = RandomErasing(p=1.0, mode=mode)(b.clone())
            assert out.shape == b.shape


# ---------- Probability ----------

class TestProbability:
    def test_p0_identity(self):
        b = make_batch_float()
        out = RandomErasing(p=0.0)(b.clone())
        assert torch.equal(out, b)

    def test_p1_modifies(self):
        b = make_batch_float()
        out = RandomErasing(p=1.0, mode="zeros")(b.clone())
        # At least one pixel must have changed in every image
        diff = (out != b).any(dim=1)  # [B, H, W]
        assert diff.view(B, -1).any(dim=1).all()

    def test_p_half_partial_erase(self):
        # With p=0.5 and many trials, expect some images touched, some not.
        torch.manual_seed(0)
        n_total = 256
        b = torch.rand(n_total, C, H, W)
        out = RandomErasing(p=0.5, mode="zeros", seed=0, device="cpu")(b.clone())
        per_image_changed = (out != b).view(n_total, -1).any(dim=1)
        n_changed = int(per_image_changed.sum())
        # Loose bounds — we just want to confirm it's truly per-sample.
        assert 80 < n_changed < 175


# ---------- Per-sample randomness ----------

class TestPerSampleRandomness:
    def test_different_rectangles_per_sample(self):
        # In a batch erased with mode=zeros, the set-of-zero pixels should differ across images.
        b = torch.rand(16, C, H, W)
        out = RandomErasing(p=1.0, mode="zeros", seed=42, device="cpu")(b.clone())
        zero_masks = (out == 0).all(dim=1)  # [N, H, W] — true where erased
        # Compare image 0's mask against the other 15 — at least most should differ
        diffs = [not torch.equal(zero_masks[0], zero_masks[i]) for i in range(1, 16)]
        assert sum(diffs) >= 13


# ---------- Mode semantics ----------

class TestModeSemantics:
    def test_zeros_mode_fills_zero(self):
        b = torch.rand(B, C, H, W).add_(1.0)  # all > 1, never zero naturally
        out = RandomErasing(p=1.0, mode="zeros")(b.clone())
        # Every image should have at least one zero pixel (the erased region)
        for i in range(B):
            assert (out[i] == 0).any()

    def test_random_color_uniform_constant_within_rect(self):
        # In random_color_uniform mode, the erased rectangle should have a constant
        # color (per channel) — i.e., variance across the rect is 0 for each channel.
        b = torch.rand(B, C, H, W)
        t = RandomErasing(p=1.0, mode="random_color_uniform", fill="uniform", seed=0, device="cpu")
        out = t(b.clone())
        params = t.last_params()
        for i in range(B):
            top, left = int(params["top"][i]), int(params["left"][i])
            eh, ew = int(params["eh"][i]), int(params["ew"][i])
            patch = out[i, :, top:top + eh, left:left + ew]
            for c in range(C):
                assert patch[c].std().item() < 1e-6  # constant per channel

    def test_random_color_pixel_has_variance(self):
        b = torch.zeros(B, C, H, W)  # zero base so any noise is detectable
        t = RandomErasing(p=1.0, mode="random_color_pixel", fill="randn", seed=0, device="cpu")
        out = t(b.clone())
        params = t.last_params()
        for i in range(B):
            top, left = int(params["top"][i]), int(params["left"][i])
            eh, ew = int(params["eh"][i]), int(params["ew"][i])
            if eh * ew >= 4:
                patch = out[i, :, top:top + eh, left:left + ew]
                # Per-pixel noise → patch std must be substantially > 0
                assert patch.std().item() > 0.1


# ---------- Rectangle bounds ----------

class TestBounds:
    def test_rect_within_image(self):
        t = RandomErasing(p=1.0, seed=0, device="cpu")
        b = torch.rand(32, C, H, W)
        t(b.clone())
        p = t.last_params()
        assert (p["top"] >= 0).all() and (p["top"] + p["eh"] <= H).all()
        assert (p["left"] >= 0).all() and (p["left"] + p["ew"] <= W).all()
        assert (p["eh"] >= 1).all() and (p["ew"] >= 1).all()


# ---------- Replay / determinism ----------

class TestReplay:
    def test_replay_match(self):
        b = torch.rand(B, C, H, W)
        t = RandomErasing(p=1.0, seed=42, device="cpu")
        out1 = t(b.clone())
        out2 = t.apply_last(b.clone())
        assert torch.equal(out1, out2)

    def test_seed_determinism(self):
        b = torch.rand(B, C, H, W)
        t1 = RandomErasing(p=0.5, seed=123, device="cpu")
        t2 = RandomErasing(p=0.5, seed=123, device="cpu")
        out1 = t1(b.clone())
        out2 = t2(b.clone())
        assert torch.equal(out1, out2)

    def test_different_seeds_differ(self):
        b = torch.rand(B, C, H, W)
        out1 = RandomErasing(p=1.0, seed=1, device="cpu")(b.clone())
        out2 = RandomErasing(p=1.0, seed=2, device="cpu")(b.clone())
        assert not torch.equal(out1, out2)


# ---------- Single-image (3D) parity ----------

class TestSingleImageParity:
    def test_3d_matches_batched_of_one(self):
        img = torch.rand(C, H, W)
        t1 = RandomErasing(p=1.0, mode="random_color_pixel", fill="randn", seed=7, device="cpu")
        t2 = RandomErasing(p=1.0, mode="random_color_pixel", fill="randn", seed=7, device="cpu")
        out_3d = t1(img.clone())
        out_4d = t2(img.unsqueeze(0).clone()).squeeze(0)
        assert torch.equal(out_3d, out_4d)


# ---------- Validation / errors ----------

class TestValidation:
    def test_uint8_with_randn_raises(self):
        b = make_batch_uint8()
        t = RandomErasing(p=1.0, mode="random_color_pixel", fill="randn")
        with pytest.raises(ValueError, match="randn"):
            t(b.clone())

    def test_invalid_mode(self):
        with pytest.raises(ValueError, match="mode"):
            RandomErasing(mode="bad")

    def test_invalid_fill(self):
        with pytest.raises(ValueError, match="fill"):
            RandomErasing(fill="bad")

    def test_zeros_mode_ignores_fill_choice(self):
        # mode="zeros" should work even if fill="randn" with uint8 (fill is unused).
        b = make_batch_uint8()
        out = RandomErasing(p=1.0, mode="zeros", fill="randn")(b.clone())
        assert out.shape == b.shape


# ---------- repr ----------

def test_repr():
    r = repr(RandomErasing(p=0.5))
    assert "RandomErasing" in r
    assert "p=0.5" in r
