"""Tests for Mixup / CutMix transform."""

import numpy as np
import pytest
import torch

from slipstream.transforms import Mixup, mixup_target, one_hot


B, C, H, W, K = 8, 3, 32, 32, 10


def make_batch(batch_size=B, num_classes=K, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "image": torch.rand(batch_size, C, H, W, generator=g),
        "label": torch.randint(0, num_classes, (batch_size,), generator=g),
    }


# ---------- Construction & validation ----------

class TestValidation:
    def test_all_zero_alphas_raises(self):
        with pytest.raises(ValueError, match="At least one"):
            Mixup(mixup_alpha=0, cutmix_alpha=0, cutmix_minmax=None)

    def test_minmax_must_be_pair(self):
        with pytest.raises(ValueError, match="cutmix_minmax"):
            Mixup(cutmix_minmax=(0.2,))

    def test_minmax_forces_cutmix_alpha_to_one(self):
        m = Mixup(mixup_alpha=0, cutmix_alpha=0, cutmix_minmax=(0.2, 0.5))
        assert m.cutmix_alpha == 1.0

    def test_odd_batch_raises(self):
        m = Mixup(num_classes=K)
        batch = make_batch(batch_size=7)
        with pytest.raises(ValueError, match="even batch size"):
            m(batch)

    def test_uint8_raises(self):
        m = Mixup(num_classes=K)
        batch = {
            "image": torch.randint(0, 256, (B, C, H, W), dtype=torch.uint8),
            "label": torch.randint(0, K, (B,)),
        }
        with pytest.raises(ValueError, match="floating-point"):
            m(batch)

    def test_3d_image_raises(self):
        m = Mixup(num_classes=K)
        batch = {"image": torch.rand(C, H, W), "label": torch.tensor(3)}
        with pytest.raises(ValueError, match="4D image tensor"):
            m(batch)

    def test_missing_keys_raise(self):
        m = Mixup(num_classes=K)
        with pytest.raises(KeyError, match="image_key"):
            m({"label": torch.zeros(B)})
        with pytest.raises(KeyError, match="label_key"):
            m({"image": torch.rand(B, C, H, W)})


# ---------- Shape / dtype ----------

class TestShape:
    def test_image_shape_preserved(self):
        m = Mixup(num_classes=K, seed=0)
        b = make_batch()
        out = m(b)
        assert out["image"].shape == (B, C, H, W)
        assert out["image"].dtype == torch.float32

    def test_label_becomes_onehot_float(self):
        m = Mixup(num_classes=K, seed=0)
        b = make_batch()
        out = m(b)
        assert out["label"].shape == (B, K)
        assert out["label"].dtype == torch.float32

    def test_label_sums_to_one(self):
        m = Mixup(num_classes=K, seed=0)
        b = make_batch()
        out = m(b)
        sums = out["label"].sum(dim=-1)
        assert torch.allclose(sums, torch.ones(B), atol=1e-5)

    def test_extra_keys_preserved(self):
        m = Mixup(num_classes=K, seed=0)
        extra = torch.arange(B)
        b = make_batch()
        b["_indices"] = extra
        b["other"] = "passthrough"
        out = m(b)
        assert torch.equal(out["_indices"], extra)
        assert out["other"] == "passthrough"


# ---------- Per-sample randomness (elem semantics) ----------

class TestPerSampleRandomness:
    def test_mix_of_ops_in_one_batch(self):
        # With both alphas active, switch_prob=0.5, prob<1, and a moderate batch,
        # we should see at least one of each: mixup, cutmix, no-op.
        m = Mixup(mixup_alpha=0.8, cutmix_alpha=1.0, prob=0.7, switch_prob=0.5,
                  num_classes=K, seed=42)
        b = make_batch(batch_size=64)
        m(b)
        n_mixup = m.last_use_mixup.sum()
        n_cutmix = m.last_use_cutmix.sum()
        n_noop = (~m.last_use_mixup & ~m.last_use_cutmix).sum()
        assert n_mixup > 0, f"expected some mixup samples, got {n_mixup}"
        assert n_cutmix > 0, f"expected some cutmix samples, got {n_cutmix}"
        assert n_noop > 0, f"expected some no-op samples, got {n_noop}"

    def test_lam_varies_per_sample(self):
        m = Mixup(mixup_alpha=0.8, cutmix_alpha=0, num_classes=K, seed=1)
        b = make_batch(batch_size=32)
        m(b)
        # All samples mix (prob=1), all use mixup, λ from Beta — must vary.
        assert m.last_lam.std().item() > 0.05

    def test_prob_zero_is_identity(self):
        m = Mixup(mixup_alpha=0.8, cutmix_alpha=1.0, prob=0.0, num_classes=K, seed=0)
        b = make_batch()
        original_image = b["image"].clone()
        original_label = b["label"].clone()
        out = m(b)
        # Image untouched (λ=1 for all → x*1 + x_flip*0 = x)
        assert torch.allclose(out["image"], original_image, atol=1e-6)
        # Label is one-hot of original (no mixing)
        expected = torch.nn.functional.one_hot(original_label, K).float()
        assert torch.allclose(out["label"], expected)

    def test_prob_partial_some_untouched(self):
        m = Mixup(mixup_alpha=0.8, cutmix_alpha=0.0, prob=0.5, num_classes=K, seed=2)
        b = make_batch(batch_size=64)
        original_image = b["image"].clone()
        out = m(b)
        # Samples that didn't roll mix have λ=1 → output equals input bit-exactly
        unchanged = torch.tensor([
            torch.equal(out["image"][i], original_image[i]) for i in range(64)
        ])
        # Should have a meaningful number unchanged (~32 expected, plenty of slack)
        assert 10 < unchanged.sum().item() < 55


# ---------- Mixup-only path ----------

class TestMixupOnly:
    def test_mathematical_identity(self):
        m = Mixup(mixup_alpha=0.8, cutmix_alpha=0.0, prob=1.0, num_classes=K, seed=7)
        b = make_batch()
        x_in = b["image"].clone()
        out = m(b)
        lam = m.last_lam.view(B, 1, 1, 1)
        expected = x_in * lam + x_in.flip(0) * (1.0 - lam)
        assert torch.allclose(out["image"], expected, atol=1e-5)

    def test_label_mixing(self):
        m = Mixup(mixup_alpha=0.8, cutmix_alpha=0.0, prob=1.0, num_classes=K, seed=7)
        b = make_batch()
        labels_in = b["label"].clone()
        out = m(b)
        lam = m.last_lam
        y1 = torch.nn.functional.one_hot(labels_in, K).float()
        y2 = torch.nn.functional.one_hot(labels_in.flip(0), K).float()
        expected = y1 * lam.view(-1, 1) + y2 * (1.0 - lam.view(-1, 1))
        assert torch.allclose(out["label"], expected, atol=1e-5)


# ---------- CutMix-only path ----------

class TestCutMixOnly:
    def test_pixels_outside_bbox_unchanged(self):
        m = Mixup(mixup_alpha=0.0, cutmix_alpha=1.0, prob=1.0, num_classes=K, seed=3)
        b = make_batch()
        x_in = b["image"].clone()
        out = m(b)
        bbox = m.last_bbox
        assert bbox is not None
        yl, yh, xl, xh = bbox
        # Build outside-bbox mask and check values unchanged there.
        for i in range(B):
            mask = torch.zeros(H, W, dtype=torch.bool)
            mask[int(yl[i]):int(yh[i]), int(xl[i]):int(xh[i])] = True
            outside = ~mask
            assert torch.equal(out["image"][i, :, outside], x_in[i, :, outside])

    def test_pixels_inside_bbox_come_from_flip(self):
        m = Mixup(mixup_alpha=0.0, cutmix_alpha=1.0, prob=1.0, num_classes=K, seed=4)
        b = make_batch()
        x_in = b["image"].clone()
        x_flip = x_in.flip(0)
        out = m(b)
        yl, yh, xl, xh = m.last_bbox
        for i in range(B):
            t, btm, l, r = int(yl[i]), int(yh[i]), int(xl[i]), int(xh[i])
            if btm > t and r > l:
                assert torch.equal(out["image"][i, :, t:btm, l:r],
                                   x_flip[i, :, t:btm, l:r])

    def test_correct_lam_matches_bbox_area(self):
        m = Mixup(mixup_alpha=0.0, cutmix_alpha=1.0, correct_lam=True,
                  prob=1.0, num_classes=K, seed=5)
        b = make_batch()
        m(b)
        yl, yh, xl, xh = m.last_bbox
        bbox_area = (yh - yl) * (xh - xl)
        expected_lam = 1.0 - bbox_area / float(H * W)
        actual = m.last_lam.numpy()
        assert np.allclose(actual, expected_lam, atol=1e-5)

    def test_minmax_bbox_within_bounds(self):
        m = Mixup(mixup_alpha=0.0, cutmix_alpha=0.0, cutmix_minmax=(0.3, 0.5),
                  prob=1.0, num_classes=K, seed=6)
        b = make_batch()
        m(b)
        yl, yh, xl, xh = m.last_bbox
        cut_h = yh - yl
        cut_w = xh - xl
        assert (cut_h >= int(H * 0.3)).all()
        assert (cut_h <= int(H * 0.5)).all()
        assert (cut_w >= int(W * 0.3)).all()
        assert (cut_w <= int(W * 0.5)).all()


# ---------- Label smoothing ----------

class TestLabelSmoothing:
    def test_smoothed_labels_max_below_one(self):
        m = Mixup(mixup_alpha=0.8, cutmix_alpha=0.0, prob=1.0,
                  label_smoothing=0.1, num_classes=K, seed=0)
        b = make_batch()
        out = m(b)
        # With smoothing > 0, no entry should be exactly 1.0
        assert out["label"].max().item() < 1.0
        # Sums still ~1.0
        assert torch.allclose(out["label"].sum(dim=-1), torch.ones(B), atol=1e-5)

    def test_smoothing_zero_gives_pure_onehot_at_lam1(self):
        # mixup_target with λ=1 and smoothing=0 → exact one-hot
        target = torch.tensor([0, 3, 7, 5])
        out = mixup_target(target, num_classes=K, lam=torch.ones(4), smoothing=0.0)
        expected = torch.nn.functional.one_hot(target, K).float()
        assert torch.equal(out, expected)


# ---------- Determinism ----------

class TestDeterminism:
    def test_same_seed_same_output(self):
        b1 = make_batch(seed=99)
        b2 = make_batch(seed=99)
        m1 = Mixup(mixup_alpha=0.8, cutmix_alpha=1.0, num_classes=K, seed=42)
        m2 = Mixup(mixup_alpha=0.8, cutmix_alpha=1.0, num_classes=K, seed=42)
        out1 = m1(b1)
        out2 = m2(b2)
        assert torch.equal(out1["image"], out2["image"])
        assert torch.equal(out1["label"], out2["label"])

    def test_different_seeds_differ(self):
        b1 = make_batch(seed=99)
        b2 = make_batch(seed=99)
        out1 = Mixup(num_classes=K, seed=1)(b1)
        out2 = Mixup(num_classes=K, seed=2)(b2)
        assert not torch.equal(out1["image"], out2["image"])


# ---------- one_hot helper ----------

class TestOneHot:
    def test_basic(self):
        t = torch.tensor([0, 1, 2])
        out = one_hot(t, num_classes=4)
        assert out.shape == (3, 4)
        expected = torch.eye(4)[:3]
        assert torch.equal(out, expected)

    def test_smoothing_values(self):
        t = torch.tensor([2])
        out = one_hot(t, num_classes=5, on_value=0.92, off_value=0.02)
        expected = torch.tensor([[0.02, 0.02, 0.92, 0.02, 0.02]])
        assert torch.allclose(out, expected)


# ---------- repr ----------

def test_repr():
    r = repr(Mixup(num_classes=K))
    assert "Mixup" in r
    assert "mixup_alpha" in r


# ---------- Loader integration ----------

class TestLoaderIntegration:
    def test_after_batch_transforms_runs(self):
        """A custom callable in after_batch_transforms should mutate the batch."""
        from slipstream.loader import SlipstreamLoader
        # We can't easily build a full loader here without a dataset; instead,
        # exercise the documented contract by direct callable composition.
        called = []

        def tagger(batch):
            called.append(True)
            batch["tagged"] = True
            return batch

        b = make_batch()
        # Compose tagger then Mixup, simulating after_batch_transforms execution
        m = Mixup(num_classes=K, seed=0)
        for t in [tagger, m]:
            b = t(b)
        assert called == [True]
        assert b["tagged"] is True
        assert b["label"].shape == (B, K)

    def test_loader_accepts_after_batch_transforms_arg(self):
        """Verify the arg exists on SlipstreamLoader (signature smoke test)."""
        import inspect
        from slipstream.loader import SlipstreamLoader
        sig = inspect.signature(SlipstreamLoader.__init__)
        assert "after_batch_transforms" in sig.parameters
