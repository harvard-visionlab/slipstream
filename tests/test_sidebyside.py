"""Tests for SideBySideSearchPair transform."""

import numpy as np
import pytest
import torch

from slipstream.transforms import SideBySideSearchPair


B, C, H, W, K = 8, 3, 32, 32, 10


def make_batch(batch_size=B, num_classes=K, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "image": torch.rand(batch_size, C, H, W, generator=g),
        "label": torch.randint(0, num_classes, (batch_size,), generator=g),
    }


# ---------- Construction & validation ----------

class TestValidation:
    def test_bad_p_left_raises(self):
        with pytest.raises(ValueError, match="p_left"):
            SideBySideSearchPair(num_classes=K, p_left=1.5)

    def test_output_key_collision_with_image_raises(self):
        with pytest.raises(ValueError, match="collides"):
            SideBySideSearchPair(num_classes=K, target_side_key="image")

    def test_duplicate_output_keys_raise(self):
        with pytest.raises(ValueError, match="distinct"):
            SideBySideSearchPair(num_classes=K, valid_key="target_side")

    def test_small_batch_raises(self):
        t = SideBySideSearchPair(num_classes=K, seed=0)
        with pytest.raises(ValueError, match="B >= 2"):
            t(make_batch(batch_size=1))

    def test_uint8_raises(self):
        t = SideBySideSearchPair(num_classes=K)
        batch = {
            "image": torch.randint(0, 256, (B, C, H, W), dtype=torch.uint8),
            "label": torch.randint(0, K, (B,)),
        }
        with pytest.raises(ValueError, match="floating-point"):
            t(batch)

    def test_3d_image_raises(self):
        t = SideBySideSearchPair(num_classes=K)
        with pytest.raises(ValueError, match="4D image tensor"):
            t({"image": torch.rand(C, H, W), "label": torch.tensor(3)})

    def test_missing_keys_raise(self):
        t = SideBySideSearchPair(num_classes=K)
        with pytest.raises(KeyError, match="image_key"):
            t({"label": torch.zeros(B)})
        with pytest.raises(KeyError, match="label_key"):
            t({"image": torch.rand(B, C, H, W)})


# ---------- Shape / composite content ----------

class TestComposite:
    def test_composite_shape(self):
        t = SideBySideSearchPair(num_classes=K, seed=0)
        out = t(make_batch())
        assert out["image"].shape == (B, C, H, 2 * W)
        assert out["image"].dtype == torch.float32

    def test_halves_match_source_for_assigned_side(self):
        b = make_batch()
        src = b["image"].clone()
        t = SideBySideSearchPair(num_classes=K, seed=0)
        out = t(b)
        comp = out["image"]
        left, right = comp[:, :, :, :W], comp[:, :, :, W:]
        side = t.last_target_side
        partner = t.last_partner_index
        for i in range(B):
            target_half, distractor_half = (left[i], right[i]) if side[i] == 0 else (right[i], left[i])
            assert torch.equal(target_half, src[i]), f"target half mismatch at {i}"
            assert torch.equal(distractor_half, src[partner[i]]), f"distractor half mismatch at {i}"

    def test_label_is_unchanged_target_label(self):
        b = make_batch()
        orig = b["label"].clone()
        t = SideBySideSearchPair(num_classes=K, seed=0)
        out = t(b)
        assert torch.equal(out["label"].cpu(), orig)

    def test_distractor_label_matches_partner(self):
        b = make_batch()
        orig = b["label"].clone()
        t = SideBySideSearchPair(num_classes=K, seed=0)
        out = t(b)
        partner = t.last_partner_index
        assert torch.equal(out["distractor_label"].cpu(), orig[partner])


# ---------- Pairing modes ----------

class TestPairing:
    def test_derangement_no_self_pairs_and_all_valid(self):
        t = SideBySideSearchPair(num_classes=K, seed=0)
        out = t(make_batch())
        partner = t.last_partner_index
        assert not np.any(partner == np.arange(B))
        assert out["valid"].all()
        # permutation: every index used exactly once
        assert sorted(partner.tolist()) == list(range(B))

    def test_require_different_class_valid_implies_different(self):
        # many classes, large batch → most pairs differ
        b = make_batch(batch_size=64, num_classes=K, seed=1)
        labels = b["label"].clone()
        t = SideBySideSearchPair(num_classes=K, seed=0, require_different_class=True)
        out = t(b)
        partner = t.last_partner_index
        valid = out["valid"].cpu().numpy()
        for i in range(64):
            if valid[i]:
                assert labels[i] != labels[partner[i]]
            else:
                assert labels[i] == labels[partner[i]]
        # no self-pairs regardless
        assert not np.any(partner == np.arange(64))

    def test_p_left_distribution(self):
        t = SideBySideSearchPair(num_classes=K, seed=0, p_left=0.5)
        sides = []
        for s in range(50):
            sides.append(t(make_batch(batch_size=64, seed=s))["target_side"].cpu().numpy())
        frac_left = (np.concatenate(sides) == 0).mean()
        assert 0.45 < frac_left < 0.55

    def test_p_left_extremes(self):
        t = SideBySideSearchPair(num_classes=K, seed=0, p_left=1.0)
        out = t(make_batch(batch_size=32))
        assert (out["target_side"] == 0).all()
        t = SideBySideSearchPair(num_classes=K, seed=0, p_left=0.0)
        out = t(make_batch(batch_size=32))
        assert (out["target_side"] == 1).all()


# ---------- Determinism ----------

class TestDeterminism:
    def test_same_seed_same_output(self):
        t1 = SideBySideSearchPair(num_classes=K, seed=42)
        t2 = SideBySideSearchPair(num_classes=K, seed=42)
        o1 = t1(make_batch(seed=3))
        o2 = t2(make_batch(seed=3))
        assert torch.equal(o1["image"], o2["image"])
        assert np.array_equal(t1.last_target_side, t2.last_target_side)
        assert np.array_equal(t1.last_partner_index, t2.last_partner_index)

    def test_rng_advances_across_calls(self):
        t = SideBySideSearchPair(num_classes=K, seed=42)
        t(make_batch(seed=3))
        first = t.last_partner_index.copy()
        t(make_batch(seed=3))
        second = t.last_partner_index
        # advancing RNG → different pairing on the identical batch (overwhelmingly likely)
        assert not np.array_equal(first, second)
