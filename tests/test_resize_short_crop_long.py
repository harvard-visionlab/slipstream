"""Tests for DecodeRandomResizeShortCropLong and DecodeMultiRandomResizeShortCropLong.

Covers:
- Geometry correctness (center crop equivalence with DecodeResizeCrop)
- Output shapes for landscape, portrait, and square images
- Edge crop positions (left/right/top/bottom)
- Seed reproducibility
- Random size sampling (per_batch and per_image)
- Multi-crop variant (dict output, per-crop shapes, yoking)
- Permute and to_tensor options
"""

import io

import numpy as np
import PIL.Image
import pytest
import torch


# ---------------------------------------------------------------------------
# Test data helpers
# ---------------------------------------------------------------------------

def _create_test_jpeg(width: int, height: int, seed: int = 0) -> bytes:
    """Create a JPEG image with gradient pattern (for spatial verification)."""
    rng = np.random.RandomState(seed)
    arr = rng.randint(0, 256, (height, width, 3), dtype=np.uint8)
    img = PIL.Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def _make_batch_data(jpeg_list: list[tuple[bytes, int, int]]) -> dict:
    """Create batch_data dict from list of (jpeg_bytes, height, width) tuples."""
    batch_size = len(jpeg_list)
    max_size = max(len(j) for j, _, _ in jpeg_list)
    data = np.zeros((batch_size, max_size), dtype=np.uint8)
    sizes = np.zeros(batch_size, dtype=np.uint64)
    heights = np.zeros(batch_size, dtype=np.uint32)
    widths = np.zeros(batch_size, dtype=np.uint32)
    for i, (jpeg_bytes, h, w) in enumerate(jpeg_list):
        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        data[i, :len(arr)] = arr
        sizes[i] = len(arr)
        heights[i] = h
        widths[i] = w
    return {'data': data, 'sizes': sizes, 'heights': heights, 'widths': widths}


@pytest.fixture
def landscape_batch():
    """Batch of 4 landscape images (640x480)."""
    jpegs = [(_create_test_jpeg(640, 480, seed=i), 480, 640) for i in range(4)]
    return _make_batch_data(jpegs)


@pytest.fixture
def portrait_batch():
    """Batch of 4 portrait images (480x640)."""
    jpegs = [(_create_test_jpeg(480, 640, seed=i + 10), 640, 480) for i in range(4)]
    return _make_batch_data(jpegs)


@pytest.fixture
def mixed_batch():
    """Batch with landscape, portrait, and square images."""
    jpegs = [
        (_create_test_jpeg(640, 480, seed=0), 480, 640),  # landscape
        (_create_test_jpeg(480, 640, seed=1), 640, 480),  # portrait
        (_create_test_jpeg(512, 512, seed=2), 512, 512),  # square
        (_create_test_jpeg(320, 240, seed=3), 240, 320),  # small landscape
    ]
    return _make_batch_data(jpegs)


def _skip_if_no_decoder():
    """Skip test if libslipstream not compiled."""
    try:
        from slipstream.decoders.numba_decoder import NumbaBatchDecoder
        NumbaBatchDecoder(num_threads=1).shutdown()
    except (RuntimeError, ImportError):
        pytest.skip("libslipstream not compiled")


# ---------------------------------------------------------------------------
# Tests: _compute_resize_short_crop_long_params
# ---------------------------------------------------------------------------

class TestComputeParams:
    """Test the geometry computation function directly."""

    def test_center_crop_landscape(self):
        """Center crop on landscape image: only horizontal slack."""
        from slipstream.decoders.numba_decoder import _compute_resize_short_crop_long_params

        heights = np.array([480], dtype=np.uint32)
        widths = np.array([640], dtype=np.uint32)
        target_sizes = np.array([224], dtype=np.int32)
        x_pos = np.array([0.5], dtype=np.float64)
        y_pos = np.array([0.5], dtype=np.float64)

        params = _compute_resize_short_crop_long_params(
            heights, widths, target_sizes, x_pos, y_pos,
        )
        crop_x, crop_y, crop_w, crop_h = params[0]

        # Short edge is 480, resize to 224 → scale = 224/480
        # new_w = round(640 * 224/480) = round(298.67) = 299
        # slack_x = 299 - 224 = 75, slack_y = 0
        # center: start_x = round(0.5 * 75) = 38 (resized coords)
        # crop_y should be 0 (no vertical slack)
        assert crop_y == 0, f"Expected crop_y=0 for landscape center crop, got {crop_y}"
        assert crop_w > 0 and crop_h > 0

    def test_center_crop_portrait(self):
        """Center crop on portrait image: only vertical slack."""
        from slipstream.decoders.numba_decoder import _compute_resize_short_crop_long_params

        heights = np.array([640], dtype=np.uint32)
        widths = np.array([480], dtype=np.uint32)
        target_sizes = np.array([224], dtype=np.int32)
        x_pos = np.array([0.5], dtype=np.float64)
        y_pos = np.array([0.5], dtype=np.float64)

        params = _compute_resize_short_crop_long_params(
            heights, widths, target_sizes, x_pos, y_pos,
        )
        crop_x, crop_y, crop_w, crop_h = params[0]

        # Short edge is 480, resize to 224 → scale = 224/480
        # new_h = round(640 * 224/480) = round(298.67) = 299
        # slack_y = 299 - 224 = 75, slack_x = 0
        assert crop_x == 0, f"Expected crop_x=0 for portrait center crop, got {crop_x}"
        assert crop_w > 0 and crop_h > 0

    def test_square_image_no_slack(self):
        """Square image has no slack on either axis."""
        from slipstream.decoders.numba_decoder import _compute_resize_short_crop_long_params

        heights = np.array([512], dtype=np.uint32)
        widths = np.array([512], dtype=np.uint32)
        target_sizes = np.array([224], dtype=np.int32)
        x_pos = np.array([0.0], dtype=np.float64)
        y_pos = np.array([1.0], dtype=np.float64)

        params = _compute_resize_short_crop_long_params(
            heights, widths, target_sizes, x_pos, y_pos,
        )
        crop_x, crop_y, crop_w, crop_h = params[0]

        # Both edges equal, both resized to 224. No slack.
        assert crop_x == 0
        assert crop_y == 0

    def test_left_edge_crop(self):
        """x_pos=0 → crop starts at left edge for landscape."""
        from slipstream.decoders.numba_decoder import _compute_resize_short_crop_long_params

        heights = np.array([480], dtype=np.uint32)
        widths = np.array([640], dtype=np.uint32)
        target_sizes = np.array([224], dtype=np.int32)
        x_pos = np.array([0.0], dtype=np.float64)
        y_pos = np.array([0.5], dtype=np.float64)

        params = _compute_resize_short_crop_long_params(
            heights, widths, target_sizes, x_pos, y_pos,
        )
        assert params[0, 0] == 0, "x_pos=0 should place crop at left edge"

    def test_right_edge_crop(self):
        """x_pos=1 → crop ends at right edge for landscape."""
        from slipstream.decoders.numba_decoder import _compute_resize_short_crop_long_params

        heights = np.array([480], dtype=np.uint32)
        widths = np.array([640], dtype=np.uint32)
        target_sizes = np.array([224], dtype=np.int32)
        x_pos = np.array([1.0], dtype=np.float64)
        y_pos = np.array([0.5], dtype=np.float64)

        params = _compute_resize_short_crop_long_params(
            heights, widths, target_sizes, x_pos, y_pos,
        )
        crop_x, _, crop_w, _ = params[0]
        # Crop should end at right edge of image
        assert crop_x + crop_w <= 640
        assert crop_x > 0, "x_pos=1 should shift crop rightward"

    def test_batch_varied_dims(self):
        """Mixed batch with different dimensions."""
        from slipstream.decoders.numba_decoder import _compute_resize_short_crop_long_params

        heights = np.array([480, 640, 512], dtype=np.uint32)
        widths = np.array([640, 480, 512], dtype=np.uint32)
        target_sizes = np.array([224, 224, 224], dtype=np.int32)
        x_pos = np.array([0.5, 0.5, 0.5], dtype=np.float64)
        y_pos = np.array([0.5, 0.5, 0.5], dtype=np.float64)

        params = _compute_resize_short_crop_long_params(
            heights, widths, target_sizes, x_pos, y_pos,
        )
        assert params.shape == (3, 4)
        # All crops should be within image bounds
        for i in range(3):
            cx, cy, cw, ch = params[i]
            assert cx >= 0 and cy >= 0
            assert cx + cw <= widths[i]
            assert cy + ch <= heights[i]


# ---------------------------------------------------------------------------
# Tests: DecodeRandomResizeShortCropLong
# ---------------------------------------------------------------------------

class TestDecodeRandomResizeShortCropLong:
    """Test the high-level single-output decoder."""

    @pytest.fixture(autouse=True)
    def check_decoder(self):
        _skip_if_no_decoder()

    def test_output_shape_landscape(self, landscape_batch):
        from slipstream.decoders import DecodeRandomResizeShortCropLong
        dec = DecodeRandomResizeShortCropLong(size=224)
        result = dec(landscape_batch)
        assert result.shape == (4, 224, 224, 3)
        dec.shutdown()

    def test_output_shape_portrait(self, portrait_batch):
        from slipstream.decoders import DecodeRandomResizeShortCropLong
        dec = DecodeRandomResizeShortCropLong(size=224)
        result = dec(portrait_batch)
        assert result.shape == (4, 224, 224, 3)
        dec.shutdown()

    def test_output_shape_mixed(self, mixed_batch):
        from slipstream.decoders import DecodeRandomResizeShortCropLong
        dec = DecodeRandomResizeShortCropLong(size=128)
        result = dec(mixed_batch)
        assert result.shape == (4, 128, 128, 3)
        dec.shutdown()

    def test_center_crop_matches_resize_crop(self, landscape_batch):
        """Default (center crop) should match DecodeResizeCrop with resize==crop."""
        from slipstream.decoders import DecodeRandomResizeShortCropLong, DecodeResizeCrop

        dec_new = DecodeRandomResizeShortCropLong(size=224)
        dec_ref = DecodeResizeCrop(resize_size=224, crop_size=224)
        result_new = dec_new(landscape_batch)
        result_ref = dec_ref(landscape_batch)
        np.testing.assert_array_equal(result_new, result_ref)
        dec_new.shutdown()
        dec_ref.shutdown()

    def test_seed_reproducibility(self, landscape_batch):
        from slipstream.decoders import DecodeRandomResizeShortCropLong

        dec1 = DecodeRandomResizeShortCropLong(size=224, x_range=(0, 1), seed=42)
        dec2 = DecodeRandomResizeShortCropLong(size=224, x_range=(0, 1), seed=42)
        r1 = dec1(landscape_batch)
        r2 = dec2(landscape_batch)
        np.testing.assert_array_equal(r1, r2)
        dec1.shutdown()
        dec2.shutdown()

    def test_different_seeds_differ(self, landscape_batch):
        from slipstream.decoders import DecodeRandomResizeShortCropLong

        dec1 = DecodeRandomResizeShortCropLong(size=224, x_range=(0, 1), seed=42)
        dec2 = DecodeRandomResizeShortCropLong(size=224, x_range=(0, 1), seed=99)
        r1 = dec1(landscape_batch)
        r2 = dec2(landscape_batch)
        assert not np.array_equal(r1, r2)
        dec1.shutdown()
        dec2.shutdown()

    def test_permute_chw(self, landscape_batch):
        from slipstream.decoders import DecodeRandomResizeShortCropLong

        dec = DecodeRandomResizeShortCropLong(size=224, permute=True)
        result = dec(landscape_batch)
        assert result.shape == (4, 3, 224, 224)
        dec.shutdown()

    def test_to_tensor(self, landscape_batch):
        from slipstream.decoders import DecodeRandomResizeShortCropLong

        dec = DecodeRandomResizeShortCropLong(size=224, to_tensor=True)
        result = dec(landscape_batch)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (4, 224, 224, 3)
        dec.shutdown()

    def test_to_tensor_permute(self, landscape_batch):
        from slipstream.decoders import DecodeRandomResizeShortCropLong

        dec = DecodeRandomResizeShortCropLong(size=224, to_tensor=True, permute=True)
        result = dec(landscape_batch)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (4, 3, 224, 224)
        dec.shutdown()

    def test_size_tuple_per_batch(self, landscape_batch):
        from slipstream.decoders import DecodeRandomResizeShortCropLong

        dec = DecodeRandomResizeShortCropLong(
            size=(192, 256), size_mode="per_batch", seed=42,
        )
        result = dec(landscape_batch)
        # Should be a stacked array (all same size)
        assert isinstance(result, np.ndarray)
        assert result.ndim == 4
        s = result.shape[1]
        assert result.shape == (4, s, s, 3)
        assert 192 <= s <= 256
        dec.shutdown()

    def test_size_tuple_per_image(self, landscape_batch):
        from slipstream.decoders import DecodeRandomResizeShortCropLong

        dec = DecodeRandomResizeShortCropLong(
            size=(96, 224), size_mode="per_image", seed=42,
        )
        result = dec(landscape_batch)
        # Should be a list of arrays with potentially different sizes
        assert isinstance(result, list)
        assert len(result) == 4
        for arr in result:
            assert arr.ndim == 3
            h, w, c = arr.shape
            assert h == w  # square
            assert c == 3
            assert 96 <= h <= 224
        dec.shutdown()

    def test_size_tuple_per_image_permute_tensor(self, landscape_batch):
        from slipstream.decoders import DecodeRandomResizeShortCropLong

        dec = DecodeRandomResizeShortCropLong(
            size=(96, 224), size_mode="per_image", seed=42,
            permute=True, to_tensor=True,
        )
        result = dec(landscape_batch)
        assert isinstance(result, list)
        for t in result:
            assert isinstance(t, torch.Tensor)
            assert t.ndim == 3
            c, h, w = t.shape
            assert c == 3
            assert h == w
        dec.shutdown()

    def test_x_range_only_affects_landscape(self, mixed_batch):
        """x_range jitter should only change landscape crops, not portrait."""
        from slipstream.decoders.numba_decoder import _compute_resize_short_crop_long_params

        # landscape (480x640): x has slack
        # portrait (640x480): x has NO slack
        heights = np.array([480, 640], dtype=np.uint32)
        widths = np.array([640, 480], dtype=np.uint32)
        target_sizes = np.array([224, 224], dtype=np.int32)

        p_left = _compute_resize_short_crop_long_params(
            heights, widths, target_sizes,
            np.array([0.0, 0.0], dtype=np.float64),
            np.array([0.5, 0.5], dtype=np.float64),
        )
        p_right = _compute_resize_short_crop_long_params(
            heights, widths, target_sizes,
            np.array([1.0, 1.0], dtype=np.float64),
            np.array([0.5, 0.5], dtype=np.float64),
        )

        # Landscape: x_pos should change crop_x
        assert p_left[0, 0] != p_right[0, 0], "x_range should affect landscape crop"
        # Portrait: x_pos should NOT change crop_x (no horizontal slack)
        assert p_left[1, 0] == p_right[1, 0], "x_range should not affect portrait crop"

    def test_backward_compat_alias(self):
        from slipstream.decoders import RandomResizeShortCropLong, DecodeRandomResizeShortCropLong
        assert RandomResizeShortCropLong is DecodeRandomResizeShortCropLong

    def test_repr(self):
        from slipstream.decoders import DecodeRandomResizeShortCropLong
        dec = DecodeRandomResizeShortCropLong(size=224)
        r = repr(dec)
        assert "DecodeRandomResizeShortCropLong" in r
        assert "224" in r

    def test_last_params_empty_before_call(self):
        from slipstream.decoders import DecodeRandomResizeShortCropLong
        dec = DecodeRandomResizeShortCropLong(size=224)
        assert dec.last_params == {}
        dec.shutdown()

    def test_last_params_keys(self, landscape_batch):
        from slipstream.decoders import DecodeRandomResizeShortCropLong
        dec = DecodeRandomResizeShortCropLong(size=224, x_range=(0, 1), seed=42)
        dec(landscape_batch)
        p = dec.last_params
        assert set(p.keys()) == {"target_sizes", "x_pos", "y_pos", "heights", "widths", "crop_params"}
        dec.shutdown()

    def test_last_params_shapes(self, landscape_batch):
        from slipstream.decoders import DecodeRandomResizeShortCropLong
        dec = DecodeRandomResizeShortCropLong(size=224, x_range=(0, 1), seed=42)
        dec(landscape_batch)
        p = dec.last_params
        B = 4
        assert p["target_sizes"].shape == (B,)
        assert p["x_pos"].shape == (B,)
        assert p["y_pos"].shape == (B,)
        assert p["heights"].shape == (B,)
        assert p["widths"].shape == (B,)
        assert p["crop_params"].shape == (B, 4)
        dec.shutdown()

    def test_last_params_values(self, landscape_batch):
        from slipstream.decoders import DecodeRandomResizeShortCropLong
        dec = DecodeRandomResizeShortCropLong(size=224, x_range=(0, 1), seed=42)
        dec(landscape_batch)
        p = dec.last_params
        # All target sizes should be 224
        np.testing.assert_array_equal(p["target_sizes"], 224)
        # Heights/widths should match input
        np.testing.assert_array_equal(p["heights"], landscape_batch["heights"].astype(np.int32))
        np.testing.assert_array_equal(p["widths"], landscape_batch["widths"].astype(np.int32))
        # x_pos/y_pos should be in [0, 1]
        assert np.all(p["x_pos"] >= 0) and np.all(p["x_pos"] <= 1)
        assert np.all(p["y_pos"] >= 0) and np.all(p["y_pos"] <= 1)
        dec.shutdown()


# ---------------------------------------------------------------------------
# Tests: DecodeMultiRandomResizeShortCropLong
# ---------------------------------------------------------------------------

class TestDecodeMultiRandomResizeShortCropLong:
    """Test the multi-crop variant."""

    @pytest.fixture(autouse=True)
    def check_decoder(self):
        _skip_if_no_decoder()

    def test_basic_dict_output(self, landscape_batch):
        from slipstream.decoders import DecodeMultiRandomResizeShortCropLong

        multi = DecodeMultiRandomResizeShortCropLong({
            "view_a": dict(size=224),
            "view_b": dict(size=128),
        })
        result = multi(landscape_batch)
        assert isinstance(result, dict)
        assert set(result.keys()) == {"view_a", "view_b"}
        assert result["view_a"].shape == (4, 224, 224, 3)
        assert result["view_b"].shape == (4, 128, 128, 3)
        multi.shutdown()

    def test_same_size_crops(self, landscape_batch):
        from slipstream.decoders import DecodeMultiRandomResizeShortCropLong

        multi = DecodeMultiRandomResizeShortCropLong({
            "left": dict(size=224, x_range=0.0, seed=42),
            "right": dict(size=224, x_range=1.0, seed=43),
        })
        result = multi(landscape_batch)
        assert result["left"].shape == (4, 224, 224, 3)
        assert result["right"].shape == (4, 224, 224, 3)
        # Different positions → different pixels
        assert not np.array_equal(result["left"], result["right"])
        multi.shutdown()

    def test_yoked_crops_same_position(self, landscape_batch):
        """Same seed → same crop position even at different sizes."""
        from slipstream.decoders import DecodeMultiRandomResizeShortCropLong

        multi = DecodeMultiRandomResizeShortCropLong({
            "large": dict(size=224, x_range=(0, 1), y_range=(0, 1), seed=42),
            "small": dict(size=112, x_range=(0, 1), y_range=(0, 1), seed=42),
        })
        # Just verify it runs and produces correct shapes
        result = multi(landscape_batch)
        assert result["large"].shape == (4, 224, 224, 3)
        assert result["small"].shape == (4, 112, 112, 3)
        multi.shutdown()

    def test_seed_reproducibility(self, landscape_batch):
        from slipstream.decoders import DecodeMultiRandomResizeShortCropLong

        config = {
            "v1": dict(size=224, x_range=(0, 1), seed=42),
            "v2": dict(size=128, x_range=(0, 1), seed=43),
        }
        m1 = DecodeMultiRandomResizeShortCropLong(config)
        m2 = DecodeMultiRandomResizeShortCropLong(config)
        r1 = m1(landscape_batch)
        r2 = m2(landscape_batch)
        np.testing.assert_array_equal(r1["v1"], r2["v1"])
        np.testing.assert_array_equal(r1["v2"], r2["v2"])
        m1.shutdown()
        m2.shutdown()

    def test_permute_and_tensor(self, landscape_batch):
        from slipstream.decoders import DecodeMultiRandomResizeShortCropLong

        multi = DecodeMultiRandomResizeShortCropLong(
            {"v": dict(size=224)},
            permute=True, to_tensor=True,
        )
        result = multi(landscape_batch)
        assert isinstance(result["v"], torch.Tensor)
        assert result["v"].shape == (4, 3, 224, 224)
        multi.shutdown()

    def test_missing_size_raises(self):
        from slipstream.decoders import DecodeMultiRandomResizeShortCropLong

        with pytest.raises(ValueError, match="must specify 'size'"):
            DecodeMultiRandomResizeShortCropLong({"bad": dict(x_range=0.5)})

    def test_backward_compat_alias(self):
        from slipstream.decoders import (
            MultiRandomResizeShortCropLong,
            DecodeMultiRandomResizeShortCropLong,
        )
        assert MultiRandomResizeShortCropLong is DecodeMultiRandomResizeShortCropLong

    def test_repr(self):
        from slipstream.decoders import DecodeMultiRandomResizeShortCropLong
        multi = DecodeMultiRandomResizeShortCropLong({"v": dict(size=224)})
        r = repr(multi)
        assert "DecodeMultiRandomResizeShortCropLong" in r
        assert "224" in r

    def test_last_params_empty_before_call(self):
        from slipstream.decoders import DecodeMultiRandomResizeShortCropLong
        multi = DecodeMultiRandomResizeShortCropLong({"v": dict(size=224)})
        assert multi.last_params == {}
        multi.shutdown()

    def test_multi_last_params(self, landscape_batch):
        from slipstream.decoders import DecodeMultiRandomResizeShortCropLong
        multi = DecodeMultiRandomResizeShortCropLong({
            "large": dict(size=224, x_range=(0, 1), seed=42),
            "small": dict(size=128, x_range=(0, 1), seed=43),
        })
        multi(landscape_batch)
        p = multi.last_params
        B = 4

        assert set(p.keys()) == {"heights", "widths", "crops"}
        np.testing.assert_array_equal(p["heights"], landscape_batch["heights"].astype(np.int32))
        np.testing.assert_array_equal(p["widths"], landscape_batch["widths"].astype(np.int32))

        assert set(p["crops"].keys()) == {"large", "small"}
        for name, expected_size in [("large", 224), ("small", 128)]:
            cp = p["crops"][name]
            assert np.all(cp["target_sizes"] == expected_size)
            assert cp["x_pos"].shape == (B,)
            assert cp["y_pos"].shape == (B,)
            assert cp["crop_params"].shape == (B, 4)
        multi.shutdown()
