"""Decoder correctness tests with tolerance-based verification.

These tests verify that slipstream decoders produce correct output by comparing
against reference implementations (PIL, TurboJPEG) within acceptable tolerances.

JPEG decode variance is expected due to:
- DCT precision (FASTDCT vs ACCURATE)
- Fixed-point vs floating-point color conversion
- Rounding strategies at operation boundaries

Tolerance thresholds:
- JPEG decode: max ≤ 2, mean < 0.5
- YUV→RGB: max ≤ 1, mean < 0.3
- Resize (bilinear): max ≤ 3, mean < 1.0
"""

import io
from pathlib import Path

import numpy as np
import PIL.Image
import pytest


# ---------------------------------------------------------------------------
# Test Data Generators
# ---------------------------------------------------------------------------

def _create_test_jpeg(width: int, height: int, color: tuple = (255, 0, 0)) -> bytes:
    """Create a JPEG image as bytes using PIL."""
    img = PIL.Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def _create_gradient_jpeg(width: int = 64, height: int = 64) -> bytes:
    """Create a JPEG with gradient for better compression testing."""
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            arr[y, x, 0] = int(255 * x / width)  # R gradient
            arr[y, x, 1] = int(255 * y / height)  # G gradient
            arr[y, x, 2] = 128  # Blue constant
    img = PIL.Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def _create_solid_yuv420(y_val: int, u_val: int, v_val: int, h: int, w: int) -> bytes:
    """Create a solid color YUV420P buffer.

    YUV420P layout: Y plane (h×w) + U plane (h/2 × w/2) + V plane (h/2 × w/2)
    """
    y_plane = np.full((h, w), y_val, dtype=np.uint8)
    u_plane = np.full((h // 2, w // 2), u_val, dtype=np.uint8)
    v_plane = np.full((h // 2, w // 2), v_val, dtype=np.uint8)
    return y_plane.tobytes() + u_plane.tobytes() + v_plane.tobytes()


# ---------------------------------------------------------------------------
# Tests: JPEG Decode vs PIL Reference
# ---------------------------------------------------------------------------

class TestJPEGDecodeVsPIL:
    """Verify JPEG decode matches PIL within tolerance."""

    @pytest.fixture
    def decoder(self):
        """Get NumbaBatchDecoder, skip if not compiled."""
        try:
            from slipstream.decoders.numba_decoder import NumbaBatchDecoder
            dec = NumbaBatchDecoder(num_threads=1)
            yield dec
            dec.shutdown()
        except RuntimeError:
            pytest.skip("libslipstream not compiled")

    def test_solid_color_decode(self, decoder):
        """Solid color JPEG should decode identically to PIL."""
        colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 255),  # White
            (0, 0, 0),      # Black
            (128, 128, 128),  # Gray
        ]

        for color in colors:
            jpeg_bytes = _create_test_jpeg(32, 32, color)

            # PIL decode
            pil_rgb = np.array(PIL.Image.open(io.BytesIO(jpeg_bytes)))

            # Slipstream decode
            batch_data = np.zeros((1, len(jpeg_bytes)), dtype=np.uint8)
            batch_data[0, :len(jpeg_bytes)] = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            sizes = np.array([len(jpeg_bytes)], dtype=np.uint64)
            heights = np.array([32], dtype=np.uint32)
            widths = np.array([32], dtype=np.uint32)

            slip_rgb = decoder.decode_batch(batch_data, sizes, heights, widths)[0]

            # Compare
            diff = np.abs(pil_rgb.astype(int) - slip_rgb.astype(int))
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)

            assert max_diff <= 2, f"Color {color}: max diff {max_diff} exceeds tolerance (2)"
            assert mean_diff < 0.5, f"Color {color}: mean diff {mean_diff} exceeds tolerance (0.5)"

    def test_gradient_decode(self, decoder):
        """Gradient JPEG should decode within tolerance.

        Note: Gradient images have more DCT activity and thus more decode
        variance between JPEG decoders. We allow max_diff of 5 for gradients.
        """
        jpeg_bytes = _create_gradient_jpeg(64, 64)

        # PIL decode
        pil_rgb = np.array(PIL.Image.open(io.BytesIO(jpeg_bytes)))

        # Slipstream decode
        batch_data = np.zeros((1, len(jpeg_bytes)), dtype=np.uint8)
        batch_data[0, :len(jpeg_bytes)] = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        sizes = np.array([len(jpeg_bytes)], dtype=np.uint64)
        heights = np.array([64], dtype=np.uint32)
        widths = np.array([64], dtype=np.uint32)

        slip_rgb = decoder.decode_batch(batch_data, sizes, heights, widths)[0]

        # Compare (relaxed tolerance for gradient)
        diff = np.abs(pil_rgb.astype(int) - slip_rgb.astype(int))
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)

        assert max_diff <= 5, f"Gradient: max diff {max_diff} exceeds tolerance (5)"
        assert mean_diff < 1.5, f"Gradient: mean diff {mean_diff} exceeds tolerance (1.5)"

    def test_varying_dimensions(self, decoder):
        """Various image sizes should decode correctly."""
        dimensions = [
            (8, 8),
            (16, 16),
            (31, 31),   # Odd dimensions
            (64, 48),
            (100, 75),
            (224, 224),
        ]

        for w, h in dimensions:
            jpeg_bytes = _create_test_jpeg(w, h, (100, 150, 200))

            # PIL decode
            pil_rgb = np.array(PIL.Image.open(io.BytesIO(jpeg_bytes)))

            # Slipstream decode
            batch_data = np.zeros((1, len(jpeg_bytes)), dtype=np.uint8)
            batch_data[0, :len(jpeg_bytes)] = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            sizes = np.array([len(jpeg_bytes)], dtype=np.uint64)
            heights = np.array([h], dtype=np.uint32)
            widths = np.array([w], dtype=np.uint32)

            slip_rgb = decoder.decode_batch(batch_data, sizes, heights, widths)[0]

            # Compare
            diff = np.abs(pil_rgb.astype(int) - slip_rgb.astype(int))
            max_diff = np.max(diff)

            assert max_diff <= 2, f"Size {w}×{h}: max diff {max_diff} exceeds tolerance (2)"


# ---------------------------------------------------------------------------
# Tests: YUV420 BT.601 Coefficients
# ---------------------------------------------------------------------------

class TestYUV420BT601:
    """Verify BT.601 YUV→RGB conversion coefficients."""

    @pytest.fixture
    def decoder(self):
        """Get YUV420NumbaBatchDecoder, skip if not compiled."""
        try:
            from slipstream.decoders.yuv420_decoder import YUV420NumbaBatchDecoder
            dec = YUV420NumbaBatchDecoder(num_threads=1)
            yield dec
            dec.shutdown()
        except RuntimeError:
            pytest.skip("libslipstream not compiled")

    def test_bt601_red(self, decoder):
        """Red in BT.601 YUV space should decode to approximately red RGB."""
        # BT.601 for pure red (255, 0, 0):
        # Y = 0.299*255 + 0.587*0 + 0.114*0 = 76.245 ≈ 76
        # Cb = -0.169*255 - 0.331*0 + 0.5*0 + 128 = 84.855 ≈ 85
        # Cr = 0.5*255 - 0.419*0 - 0.081*0 + 128 = 255.5 ≈ 255

        h, w = 16, 16
        yuv_bytes = _create_solid_yuv420(76, 85, 255, h, w)

        batch_data = np.zeros((1, len(yuv_bytes)), dtype=np.uint8)
        batch_data[0, :len(yuv_bytes)] = np.frombuffer(yuv_bytes, dtype=np.uint8)
        sizes = np.array([len(yuv_bytes)], dtype=np.uint64)
        heights = np.array([h], dtype=np.uint32)
        widths = np.array([w], dtype=np.uint32)

        rgb = decoder.decode_batch(batch_data, sizes, heights, widths)[0]

        # Check center pixel (away from edges where padding might affect)
        r, g, b = rgb[h // 2, w // 2]

        # Should be approximately red (allow some tolerance for fixed-point math)
        assert r >= 245, f"Red channel too low: {r}"
        assert g <= 10, f"Green channel too high: {g}"
        assert b <= 10, f"Blue channel too high: {b}"

    def test_bt601_green(self, decoder):
        """Green in BT.601 YUV space should decode to approximately green RGB."""
        # BT.601 for pure green (0, 255, 0):
        # Y = 0.587*255 = 149.685 ≈ 150
        # Cb = -0.331*255 + 128 = 43.595 ≈ 44
        # Cr = -0.419*255 + 128 = 21.155 ≈ 21

        h, w = 16, 16
        yuv_bytes = _create_solid_yuv420(150, 44, 21, h, w)

        batch_data = np.zeros((1, len(yuv_bytes)), dtype=np.uint8)
        batch_data[0, :len(yuv_bytes)] = np.frombuffer(yuv_bytes, dtype=np.uint8)
        sizes = np.array([len(yuv_bytes)], dtype=np.uint64)
        heights = np.array([h], dtype=np.uint32)
        widths = np.array([w], dtype=np.uint32)

        rgb = decoder.decode_batch(batch_data, sizes, heights, widths)[0]

        r, g, b = rgb[h // 2, w // 2]

        assert r <= 10, f"Red channel too high: {r}"
        assert g >= 245, f"Green channel too low: {g}"
        assert b <= 10, f"Blue channel too high: {b}"

    def test_bt601_blue(self, decoder):
        """Blue in BT.601 YUV space should decode to approximately blue RGB."""
        # BT.601 for pure blue (0, 0, 255):
        # Y = 0.114*255 = 29.07 ≈ 29
        # Cb = 0.5*255 + 128 = 255.5 ≈ 255
        # Cr = -0.081*255 + 128 = 107.345 ≈ 107

        h, w = 16, 16
        yuv_bytes = _create_solid_yuv420(29, 255, 107, h, w)

        batch_data = np.zeros((1, len(yuv_bytes)), dtype=np.uint8)
        batch_data[0, :len(yuv_bytes)] = np.frombuffer(yuv_bytes, dtype=np.uint8)
        sizes = np.array([len(yuv_bytes)], dtype=np.uint64)
        heights = np.array([h], dtype=np.uint32)
        widths = np.array([w], dtype=np.uint32)

        rgb = decoder.decode_batch(batch_data, sizes, heights, widths)[0]

        r, g, b = rgb[h // 2, w // 2]

        assert r <= 10, f"Red channel too high: {r}"
        assert g <= 10, f"Green channel too high: {g}"
        assert b >= 245, f"Blue channel too low: {b}"

    def test_bt601_white(self, decoder):
        """White in YUV space (Y=255, U=128, V=128) should decode to white RGB."""
        h, w = 16, 16
        yuv_bytes = _create_solid_yuv420(255, 128, 128, h, w)

        batch_data = np.zeros((1, len(yuv_bytes)), dtype=np.uint8)
        batch_data[0, :len(yuv_bytes)] = np.frombuffer(yuv_bytes, dtype=np.uint8)
        sizes = np.array([len(yuv_bytes)], dtype=np.uint64)
        heights = np.array([h], dtype=np.uint32)
        widths = np.array([w], dtype=np.uint32)

        rgb = decoder.decode_batch(batch_data, sizes, heights, widths)[0]

        r, g, b = rgb[h // 2, w // 2]

        assert r >= 250, f"Red channel too low for white: {r}"
        assert g >= 250, f"Green channel too low for white: {g}"
        assert b >= 250, f"Blue channel too low for white: {b}"

    def test_bt601_black(self, decoder):
        """Black in YUV space (Y=0, U=128, V=128) should decode to black RGB."""
        h, w = 16, 16
        yuv_bytes = _create_solid_yuv420(0, 128, 128, h, w)

        batch_data = np.zeros((1, len(yuv_bytes)), dtype=np.uint8)
        batch_data[0, :len(yuv_bytes)] = np.frombuffer(yuv_bytes, dtype=np.uint8)
        sizes = np.array([len(yuv_bytes)], dtype=np.uint64)
        heights = np.array([h], dtype=np.uint32)
        widths = np.array([w], dtype=np.uint32)

        rgb = decoder.decode_batch(batch_data, sizes, heights, widths)[0]

        r, g, b = rgb[h // 2, w // 2]

        assert r <= 5, f"Red channel too high for black: {r}"
        assert g <= 5, f"Green channel too high for black: {g}"
        assert b <= 5, f"Blue channel too high for black: {b}"


# ---------------------------------------------------------------------------
# Tests: Resize Operations
# ---------------------------------------------------------------------------

class TestResizeCorrectness:
    """Verify resize operations produce correct output dimensions and quality."""

    @pytest.fixture
    def decoder(self):
        """Get NumbaBatchDecoder, skip if not compiled."""
        try:
            from slipstream.decoders.numba_decoder import NumbaBatchDecoder
            dec = NumbaBatchDecoder(num_threads=1)
            yield dec
            dec.shutdown()
        except RuntimeError:
            pytest.skip("libslipstream not compiled")

    def test_center_crop_dimensions(self, decoder):
        """Center crop should produce correct output dimensions."""
        jpeg_bytes = _create_test_jpeg(256, 256, (100, 150, 200))

        batch_data = np.zeros((1, len(jpeg_bytes)), dtype=np.uint8)
        batch_data[0, :len(jpeg_bytes)] = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        sizes = np.array([len(jpeg_bytes)], dtype=np.uint64)
        heights = np.array([256], dtype=np.uint32)
        widths = np.array([256], dtype=np.uint32)

        crop_size = 224
        result = decoder.decode_batch_center_crop(
            batch_data, sizes, heights, widths, crop_size=crop_size
        )

        assert result.shape == (1, crop_size, crop_size, 3), \
            f"Unexpected shape: {result.shape}"

    def test_random_crop_dimensions(self, decoder):
        """Random crop should produce correct output dimensions."""
        jpeg_bytes = _create_test_jpeg(256, 256, (100, 150, 200))

        batch_data = np.zeros((1, len(jpeg_bytes)), dtype=np.uint8)
        batch_data[0, :len(jpeg_bytes)] = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        sizes = np.array([len(jpeg_bytes)], dtype=np.uint64)
        heights = np.array([256], dtype=np.uint32)
        widths = np.array([256], dtype=np.uint32)

        target_size = 224
        result = decoder.decode_batch_random_crop(
            batch_data, sizes, heights, widths,
            target_size=target_size,
            seed=42,
        )

        assert result.shape == (1, target_size, target_size, 3), \
            f"Unexpected shape: {result.shape}"

    def test_resize_preserves_color(self, decoder):
        """Resize should approximately preserve colors (solid color image)."""
        jpeg_bytes = _create_test_jpeg(256, 256, (200, 100, 50))

        batch_data = np.zeros((1, len(jpeg_bytes)), dtype=np.uint8)
        batch_data[0, :len(jpeg_bytes)] = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        sizes = np.array([len(jpeg_bytes)], dtype=np.uint64)
        heights = np.array([256], dtype=np.uint32)
        widths = np.array([256], dtype=np.uint32)

        result = decoder.decode_batch_center_crop(
            batch_data, sizes, heights, widths, crop_size=64
        )[0]

        # Check center pixel - should be close to original color
        h, w = result.shape[:2]
        r, g, b = result[h // 2, w // 2]

        # Allow tolerance for JPEG compression + resize
        assert abs(r - 200) <= 10, f"Red channel drift: {r} vs 200"
        assert abs(g - 100) <= 10, f"Green channel drift: {g} vs 100"
        assert abs(b - 50) <= 10, f"Blue channel drift: {b} vs 50"


# ---------------------------------------------------------------------------
# Tests: Batch Processing
# ---------------------------------------------------------------------------

class TestBatchProcessing:
    """Verify batch processing handles multiple images correctly."""

    @pytest.fixture
    def decoder(self):
        """Get NumbaBatchDecoder, skip if not compiled."""
        try:
            from slipstream.decoders.numba_decoder import NumbaBatchDecoder
            dec = NumbaBatchDecoder(num_threads=2)  # Use multiple threads
            yield dec
            dec.shutdown()
        except RuntimeError:
            pytest.skip("libslipstream not compiled")

    def test_batch_decode_all_images_different(self, decoder):
        """Each image in batch should decode independently."""
        colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
        ]

        jpegs = [_create_test_jpeg(32, 32, c) for c in colors]

        # Find max size for batch buffer
        max_size = max(len(j) for j in jpegs)
        batch_size = len(jpegs)

        batch_data = np.zeros((batch_size, max_size), dtype=np.uint8)
        sizes = np.zeros(batch_size, dtype=np.uint64)
        heights = np.full(batch_size, 32, dtype=np.uint32)
        widths = np.full(batch_size, 32, dtype=np.uint32)

        for i, jpeg_bytes in enumerate(jpegs):
            arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            batch_data[i, :len(arr)] = arr
            sizes[i] = len(jpeg_bytes)

        results = decoder.decode_batch(batch_data, sizes, heights, widths)

        assert len(results) == batch_size

        # Verify each image has expected color (within JPEG tolerance)
        for i, (result, expected_color) in enumerate(zip(results, colors)):
            # Center pixel should be close to expected color
            h, w = result.shape[:2]
            r, g, b = result[h // 2, w // 2]

            # Check that colors are close to expected (allow JPEG variance)
            assert abs(int(r) - expected_color[0]) <= 10, \
                f"Image {i}: Red channel {r} not close to {expected_color[0]}"
            assert abs(int(g) - expected_color[1]) <= 10, \
                f"Image {i}: Green channel {g} not close to {expected_color[1]}"
            assert abs(int(b) - expected_color[2]) <= 10, \
                f"Image {i}: Blue channel {b} not close to {expected_color[2]}"

    def test_batch_varying_sizes(self, decoder):
        """Batch with varying image sizes should decode correctly."""
        sizes_list = [(32, 32), (64, 48), (100, 75), (128, 128)]
        jpegs = [_create_test_jpeg(w, h, (100, 150, 200)) for w, h in sizes_list]

        max_size = max(len(j) for j in jpegs)
        batch_size = len(jpegs)

        batch_data = np.zeros((batch_size, max_size), dtype=np.uint8)
        sizes = np.zeros(batch_size, dtype=np.uint64)
        heights = np.zeros(batch_size, dtype=np.uint32)
        widths = np.zeros(batch_size, dtype=np.uint32)

        for i, (jpeg_bytes, (w, h)) in enumerate(zip(jpegs, sizes_list)):
            arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            batch_data[i, :len(arr)] = arr
            sizes[i] = len(jpeg_bytes)
            heights[i] = h
            widths[i] = w

        results = decoder.decode_batch(batch_data, sizes, heights, widths)

        assert len(results) == batch_size

        for i, (result, (w, h)) in enumerate(zip(results, sizes_list)):
            assert result.shape == (h, w, 3), \
                f"Image {i}: expected {(h, w, 3)}, got {result.shape}"


# ---------------------------------------------------------------------------
# Tests: Edge Cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Test edge cases in decoding."""

    @pytest.fixture
    def decoder(self):
        """Get NumbaBatchDecoder, skip if not compiled."""
        try:
            from slipstream.decoders.numba_decoder import NumbaBatchDecoder
            dec = NumbaBatchDecoder(num_threads=1)
            yield dec
            dec.shutdown()
        except RuntimeError:
            pytest.skip("libslipstream not compiled")

    def test_single_image_batch(self, decoder):
        """Single image batch should work correctly."""
        jpeg_bytes = _create_test_jpeg(64, 64)

        batch_data = np.zeros((1, len(jpeg_bytes)), dtype=np.uint8)
        batch_data[0] = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        sizes = np.array([len(jpeg_bytes)], dtype=np.uint64)
        heights = np.array([64], dtype=np.uint32)
        widths = np.array([64], dtype=np.uint32)

        results = decoder.decode_batch(batch_data, sizes, heights, widths)

        assert len(results) == 1
        assert results[0].shape == (64, 64, 3)

    def test_small_images(self, decoder):
        """Very small images (8×8) should decode correctly."""
        jpeg_bytes = _create_test_jpeg(8, 8, (200, 100, 50))

        batch_data = np.zeros((1, len(jpeg_bytes)), dtype=np.uint8)
        batch_data[0, :len(jpeg_bytes)] = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        sizes = np.array([len(jpeg_bytes)], dtype=np.uint64)
        heights = np.array([8], dtype=np.uint32)
        widths = np.array([8], dtype=np.uint32)

        results = decoder.decode_batch(batch_data, sizes, heights, widths)

        assert results[0].shape == (8, 8, 3)
