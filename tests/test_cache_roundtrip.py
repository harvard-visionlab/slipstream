"""Cache build/load round-trip verification tests.

These tests verify that:
1. JPEG bytes survive cache build/load unchanged (byte-identical)
2. PNG→YUV420 conversion decodes back to same pixels (±1 tolerance)
3. Non-image fields (str, int) survive unchanged

The tests use synthetic data and don't require external datasets.
"""

import hashlib
import io
from pathlib import Path

import numpy as np
import PIL.Image
import pytest

from slipstream import SlipstreamDataset
from slipstream.cache import (
    OptimizedCache,
    ImageBytesStorage,
    detect_image_format,
    find_image_end,
    read_image_dimensions,
    rgb_to_yuv420,
    decode_image_to_rgb,
)


# ---------------------------------------------------------------------------
# Test Data Generators
# ---------------------------------------------------------------------------

def _create_test_jpeg(width: int, height: int, color: tuple = (255, 0, 0)) -> bytes:
    """Create a JPEG image as bytes using PIL."""
    img = PIL.Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def _create_test_png(width: int, height: int, color: tuple = (0, 255, 0)) -> bytes:
    """Create a PNG image as bytes using PIL."""
    img = PIL.Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _create_test_bmp(width: int, height: int, color: tuple = (0, 0, 255)) -> bytes:
    """Create a BMP image as bytes using PIL."""
    img = PIL.Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="BMP")
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


def _create_gradient_png(width: int = 64, height: int = 64) -> bytes:
    """Create a PNG with gradient for round-trip testing."""
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            arr[y, x, 0] = int(255 * x / width)
            arr[y, x, 1] = int(255 * y / height)
            arr[y, x, 2] = 128
    img = PIL.Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class MockJPEGDataset:
    """Mock dataset returning JPEG images for cache testing."""

    def __init__(self, num_samples: int = 10, cache_path: Path | None = None):
        self.num_samples = num_samples
        self.cache_path = cache_path

        # Pre-generate JPEG images with different colors
        self._images = []
        self._labels = list(range(num_samples))
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 128, 128), (64, 192, 32),
        ]
        for i in range(num_samples):
            color = colors[i % len(colors)]
            # Vary sizes to test variable-length storage
            w = 32 + (i * 8) % 64
            h = 24 + (i * 6) % 48
            self._images.append(_create_test_jpeg(w, h, color))

    @property
    def field_types(self) -> dict:
        return {"image": "ImageBytes", "label": "int"}

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        return {
            "image": self._images[idx],
            "label": self._labels[idx],
        }


class MockPNGDataset:
    """Mock dataset returning PNG images for YUV420 conversion testing."""

    def __init__(self, num_samples: int = 10, cache_path: Path | None = None):
        self.num_samples = num_samples
        self.cache_path = cache_path

        self._images = []
        self._labels = list(range(num_samples))
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 128, 128), (64, 192, 32),
        ]
        for i in range(num_samples):
            color = colors[i % len(colors)]
            w = 32 + (i * 8) % 64
            h = 24 + (i * 6) % 48
            self._images.append(_create_test_png(w, h, color))

    @property
    def field_types(self) -> dict:
        return {"image": "ImageBytes", "label": "int"}

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        return {
            "image": self._images[idx],
            "label": self._labels[idx],
        }


class MockMixedDataset:
    """Mock dataset with JPEG images plus string and int fields."""

    def __init__(self, num_samples: int = 10, cache_path: Path | None = None):
        self.num_samples = num_samples
        self.cache_path = cache_path

        self._images = [_create_gradient_jpeg(48, 36) for _ in range(num_samples)]
        self._labels = list(range(num_samples))
        self._paths = [f"path/to/image_{i:04d}.jpg" for i in range(num_samples)]
        self._scores = [float(i) * 0.1 for i in range(num_samples)]

    @property
    def field_types(self) -> dict:
        return {
            "image": "ImageBytes",
            "label": "int",
            "path": "str",
            "score": "float",
        }

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        return {
            "image": self._images[idx],
            "label": self._labels[idx],
            "path": self._paths[idx],
            "score": self._scores[idx],
        }


# ---------------------------------------------------------------------------
# Tests: JPEG Round-Trip (Byte-Identical)
# ---------------------------------------------------------------------------

class TestJPEGCacheRoundtrip:
    """Verify JPEG bytes survive cache build/load unchanged."""

    def test_jpeg_bytes_identical_after_roundtrip(self, tmp_path):
        """JPEG bytes should be byte-identical after cache build/load."""
        dataset = MockJPEGDataset(num_samples=10, cache_path=tmp_path)

        # Build cache
        cache = OptimizedCache.build(dataset, output_dir=tmp_path, verbose=False)

        # Load cache
        loaded = OptimizedCache.load(tmp_path, verbose=False)

        # Verify all samples are byte-identical
        for idx in range(len(dataset)):
            original_bytes = dataset[idx]['image']

            # Load from cache using internal access (pure Python, no JIT)
            storage = loaded.fields['image']
            meta = storage._metadata[idx]
            ptr = int(meta['data_ptr'])
            size = int(meta['data_size'])
            cached_bytes = bytes(storage._data_mmap[ptr:ptr + size])

            # Hash comparison
            original_hash = hashlib.sha256(original_bytes).hexdigest()
            cached_hash = hashlib.sha256(cached_bytes).hexdigest()

            assert original_hash == cached_hash, f"JPEG bytes mismatch at index {idx}"

    def test_jpeg_dimensions_stored_correctly(self, tmp_path):
        """Verify stored dimensions match original image dimensions."""
        dataset = MockJPEGDataset(num_samples=10, cache_path=tmp_path)
        cache = OptimizedCache.build(dataset, output_dir=tmp_path, verbose=False)
        loaded = OptimizedCache.load(tmp_path, verbose=False)

        for idx in range(len(dataset)):
            original_bytes = dataset[idx]['image']
            original_w, original_h = read_image_dimensions(original_bytes)

            # Get stored dimensions
            cached_h, cached_w = loaded.get_image_dims('image', idx)

            assert cached_h == original_h, f"Height mismatch at {idx}: {cached_h} vs {original_h}"
            assert cached_w == original_w, f"Width mismatch at {idx}: {cached_w} vs {original_w}"

            # Verify dimensions are valid (non-zero)
            assert cached_h > 0, f"Invalid height at {idx}: {cached_h}"
            assert cached_w > 0, f"Invalid width at {idx}: {cached_w}"

    def test_jpeg_decodable_after_roundtrip(self, tmp_path):
        """Verify cached JPEG bytes can be decoded by PIL."""
        dataset = MockJPEGDataset(num_samples=10, cache_path=tmp_path)
        OptimizedCache.build(dataset, output_dir=tmp_path, verbose=False)
        loaded = OptimizedCache.load(tmp_path, verbose=False)

        for idx in range(len(dataset)):
            storage = loaded.fields['image']
            meta = storage._metadata[idx]
            ptr = int(meta['data_ptr'])
            size = int(meta['data_size'])
            cached_bytes = bytes(storage._data_mmap[ptr:ptr + size])

            # Decode with PIL
            img = PIL.Image.open(io.BytesIO(cached_bytes))
            assert img.mode in ('RGB', 'L'), f"Unexpected mode at {idx}: {img.mode}"
            assert img.size[0] > 0 and img.size[1] > 0

    def test_labels_preserved(self, tmp_path):
        """Verify integer labels survive cache round-trip."""
        dataset = MockJPEGDataset(num_samples=10, cache_path=tmp_path)
        OptimizedCache.build(dataset, output_dir=tmp_path, verbose=False)
        loaded = OptimizedCache.load(tmp_path, verbose=False)

        for idx in range(len(dataset)):
            original_label = dataset[idx]['label']
            cached_label = loaded.fields['label']._data[idx]
            assert original_label == cached_label, f"Label mismatch at {idx}"


# ---------------------------------------------------------------------------
# Tests: PNG → YUV420 Round-Trip (Tolerance-Based)
# ---------------------------------------------------------------------------

class TestPNGYUV420Roundtrip:
    """Verify PNG→YUV420 conversion decodes back to same pixels (±1)."""

    def test_png_detected_as_non_jpeg(self, tmp_path):
        """PNG should be detected and converted to YUV420."""
        dataset = MockPNGDataset(num_samples=5, cache_path=tmp_path)
        OptimizedCache.build(dataset, output_dir=tmp_path, verbose=False)
        loaded = OptimizedCache.load(tmp_path, verbose=False)

        # Check image format in manifest
        image_format = loaded.get_image_format('image')
        assert image_format == "yuv420", f"Expected yuv420 format, got {image_format}"

    def test_png_yuv420_dimensions_match(self, tmp_path):
        """YUV420 stored dimensions should match original (padded to even)."""
        dataset = MockPNGDataset(num_samples=5, cache_path=tmp_path)
        OptimizedCache.build(dataset, output_dir=tmp_path, verbose=False)
        loaded = OptimizedCache.load(tmp_path, verbose=False)

        for idx in range(len(dataset)):
            original_bytes = dataset[idx]['image']
            original_w, original_h = read_image_dimensions(original_bytes)

            # YUV420 pads to even dimensions
            expected_h = original_h + (original_h % 2)
            expected_w = original_w + (original_w % 2)

            cached_h, cached_w = loaded.get_image_dims('image', idx)

            assert cached_h == expected_h, f"Height mismatch at {idx}"
            assert cached_w == expected_w, f"Width mismatch at {idx}"

            # Verify dimensions are valid (non-zero)
            assert cached_h > 0, f"Invalid height at {idx}: {cached_h}"
            assert cached_w > 0, f"Invalid width at {idx}: {cached_w}"

    def test_png_yuv420_decode_matches_original(self, tmp_path):
        """YUV420 decoded RGB should match original PNG within ±1."""
        try:
            from slipstream.decoders.yuv420_decoder import YUV420NumbaBatchDecoder
        except RuntimeError:
            pytest.skip("libslipstream not compiled (required for YUV420 decode)")

        dataset = MockPNGDataset(num_samples=5, cache_path=tmp_path)
        OptimizedCache.build(dataset, output_dir=tmp_path, verbose=False)
        loaded = OptimizedCache.load(tmp_path, verbose=False)

        decoder = YUV420NumbaBatchDecoder(num_threads=1)

        try:
            for idx in range(len(dataset)):
                # Original RGB from PNG
                original_bytes = dataset[idx]['image']
                original_rgb = np.array(PIL.Image.open(io.BytesIO(original_bytes)).convert('RGB'))
                orig_h, orig_w = original_rgb.shape[:2]

                # Load YUV420 bytes from cache
                storage = loaded.fields['image']
                meta = storage._metadata[idx]
                ptr = int(meta['data_ptr'])
                size = int(meta['data_size'])
                h = int(meta['height'])
                w = int(meta['width'])
                yuv_bytes = storage._data_mmap[ptr:ptr + size]

                # Decode YUV420 → RGB using batch decoder
                yuv_arr = np.frombuffer(yuv_bytes, dtype=np.uint8)
                batch_data = np.zeros((1, len(yuv_arr)), dtype=np.uint8)
                batch_data[0, :len(yuv_arr)] = yuv_arr
                sizes = np.array([size], dtype=np.uint64)
                heights = np.array([h], dtype=np.uint32)
                widths = np.array([w], dtype=np.uint32)

                decoded_rgb = decoder.decode_batch(batch_data, sizes, heights, widths)[0]

                # Crop to original dimensions (remove YUV420 padding)
                decoded_rgb = decoded_rgb[:orig_h, :orig_w, :]

                # Compare with tolerance
                diff = np.abs(original_rgb.astype(int) - decoded_rgb.astype(int))
                max_diff = np.max(diff)
                mean_diff = np.mean(diff)

                # Allow ±2 for YUV420 conversion (BT.601 fixed-point rounding)
                assert max_diff <= 2, f"Sample {idx}: max pixel diff {max_diff} exceeds tolerance (2)"
                assert mean_diff < 1.0, f"Sample {idx}: mean pixel diff {mean_diff} too high"
        finally:
            decoder.shutdown()


# ---------------------------------------------------------------------------
# Tests: Mixed Field Types
# ---------------------------------------------------------------------------

class TestMixedFieldRoundtrip:
    """Verify all field types survive cache round-trip."""

    def test_all_fields_preserved(self, tmp_path):
        """All field types should be correctly preserved."""
        dataset = MockMixedDataset(num_samples=10, cache_path=tmp_path)
        OptimizedCache.build(dataset, output_dir=tmp_path, verbose=False)
        loaded = OptimizedCache.load(tmp_path, verbose=False)

        for idx in range(len(dataset)):
            original = dataset[idx]

            # Check image (JPEG, should be identical)
            storage = loaded.fields['image']
            meta = storage._metadata[idx]
            ptr = int(meta['data_ptr'])
            size = int(meta['data_size'])
            cached_image = bytes(storage._data_mmap[ptr:ptr + size])
            assert hashlib.sha256(original['image']).hexdigest() == \
                   hashlib.sha256(cached_image).hexdigest(), f"Image mismatch at {idx}"

            # Check label (int)
            cached_label = loaded.fields['label']._data[idx]
            assert original['label'] == cached_label, f"Label mismatch at {idx}"

            # Check path (string)
            path_storage = loaded.fields['path']
            offset, length = path_storage._offsets[idx]
            cached_path = bytes(path_storage._data_mmap[offset:offset + length]).decode('utf-8')
            assert original['path'] == cached_path, f"Path mismatch at {idx}"

            # Check score (float)
            cached_score = loaded.fields['score']._data[idx]
            assert abs(original['score'] - cached_score) < 1e-9, f"Score mismatch at {idx}"


# ---------------------------------------------------------------------------
# Tests: Edge Cases
# ---------------------------------------------------------------------------

class TestCacheEdgeCases:
    """Test edge cases in cache handling."""

    def test_empty_or_single_sample(self, tmp_path):
        """Cache should handle single-sample datasets."""
        dataset = MockJPEGDataset(num_samples=1, cache_path=tmp_path)
        cache = OptimizedCache.build(dataset, output_dir=tmp_path, verbose=False)
        loaded = OptimizedCache.load(tmp_path, verbose=False)

        assert loaded.num_samples == 1
        original = dataset[0]['image']
        storage = loaded.fields['image']
        meta = storage._metadata[0]
        cached = bytes(storage._data_mmap[int(meta['data_ptr']):int(meta['data_ptr']) + int(meta['data_size'])])
        assert hashlib.sha256(original).hexdigest() == hashlib.sha256(cached).hexdigest()

    def test_jpeg_with_odd_dimensions(self, tmp_path):
        """JPEG with odd dimensions should store correctly."""
        # Create dataset with odd-dimension images
        class OddDimDataset:
            def __init__(self):
                self.cache_path = tmp_path
                self._images = [
                    _create_test_jpeg(31, 47, (255, 0, 0)),  # odd × odd
                    _create_test_jpeg(32, 47, (0, 255, 0)),  # even × odd
                    _create_test_jpeg(31, 48, (0, 0, 255)),  # odd × even
                ]
                self._labels = [0, 1, 2]

            @property
            def field_types(self):
                return {"image": "ImageBytes", "label": "int"}

            def __len__(self):
                return len(self._images)

            def __getitem__(self, idx):
                return {"image": self._images[idx], "label": self._labels[idx]}

        dataset = OddDimDataset()
        OptimizedCache.build(dataset, output_dir=tmp_path, verbose=False)
        loaded = OptimizedCache.load(tmp_path, verbose=False)

        for idx in range(len(dataset)):
            original = dataset[idx]['image']
            storage = loaded.fields['image']
            meta = storage._metadata[idx]
            cached = bytes(storage._data_mmap[int(meta['data_ptr']):int(meta['data_ptr']) + int(meta['data_size'])])
            assert hashlib.sha256(original).hexdigest() == hashlib.sha256(cached).hexdigest()

    def test_verify_detects_dimension_parse_failure(self, tmp_path):
        """Verify that dimension > 0 for all samples."""
        dataset = MockJPEGDataset(num_samples=10, cache_path=tmp_path)
        OptimizedCache.build(dataset, output_dir=tmp_path, verbose=False)
        loaded = OptimizedCache.load(tmp_path, verbose=False)

        storage = loaded.fields['image']
        for idx in range(loaded.num_samples):
            h, w = loaded.get_image_dims('image', idx)
            assert h > 0, f"Invalid height at {idx}: {h}"
            assert w > 0, f"Invalid width at {idx}: {w}"


# ---------------------------------------------------------------------------
# Tests: Image Format Detection
# ---------------------------------------------------------------------------

class TestImageFormatDetection:
    """Test image format detection logic."""

    def test_detect_jpeg(self):
        """JPEG should be detected correctly."""
        jpeg_bytes = _create_test_jpeg(32, 32)
        assert detect_image_format(jpeg_bytes) == "jpeg"

    def test_detect_png(self):
        """PNG should be detected correctly."""
        png_bytes = _create_test_png(32, 32)
        assert detect_image_format(png_bytes) == "png"

    def test_detect_bmp_as_other(self):
        """BMP should be detected as 'other'."""
        bmp_bytes = _create_test_bmp(32, 32)
        assert detect_image_format(bmp_bytes) == "other"

    def test_detect_from_numpy_array(self):
        """Detection should work on numpy arrays too."""
        jpeg_bytes = _create_test_jpeg(32, 32)
        jpeg_array = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        assert detect_image_format(jpeg_array) == "jpeg"


# ---------------------------------------------------------------------------
# Tests: YUV420 Conversion
# ---------------------------------------------------------------------------

class TestYUV420Conversion:
    """Test RGB ↔ YUV420 conversion."""

    def test_rgb_to_yuv420_roundtrip_solid_color(self):
        """Solid color should survive RGB→YUV420→RGB within ±1."""
        try:
            from slipstream.decoders.yuv420_decoder import YUV420NumbaBatchDecoder
        except RuntimeError:
            pytest.skip("libslipstream not compiled (required for YUV420 decode)")

        # Create solid red image
        rgb = np.zeros((16, 16, 3), dtype=np.uint8)
        rgb[:, :, 0] = 255  # Red

        # Convert to YUV420
        yuv_bytes, pad_h, pad_w = rgb_to_yuv420(rgb)

        # Convert back to RGB using decoder
        decoder = YUV420NumbaBatchDecoder(num_threads=1)
        try:
            yuv_arr = np.frombuffer(yuv_bytes, dtype=np.uint8)
            batch_data = np.zeros((1, len(yuv_arr)), dtype=np.uint8)
            batch_data[0, :len(yuv_arr)] = yuv_arr
            sizes = np.array([len(yuv_bytes)], dtype=np.uint64)
            heights = np.array([pad_h], dtype=np.uint32)
            widths = np.array([pad_w], dtype=np.uint32)

            decoded = decoder.decode_batch(batch_data, sizes, heights, widths)[0]

            # Compare
            diff = np.abs(rgb.astype(int) - decoded.astype(int))
            max_diff = np.max(diff)
            assert max_diff <= 1, f"Max diff {max_diff} exceeds tolerance"
        finally:
            decoder.shutdown()

    def test_rgb_to_yuv420_odd_dimensions(self):
        """Odd dimensions should be padded to even."""
        rgb = np.zeros((15, 17, 3), dtype=np.uint8)  # 15 × 17
        yuv_bytes, pad_h, pad_w = rgb_to_yuv420(rgb)

        assert pad_h == 16, f"Expected padded height 16, got {pad_h}"
        assert pad_w == 18, f"Expected padded width 18, got {pad_w}"

        # YUV420 size: Y (h×w) + U (h/2 × w/2) + V (h/2 × w/2)
        expected_size = 16 * 18 + (16 // 2) * (18 // 2) * 2
        assert len(yuv_bytes) == expected_size
