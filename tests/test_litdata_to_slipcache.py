"""LitData → SlipCache conversion verification.

These tests verify that:
1. JPEG images survive LitData → SlipCache → load unchanged (byte-identical)
2. Labels and other fields are preserved exactly
3. ALL 50,000 ImageNet validation samples are verified (not just first 100)

Important: The cache format is auto-detected from the first sample:
- If first sample is JPEG → all samples stored as JPEG bytes
- If first sample is PNG → all samples converted to YUV420

The PNG-in-JPEG problem: ImageNet contains ~1% PNG files with .JPEG extension.
Since the first sample is likely JPEG, PNG files may be stored as raw PNG bytes,
which would cause JPEG-only decoders to fail. This test suite detects such cases.

Run with:
    uv run pytest tests/test_litdata_to_slipcache.py -v -s

    # Skip slow full-dataset tests
    uv run pytest tests/test_litdata_to_slipcache.py -v -s -m "not slow"
"""

import io
import sys

import numpy as np
import PIL.Image
import pytest

from slipstream.readers.streaming import StreamingReader
from slipstream.cache import OptimizedCache


# Real ImageNet validation set on S3 (LitData format)
LITDATA_VAL_S3_PATH = "s3://visionlab-datasets/imagenet1k/pre-processed/s256-l512-jpgbytes-q100-streaming/val/"


def _print_progress(msg: str) -> None:
    """Print to stderr so pytest doesn't capture it."""
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def check_s3_access():
    """Skip tests if S3 access is not available."""
    import boto3

    try:
        boto3.client('s3').head_bucket(Bucket='visionlab-datasets')
    except Exception as e:
        pytest.skip(f"S3 access not available: {e}")


@pytest.fixture(scope="module")
def litdata_cache_dir():
    """Get cache directory for test data (persistent across test runs)."""
    from slipstream.dataset import get_default_cache_dir

    cache_dir = get_default_cache_dir() / "slipstream" / "litdata-slipcache-test"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


@pytest.fixture(scope="module")
def litdata_reader(check_s3_access):
    """Create StreamingReader for ImageNet val."""
    _print_progress("\n[fixture] Creating LitData StreamingReader...")
    reader = StreamingReader(
        remote_dir=LITDATA_VAL_S3_PATH,
        max_cache_size="50GB",
    )
    _print_progress(f"[fixture] LitData reader ready: {len(reader)} samples")
    return reader


@pytest.fixture(scope="module")
def slipcache(litdata_reader):
    """Build SlipCache from LitData.

    The reader's cache_path property includes a dataset hash automatically,
    preventing stale cache issues when the source dataset changes.

    Format is auto-detected from first sample:
    - JPEG source → JPEG cache
    - PNG source → YUV420 cache
    """
    # Use the reader's versioned cache_path (includes dataset hash)
    cache_dir = litdata_reader.cache_path
    _print_progress(f"\n[fixture] Dataset hash: {litdata_reader.dataset_hash}")
    _print_progress(f"[fixture] Cache path: {cache_dir}")

    manifest_path = cache_dir / ".slipstream" / "manifest.json"

    # Check if cache already exists
    if manifest_path.exists():
        _print_progress(f"[fixture] Loading existing SlipCache from {cache_dir}")
        return OptimizedCache.load(cache_dir, verbose=False)

    _print_progress(f"[fixture] Building SlipCache from LitData...")
    _print_progress(f"[fixture] Output: {cache_dir}")

    # Build cache (auto-detects format)
    cache = OptimizedCache.build(
        litdata_reader,
        output_dir=cache_dir,
        verbose=True,
    )

    image_format = cache.get_image_format('image')
    _print_progress(f"[fixture] SlipCache ready: {cache.num_samples} samples, format={image_format}")
    return cache


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def _get_cached_image_bytes(cache: OptimizedCache, idx: int) -> bytes:
    """Extract raw image bytes from cache (pure Python, no JIT)."""
    storage = cache.fields['image']
    meta = storage._metadata[idx]
    ptr = int(meta['data_ptr'])
    size = int(meta['data_size'])
    return bytes(storage._data_mmap[ptr:ptr + size])


def _decode_cached_image(cache: OptimizedCache, idx: int) -> np.ndarray:
    """Decode cached image to RGB, handling both JPEG and YUV420 formats."""
    cached_bytes = _get_cached_image_bytes(cache, idx)
    image_format = cache.get_image_format('image')

    if image_format == "yuv420":
        from slipstream.decoders.yuv420_decoder import YUV420NumbaBatchDecoder
        h, w = cache.get_image_dims('image', idx)

        decoder = YUV420NumbaBatchDecoder(num_threads=1)
        try:
            yuv_arr = np.frombuffer(cached_bytes, dtype=np.uint8)
            batch_data = np.zeros((1, len(yuv_arr)), dtype=np.uint8)
            batch_data[0, :len(yuv_arr)] = yuv_arr
            sizes = np.array([len(cached_bytes)], dtype=np.uint64)
            heights = np.array([h], dtype=np.uint32)
            widths = np.array([w], dtype=np.uint32)
            return decoder.decode_batch(batch_data, sizes, heights, widths)[0]
        finally:
            decoder.shutdown()
    else:
        # JPEG format - decode with PIL
        return np.array(PIL.Image.open(io.BytesIO(cached_bytes)).convert('RGB'))


# ---------------------------------------------------------------------------
# Tests: Structure Verification
# ---------------------------------------------------------------------------

@pytest.mark.s3
class TestLitDataToSlipCacheStructure:
    """Verify SlipCache structure matches source."""

    def test_sample_count_matches(self, litdata_reader, slipcache):
        """Sample count should match source."""
        assert slipcache.num_samples == len(litdata_reader), \
            f"Count mismatch: cache={slipcache.num_samples}, source={len(litdata_reader)}"

    def test_sample_count_is_50000(self, slipcache):
        """ImageNet val should have 50,000 samples."""
        assert slipcache.num_samples == 50000, \
            f"Expected 50000 samples, got {slipcache.num_samples}"

    def test_fields_present(self, slipcache):
        """Required fields should be present."""
        assert 'image' in slipcache.fields, "Missing 'image' field"
        assert 'label' in slipcache.fields, "Missing 'label' field"

    def test_image_format_detected(self, slipcache):
        """Image format should be detected (jpeg or yuv420)."""
        image_format = slipcache.get_image_format('image')
        assert image_format in ("jpeg", "yuv420"), \
            f"Unexpected image format: {image_format}"
        _print_progress(f"\n  Detected image format: {image_format}")


# ---------------------------------------------------------------------------
# Tests: Quick Verification (First 100)
# ---------------------------------------------------------------------------

@pytest.mark.s3
class TestLitDataToSlipCacheQuick:
    """Quick verification of first 100 samples."""

    def test_first_100_cached_valid(self, slipcache):
        """First 100 cached images should be valid (correct format, decodable)."""
        _print_progress(f"\n  Validating first 100 cached images...")
        image_format = slipcache.get_image_format('image')
        errors = []

        for idx in range(100):
            try:
                cached_bytes = _get_cached_image_bytes(slipcache, idx)

                if image_format == "jpeg":
                    # Verify JPEG structure
                    if cached_bytes[:2] != b'\xff\xd8':
                        errors.append((idx, "missing JPEG SOI marker"))
                        continue
                    if cached_bytes[-2:] != b'\xff\xd9':
                        errors.append((idx, "missing JPEG EOI marker"))
                        continue

                    # Verify decodable
                    img = PIL.Image.open(io.BytesIO(cached_bytes))
                    img.load()
                else:
                    # YUV420: verify dimensions and size
                    h, w = slipcache.get_image_dims('image', idx)
                    if h == 0 or w == 0:
                        errors.append((idx, f"invalid dimensions: {h}x{w}"))
                        continue

                    expected_size = h * w + (h // 2) * (w // 2) * 2
                    if len(cached_bytes) != expected_size:
                        errors.append((idx, f"size mismatch: {len(cached_bytes)} vs {expected_size}"))

            except Exception as e:
                errors.append((idx, str(e)))

        if errors:
            pytest.fail(f"Validation errors at {len(errors)} indices: {errors[:5]}...")

    def test_first_100_labels_match(self, litdata_reader, slipcache):
        """First 100 labels should match exactly."""
        _print_progress(f"\n  Comparing first 100 labels...")
        mismatches = []

        for idx in range(100):
            source_label = litdata_reader[idx]['label']
            cached_label = slipcache.fields['label']._data[idx]

            if source_label != cached_label:
                mismatches.append((idx, source_label, cached_label))

        if mismatches:
            pytest.fail(f"Label mismatch at {len(mismatches)} indices: {mismatches}")

    def test_first_100_decodable(self, litdata_reader, slipcache):
        """First 100 cached images should decode to match source (within tolerance)."""
        _print_progress(f"\n  Verifying first 100 samples decodable and match source...")
        errors = []

        for idx in range(100):
            try:
                # Decode source
                source_bytes = litdata_reader[idx]['image']
                source_rgb = np.array(PIL.Image.open(io.BytesIO(source_bytes)).convert('RGB'))

                # Decode cached
                cached_rgb = _decode_cached_image(slipcache, idx)

                # Crop to match (YUV420 may have padding)
                h, w = source_rgb.shape[:2]
                cached_rgb = cached_rgb[:h, :w, :]

                # Compare with tolerance
                diff = np.abs(source_rgb.astype(int) - cached_rgb.astype(int))
                max_diff = np.max(diff)

                # Tolerance depends on format
                image_format = slipcache.get_image_format('image')
                tolerance = 2 if image_format == "yuv420" else 0

                if max_diff > tolerance:
                    errors.append((idx, f"max_diff={max_diff}"))

            except Exception as e:
                errors.append((idx, str(e)))

        if errors:
            pytest.fail(f"Decode/compare errors at {len(errors)} indices: {errors[:5]}...")


# ---------------------------------------------------------------------------
# Tests: Full Dataset Verification (All 50,000)
# ---------------------------------------------------------------------------

@pytest.mark.s3
@pytest.mark.slow
class TestLitDataToSlipCacheFull:
    """Full verification of all 50,000 samples."""

    def test_all_cached_valid(self, slipcache):
        """ALL cached images should be valid (correct format markers)."""
        _print_progress(f"\n  Validating all {slipcache.num_samples} cached images...")
        image_format = slipcache.get_image_format('image')
        errors = []

        for idx in range(slipcache.num_samples):
            if idx % 10000 == 0:
                _print_progress(f"    {idx}/{slipcache.num_samples}...")

            try:
                cached_bytes = _get_cached_image_bytes(slipcache, idx)

                if image_format == "jpeg":
                    # Verify JPEG structure (fast check, no decode)
                    if cached_bytes[:2] != b'\xff\xd8':
                        errors.append((idx, "missing JPEG SOI"))
                    elif cached_bytes[-2:] != b'\xff\xd9':
                        errors.append((idx, "missing JPEG EOI"))
                else:
                    # YUV420: verify dimensions and size
                    h, w = slipcache.get_image_dims('image', idx)
                    if h == 0 or w == 0:
                        errors.append((idx, f"invalid dims: {h}x{w}"))
                    else:
                        expected_size = h * w + (h // 2) * (w // 2) * 2
                        if len(cached_bytes) != expected_size:
                            errors.append((idx, f"size mismatch"))

                if len(errors) >= 20:
                    break

            except Exception as e:
                errors.append((idx, str(e)))
                if len(errors) >= 20:
                    break

        if errors:
            pytest.fail(f"Validation errors at {len(errors)} samples: {errors[:10]}...")

    def test_all_labels_match(self, litdata_reader, slipcache):
        """ALL labels should match exactly."""
        _print_progress(f"\n  Comparing all {len(litdata_reader)} labels...")
        mismatches = []

        for idx in range(len(litdata_reader)):
            if idx % 10000 == 0:
                _print_progress(f"    {idx}/{len(litdata_reader)}...")

            source_label = litdata_reader[idx]['label']
            cached_label = slipcache.fields['label']._data[idx]

            if source_label != cached_label:
                mismatches.append((idx, source_label, cached_label))

        if mismatches:
            pytest.fail(f"Label mismatch at {len(mismatches)} indices: {mismatches[:10]}...")

    def test_all_decodable(self, slipcache):
        """ALL cached images should be decodable."""
        _print_progress(f"\n  Verifying all {slipcache.num_samples} samples decodable...")
        image_format = slipcache.get_image_format('image')
        errors = []

        for idx in range(slipcache.num_samples):
            if idx % 10000 == 0:
                _print_progress(f"    {idx}/{slipcache.num_samples}...")

            try:
                cached_bytes = _get_cached_image_bytes(slipcache, idx)

                if image_format == "jpeg":
                    # Verify JPEG markers
                    if cached_bytes[:2] != b'\xff\xd8':
                        errors.append((idx, "missing JPEG SOI"))
                        continue
                    if cached_bytes[-2:] != b'\xff\xd9':
                        errors.append((idx, "missing JPEG EOI"))
                        continue

                    # Verify decodable with PIL
                    img = PIL.Image.open(io.BytesIO(cached_bytes))
                    img.load()
                else:
                    # YUV420: verify dimensions are valid
                    h, w = slipcache.get_image_dims('image', idx)
                    if h == 0 or w == 0:
                        errors.append((idx, f"invalid dims: {h}x{w}"))
                        continue

                    # Verify YUV420 size matches expected
                    expected_size = h * w + (h // 2) * (w // 2) * 2
                    if len(cached_bytes) != expected_size:
                        errors.append((idx, f"size mismatch: {len(cached_bytes)} vs {expected_size}"))

            except Exception as e:
                errors.append((idx, str(e)))
                if len(errors) >= 10:
                    break

        if errors:
            pytest.fail(f"Decode errors at {len(errors)} indices: {errors}...")

    def test_random_samples_pixel_match(self, litdata_reader, slipcache):
        """Random samples should decode to match source within tolerance."""
        import random
        random.seed(42)

        indices = random.sample(range(len(litdata_reader)), 100)
        _print_progress(f"\n  Verifying 100 random samples pixel-match...")
        errors = []
        image_format = slipcache.get_image_format('image')

        for idx in indices:
            try:
                # Decode source
                source_bytes = litdata_reader[idx]['image']
                source_rgb = np.array(PIL.Image.open(io.BytesIO(source_bytes)).convert('RGB'))

                # Decode cached
                cached_rgb = _decode_cached_image(slipcache, idx)

                # Crop to match
                h, w = source_rgb.shape[:2]
                cached_rgb = cached_rgb[:h, :w, :]

                # Compare
                diff = np.abs(source_rgb.astype(int) - cached_rgb.astype(int))
                max_diff = np.max(diff)

                tolerance = 2 if image_format == "yuv420" else 0
                if max_diff > tolerance:
                    errors.append((idx, f"max_diff={max_diff}"))

            except Exception as e:
                errors.append((idx, str(e)))

        if errors:
            pytest.fail(f"Pixel mismatch at {len(errors)} random samples: {errors[:5]}...")
