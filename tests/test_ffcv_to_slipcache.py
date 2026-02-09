"""FFCV → SlipCache conversion verification.

These tests verify that:
1. JPEG images survive FFCV → SlipCache → load unchanged (byte-identical)
2. Labels and other fields are preserved exactly
3. ALL 50,000 ImageNet validation samples are verified (not just first 100)

Note: FFCV files already contain JPEG bytes (transcoded during dataset creation),
so there's no PNG-in-JPEG issue. All samples should be byte-identical.

Run with:
    uv run pytest tests/test_ffcv_to_slipcache.py -v -s

    # Skip slow full-dataset tests
    uv run pytest tests/test_ffcv_to_slipcache.py -v -s -m "not slow"
"""

import hashlib
import io
import sys
from pathlib import Path

import numpy as np
import PIL.Image
import pytest

from slipstream.readers.ffcv import FFCVFileReader
from slipstream.cache import OptimizedCache, detect_image_format


# Real FFCV file from S3 (ImageNet validation set)
FFCV_VAL_S3_PATH = (
    "s3://visionlab-datasets/imagenet1k/pre-processed/"
    "s256-l512-jpgbytes-q100-ffcv/imagenet1k-s256-l512-jpg-q100-cs100-val-7ac6386e.ffcv"
)


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
def ffcv_cache_dir():
    """Get cache directory for test data (persistent across test runs)."""
    from slipstream.dataset import get_default_cache_dir

    cache_dir = get_default_cache_dir() / "slipstream" / "ffcv-slipcache-test"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


@pytest.fixture(scope="module")
def ffcv_reader(check_s3_access, ffcv_cache_dir):
    """Create FFCVFileReader for ImageNet val."""
    _print_progress("\n[fixture] Creating FFCV reader...")

    reader = FFCVFileReader(
        FFCV_VAL_S3_PATH,
        cache_dir=str(ffcv_cache_dir),
        verbose=True
    )

    _print_progress(f"[fixture] FFCV reader ready: {len(reader)} samples")
    return reader


@pytest.fixture(scope="module")
def slipcache(ffcv_reader):
    """Build SlipCache from FFCV.

    The reader's cache_path property includes a dataset hash automatically,
    preventing stale cache issues when the source changes.

    FFCV files contain JPEG bytes, so this will be a JPEG cache.
    """
    # Use the reader's versioned cache_path (includes dataset hash)
    cache_dir = ffcv_reader.cache_path
    _print_progress(f"\n[fixture] Dataset hash: {ffcv_reader.dataset_hash}")
    _print_progress(f"[fixture] Cache path: {cache_dir}")

    manifest_path = cache_dir / ".slipstream" / "manifest.json"

    # Check if cache already exists
    if manifest_path.exists():
        _print_progress(f"[fixture] Loading existing SlipCache from {cache_dir}")
        return OptimizedCache.load(cache_dir, verbose=False)

    _print_progress(f"[fixture] Building SlipCache from FFCV...")
    _print_progress(f"[fixture] Output: {cache_dir}")

    # Build cache (auto-detects format - will be JPEG for FFCV)
    cache = OptimizedCache.build(
        ffcv_reader,
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
class TestFFCVToSlipCacheStructure:
    """Verify SlipCache structure matches source."""

    def test_sample_count_matches(self, ffcv_reader, slipcache):
        """Sample count should match source."""
        assert slipcache.num_samples == len(ffcv_reader), \
            f"Count mismatch: cache={slipcache.num_samples}, source={len(ffcv_reader)}"

    def test_sample_count_is_50000(self, slipcache):
        """ImageNet val should have 50,000 samples."""
        assert slipcache.num_samples == 50000, \
            f"Expected 50000 samples, got {slipcache.num_samples}"

    def test_fields_present(self, slipcache):
        """Required fields should be present."""
        assert 'image' in slipcache.fields, "Missing 'image' field"
        assert 'label' in slipcache.fields, "Missing 'label' field"

    def test_image_format_is_jpeg(self, slipcache):
        """FFCV source should produce JPEG cache."""
        image_format = slipcache.get_image_format('image')
        assert image_format == "jpeg", \
            f"Expected jpeg format for FFCV source, got {image_format}"


# ---------------------------------------------------------------------------
# Tests: Quick Verification (First 100)
# ---------------------------------------------------------------------------

@pytest.mark.s3
class TestFFCVToSlipCacheQuick:
    """Quick verification of first 100 samples."""

    def test_first_100_bytes_identical(self, ffcv_reader, slipcache):
        """First 100 JPEG samples should be byte-identical."""
        _print_progress(f"\n  Comparing first 100 JPEG samples...")
        mismatches = []

        for idx in range(100):
            source_bytes = ffcv_reader[idx]['image']
            cached_bytes = _get_cached_image_bytes(slipcache, idx)

            # FFCV already contains JPEG - should be byte-identical
            source_hash = hashlib.sha256(source_bytes).hexdigest()
            cached_hash = hashlib.sha256(cached_bytes).hexdigest()

            if source_hash != cached_hash:
                mismatches.append((idx, f"bytes differ: {len(source_bytes)} vs {len(cached_bytes)}"))

        if mismatches:
            pytest.fail(f"JPEG bytes mismatch at {len(mismatches)} indices: {mismatches[:5]}...")

    def test_first_100_labels_match(self, ffcv_reader, slipcache):
        """First 100 labels should match exactly."""
        _print_progress(f"\n  Comparing first 100 labels...")
        mismatches = []

        for idx in range(100):
            source_label = ffcv_reader[idx]['label']
            cached_label = slipcache.fields['label']._data[idx]

            if source_label != cached_label:
                mismatches.append((idx, source_label, cached_label))

        if mismatches:
            pytest.fail(f"Label mismatch at {len(mismatches)} indices: {mismatches}")

    def test_first_100_decodable(self, slipcache):
        """First 100 cached images should be decodable."""
        _print_progress(f"\n  Verifying first 100 samples decodable...")
        errors = []

        for idx in range(100):
            try:
                cached_bytes = _get_cached_image_bytes(slipcache, idx)

                # Verify JPEG markers
                if cached_bytes[:2] != b'\xff\xd8':
                    errors.append((idx, "missing SOI"))
                    continue
                if cached_bytes[-2:] != b'\xff\xd9':
                    errors.append((idx, "missing EOI"))
                    continue

                img = PIL.Image.open(io.BytesIO(cached_bytes))
                img.load()
            except Exception as e:
                errors.append((idx, str(e)))

        if errors:
            pytest.fail(f"Decode errors at {len(errors)} indices: {errors[:5]}...")


# ---------------------------------------------------------------------------
# Tests: Full Dataset Verification (All 50,000)
# ---------------------------------------------------------------------------

@pytest.mark.s3
@pytest.mark.slow
class TestFFCVToSlipCacheFull:
    """Full verification of all 50,000 samples."""

    def test_all_bytes_identical(self, ffcv_reader, slipcache):
        """ALL JPEG samples should be byte-identical."""
        _print_progress(f"\n  Comparing all {len(ffcv_reader)} samples...")
        mismatches = []

        for idx in range(len(ffcv_reader)):
            if idx % 5000 == 0:
                _print_progress(f"    {idx}/{len(ffcv_reader)}...")

            source_bytes = ffcv_reader[idx]['image']
            cached_bytes = _get_cached_image_bytes(slipcache, idx)

            source_hash = hashlib.sha256(source_bytes).hexdigest()
            cached_hash = hashlib.sha256(cached_bytes).hexdigest()

            if source_hash != cached_hash:
                mismatches.append((idx, "bytes differ"))
                if len(mismatches) >= 10:
                    break

        if mismatches:
            pytest.fail(f"Bytes mismatch at {len(mismatches)} samples: {mismatches[:10]}...")

    def test_all_labels_match(self, ffcv_reader, slipcache):
        """ALL labels should match exactly."""
        _print_progress(f"\n  Comparing all {len(ffcv_reader)} labels...")
        mismatches = []

        for idx in range(len(ffcv_reader)):
            if idx % 10000 == 0:
                _print_progress(f"    {idx}/{len(ffcv_reader)}...")

            source_label = ffcv_reader[idx]['label']
            cached_label = slipcache.fields['label']._data[idx]

            if source_label != cached_label:
                mismatches.append((idx, source_label, cached_label))

        if mismatches:
            pytest.fail(f"Label mismatch at {len(mismatches)} indices: {mismatches[:10]}...")

    def test_all_decodable(self, slipcache):
        """ALL cached images should be decodable."""
        _print_progress(f"\n  Verifying all {slipcache.num_samples} samples decodable...")
        errors = []

        for idx in range(slipcache.num_samples):
            if idx % 10000 == 0:
                _print_progress(f"    {idx}/{slipcache.num_samples}...")

            try:
                cached_bytes = _get_cached_image_bytes(slipcache, idx)

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

            except Exception as e:
                errors.append((idx, str(e)))
                if len(errors) >= 10:
                    break

        if errors:
            pytest.fail(f"Decode errors at {len(errors)} indices: {errors}...")

    def test_random_samples_pixel_match(self, ffcv_reader, slipcache):
        """Random samples should decode to identical pixels (JPEG→JPEG)."""
        import random
        random.seed(42)

        indices = random.sample(range(len(ffcv_reader)), 100)
        _print_progress(f"\n  Verifying 100 random samples pixel-match...")
        errors = []

        for idx in indices:
            try:
                # Decode source
                source_bytes = ffcv_reader[idx]['image']
                source_rgb = np.array(PIL.Image.open(io.BytesIO(source_bytes)).convert('RGB'))

                # Decode cached
                cached_rgb = _decode_cached_image(slipcache, idx)

                # Compare (should be identical for JPEG→JPEG)
                if not np.array_equal(source_rgb, cached_rgb):
                    diff = np.abs(source_rgb.astype(int) - cached_rgb.astype(int))
                    max_diff = np.max(diff)
                    errors.append((idx, f"max_diff={max_diff}"))

            except Exception as e:
                errors.append((idx, str(e)))

        if errors:
            pytest.fail(f"Pixel mismatch at {len(errors)} random samples: {errors[:5]}...")
