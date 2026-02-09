"""Cache integrity verification - rigorous byte-level testing.

This test suite verifies cache correctness with NO tolerance for errors:
- JPEG source → JPEG cache: bytes MUST be identical (SHA256 hash match)
- PNG source → JPEG cache: must be transcoded, pixels must match within JPEG tolerance
- All 50,000 samples tested, not just first 100 or random samples

The cache is rebuilt fresh for each test run to verify current code, not stale cache.

Run with:
    uv run pytest tests/test_cache_integrity.py -v -s

This test takes ~10-20 minutes to run all 50,000 samples.
"""

import hashlib
import io
import sys
import tempfile
from pathlib import Path

import numpy as np
import PIL.Image
import pytest

from slipstream.readers.imagefolder import open_imagefolder
from slipstream.cache import OptimizedCache, detect_image_format


# Real ImageNet validation set on S3
IMAGENET_VAL_S3_PATH = "s3://visionlab-datasets/imagenet1k-raw/val.tar.gz"


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
def imagenet_cache_dir():
    """Get cache directory for ImageNet data (persistent for source data only)."""
    from slipstream.dataset import get_default_cache_dir

    cache_dir = get_default_cache_dir() / "slipstream" / "imagefolder"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


@pytest.fixture(scope="module")
def imagefolder_reader(check_s3_access, imagenet_cache_dir):
    """Create ImageFolder reader for ImageNet val."""
    _print_progress("\n[fixture] Creating ImageFolder reader...")

    reader = open_imagefolder(
        IMAGENET_VAL_S3_PATH,
        cache_dir=str(imagenet_cache_dir),
        verbose=True
    )

    _print_progress(f"[fixture] ImageFolder reader ready: {len(reader)} samples")
    return reader


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


def _decode_to_rgb(img_bytes: bytes) -> np.ndarray:
    """Decode image bytes to RGB numpy array."""
    return np.array(PIL.Image.open(io.BytesIO(img_bytes)).convert('RGB'))


# ---------------------------------------------------------------------------
# Test: JPEG Byte-Identical Verification
# ---------------------------------------------------------------------------

@pytest.mark.s3
@pytest.mark.slow
class TestCacheIntegrityJPEG:
    """Verify JPEG bytes are preserved exactly through cache round-trip.

    This is the definitive test: if source is JPEG and cache mode is JPEG,
    the cached bytes MUST be byte-identical to the source bytes.

    Any difference indicates a bug in the cache builder.
    """

    def test_all_jpeg_samples_byte_identical(self, imagefolder_reader):
        """ALL JPEG samples must have byte-identical cached bytes."""
        _print_progress("\n" + "="*70)
        _print_progress("CACHE INTEGRITY TEST: JPEG Byte-Identical Verification")
        _print_progress("="*70)

        # Build cache in temp directory (fresh build, no stale cache)
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_dir = Path(tmp_dir) / "slipcache"

            _print_progress(f"\n[1/3] Building fresh cache in {cache_dir}...")
            cache = OptimizedCache.build(
                imagefolder_reader,
                output_dir=cache_dir,
                verbose=True,
            )

            # Verify cache format is JPEG (first sample should be JPEG)
            image_format = cache.get_image_format('image')
            _print_progress(f"[2/3] Cache format: {image_format}")
            assert image_format == "jpeg", \
                f"Expected JPEG cache format, got {image_format}"

            # Load the cache
            loaded = OptimizedCache.load(cache_dir, verbose=False)

            _print_progress(f"[3/3] Verifying ALL {len(imagefolder_reader)} samples...")
            _print_progress("")

            # Track results
            jpeg_matched = 0
            jpeg_mismatched = []
            png_transcoded = 0
            png_failed = []
            other_format = 0
            errors = []

            for idx in range(len(imagefolder_reader)):
                if idx % 5000 == 0:
                    _print_progress(f"  Progress: {idx}/{len(imagefolder_reader)} "
                                   f"(JPEG OK: {jpeg_matched}, PNG transcoded: {png_transcoded}, "
                                   f"errors: {len(jpeg_mismatched) + len(png_failed) + len(errors)})")

                try:
                    # Get source bytes
                    source_bytes = imagefolder_reader[idx]['image']
                    source_format = detect_image_format(source_bytes)

                    # Get cached bytes
                    cached_bytes = _get_cached_image_bytes(loaded, idx)
                    cached_format = detect_image_format(cached_bytes)

                    if source_format == "jpeg":
                        # JPEG source: must be byte-identical
                        source_hash = hashlib.sha256(source_bytes).hexdigest()
                        cached_hash = hashlib.sha256(cached_bytes).hexdigest()

                        if source_hash == cached_hash:
                            jpeg_matched += 1
                        else:
                            jpeg_mismatched.append({
                                'idx': idx,
                                'source_size': len(source_bytes),
                                'cached_size': len(cached_bytes),
                                'source_hash': source_hash[:16],
                                'cached_hash': cached_hash[:16],
                            })

                    elif source_format == "png":
                        # PNG source: must be transcoded to JPEG
                        if cached_format != "jpeg":
                            png_failed.append({
                                'idx': idx,
                                'reason': f"not transcoded to JPEG (cached as {cached_format})",
                            })
                            continue

                        # Verify pixels match within JPEG tolerance
                        source_rgb = _decode_to_rgb(source_bytes)
                        cached_rgb = _decode_to_rgb(cached_bytes)

                        # Dimensions must match
                        if source_rgb.shape != cached_rgb.shape:
                            png_failed.append({
                                'idx': idx,
                                'reason': f"dimension mismatch: {source_rgb.shape} vs {cached_rgb.shape}",
                            })
                            continue

                        # Pixels must be close (JPEG compression tolerance)
                        diff = np.abs(source_rgb.astype(int) - cached_rgb.astype(int))
                        max_diff = int(np.max(diff))
                        mean_diff = float(np.mean(diff))

                        # JPEG quality 100 should be very close
                        if max_diff > 5 or mean_diff > 1.0:
                            png_failed.append({
                                'idx': idx,
                                'reason': f"pixel mismatch: max={max_diff}, mean={mean_diff:.2f}",
                            })
                        else:
                            png_transcoded += 1

                    else:
                        # Other format (BMP, etc.)
                        other_format += 1

                except Exception as e:
                    errors.append({
                        'idx': idx,
                        'error': str(e),
                    })

            # Report results
            _print_progress("")
            _print_progress("="*70)
            _print_progress("RESULTS")
            _print_progress("="*70)
            _print_progress(f"Total samples:      {len(imagefolder_reader)}")
            _print_progress(f"JPEG byte-identical: {jpeg_matched}")
            _print_progress(f"PNG transcoded OK:   {png_transcoded}")
            _print_progress(f"Other formats:       {other_format}")
            _print_progress(f"JPEG mismatched:     {len(jpeg_mismatched)}")
            _print_progress(f"PNG failed:          {len(png_failed)}")
            _print_progress(f"Errors:              {len(errors)}")
            _print_progress("")

            # Report failures in detail
            if jpeg_mismatched:
                _print_progress("JPEG MISMATCHES (first 10):")
                for m in jpeg_mismatched[:10]:
                    _print_progress(f"  idx={m['idx']}: {m['source_size']} bytes -> {m['cached_size']} bytes")
                _print_progress("")

            if png_failed:
                _print_progress("PNG FAILURES (first 10):")
                for f in png_failed[:10]:
                    _print_progress(f"  idx={f['idx']}: {f['reason']}")
                _print_progress("")

            if errors:
                _print_progress("ERRORS (first 10):")
                for e in errors[:10]:
                    _print_progress(f"  idx={e['idx']}: {e['error']}")
                _print_progress("")

            # Assert success
            total_failures = len(jpeg_mismatched) + len(png_failed) + len(errors)
            assert total_failures == 0, \
                f"Cache integrity check failed: {len(jpeg_mismatched)} JPEG mismatches, " \
                f"{len(png_failed)} PNG failures, {len(errors)} errors"

            _print_progress("ALL SAMPLES VERIFIED SUCCESSFULLY")


# ---------------------------------------------------------------------------
# Test: Known Problematic Samples
# ---------------------------------------------------------------------------

@pytest.mark.s3
class TestKnownProblematicSamples:
    """Test specific samples known to trigger edge cases.

    These are samples that have caused bugs in the past:
    - Sample 18025: Has EXIF thumbnail with early FFD9 marker
    - PNG-in-JPEG files: PNG files with .JPEG extension
    """

    def test_sample_18025_not_truncated(self, imagefolder_reader):
        """Sample 18025 must not be truncated by EXIF FFD9 marker."""
        _print_progress("\n[test] Checking sample 18025 (EXIF edge case)...")

        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_dir = Path(tmp_dir) / "slipcache"

            # Build cache
            cache = OptimizedCache.build(
                imagefolder_reader,
                output_dir=cache_dir,
                verbose=False,
            )
            loaded = OptimizedCache.load(cache_dir, verbose=False)

            # Get source and cached bytes for sample 18025
            idx = 18025
            source_bytes = imagefolder_reader[idx]['image']
            cached_bytes = _get_cached_image_bytes(loaded, idx)

            source_format = detect_image_format(source_bytes)
            _print_progress(f"  Source format: {source_format}")
            _print_progress(f"  Source size: {len(source_bytes)} bytes")
            _print_progress(f"  Cached size: {len(cached_bytes)} bytes")

            if source_format == "jpeg":
                # JPEG: must be byte-identical
                source_hash = hashlib.sha256(source_bytes).hexdigest()
                cached_hash = hashlib.sha256(cached_bytes).hexdigest()

                assert source_hash == cached_hash, \
                    f"Sample 18025 was corrupted: {len(source_bytes)} bytes -> {len(cached_bytes)} bytes"

                _print_progress("  PASS: Bytes are identical")
            else:
                # PNG: verify transcoding
                assert detect_image_format(cached_bytes) == "jpeg", \
                    f"Sample 18025 should be transcoded to JPEG"

                source_rgb = _decode_to_rgb(source_bytes)
                cached_rgb = _decode_to_rgb(cached_bytes)

                assert source_rgb.shape == cached_rgb.shape, \
                    f"Dimension mismatch after transcoding"

                diff = np.abs(source_rgb.astype(int) - cached_rgb.astype(int))
                max_diff = int(np.max(diff))

                assert max_diff <= 5, \
                    f"Pixel mismatch after transcoding: max_diff={max_diff}"

                _print_progress(f"  PASS: Transcoded correctly (max_diff={max_diff})")

    def test_find_all_png_in_jpeg_samples(self, imagefolder_reader):
        """Identify all PNG files with JPEG extension and verify handling."""
        _print_progress("\n[test] Scanning for PNG-in-JPEG samples...")

        png_samples = []

        # Scan all samples
        for idx in range(len(imagefolder_reader)):
            if idx % 10000 == 0:
                _print_progress(f"  Scanned {idx}/{len(imagefolder_reader)}...")

            source_bytes = imagefolder_reader[idx]['image']
            if detect_image_format(source_bytes) == "png":
                sample = imagefolder_reader[idx]
                path = sample.get('path', f'idx_{idx}')
                png_samples.append({'idx': idx, 'path': path, 'size': len(source_bytes)})

        _print_progress(f"\n  Found {len(png_samples)} PNG files in dataset")

        if png_samples:
            _print_progress("  PNG samples (first 20):")
            for s in png_samples[:20]:
                _print_progress(f"    idx={s['idx']}: {s['path']} ({s['size']} bytes)")

        # Now verify each PNG is transcoded correctly in cache
        if png_samples:
            _print_progress(f"\n  Verifying PNG transcoding in cache...")

            with tempfile.TemporaryDirectory() as tmp_dir:
                cache_dir = Path(tmp_dir) / "slipcache"

                cache = OptimizedCache.build(
                    imagefolder_reader,
                    output_dir=cache_dir,
                    verbose=False,
                )
                loaded = OptimizedCache.load(cache_dir, verbose=False)

                failures = []
                for s in png_samples:
                    idx = s['idx']
                    source_bytes = imagefolder_reader[idx]['image']
                    cached_bytes = _get_cached_image_bytes(loaded, idx)

                    # Must be transcoded to JPEG
                    if detect_image_format(cached_bytes) != "jpeg":
                        failures.append(f"idx={idx}: not transcoded")
                        continue

                    # Pixels must match
                    try:
                        source_rgb = _decode_to_rgb(source_bytes)
                        cached_rgb = _decode_to_rgb(cached_bytes)

                        if source_rgb.shape != cached_rgb.shape:
                            failures.append(f"idx={idx}: dimension mismatch")
                            continue

                        diff = np.abs(source_rgb.astype(int) - cached_rgb.astype(int))
                        max_diff = int(np.max(diff))

                        if max_diff > 5:
                            failures.append(f"idx={idx}: max_diff={max_diff}")
                    except Exception as e:
                        failures.append(f"idx={idx}: {e}")

                if failures:
                    _print_progress(f"\n  FAILURES ({len(failures)}):")
                    for f in failures[:10]:
                        _print_progress(f"    {f}")
                    pytest.fail(f"{len(failures)} PNG samples failed transcoding")
                else:
                    _print_progress(f"  All {len(png_samples)} PNG samples transcoded correctly")


# ---------------------------------------------------------------------------
# Test: Quick Sanity Check (for CI)
# ---------------------------------------------------------------------------

@pytest.mark.s3
class TestCacheIntegrityQuick:
    """Quick sanity check for CI - tests first 1000 samples."""

    def test_first_1000_samples_integrity(self, imagefolder_reader):
        """Verify first 1000 samples (quick check for CI)."""
        _print_progress("\n[quick] Testing first 1000 samples...")

        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_dir = Path(tmp_dir) / "slipcache"

            cache = OptimizedCache.build(
                imagefolder_reader,
                output_dir=cache_dir,
                verbose=False,
            )
            loaded = OptimizedCache.load(cache_dir, verbose=False)

            failures = []

            for idx in range(min(1000, len(imagefolder_reader))):
                source_bytes = imagefolder_reader[idx]['image']
                cached_bytes = _get_cached_image_bytes(loaded, idx)

                source_format = detect_image_format(source_bytes)

                if source_format == "jpeg":
                    # Must be byte-identical
                    if hashlib.sha256(source_bytes).hexdigest() != \
                       hashlib.sha256(cached_bytes).hexdigest():
                        failures.append(f"idx={idx}: JPEG bytes differ")
                else:
                    # Must be transcoded and decodable
                    if detect_image_format(cached_bytes) != "jpeg":
                        failures.append(f"idx={idx}: not transcoded to JPEG")
                        continue

                    try:
                        source_rgb = _decode_to_rgb(source_bytes)
                        cached_rgb = _decode_to_rgb(cached_bytes)

                        if source_rgb.shape != cached_rgb.shape:
                            failures.append(f"idx={idx}: dimension mismatch")
                    except Exception as e:
                        failures.append(f"idx={idx}: {e}")

            if failures:
                for f in failures[:10]:
                    _print_progress(f"  FAIL: {f}")
                pytest.fail(f"{len(failures)} samples failed integrity check")

            _print_progress(f"  PASS: All 1000 samples verified")
