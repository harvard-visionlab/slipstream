"""LitData reader verification against native litdata.StreamingDataset.

These tests verify that StreamingReader returns identical data to
the native litdata.StreamingDataset using real ImageNet validation data.

Two test modes:
- S3 path: Tests StreamingReader with remote_dir (streaming from S3)
- Local path: Tests StreamingReader with local_dir (cached data)

Run with:
    # All tests (requires AWS credentials, streams from S3)
    uv run pytest tests/test_litdata_verification.py -v -s

    # The -s flag shows progress output (recommended)

    # Skip S3 tests if no credentials
    uv run pytest tests/test_litdata_verification.py -v -m "not s3"
"""

import hashlib
import io
import sys
from pathlib import Path

import numpy as np
import PIL.Image
import pytest
from litdata import StreamingDataset as NativeStreamingDataset

from slipstream.readers.streaming import StreamingReader


# Real ImageNet validation set on S3 (LitData format)
LITDATA_VAL_S3_PATH = "s3://visionlab-datasets/imagenet1k/pre-processed/s256-l512-jpgbytes-q100-streaming/val/"


def _print_progress(msg: str) -> None:
    """Print to stderr so pytest doesn't capture it."""
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def litdata_cache_dir():
    """Get cache directory for LitData (persistent across test runs)."""
    from slipstream.dataset import get_default_cache_dir

    cache_dir = get_default_cache_dir() / "slipstream" / "litdata-test"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


@pytest.fixture(scope="module")
def check_s3_access():
    """Skip tests if S3 access is not available."""
    import boto3

    try:
        boto3.client('s3').head_bucket(Bucket='visionlab-datasets')
    except Exception as e:
        pytest.skip(f"S3 access not available: {e}")


@pytest.fixture(scope="module")
def native_litdata(check_s3_access, litdata_cache_dir):
    """Native litdata.StreamingDataset for reference."""
    _print_progress("\n[fixture] Creating native LitData StreamingDataset...")
    ds = NativeStreamingDataset(
        input_dir=LITDATA_VAL_S3_PATH,
        max_cache_size="50GB",
    )
    _print_progress(f"[fixture] Native LitData ready: {len(ds)} samples")
    return ds


@pytest.fixture(scope="module")
def slipstream_reader_s3(check_s3_access, litdata_cache_dir):
    """StreamingReader via S3 path (tests remote streaming)."""
    _print_progress("\n[fixture] Creating StreamingReader via S3 path...")
    reader = StreamingReader(
        remote_dir=LITDATA_VAL_S3_PATH,
        max_cache_size="50GB",
    )
    _print_progress(f"[fixture] S3 StreamingReader ready: {len(reader)} samples")
    return reader


@pytest.fixture(scope="module")
def slipstream_reader_local(check_s3_access, native_litdata, litdata_cache_dir):
    """StreamingReader via local cached path (tests local access).

    This fixture depends on native_litdata to ensure data is cached first.
    """
    # Get the cache path from native dataset
    # Native litdata uses input_dir.path, not cache_path (that's a slipstream addition)
    cache_path = native_litdata.input_dir.path
    if cache_path is None:
        pytest.skip("No local cache available")

    _print_progress(f"\n[fixture] Creating StreamingReader via local path: {cache_path}")
    reader = StreamingReader(
        local_dir=str(cache_path),
    )
    _print_progress(f"[fixture] Local StreamingReader ready: {len(reader)} samples")
    return reader


# Parametrized fixture to run tests against both S3 and local paths
@pytest.fixture(scope="module", params=["s3", "local"])
def litdata_reader(request, slipstream_reader_s3, slipstream_reader_local):
    """Parametrized fixture: runs each test twice (S3 and local paths)."""
    if request.param == "s3":
        return slipstream_reader_s3
    else:
        return slipstream_reader_local


# ---------------------------------------------------------------------------
# Tests: Structure and metadata (run against both S3 and local)
# ---------------------------------------------------------------------------

@pytest.mark.s3
class TestLitDataStructure:
    """Verify basic structure matches expected ImageNet val properties."""

    def test_sample_count_matches(self, litdata_reader, native_litdata):
        """Sample count should match native LitData."""
        assert len(litdata_reader) == len(native_litdata), \
            f"Length mismatch: slipstream={len(litdata_reader)}, native={len(native_litdata)}"

    def test_sample_count_is_50000(self, litdata_reader):
        """ImageNet val should have 50,000 samples."""
        assert len(litdata_reader) == 50000, \
            f"Expected 50000 samples, got {len(litdata_reader)}"

    def test_field_types_detected(self, litdata_reader):
        """Field types should be detected correctly."""
        field_types = litdata_reader.field_types
        assert 'image' in field_types, "Missing 'image' field"
        assert 'label' in field_types, "Missing 'label' field"


# ---------------------------------------------------------------------------
# Tests: Labels (run against both S3 and local)
# ---------------------------------------------------------------------------

@pytest.mark.s3
class TestLitDataLabels:
    """Verify labels match native LitData."""

    def test_all_labels_in_range(self, litdata_reader):
        """All labels should be in range [0, 999]."""
        _print_progress(f"\n  Checking labels for 1000 samples...")
        errors = []
        for idx in range(min(1000, len(litdata_reader))):
            sample = litdata_reader[idx]
            label = sample['label']
            if not (0 <= label < 1000):
                errors.append((idx, label))

        if errors:
            pytest.fail(f"Labels out of range: {errors[:10]}...")

    def test_labels_match_native(self, litdata_reader, native_litdata):
        """Labels should match native LitData exactly."""
        _print_progress(f"\n  Comparing labels for 100 samples...")
        mismatches = []
        for idx in range(100):
            slip_label = litdata_reader[idx]['label']
            native_label = native_litdata[idx]['label']

            if slip_label != native_label:
                mismatches.append((idx, slip_label, native_label))

        if mismatches:
            pytest.fail(f"Label mismatch at {len(mismatches)} indices: {mismatches}")


# ---------------------------------------------------------------------------
# Tests: Image bytes (run against both S3 and local)
# ---------------------------------------------------------------------------

@pytest.mark.s3
class TestLitDataBytes:
    """Verify image bytes match native LitData."""

    def test_image_bytes_match_native(self, litdata_reader, native_litdata):
        """Image bytes should match native LitData (SHA256)."""
        _print_progress(f"\n  Comparing bytes for 100 samples...")
        mismatches = []
        for idx in range(100):
            slip_bytes = litdata_reader[idx]['image']
            native_bytes = native_litdata[idx]['image']

            slip_hash = hashlib.sha256(slip_bytes).hexdigest()
            native_hash = hashlib.sha256(native_bytes).hexdigest()

            if slip_hash != native_hash:
                mismatches.append((idx, len(slip_bytes), len(native_bytes)))

        if mismatches:
            pytest.fail(f"Bytes mismatch at {len(mismatches)} indices: {mismatches}")

    def test_first_100_samples_valid_images(self, litdata_reader):
        """First 100 samples should be valid images (JPEG or PNG)."""
        _print_progress(f"\n  Validating 100 images...")
        errors = []
        for idx in range(100):
            try:
                sample = litdata_reader[idx]
                img_bytes = sample['image']

                # Check for valid image header (JPEG SOI or PNG signature)
                is_jpeg = img_bytes[:2] == b'\xff\xd8'
                is_png = img_bytes[:8] == b'\x89PNG\r\n\x1a\n'

                if not (is_jpeg or is_png):
                    errors.append((idx, f"unknown format: {img_bytes[:4].hex()}"))
                    continue

                # Verify image can be loaded
                img = PIL.Image.open(io.BytesIO(img_bytes))
                img.load()

            except Exception as e:
                errors.append((idx, str(e)))

        if errors:
            pytest.fail(f"Errors in {len(errors)} samples: {errors[:5]}...")


# ---------------------------------------------------------------------------
# Tests: Decoded images (run against both S3 and local)
# ---------------------------------------------------------------------------

@pytest.mark.s3
class TestLitDataDecode:
    """Verify decoded images match native LitData."""

    def test_decoded_images_match(self, litdata_reader, native_litdata):
        """Decoded pixels should match native (exact for same JPEG)."""
        _print_progress(f"\n  Comparing decoded pixels for 50 samples...")
        for idx in range(50):
            slip_bytes = litdata_reader[idx]['image']
            native_bytes = native_litdata[idx]['image']

            slip_img = np.array(PIL.Image.open(io.BytesIO(slip_bytes)))
            native_img = np.array(PIL.Image.open(io.BytesIO(native_bytes)))

            assert np.array_equal(slip_img, native_img), \
                f"Image {idx}: decoded pixels don't match"

    def test_random_samples_valid(self, litdata_reader):
        """Random samples across dataset should be valid."""
        import random
        random.seed(42)

        indices = random.sample(range(len(litdata_reader)), 50)
        _print_progress(f"\n  Validating 50 random samples...")
        errors = []

        for idx in indices:
            try:
                sample = litdata_reader[idx]
                img_bytes = sample['image']

                img = PIL.Image.open(io.BytesIO(img_bytes))
                img.load()

                w, h = img.size
                if w < 10 or h < 10 or w > 10000 or h > 10000:
                    errors.append((idx, f"unusual dimensions: {w}x{h}"))

            except Exception as e:
                errors.append((idx, str(e)))

        if errors:
            pytest.fail(f"Errors in {len(errors)} random samples: {errors}...")


# ---------------------------------------------------------------------------
# Tests: Paths and indices (run against both S3 and local)
# ---------------------------------------------------------------------------

@pytest.mark.s3
class TestLitDataPaths:
    """Verify path and index fields are correct."""

    def test_indices_match_native(self, litdata_reader, native_litdata):
        """Index field should match native LitData (original dataset index)."""
        _print_progress(f"\n  Comparing index field for 100 samples...")
        mismatches = []
        for idx in range(100):
            slip_sample = litdata_reader[idx]
            native_sample = native_litdata[idx]

            slip_index = slip_sample.get('index')
            native_index = native_sample.get('index')

            if slip_index != native_index:
                mismatches.append((idx, slip_index, native_index))

        if mismatches:
            pytest.fail(f"Index mismatch at {len(mismatches)} indices: {mismatches[:5]}...")

    def test_paths_match_native(self, litdata_reader, native_litdata):
        """Path field should match native LitData."""
        _print_progress(f"\n  Comparing paths for 100 samples...")
        mismatches = []
        for idx in range(100):
            slip_path = litdata_reader[idx].get('path', '')
            native_path = native_litdata[idx].get('path', '')

            if slip_path != native_path:
                mismatches.append((idx, slip_path, native_path))

        if mismatches:
            pytest.fail(f"Path mismatch at {len(mismatches)} indices: {mismatches[:5]}...")


# ---------------------------------------------------------------------------
# Tests: Full dataset scan (slower, optional)
# ---------------------------------------------------------------------------

@pytest.mark.s3
@pytest.mark.slow
class TestLitDataFullScan:
    """Full dataset validation (marked slow)."""

    def test_all_samples_readable(self, litdata_reader):
        """All 50,000 samples should be readable without error."""
        _print_progress(f"\n  Reading all {len(litdata_reader)} samples...")
        errors = []
        for idx in range(len(litdata_reader)):
            try:
                sample = litdata_reader[idx]
                img_bytes = sample['image']
                assert img_bytes[:2] == b'\xff\xd8', f"Sample {idx}: not a JPEG"
            except Exception as e:
                errors.append((idx, str(e)))
                if len(errors) >= 10:
                    break

            if idx % 10000 == 0:
                _print_progress(f"    {idx}/{len(litdata_reader)}...")

        if errors:
            pytest.fail(f"Errors reading {len(errors)} samples: {errors}...")
