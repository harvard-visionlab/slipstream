"""ImageFolder reader verification against torchvision.

These tests verify that SlipstreamImageFolder returns identical data to
torchvision.datasets.ImageFolder using real ImageNet validation data.

Two test modes:
- S3 path: Tests open_imagefolder("s3://...") download/extract/cache flow
- Local path: Tests SlipstreamImageFolder("/local/path") direct access

Run with:
    # All tests (requires AWS credentials, downloads ~7GB on first run)
    uv run pytest tests/test_imagefolder_verification.py -v -s

    # The -s flag shows download/extraction progress (recommended for first run)

    # Skip S3 tests if no credentials
    uv run pytest tests/test_imagefolder_verification.py -v -m "not s3"
"""

import hashlib
import io
import sys
from pathlib import Path

import numpy as np
import PIL.Image
import pytest
import torchvision.datasets

from slipstream.readers.imagefolder import SlipstreamImageFolder, open_imagefolder


# Real ImageNet validation set on S3
IMAGENET_VAL_S3_PATH = "s3://visionlab-datasets/imagenet1k-raw/val.tar.gz"


def _print_progress(msg: str) -> None:
    """Print to stderr so pytest doesn't capture it."""
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def imagenet_cache_dir(tmp_path_factory):
    """Get or create cache directory for ImageNet data."""
    import os

    cache_dir = os.environ.get("SLIPSTREAM_CACHE_DIR")
    if cache_dir:
        return Path(cache_dir) / "imagefolder"
    return tmp_path_factory.mktemp("imagenet_cache")


@pytest.fixture(scope="module")
def imagenet_extracted_path(imagenet_cache_dir):
    """Ensure ImageNet val is downloaded and extracted, return local path.

    This is the base fixture that handles S3 download/extraction.
    Returns the path to the extracted ImageFolder directory.
    """
    import boto3

    try:
        boto3.client('s3').head_bucket(Bucket='visionlab-datasets')
    except Exception as e:
        pytest.skip(f"S3 access not available: {e}")

    _print_progress("\n[fixture] Ensuring ImageNet val is available...")

    try:
        # Use open_imagefolder to download/extract (caches automatically)
        reader = open_imagefolder(
            IMAGENET_VAL_S3_PATH,
            cache_dir=str(imagenet_cache_dir),
            verbose=True  # Shows progress to stdout
        )
        # Return the extracted local path
        local_path = reader._root_path
        _print_progress(f"[fixture] ImageNet val ready at: {local_path}")
        return local_path
    except Exception as e:
        pytest.skip(f"Failed to download ImageNet: {e}")


@pytest.fixture(scope="module")
def imagenet_val_via_s3(imagenet_cache_dir, imagenet_extracted_path):
    """Reader created via S3 path (tests open_imagefolder S3 flow)."""
    _print_progress("\n[fixture] Creating reader via S3 path...")
    reader = open_imagefolder(
        IMAGENET_VAL_S3_PATH,
        cache_dir=str(imagenet_cache_dir),
        verbose=False  # Already downloaded, skip verbose
    )
    _print_progress(f"[fixture] S3 reader ready: {len(reader)} samples")
    return reader


@pytest.fixture(scope="module")
def imagenet_val_via_local(imagenet_extracted_path):
    """Reader created via local path (tests SlipstreamImageFolder direct)."""
    _print_progress("\n[fixture] Creating reader via local path...")
    reader = SlipstreamImageFolder(str(imagenet_extracted_path))
    _print_progress(f"[fixture] Local reader ready: {len(reader)} samples")
    return reader


@pytest.fixture(scope="module")
def torchvision_dataset(imagenet_extracted_path):
    """Reference torchvision.datasets.ImageFolder for comparison."""
    return torchvision.datasets.ImageFolder(str(imagenet_extracted_path))


# Parametrized fixture to run tests against both S3 and local paths
@pytest.fixture(scope="module", params=["s3", "local"])
def imagenet_reader(request, imagenet_val_via_s3, imagenet_val_via_local):
    """Parametrized fixture: runs each test twice (S3 path and local path)."""
    if request.param == "s3":
        return imagenet_val_via_s3
    else:
        return imagenet_val_via_local


# ---------------------------------------------------------------------------
# Tests: Structure and metadata (run against both S3 and local)
# ---------------------------------------------------------------------------

@pytest.mark.s3
class TestImageFolderStructure:
    """Verify basic structure matches expected ImageNet val properties."""

    def test_sample_count_matches(self, imagenet_reader):
        """ImageNet val should have 50,000 samples."""
        assert len(imagenet_reader) == 50000, \
            f"Expected 50000 samples, got {len(imagenet_reader)}"

    def test_class_count_matches(self, imagenet_reader):
        """ImageNet val should have 1000 classes."""
        assert len(imagenet_reader.classes) == 1000, \
            f"Expected 1000 classes, got {len(imagenet_reader.classes)}"

    def test_class_to_idx_consistent(self, imagenet_reader):
        """Class to index mapping should be consistent."""
        indices = set(imagenet_reader.class_to_idx.values())
        assert indices == set(range(1000)), \
            "class_to_idx should map to indices 0-999"


# ---------------------------------------------------------------------------
# Tests: Labels (run against both S3 and local)
# ---------------------------------------------------------------------------

@pytest.mark.s3
class TestImageFolderLabels:
    """Verify labels are correct."""

    def test_all_labels_in_range(self, imagenet_reader):
        """All labels should be in range [0, 999]."""
        _print_progress(f"\n  Checking labels for 1000 samples...")
        errors = []
        for idx in range(min(1000, len(imagenet_reader))):
            label = imagenet_reader[idx]['label']
            if not (0 <= label < 1000):
                errors.append((idx, label))

        if errors:
            pytest.fail(f"Labels out of range: {errors[:10]}...")

    def test_labels_match_torchvision(self, imagenet_reader, torchvision_dataset):
        """Labels should match torchvision.datasets.ImageFolder ordering."""
        _print_progress(f"\n  Comparing labels for 100 samples...")
        mismatches = []
        for idx in range(100):
            slip_label = imagenet_reader[idx]['label']
            _, tv_label = torchvision_dataset[idx]

            if slip_label != tv_label:
                mismatches.append((idx, slip_label, tv_label))

        if mismatches:
            pytest.fail(f"Label mismatch at {len(mismatches)} indices: {mismatches}")


# ---------------------------------------------------------------------------
# Tests: Image bytes (run against both S3 and local)
# ---------------------------------------------------------------------------

@pytest.mark.s3
class TestImageFolderBytes:
    """Verify image bytes match direct file reads."""

    def test_image_bytes_match_file(self, imagenet_reader, torchvision_dataset):
        """Image bytes should match torchvision sample paths."""
        _print_progress(f"\n  Comparing bytes for 100 samples...")
        mismatches = []
        for idx in range(100):
            slip_bytes = imagenet_reader[idx]['image']

            file_path, _ = torchvision_dataset.samples[idx]
            file_bytes = Path(file_path).read_bytes()

            slip_hash = hashlib.sha256(slip_bytes).hexdigest()
            file_hash = hashlib.sha256(file_bytes).hexdigest()

            if slip_hash != file_hash:
                mismatches.append((idx, len(slip_bytes), len(file_bytes)))

        if mismatches:
            pytest.fail(f"Bytes mismatch at {len(mismatches)} indices: {mismatches}")

    def test_first_100_samples_valid_jpeg(self, imagenet_reader):
        """First 100 samples should be valid JPEGs."""
        _print_progress(f"\n  Validating 100 JPEGs...")
        errors = []
        for idx in range(100):
            try:
                sample = imagenet_reader[idx]
                img_bytes = sample['image']

                if img_bytes[:2] != b'\xff\xd8':
                    errors.append((idx, "missing SOI"))
                elif img_bytes[-2:] != b'\xff\xd9':
                    errors.append((idx, "missing EOI"))

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
class TestImageFolderDecode:
    """Verify decoded images match torchvision."""

    def test_decoded_images_match(self, imagenet_reader, torchvision_dataset):
        """Decoded pixels should match torchvision (exact for same JPEG)."""
        _print_progress(f"\n  Comparing decoded pixels for 50 samples...")
        for idx in range(50):
            slip_bytes = imagenet_reader[idx]['image']
            slip_img = np.array(PIL.Image.open(io.BytesIO(slip_bytes)))

            tv_img, _ = torchvision_dataset[idx]
            tv_img = np.array(tv_img)

            assert np.array_equal(slip_img, tv_img), \
                f"Image {idx}: decoded pixels don't match"

    def test_random_samples_valid(self, imagenet_reader):
        """Random samples across dataset should be valid."""
        import random
        random.seed(42)

        indices = random.sample(range(len(imagenet_reader)), 50)
        _print_progress(f"\n  Validating 50 random samples...")
        errors = []

        for idx in indices:
            try:
                sample = imagenet_reader[idx]
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
# Tests: Paths (run against both S3 and local)
# ---------------------------------------------------------------------------

@pytest.mark.s3
class TestImageFolderPaths:
    """Verify path field is correct."""

    def test_paths_have_class_and_filename(self, imagenet_reader):
        """Paths should contain class folder and filename."""
        _print_progress(f"\n  Checking path structure for 100 samples...")
        for idx in range(100):
            sample = imagenet_reader[idx]
            if 'path' in sample:
                path = Path(sample['path'])
                assert len(path.parts) >= 2, f"Path too short: {path}"
                assert path.suffix.lower() in {'.jpg', '.jpeg', '.png'}, \
                    f"Expected image extension: {path}"

    def test_path_class_matches_label(self, imagenet_reader):
        """Path class folder should match the label."""
        _print_progress(f"\n  Verifying path/label consistency for 100 samples...")
        for idx in range(100):
            sample = imagenet_reader[idx]
            if 'path' in sample:
                path = Path(sample['path'])
                label = sample['label']
                class_name = imagenet_reader.classes[label]

                assert class_name in path.parts, \
                    f"Class '{class_name}' not in path {path}"


# ---------------------------------------------------------------------------
# Tests: S3 Caching (S3-specific)
# ---------------------------------------------------------------------------

@pytest.mark.s3
class TestImageFolderS3Cache:
    """Test that cached S3 data works correctly."""

    def test_s3_uses_cache_on_second_load(self, imagenet_val_via_s3, imagenet_cache_dir):
        """Second load should use cached extraction, not re-download."""
        _print_progress(f"\n  Testing cache reuse...")

        # Create a second reader pointing to same S3 path
        reader2 = open_imagefolder(
            IMAGENET_VAL_S3_PATH,
            cache_dir=str(imagenet_cache_dir),
            verbose=False
        )

        assert len(reader2) == len(imagenet_val_via_s3)

        # First sample should be identical
        sample1 = imagenet_val_via_s3[0]
        sample2 = reader2[0]

        assert hashlib.sha256(sample1['image']).hexdigest() == \
               hashlib.sha256(sample2['image']).hexdigest()
        assert sample1['label'] == sample2['label']
        _print_progress("  Cache reuse verified!")
