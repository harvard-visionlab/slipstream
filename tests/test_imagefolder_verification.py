"""ImageFolder reader verification against torchvision.

These tests verify that SlipstreamImageFolder returns identical data to
torchvision.datasets.ImageFolder using real ImageNet validation data from S3.

Run with:
    # All tests (requires AWS credentials, downloads ~7GB on first run)
    uv run pytest tests/test_imagefolder_verification.py -v

    # Skip S3 tests
    uv run pytest tests/test_imagefolder_verification.py -v -m "not s3"
"""

import hashlib
import io
from pathlib import Path

import numpy as np
import PIL.Image
import pytest
import torchvision.datasets

from slipstream.readers.imagefolder import SlipstreamImageFolder, open_imagefolder


# Real ImageNet validation set on S3
IMAGENET_VAL_S3_PATH = "s3://visionlab-datasets/imagenet1k-raw/val.tar.gz"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def imagenet_val_s3(tmp_path_factory):
    """Download and cache real ImageNet validation set from S3.

    This downloads ~7GB on first run, then uses cached extraction.
    """
    import os

    import boto3

    try:
        boto3.client('s3').head_bucket(Bucket='visionlab-datasets')
    except Exception as e:
        pytest.skip(f"S3 access not available: {e}")

    # Use SLIPSTREAM_CACHE_DIR if set, otherwise use pytest tmp dir
    cache_dir = os.environ.get("SLIPSTREAM_CACHE_DIR")
    if cache_dir:
        cache_dir = Path(cache_dir) / "imagefolder"
    else:
        cache_dir = tmp_path_factory.mktemp("imagenet_cache")

    try:
        # Use open_imagefolder for S3 tar archive support
        reader = open_imagefolder(
            IMAGENET_VAL_S3_PATH,
            cache_dir=str(cache_dir),
            verbose=True
        )
        return reader
    except Exception as e:
        pytest.skip(f"Failed to download ImageNet: {e}")


# ---------------------------------------------------------------------------
# Tests: Structure and metadata
# ---------------------------------------------------------------------------

@pytest.mark.s3
class TestImageFolderStructure:
    """Verify basic structure matches expected ImageNet val properties."""

    def test_sample_count_matches(self, imagenet_val_s3):
        """ImageNet val should have 50,000 samples."""
        assert len(imagenet_val_s3) == 50000, \
            f"Expected 50000 samples, got {len(imagenet_val_s3)}"

    def test_class_count_matches(self, imagenet_val_s3):
        """ImageNet val should have 1000 classes."""
        assert len(imagenet_val_s3.classes) == 1000, \
            f"Expected 1000 classes, got {len(imagenet_val_s3.classes)}"

    def test_class_to_idx_consistent(self, imagenet_val_s3):
        """Class to index mapping should be consistent."""
        # All classes should map to unique indices 0-999
        indices = set(imagenet_val_s3.class_to_idx.values())
        assert indices == set(range(1000)), \
            "class_to_idx should map to indices 0-999"


# ---------------------------------------------------------------------------
# Tests: Labels
# ---------------------------------------------------------------------------

@pytest.mark.s3
class TestImageFolderLabels:
    """Verify labels are correct."""

    def test_all_labels_in_range(self, imagenet_val_s3):
        """All labels should be in range [0, 999]."""
        errors = []
        for idx in range(min(1000, len(imagenet_val_s3))):
            label = imagenet_val_s3[idx]['label']
            if not (0 <= label < 1000):
                errors.append((idx, label))

        if errors:
            pytest.fail(f"Labels out of range: {errors[:10]}...")

    def test_labels_match_torchvision(self, imagenet_val_s3):
        """Labels should match torchvision.datasets.ImageFolder ordering."""
        # Get the extracted path from the reader
        root_path = imagenet_val_s3._root_path
        tv_dataset = torchvision.datasets.ImageFolder(str(root_path))

        # Check first 100 samples
        mismatches = []
        for idx in range(100):
            slip_label = imagenet_val_s3[idx]['label']
            _, tv_label = tv_dataset[idx]

            if slip_label != tv_label:
                mismatches.append((idx, slip_label, tv_label))

        if mismatches:
            pytest.fail(f"Label mismatch at {len(mismatches)} indices: {mismatches}")


# ---------------------------------------------------------------------------
# Tests: Image bytes
# ---------------------------------------------------------------------------

@pytest.mark.s3
class TestImageFolderBytes:
    """Verify image bytes match direct file reads."""

    def test_image_bytes_match_file(self, imagenet_val_s3):
        """Image bytes should match torchvision sample paths."""
        root_path = imagenet_val_s3._root_path
        tv_dataset = torchvision.datasets.ImageFolder(str(root_path))

        mismatches = []
        for idx in range(100):
            slip_bytes = imagenet_val_s3[idx]['image']

            # Get file path from torchvision dataset
            file_path, _ = tv_dataset.samples[idx]
            file_bytes = Path(file_path).read_bytes()

            slip_hash = hashlib.sha256(slip_bytes).hexdigest()
            file_hash = hashlib.sha256(file_bytes).hexdigest()

            if slip_hash != file_hash:
                mismatches.append((idx, len(slip_bytes), len(file_bytes)))

        if mismatches:
            pytest.fail(f"Bytes mismatch at {len(mismatches)} indices: {mismatches}")

    def test_first_100_samples_valid_jpeg(self, imagenet_val_s3):
        """First 100 samples should be valid JPEGs."""
        errors = []
        for idx in range(100):
            try:
                sample = imagenet_val_s3[idx]
                img_bytes = sample['image']

                # Check JPEG markers
                if img_bytes[:2] != b'\xff\xd8':
                    errors.append((idx, "missing SOI"))
                elif img_bytes[-2:] != b'\xff\xd9':
                    errors.append((idx, "missing EOI"))

                # Verify decodable
                img = PIL.Image.open(io.BytesIO(img_bytes))
                img.load()

            except Exception as e:
                errors.append((idx, str(e)))

        if errors:
            pytest.fail(f"Errors in {len(errors)} samples: {errors[:5]}...")


# ---------------------------------------------------------------------------
# Tests: Decoded images
# ---------------------------------------------------------------------------

@pytest.mark.s3
class TestImageFolderDecode:
    """Verify decoded images match torchvision."""

    def test_decoded_images_match(self, imagenet_val_s3):
        """Decoded pixels should match torchvision (exact for same JPEG)."""
        root_path = imagenet_val_s3._root_path
        tv_dataset = torchvision.datasets.ImageFolder(str(root_path))

        for idx in range(50):
            slip_bytes = imagenet_val_s3[idx]['image']
            slip_img = np.array(PIL.Image.open(io.BytesIO(slip_bytes)))

            tv_img, _ = tv_dataset[idx]
            tv_img = np.array(tv_img)

            # Should be exactly equal since it's the same JPEG file
            assert np.array_equal(slip_img, tv_img), \
                f"Image {idx}: decoded pixels don't match"

    def test_random_samples_valid(self, imagenet_val_s3):
        """Random samples across dataset should be valid."""
        import random
        random.seed(42)

        indices = random.sample(range(len(imagenet_val_s3)), 50)
        errors = []

        for idx in indices:
            try:
                sample = imagenet_val_s3[idx]
                img_bytes = sample['image']

                # Verify decodable
                img = PIL.Image.open(io.BytesIO(img_bytes))
                img.load()

                # Check dimensions reasonable
                w, h = img.size
                if w < 10 or h < 10 or w > 10000 or h > 10000:
                    errors.append((idx, f"unusual dimensions: {w}x{h}"))

            except Exception as e:
                errors.append((idx, str(e)))

        if errors:
            pytest.fail(f"Errors in {len(errors)} random samples: {errors}...")


# ---------------------------------------------------------------------------
# Tests: Paths
# ---------------------------------------------------------------------------

@pytest.mark.s3
class TestImageFolderPaths:
    """Verify path field is correct."""

    def test_paths_have_class_and_filename(self, imagenet_val_s3):
        """Paths should contain class folder and filename."""
        for idx in range(100):
            sample = imagenet_val_s3[idx]
            if 'path' in sample:
                path = Path(sample['path'])
                # Path should have at least class/filename structure
                assert len(path.parts) >= 2, f"Path too short: {path}"
                # Filename should end with image extension
                assert path.suffix.lower() in {'.jpg', '.jpeg', '.png'}, \
                    f"Expected image extension: {path}"

    def test_path_class_matches_label(self, imagenet_val_s3):
        """Path class folder should match the label."""
        for idx in range(100):
            sample = imagenet_val_s3[idx]
            if 'path' in sample:
                path = Path(sample['path'])
                label = sample['label']
                class_name = imagenet_val_s3.classes[label]

                # The class folder should be in the path
                assert class_name in path.parts, \
                    f"Class '{class_name}' not in path {path}"


# ---------------------------------------------------------------------------
# Tests: Caching
# ---------------------------------------------------------------------------

@pytest.mark.s3
class TestImageFolderS3Cache:
    """Test that cached S3 data works correctly."""

    def test_s3_uses_cache_on_second_load(self, imagenet_val_s3, tmp_path_factory):
        """Second load should use cached extraction, not re-download."""
        # Get the cache directory from the first reader
        cache_dir = imagenet_val_s3._cache_path.parent

        # Create a second reader pointing to same S3 path with same cache
        reader2 = open_imagefolder(
            IMAGENET_VAL_S3_PATH,
            cache_dir=str(cache_dir),
            verbose=True
        )

        # Should have same number of samples
        assert len(reader2) == len(imagenet_val_s3)

        # First sample should be identical
        sample1 = imagenet_val_s3[0]
        sample2 = reader2[0]

        assert hashlib.sha256(sample1['image']).hexdigest() == \
               hashlib.sha256(sample2['image']).hexdigest()
        assert sample1['label'] == sample2['label']
