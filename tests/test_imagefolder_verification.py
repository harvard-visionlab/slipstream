"""ImageFolder reader verification against torchvision.

These tests verify that SlipstreamImageFolder returns identical data to
torchvision.datasets.ImageFolder. No special setup required - uses
synthetic test images in a temp directory.

Run with: uv run pytest tests/test_imagefolder_verification.py -v
"""

import hashlib
import io
import tempfile
from pathlib import Path

import numpy as np
import PIL.Image
import pytest
import torchvision.datasets

from slipstream.readers.imagefolder import SlipstreamImageFolder


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _create_test_jpeg(width: int, height: int, color: tuple) -> bytes:
    """Create a JPEG image as bytes."""
    img = PIL.Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


@pytest.fixture
def imagefolder_dir(tmp_path):
    """Create a temporary ImageFolder structure with test images.

    Structure:
        tmp_path/
            class_0/
                img_0.jpg
                img_1.jpg
            class_1/
                img_2.jpg
                img_3.jpg
            class_2/
                img_4.jpg
    """
    colors = [
        (255, 0, 0),    # red
        (0, 255, 0),    # green
        (0, 0, 255),    # blue
        (255, 255, 0),  # yellow
        (128, 128, 128),  # gray
    ]

    img_idx = 0
    for class_idx in range(3):
        class_dir = tmp_path / f"class_{class_idx}"
        class_dir.mkdir()

        # 2 images in first two classes, 1 in last
        num_images = 2 if class_idx < 2 else 1
        for _ in range(num_images):
            img_bytes = _create_test_jpeg(64 + img_idx * 10, 48 + img_idx * 5, colors[img_idx])
            img_path = class_dir / f"img_{img_idx}.jpg"
            img_path.write_bytes(img_bytes)
            img_idx += 1

    return tmp_path


# ---------------------------------------------------------------------------
# Tests: Sample count and structure
# ---------------------------------------------------------------------------

class TestImageFolderStructure:
    """Verify basic structure matches torchvision."""

    def test_sample_count_matches(self, imagefolder_dir):
        """Sample count should match torchvision."""
        slip_reader = SlipstreamImageFolder(str(imagefolder_dir))
        tv_dataset = torchvision.datasets.ImageFolder(str(imagefolder_dir))

        assert len(slip_reader) == len(tv_dataset), \
            f"Length mismatch: slipstream={len(slip_reader)}, torchvision={len(tv_dataset)}"

    def test_class_count_matches(self, imagefolder_dir):
        """Number of classes should match."""
        slip_reader = SlipstreamImageFolder(str(imagefolder_dir))
        tv_dataset = torchvision.datasets.ImageFolder(str(imagefolder_dir))

        assert len(slip_reader.classes) == len(tv_dataset.classes)
        assert slip_reader.classes == tv_dataset.classes

    def test_class_to_idx_matches(self, imagefolder_dir):
        """Class to index mapping should match."""
        slip_reader = SlipstreamImageFolder(str(imagefolder_dir))
        tv_dataset = torchvision.datasets.ImageFolder(str(imagefolder_dir))

        assert slip_reader.class_to_idx == tv_dataset.class_to_idx


# ---------------------------------------------------------------------------
# Tests: Labels match
# ---------------------------------------------------------------------------

class TestImageFolderLabels:
    """Verify labels match torchvision."""

    def test_all_labels_match(self, imagefolder_dir):
        """All labels should match torchvision ordering."""
        slip_reader = SlipstreamImageFolder(str(imagefolder_dir))
        tv_dataset = torchvision.datasets.ImageFolder(str(imagefolder_dir))

        mismatches = []
        for idx in range(len(slip_reader)):
            slip_label = slip_reader[idx]['label']
            _, tv_label = tv_dataset[idx]

            if slip_label != tv_label:
                mismatches.append((idx, slip_label, tv_label))

        if mismatches:
            pytest.fail(f"Label mismatch at {len(mismatches)} indices: {mismatches}")

    def test_labels_correspond_to_class_folders(self, imagefolder_dir):
        """Labels should match the class folder structure."""
        slip_reader = SlipstreamImageFolder(str(imagefolder_dir))

        for idx in range(len(slip_reader)):
            sample = slip_reader[idx]
            label = sample['label']
            path = sample.get('path', '')

            # Label should be valid class index
            assert 0 <= label < len(slip_reader.classes), \
                f"Label {label} out of range for {len(slip_reader.classes)} classes"


# ---------------------------------------------------------------------------
# Tests: Image bytes match
# ---------------------------------------------------------------------------

class TestImageFolderBytes:
    """Verify image bytes match direct file reads."""

    def test_image_bytes_match_file(self, imagefolder_dir):
        """Image bytes should match direct file read."""
        slip_reader = SlipstreamImageFolder(str(imagefolder_dir))
        tv_dataset = torchvision.datasets.ImageFolder(str(imagefolder_dir))

        mismatches = []
        for idx in range(len(slip_reader)):
            slip_bytes = slip_reader[idx]['image']

            # Get file path from torchvision dataset
            file_path, _ = tv_dataset.samples[idx]
            file_bytes = Path(file_path).read_bytes()

            slip_hash = hashlib.sha256(slip_bytes).hexdigest()
            file_hash = hashlib.sha256(file_bytes).hexdigest()

            if slip_hash != file_hash:
                mismatches.append((idx, len(slip_bytes), len(file_bytes)))

        if mismatches:
            pytest.fail(f"Bytes mismatch at {len(mismatches)} indices: {mismatches}")

    def test_all_images_valid_jpeg(self, imagefolder_dir):
        """All images should have valid JPEG markers."""
        slip_reader = SlipstreamImageFolder(str(imagefolder_dir))

        errors = []
        for idx in range(len(slip_reader)):
            img_bytes = slip_reader[idx]['image']

            if img_bytes[:2] != b'\xff\xd8':
                errors.append((idx, "missing SOI"))
            elif img_bytes[-2:] != b'\xff\xd9':
                errors.append((idx, "missing EOI"))

        if errors:
            pytest.fail(f"JPEG marker errors: {errors}")


# ---------------------------------------------------------------------------
# Tests: Decoded images match
# ---------------------------------------------------------------------------

class TestImageFolderDecode:
    """Verify decoded images match torchvision."""

    def test_decoded_images_match(self, imagefolder_dir):
        """Decoded pixels should match torchvision (exact for same JPEG)."""
        slip_reader = SlipstreamImageFolder(str(imagefolder_dir))
        tv_dataset = torchvision.datasets.ImageFolder(str(imagefolder_dir))

        for idx in range(len(slip_reader)):
            slip_bytes = slip_reader[idx]['image']
            slip_img = np.array(PIL.Image.open(io.BytesIO(slip_bytes)))

            tv_img, _ = tv_dataset[idx]
            tv_img = np.array(tv_img)

            # Should be exactly equal since it's the same JPEG file
            assert np.array_equal(slip_img, tv_img), \
                f"Image {idx}: decoded pixels don't match"

    def test_image_dimensions_correct(self, imagefolder_dir):
        """Image dimensions should match the created test images."""
        slip_reader = SlipstreamImageFolder(str(imagefolder_dir))

        for idx in range(len(slip_reader)):
            img_bytes = slip_reader[idx]['image']
            img = PIL.Image.open(io.BytesIO(img_bytes))
            w, h = img.size

            # We created images with dimensions 64+idx*10 x 48+idx*5
            expected_w = 64 + idx * 10
            expected_h = 48 + idx * 5

            assert w == expected_w and h == expected_h, \
                f"Image {idx}: expected {expected_w}x{expected_h}, got {w}x{h}"


# ---------------------------------------------------------------------------
# Tests: Path field
# ---------------------------------------------------------------------------

class TestImageFolderPaths:
    """Verify path field is correct."""

    def test_paths_have_class_and_filename(self, imagefolder_dir):
        """Paths should contain class folder and filename."""
        slip_reader = SlipstreamImageFolder(str(imagefolder_dir))

        for idx in range(len(slip_reader)):
            sample = slip_reader[idx]
            if 'path' in sample:
                path = Path(sample['path'])
                # Path should have at least class/filename structure
                assert len(path.parts) >= 2, f"Path too short: {path}"
                # Filename should end with .jpg
                assert path.suffix == '.jpg', f"Expected .jpg suffix: {path}"

    def test_path_class_matches_label(self, imagefolder_dir):
        """Path class folder should match the label."""
        slip_reader = SlipstreamImageFolder(str(imagefolder_dir))

        for idx in range(len(slip_reader)):
            sample = slip_reader[idx]
            if 'path' in sample:
                path = Path(sample['path'])
                label = sample['label']
                class_name = slip_reader.classes[label]

                # The class folder should be in the path
                assert class_name in path.parts, \
                    f"Class '{class_name}' not in path {path}"
