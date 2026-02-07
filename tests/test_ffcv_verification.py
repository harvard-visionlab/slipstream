"""FFCV reader byte-for-byte verification against native ffcv-ssl.

These tests verify that FFCVFileReader returns identical bytes to the native
ffcv-ssl Reader implementation. This is the strongest possible test for reader
correctness - if SHA256 hashes match for all samples, the reader is correct.

These tests require ffcv-ssl and run in the devcontainer (Linux only).
Skip on systems without ffcv installed.

Run with: pytest tests/test_ffcv_verification.py -v
"""

import hashlib
import io
import os
from pathlib import Path

import numpy as np
import PIL.Image
import pytest

from slipstream.readers.ffcv import FFCVFileReader

# Check if ffcv-ssl is available
try:
    from ffcv.reader import Reader as FFCVNativeReader
    FFCV_AVAILABLE = True
except ImportError:
    FFCV_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not FFCV_AVAILABLE,
    reason="ffcv-ssl not installed (run in devcontainer with uv sync --group ffcv-test)"
)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ffcv_test_path():
    """Return path to a real FFCV file for testing.

    Set FFCV_TEST_PATH environment variable to point to a .ffcv file,
    or skip if not available.
    """
    path = os.environ.get("FFCV_TEST_PATH")
    if path is None:
        pytest.skip("FFCV_TEST_PATH not set - set to path of .ffcv file to test")
    path = Path(path)
    if not path.exists():
        pytest.skip(f"FFCV_TEST_PATH={path} does not exist")
    return path


# ---------------------------------------------------------------------------
# Helper to create synthetic FFCV for local testing
# ---------------------------------------------------------------------------

def _create_test_jpeg(width: int = 32, height: int = 32, color: tuple = (255, 0, 0)) -> bytes:
    """Create a JPEG image as bytes using PIL."""
    img = PIL.Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _build_synthetic_ffcv_for_native(
    path: Path,
    images: list[tuple[bytes, int, int]],
    labels: list[int],
) -> None:
    """Build a synthetic FFCV file using the native ffcv-ssl writer.

    This ensures we're testing against files created by the real FFCV,
    not our synthetic builder (which could have the same bugs).
    """
    from ffcv.writer import DatasetWriter
    from ffcv.fields import RGBImageField, IntField

    class SyntheticDataset:
        def __init__(self, images, labels):
            self.images = images
            self.labels = labels

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            jpeg_bytes, w, h = self.images[idx]
            img = PIL.Image.open(io.BytesIO(jpeg_bytes))
            return np.array(img), self.labels[idx]

    dataset = SyntheticDataset(images, labels)
    writer = DatasetWriter(
        str(path),
        {
            'image': RGBImageField(write_mode='jpg'),
            'label': IntField(),
        },
        num_workers=1,
    )
    writer.from_indexed_dataset(dataset)


@pytest.fixture
def native_written_ffcv(tmp_path):
    """Create a synthetic FFCV file using native ffcv-ssl writer.

    This is the gold standard - if our reader matches bytes from a file
    written by native ffcv-ssl, we know the reader is correct.
    """
    if not FFCV_AVAILABLE:
        pytest.skip("ffcv-ssl required to create test file")

    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 128, 128), (64, 192, 32),
    ]
    images = []
    for color in colors:
        jpeg = _create_test_jpeg(width=64, height=48, color=color)
        images.append((jpeg, 64, 48))

    labels = list(range(8))
    ffcv_path = tmp_path / "native_written.ffcv"
    _build_synthetic_ffcv_for_native(ffcv_path, images, labels)

    return ffcv_path, images, labels


# ---------------------------------------------------------------------------
# Tests: Bytes match native reader
# ---------------------------------------------------------------------------

class TestFFCVBytesMatchNative:
    """Verify FFCVFileReader returns identical bytes to native ffcv-ssl."""

    def test_all_samples_bytes_match(self, native_written_ffcv):
        """Hash comparison of all samples from native-written FFCV file."""
        ffcv_path, original_images, labels = native_written_ffcv

        slip_reader = FFCVFileReader(str(ffcv_path), verbose=False)
        native_reader = FFCVNativeReader(str(ffcv_path))

        assert len(slip_reader) == len(native_reader), \
            f"Length mismatch: slipstream={len(slip_reader)}, native={len(native_reader)}"

        mismatches = []
        for idx in range(len(slip_reader)):
            slip_sample = slip_reader[idx]
            native_sample = native_reader[idx]

            # Native reader returns tuple, first element is image
            # For JPEG mode, native returns the raw JPEG bytes
            slip_bytes = slip_sample['image']
            native_bytes = native_sample[0]

            # If native returns numpy array, it's decoded - skip byte comparison
            if isinstance(native_bytes, np.ndarray):
                # Native decoded the image - compare decoded pixels instead
                slip_img = np.array(PIL.Image.open(io.BytesIO(slip_bytes)))
                diff = np.abs(slip_img.astype(int) - native_bytes.astype(int))
                if np.max(diff) > 2:  # Allow ±2 for JPEG decode variance
                    mismatches.append((idx, f"pixel diff {np.max(diff)}"))
            else:
                # Native returned raw bytes - compare hashes
                slip_hash = hashlib.sha256(slip_bytes).hexdigest()
                native_hash = hashlib.sha256(native_bytes).hexdigest()

                if slip_hash != native_hash:
                    mismatches.append((idx, "hash mismatch"))

        if mismatches:
            sample_errors = mismatches[:5]
            pytest.fail(f"Bytes mismatch at {len(mismatches)} indices: {sample_errors}...")

    def test_metadata_matches(self, native_written_ffcv):
        """Verify field count and sample count match."""
        ffcv_path, _, labels = native_written_ffcv

        slip_reader = FFCVFileReader(str(ffcv_path), verbose=False)
        native_reader = FFCVNativeReader(str(ffcv_path))

        assert len(slip_reader) == len(native_reader)

        # Verify labels match
        for idx in range(len(labels)):
            slip_label = slip_reader[idx]['label']
            native_label = native_reader[idx][1]  # Second field is label
            assert slip_label == native_label, f"Label mismatch at {idx}: {slip_label} vs {native_label}"

    def test_decoded_images_match(self, native_written_ffcv):
        """Verify decoded images match within JPEG tolerance."""
        ffcv_path, _, _ = native_written_ffcv

        slip_reader = FFCVFileReader(str(ffcv_path), verbose=False)
        native_reader = FFCVNativeReader(str(ffcv_path))

        for idx in range(len(slip_reader)):
            slip_bytes = slip_reader[idx]['image']
            native_sample = native_reader[idx][0]

            # Decode slipstream bytes
            slip_img = np.array(PIL.Image.open(io.BytesIO(slip_bytes)))

            # Native may return decoded array or bytes
            if isinstance(native_sample, np.ndarray):
                native_img = native_sample
            else:
                native_img = np.array(PIL.Image.open(io.BytesIO(native_sample)))

            # Compare decoded images (allow JPEG decode variance)
            diff = np.abs(slip_img.astype(int) - native_img.astype(int))
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)

            assert max_diff <= 2, f"Sample {idx}: max pixel diff {max_diff} exceeds tolerance (2)"
            assert mean_diff < 0.5, f"Sample {idx}: mean pixel diff {mean_diff} exceeds tolerance (0.5)"


class TestFFCVBytesMatchNativeReal:
    """Test against real FFCV files (requires FFCV_TEST_PATH env var)."""

    @pytest.mark.slow
    def test_real_ffcv_bytes_match(self, ffcv_test_path):
        """Verify bytes match for first 100 samples of a real FFCV file."""
        slip_reader = FFCVFileReader(str(ffcv_test_path), verbose=False)
        native_reader = FFCVNativeReader(str(ffcv_test_path))

        assert len(slip_reader) == len(native_reader)

        # Test first 100 samples (or all if fewer)
        num_samples = min(100, len(slip_reader))
        mismatches = []

        for idx in range(num_samples):
            slip_sample = slip_reader[idx]
            native_sample = native_reader[idx]

            slip_bytes = slip_sample['image']
            native_bytes = native_sample[0]

            if isinstance(native_bytes, np.ndarray):
                # Compare decoded pixels
                slip_img = np.array(PIL.Image.open(io.BytesIO(slip_bytes)))
                diff = np.abs(slip_img.astype(int) - native_bytes.astype(int))
                if np.max(diff) > 2:
                    mismatches.append((idx, f"pixel diff {np.max(diff)}"))
            else:
                # Compare raw bytes
                slip_hash = hashlib.sha256(slip_bytes).hexdigest()
                native_hash = hashlib.sha256(native_bytes).hexdigest()
                if slip_hash != native_hash:
                    mismatches.append((idx, "hash mismatch"))

        if mismatches:
            pytest.fail(f"Bytes mismatch at {len(mismatches)}/{num_samples} samples: {mismatches[:5]}...")

    @pytest.mark.slow
    def test_real_ffcv_all_samples(self, ffcv_test_path):
        """Verify all samples can be read without error."""
        slip_reader = FFCVFileReader(str(ffcv_test_path), verbose=False)

        errors = []
        for idx in range(len(slip_reader)):
            try:
                sample = slip_reader[idx]
                # Verify it's valid JPEG
                assert sample['image'][:2] == b'\xff\xd8', f"Sample {idx}: not a JPEG"
                assert sample['image'][-2:] == b'\xff\xd9', f"Sample {idx}: missing JPEG EOI"
            except Exception as e:
                errors.append((idx, str(e)))

        if errors:
            pytest.fail(f"Errors reading {len(errors)}/{len(slip_reader)} samples: {errors[:5]}...")
