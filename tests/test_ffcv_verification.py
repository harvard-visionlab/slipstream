"""FFCV reader byte-for-byte verification against native ffcv-ssl.

These tests verify that FFCVFileReader returns identical bytes to the native
ffcv-ssl Reader implementation. This is the strongest possible test for reader
correctness - if SHA256 hashes match for all samples, the reader is correct.

These tests require:
1. ffcv-ssl installed (run in devcontainer - Linux only)
2. Access to S3 bucket with real FFCV files

Run with: pytest tests/test_ffcv_verification.py -v
"""

import hashlib
import io
import os

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
    reason="ffcv-ssl not installed (run in devcontainer)"
)

# Real FFCV file from S3 (ImageNet validation set)
# This is the same file used in notebooks/13_ffcv_datasets.ipynb
FFCV_VAL_S3_PATH = (
    "s3://visionlab-datasets/imagenet1k/pre-processed/"
    "s256-l512-jpgbytes-q100-ffcv/imagenet1k-s256-l512-jpg-q100-cs100-val-7ac6386e.ffcv"
)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ffcv_val_path(tmp_path_factory):
    """Download and cache the real ImageNet-val FFCV file.

    Uses FFCVFileReader's built-in S3 download and caching mechanism.
    Scoped to module to avoid re-downloading for each test.
    """
    # Check if AWS credentials are available
    import boto3
    try:
        boto3.client('s3').head_bucket(Bucket='visionlab-datasets')
    except Exception as e:
        pytest.skip(f"S3 access not available: {e}")

    # FFCVFileReader handles S3 download and caching automatically
    # It downloads to ~/.cache/slipstream/ by default
    cache_dir = tmp_path_factory.mktemp("ffcv_cache")

    try:
        # This will download if not already cached
        reader = FFCVFileReader(FFCV_VAL_S3_PATH, cache_dir=str(cache_dir), verbose=True)
        local_path = reader._path  # Local path after S3 download
        return str(local_path)
    except Exception as e:
        pytest.skip(f"Failed to download FFCV file: {e}")


# ---------------------------------------------------------------------------
# Tests: Bytes match native reader
# ---------------------------------------------------------------------------

class TestFFCVBytesMatchNative:
    """Verify FFCVFileReader returns identical bytes to native ffcv-ssl."""

    def test_sample_count_matches(self, ffcv_val_path):
        """Verify sample counts match between readers."""
        slip_reader = FFCVFileReader(ffcv_val_path, verbose=False)
        native_reader = FFCVNativeReader(str(ffcv_val_path))

        assert len(slip_reader) == len(native_reader), \
            f"Length mismatch: slipstream={len(slip_reader)}, native={len(native_reader)}"

    def test_first_100_samples_bytes_match(self, ffcv_val_path):
        """Hash comparison of first 100 samples."""
        slip_reader = FFCVFileReader(ffcv_val_path, verbose=False)
        native_reader = FFCVNativeReader(str(ffcv_val_path))

        num_samples = min(100, len(slip_reader))
        mismatches = []

        for idx in range(num_samples):
            slip_sample = slip_reader[idx]
            native_sample = native_reader[idx]

            # Get image bytes from both readers
            slip_bytes = slip_sample['image']
            native_bytes = native_sample[0]

            # ffcv-ssl may return decoded numpy array or raw bytes depending on field type
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
                    mismatches.append((idx, f"hash mismatch: {len(slip_bytes)} vs {len(native_bytes)} bytes"))

        if mismatches:
            sample_errors = mismatches[:5]
            pytest.fail(f"Bytes mismatch at {len(mismatches)}/{num_samples} indices: {sample_errors}...")

    def test_labels_match(self, ffcv_val_path):
        """Verify labels match for first 100 samples."""
        slip_reader = FFCVFileReader(ffcv_val_path, verbose=False)
        native_reader = FFCVNativeReader(str(ffcv_val_path))

        num_samples = min(100, len(slip_reader))
        mismatches = []

        for idx in range(num_samples):
            slip_label = slip_reader[idx]['label']
            native_label = native_reader[idx][1]  # Second field is label

            if slip_label != native_label:
                mismatches.append((idx, slip_label, native_label))

        if mismatches:
            pytest.fail(f"Label mismatch at {len(mismatches)}/{num_samples} samples: {mismatches[:5]}...")

    def test_decoded_images_match(self, ffcv_val_path):
        """Verify decoded images match within JPEG tolerance."""
        slip_reader = FFCVFileReader(ffcv_val_path, verbose=False)
        native_reader = FFCVNativeReader(str(ffcv_val_path))

        num_samples = min(50, len(slip_reader))

        for idx in range(num_samples):
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

            assert max_diff <= 5, f"Sample {idx}: max pixel diff {max_diff} exceeds tolerance (5)"
            assert mean_diff < 1.5, f"Sample {idx}: mean pixel diff {mean_diff} exceeds tolerance (1.5)"

    def test_jpeg_markers_valid(self, ffcv_val_path):
        """Verify all images have valid JPEG SOI/EOI markers."""
        slip_reader = FFCVFileReader(ffcv_val_path, verbose=False)

        num_samples = min(100, len(slip_reader))
        errors = []

        for idx in range(num_samples):
            try:
                sample = slip_reader[idx]
                img_bytes = sample['image']

                # Check JPEG markers
                if img_bytes[:2] != b'\xff\xd8':
                    errors.append((idx, "missing SOI marker"))
                elif img_bytes[-2:] != b'\xff\xd9':
                    errors.append((idx, "missing EOI marker"))
            except Exception as e:
                errors.append((idx, str(e)))

        if errors:
            pytest.fail(f"JPEG marker errors at {len(errors)}/{num_samples} samples: {errors[:5]}...")


@pytest.mark.slow
class TestFFCVFullValidation:
    """Full validation tests (all samples) - marked slow."""

    def test_all_samples_readable(self, ffcv_val_path):
        """Verify all samples can be read without error."""
        slip_reader = FFCVFileReader(ffcv_val_path, verbose=False)

        errors = []
        for idx in range(len(slip_reader)):
            try:
                sample = slip_reader[idx]
                # Verify it's valid JPEG
                img_bytes = sample['image']
                assert img_bytes[:2] == b'\xff\xd8', f"Sample {idx}: not a JPEG"
                assert img_bytes[-2:] == b'\xff\xd9', f"Sample {idx}: missing JPEG EOI"
                # Verify it can be decoded
                img = PIL.Image.open(io.BytesIO(img_bytes))
                img.load()  # Force decode
            except Exception as e:
                errors.append((idx, str(e)))
                if len(errors) >= 10:
                    break  # Stop early if too many errors

        if errors:
            pytest.fail(f"Errors reading {len(errors)}/{len(slip_reader)} samples: {errors}...")

    def test_all_samples_bytes_match(self, ffcv_val_path):
        """Hash comparison of ALL samples (slow - runs on full dataset)."""
        slip_reader = FFCVFileReader(ffcv_val_path, verbose=False)
        native_reader = FFCVNativeReader(str(ffcv_val_path))

        mismatches = []

        for idx in range(len(slip_reader)):
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
            pytest.fail(
                f"Bytes mismatch at {len(mismatches)}/{len(slip_reader)} samples: "
                f"{mismatches[:10]}..."
            )
