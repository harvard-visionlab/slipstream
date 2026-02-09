"""FFCV reader byte-for-byte verification against native ffcv-ssl.

These tests verify that FFCVFileReader returns identical bytes to the native
ffcv-ssl implementation. This is the strongest possible test for reader
correctness - if SHA256 hashes match for all samples, the reader is correct.

These tests require:
1. ffcv-ssl installed (run in devcontainer - Linux only)
2. Access to S3 bucket with real FFCV files

Run with docker (copy-paste this command):

    docker run --rm \
      -v "$(pwd)":/workspace \
      -v ~/.aws:/root/.aws:ro \
      -v "$(pwd)/.devcontainer/cache":/root/.cache \
      -e SLIPSTREAM_CACHE_DIR=/root/.cache/slipstream \
      -w /workspace \
      slipstream-ffcv \
      bash -c "uv venv --clear && uv pip install -r .devcontainer/requirements-ffcv.txt && uv run python libslipstream/setup.py build_ext --inplace && uv run pytest tests/test_ffcv_verification.py -v"

First time setup (build the docker image):

    docker build -t slipstream-ffcv -f .devcontainer/Dockerfile .

The FFCV file (~4GB) is cached in .devcontainer/cache/ after first download.
"""

import hashlib
import io
import sys

import numpy as np
import PIL.Image
import pytest

from slipstream.readers.ffcv import FFCVFileReader


def _print_progress(msg: str) -> None:
    """Print to stderr so pytest doesn't capture it."""
    print(msg, file=sys.stderr, flush=True)

# Check if ffcv-ssl is available
try:
    from ffcv.loader import Loader, OrderOption
    from ffcv.fields import IntField, RGBImageField
    from ffcv.fields.decoders import IntDecoder

    # Import SimpleSampleReader from lrm-ssl if available, otherwise define it
    try:
        from lrm_ssl.datasets.dataloaders.custom_decoders import SimpleSampleReader, get_max_sample_size
    except ImportError:
        # Minimal implementation of SimpleSampleReader for testing
        from ffcv.fields.rgb_image import SimpleRGBImageDecoder
        from ffcv.pipeline.state import State
        from ffcv.pipeline.allocation_query import AllocationQuery
        from ffcv.pipeline.compiler import Compiler
        from dataclasses import replace
        from ffcv.reader import Reader

        class SimpleSampleReader(SimpleRGBImageDecoder):
            """Read raw JPEG bytes from FFCV file."""
            def __init__(self, max_size):
                super().__init__()
                self.max_size = max_size

            def declare_state_and_memory(self, previous_state: State):
                my_dtype = np.dtype('<u1')
                return (
                    replace(previous_state, jit_mode=True, shape=(self.max_size,), dtype=my_dtype),
                    AllocationQuery((self.max_size,), my_dtype),
                )

            def generate_code(self):
                mem_read = self.memory_read
                my_range = Compiler.get_iterator()
                def decode(batch_indices, destination, metadata, storage_state):
                    for dst_ix in my_range(len(batch_indices)):
                        source_ix = batch_indices[dst_ix]
                        field = metadata[source_ix]
                        image_data = mem_read(field['data_ptr'], storage_state)
                        data_size = image_data.shape[0]
                        destination[dst_ix, 0:data_size] = image_data
                    return destination
                decode.is_parallel = True
                return decode

        def get_max_sample_size(ffcv_path, custom_fields):
            reader = Reader(ffcv_path, custom_handlers=custom_fields)
            return int(max(reader.alloc_table['size']))

    FFCV_AVAILABLE = True
except ImportError:
    FFCV_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not FFCV_AVAILABLE,
    reason="ffcv-ssl not installed (run in devcontainer)"
)

# Custom field handlers for the ImageNet FFCV file
CUSTOM_HANDLERS = {'image': RGBImageField, 'label': IntField} if FFCV_AVAILABLE else {}

# Real FFCV file from S3 (ImageNet validation set)
FFCV_VAL_S3_PATH = (
    "s3://visionlab-datasets/imagenet1k/pre-processed/"
    "s256-l512-jpgbytes-q100-ffcv/imagenet1k-s256-l512-jpg-q100-cs100-val-7ac6386e.ffcv"
)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ffcv_val_path(tmp_path_factory):
    """Download and cache the real ImageNet-val FFCV file."""
    import boto3
    try:
        boto3.client('s3').head_bucket(Bucket='visionlab-datasets')
    except Exception as e:
        pytest.skip(f"S3 access not available: {e}")

    cache_dir = tmp_path_factory.mktemp("ffcv_cache")

    try:
        reader = FFCVFileReader(FFCV_VAL_S3_PATH, cache_dir=str(cache_dir), verbose=True)
        return str(reader._path)
    except Exception as e:
        pytest.skip(f"Failed to download FFCV file: {e}")


@pytest.fixture(scope="module")
def native_loader(ffcv_val_path):
    """Create an ffcv-ssl Loader that reads raw JPEG bytes."""
    from ffcv.fields.decoders import BytesDecoder

    max_size = get_max_sample_size(ffcv_val_path, CUSTOM_HANDLERS)

    # The FFCV file has 4 fields: image, label, index, path
    # We need to define a pipeline for each field
    loader = Loader(
        ffcv_val_path,
        batch_size=1,
        num_workers=1,
        order=OrderOption.SEQUENTIAL,
        pipelines={
            'image': [SimpleSampleReader(max_size)],
            'label': [IntDecoder()],
            'index': [IntDecoder()],
            'path': [BytesDecoder()],
        },
        custom_fields=CUSTOM_HANDLERS,
    )
    return loader


# ---------------------------------------------------------------------------
# Tests: Bytes match native reader
# ---------------------------------------------------------------------------

class TestFFCVBytesMatchNative:
    """Verify FFCVFileReader returns identical bytes to native ffcv-ssl."""

    def test_sample_count_matches(self, ffcv_val_path, native_loader):
        """Verify sample counts match between readers."""
        slip_reader = FFCVFileReader(ffcv_val_path, verbose=False)

        assert len(slip_reader) == native_loader.reader.num_samples, \
            f"Length mismatch: slipstream={len(slip_reader)}, native={native_loader.reader.num_samples}"

    def test_first_100_samples_bytes_match(self, ffcv_val_path, native_loader):
        """Hash comparison of first 100 samples."""
        slip_reader = FFCVFileReader(ffcv_val_path, verbose=False)

        num_samples = min(100, len(slip_reader))
        _print_progress(f"\n  Comparing bytes for {num_samples} samples...")
        mismatches = []

        # Read samples from native loader
        native_iter = iter(native_loader)

        for idx in range(num_samples):
            # Get slipstream sample
            slip_sample = slip_reader[idx]
            slip_bytes = slip_sample['image']

            # Get native sample (batch of 1)
            native_batch = next(native_iter)
            native_bytes_padded = native_batch[0][0]  # First sample, image field

            # Native returns padded array - trim to actual JPEG size
            # Find JPEG EOI marker (0xFFD9)
            native_bytes = _extract_jpeg_bytes(native_bytes_padded)

            # Compare hashes
            slip_hash = hashlib.sha256(slip_bytes).hexdigest()
            native_hash = hashlib.sha256(native_bytes).hexdigest()

            if slip_hash != native_hash:
                mismatches.append((idx, f"hash mismatch: slip={len(slip_bytes)}B, native={len(native_bytes)}B"))

        if mismatches:
            pytest.fail(f"Bytes mismatch at {len(mismatches)}/{num_samples} indices: {mismatches[:5]}...")

    def test_labels_match(self, ffcv_val_path, native_loader):
        """Verify labels match for first 100 samples."""
        slip_reader = FFCVFileReader(ffcv_val_path, verbose=False)

        num_samples = min(100, len(slip_reader))
        _print_progress(f"\n  Comparing labels for {num_samples} samples...")
        mismatches = []

        native_iter = iter(native_loader)

        for idx in range(num_samples):
            slip_label = slip_reader[idx]['label']
            native_batch = next(native_iter)
            native_label = int(native_batch[1][0].item())  # First sample, label field

            if slip_label != native_label:
                mismatches.append((idx, slip_label, native_label))

        if mismatches:
            pytest.fail(f"Label mismatch at {len(mismatches)}/{num_samples} samples: {mismatches[:5]}...")

    def test_decoded_images_match(self, ffcv_val_path, native_loader):
        """Verify decoded images match within JPEG tolerance."""
        slip_reader = FFCVFileReader(ffcv_val_path, verbose=False)

        num_samples = min(50, len(slip_reader))
        _print_progress(f"\n  Comparing decoded pixels for {num_samples} samples...")
        native_iter = iter(native_loader)

        for idx in range(num_samples):
            slip_bytes = slip_reader[idx]['image']
            native_batch = next(native_iter)
            native_bytes_padded = native_batch[0][0]
            native_bytes = _extract_jpeg_bytes(native_bytes_padded)

            # Decode both
            slip_img = np.array(PIL.Image.open(io.BytesIO(slip_bytes)))
            native_img = np.array(PIL.Image.open(io.BytesIO(native_bytes)))

            # Compare decoded images
            diff = np.abs(slip_img.astype(int) - native_img.astype(int))
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)

            assert max_diff <= 5, f"Sample {idx}: max pixel diff {max_diff} exceeds tolerance (5)"
            assert mean_diff < 1.5, f"Sample {idx}: mean pixel diff {mean_diff} exceeds tolerance (1.5)"

    def test_jpeg_markers_valid(self, ffcv_val_path):
        """Verify all images have valid JPEG SOI/EOI markers."""
        slip_reader = FFCVFileReader(ffcv_val_path, verbose=False)

        num_samples = min(100, len(slip_reader))
        _print_progress(f"\n  Validating JPEG markers for {num_samples} samples...")
        errors = []

        for idx in range(num_samples):
            try:
                sample = slip_reader[idx]
                img_bytes = sample['image']

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
        _print_progress(f"\n  Reading all {len(slip_reader)} samples...")

        errors = []
        for idx in range(len(slip_reader)):
            if idx % 10000 == 0:
                _print_progress(f"    {idx}/{len(slip_reader)}...")
            try:
                sample = slip_reader[idx]
                img_bytes = sample['image']
                assert img_bytes[:2] == b'\xff\xd8', f"Sample {idx}: not a JPEG"
                assert img_bytes[-2:] == b'\xff\xd9', f"Sample {idx}: missing JPEG EOI"
                # Verify it can be decoded
                img = PIL.Image.open(io.BytesIO(img_bytes))
                img.load()
            except Exception as e:
                errors.append((idx, str(e)))
                if len(errors) >= 10:
                    break

        if errors:
            pytest.fail(f"Errors reading {len(errors)}/{len(slip_reader)} samples: {errors}...")

    def test_all_samples_bytes_match(self, ffcv_val_path, native_loader):
        """Hash comparison of ALL samples (slow - runs on full dataset)."""
        slip_reader = FFCVFileReader(ffcv_val_path, verbose=False)
        _print_progress(f"\n  Comparing bytes for all {len(slip_reader)} samples...")

        # Reset loader to sequential order
        native_loader.traversal_order = native_loader.traversal_order.__class__(native_loader)
        native_iter = iter(native_loader)

        mismatches = []

        for idx in range(len(slip_reader)):
            if idx % 10000 == 0:
                _print_progress(f"    {idx}/{len(slip_reader)}...")
            slip_bytes = slip_reader[idx]['image']
            native_batch = next(native_iter)
            native_bytes_padded = native_batch[0][0]
            native_bytes = _extract_jpeg_bytes(native_bytes_padded)

            slip_hash = hashlib.sha256(slip_bytes).hexdigest()
            native_hash = hashlib.sha256(native_bytes).hexdigest()

            if slip_hash != native_hash:
                mismatches.append((idx, "hash mismatch"))

        if mismatches:
            pytest.fail(
                f"Bytes mismatch at {len(mismatches)}/{len(slip_reader)} samples: "
                f"{mismatches[:10]}..."
            )


def _extract_jpeg_bytes(data: np.ndarray) -> bytes:
    """Extract JPEG bytes from padded buffer by finding SOI and EOI markers.

    The SimpleSampleReader returns a fixed-size buffer with JPEG data at the start
    and garbage/zeros after the EOI marker. We need to find the actual JPEG bounds.
    """
    data_bytes = bytes(data)

    # Find JPEG SOI marker (0xFF 0xD8) - should be at start
    soi_pos = data_bytes.find(b'\xff\xd8')
    if soi_pos == -1:
        return data_bytes

    # Find JPEG EOI marker (0xFF 0xD9) starting from after SOI
    # Use find (not rfind) to get the first EOI after SOI, avoiding false matches in padding
    eoi_pos = data_bytes.find(b'\xff\xd9', soi_pos + 2)
    if eoi_pos == -1:
        # No EOI found - try rfind as fallback
        eoi_pos = data_bytes.rfind(b'\xff\xd9')
        if eoi_pos == -1:
            return data_bytes

    # Return from SOI to EOI inclusive
    return data_bytes[soi_pos:eoi_pos + 2]
