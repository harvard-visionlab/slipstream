"""Tests for resilient cache loading (auto-rebuild on partial deletion).

Verifies that:
- check_integrity() detects missing and size-mismatched files
- _wipe_cache() removes the .slipstream subdirectory
- SlipstreamLoader auto-rebuilds when cache files are deleted
- _check_extraction_integrity() detects incomplete tar extractions

All tests use small synthetic ImageFolder datasets via tmp_path (no S3).
"""

import io
import json
import os
import shutil
import warnings
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from slipstream.cache import (
    CACHE_SUBDIR,
    MANIFEST_FILE,
    OptimizedCache,
    _get_expected_files,
)
from slipstream.readers.imagefolder import (
    SlipstreamImageFolder,
    _check_extraction_integrity,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_jpeg_bytes(width: int = 8, height: int = 8, color: tuple = (255, 0, 0)) -> bytes:
    """Create a valid JPEG image as bytes."""
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def _make_imagefolder(root: Path, num_classes: int = 2, images_per_class: int = 3) -> Path:
    """Create a minimal ImageFolder structure with real JPEG images."""
    for cls_idx in range(num_classes):
        cls_dir = root / f"class_{cls_idx}"
        cls_dir.mkdir(parents=True, exist_ok=True)
        for img_idx in range(images_per_class):
            jpeg_bytes = _create_jpeg_bytes(
                width=16 + img_idx * 2,
                height=16 + img_idx * 2,
                color=(cls_idx * 100, img_idx * 50, 128),
            )
            (cls_dir / f"img_{img_idx}.jpg").write_bytes(jpeg_bytes)
    return root


def _build_cache(reader: SlipstreamImageFolder) -> OptimizedCache:
    """Build a cache from a reader, return the loaded cache."""
    return OptimizedCache.build(reader, verbose=False)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def imagefolder_root(tmp_path):
    """Create a temporary ImageFolder and return its root."""
    return _make_imagefolder(tmp_path / "dataset")


@pytest.fixture
def reader(imagefolder_root, tmp_path):
    """Create a SlipstreamImageFolder reader with a cache dir."""
    return SlipstreamImageFolder(
        imagefolder_root,
        cache_dir=tmp_path / "cache",
    )


@pytest.fixture
def built_cache(reader):
    """Build and return a valid OptimizedCache."""
    return _build_cache(reader)


# ---------------------------------------------------------------------------
# Tests: _get_expected_files
# ---------------------------------------------------------------------------

class TestGetExpectedFiles:
    def test_image_bytes(self):
        files = _get_expected_files("image", "ImageBytes")
        assert files == ["image.bin", "image.meta.npy"]

    def test_numpy(self):
        files = _get_expected_files("label", "int")
        assert files == ["label.npy"]

    def test_string(self):
        files = _get_expected_files("path", "str")
        assert files == ["path.bin", "path.offsets.npy"]

    def test_bytes(self):
        files = _get_expected_files("data", "bytes")
        assert files == ["data.bin", "data.meta.npy"]


# ---------------------------------------------------------------------------
# Tests: check_integrity
# ---------------------------------------------------------------------------

class TestCheckIntegrity:
    def test_valid_cache(self, reader, built_cache):
        """check_integrity returns True for a freshly-built cache."""
        cache_dir = reader.cache_path
        is_valid, problems = OptimizedCache.check_integrity(cache_dir)
        assert is_valid is True
        assert problems == []

    def test_missing_bin_file(self, reader, built_cache):
        """check_integrity detects a missing .bin file."""
        cache_dir = reader.cache_path
        slipstream_dir = cache_dir / CACHE_SUBDIR

        # Delete the image.bin file
        bin_file = slipstream_dir / "image.bin"
        assert bin_file.exists()
        bin_file.unlink()

        is_valid, problems = OptimizedCache.check_integrity(cache_dir)
        assert is_valid is False
        assert any("image.bin" in p for p in problems)

    def test_missing_npy_file(self, reader, built_cache):
        """check_integrity detects a missing .npy file."""
        cache_dir = reader.cache_path
        slipstream_dir = cache_dir / CACHE_SUBDIR

        # Delete the label.npy file
        npy_file = slipstream_dir / "label.npy"
        assert npy_file.exists()
        npy_file.unlink()

        is_valid, problems = OptimizedCache.check_integrity(cache_dir)
        assert is_valid is False
        assert any("label.npy" in p for p in problems)

    def test_file_size_mismatch(self, reader, built_cache):
        """check_integrity detects file size mismatches."""
        cache_dir = reader.cache_path
        slipstream_dir = cache_dir / CACHE_SUBDIR

        # Corrupt the image.bin file (truncate it)
        bin_file = slipstream_dir / "image.bin"
        original_size = bin_file.stat().st_size
        assert original_size > 10
        bin_file.write_bytes(b"corrupted")

        is_valid, problems = OptimizedCache.check_integrity(cache_dir)
        assert is_valid is False
        assert any("size mismatch" in p and "image.bin" in p for p in problems)

    def test_missing_manifest(self, reader, built_cache):
        """check_integrity returns False when manifest is missing."""
        cache_dir = reader.cache_path
        slipstream_dir = cache_dir / CACHE_SUBDIR

        manifest = slipstream_dir / MANIFEST_FILE
        manifest.unlink()

        is_valid, problems = OptimizedCache.check_integrity(cache_dir)
        assert is_valid is False
        assert any("manifest" in p for p in problems)

    def test_corrupt_manifest(self, reader, built_cache):
        """check_integrity handles a corrupt manifest gracefully."""
        cache_dir = reader.cache_path
        slipstream_dir = cache_dir / CACHE_SUBDIR

        manifest = slipstream_dir / MANIFEST_FILE
        manifest.write_text("this is not json {{{")

        is_valid, problems = OptimizedCache.check_integrity(cache_dir)
        assert is_valid is False
        assert any("corrupt" in p for p in problems)

    def test_no_cache_dir(self, tmp_path):
        """check_integrity returns False when no cache exists."""
        is_valid, problems = OptimizedCache.check_integrity(tmp_path / "nonexistent")
        assert is_valid is False

    def test_old_manifest_without_file_sizes(self, reader, built_cache):
        """check_integrity handles old manifests without file_sizes."""
        cache_dir = reader.cache_path
        slipstream_dir = cache_dir / CACHE_SUBDIR

        # Remove file_sizes from manifest (simulating old format)
        manifest_path = slipstream_dir / MANIFEST_FILE
        with open(manifest_path) as f:
            manifest = json.load(f)
        manifest.pop("file_sizes", None)
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)

        # Should still pass (file existence check only)
        is_valid, problems = OptimizedCache.check_integrity(cache_dir)
        assert is_valid is True
        assert problems == []


# ---------------------------------------------------------------------------
# Tests: _wipe_cache
# ---------------------------------------------------------------------------

class TestWipeCache:
    def test_removes_slipstream_dir(self, reader, built_cache):
        """_wipe_cache removes the .slipstream subdirectory."""
        cache_dir = reader.cache_path
        slipstream_dir = cache_dir / CACHE_SUBDIR
        assert slipstream_dir.exists()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            OptimizedCache._wipe_cache(cache_dir, "test reason")

        assert not slipstream_dir.exists()
        # Parent directory should still exist
        assert cache_dir.exists()

    def test_emits_warning(self, reader, built_cache):
        """_wipe_cache emits a warning with the reason."""
        cache_dir = reader.cache_path

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            OptimizedCache._wipe_cache(cache_dir, "GC deleted files")

        assert len(w) == 1
        assert "GC deleted files" in str(w[0].message)

    def test_idempotent(self, reader, built_cache):
        """_wipe_cache does not fail if called twice."""
        cache_dir = reader.cache_path

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            OptimizedCache._wipe_cache(cache_dir, "first")
            OptimizedCache._wipe_cache(cache_dir, "second")  # Should not raise


# ---------------------------------------------------------------------------
# Tests: build() stores file_sizes in manifest
# ---------------------------------------------------------------------------

class TestBuildFilesSizes:
    def test_manifest_contains_file_sizes(self, reader, built_cache):
        """build() should store file_sizes in the manifest."""
        cache_dir = reader.cache_path
        manifest_path = cache_dir / CACHE_SUBDIR / MANIFEST_FILE

        with open(manifest_path) as f:
            manifest = json.load(f)

        assert "file_sizes" in manifest
        file_sizes = manifest["file_sizes"]
        assert isinstance(file_sizes, dict)
        assert len(file_sizes) > 0

        # Verify recorded sizes match actual files
        slipstream_dir = cache_dir / CACHE_SUBDIR
        for fname, expected_size in file_sizes.items():
            fpath = slipstream_dir / fname
            assert fpath.exists(), f"File {fname} recorded in file_sizes but missing"
            assert os.path.getsize(fpath) == expected_size


# ---------------------------------------------------------------------------
# Tests: Loader auto-rebuild (end-to-end)
# ---------------------------------------------------------------------------

class TestLoaderAutoRebuild:
    def test_auto_rebuild_on_deleted_bin(self, imagefolder_root, tmp_path):
        """SlipstreamLoader auto-rebuilds when a cache .bin file is deleted."""
        from slipstream.loader import SlipstreamLoader

        cache_dir = tmp_path / "cache"
        reader = SlipstreamImageFolder(imagefolder_root, cache_dir=cache_dir)

        # First: build cache normally
        loader1 = SlipstreamLoader(reader, batch_size=2, shuffle=False, verbose=False)
        cache_dir_used = reader.cache_path
        slipstream_dir = cache_dir_used / CACHE_SUBDIR

        # Verify cache exists and is valid
        assert slipstream_dir.exists()
        is_valid, _ = OptimizedCache.check_integrity(cache_dir_used)
        assert is_valid

        # Delete a .bin file to simulate GC
        bin_file = slipstream_dir / "image.bin"
        assert bin_file.exists()
        bin_file.unlink()

        # Creating a new loader should auto-detect corruption and rebuild
        loader2 = SlipstreamLoader(reader, batch_size=2, shuffle=False, verbose=False)

        # Cache should be valid again
        is_valid, problems = OptimizedCache.check_integrity(cache_dir_used)
        assert is_valid, f"Cache not valid after rebuild: {problems}"

        # And the loader should work
        batch = next(iter(loader2))
        assert "image" in batch or "_indices" in batch

        loader1.shutdown()
        loader2.shutdown()

    def test_auto_rebuild_on_deleted_npy(self, imagefolder_root, tmp_path):
        """SlipstreamLoader auto-rebuilds when a cache .npy file is deleted."""
        from slipstream.loader import SlipstreamLoader

        cache_dir = tmp_path / "cache"
        reader = SlipstreamImageFolder(imagefolder_root, cache_dir=cache_dir)

        # Build cache
        loader1 = SlipstreamLoader(reader, batch_size=2, shuffle=False, verbose=False)
        cache_dir_used = reader.cache_path
        slipstream_dir = cache_dir_used / CACHE_SUBDIR

        # Delete label.npy
        npy_file = slipstream_dir / "label.npy"
        assert npy_file.exists()
        npy_file.unlink()

        # Should auto-rebuild
        loader2 = SlipstreamLoader(reader, batch_size=2, shuffle=False, verbose=False)

        is_valid, problems = OptimizedCache.check_integrity(cache_dir_used)
        assert is_valid, f"Cache not valid after rebuild: {problems}"

        loader1.shutdown()
        loader2.shutdown()


# ---------------------------------------------------------------------------
# Tests: _check_extraction_integrity
# ---------------------------------------------------------------------------

class TestExtractionIntegrity:
    def test_complete_extraction(self, tmp_path):
        """Returns True for a directory with a matching marker."""
        extract_dir = tmp_path / "extracted"
        extract_dir.mkdir()

        # Create some "image" files
        (extract_dir / "img1.jpg").write_bytes(b"fake")
        (extract_dir / "img2.jpg").write_bytes(b"fake")
        subdir = extract_dir / "subdir"
        subdir.mkdir()
        (subdir / "img3.jpg").write_bytes(b"fake")

        # Write marker with correct count (3 files)
        (extract_dir / ".extraction_complete").write_text("3\n")

        assert _check_extraction_integrity(extract_dir) is True

    def test_missing_marker(self, tmp_path):
        """Returns False when the marker file is missing."""
        extract_dir = tmp_path / "extracted"
        extract_dir.mkdir()
        (extract_dir / "img1.jpg").write_bytes(b"fake")

        assert _check_extraction_integrity(extract_dir) is False

    def test_files_deleted_count_mismatch(self, tmp_path):
        """Returns False when files were deleted (count mismatch)."""
        extract_dir = tmp_path / "extracted"
        extract_dir.mkdir()

        # Create files and marker
        (extract_dir / "img1.jpg").write_bytes(b"fake")
        (extract_dir / "img2.jpg").write_bytes(b"fake")
        (extract_dir / ".extraction_complete").write_text("2\n")

        # Verify it passes first
        assert _check_extraction_integrity(extract_dir) is True

        # Delete a file to simulate GC
        (extract_dir / "img2.jpg").unlink()

        assert _check_extraction_integrity(extract_dir) is False

    def test_corrupt_marker(self, tmp_path):
        """Returns False when the marker file contains invalid data."""
        extract_dir = tmp_path / "extracted"
        extract_dir.mkdir()
        (extract_dir / "img1.jpg").write_bytes(b"fake")
        (extract_dir / ".extraction_complete").write_text("not_a_number\n")

        assert _check_extraction_integrity(extract_dir) is False

    def test_empty_directory(self, tmp_path):
        """Returns False for an empty directory (no marker)."""
        extract_dir = tmp_path / "extracted"
        extract_dir.mkdir()

        assert _check_extraction_integrity(extract_dir) is False


# ---------------------------------------------------------------------------
# Tests: __getitem__ error message
# ---------------------------------------------------------------------------

class TestGetItemErrorMessage:
    def test_missing_file_gives_helpful_error(self, imagefolder_root):
        """__getitem__ gives a helpful error when an image file is missing."""
        reader = SlipstreamImageFolder(imagefolder_root)

        # Delete one image
        first_path = Path(reader.samples[0][0])
        first_path.unlink()

        with pytest.raises(FileNotFoundError, match="garbage collection"):
            reader[0]
