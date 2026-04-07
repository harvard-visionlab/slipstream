"""Tests for SlipstreamLoader.warmup_cache().

Verifies that:
1. warmup_cache() returns expected stats dict
2. Loader output is identical with and without warmup
3. warmup_cache() does not modify loader state
"""

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from slipstream.decoders import DecodeCenterCrop
from slipstream.loader import SlipstreamLoader
from slipstream.readers.imagefolder import SlipstreamImageFolder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_jpeg_bytes(width: int = 16, height: int = 16, color: tuple = (255, 0, 0)) -> bytes:
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def _make_imagefolder(root: Path, num_classes: int = 2, images_per_class: int = 4) -> Path:
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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def loader(tmp_path):
    """Create a SlipstreamLoader with a small synthetic ImageFolder."""
    dataset_root = _make_imagefolder(tmp_path / "dataset")
    reader = SlipstreamImageFolder(dataset_root, cache_dir=tmp_path / "cache")
    ldr = SlipstreamLoader(
        reader,
        batch_size=2,
        shuffle=False,
        pipelines={"image": [DecodeCenterCrop(size=8)]},
        verbose=False,
    )
    yield ldr
    ldr.shutdown()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestWarmupCache:
    def test_returns_stats_dict(self, loader):
        """warmup_cache() returns dict with expected keys and plausible values."""
        stats = loader.warmup_cache(verbose=False)

        assert isinstance(stats, dict)
        for key in ("elapsed_sec", "total_bytes", "throughput_mb_s", "cache_dir", "num_files"):
            assert key in stats, f"Missing key: {key}"

        assert stats["num_files"] > 0
        assert stats["total_bytes"] > 0
        assert stats["elapsed_sec"] >= 0
        assert Path(stats["cache_dir"]).exists()

    def test_output_identical_with_and_without_warmup(self, tmp_path):
        """Loader produces the same batches whether or not warmup_cache() was called."""
        dataset_root = _make_imagefolder(tmp_path / "dataset")

        # --- Without warmup ---
        reader1 = SlipstreamImageFolder(dataset_root, cache_dir=tmp_path / "cache1")
        loader1 = SlipstreamLoader(
            reader1,
            batch_size=2,
            shuffle=False,
            pipelines={"image": [DecodeCenterCrop(size=8)]},
            verbose=False,
        )
        batches_no_warmup = list(loader1)

        # --- With warmup ---
        reader2 = SlipstreamImageFolder(dataset_root, cache_dir=tmp_path / "cache2")
        loader2 = SlipstreamLoader(
            reader2,
            batch_size=2,
            shuffle=False,
            pipelines={"image": [DecodeCenterCrop(size=8)]},
            verbose=False,
        )
        loader2.warmup_cache(verbose=False)
        batches_with_warmup = list(loader2)

        assert len(batches_no_warmup) == len(batches_with_warmup)
        for b1, b2 in zip(batches_no_warmup, batches_with_warmup):
            assert b1.keys() == b2.keys()
            for key in b1:
                v1, v2 = b1[key], b2[key]
                if hasattr(v1, "numpy"):
                    np.testing.assert_array_equal(v1.numpy(), v2.numpy())
                elif isinstance(v1, np.ndarray):
                    np.testing.assert_array_equal(v1, v2)
                else:
                    assert v1 == v2

        loader1.shutdown()
        loader2.shutdown()

    def test_does_not_modify_loader_state(self, loader):
        """warmup_cache() must not change epoch counter or other mutable state."""
        epoch_before = loader._epoch
        loader.warmup_cache(verbose=False)
        assert loader._epoch == epoch_before

    def test_warmup_then_iterate(self, loader):
        """After warmup, iteration completes without errors."""
        loader.warmup_cache(verbose=False)
        batches = list(loader)
        assert len(batches) > 0
        assert "image" in batches[0]
