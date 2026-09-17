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


# ---------------------------------------------------------------------------
# Indices-aware warmup
# ---------------------------------------------------------------------------

from slipstream.loader import _coalesce_ranges  # noqa: E402


class _BigBytesDataset:
    """bytes records of 100 KB each (bigger than the coalesce gap) + int label."""

    REC = 100 * 1024

    def __init__(self, num_samples: int = 10, cache_path: Path | None = None):
        self.num_samples = num_samples
        self.cache_path = cache_path
        rng = np.random.default_rng(1)
        self._blobs = [
            rng.integers(0, 256, self.REC, dtype=np.uint8).tobytes()
            for _ in range(num_samples)
        ]

    @property
    def field_types(self) -> dict:
        return {"video": "bytes", "label": "int", "name": "str"}

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        return {"video": self._blobs[idx], "label": idx, "name": f"n{idx}"}


class TestCoalesceRanges:
    def test_sorts_merges_and_drops_empty(self):
        starts = np.array([100, 0, 50, 300_000, 40])
        ends = np.array([150, 50, 60, 300_010, 40])  # last is empty
        assert _coalesce_ranges(starts, ends, gap=0) == [(0, 60), (100, 150), (300_000, 300_010)]
        assert _coalesce_ranges(starts, ends, gap=64) == [(0, 150), (300_000, 300_010)]
        assert _coalesce_ranges(np.array([], dtype=np.int64), np.array([], dtype=np.int64)) == []


class TestSubsetWarmup:
    def test_defaults_to_loader_indices_and_touches_only_selected(self, tmp_path):
        ds = _BigBytesDataset(cache_path=tmp_path / "cache")
        subset = [0, 1, 5, 8]
        loader = SlipstreamLoader(
            ds, batch_size=2, shuffle=False, drop_last=False, indices=subset, verbose=False,
        )
        try:
            plan = dict(loader._warmup_plan(np.asarray(subset)))
            ranges = plan[Path(ds.cache_path) / "video.bin"]
            R = _BigBytesDataset.REC
            # records 0,1 contiguous -> one run; 5 and 8 isolated
            assert ranges == [(0, 2 * R), (5 * R, 6 * R), (8 * R, 9 * R)]
            assert plan[Path(ds.cache_path) / "label.npy"] is None  # whole tiny file
            assert Path(ds.cache_path, "name.bin") in plan

            stats = loader.warmup_cache(verbose=False)
            assert stats["subset"] is True
            assert stats["num_records"] == len(subset)
            video_bytes = sum(e - s for s, e in ranges)
            assert video_bytes == 4 * R
            # everything else in the plan is tiny: total is 4 records + small change
            assert video_bytes <= stats["total_bytes"] < video_bytes + 64 * 1024
            assert stats["num_ranges"] >= 3

            full = loader.warmup_cache(verbose=False, indices=np.arange(len(ds)))
            assert full["total_bytes"] > 10 * R
            assert full["subset"] is True and full["num_records"] == len(ds)
        finally:
            loader.shutdown()

    def test_explicit_indices_override_and_no_subset_reads_whole_files(self, tmp_path):
        ds = _BigBytesDataset(cache_path=tmp_path / "cache")
        loader = SlipstreamLoader(ds, batch_size=2, shuffle=False, verbose=False)
        try:
            whole = loader.warmup_cache(verbose=False)
            assert whole["subset"] is False
            assert whole["num_records"] == len(ds)
            expected = sum(
                f.stat().st_size
                for f in list(Path(ds.cache_path).glob("*.bin")) + list(Path(ds.cache_path).glob("*.npy"))
            )
            assert whole["total_bytes"] == expected

            one = loader.warmup_cache(verbose=False, indices=[3])
            assert one["subset"] is True
            assert one["num_records"] == 1
            assert one["num_ranges"] >= 1
            assert _BigBytesDataset.REC <= one["total_bytes"] < 2 * _BigBytesDataset.REC
        finally:
            loader.shutdown()

    def test_output_identical_after_subset_warmup(self, tmp_path):
        ds = _BigBytesDataset(cache_path=tmp_path / "cache")
        subset = [9, 2, 4]
        loader = SlipstreamLoader(
            ds, batch_size=2, shuffle=False, drop_last=False, indices=subset, verbose=False,
        )
        try:
            before = list(loader)
            loader.warmup_cache(verbose=False)
            after = list(loader)
            assert len(before) == len(after) == 2
            for b1, b2 in zip(before, after):
                np.testing.assert_array_equal(b1["_indices"].numpy(), b2["_indices"].numpy())
                np.testing.assert_array_equal(b1["video"]["data"], b2["video"]["data"])
                for j, i in enumerate(b1["_indices"].numpy()):
                    n = int(b1["video"]["sizes"][j])
                    assert bytes(b1["video"]["data"][j, :n]) == ds._blobs[i]
        finally:
            loader.shutdown()

    def test_imagefolder_subset_warmup_smaller_than_full(self, tmp_path):
        dataset_root = _make_imagefolder(tmp_path / "dataset")
        reader = SlipstreamImageFolder(dataset_root, cache_dir=tmp_path / "cache")
        loader = SlipstreamLoader(
            reader, batch_size=2, shuffle=False, indices=[0, 1],
            pipelines={"image": [DecodeCenterCrop(size=8)]}, verbose=False,
        )
        try:
            sub = loader.warmup_cache(verbose=False)
            full = loader.warmup_cache(verbose=False, indices=np.arange(len(loader.cache)))
            assert sub["subset"] and sub["total_bytes"] < full["total_bytes"]
            storage = loader.cache.fields["image"]
            meta = storage._metadata
            plan = dict(loader._warmup_plan(np.array([0, 1])))
            ranges = plan[Path(loader.cache.cache_dir) / "image.bin"]
            assert ranges == [(0, int(meta["data_ptr"][1] + meta["data_size"][1]))]
            assert len(list(loader)) == 1
        finally:
            loader.shutdown()


class TestWarmupTouch:
    def test_touch_walks_selected_ranges_and_reports_time(self, tmp_path):
        ds = _BigBytesDataset(cache_path=tmp_path / "cache")
        loader = SlipstreamLoader(ds, batch_size=2, shuffle=False, drop_last=False, indices=[1, 4], verbose=False)
        try:
            with_touch = loader.warmup_cache(verbose=False)
            without = loader.warmup_cache(verbose=False, touch=False)
            assert with_touch["touch_sec"] >= 0 and without["touch_sec"] == 0
            assert with_touch["total_bytes"] == without["total_bytes"]
            # output unaffected
            b = next(iter(loader))
            n = int(b["video"]["sizes"][0])
            assert bytes(b["video"]["data"][0, :n]) == ds._blobs[1]
        finally:
            loader.shutdown()

    def test_touch_helper_reads_every_page_once(self):
        from slipstream.loader import _touch_mmap_ranges
        arr = np.arange(3 * 4096 + 100, dtype=np.uint8)          # plain array stands in for the mmap
        _touch_mmap_ranges(arr, [(10, 3 * 4096 + 50)])           # must not raise, covers partial pages
