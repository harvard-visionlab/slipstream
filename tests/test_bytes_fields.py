"""Tests for raw ``bytes`` fields (video containers, np.save blobs) in SlipstreamLoader.

Verifies that:
1. A ``bytes`` field is bank-eligible: auto-detected as the primary field when
   no image field exists, and served through the slot-rotated prefetch banks.
2. Non-primary bytes-backed fields come back as an owned ``{data, sizes}`` dict
   with byte-exact payloads even when the prefetch worker runs ahead
   (``batches_ahead > 1``) — the shared-scratch-buffer race.
3. A non-jpeg/yuv420 manifest ``image_format`` on a bytes field (e.g. "mp4")
   does not trip the YUV420 branches, and ``image_format="yuv420"`` on a
   bytes-only cache does not try to build a sibling YUV420 store.
4. Mixed caches (ImageBytes + bytes) keep the image field as primary and
   return the bytes field as a secondary dict.
"""

import io
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from slipstream.cache import ImageBytesStorage, OptimizedCache
from slipstream.loader import SlipstreamLoader

from tests.test_cache_roundtrip import _create_test_jpeg


# ---------------------------------------------------------------------------
# Mock datasets
# ---------------------------------------------------------------------------

def _npsave_blob(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


class MockVideoDataset:
    """video: bytes (opaque, ragged, one large outlier), poses: bytes (np.save),
    plus str/int scalars. Mirrors the SpatialVID-HQ record layout."""

    def __init__(self, num_samples: int = 23, cache_path: Path | None = None, seed: int = 0):
        self.num_samples = num_samples
        self.cache_path = cache_path
        rng = np.random.default_rng(seed)
        self._videos: list[bytes] = []
        self._poses: list[np.ndarray] = []
        for i in range(num_samples):
            n = 40_000 if i == 7 else int(rng.integers(500, 3000))  # outlier at 7
            self._videos.append(rng.integers(0, 256, n, dtype=np.uint8).tobytes())
            n_annot = int(rng.integers(1, 12))
            self._poses.append(rng.standard_normal((n_annot, 7)).astype(np.float32))
        self._names = [f"clip_{i:05d}" for i in range(num_samples)]
        self._labels = [i % 3 for i in range(num_samples)]

    @property
    def field_types(self) -> dict:
        return {"video": "bytes", "poses": "bytes", "name": "str", "label": "int"}

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        return {
            "video": self._videos[idx],
            "poses": _npsave_blob(self._poses[idx]),
            "name": self._names[idx],
            "label": self._labels[idx],
        }


class MockImagePlusBytesDataset:
    """image: ImageBytes (JPEG) plus meta: bytes (np.save blob)."""

    def __init__(self, num_samples: int = 10, cache_path: Path | None = None):
        self.num_samples = num_samples
        self.cache_path = cache_path
        self._images = [
            _create_test_jpeg(32 + 4 * i, 24 + 2 * i, (i * 20 % 256, 0, 0))
            for i in range(num_samples)
        ]
        self._meta = [np.arange(i + 1, dtype=np.int32) for i in range(num_samples)]

    @property
    def field_types(self) -> dict:
        return {"image": "ImageBytes", "meta": "bytes", "label": "int"}

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        return {
            "image": self._images[idx],
            "meta": _npsave_blob(self._meta[idx]),
            "label": idx,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rows(field: dict) -> list[bytes]:
    """Split a {data, sizes} dict into per-sample bytes."""
    data, sizes = field["data"], field["sizes"]
    return [bytes(data[i, : int(sizes[i])]) for i in range(len(sizes))]


def _check_video_batches(ds: MockVideoDataset, loader: SlipstreamLoader) -> int:
    """Iterate one epoch and assert every payload is exact. Returns #samples.

    The primary field (``video``) is a zero-copy view into the current bank
    slot, valid only until that slot is reused, so it is checked while the
    batch is live. Secondary bytes fields (``poses``) must be owned copies,
    so they are deliberately checked only after the whole epoch has been
    consumed — with ``batches_ahead > 1`` a view into the storage's shared
    scratch buffer would already hold a later batch's bytes.
    """
    seen = 0
    deferred: list[tuple[np.ndarray, dict, list, Any]] = []
    for batch in loader:
        idx = batch["_indices"].numpy()
        assert isinstance(batch["video"], dict) and isinstance(batch["poses"], dict)
        assert set(batch["poses"].keys()) == {"data", "sizes"}
        videos = _rows(batch["video"])
        assert len(videos) == len(idx)
        for j, i in enumerate(idx):
            assert videos[j] == ds._videos[i], f"video mismatch at sample {i}"
        deferred.append((idx, batch["poses"], batch["name"], batch["label"]))
        seen += len(idx)

    for idx, poses_field, names, labels in deferred:
        poses = _rows(poses_field)
        assert len(poses) == len(idx)
        for j, i in enumerate(idx):
            arr = np.load(io.BytesIO(poses[j]))
            np.testing.assert_array_equal(arr, ds._poses[i], err_msg=f"poses mismatch at sample {i}")
            assert names[j] == ds._names[i]
            assert int(labels[j]) == ds._labels[i]
    return seen


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBytesPrimaryField:
    @pytest.mark.parametrize("use_threading", [True, False])
    def test_bytes_field_auto_primary_exact_payloads(self, tmp_path, use_threading):
        """bytes field becomes primary; every payload is byte-exact through banks."""
        ds = MockVideoDataset(cache_path=tmp_path / "cache")
        loader = SlipstreamLoader(
            ds, batch_size=4, shuffle=False, drop_last=False,
            batches_ahead=3, use_threading=use_threading, verbose=False,
        )
        try:
            assert loader.image_field == "video"
            assert isinstance(loader._image_storage, ImageBytesStorage)
            assert loader._data_banks is not None
            assert len(loader._data_banks) == loader.batches_ahead + 2
            assert loader.image_format == "jpeg"  # untouched for bytes primary

            assert _check_video_batches(ds, loader) == len(ds)
        finally:
            loader.shutdown()

    def test_secondary_bytes_field_survives_prefetch_race(self, tmp_path):
        """Worker runs batches_ahead ahead; secondary 'poses' payloads must not be clobbered.

        ``list(loader)`` lets the worker fill the queue before the main thread
        consumes anything, so a view into the storage's shared scratch buffer
        would already hold a later batch's bytes.
        """
        ds = MockVideoDataset(num_samples=40, cache_path=tmp_path / "cache")
        loader = SlipstreamLoader(
            ds, batch_size=2, shuffle=True, seed=0, drop_last=False,
            batches_ahead=6, verbose=False,
        )
        try:
            assert len(loader) == 20
            assert _check_video_batches(ds, loader) == len(ds)
            # data is an owned copy, trimmed to the batch's max size
            for b in list(loader):
                p = b["poses"]
                assert p["data"].shape[1] == int(p["sizes"].max())
                assert p["data"].flags.owndata
        finally:
            loader.shutdown()

    def test_explicit_image_field_selects_other_bytes_field(self, tmp_path):
        """image_field='poses' gives poses the banks; video becomes a secondary dict."""
        ds = MockVideoDataset(cache_path=tmp_path / "cache")
        loader = SlipstreamLoader(
            ds, batch_size=5, shuffle=False, drop_last=False,
            image_field="poses", verbose=False,
        )
        try:
            assert loader.image_field == "poses"
            videos: list[tuple[np.ndarray, dict]] = []
            for b in loader:
                idx = b["_indices"].numpy()
                assert set(b["poses"].keys()) == {"data", "sizes", "heights", "widths"}
                assert set(b["video"].keys()) == {"data", "sizes"}
                # primary (poses) checked live, secondary (video) deferred
                for j, i in enumerate(idx):
                    np.testing.assert_array_equal(
                        np.load(io.BytesIO(_rows(b["poses"])[j])), ds._poses[i]
                    )
                videos.append((idx, b["video"]))
            for idx, vfield in videos:
                rows = _rows(vfield)
                for j, i in enumerate(idx):
                    assert rows[j] == ds._videos[i]
        finally:
            loader.shutdown()

    def test_subset_indices_with_bytes_primary(self, tmp_path):
        ds = MockVideoDataset(cache_path=tmp_path / "cache")
        subset = np.array([1, 7, 8, 20])  # includes the outlier
        loader = SlipstreamLoader(
            ds, batch_size=2, shuffle=False, drop_last=False,
            indices=subset, verbose=False,
        )
        try:
            got = np.concatenate([b["_indices"].numpy() for b in loader])
            np.testing.assert_array_equal(got, subset)
            assert _check_video_batches(ds, loader) == len(subset)
        finally:
            loader.shutdown()


class TestBytesImageFormatGuards:
    def test_non_image_manifest_format_does_not_trip_yuv(self, tmp_path):
        """manifest image_format='mp4' on a bytes field: no YUV conversion, raw bytes returned."""
        ds = MockVideoDataset(cache_path=tmp_path / "cache")
        OptimizedCache.build(ds, ds.cache_path, verbose=False, image_format="mp4")
        cache = OptimizedCache.load(ds.cache_path, verbose=False)
        assert cache.get_image_format("video") == "mp4"
        assert cache.get_image_format("poses") == "mp4"

        loader = SlipstreamLoader(ds, batch_size=4, shuffle=False, drop_last=False, verbose=False)
        try:
            assert loader.image_format == "jpeg"
            assert loader.image_field == "video"
            assert not list(Path(ds.cache_path).glob("*yuv420*"))
            assert _check_video_batches(ds, loader) == len(ds)
        finally:
            loader.shutdown()

    def test_yuv420_request_ignored_for_bytes_primary(self, tmp_path):
        ds = MockVideoDataset(cache_path=tmp_path / "cache")
        loader = SlipstreamLoader(
            ds, batch_size=4, shuffle=False, drop_last=False,
            image_format="yuv420", verbose=False,
        )
        try:
            assert not list(Path(ds.cache_path).glob("*yuv420*"))
            assert loader._image_storage is loader.cache.fields["video"]
            assert _check_video_batches(ds, loader) == len(ds)
        finally:
            loader.shutdown()


class TestMixedImageAndBytes:
    def test_image_field_stays_primary_bytes_is_secondary_dict(self, tmp_path):
        ds = MockImagePlusBytesDataset(cache_path=tmp_path / "cache")
        loader = SlipstreamLoader(
            ds, batch_size=3, shuffle=False, drop_last=False, batches_ahead=3, verbose=False,
        )
        try:
            assert loader.image_field == "image"
            assert loader._bytes_fields == {"meta"}
            deferred = []
            for b in loader:
                idx = b["_indices"].numpy()
                assert set(b["image"].keys()) == {"data", "sizes", "heights", "widths"}
                assert set(b["meta"].keys()) == {"data", "sizes"}
                imgs = _rows(b["image"])  # primary: live view, check now
                for j, i in enumerate(idx):
                    assert imgs[j] == ds._images[i]
                deferred.append((idx, b["meta"]))
            for idx, mfield in deferred:  # secondary: owned copy, check after epoch
                metas = _rows(mfield)
                for j, i in enumerate(idx):
                    np.testing.assert_array_equal(np.load(io.BytesIO(metas[j])), ds._meta[i])
        finally:
            loader.shutdown()

    def test_bogus_image_field_falls_back_to_image(self, tmp_path):
        ds = MockImagePlusBytesDataset(cache_path=tmp_path / "cache")
        loader = SlipstreamLoader(ds, batch_size=2, image_field="nope", verbose=False)
        try:
            assert loader.image_field == "image"
        finally:
            loader.shutdown()
