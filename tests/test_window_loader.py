"""Tests for SlipstreamLoader(window=(T, stride)) and fixed-shape array fields.

The mock is a tiny "frame store": clips of varying length, one JPEG per frame,
records of a clip contiguous, per-frame scalar / array / string fields. Every
frame of a clip carries the same textured image (distinct per clip), so
"identical augmentation across the T frames of a window" is observable as
identical output pixels, while different windows of the same clip normally
get different crops.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from slipstream.cache import OptimizedCache, array_type_of, parse_array_type
from slipstream.decoders import DecodeRandomResizedCrop
from slipstream.loader import SlipstreamLoader
from slipstream.transforms.geometric import RandomHorizontalFlip


# ---------------------------------------------------------------------------
# Mock frame store
# ---------------------------------------------------------------------------

def _textured_jpeg(seed: int, w: int = 64, h: int = 48) -> bytes:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    base = np.stack([xx * 255 // w, yy * 255 // h, ((xx + yy) * 255 // (w + h))], axis=-1)
    noise = rng.integers(0, 80, (h, w, 3))
    arr = np.clip(base * 0.7 + noise, 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=92)
    return buf.getvalue()


class MockFrameStore:
    """Clips of lengths `lengths`; frame record fields mirror the SpatialVID 5 Hz frame store."""

    def __init__(self, lengths=(5, 8, 3, 12, 6), cache_path: Path | None = None):
        self.cache_path = cache_path
        self.lengths = list(lengths)
        self._rows = []
        rng = np.random.default_rng(0)
        for clip_rec, n_k in enumerate(self.lengths):
            jpeg = _textured_jpeg(clip_rec)
            for k in range(n_k):
                self._rows.append(dict(
                    image=jpeg, clip_rec=clip_rec, k=k, n_k=n_k, t_sec=0.2 * k,
                    pose=(rng.standard_normal(7)).astype(np.float32),
                    intrinsics=np.array([0.4, 0.7, 0.5, 0.5], dtype=np.float32),
                    clip_id=f"clip_{clip_rec:03d}",
                ))

    @property
    def field_types(self) -> dict:
        return {"image": "ImageBytes", "clip_rec": "int", "k": "int", "n_k": "int", "t_sec": "float",
                "pose": "float32[7]", "intrinsics": "float32[4]", "clip_id": "str"}

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> dict:
        return self._rows[idx]

    def anchors(self, T: int, stride: int = 1) -> np.ndarray:
        """Records a with k + (T-1)*stride < n_k, i.e. windows that stay inside one clip."""
        return np.array([i for i, r in enumerate(self._rows) if r["k"] + (T - 1) * stride < r["n_k"]], dtype=np.int64)


class MockInferredArrays:
    """Field types given as Python types (readers do this); arrays must be inferred as float32[3]."""

    def __init__(self, n: int = 6, cache_path: Path | None = None):
        self.cache_path = cache_path
        self._v = [np.arange(3, dtype=np.float32) * i for i in range(n)]
        self._img = [_textured_jpeg(i) for i in range(n)]

    @property
    def field_types(self) -> dict:
        return {"image": "ImageBytes", "vec": np.ndarray, "label": int}

    def __len__(self) -> int:
        return len(self._v)

    def __getitem__(self, idx: int) -> dict:
        return {"image": self._img[idx], "vec": self._v[idx], "label": idx}


def _make_loader(ds, T, stride=1, **kw):
    defaults = dict(batch_size=2, shuffle=False, drop_last=False, verbose=False,
                    indices=ds.anchors(T, stride), window=(T, stride))
    defaults.update(kw)
    return SlipstreamLoader(ds, **defaults)


# ---------------------------------------------------------------------------
# Array fields
# ---------------------------------------------------------------------------

class TestArrayFields:
    def test_parse_and_infer(self):
        assert parse_array_type("float32[7]") == (np.dtype("float32"), (7,))
        assert parse_array_type("int16[2, 3]") == (np.dtype("int16"), (2, 3))
        assert parse_array_type("float") is None and parse_array_type("ImageBytes") is None
        assert array_type_of(np.zeros((7,), np.float32)) == "float32[7]"
        assert array_type_of(np.zeros((2, 3), np.int64)) == "int64[2,3]"

    @pytest.mark.parametrize("num_workers", [1, 2])
    def test_roundtrip_exact_dtype_and_shape(self, tmp_path, num_workers):
        ds = MockFrameStore(cache_path=tmp_path / "cache")
        cache = OptimizedCache.build(ds, ds.cache_path, verbose=False, num_workers=num_workers)
        pose = cache.fields["pose"]._data
        assert pose.dtype == np.float32 and pose.shape == (len(ds), 7)
        assert cache.fields["intrinsics"]._data.shape == (len(ds), 4)
        for i in [0, 7, len(ds) - 1]:
            np.testing.assert_array_equal(pose[i], ds[i]["pose"])
        # records keep iterator order (contiguous per clip)
        clip_rec = cache.fields["clip_rec"]._data
        assert np.all(np.diff(clip_rec) >= 0)
        assert cache.verify(ds, num_checks=10, verbose=False)
        # manifest carries the type string; reload works
        cache2 = OptimizedCache.load(ds.cache_path, verbose=False)
        assert cache2.field_types["pose"] == "float32[7]"
        np.testing.assert_array_equal(cache2.fields["pose"].load_batch(np.array([3, 5]))["data"], pose[[3, 5]])

    def test_inferred_from_ndarray_values(self, tmp_path):
        ds = MockInferredArrays(cache_path=tmp_path / "cache")
        cache = OptimizedCache.build(ds, ds.cache_path, verbose=False)
        assert cache.field_types["vec"] == "float32[3]"
        np.testing.assert_array_equal(cache.fields["vec"]._data[4], ds[4]["vec"])

    def test_shape_mismatch_raises(self, tmp_path):
        ds = MockFrameStore(lengths=(3,), cache_path=tmp_path / "cache")
        ds._rows[1]["pose"] = np.zeros(6, np.float32)
        with pytest.raises(ValueError, match="expects shape"):
            OptimizedCache.build(ds, ds.cache_path, verbose=False)

    def test_loader_returns_array_fields_as_tensors(self, tmp_path):
        ds = MockFrameStore(cache_path=tmp_path / "cache")
        loader = SlipstreamLoader(ds, batch_size=4, shuffle=False, drop_last=False, verbose=False)
        try:
            b = next(iter(loader))
            assert b["pose"].shape == (4, 7) and b["pose"].dtype == torch.float32
            assert b["intrinsics"].shape == (4, 4)
        finally:
            loader.shutdown()


# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------

class TestWindowShapes:
    @pytest.mark.parametrize("use_threading", [True, False])
    @pytest.mark.parametrize("T,stride", [(4, 1), (3, 2)])
    def test_every_field_is_B_T(self, tmp_path, use_threading, T, stride):
        ds = MockFrameStore(cache_path=tmp_path / "cache")
        loader = _make_loader(ds, T, stride, use_threading=use_threading)
        try:
            anchors = ds.anchors(T, stride)
            assert len(loader) == -(-len(anchors) // 2)
            seen = []
            for b in loader:
                B = b["_anchors"].shape[0]
                assert b["_indices"].shape == (B, T)
                exp = b["_anchors"][:, None] + stride * torch.arange(T)[None, :]
                assert torch.equal(b["_indices"], exp)
                # raw image dict folded
                assert b["image"]["data"].shape[:2] == (B, T) and b["image"]["sizes"].shape == (B, T)
                # scalar / array / str fields folded
                assert b["pose"].shape == (B, T, 7) and b["intrinsics"].shape == (B, T, 4)
                assert b["t_sec"].shape == (B, T) and b["k"].shape == (B, T)
                assert isinstance(b["clip_id"], list) and len(b["clip_id"]) == B and len(b["clip_id"][0]) == T
                # a window never crosses a clip boundary and k advances by stride
                assert torch.equal(b["clip_rec"][:, 1:], b["clip_rec"][:, :-1])
                assert torch.equal(b["k"][:, 1:] - b["k"][:, :-1], torch.full((B, T - 1), stride))
                for i in range(B):
                    assert len(set(b["clip_id"][i])) == 1
                    for t in range(T):
                        rec = int(b["_indices"][i, t])
                        np.testing.assert_array_equal(b["pose"][i, t].numpy(), ds[rec]["pose"])
                        assert bytes(b["image"]["data"][i, t, : int(b["image"]["sizes"][i, t])]) == ds[rec]["image"]
                seen.extend(b["_anchors"].tolist())
            assert seen == anchors.tolist()
        finally:
            loader.shutdown()

    def test_default_anchors_when_indices_none(self, tmp_path):
        ds = MockFrameStore(cache_path=tmp_path / "cache")
        loader = SlipstreamLoader(ds, batch_size=8, shuffle=False, drop_last=False, verbose=False, window=(3, 2))
        try:
            assert loader._num_anchors() == len(ds) - 4
            last = None
            for b in loader:
                last = b
            assert int(last["_indices"].max()) == len(ds) - 1
        finally:
            loader.shutdown()

    def test_bad_anchor_rejected(self, tmp_path):
        ds = MockFrameStore(cache_path=tmp_path / "cache")
        with pytest.raises(ValueError, match="anchor"):
            SlipstreamLoader(ds, batch_size=2, verbose=False, indices=[len(ds) - 1], window=(4, 1))

    def test_shuffle_shard_droplast_on_anchors(self, tmp_path):
        ds = MockFrameStore(cache_path=tmp_path / "cache")
        anchors = ds.anchors(4)
        loader = _make_loader(ds, 4, shuffle=True, seed=0, batch_size=3, drop_last=True)
        try:
            assert len(loader) == len(anchors) // 3
            order0 = loader._generate_indices(0)
            assert sorted(order0.tolist()) == anchors.tolist() and order0.tolist() != anchors.tolist()
            assert loader._generate_indices(1).tolist() != order0.tolist()
            # distributed sharding: disjoint strided anchor subsets that cover all anchors
            loader.distributed, loader.world_size = True, 2
            shards = []
            for r in range(2):
                loader.rank = r
                shards.append(loader._generate_indices(0))
            per_rank = -(-len(anchors) // 2)
            assert all(len(s) == per_rank for s in shards)
            assert set(np.concatenate(shards).tolist()) == set(anchors.tolist())
            # disjoint apart from the (at most one) padding anchor
            assert len(set(shards[0].tolist()) & set(shards[1].tolist())) <= len(anchors) % 2
        finally:
            loader.shutdown()

    def test_warmup_covers_expanded_records(self, tmp_path):
        ds = MockFrameStore(cache_path=tmp_path / "cache")
        anchors = np.array([0, 5, 6])
        loader = _make_loader(ds, 3, 1, indices=anchors)
        try:
            stats = loader.warmup_cache(verbose=False)
            expected = np.unique((anchors[:, None] + np.arange(3)[None, :]).ravel())
            assert stats["num_records"] == len(expected)
            plan = dict(loader._warmup_plan(expected))
            meta = loader.cache.fields["image"]._metadata
            lo = int(meta["data_ptr"][expected.min()]); hi = int(meta["data_ptr"][expected.max()] + meta["data_size"][expected.max()])
            ranges = plan[Path(loader.cache.cache_dir) / "image.bin"]
            assert ranges[0][0] == lo and ranges[-1][1] == hi
        finally:
            loader.shutdown()


class TestWindowAugmentation:
    def _pipeline(self, seed):
        return [DecodeRandomResizedCrop(size=24, scale=(0.3, 1.0), seed=seed, to_tensor=True, permute=True),
                RandomHorizontalFlip(p=0.5, seed=seed + 1)]

    def test_same_augmentation_within_window_different_across(self, tmp_path):
        T = 4
        ds = MockFrameStore(lengths=(12, 12, 12), cache_path=tmp_path / "cache")
        loader = _make_loader(ds, T, batch_size=6, pipelines={"image": self._pipeline(7)})
        try:
            n_diff_pairs = 0
            for b in loader:
                img = b["image"]
                B = img.shape[0]
                assert img.shape == (B, T, 3, 24, 24)
                # frames of a clip are identical JPEGs -> same params => same pixels across T
                for i in range(B):
                    for t in range(1, T):
                        assert torch.equal(img[i, 0], img[i, t]), f"window {i} frame {t} differs"
                # windows from the same clip get their own params: not all equal
                same_clip = b["clip_rec"][:, 0]
                for i in range(B):
                    for j in range(i + 1, B):
                        if same_clip[i] == same_clip[j] and not torch.equal(img[i, 0], img[j, 0]):
                            n_diff_pairs += 1
            assert n_diff_pairs > 0
            # every decoder / transform in the pipeline got the window size
            for obj in loader._walk_transforms(list(loader.pipelines.values())):
                if hasattr(obj, "seed_repeat"):
                    assert obj.seed_repeat == T
        finally:
            loader.shutdown()

    @pytest.mark.parametrize("use_threading", [True, False])
    def test_seeded_runs_identical(self, tmp_path, use_threading):
        T = 3
        ds = MockFrameStore(lengths=(9, 7, 11), cache_path=tmp_path / "cache")
        out = []
        for _ in range(2):
            loader = _make_loader(ds, T, batch_size=4, shuffle=True, seed=123,
                                  pipelines={"image": self._pipeline(11)}, use_threading=use_threading)
            try:
                out.append([(b["_anchors"].clone(), b["image"].clone(), b["pose"].clone()) for b in loader])
            finally:
                loader.shutdown()
        assert len(out[0]) == len(out[1]) > 0
        for (a0, i0, p0), (a1, i1, p1) in zip(*out):
            assert torch.equal(a0, a1) and torch.equal(i0, i1) and torch.equal(p0, p1)

    def test_window_one_matches_plain_loader(self, tmp_path):
        """window=(1,1) is exactly the old behaviour: no fold, no '_anchors'."""
        ds = MockFrameStore(cache_path=tmp_path / "cache")
        a = SlipstreamLoader(ds, batch_size=4, shuffle=False, drop_last=False, verbose=False,
                             pipelines={"image": self._pipeline(3)})
        b = SlipstreamLoader(ds, batch_size=4, shuffle=False, drop_last=False, verbose=False,
                             pipelines={"image": self._pipeline(3)}, window=(1, 1))
        try:
            for x, y in zip(a, b):
                assert "_anchors" not in x and "_anchors" not in y
                assert torch.equal(x["image"], y["image"]) and x["pose"].shape == y["pose"].shape == (x["image"].shape[0], 7)
        finally:
            a.shutdown(); b.shutdown()


class TestOtherDecodersHonourWindows:
    def test_short_crop_long_and_multicrop_params_shared_within_window(self, tmp_path):
        from slipstream.decoders import DecodeRandomResizeShortCropLong
        from slipstream.decoders.multicrop import DecodeMultiRandomResizedCrop
        T = 4
        ds = MockFrameStore(lengths=(12, 12), cache_path=tmp_path / "cache")
        scl = DecodeRandomResizeShortCropLong(size=(16, 24), size_mode="per_image", x_range=(0, 1), seed=5)
        mc = DecodeMultiRandomResizedCrop({"a": dict(size=16, scale=(0.2, 1.0), seed=1), "b": dict(size=8, scale=(0.05, 0.4), seed=2)})
        loader = _make_loader(ds, T, batch_size=3, pipelines={"image": [scl]}, exclude_fields=None)
        try:
            b = next(iter(loader))
            lp = scl._last_params
            for key in ("target_sizes", "x_pos", "y_pos"):
                arr = np.asarray(lp[key]).reshape(-1, T)
                assert np.all(arr == arr[:, :1]), key
            assert not np.all(np.asarray(lp["x_pos"]).reshape(-1, T)[:, 0] == np.asarray(lp["x_pos"])[0])
        finally:
            loader.shutdown()
        loader = _make_loader(ds, T, batch_size=3, pipelines={"image": [mc]})
        try:
            out = next(iter(loader))
            for name, size in (("a", 16), ("b", 8)):
                img = out[name]
                assert img.shape[1] == T and img.shape[2:] in ((3, size, size), (size, size, 3))
                for i in range(img.shape[0]):
                    for t in range(1, T):
                        assert np.array_equal(np.asarray(img[i, 0]), np.asarray(img[i, t]))
        finally:
            loader.shutdown()
