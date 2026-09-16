"""Tests for DecodeVideoWindow (torchcodec, time-based windows over a `bytes` video field).

Synthetic HEVC clips are encoded with torchcodec's VideoEncoder: frame t has red = 2*t,
so the decoded content tells which frame was picked. Clips differ in fps and duration.
Skipped when libtorchcodec cannot load (on macOS: DYLD_LIBRARY_PATH=/opt/homebrew/lib).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from slipstream.loader import SlipstreamLoader

torchcodec = pytest.importorskip("torchcodec")
try:  # libtorchcodec needs FFmpeg shared libraries on the loader path
    from torchcodec.encoders import VideoEncoder
    from torchcodec.decoders import VideoDecoder
    _probe = VideoEncoder(torch.zeros(2, 3, 16, 16, dtype=torch.uint8), frame_rate=10)
    _probe.to_tensor("mp4", codec="libx264", pixel_format="yuv420p")
except Exception as exc:  # pragma: no cover
    pytest.skip(f"torchcodec not usable here: {str(exc)[:120]}", allow_module_level=True)

from slipstream.decoders import DecodeVideoWindow  # noqa: E402
from slipstream.transforms import RandomHorizontalFlip, RandomResizedCropBatch  # noqa: E402


def _clip(n_frames: int, fps: float, w: int = 96, h: int = 64, static: bool = False, seed: int = 0) -> bytes:
    rng = np.random.default_rng(seed)
    base = torch.from_numpy(rng.integers(0, 255, (3, h, w), dtype=np.uint8))
    frames = base.unsqueeze(0).repeat(n_frames, 1, 1, 1).clone()
    frames[:, 2] = min(250, 40 * seed)                # blue channel encodes the clip id
    if not static:
        for t in range(n_frames):
            frames[t, 0] = min(255, 2 * t)            # red channel encodes the frame index
    enc = VideoEncoder(frames, frame_rate=fps)
    opts = {"g": str(int(fps)), "x265-params": "log-level=none"}
    try:   # near-lossless so the red channel reads back as the frame index
        b = enc.to_tensor("mp4", codec="libx265", pixel_format="yuv420p", crf=8, extra_options=opts)
    except Exception:
        b = enc.to_tensor("mp4", codec="libx264", pixel_format="yuv420p", crf=8, extra_options={"g": str(int(fps))})
    return b.numpy().tobytes()


class MockVideoStore:
    """video: bytes (HEVC), duration_s, fps, clip_id — one record per clip, mixed fps and lengths."""

    SPEC = [(90, 30.0), (150, 60.0), (72, 24.0), (60, 30.0), (100, 25.0)]

    def __init__(self, cache_path: Path | None = None, static: bool = False):
        self.cache_path = cache_path
        self._rows = []
        for i, (n, fps) in enumerate(self.SPEC):
            self._rows.append(dict(video=_clip(n, fps, static=static, seed=i), duration_s=n / fps, fps=fps, clip_id=f"clip{i}"))

    @property
    def field_types(self):
        return {"video": "bytes", "duration_s": "float", "fps": "float", "clip_id": "str"}

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, i):
        return self._rows[i]


def _red_index(frames: torch.Tensor) -> torch.Tensor:
    """Frame index implied by the red channel (mean over the image), per frame."""
    return frames[..., 0, :, :].float().mean(dim=(-2, -1)) / 2.0


class TestDecodeVideoWindow:
    def test_shapes_times_and_content(self, tmp_path):
        ds = MockVideoStore(cache_path=tmp_path / "cache")
        T, rate = 8, 10.0
        stage = DecodeVideoWindow(T=T, rate_hz=rate, seed=1, num_workers=2)
        loader = SlipstreamLoader(ds, batch_size=5, shuffle=False, drop_last=False, verbose=False,
                                  pipelines={"video": [stage]})
        try:
            b = next(iter(loader))
            f = b["video"]
            assert f.shape == (5, T, 3, 64, 96) and f.dtype == torch.uint8
            assert b["video_t_sec"].shape == (5, T) and b["video_t0"].shape == (5,)
            assert torch.equal(b["video_rec"], torch.arange(5))
            for i in range(5):
                fps, dur = ds[i]["fps"], ds[i]["duration_s"]
                t = b["video_t_sec"][i]
                # true frame times: monotone, ~1/rate apart (within one source frame), never the last 2 frames
                assert torch.all(t[1:] >= t[:-1])
                assert torch.all((t[1:] - t[:-1] - 1 / rate).abs() <= 1 / fps + 1e-3)
                assert float(t[-1]) <= dur - 2 / fps + 1e-6
                assert float(b["video_t0"][i]) <= dur - T / rate + 1e-6
                # the decoded frames are the frames shown at those times
                idx = _red_index(f[i]).round()
                assert torch.allclose(idx, (t * fps).round(), atol=2.5)
        finally:
            loader.shutdown()

    def test_given_t0_via_sample_data_survives_shuffle_and_repeats(self, tmp_path):
        ds = MockVideoStore(cache_path=tmp_path / "cache")
        T, rate = 6, 10.0
        indices = np.array([0, 1, 1, 3, 4, 0])            # clip 1 and clip 0 twice with different starts
        t0 = np.array([0.0, 0.5, 1.5, 0.2, 1.0, 1.9])
        stage = DecodeVideoWindow(T=T, rate_hz=rate, t0_key="t0", seed=0)
        loader = SlipstreamLoader(ds, batch_size=4, shuffle=True, seed=3, drop_last=False, verbose=False,
                                  indices=indices, sample_data={"t0": t0}, pipelines={"video": [stage]})
        try:
            seen = {}
            for b in loader:
                assert torch.allclose(b["t0"].float(), b["video_t0"])  # side data reached the stage per sample
                for i in range(b["video_rec"].shape[0]):
                    rec, start = int(b["video_rec"][i]), float(b["video_t0"][i])
                    assert abs(float(b["video_t_sec"][i, 0]) - start) <= 1 / ds[rec]["fps"] + 1e-6
                    seen[(rec, round(start, 3))] = True
            assert set(seen) == {(int(r), round(float(s), 3)) for r, s in zip(indices, t0)}
        finally:
            loader.shutdown()

    def test_given_t0_that_does_not_fit_raises(self, tmp_path):
        ds = MockVideoStore(cache_path=tmp_path / "cache")
        stage = DecodeVideoWindow(T=10, rate_hz=5.0, t0_key="t0")       # 2 s window
        loader = SlipstreamLoader(ds, batch_size=1, shuffle=False, verbose=False, indices=[3],
                                  sample_data={"t0": [1.5]}, pipelines={"video": [stage]})  # clip 3 is 2 s long
        try:
            with pytest.raises(ValueError, match="does not fit"):
                next(iter(loader))
        finally:
            loader.shutdown()

    def test_seeded_runs_identical_and_set_epoch_resumes(self, tmp_path):
        ds = MockVideoStore(cache_path=tmp_path / "cache")
        def run(epoch=None):
            stage = DecodeVideoWindow(T=5, rate_hz=10.0, seed=42)
            loader = SlipstreamLoader(ds, batch_size=2, shuffle=True, seed=7, drop_last=False, verbose=False,
                                      pipelines={"video": [stage]})
            try:
                if epoch is not None:
                    list(loader)                                           # burn epoch 0
                    loader.set_epoch(epoch)
                return [(b["video_rec"].clone(), b["video_t0"].clone(), b["video"].clone()) for b in loader]
            finally:
                loader.shutdown()
        a, b = run(), run()
        for (r0, t0, f0), (r1, t1, f1) in zip(a, b):
            assert torch.equal(r0, r1) and torch.equal(t0, t1) and torch.equal(f0, f1)
        c = run(epoch=0)                                                   # resume at epoch 0 == first run
        for (r0, t0, f0), (r1, t1, f1) in zip(a, c):
            assert torch.equal(r0, r1) and torch.equal(t0, t1)
        # a different seed gives different window starts
        stage = DecodeVideoWindow(T=5, rate_hz=10.0, seed=43)
        loader = SlipstreamLoader(ds, batch_size=5, shuffle=False, verbose=False, pipelines={"video": [stage]})
        try:
            other = next(iter(loader))["video_t0"]
            assert not torch.allclose(torch.cat([x[1] for x in a]).sort().values, other.sort().values)
        finally:
            loader.shutdown()

    def test_inner_transforms_share_params_within_window(self, tmp_path):
        ds = MockVideoStore(cache_path=tmp_path / "cache", static=True)      # every frame of a clip identical
        T = 6
        stage = DecodeVideoWindow(T=T, rate_hz=10.0, seed=1, transforms=[
            RandomResizedCropBatch(32, scale=(0.3, 0.9), seed=5), RandomHorizontalFlip(p=0.5, seed=6)])
        loader = SlipstreamLoader(ds, batch_size=5, shuffle=False, verbose=False, pipelines={"video": [stage]})
        try:
            b = next(iter(loader))
            f = b["video"]
            assert f.shape == (5, T, 3, 32, 32) and f.dtype == torch.uint8
            # one crop matrix and one flip decision per window, shared by its T frames
            mat = stage.transforms[0].last_params()["mat"].reshape(5, T, 2, 3)
            do = stage.transforms[1].last_params()["do"].reshape(5, T)
            assert torch.equal(mat, mat[:, :1].expand_as(mat)) and torch.equal(do, do[:, :1].expand_as(do))
            assert not torch.equal(mat[:, 0], mat[:1, 0].expand_as(mat[:, 0]))       # windows differ
            # decoded "static" frames differ only by codec noise, so same params => near-identical pixels
            for i in range(5):
                for t in range(1, T):
                    assert (f[i, 0].int() - f[i, t].int()).abs().max() <= 12
            # inner transforms got seed_repeat = T even though the loader walked the pipelines
            assert all(t.seed_repeat == T for t in stage.transforms)
        finally:
            loader.shutdown()

    def test_resize_short_side_and_hw(self, tmp_path):
        ds = MockVideoStore(cache_path=tmp_path / "cache")
        for resize, expect in [(32, (32, 48)), ((40, 40), (40, 40))]:
            stage = DecodeVideoWindow(T=3, rate_hz=5.0, seed=1, resize=resize)
            loader = SlipstreamLoader(ds, batch_size=2, shuffle=False, verbose=False, pipelines={"video": [stage]})
            try:
                f = next(iter(loader))["video"]
                assert f.shape[-2:] == expect
            finally:
                loader.shutdown()

    def test_unthreaded_loader_and_repr(self, tmp_path):
        ds = MockVideoStore(cache_path=tmp_path / "cache")
        stage = DecodeVideoWindow(T=4, rate_hz=8.0, seed=2)
        loader = SlipstreamLoader(ds, batch_size=3, shuffle=False, drop_last=False, verbose=False,
                                  use_threading=False, pipelines={"video": [stage]})
        try:
            n = sum(b["video"].shape[0] for b in loader)
            assert n == len(ds)
            assert "DecodeVideoWindow(T=4" in repr(stage)
        finally:
            loader.shutdown()


class TestAsyncPipelining:
    """The loader submits decodes from its prefetch thread; results must stay aligned with their records."""

    def test_batches_ahead_alignment_and_parity_with_sync(self, tmp_path):
        ds = MockVideoStore(cache_path=tmp_path / "cache")
        indices = np.array([4, 0, 3, 1, 2, 0, 4, 3])
        def run(use_threading, batches_ahead):
            stage = DecodeVideoWindow(T=6, rate_hz=10.0, seed=9, num_workers=4)
            loader = SlipstreamLoader(ds, batch_size=2, shuffle=True, seed=5, drop_last=False, verbose=False,
                                      indices=indices, batches_ahead=batches_ahead, use_threading=use_threading,
                                      pipelines={"video": [stage]})
            try:
                assert (loader._async_stage is stage) == True
                out = []
                for b in loader:
                    assert torch.equal(b["video_rec"], b["_indices"])
                    blue = b["video"][:, :, 2].float().mean(dim=(-2, -1))          # [B, T]
                    expect = (40 * b["_indices"].float()).clamp(min=16, max=250)[:, None]   # yuv limited range floors 0 at 16
                    assert torch.allclose(blue, expect.expand_as(blue), atol=6), (blue, expect)
                    out.append((b["_indices"].clone(), b["video_t0"].clone(), b["video"].clone()))
                return out
            finally:
                loader.shutdown()
        a = run(True, 4)
        b = run(False, 1)
        assert len(a) == len(b) == 4
        for (i0, t0, f0), (i1, t1, f1) in zip(a, b):
            assert torch.equal(i0, i1) and torch.equal(t0, t1) and torch.equal(f0, f1)


    def test_device_list_pins_threads(self, tmp_path):
        ds = MockVideoStore(cache_path=tmp_path / "cache")
        stage = DecodeVideoWindow(T=4, rate_hz=10.0, seed=1, device=["cpu", "cpu", "cpu"], num_workers=3)
        loader = SlipstreamLoader(ds, batch_size=5, shuffle=False, verbose=False, pipelines={"video": [stage]})
        try:
            b = next(iter(loader))
            assert b["video"].shape[:2] == (5, 4) and stage.output_device == torch.device("cpu")
            assert "device='cpu,cpu,cpu'" in repr(stage)
        finally:
            loader.shutdown()


    def test_reuse_output_ring_gives_same_frames(self, tmp_path):
        ds = MockVideoStore(cache_path=tmp_path / "cache")
        def run(reuse):
            stage = DecodeVideoWindow(T=4, rate_hz=10.0, seed=3, num_workers=3, reuse_output=reuse, ring_size=6)
            loader = SlipstreamLoader(ds, batch_size=2, shuffle=True, seed=1, drop_last=False, verbose=False,
                                      batches_ahead=2, pipelines={"video": [stage]})
            try:
                return [(b["_indices"].clone(), b["video"].clone()) for b in loader]
            finally:
                loader.shutdown()
        for (i0, f0), (i1, f1) in zip(run(False), run(True)):
            assert torch.equal(i0, i1) and torch.equal(f0, f1)
