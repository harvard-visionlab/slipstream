"""SourceLoader (0.12.0): slipstream pipelines on batches from a live source / replay buffer.

Frames are real imagenet val images (center-cropped by the cache path), skipped when the cache is absent.
"""

import os
import threading
from pathlib import Path

import numpy as np
import pytest
import torch

from slipstream import SourceLoader
from slipstream.seeds import STREAM_SOURCE, derive_seed
from slipstream.transforms import Normalize, RandomHorizontalFlip, RandomResizedCropBatch, ToFloatDiv

CANDIDATES = [Path.home() / ".slipstream" / "imagenet10-s256_l512-jpeg-val",
              Path(os.environ.get("SLIPSTREAM_CACHE_DIR", "/nonexistent")) / "imagenet1k-s256_l512-jpeg-val"]
N, T, S = 4, 3, 64


@pytest.fixture(scope="module")
def frames():
    """[M, 3, S, S] uint8 real frames."""
    from slipstream import SlipstreamDataset, SlipstreamLoader
    from slipstream.decoders import DecodeCenterCrop

    cache = next((p for p in CANDIDATES if p.exists()), None)
    if cache is None:
        pytest.skip("no jpeg val cache present")
    ld = SlipstreamLoader(SlipstreamDataset(input_dir=str(cache)), batch_size=64, shuffle=False, verbose=False,
                          pipelines={"image": [DecodeCenterCrop(S)]}, use_threading=False)
    b = next(iter(ld))["image"]
    ld.shutdown()
    return torch.as_tensor(np.asarray(b)).permute(0, 3, 1, 2).contiguous()


class Replay:
    """Stateful replay source: its own generator picks N clips per batch; each clip repeats one frame T times."""

    def __init__(self, frames, on_policy=False):
        self.frames, self.on_policy = frames, on_policy
        self.g = torch.Generator().manual_seed(0)
        self.calls, self.threads, self.reseeds = [], set(), []

    def next_batch(self, step):
        self.calls.append(step)
        self.threads.add(threading.get_ident())
        idx = torch.randint(len(self.frames), (N,), generator=self.g)
        obs = self.frames[idx][:, None].expand(N, T, *self.frames.shape[1:]).contiguous()
        return {"obs": obs, "action": torch.arange(N * T).reshape(N, T), "reward": torch.zeros(N, T),
                "done": torch.zeros(N, T, dtype=torch.bool), "index": idx}

    def reseed(self, seed, key):
        self.reseeds.append((seed, key))
        self.g.manual_seed(seed + 1000 * key[0] + key[1])

    def state_dict(self):
        return {"g": self.g.get_state()}

    def load_state_dict(self, sd):
        self.g.set_state(sd["g"])


def _pipe(size=32):
    return {"obs": [RandomResizedCropBatch(size, scale=(0.2, 1.0), seed=1), RandomHorizontalFlip(seed=2)]}


def _take(loader, k):
    out = []
    for i, b in enumerate(loader):
        out.append({k_: v.clone() if isinstance(v, torch.Tensor) else v for k_, v in b.items()})
        if len(out) == k:
            break
    return out


def test_shapes_and_per_sequence_params(frames):
    ld = SourceLoader(Replay(frames), pipelines=_pipe(), seed=0, steps_per_epoch=3)
    batches = list(ld)
    assert len(batches) == 3 == len(ld)
    for b in batches:
        assert b["obs"].shape == (N, T, 3, 32, 32) and b["obs"].dtype == torch.uint8
        assert b["action"].shape == (N, T) and b["index"].shape == (N,)
        # identical input frames + one draw per sequence -> identical output frames within a sequence
        assert all(torch.equal(b["obs"][i, 0], b["obs"][i, t]) for i in range(N) for t in range(T))
    assert ld.epoch == 1 and ld.step == 0


def test_deterministic_and_rank_dependent(frames):
    a = _take(SourceLoader(Replay(frames), pipelines=_pipe(), seed=0, steps_per_epoch=4), 4)
    b = _take(SourceLoader(Replay(frames), pipelines=_pipe(), seed=0, steps_per_epoch=4), 4)
    assert all(torch.equal(x["obs"], y["obs"]) for x, y in zip(a, b))
    other = SourceLoader(Replay(frames), pipelines=_pipe(), seed=0, steps_per_epoch=4)
    other.rank = 1
    c = _take(other, 4)
    assert not all(torch.equal(x["index"], y["index"]) for x, y in zip(a, c))


def test_source_reseed_key(frames):
    src = Replay(frames)
    ld = SourceLoader(src, pipelines=_pipe(), seed=7, steps_per_epoch=2)
    ld.set_epoch(5)
    list(ld)
    assert src.reseeds == [(derive_seed(7, STREAM_SOURCE), (0, 5))]
    src2 = Replay(frames)
    ld2 = SourceLoader(src2, pipelines=_pipe(), seed=7, source_seed=11, steps_per_epoch=2)
    ld2.set_epoch(5)
    list(ld2)
    assert src2.reseeds == [(derive_seed(11, STREAM_SOURCE), (0, 5))]
    assert ld2.state_dict()["source_seed"] == 11


@pytest.mark.parametrize("threaded", [False, True])
def test_mid_epoch_resume_is_exact(frames, threaded):
    fresh = SourceLoader(Replay(frames), pipelines=_pipe(), seed=0, steps_per_epoch=6, use_threading=threaded)
    fresh.set_epoch(2)
    want = _take(fresh, 6)

    run = SourceLoader(Replay(frames), pipelines=_pipe(), seed=0, steps_per_epoch=6, use_threading=threaded)
    run.set_epoch(2)
    _take(run, 3)                                   # consume 3 (prefetch may have read further)
    sd = run.state_dict()
    assert (sd["epoch"], sd["step"]) == (2, 3)
    run.shutdown()

    resumed = SourceLoader(Replay(frames), pipelines=_pipe(), seed=0, steps_per_epoch=6, use_threading=threaded)
    resumed.load_state_dict(sd)
    got = list(resumed)
    assert len(got) == 3
    assert all(torch.equal(x["obs"], y["obs"]) and torch.equal(x["index"], y["index"]) for x, y in zip(want[3:], got))


def test_threaded_matches_inline(frames):
    a = _take(SourceLoader(Replay(frames), pipelines=_pipe(), seed=0, steps_per_epoch=5, use_threading=False), 5)
    src = Replay(frames)
    b = _take(SourceLoader(src, pipelines=_pipe(), seed=0, steps_per_epoch=5, use_threading=True), 5)
    assert all(torch.equal(x["obs"], y["obs"]) for x, y in zip(a, b))
    assert threading.get_ident() not in src.threads


def test_on_policy_is_inline_and_never_ahead(frames):
    src = Replay(frames, on_policy=True)
    ld = SourceLoader(src, pipelines=_pipe(), seed=0, steps_per_epoch=4, batches_ahead=3)
    for i, _ in enumerate(ld):
        assert src.calls == list(range(i + 1))      # produced only when asked for
    assert src.threads == {threading.get_ident()}


def test_source_buffer_never_modified(frames):
    class View:
        def __init__(self):
            self.buf = frames[:N].clone()[:, None].expand(N, T, *frames.shape[1:]).contiguous()
            self.orig = self.buf.clone()

        def next_batch(self, step):
            return {"obs": self.buf}                 # the source's own storage, as-is

    src = View()
    ld = SourceLoader(src, pipelines={"obs": [RandomHorizontalFlip(p=1.0, seed=0),
                                              Normalize([0.5] * 3, [0.2] * 3, inplace=True)]},
                      seed=0, steps_per_epoch=2)
    for b in ld:
        assert b["obs"].dtype == torch.float32
    assert torch.equal(src.buf, src.orig)


def test_multiview_views_independent(frames):
    pipes = {"obs": [[RandomHorizontalFlip(p=1.0, seed=0)], [ToFloatDiv(255.0)]]}
    b = next(iter(SourceLoader(Replay(frames), pipelines=pipes, seed=0, steps_per_epoch=1)))
    flipped, plain = b["obs"]
    assert torch.allclose(flipped.flip(-1).float() / 255.0, plain)   # view 2 did not see view 1's flip


def test_set_resolution_and_decoders_rejected(frames):
    ld = SourceLoader(Replay(frames), pipelines=_pipe(48), seed=0, steps_per_epoch=1)
    ld.set_resolution(24)
    assert next(iter(ld))["obs"].shape[-2:] == (24, 24) and ld.resolution == 24
    from slipstream.decoders import DecodeCenterCrop
    with pytest.raises(TypeError, match="decoder stage"):
        SourceLoader(Replay(frames), pipelines={"obs": [DecodeCenterCrop(32)]})


def test_apply_runs_pipelines_without_advancing(frames):
    ld = SourceLoader(Replay(frames), pipelines={"obs": [ToFloatDiv(255.0)]}, seed=0, steps_per_epoch=2)
    out = ld.apply({"obs": frames[:2][:, None]})
    assert out["obs"].shape == (2, 1, 3, S, S) and out["obs"].dtype == torch.float32 and ld.step == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("source_on_gpu", [False, True])
def test_cuda(frames, source_on_gpu):
    fr = frames.cuda() if source_on_gpu else frames
    ld = SourceLoader(Replay(fr), pipelines={"obs": [RandomResizedCropBatch(32, seed=1, device="cuda"),
                                                     ToFloatDiv(255.0, dtype=torch.bfloat16)]},
                      device="cuda", seed=0, steps_per_epoch=3)
    for b in ld:
        assert b["obs"].is_cuda and b["obs"].dtype == torch.bfloat16 and b["obs"].shape == (N, T, 3, 32, 32)
