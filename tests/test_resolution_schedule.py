"""Progressive resolution (0.11.0): ResolutionSchedule + SlipstreamLoader(resolution_schedule=...).

Loader tests use the real imagenet10 val caches under ~/.slipstream (skipped when absent).
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from slipstream import ResolutionSchedule

CACHE = Path.home() / ".slipstream" / "imagenet10-s256_l512-{fmt}-val"


def _reference(epoch, min_res, max_res, end_ramp, start_ramp, step=32):
    """lrm-ssl lrm_ssl_torch/train.py get_resolution (step generalised)."""
    if epoch <= start_ramp:
        return min_res
    if epoch >= end_ramp:
        return max_res
    interp = np.interp([epoch], [start_ramp, end_ramp], [min_res, max_res])
    return int(np.round(interp[0] / step)) * step


@pytest.mark.parametrize("args", [(160, 192, 65, 76, 32), (160, 224, 10, 40, 32), (96, 224, 0, 16, 16),
                                  (128, 128, 5, 5, 8), (64, 256, 2, 90, 8)])
def test_matches_lrm_ssl(args):
    mn, mx, s0, s1, step = args
    sched = ResolutionSchedule(mn, mx, s0, s1, step)
    for e in range(0, 100):
        assert sched(e) == _reference(e, mn, mx, s1, s0, step), e


@pytest.mark.parametrize("bad", [dict(min_res=192, max_res=160), dict(min_res=170), dict(max_res=200),
                                 dict(start_ramp=10, end_ramp=5), dict(step=0), dict(min_res=160.0),
                                 dict(start_ramp=-1)])
def test_validation(bad):
    kw = dict(min_res=160, max_res=192, start_ramp=5, end_ramp=10, step=32) | bad
    with pytest.raises((ValueError, TypeError)):
        ResolutionSchedule(**kw)


def _loader(fmt, sched=None, pipe=None, threaded=True):
    from slipstream import SlipstreamDataset, SlipstreamLoader
    from slipstream.decoders import DecodeRandomResizedCrop

    path = Path(str(CACHE).format(fmt=fmt))
    if not path.exists():
        pytest.skip(f"{path} not present")
    pipe = pipe if pipe is not None else [DecodeRandomResizedCrop(224, seed=1)]
    return SlipstreamLoader(SlipstreamDataset(input_dir=str(path)), batch_size=32, seed=0, image_format=fmt,
                            verbose=False, use_threading=threaded, pipelines={"image": pipe},
                            resolution_schedule=sched)


def _epoch(loader, epoch, n=3):
    loader.set_epoch(epoch)
    out = []
    for i, b in enumerate(loader):
        if i < n:
            out.append(torch.as_tensor(np.asarray(b["image"])).clone())
        else:
            break
    return out


@pytest.mark.parametrize("fmt", ["jpeg", "yuv420"])
def test_sizes_follow_schedule_threaded(fmt):
    sched = ResolutionSchedule(min_res=96, max_res=160, start_ramp=1, end_ramp=3, step=32)
    loader = _loader(fmt, sched)
    for e, want in [(0, 96), (1, 96), (2, 128), (3, 160), (1, 96)]:          # also going back down
        batches = _epoch(loader, e)
        assert all(b.shape[1:3] == (want, want) for b in batches), (e, [tuple(b.shape) for b in batches])
        assert loader.resolution == want


def test_resume_is_exact_across_size_changes():
    sched = ResolutionSchedule(min_res=96, max_res=160, start_ramp=0, end_ramp=2, step=32)
    fresh = _loader("jpeg", sched)
    for e in (0, 1):
        _epoch(fresh, e)
    want = _epoch(fresh, 2)
    resumed = _loader("jpeg", sched)
    got = _epoch(resumed, 2)
    assert all(torch.equal(a, b) for a, b in zip(want, got))


def test_crop_boxes_do_not_depend_on_size():
    # Same (seed, rank, epoch) at two sizes: same source boxes, so a downscale of the big crop
    # is close to the small crop (not exact: different resample paths).
    a = _loader("jpeg")
    a.set_resolution(192)
    big = _epoch(a, 4)[0].permute(0, 3, 1, 2).float()
    b = _loader("jpeg")
    b.set_resolution(96)
    small = _epoch(b, 4)[0].permute(0, 3, 1, 2).float()
    down = torch.nn.functional.interpolate(big, size=(96, 96), mode="area")
    assert (down - small).abs().mean() < 8                                   # different crops differ by ~40+


def test_manual_set_resolution_and_eval_stages_untouched():
    from slipstream.decoders import DecodeCenterCrop

    loader = _loader("jpeg")
    loader.set_resolution(128)
    assert _epoch(loader, 0)[0].shape[1:3] == (128, 128)
    with pytest.raises(ValueError, match="no resizable crop stage"):
        _loader("jpeg", ResolutionSchedule(96, 160, 0, 2), pipe=[DecodeCenterCrop(224)])
    ev = _loader("jpeg", pipe=[DecodeCenterCrop(224)])
    with pytest.raises(ValueError, match="no resizable crop stage"):
        ev.set_resolution(128)
