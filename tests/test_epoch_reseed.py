"""0.9.0: every seeded stream is a pure function of (seed, rank, epoch).

The loader reseeds decoders, BatchAugment / RandomApply transforms and
after_batch_transforms (Mixup) at each epoch start and in set_epoch, so a run
resumed at epoch N equals the fresh run's epoch N. Uses the real imagenet10 val
cache under ~/.slipstream (skipped when absent).
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from slipstream import derive_seed
from slipstream.pipelines._common import CROP_OFFSET, _seed

CACHE = Path.home() / ".slipstream" / "imagenet10-s256_l512-jpeg-val"
NB = 3


def _to_float(x):
    return torch.as_tensor(np.asarray(x)).permute(0, 3, 1, 2).float() / 255


def _loader(pipe, after=None, bs=32, **kw):
    from slipstream.loader import SlipstreamLoader

    if not CACHE.exists():
        pytest.skip(f"{CACHE} not present")
    return SlipstreamLoader(SimpleNamespace(cache_path=CACHE, remote_dir=None), batch_size=bs, seed=0,
                            verbose=False, pipelines={"image": pipe}, after_batch_transforms=after, **kw)


def _flip_rotate_pipe():
    from slipstream.decoders.crop import DecodeRandomResizedCrop
    from slipstream.transforms.base import RandomApply
    from slipstream.transforms.color_jitter import RandomColorJitter
    from slipstream.transforms.geometric import RandomHorizontalFlip, RandomRotate

    return [DecodeRandomResizedCrop(64, seed=7), _to_float,
            RandomHorizontalFlip(p=0.5, seed=1111), RandomRotate(p=0.5, max_deg=30, seed=2222),
            RandomApply([RandomColorJitter(p=1.0, hue=0.1, seed=3333)], p=0.5, seed=4444)]


def _epoch(loader, epoch=None, keys=("image",), n=NB, full=True):
    if epoch is not None:
        loader.set_epoch(epoch)
    out = []
    for i, b in enumerate(loader):
        if i < n:
            out.append({k: torch.as_tensor(np.asarray(b[k])).clone() for k in keys})
        elif not full:
            break
    return out


def _same(a, b):
    return all(torch.equal(x[k], y[k]) for x, y in zip(a, b) for k in x)


def test_transforms_resume_exactly():
    fresh = _loader(_flip_rotate_pipe())
    _epoch(fresh)                                   # epoch 0 (no set_epoch: iteration reseeds)
    e1 = _epoch(fresh)                              # epoch 1
    resumed = _loader(_flip_rotate_pipe())
    assert _same(e1, _epoch(resumed, 1))
    # partial / extra iteration in between doesn't leak into the next epoch
    other = _loader(_flip_rotate_pipe())
    _epoch(other, 0, full=False)
    assert _same(e1, _epoch(other, 1))
    assert not _same(e1, _epoch(_loader(_flip_rotate_pipe()), 2))


def test_after_batch_mixup_resumes():
    from slipstream.decoders.crop import DecodeCenterCrop
    from slipstream.transforms.mixup import Mixup

    mk = lambda: _loader([DecodeCenterCrop(64), _to_float], after=[Mixup(num_classes=1000, seed=9)])
    fresh = mk()
    _epoch(fresh)
    e1 = _epoch(fresh, keys=("image", "label"))
    assert _same(e1, _epoch(mk(), 1, keys=("image", "label")))


def test_multi_call_decoder_resumes():
    # DecodeUniformMultiRandomResizedCrop advances its counter once per crop per batch.
    from slipstream.decoders.multicrop import DecodeUniformMultiRandomResizedCrop

    mk = lambda: _loader([DecodeUniformMultiRandomResizedCrop(num_crops=3, size=48, seeds=[1, 2, 3])])
    fresh = mk()
    b0 = next(iter(fresh))
    keys = tuple(k for k in b0 if isinstance(b0[k], (np.ndarray, torch.Tensor)) and np.asarray(b0[k]).ndim == 4) or ("image",)
    _epoch(fresh, 0)
    e1 = _epoch(fresh, keys=keys)
    assert _same(e1, _epoch(mk(), 1, keys=keys))


def test_rank_changes_augmentation_not_order(monkeypatch):
    from slipstream.loader import SlipstreamLoader

    r0 = _loader(_flip_rotate_pipe())
    r1 = _loader(_flip_rotate_pipe())
    monkeypatch.setattr(r1, "_seed_rank", lambda: 1)
    a, b = _epoch(r0, 0, keys=("image", "_indices")), _epoch(r1, 0, keys=("image", "_indices"))
    assert all(torch.equal(x["_indices"], y["_indices"]) for x, y in zip(a, b))    # one shared permutation
    assert not any(torch.equal(x["image"], y["image"]) for x, y in zip(a, b))
    assert np.array_equal(SlipstreamLoader._generate_indices(r0, 3), SlipstreamLoader._generate_indices(r1, 3))


def test_standalone_equals_rank0_epoch0():
    # Objects never driven by a loader behave as rank 0, epoch 0.
    from slipstream.seeds import reseed
    from slipstream.transforms.geometric import RandomHorizontalFlip

    x = torch.rand(64, 3, 8, 8)
    a = RandomHorizontalFlip(p=0.5, seed=5)(x.clone())
    t = RandomHorizontalFlip(p=0.5, seed=5)
    reseed(t, (0, 3))
    reseed(t, (0, 0))
    assert torch.equal(a, t(x.clone()))


def test_preset_seeds_hashed():
    assert _seed(0, CROP_OFFSET, 1) != _seed(1, CROP_OFFSET, 0)       # was equal (base + offset + k)
    assert _seed(0, 1111, 0) != _seed(0, 2222, 0)
    assert _seed(None, CROP_OFFSET) is None
    assert _seed(3, CROP_OFFSET, 2) == derive_seed(3, CROP_OFFSET, 2) < 2**63


def _raw_batch(n=16):
    loader = _loader([], bs=n, shuffle=False)
    loader.pipelines = {}
    b = next(iter(loader))
    raw = b["image"]
    return {k: np.asarray(raw[k]) for k in ("data", "sizes", "heights", "widths")}


def test_cpu_decoder_seeded_random_crop():
    from slipstream.decoders.cpu import CPUDecoder, check_turbojpeg_available
    from slipstream.seeds import reseed

    if not check_turbojpeg_available():
        pytest.skip("TurboJPEG not available")
    raw = _raw_batch()
    args = (raw["data"], raw["sizes"], raw["heights"], raw["widths"])
    shapes = lambda imgs: [im.shape for im in imgs]
    d1, d2 = CPUDecoder(num_workers=2), CPUDecoder(num_workers=2)
    first = d1.decode_batch_random_crop(*args, seed=5)
    other = d2.decode_batch_random_crop(*args, seed=5)                  # fresh decoder, same seed
    assert all(np.array_equal(a, b) for a, b in zip(first, other))
    second = d1.decode_batch_random_crop(*args, seed=5)                 # next batch: new draws
    assert shapes(second) != shapes(first)
    reseed(d1, (0, 0))
    again = d1.decode_batch_random_crop(*args, seed=5)
    assert all(np.array_equal(a, b) for a, b in zip(first, again))
    reseed(d1, (1, 0))                                                   # another rank
    assert shapes(d1.decode_batch_random_crop(*args, seed=5)) != shapes(first)


def test_gpu_rois_seeded_match_cpu_boxes():
    # GPUDecoder.decode_batch_random_crop builds its ROIs with this call; no CUDA needed to check it.
    from slipstream.seeds import sample_seeds
    from slipstream.utils.crop import generate_batch_random_crop_params, generate_random_crop_params

    w = np.array([500, 375, 640, 333] * 4, np.int32)
    h = np.array([375, 500, 480, 500] * 4, np.int32)
    seeds = sample_seeds(5, 1, len(w))
    rois = generate_batch_random_crop_params(w, h, seeds=seeds)
    assert np.array_equal(rois, generate_batch_random_crop_params(w, h, seeds=seeds))
    assert not np.array_equal(rois, generate_batch_random_crop_params(w, h, seeds=sample_seeds(5, 2, len(w))))
    for i in range(len(w)):                                              # same boxes as the CPU decoder path
        c = generate_random_crop_params(int(w[i]), int(h[i]), rng=np.random.default_rng(int(seeds[i])))
        assert rois[i].tolist() == c.to_roi_array().tolist()
    # per-sample: changing one image doesn't move the others' boxes
    w2 = w.copy(); w2[3] = 200
    r2 = generate_batch_random_crop_params(w2, h, seeds=seeds)
    assert np.array_equal(np.delete(rois, 3, 0), np.delete(r2, 3, 0))
