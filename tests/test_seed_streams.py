"""Hashed seed derivation (0.8.0): nearby seeds / counters / views give unrelated streams.

Before 0.8.0 per-sample crop seeds were ``seed + B*counter + i`` and the epoch
shuffle used ``seed + epoch``, so seed s+1 replayed seed s shifted by one sample
(or one epoch). These tests pin the new contract.
"""

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from slipstream.decoders._seeds import STREAM_SIZE, epoch_rng, sample_seeds
from slipstream.decoders.numba_decoder import (
    _generate_random_crop_params_batch,
    _generate_resize_short_crop_long_params_batch,
)

LR = (math.log(3 / 4), math.log(4 / 3))
B = 16


def test_sample_seeds_pinned():
    # Changing these values changes every seeded augmentation stream: bump the minor version.
    pinned = [1501029259, 415423813, 425966647, 3516250141]
    assert sample_seeds(0, 1, 4).tolist() == pinned
    assert sample_seeds(0, 1, 8)[:4].tolist() == pinned        # prefix-stable in n


def test_sample_seeds_keys_distinct():
    base = sample_seeds(0, 1, B)
    for other in (sample_seeds(1, 1, B), sample_seeds(0, 2, B), sample_seeds(None, 1, B),
                  sample_seeds(0, 1, B, STREAM_SIZE), sample_seeds(-1, 1, B)):
        assert len(np.intersect1d(base, other)) == 0
    assert len(np.unique(base)) == B
    assert base.min() >= 0 and base.max() < 2**32


def _crops(seed, counter=1):
    w = np.full(B + 1, 500, np.int32)
    h = np.full(B + 1, 375, np.int32)
    return _generate_random_crop_params_batch(w, h, 0.08, 1.0, *LR, sample_seeds(seed, counter, B + 1))


def test_nearby_seeds_not_shifted_copies():
    a0, a1 = _crops(1234), _crops(1235)
    assert not np.array_equal(a1[:B], a0[1:])                    # was equal before 0.8.0
    assert not np.array_equal(_crops(1234, 2)[:B], a0[1:])
    assert (a0 != a1).any(axis=1).mean() > 0.9


def test_numba_and_python_rng_agree():
    # Fast (Numba) and slow (Python RandomState) paths of the resize-short-crop-long
    # decoders draw positions from the same seeds; they must match.
    seeds = sample_seeds(7, 3, B)
    w = np.full(B, 500, np.int32)
    h = np.full(B, 375, np.int32)
    xs, ys = np.empty(B), np.empty(B)
    _generate_resize_short_crop_long_params_batch(w, h, 224, 0.0, 1.0, 0.0, 1.0, seeds, xs, ys)
    for i in range(B):
        rng = np.random.RandomState(seeds[i])
        assert xs[i] == pytest.approx(rng.uniform(0.0, 1.0))
        assert ys[i] == pytest.approx(rng.uniform(0.0, 1.0))


def test_epoch_orders_not_shifted():
    n = 1000
    def order(seed, epoch):
        pos = np.arange(n)
        epoch_rng(seed, epoch).shuffle(pos)
        return pos
    for e in range(3):
        assert not np.array_equal(order(1, e), order(0, e + 1))   # was equal before 0.8.0
    assert np.array_equal(order(3, 2), order(3, 2))


def test_loader_shuffle_uses_epoch_rng():
    from slipstream.loader import SlipstreamLoader

    cache = Path.home() / ".slipstream" / "imagenet10-s256_l512-jpeg-val"
    if not cache.exists():
        pytest.skip(f"{cache} not present")
    mk = lambda seed: SlipstreamLoader(SimpleNamespace(cache_path=cache, remote_dir=None),
                                       batch_size=32, seed=seed, verbose=False)
    l0, l1 = mk(0), mk(1)
    assert not np.array_equal(l1._generate_indices(0), l0._generate_indices(1))
    assert np.array_equal(l0._generate_indices(4), mk(0)._generate_indices(4))


def test_multicrop_views_independent_and_yoked():
    from slipstream.decoders.multicrop import DecodeMultiRandomResizedCrop

    cache = Path.home() / ".slipstream" / "imagenet10-s256_l512-jpeg-val"
    if not cache.exists():
        pytest.skip(f"{cache} not present")
    from slipstream.loader import SlipstreamLoader

    dec = DecodeMultiRandomResizedCrop({
        "v0": dict(size=64, seed=1234), "v1": dict(size=64, seed=1235),
        "yoke": dict(size=64, seed=1234),
    })
    loader = SlipstreamLoader(SimpleNamespace(cache_path=cache, remote_dir=None), batch_size=B,
                              shuffle=False, verbose=False, pipelines={"image": [dec]})
    b = next(iter(loader))
    v0, v1, yoke = (np.asarray(b[k]) for k in ("v0", "v1", "yoke"))
    assert np.array_equal(v0, yoke)                              # same seed -> same crops
    assert not np.array_equal(v1[: B - 1], v0[1:])               # view k+1 != view k of the next sample
