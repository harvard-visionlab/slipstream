"""0.16.2: distributed sharding with drop_last=False gives every rank ceil(N/W) indices, even when N < W/2.

Before, padding copied the order once (pos[:pad]), so N=1 on W=4 gave shard sizes [1, 1, 0, 0]: the empty ranks
skipped the forward collectives and the run could hang. Sampler-level (no process group needed).
"""

import math
import types

import numpy as np
import pytest

from slipstream.loader import SlipstreamLoader
from slipstream.seeds import epoch_rng


def _shards(n, W, drop_last=False, shuffle=True):
    out = []
    for r in range(W):
        s = types.SimpleNamespace(shuffle=shuffle, seed=0, distributed=True, drop_last=drop_last, world_size=W, rank=r,
                                  indices=None, _num_anchors=lambda: n)
        out.append(SlipstreamLoader._generate_indices(s, 0))
    return out


@pytest.mark.parametrize("n,W", [(1, 4), (2, 8), (3, 8), (1, 8), (5, 8), (7, 8), (9, 2), (17, 4), (100, 8)])
def test_equal_shards(n, W):
    sh = _shards(n, W)
    assert [len(x) for x in sh] == [math.ceil(n / W)] * W
    assert set(np.concatenate(sh).tolist()) == set(range(n))


@pytest.mark.parametrize("n,W", [(5, 8), (7, 8), (9, 2), (17, 4), (100, 8)])
def test_unchanged_when_padding_fits(n, W):
    """For n >= W/2 the cyclic pad equals the old one-copy pad."""
    pos = np.arange(n)
    epoch_rng(0, 0).shuffle(pos)
    old = np.concatenate([pos, pos[:math.ceil(n / W) * W - n]])
    for r, shard in enumerate(_shards(n, W)):
        np.testing.assert_array_equal(shard, old[r::W])


def test_empty_and_drop_last():
    assert all(len(x) == 0 for x in _shards(0, 4))
    assert all(len(x) == 0 for x in _shards(3, 8, drop_last=True))
