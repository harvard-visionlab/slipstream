"""distributed + drop_last must truncate (DistributedSampler semantics), never train on padding.

Index logic only: no cache is read, so no dataset is needed.
"""

import numpy as np
import pytest

from slipstream.loader import SlipstreamLoader


def _shard(n, world, rank, batch, drop_last, shuffle=True):
    ld = SlipstreamLoader.__new__(SlipstreamLoader)
    ld.indices, ld.distributed, ld.world_size, ld.rank = np.arange(n), True, world, rank
    ld.shuffle, ld.seed, ld.drop_last, ld.batch_size = shuffle, 0, drop_last, batch
    pos = ld._generate_indices(3)
    nb = len(pos) // batch if drop_last else -(-len(pos) // batch)
    assert len(ld) == nb
    return pos[:nb * batch] if drop_last else pos


@pytest.mark.parametrize("n,world,batch", [(1001, 8, 7), (1000, 8, 7), (10, 3, 3), (7, 8, 1), (1001, 1, 7)])
def test_drop_last_no_duplicates(n, world, batch):
    seen = np.concatenate([_shard(n, world, r, batch, True) for r in range(world)])
    assert len(seen) == (n // world) // batch * batch * world
    assert len(np.unique(seen)) == len(seen)


@pytest.mark.parametrize("n,world,batch", [(1001, 8, 7), (10, 3, 4)])
def test_eval_still_pads_and_covers_all(n, world, batch):
    shards = [_shard(n, world, r, batch, False) for r in range(world)]
    assert {len(s) for s in shards} == {-(-n // world)}
    assert set(np.concatenate(shards)) == set(range(n))


def test_prefix_unchanged_vs_padding():
    # truncation only removes the tail: every rank's order is a prefix of the padded order
    for r in range(8):
        a = _shard(1001, 8, r, 1, True)
        b = _shard(1001, 8, r, 1, False)
        assert np.array_equal(a, b[:len(a)])
