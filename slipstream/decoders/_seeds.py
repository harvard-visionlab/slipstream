"""Seed derivation for per-sample random draws.

Every seeded draw in the decoders is keyed by ``(seed, counter, *stream)`` and
hashed through :class:`numpy.random.SeedSequence`, so nearby seeds, counters and
sample positions give unrelated streams. (Before 0.8.0 the per-sample seed was
``seed + batch_size * counter + i``, which made seed ``s + 1`` a one-sample shift
of seed ``s`` and let consecutive batches / views share draws.)

``counter`` is a decoder's ``_seed_counter`` (reset by ``SlipstreamLoader.set_epoch``);
``stream`` separates independent draws that share a seed and counter (crop size
vs position, embed placement per crop). Crops that pass the same seed and stream
get identical draws, which is what yoked crops rely on.
"""

from __future__ import annotations

import numpy as np

_MASK64 = (1 << 64) - 1

# Stream tags: independent draw families under one (seed, counter).
STREAM_POSITION = 0
STREAM_SIZE = 1
STREAM_EMBED = 2


def _key(seed: int | None, *parts: int) -> list[int]:
    # SeedSequence needs non-negative ints; the leading flag keeps seed=None apart from seed=0.
    head = [0, 0] if seed is None else [1, int(seed) & _MASK64]
    return head + [int(p) & _MASK64 for p in parts]


def sample_seeds(seed: int | None, counter: int, n: int, *stream: int) -> np.ndarray:
    """``n`` per-sample uint32 seeds (as int64 [n]) for batch ``counter`` of ``seed``."""
    state = np.random.SeedSequence(_key(seed, counter, *stream)).generate_state(max(int(n), 1), np.uint32)
    return state[:n].astype(np.int64)


def epoch_rng(seed: int | None, epoch: int) -> np.random.Generator:
    """Generator for the loader's epoch-``epoch`` shuffle (``seed=None``: fresh entropy)."""
    if seed is None:
        return np.random.default_rng()
    return np.random.default_rng(np.random.SeedSequence(_key(seed, epoch)))
