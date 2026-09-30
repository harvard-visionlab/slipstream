"""Seed derivation and per-epoch reseeding.

All seeded randomness in slipstream is a pure function of ``(seed, rank, epoch, ...)``,
hashed through :class:`numpy.random.SeedSequence` so nearby seeds, ranks, epochs,
batches and sample positions give unrelated streams.

- **Decoders** (crop / size / position / embed / video t0) draw per-sample seeds with
  :func:`sample_seeds` keyed by ``(seed, rank, epoch, counter, *stream)``, where
  ``counter`` is the decoder's ``_seed_counter`` (batches drawn this epoch).
- **Transforms** (``BatchAugment``, ``RandomApply``, ``Mixup``, ...) keep a sequential
  generator, seeded with :func:`stream_seed` = ``derive_seed(seed, rank, epoch)``.
- **The loader** calls :func:`reseed` on every object in its pipelines and
  ``after_batch_transforms`` at the start of each epoch (and in ``set_epoch``), so
  a run resumed at epoch N reproduces the fresh run's epoch N regardless of what
  ran before. ``rank`` is the global ``torch.distributed`` rank (0 when not
  distributed); the epoch shuffle (:func:`epoch_rng`) never includes it, so all
  ranks share one permutation.
- **Pipelines / configs** derive per-view and per-transform seeds from one base
  seed with :func:`derive_seed` (``derive_seed(base, offset, view)``) instead of
  adding offsets, which made base ``b`` view ``k+1`` equal base ``b+1`` view ``k``.

Objects used outside a loader behave as rank 0, epoch 0.
"""

from __future__ import annotations

from typing import Any

import numpy as np

_MASK64 = (1 << 64) - 1

# Stream tags: independent draw families under one (seed, counter).
STREAM_POSITION = 0
STREAM_SIZE = 1
STREAM_EMBED = 2

DEFAULT_KEY = (0, 0)    # (rank, epoch) for objects not driven by a loader


def _key(seed: int | None, *parts: int) -> list[int]:
    # SeedSequence needs non-negative ints; the leading flag keeps seed=None apart from seed=0.
    head = [0, 0] if seed is None else [1, int(seed) & _MASK64]
    return head + [int(p) & _MASK64 for p in parts]


def derive_seed(base: int, *parts: int) -> int:
    """Hash ``base`` and ``parts`` into an independent 63-bit seed.

    Use it for per-view / per-transform seeds: ``derive_seed(base, 1234, k)`` for
    view ``k`` instead of ``base + 1234 + k``.
    """
    return int(np.random.SeedSequence(_key(base, *parts)).generate_state(1, np.uint64)[0] >> np.uint64(1))


def sample_seeds(seed: int | None, counter: int, n: int, *stream: int, key: tuple = DEFAULT_KEY) -> np.ndarray:
    """``n`` per-sample uint32 seeds (as int64 [n]) for batch ``counter`` of ``seed`` under ``key``."""
    state = np.random.SeedSequence(_key(seed, *key, counter, *stream)).generate_state(max(int(n), 1), np.uint32)
    return state[:n].astype(np.int64)


def next_sample_seeds(obj: Any, seed: int | None, n: int, *stream: int) -> np.ndarray | None:
    """Advance ``obj._seed_counter`` and return its next ``n`` per-sample seeds (``None`` if unseeded)."""
    if seed is None:
        return None
    obj._seed_counter = getattr(obj, '_seed_counter', 0) + 1
    return sample_seeds(seed, obj._seed_counter, n, *stream, key=getattr(obj, '_seed_key', DEFAULT_KEY))


def stream_seed(obj: Any) -> int:
    """Seed for a transform's sequential generator: ``derive_seed(obj.seed, rank, epoch)``."""
    return derive_seed(obj.seed, *getattr(obj, '_seed_key', DEFAULT_KEY))


def epoch_rng(seed: int | None, epoch: int) -> np.random.Generator:
    """Generator for the loader's epoch-``epoch`` shuffle (``seed=None``: fresh entropy)."""
    if seed is None:
        return np.random.default_rng()
    return np.random.default_rng(np.random.SeedSequence(_key(seed, epoch)))


def reseed(obj: Any, key: tuple) -> None:
    """Restart ``obj``'s random streams for ``key = (rank, epoch)``.

    Decoders: ``_seed_counter`` / ``_embed_seed_counter`` go back to 0 under the new key.
    Seeded transforms: their torch / numpy generator is re-seeded with :func:`stream_seed`
    (a generator created lazily later picks up the key too). Unseeded objects are untouched.
    """
    import torch

    key = tuple(int(k) for k in key)
    if hasattr(obj, '_seed_counter') or hasattr(obj, '_embed_seed_counter'):
        obj._seed_key = key
        for attr in ('_seed_counter', '_embed_seed_counter'):
            if hasattr(obj, attr):
                setattr(obj, attr, 0)
    if getattr(obj, 'seed', None) is None or not hasattr(obj, 'rng'):
        return
    obj._seed_key = key
    rng = obj.rng
    if isinstance(rng, torch.Generator):
        rng.manual_seed(stream_seed(obj))
    elif isinstance(rng, np.random.Generator):
        obj.rng = np.random.default_rng(stream_seed(obj))
