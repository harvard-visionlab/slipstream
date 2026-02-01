"""Tensor-only mask utilities for batch augmentations."""

import torch


def mask_batch(b: torch.Tensor, p=0.5, rng=None):
    """Return (do_mask, indices) for randomly selecting images in a batch."""
    n = b.shape[0] if (hasattr(b, "shape") and len(b.shape) == 4) else 1
    do = mask_tensor(b.new_ones(n), p=p, rng=rng)
    idx = torch.where(do)[0]
    return do, idx


def mask_tensor(x, p=0.5, neutral=0.0, rng=None):
    """Mask elements of `x` with `neutral` with probability `1-p`."""
    if p == 1.0:
        return x
    if neutral != 0:
        x.add_(-neutral)
    mask = x.new_empty(*x.size()).bernoulli_(p, generator=rng)
    x.mul_(mask)
    return x.add_(neutral) if neutral != 0 else x
