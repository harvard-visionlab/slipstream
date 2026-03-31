"""Dataset preparation for slipstream cache building.

Provides dataset classes with preprocessing baked into __getitem__,
ready for use with ``OptimizedCache.build()``.

Usage::

    from slipstream.prep import ImageNet1k_s256_l512
    from slipstream.cache import OptimizedCache

    dataset = ImageNet1k_s256_l512("/path/to/imagenet", split="val")
    cache = OptimizedCache.build(dataset, num_workers=8)
"""
from .imagenet import ImageNet1k_s256_l512

__all__ = ['ImageNet1k_s256_l512']
