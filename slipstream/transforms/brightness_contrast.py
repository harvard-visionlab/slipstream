"""Random brightness and contrast transforms."""

import torch
from .base import BatchAugment
from . import functional as F
from ._compat import mask_batch


class RandomBrightness(BatchAugment):
    """Randomly adjust brightness with per-image scale factors.

    new_img = img * scale_factor
    """

    def __init__(self, p=1.0, scale_range=(0.6, 1.4), max_value=1.0, seed=None, device=None):
        self.scale_range = scale_range
        self.p = p
        self.max_value = max_value
        self.seed = seed
        self.rng = None
        if self.seed is not None:
            self.rng = torch.Generator("cpu" if device is None else device)
            self.rng.manual_seed(self.seed)

    def before_call(self, b, **kwargs):
        self.do, self.idx = mask_batch(b, p=self.p, rng=self.rng)
        self.sf = torch.FloatTensor(len(self.idx)).uniform_(
            self.scale_range[0], self.scale_range[1], generator=self.rng
        )

    def last_params(self):
        return {"do": self.do, "idx": self.idx, "sf": self.sf}

    def apply_last(self, b):
        params = self.last_params()
        return F.random_adjust_brightness(
            b, scale_factor=params["sf"], idx=params["idx"], max_value=self.max_value
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(p={self.p}, scale_range={self.scale_range}, "
            f"max_value={self.max_value}, seed={self.seed})"
        )


class RandomContrast(BatchAugment):
    """Randomly adjust contrast with per-image scale factors.

    new_img = (1-sf)*img.mean() + sf*img
    """

    def __init__(self, p=1.0, scale_range=(0.6, 1.4), max_value=1.0, seed=None, device=None):
        self.scale_range = scale_range
        self.p = p
        self.max_value = max_value
        self.seed = seed
        self.rng = None
        if self.seed is not None:
            self.rng = torch.Generator("cpu" if device is None else device)
            self.rng.manual_seed(self.seed)

    def before_call(self, b, **kwargs):
        self.do, self.idx = mask_batch(b, p=self.p, rng=self.rng)
        self.sf = torch.FloatTensor(len(self.idx)).uniform_(
            self.scale_range[0], self.scale_range[1], generator=self.rng
        )

    def last_params(self):
        return {"do": self.do, "idx": self.idx, "sf": self.sf}

    def apply_last(self, b):
        params = self.last_params()
        return F.random_adjust_contrast(
            b, scale_factor=params["sf"], idx=params["idx"], max_value=self.max_value
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(p={self.p}, scale_range={self.scale_range}, "
            f"max_value={self.max_value}, seed={self.seed})"
        )
