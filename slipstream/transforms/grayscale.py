"""Grayscale transforms."""

import torch
from .base import BatchAugment
from . import functional as F
from . import functional_tensor as FT
from ._compat import mask_batch


class ToGrayscale(BatchAugment):
    """Convert image to grayscale using functional.to_grayscale."""

    def __init__(self, num_output_channels=1):
        self.num_output_channels = num_output_channels

    def apply_last(self, b):
        return F.to_grayscale(b, num_output_channels=self.num_output_channels)

    def __call__(self, b, **kwargs):
        return self.apply_last(b)

    def __repr__(self):
        return f"{self.__class__.__name__}(num_output_channels={self.num_output_channels})"


class ToGrayscaleTorch(BatchAugment):
    """Convert image to grayscale using functional_tensor.rgb_to_grayscale."""

    def __init__(self, num_output_channels=1):
        self.num_output_channels = num_output_channels

    def apply_last(self, b):
        return FT.rgb_to_grayscale(b, num_output_channels=self.num_output_channels)

    def __call__(self, b, **kwargs):
        return self.apply_last(b)

    def __repr__(self):
        return f"{self.__class__.__name__}(num_output_channels={self.num_output_channels})"


class RandomGrayscale(BatchAugment):
    """Randomly convert images in a batch to grayscale."""

    def __init__(self, p=0.5, num_output_channels=3, seed=None, device=None):
        self.p = p
        self.num_output_channels = num_output_channels
        self.seed = seed
        self.rng = None
        if self.seed is not None:
            self.rng = torch.Generator("cpu" if device is None else device)
            self.rng.manual_seed(self.seed)

    def before_call(self, b, **kwargs):
        self.do, self.idx = mask_batch(b, p=self.p, rng=self.rng)

    def last_params(self):
        return {"do": self.do, "idx": self.idx}

    def apply_last(self, b):
        params = self.last_params()
        return F.random_grayscale(b, params["idx"], num_output_channels=self.num_output_channels)

    def __call__(self, b, **kwargs):
        self.before_call(b, **kwargs)
        return F.random_grayscale(b, self.idx, num_output_channels=self.num_output_channels)

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p}, num_output_channels={self.num_output_channels}, seed={self.seed})"
