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
        self.sf = torch.empty(len(self.idx), device=b.device, dtype=b.dtype).uniform_(
            self.scale_range[0], self.scale_range[1], generator=self.rng
        )

    def last_params(self):
        return {"do": self.do, "idx": self.idx, "sf": self.sf}

    def apply_last(self, b):
        params = self.last_params()
        return F.random_adjust_brightness(
            b, scale_factor=params["sf"], idx=params["idx"], max_value=self.max_value
        )

    def __call__(self, b, **kwargs):
        n = b.shape[0] if b.ndim == 4 else 1
        sf = torch.empty(n, device=b.device, dtype=b.dtype).uniform_(
            self.scale_range[0], self.scale_range[1], generator=self.rng
        )

        if self.p < 1.0:
            # Partial application: mask unselected images to scale_factor=1.0
            mask = torch.empty(n, device=b.device, dtype=b.dtype).bernoulli_(self.p, generator=self.rng)
            sf = sf * mask + (1.0 - mask)

        # Store params for replay
        self.do = torch.ones(n, device=b.device, dtype=b.dtype) if self.p == 1.0 else mask
        self.idx = torch.where(self.do)[0] if self.p < 1.0 else torch.arange(n, device=b.device)
        self.sf = sf[self.idx] if self.p < 1.0 else sf

        # Direct multiply — no index_select/index_copy
        if b.ndim == 4:
            out = (b * sf[:, None, None, None]).clamp_(0, self.max_value)
        else:
            out = (b * sf).clamp_(0, self.max_value)
        return out

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
        self.sf = torch.empty(len(self.idx), device=b.device, dtype=b.dtype).uniform_(
            self.scale_range[0], self.scale_range[1], generator=self.rng
        )

    def last_params(self):
        return {"do": self.do, "idx": self.idx, "sf": self.sf}

    def apply_last(self, b):
        params = self.last_params()
        return F.random_adjust_contrast(
            b, scale_factor=params["sf"], idx=params["idx"], max_value=self.max_value
        )

    def __call__(self, b, **kwargs):
        n = b.shape[0] if b.ndim == 4 else 1
        sf = torch.empty(n, device=b.device, dtype=b.dtype).uniform_(
            self.scale_range[0], self.scale_range[1], generator=self.rng
        )

        if self.p < 1.0:
            mask = torch.empty(n, device=b.device, dtype=b.dtype).bernoulli_(self.p, generator=self.rng)
            sf = sf * mask + (1.0 - mask)  # unselected → sf=1.0 → identity

        # Store params for replay
        self.do = torch.ones(n, device=b.device, dtype=b.dtype) if self.p == 1.0 else mask
        self.idx = torch.where(self.do)[0] if self.p < 1.0 else torch.arange(n, device=b.device)
        self.sf = sf[self.idx] if self.p < 1.0 else sf

        # Direct contrast: mean*(1-sf) + x*sf, no index_select/index_copy
        if b.ndim == 4:
            # Per-image grayscale mean
            mean = F.to_grayscale(b, num_output_channels=1).mean(dim=(-3, -2, -1), keepdim=True)
            sf4 = sf[:, None, None, None]
            out = (mean * (1.0 - sf4) + b * sf4).clamp_(0, self.max_value)
        else:
            mean = F.to_grayscale(b, num_output_channels=1).mean()
            out = (mean * (1.0 - sf) + b * sf).clamp_(0, self.max_value)
        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(p={self.p}, scale_range={self.scale_range}, "
            f"max_value={self.max_value}, seed={self.seed})"
        )
