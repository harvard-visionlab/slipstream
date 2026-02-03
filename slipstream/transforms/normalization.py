"""Normalization transforms and ImageNet constants."""

import torch
from .base import BatchAugment
from . import functional as F

default_device = "cuda" if torch.cuda.is_available() else "cpu"

# ImageNet channel-wise statistics
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class Normalize(BatchAugment):
    """x = (x - mean) / std"""

    def __init__(self, mean, std, inplace=True, dtype=torch.float32, device=default_device):
        dtype = torch.__dict__.get(dtype, torch.float32) if isinstance(dtype, str) else dtype
        self.mean = torch.tensor(mean).to(device, dtype, non_blocking=True)
        self.std = torch.tensor(std).to(device, dtype, non_blocking=True)
        self.inplace = inplace
        self.dtype = dtype
        self.device = device

    def _op(self, b):
        if self.mean.device != b.device:
            self.mean = self.mean.to(b.device, non_blocking=True)
        if self.std.device != b.device:
            self.std = self.std.to(b.device, non_blocking=True)
        if b.dtype != self.dtype:
            b = b.to(self.dtype)
        return F.normalize(b, self.mean, self.std, self.inplace)

    def apply_last(self, b):
        return self._op(b)

    def __call__(self, b, **kwargs):
        return self._op(b)

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std}, inplace={self.inplace})"


class NormalizeLGN(Normalize):
    """Normalize with default LGN 5-channel imagenet stats."""

    def __init__(
        self,
        mean=(0.2, 0.2, 0.2, 0.2, 0.2),
        std=(0.2, 0.2, 0.2, 0.2, 0.2),
        inplace=True,
        dtype=torch.float32,
        device=default_device,
    ):
        super().__init__(mean=mean, std=std, inplace=inplace, dtype=dtype, device=device)
