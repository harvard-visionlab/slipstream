"""Type and device conversion transforms."""

import torch
from .base import BatchAugment
from . import functional as F


class ToTorchImage(BatchAugment):
    """Convert HWC/BHWC tensor to CHW/BCHW float tensor in [0,1]."""

    def __init__(self, device, dtype=torch.float32, from_numpy=True):
        self.device = device
        self.dtype = torch.__dict__.get(dtype, torch.float32) if isinstance(dtype, str) else dtype
        self.from_numpy = from_numpy

    def apply_last(self, x):
        return F.to_torch_image(x, device=self.device, dtype=self.dtype)

    def __call__(self, x, **kwargs):
        return self.apply_last(x)

    def __repr__(self):
        return f"{self.__class__.__name__}(device={self.device}, dtype={self.dtype}, from_numpy={self.from_numpy})"


class ToNumpy(BatchAugment):
    """Convert tensor to numpy array."""

    def apply_last(self, x):
        if isinstance(x, torch.Tensor):
            return x.cpu().numpy()
        return x

    def __call__(self, x, **kwargs):
        return self.apply_last(x)


class ToChannelsFirst(BatchAugment):
    """Convert BHWC to BCHW."""

    def apply_last(self, x):
        return F.to_channels_first(x)

    def __call__(self, x, **kwargs):
        return self.apply_last(x)


class ToChannelsLast(BatchAugment):
    """Convert BCHW to BHWC."""

    def apply_last(self, x):
        return F.to_channels_last(x)

    def __call__(self, x, **kwargs):
        return self.apply_last(x)


class ToDevice(BatchAugment):
    """Move tensor to device."""

    def __init__(self, device):
        self.device = device

    def apply_last(self, x):
        return F.to_device(x, self.device)

    def __call__(self, x, **kwargs):
        return self.apply_last(x)

    def __repr__(self):
        return f"{self.__class__.__name__}(device='{self.device}')"


class ToFloat(BatchAugment):
    """Convert tensor to float using .float()."""

    def __init__(self, value=None):
        self.value = float(value) if value is not None else None

    def apply_last(self, x):
        return F.to_float(x)

    def __call__(self, x, **kwargs):
        return self.apply_last(x)


class ToFloatDiv(BatchAugment):
    """Convert tensor to float using division."""

    def __init__(self, value, dtype=torch.float32):
        self.value = float(value)
        self.dtype = dtype

    def apply_last(self, x):
        return F.div_(x, val=self.value, dtype=self.dtype)

    def __call__(self, x, **kwargs):
        return self.apply_last(x)

    def __repr__(self):
        return f"{self.__class__.__name__}(value={self.value}, dtype={self.dtype})"
