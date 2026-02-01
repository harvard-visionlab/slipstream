"""Color jitter transforms (HSV, YIQ paths)."""

import numbers
import torch
from .base import BatchAugment
from . import functional as F
from . import functional_tensor as FT
from ._compat import mask_batch


def _check_input(value, name, center=1, bound=(0, float("inf")), clip_first_on_zero=True):
    """Validate and normalize color jitter parameter ranges."""
    if isinstance(value, numbers.Number):
        if value < 0:
            raise ValueError(f"If {name} is a single number, it must be non negative.")
        value = [center - value, center + value]
        if clip_first_on_zero:
            value[0] = max(value[0], 0)
    elif isinstance(value, (tuple, list)) and len(value) == 2:
        if not bound[0] <= value[0] <= value[1] <= bound[1]:
            raise ValueError(f"{name} values should be between {bound}")
    else:
        raise TypeError(f"{name} should be a single number or a list/tuple with length 2.")

    if value[0] == value[1] == center:
        value = None
    return value


class ColorJitter(BatchAugment):
    """Batch-level color jitter (HSV path via functional_tensor).

    When applied, either all images are jittered or none (with probability p),
    but each image gets different random parameters.
    """

    def __init__(self, p=1.0, hue=0.0, saturation=0.0, value=0.0, contrast=0.0, seed=None, device=None):
        self.p = p
        self.hue = _check_input(hue, "hue", center=0, bound=(-0.5, 0.5), clip_first_on_zero=False)
        self.saturation = _check_input(saturation, "saturation")
        self.value = _check_input(value, "value")
        self.contrast = _check_input(contrast, "contrast")
        self.seed = seed
        self.rng = None
        if self.seed is not None:
            self.rng = torch.Generator("cpu" if device is None else device)
            self.rng.manual_seed(self.seed)

    @staticmethod
    def sample_params(b, n, hue, saturation, value, contrast, generator):
        h = b.new_zeros(n).uniform_(hue[0], hue[1], generator=generator) if hue is not None else None
        s = b.new_zeros(n).uniform_(saturation[0], saturation[1], generator=generator) if saturation is not None else None
        v = b.new_zeros(n).uniform_(value[0], value[1], generator=generator) if value is not None else None
        c = b.new_zeros(n).uniform_(contrast[0], contrast[1], generator=generator) if contrast is not None else None
        return h, s, v, c

    def before_call(self, b, **kwargs):
        self.do = any(kwargs) or self.p == 1.0 or torch.rand(1, generator=self.rng).item() < self.p
        n = b.shape[0] if len(b.shape) == 4 else 1
        h, s, v, c = self.sample_params(b, n, self.hue, self.saturation, self.value, self.contrast, self.rng)
        self.h = kwargs.get("h", h)
        self.s = kwargs.get("s", s)
        self.v = kwargs.get("v", v)
        self.c = kwargs.get("c", c)

        if "mask" in kwargs:
            mask = kwargs["mask"]
            if self.h is not None: self.h[mask] = 0
            if self.s is not None: self.s[mask] = 1
            if self.v is not None: self.v[mask] = 1
            if self.c is not None: self.c[mask] = 1

    def last_params(self):
        return {"do": self.do, "h": self.h, "s": self.s, "v": self.v, "c": self.c}

    def apply_last(self, b):
        params = self.last_params()
        return FT.color_jitter(b, params["h"], params["s"], params["v"], params["c"]) if params["do"] else b

    def __call__(self, b, **kwargs):
        self.before_call(b, **kwargs)
        return FT.color_jitter(b, self.h, self.s, self.v, self.c) if self.do else b

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(p={self.p}, hue={self.hue}, "
            f"saturation={self.saturation}, value={self.value}, contrast={self.contrast})"
        )


class RandomColorJitter(BatchAugment):
    """Per-image random color jitter (HSV path via functional_tensor).

    Each image is independently jittered with probability p.
    """

    def __init__(self, p=1.0, hue=0.0, saturation=0.0, value=0.0, contrast=0.0, seed=None, device=None):
        self.p = p
        self.hue = _check_input(hue, "hue", center=0, bound=(-0.5, 0.5), clip_first_on_zero=False)
        self.saturation = _check_input(saturation, "saturation")
        self.value = _check_input(value, "value")
        self.contrast = _check_input(contrast, "contrast")
        self.seed = seed
        self.rng = None
        if self.seed is not None:
            self.rng = torch.Generator("cpu" if device is None else device)
            self.rng.manual_seed(self.seed)

    @staticmethod
    def sample_params(b, n, hue, saturation, value, contrast, generator):
        h = b.new_zeros(n).uniform_(hue[0], hue[1], generator=generator) if hue is not None else None
        s = b.new_zeros(n).uniform_(saturation[0], saturation[1], generator=generator) if saturation is not None else None
        v = b.new_zeros(n).uniform_(value[0], value[1], generator=generator) if value is not None else None
        c = b.new_zeros(n).uniform_(contrast[0], contrast[1], generator=generator) if contrast is not None else None
        return h, s, v, c

    def before_call(self, b, **kwargs):
        self.do, self.idx = mask_batch(b, p=self.p, rng=self.rng)
        n = len(self.idx)
        h, s, v, c = self.sample_params(b, n, self.hue, self.saturation, self.value, self.contrast, self.rng)
        self.h = h.to(b.device, non_blocking=True) if h is not None else h
        self.s = s.to(b.device, non_blocking=True) if s is not None else s
        self.v = v.to(b.device, non_blocking=True) if v is not None else v
        self.c = c.to(b.device, non_blocking=True) if c is not None else c

    def last_params(self):
        return {"do": self.do, "idx": self.idx, "h": self.h, "s": self.s, "v": self.v, "c": self.c}

    def apply_last(self, b):
        params = self.last_params()
        return FT.random_color_jitter(b, params["idx"], params["h"], params["s"], params["v"], params["c"])

    def __call__(self, b, **kwargs):
        self.before_call(b, **kwargs)
        return FT.random_color_jitter(b, self.idx, self.h, self.s, self.v, self.c)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(p={self.p}, hue={self.hue}, saturation={self.saturation}, "
            f"value={self.value}, contrast={self.contrast}, seed={self.seed})"
        )


# TODO: Review YIQ jitter path (RandomColorJitterYIQ) for correctness. HSV path verified correct.
class RandomColorJitterHSV(RandomColorJitter):
    """Alias for RandomColorJitter (HSV path)."""
    pass


class RandomColorJitterYIQ(BatchAugment):
    """Per-image color jitter via YIQ color space transform.

    Combines hue, saturation, value, brightness, and contrast in a single
    matrix multiplication for efficiency.
    """

    def __init__(self, p=0.80, hue=0.0, saturation=0.0, value=0.0, brightness=0.0, contrast=0.0,
                 seed=None, device=None):
        self.p = p
        self.hue = _check_input(hue, "hue", center=0, bound=(-180.0, 180.0), clip_first_on_zero=False)
        self.saturation = _check_input(saturation, "saturation")
        self.value = _check_input(value, "value")
        self.brightness = _check_input(brightness, "brightness")
        self.contrast = _check_input(contrast, "contrast")
        self.seed = seed
        self.rng = None
        self._init_rng(device)

    def _init_rng(self, device):
        if self.seed is not None and device is not None and self.rng is None:
            self.rng = torch.Generator(device)
            self.rng.manual_seed(self.seed)

    def sample_params(self, b, n, hue, saturation, value, brightness, contrast):
        h = b.new_zeros(n).uniform_(hue[0], hue[1], generator=self.rng) if hue is not None else b.new_zeros(n)
        s = b.new_zeros(n).uniform_(saturation[0], saturation[1], generator=self.rng) if saturation is not None else b.new_ones(n)
        v = b.new_zeros(n).uniform_(value[0], value[1], generator=self.rng) if value is not None else b.new_ones(n)
        br = b.new_zeros(n).uniform_(brightness[0], brightness[1], generator=self.rng) if brightness is not None else b.new_ones(n)
        c = b.new_zeros(n).uniform_(contrast[0], contrast[1], generator=self.rng) if contrast is not None else b.new_ones(n)
        return h, s, v, br, c

    def before_call(self, batch, **kwargs):
        self._init_rng(batch.device)
        self.do, self.idx = mask_batch(batch, p=self.p, rng=self.rng)
        n = len(self.idx)
        h, s, v, b, c = self.sample_params(
            batch, n, self.hue, self.saturation, self.value, self.brightness, self.contrast
        )
        self.h = h
        self.s = s
        self.v = v
        self.b = b
        self.c = c

    def last_params(self):
        return {"do": self.do, "idx": self.idx, "h": self.h, "s": self.s, "v": self.v, "b": self.b, "c": self.c}

    def apply_last(self, b):
        params = self.last_params()
        return F.random_hsv_jitter2(b, params["idx"], params["h"], params["s"], params["v"], params["b"], params["c"])

    def __call__(self, b, **kwargs):
        self.before_call(b, **kwargs)
        return F.random_hsv_jitter2(b, self.idx, self.h, self.s, self.v, self.b, self.c)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(p={self.p}, hue={self.hue}, saturation={self.saturation}, "
            f"value={self.value}, brightness={self.brightness}, contrast={self.contrast}, seed={self.seed})"
        )
