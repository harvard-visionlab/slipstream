"""Geometric transforms: flip, rotate, zoom."""

import torch
from .base import BatchAugment
from . import functional as F
from ._compat import mask_tensor


class RandomHorizontalFlip(BatchAugment):
    """Flip images horizontally with probability p (per-image).

    Uses torch.flip for the actual operation (exact pixel copy, no interpolation)
    instead of the affine grid_sample path. The affine matrix is still computed
    and stored for compose/replay compatibility.
    """

    def __init__(self, p=0.5, seed=None, device=None):
        self.p = p
        self.seed = seed
        self.rng = None
        self.device = device
        if self.seed is not None:
            self.rng = torch.Generator("cpu" if self.device is None else self.device)
            self.rng.manual_seed(self.seed)

    def before_call(self, b, **kwargs):
        n = b.shape[0] if (hasattr(b, "shape") and len(b.shape) == 4) else 1
        self.do = self._expand(mask_tensor(b.new_ones(self._ng(n)), p=self.p, rng=self.rng), n)

        if "flip_val" in kwargs:
            flip_val = kwargs["flip_val"].to(b.device, dtype=b.dtype, non_blocking=True)
            self.do = (flip_val == -1).float()

    def last_params(self):
        return {"do": self.do}

    @staticmethod
    def _apply_flip(b, do):
        """Apply per-image horizontal flip in-place using masked indexing.

        For single images (B=1 or 3D), avoids boolean indexing overhead
        by using torch.flip directly. For batches, uses in-place masked
        assignment which is efficient when many images need flipping.
        """
        if b.ndim == 3:
            return torch.flip(b, [-1]) if do[0] > 0.5 else b
        if b.shape[0] == 1:
            return torch.flip(b, [-1]) if do[0] > 0.5 else b
        mask = do > 0.5
        if mask.any():
            b[mask] = b[mask].flip(-1)
        return b

    def apply_last(self, b):
        return self._apply_flip(b, self.last_params()["do"])

    def __call__(self, b, **kwargs):
        self.before_call(b, **kwargs)
        return self._apply_flip(b, self.do)

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p}, seed={self.seed})"


class RandomRotate(BatchAugment):
    """Randomly rotate images around a point."""

    def __init__(self, p=0.5, max_deg=45, x=0.5, y=0.5, angles=None, pad_mode="zeros",
                 seed=None, device=None):
        self.p = p
        self.max_deg = max_deg
        self.angles = torch.tensor(angles).float() if angles is not None else None
        self.x_range = (x, x) if isinstance(x, (int, float)) else tuple(x)
        self.y_range = (y, y) if isinstance(y, (int, float)) else tuple(y)
        self.pad_mode = pad_mode
        self.seed = seed
        self.rng = None
        if self.seed is not None:
            self.rng = torch.Generator("cpu" if device is None else device)
            self.rng.manual_seed(self.seed)

    def before_call(self, b, **kwargs):
        n = b.shape[0] if (hasattr(b, "shape") and len(b.shape) == 4) else 1
        ng = self._ng(n)
        self.do = self._expand(mask_tensor(b.new_ones(ng), p=self.p, rng=self.rng), n)

        if self.angles is not None:
            if self.angles.device != b.device:
                self.angles = self.angles.to(b.device)
            self.deg = self._expand(self.angles[
                torch.randint(0, len(self.angles), (ng,), generator=self.rng, device=b.device)
            ], n) * self.do
        else:
            self.deg = self._expand(b.new(ng).uniform_(-self.max_deg, self.max_deg, generator=self.rng), n) * self.do

        self.xs = self._expand(b.new(ng).uniform_(self.x_range[0], self.x_range[1], generator=self.rng), n)
        self.ys = self._expand(b.new(ng).uniform_(self.y_range[0], self.y_range[1], generator=self.rng), n)
        self.mat = F._prepare_mat(b, F.rotate_mat(self.deg, self.xs, self.ys))

        if any(kwargs):
            if "deg" in kwargs:
                self.deg = kwargs["deg"].to(b.device, non_blocking=True)
            if "xs" in kwargs:
                self.xs = kwargs["xs"].to(b.device, non_blocking=True)
            if "ys" in kwargs:
                self.ys = kwargs["ys"].to(b.device, non_blocking=True)
            self.do = (self.deg.abs() != 0).float()
            self.mat = F._prepare_mat(b, F.rotate_mat(self.deg, self.xs, self.ys))

    def last_params(self):
        return {"do": self.do, "deg": self.deg, "xs": self.xs, "ys": self.ys, "mat": self.mat}

    def apply_last(self, b):
        return F.affine_transform(b, self.last_params()["mat"], pad_mode=self.pad_mode)

    def __call__(self, b, **kwargs):
        self.before_call(b, **kwargs)
        return F.affine_transform(b, self.mat, pad_mode=self.pad_mode)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(p={self.p}, max_deg={self.max_deg}, "
            f"angles={self.angles}, x_range={self.x_range}, y_range={self.y_range}, "
            f"pad_mode='{self.pad_mode}')"
        )


class RandomZoom(BatchAugment):
    """Randomly zoom images around a point."""

    def __init__(self, p=0.5, zoom=(0.5, 1.0), x=0.5, y=0.5, pad_mode="zeros",
                 seed=None, device=None):
        self.p = p
        self.zoom_range = (zoom, zoom) if isinstance(zoom, (int, float)) else tuple(zoom)
        self.x_range = (x, x) if isinstance(x, (int, float)) else tuple(x)
        self.y_range = (y, y) if isinstance(y, (int, float)) else tuple(y)
        self.pad_mode = pad_mode
        self.seed = seed
        self.rng = None
        if self.seed is not None:
            self.rng = torch.Generator("cpu" if device is None else device)
            self.rng.manual_seed(self.seed)

    def before_call(self, b, **kwargs):
        n = b.shape[0] if (hasattr(b, "shape") and len(b.shape) == 4) else 1
        ng = self._ng(n)
        self.do = self._expand(mask_tensor(b.new_ones(ng), p=self.p, rng=self.rng), n)
        self.zoom = (
            self._expand(b.new(ng).uniform_(self.zoom_range[0], self.zoom_range[1], generator=self.rng), n) * self.do
            + (1 - self.do)
        )
        self.xs = self._expand(b.new(ng).uniform_(self.x_range[0], self.x_range[1], generator=self.rng), n)
        self.ys = self._expand(b.new(ng).uniform_(self.y_range[0], self.y_range[1], generator=self.rng), n)
        self.mat = F._prepare_mat(b, F.zoom_mat(self.zoom, self.xs, self.ys))

        if any(kwargs):
            if "zoom" in kwargs:
                self.zoom = kwargs["zoom"].to(b.device, non_blocking=True)
            if "xs" in kwargs:
                self.xs = kwargs["xs"].to(b.device, non_blocking=True)
            if "ys" in kwargs:
                self.ys = kwargs["ys"].to(b.device, non_blocking=True)
            self.do = (self.zoom.abs() != 1.0).float()
            self.mat = F._prepare_mat(b, F.zoom_mat(self.zoom, self.xs, self.ys))

    def last_params(self):
        return {"do": self.do, "zoom": self.zoom, "xs": self.xs, "ys": self.ys, "mat": self.mat}

    def apply_last(self, b):
        return F.affine_transform(b, self.last_params()["mat"], pad_mode=self.pad_mode)

    def __call__(self, b, **kwargs):
        self.before_call(b, **kwargs)
        return F.affine_transform(b, self.mat, pad_mode=self.pad_mode)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(p={self.p}, zoom={self.zoom_range}, "
            f"x={self.x_range}, y={self.y_range}, pad_mode='{self.pad_mode}')"
        )


class RandomRotateObject(BatchAugment):
    """Randomly rotate, rescale, and reposition an object."""

    def __init__(self, p=0.5, max_deg=45, ctr_x=0.5, ctr_y=0.5, scale=(1.0, 2.0),
                 dest_x=(0.25, 0.75), dest_y=(0.25, 0.75), pad_mode="border",
                 seed=None, device=None):
        self.p = p
        self.max_deg = max_deg
        self.cx_range = (ctr_x, ctr_x) if isinstance(ctr_x, (int, float)) else tuple(ctr_x)
        self.cy_range = (ctr_y, ctr_y) if isinstance(ctr_y, (int, float)) else tuple(ctr_y)
        self.scale_range = (scale, scale) if isinstance(scale, (int, float)) else tuple(scale)
        self.destx_range = (dest_x, dest_x) if isinstance(dest_x, (int, float)) else tuple(dest_x)
        self.desty_range = (dest_y, dest_y) if isinstance(dest_y, (int, float)) else tuple(dest_y)
        self.pad_mode = pad_mode
        self.seed = seed
        self.rng = None
        if self.seed is not None:
            self.rng = torch.Generator("cpu" if device is None else device)
            self.rng.manual_seed(self.seed)

    def before_call(self, b, **kwargs):
        n = b.shape[0] if (hasattr(b, "shape") and len(b.shape) == 4) else 1
        ng = self._ng(n)
        u = lambda lo, hi: self._expand(b.new(ng).uniform_(lo, hi, generator=self.rng), n)
        self.do = self._expand(mask_tensor(b.new_ones(ng), p=self.p, rng=self.rng), n)
        self.deg = u(-self.max_deg, self.max_deg) * self.do
        self.xs = u(self.cx_range[0], self.cx_range[1]) * self.do + (1 - self.do) * 0.5
        self.ys = u(self.cy_range[0], self.cy_range[1]) * self.do + (1 - self.do) * 0.5
        self.scale = u(self.scale_range[0], self.scale_range[1]) * self.do + (1 - self.do)
        self.dest_x = u(self.destx_range[0], self.destx_range[1]) * self.do + (1 - self.do) * 0.5
        self.dest_y = u(self.desty_range[0], self.desty_range[1]) * self.do + (1 - self.do) * 0.5

        mat = F.rotate_object_mat(self.deg, self.xs, self.ys, self.scale, self.dest_x, self.dest_y)
        self.mat = F._prepare_mat(b, mat)

        if any(kwargs):
            if "deg" in kwargs:
                self.deg = kwargs["deg"].to(b.device, non_blocking=True)
            if "xs" in kwargs:
                self.xs = kwargs["xs"].to(b.device, non_blocking=True)
            if "ys" in kwargs:
                self.ys = kwargs["ys"].to(b.device, non_blocking=True)
            if "scale" in kwargs:
                self.scale = kwargs["scale"].to(b.device, non_blocking=True)
            if "dest_x" in kwargs:
                self.dest_x = kwargs["dest_x"].to(b.device, non_blocking=True)
            if "dest_y" in kwargs:
                self.dest_y = kwargs["dest_y"].to(b.device, non_blocking=True)
            self.do = (self.deg.abs() != 0).float()
            mat = F.rotate_object_mat(self.deg, self.xs, self.ys, self.scale, self.dest_x, self.dest_y)
            self.mat = F._prepare_mat(b, mat)

    def last_params(self):
        return {"do": self.do, "deg": self.deg, "xs": self.xs, "ys": self.ys, "mat": self.mat}

    def apply_last(self, b):
        return F.affine_transform(b, self.last_params()["mat"], pad_mode=self.pad_mode)

    def __call__(self, b, **kwargs):
        self.before_call(b, **kwargs)
        return F.affine_transform(b, self.mat, pad_mode=self.pad_mode)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(p={self.p}, max_deg={self.max_deg}, "
            f"ctr_x={self.cx_range}, ctr_y={self.cy_range}, scale={self.scale_range},\n"
            f"\t\t      dest_x={self.destx_range}, dest_y={self.desty_range}, "
            f"pad_mode='{self.pad_mode}')"
        )


class RandomResizedCropBatch(BatchAugment):
    """Per-image random resized crop on decoded tensors (torchvision semantics, batched).

    For already-decoded frames (e.g. from ``DecodeVideoWindow``) where the JPEG
    decode-time crops do not apply. Samples an area fraction in ``scale`` and an
    aspect ratio in ``ratio`` per image (analytic, no rejection loop), places the
    crop uniformly inside the image and resamples it to ``size`` with bilinear
    ``grid_sample``. uint8 input gives uint8 output. Honours ``seed_repeat``
    (windows share one crop) and supports replay via ``apply_last``.
    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0), seed=None, device=None):
        self.size = (size, size) if isinstance(size, int) else tuple(size)
        self.scale = tuple(scale)
        self.log_ratio = (float(torch.log(torch.tensor(ratio[0]))), float(torch.log(torch.tensor(ratio[1]))))
        self.seed = seed
        self.rng = None
        if self.seed is not None:
            self.rng = torch.Generator("cpu" if device is None else device)
            self.rng.manual_seed(self.seed)

    def before_call(self, b, **kwargs):
        n = b.shape[0] if b.ndim == 4 else 1
        ng = self._ng(n)
        u = lambda lo, hi: self._expand(torch.empty(ng).uniform_(lo, hi, generator=self.rng), n)
        area = u(self.scale[0], self.scale[1])
        ratio = torch.exp(u(self.log_ratio[0], self.log_ratio[1]))
        w = torch.sqrt(area * ratio).clamp_(max=1.0)          # crop width as a fraction of W
        h = torch.sqrt(area / ratio).clamp_(max=1.0)          # crop height as a fraction of H
        cx = u(0.0, 1.0) * (1 - w) + w / 2                     # centre, fraction of W
        cy = u(0.0, 1.0) * (1 - h) + h / 2
        self.params = {"w": w, "h": h, "cx": cx, "cy": cy}
        # normalized affine: x_in = w * x_out + (2*cx - 1)
        mat = torch.zeros(n, 2, 3)
        mat[:, 0, 0] = w; mat[:, 1, 1] = h
        mat[:, 0, 2] = 2 * cx - 1; mat[:, 1, 2] = 2 * cy - 1
        self.mat = mat.to(b.device)

    def last_params(self):
        return {"mat": self.mat, **self.params}

    def apply_last(self, b):
        squeeze = b.ndim == 3
        x = b.unsqueeze(0) if squeeze else b
        orig_dtype = x.dtype
        xf = x if x.is_floating_point() else x.float()
        grid = torch.nn.functional.affine_grid(self.mat.to(xf.dtype), (xf.shape[0], xf.shape[1], *self.size), align_corners=False)
        out = torch.nn.functional.grid_sample(xf, grid, mode="bilinear", padding_mode="border", align_corners=False)
        if not orig_dtype.is_floating_point:
            out = out.round_().clamp_(0, 255).to(orig_dtype)
        return out.squeeze(0) if squeeze else out

    def __call__(self, b, **kwargs):
        self.before_call(b, **kwargs)
        return self.apply_last(b)

    def __repr__(self):
        return f"{self.__class__.__name__}(size={self.size}, scale={self.scale}, seed={self.seed})"
