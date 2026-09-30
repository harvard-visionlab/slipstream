"""Progressive-resolution training (FFCV / lrm-ssl style).

    sched = ResolutionSchedule(min_res=160, max_res=192, start_ramp=65, end_ramp=76)
    loader = SlipstreamLoader(ds, ..., resolution_schedule=sched)
    loader.set_epoch(epoch)          # sets the crop size to sched(epoch)
    loader.set_resolution(176)       # or set it by hand

The schedule is the one in lrm-ssl ``train.get_resolution`` and ffcv-imagenet: ``min_res`` up to
and including ``start_ramp``, ``max_res`` from ``end_ramp`` on, and in between a linear
interpolation rounded to the nearest multiple of ``step``. Epochs are 0-based, as ``set_epoch``
counts them.

Only single-size random-crop stages are resized (they set ``resolution_schedulable = True`` and
expose ``set_size``): ``DecodeRandomResizedCrop``, ``DecodeDirectRandomResizedCrop``,
``DecodeYUVRandomResizedCrop``. Crop boxes are drawn in source-image coordinates before the resize,
so a given (seed, rank, epoch) yields the same crops at any resolution and resume stays exact.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ResolutionSchedule:
    """Crop size per epoch: ``min_res`` → ``max_res`` between ``start_ramp`` and ``end_ramp``."""

    min_res: int
    max_res: int
    start_ramp: int
    end_ramp: int
    step: int = 32

    def __post_init__(self) -> None:
        for name in ("min_res", "max_res", "start_ramp", "end_ramp", "step"):
            v = getattr(self, name)
            if not isinstance(v, (int, np.integer)) or isinstance(v, bool):
                raise TypeError(f"ResolutionSchedule.{name} must be an int, got {v!r}")
        if self.step <= 0:
            raise ValueError(f"step must be positive, got {self.step}")
        if not 0 < self.min_res <= self.max_res:
            raise ValueError(f"need 0 < min_res <= max_res, got {self.min_res}, {self.max_res}")
        if self.min_res % self.step or self.max_res % self.step:
            raise ValueError(f"min_res and max_res must be multiples of step={self.step}, "
                             f"got {self.min_res}, {self.max_res}")
        if not 0 <= self.start_ramp <= self.end_ramp:
            raise ValueError(f"need 0 <= start_ramp <= end_ramp, got {self.start_ramp}, {self.end_ramp}")

    def __call__(self, epoch: int) -> int:
        if epoch <= self.start_ramp:
            return int(self.min_res)
        if epoch >= self.end_ramp:
            return int(self.max_res)
        interp = np.interp([epoch], [self.start_ramp, self.end_ramp], [self.min_res, self.max_res])[0]
        return int(np.round(interp / self.step)) * int(self.step)
