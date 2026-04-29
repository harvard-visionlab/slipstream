"""Random erasing transform.

Reference: 'Random Erasing Data Augmentation' (Zhong et al., 2017)
https://arxiv.org/pdf/1708.04896.pdf
"""

import math
import torch
from .base import BatchAugment


_VALID_MODES = ("zeros", "random_color_uniform", "random_color_pixel")
_VALID_FILLS = ("randn", "uniform")


def _as_range(x, name):
    if isinstance(x, (int, float)):
        return (float(x), float(x))
    if isinstance(x, (tuple, list)) and len(x) == 2:
        return (float(x[0]), float(x[1]))
    raise ValueError(f"{name} must be a scalar or a (min, max) pair, got {x!r}")


class RandomErasing(BatchAugment):
    """Randomly erase rectangular regions of images.

    Vectorized over the batch — no per-sample Python loop. Each image in the batch
    independently rolls its own Bernoulli(p) and, if selected, gets a unique
    rectangle (random area and aspect ratio within the configured ranges) erased.

    Args:
        p: probability of applying erasing to each image (per-sample).
        area_range: (min, max) erase area as fraction of image area. Scalar → fixed.
        aspect_range: (min, max) aspect ratio (h/w) of the erase rectangle.
            Sampled log-uniformly. Scalar → fixed.
        mode: shape of the fill content.
            - "zeros": fill with 0 (channel-constant).
            - "random_color_uniform": one random color per image (per-channel),
              broadcast across the rectangle.
            - "random_color_pixel": independent random value per pixel and channel.
        fill: distribution to draw fill values from.
            - "randn": standard normal (assumes inputs are mean/std-normalized).
              Not valid for uint8 inputs.
            - "uniform": uniform in [0, max_value] for float, [0, 256) for uint8.
            Ignored when mode="zeros".
        max_value: upper bound for fill="uniform" with float inputs (default 1.0).
            Ignored for uint8 (always 256) and for fill="randn".
        seed: RNG seed for reproducibility.
        device: device for the RNG generator (default: matches input at call time).
    """

    def __init__(
        self,
        p: float = 0.5,
        area_range=(0.02, 1.0 / 3.0),
        aspect_range=(0.3, 3.3),
        mode: str = "random_color_pixel",
        fill: str = "randn",
        max_value: float = 1.0,
        seed: int = None,
        device=None,
    ):
        if mode not in _VALID_MODES:
            raise ValueError(f"mode must be one of {_VALID_MODES}, got {mode!r}")
        if fill not in _VALID_FILLS:
            raise ValueError(f"fill must be one of {_VALID_FILLS}, got {fill!r}")

        self.p = float(p)
        self.area_range = _as_range(area_range, "area_range")
        self.aspect_range = _as_range(aspect_range, "aspect_range")
        self.log_aspect_range = (math.log(self.aspect_range[0]), math.log(self.aspect_range[1]))
        self.mode = mode
        self.fill = fill
        self.max_value = float(max_value)

        self.seed = seed
        self.rng = None
        self._rng_device = device
        if self.seed is not None and device is not None:
            self.rng = torch.Generator(device)
            self.rng.manual_seed(self.seed)

        self.do = None
        self.top = None
        self.left = None
        self.eh = None
        self.ew = None
        self.fill_tensor = None

    def _init_rng(self, device):
        if self.seed is not None and self.rng is None:
            self.rng = torch.Generator(device)
            self.rng.manual_seed(self.seed)

    def _sample_fill(self, shape, dtype, device):
        """Draw a fill tensor of `shape` matching input dtype/device."""
        if self.fill == "randn":
            if not dtype.is_floating_point:
                raise ValueError(
                    "fill='randn' requires float inputs. For uint8, use fill='uniform' "
                    "(yields randint(0, 256))."
                )
            return torch.randn(shape, generator=self.rng, dtype=dtype, device=device)
        # uniform
        if dtype == torch.uint8:
            return torch.randint(
                0, 256, shape, generator=self.rng, dtype=torch.uint8, device=device
            )
        return torch.empty(shape, dtype=dtype, device=device).uniform_(
            0.0, self.max_value, generator=self.rng
        )

    def before_call(self, b: torch.Tensor, **kwargs):
        self._init_rng(b.device)

        is_3d = b.ndim == 3
        if is_3d:
            C, H, W = b.shape
            N = 1
        else:
            N, C, H, W = b.shape

        device = b.device

        # Per-image Bernoulli mask: which images get erased.
        if self.p >= 1.0:
            do = torch.ones(N, device=device, dtype=torch.bool)
        elif self.p <= 0.0:
            do = torch.zeros(N, device=device, dtype=torch.bool)
        else:
            do = torch.empty(N, device=device, dtype=torch.float32).bernoulli_(
                self.p, generator=self.rng
            ).bool()

        # Per-image rectangle params, all shape [N], sampled in single vectorized calls.
        area_frac = torch.empty(N, device=device, dtype=torch.float32).uniform_(
            self.area_range[0], self.area_range[1], generator=self.rng
        )
        log_aspect = torch.empty(N, device=device, dtype=torch.float32).uniform_(
            self.log_aspect_range[0], self.log_aspect_range[1], generator=self.rng
        )
        aspect = log_aspect.exp()

        area = area_frac * float(H * W)
        eh = (area * aspect).sqrt().round().long().clamp_(1, H)
        ew = (area / aspect).sqrt().round().long().clamp_(1, W)

        # Sample top-left in [0, dim - extent], inclusive of the lower bound.
        u_top = torch.empty(N, device=device, dtype=torch.float32).uniform_(
            0.0, 1.0, generator=self.rng
        )
        u_left = torch.empty(N, device=device, dtype=torch.float32).uniform_(
            0.0, 1.0, generator=self.rng
        )
        top = (u_top * (H - eh + 1).float()).long().clamp_(0, H - 1)
        left = (u_left * (W - ew + 1).float()).long().clamp_(0, W - 1)

        # Build fill tensor (only if needed).
        if self.mode == "zeros":
            fill_tensor = None
        elif self.mode == "random_color_uniform":
            fill_tensor = self._sample_fill((N, C, 1, 1), b.dtype, device)
        else:  # random_color_pixel
            fill_tensor = self._sample_fill((N, C, H, W), b.dtype, device)

        self.do = do
        self.top = top
        self.left = left
        self.eh = eh
        self.ew = ew
        self.fill_tensor = fill_tensor
        self._is_3d = is_3d
        self._H = H
        self._W = W

    def last_params(self):
        return {
            "do": self.do, "top": self.top, "left": self.left,
            "eh": self.eh, "ew": self.ew, "fill_tensor": self.fill_tensor,
            "is_3d": self._is_3d, "H": self._H, "W": self._W,
        }

    def _apply(self, b, params):
        do = params["do"]
        if not do.any():
            return b

        H, W = params["H"], params["W"]
        device = b.device

        # Build per-image rectangle mask via broadcasting — no Python loop over N.
        rows = torch.arange(H, device=device).view(1, H, 1)   # [1, H, 1]
        cols = torch.arange(W, device=device).view(1, 1, W)   # [1, 1, W]

        top_b = params["top"].view(-1, 1, 1)
        bot_b = (params["top"] + params["eh"]).view(-1, 1, 1)
        left_b = params["left"].view(-1, 1, 1)
        right_b = (params["left"] + params["ew"]).view(-1, 1, 1)

        in_rect = (rows >= top_b) & (rows < bot_b) & (cols >= left_b) & (cols < right_b)  # [N,H,W]
        mask = (in_rect & do.view(-1, 1, 1)).unsqueeze(1)  # [N, 1, H, W]

        if params["is_3d"]:
            mask = mask.squeeze(0)  # [1, H, W]

        if self.mode == "zeros":
            fill = b.new_zeros(())  # scalar zero, broadcasts
        else:
            fill = params["fill_tensor"]
            if params["is_3d"]:
                # fill is [1, C, ...] — drop batch dim
                fill = fill.squeeze(0)

        return torch.where(mask, fill, b)

    def apply_last(self, b: torch.Tensor) -> torch.Tensor:
        return self._apply(b, self.last_params())

    def __call__(self, b: torch.Tensor, **kwargs) -> torch.Tensor:
        self.before_call(b, **kwargs)
        return self._apply(b, self.last_params())

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(p={self.p}, area_range={self.area_range}, "
            f"aspect_range={self.aspect_range}, mode={self.mode!r}, fill={self.fill!r}, "
            f"max_value={self.max_value}, seed={self.seed})"
        )
