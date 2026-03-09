"""RandomEmbed: embed an image onto a larger canvas with configurable background."""

import torch
from torch.nn.modules.utils import _pair

from .base import BatchAugment


def _even_ceil(n: int) -> int:
    """Round up to the next even integer."""
    return n if n % 2 == 0 else n + 1


class RandomEmbed(BatchAugment):
    """Embed an image at a random position on a larger canvas.

    Resize-free embedding: the input image is placed pixel-for-pixel onto a
    canvas of size ``canvas_size``.  The placement position is controllable
    via three mutually exclusive modes:

    1. **Range mode** (default): ``x_range`` / ``y_range`` in [0, 1] where
       0 → leftmost/topmost position that keeps the image fully inside the
       canvas, and 1 → rightmost/bottommost.  A scalar fixes the position;
       a ``(min, max)`` tuple enables uniform random sampling per image.
    2. **Coords mode**: a list of ``(x, y)`` center-pixel coordinates;
       one is sampled uniformly per image.  Partial clipping at edges is
       allowed.
    3. **Coords-dict mode**: ``{(H, W): [(x, y), ...]}`` — per-size
       coordinate lists, looked up by the input image dimensions.

    The canvas background is filled according to ``background``:

    - ``"zeros"``: all-zero canvas.
    - ``"mean"``: constant per-channel fill (e.g. ImageNet mean).
    - ``"power_law"``: 1/f^α colored noise in [0, 1].

    All backgrounds are generated in the [0, 1] value range.  When the
    input images are normalised (e.g. ImageNet normalisation), pass ``std``
    (and ``mean`` if not already set) so the background is normalised to
    match: ``(bg - mean) / std``.

    All random parameters are sampled **per image**, never per batch.

    Args:
        canvas_size: Output spatial size.  ``int`` for square, ``(H, W)``
            tuple for rectangular.  Channels are inferred from the input.
        x_range: Horizontal position in [0, 1].  Float or ``(min, max)``.
        y_range: Vertical position in [0, 1].  Float or ``(min, max)``.
        coords: List of ``(x, y)`` center pixel coordinates.
        coords_dict: Dict mapping ``(H, W)`` image-size tuples to lists of
            ``(x, y)`` center coordinates.
        background: One of ``"zeros"``, ``"mean"``, ``"power_law"``.
        mean: Fill value for ``"mean"`` background.  Float (all channels)
            or list of floats (one per channel).  Also used as the
            normalisation mean when ``std`` is provided.
        std: Per-channel standard deviation for normalisation.  When
            provided (together with ``mean``), the background is
            normalised as ``(bg - mean) / std`` so it matches images
            that have been normalised the same way.
        alpha_range: Power-law exponent for ``"power_law"`` background.
            Float or ``(min, max)`` for per-image sampling.
            α=0 white, α=1 pink, α=2 Brownian.
        color_noise: If ``True`` (default), generate independent noise per
            channel (colorful backgrounds).  If ``False``, duplicate a
            single noise field across channels (grayscale backgrounds).
            Only affects ``"power_law"`` background.
        fade_radius: Optional ``(inner, outer)`` tuple controlling circular
            fade.  Both values are fractions of ``min(H, W)`` of the input
            image.  Pixels within ``inner`` are fully opaque; pixels beyond
            ``outer`` are fully transparent (showing only background); the
            region between fades via a Gaussian falloff.  ``None`` (default)
            disables the fade and uses a hard rectangular boundary.
        seed: Random seed for reproducible positions and noise.
        device: Device for the RNG (must match input tensor device when
            using a seed).

    Example::

        # Center-embed 96×96 images on a 224×224 zero canvas
        embed = RandomEmbed(canvas_size=224)

        # Random position with Brownian noise background
        embed = RandomEmbed(
            canvas_size=224,
            x_range=(0, 1), y_range=(0, 1),
            background="power_law", alpha_range=2.0,
            seed=42,
        )

        # Same, but for ImageNet-normalised images
        from slipstream import IMAGENET_MEAN, IMAGENET_STD
        embed = RandomEmbed(
            canvas_size=224,
            background="power_law", alpha_range=2.0,
            mean=IMAGENET_MEAN, std=IMAGENET_STD,
        )
    """

    _VALID_BACKGROUNDS = ("zeros", "mean", "power_law")

    def __init__(
        self,
        canvas_size,
        # Position (mutually exclusive) ──────────────────────
        x_range=0.5,
        y_range=0.5,
        coords=None,
        coords_dict=None,
        # Background ─────────────────────────────────────────
        background="zeros",
        mean=None,
        std=None,
        alpha_range=2.0,
        color_noise=True,
        fade_radius=None,
        # Common ─────────────────────────────────────────────
        seed=None,
        device=None,
    ):
        self.canvas_size = _pair(canvas_size)  # (H, W)

        # ── position mode ──
        has_coords = coords is not None
        has_coords_dict = coords_dict is not None
        if has_coords and has_coords_dict:
            raise ValueError("coords and coords_dict are mutually exclusive")

        if has_coords_dict:
            self._position_mode = "coords_dict"
        elif has_coords:
            self._position_mode = "coords"
        else:
            self._position_mode = "range"

        self.x_range = (x_range, x_range) if isinstance(x_range, (int, float)) else tuple(x_range)
        self.y_range = (y_range, y_range) if isinstance(y_range, (int, float)) else tuple(y_range)
        self.coords = coords
        self.coords_dict = coords_dict

        # ── background ──
        if background not in self._VALID_BACKGROUNDS:
            raise ValueError(
                f"background must be one of {self._VALID_BACKGROUNDS}, got '{background}'"
            )
        if background == "mean" and mean is None:
            raise ValueError("'mean' is required for background='mean'")
        if std is not None and mean is None:
            raise ValueError(
                "'mean' is required when 'std' is provided (needed for "
                "normalisation: (bg - mean) / std)"
            )
        self.background = background
        self.mean = mean
        self.std = std
        self.color_noise = color_noise
        if fade_radius is not None:
            fade_radius = tuple(fade_radius)
            if len(fade_radius) != 2 or fade_radius[0] >= fade_radius[1]:
                raise ValueError(
                    "fade_radius must be (inner, outer) with inner < outer, "
                    f"got {fade_radius}"
                )
        self.fade_radius = fade_radius
        self.alpha_range = (
            (alpha_range, alpha_range) if isinstance(alpha_range, (int, float))
            else tuple(alpha_range)
        )

        # ── RNG ──
        self.seed = seed
        self.rng = None
        if self.seed is not None:
            self.rng = torch.Generator("cpu" if device is None else device)
            self.rng.manual_seed(self.seed)

        # ── stored params (set by before_call) ──
        self._xs = None
        self._ys = None
        self._x_frac = None
        self._y_frac = None
        self._img_h = None
        self._img_w = None

    # ------------------------------------------------------------------ #
    #  BatchAugment interface                                              #
    # ------------------------------------------------------------------ #

    def before_call(self, b, **kwargs):
        n = b.shape[0] if b.ndim == 4 else 1
        h, w = (b.shape[-2], b.shape[-1])
        M, N = self.canvas_size

        self._img_h = h
        self._img_w = w

        if self._position_mode == "range":
            x_frac = torch.empty(n, device=b.device, dtype=b.dtype).uniform_(
                self.x_range[0], self.x_range[1], generator=self.rng,
            )
            y_frac = torch.empty(n, device=b.device, dtype=b.dtype).uniform_(
                self.y_range[0], self.y_range[1], generator=self.rng,
            )
            slack_x = max(0, N - w)
            slack_y = max(0, M - h)
            self._xs = (x_frac * slack_x).long()
            self._ys = (y_frac * slack_y).long()
            self._x_frac = x_frac
            self._y_frac = y_frac

        elif self._position_mode == "coords":
            indices = torch.randint(
                0, len(self.coords), (n,), generator=self.rng,
            )
            centers = [self.coords[i.item()] for i in indices]
            cx = torch.tensor([c[0] for c in centers], dtype=torch.long)
            cy = torch.tensor([c[1] for c in centers], dtype=torch.long)
            self._xs = cx - w // 2
            self._ys = cy - h // 2
            self._x_frac = None
            self._y_frac = None

        elif self._position_mode == "coords_dict":
            key = (h, w)
            if key not in self.coords_dict:
                raise KeyError(
                    f"No coords for image size {key} in coords_dict. "
                    f"Available sizes: {list(self.coords_dict.keys())}"
                )
            valid = self.coords_dict[key]
            indices = torch.randint(
                0, len(valid), (n,), generator=self.rng,
            )
            centers = [valid[i.item()] for i in indices]
            cx = torch.tensor([c[0] for c in centers], dtype=torch.long)
            cy = torch.tensor([c[1] for c in centers], dtype=torch.long)
            self._xs = cx - w // 2
            self._ys = cy - h // 2
            self._x_frac = None
            self._y_frac = None

    def last_params(self):
        params = {
            "xs": self._xs,
            "ys": self._ys,
            "img_h": self._img_h,
            "img_w": self._img_w,
        }
        if self._x_frac is not None:
            params["x_frac"] = self._x_frac
            params["y_frac"] = self._y_frac
        return params

    def apply_last(self, b):
        expanded = False
        if b.ndim == 3:
            b = b.unsqueeze(0)
            expanded = True

        B, C, h, w = b.shape
        M, N = self.canvas_size

        canvas = self._generate_background(B, C, M, N, b.device, b.dtype)

        fade_mask = None  # lazily built on first use
        for i in range(B):
            x = self._xs[i].item()
            y = self._ys[i].item()

            # Source and destination regions (handles clipping)
            src_x0 = max(0, -x)
            src_y0 = max(0, -y)
            dst_x0 = max(0, x)
            dst_y0 = max(0, y)
            copy_w = min(w - src_x0, N - dst_x0)
            copy_h = min(h - src_y0, M - dst_y0)

            if copy_w > 0 and copy_h > 0:
                fg = b[i, :, src_y0:src_y0 + copy_h, src_x0:src_x0 + copy_w]
                if self.fade_radius is not None:
                    if fade_mask is None:
                        fade_mask = self._make_fade_mask(h, w, b.device, b.dtype)
                    m = fade_mask[:, :, src_y0:src_y0 + copy_h, src_x0:src_x0 + copy_w]
                    bg = canvas[i, :, dst_y0:dst_y0 + copy_h, dst_x0:dst_x0 + copy_w]
                    canvas[i, :, dst_y0:dst_y0 + copy_h, dst_x0:dst_x0 + copy_w] = \
                        m * fg + (1 - m) * bg
                else:
                    canvas[i, :, dst_y0:dst_y0 + copy_h, dst_x0:dst_x0 + copy_w] = fg

        if expanded:
            canvas = canvas.squeeze(0)
        return canvas

    # ------------------------------------------------------------------ #
    #  Circular fade mask                                                   #
    # ------------------------------------------------------------------ #

    def _make_fade_mask(self, h, w, device, dtype):
        """Build a circular fade mask of shape (1, 1, h, w).

        Pixels within ``r_inner`` are 1.0, beyond ``r_outer`` are 0.0,
        and the transition uses a Gaussian falloff (3-sigma rule).
        """
        dim = min(h, w)
        r_inner = self.fade_radius[0] * dim
        r_outer = self.fade_radius[1] * dim

        cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
        y = torch.arange(h, device=device, dtype=dtype) - cy
        x = torch.arange(w, device=device, dtype=dtype) - cx
        r = torch.sqrt(y[:, None] ** 2 + x[None, :] ** 2)

        sigma = (r_outer - r_inner) / 3.0
        t = (r - r_inner).clamp(min=0)
        mask = torch.exp(-0.5 * (t / sigma) ** 2)
        mask[r <= r_inner] = 1.0
        mask[r >= r_outer] = 0.0

        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, h, w)

    # ------------------------------------------------------------------ #
    #  Background generation                                               #
    # ------------------------------------------------------------------ #

    def _generate_background(self, B, C, M, N, device, dtype):
        if self.background == "zeros":
            canvas = torch.zeros(B, C, M, N, device=device, dtype=dtype)

        elif self.background == "mean":
            canvas = torch.empty(B, C, M, N, device=device, dtype=dtype)
            if isinstance(self.mean, (int, float)):
                canvas.fill_(self.mean)
            else:
                for c in range(min(C, len(self.mean))):
                    canvas[:, c].fill_(self.mean[c])

        elif self.background == "power_law":
            canvas = self._generate_power_law(B, C, M, N, device, dtype)

        else:
            raise ValueError(f"Unknown background: {self.background}")

        # Apply normalisation if std is provided: (bg - mean) / std
        if self.std is not None:
            mean_vals = self.mean if not isinstance(self.mean, (int, float)) else [self.mean] * C
            mean_t = torch.tensor(mean_vals, device=device, dtype=dtype).view(1, C, 1, 1)
            std_t = torch.tensor(self.std, device=device, dtype=dtype).view(1, C, 1, 1)
            canvas = (canvas - mean_t) / std_t

        return canvas

    def _generate_power_law(self, B, C, M, N, device, dtype):
        from slipstream.utils.noise import power_law_noise

        gen_size = _even_ceil(max(M, N))
        fixed_alpha = self.alpha_range[0] == self.alpha_range[1]

        if fixed_alpha:
            bg = power_law_noise(
                gen_size,
                alpha=self.alpha_range[0],
                out_channels=C,
                scale=(0, 1),
                device=device,
                batch_size=B,
                generator=self.rng,
                independent_channels=self.color_noise,
            )
        else:
            # Sample a different alpha per image
            slices = []
            for _ in range(B):
                alpha = torch.empty(1).uniform_(
                    self.alpha_range[0], self.alpha_range[1],
                    generator=self.rng,
                ).item()
                noise = power_law_noise(
                    gen_size,
                    alpha=alpha,
                    out_channels=C,
                    scale=(0, 1),
                    device=device,
                    generator=self.rng,
                    independent_channels=self.color_noise,
                )
                slices.append(noise)
            bg = torch.stack(slices, dim=0)

        # Crop to canvas size, make contiguous (expand() shares memory), cast
        return bg[:, :, :M, :N].contiguous().to(dtype=dtype)

    # ------------------------------------------------------------------ #
    #  Repr                                                                #
    # ------------------------------------------------------------------ #

    def __repr__(self):
        parts = [f"canvas_size={self.canvas_size}"]

        if self._position_mode == "range":
            parts.append(f"x_range={self.x_range}")
            parts.append(f"y_range={self.y_range}")
        elif self._position_mode == "coords":
            parts.append(f"coords=[{len(self.coords)} positions]")
        elif self._position_mode == "coords_dict":
            sizes = list(self.coords_dict.keys())
            parts.append(f"coords_dict={{{', '.join(str(s) for s in sizes)}}}")

        parts.append(f"background='{self.background}'")
        if self.background == "mean":
            parts.append(f"mean={self.mean}")
        elif self.background == "power_law":
            parts.append(f"alpha_range={self.alpha_range}")
            if not self.color_noise:
                parts.append("color_noise=False")
        if self.fade_radius is not None:
            parts.append(f"fade_radius={self.fade_radius}")
        if self.std is not None:
            parts.append(f"std={self.std}")

        if self.seed is not None:
            parts.append(f"seed={self.seed}")

        return f"{self.__class__.__name__}({', '.join(parts)})"
