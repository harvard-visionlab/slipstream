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
        p_fade: Probability of applying the circular fade **per image**.
            1.0 (default) always applies it; 0.5 applies to ~half the
            images in a batch, leaving the rest with a hard rectangular
            boundary.  Ignored when ``fade_radius`` is ``None``.
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
        p_fade=1.0,
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
        self.p_fade = p_fade
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
        self._do_fade = None

    # ------------------------------------------------------------------ #
    #  BatchAugment interface                                              #
    # ------------------------------------------------------------------ #

    def before_call(self, b, **kwargs):
        n = b.shape[0] if b.ndim == 4 else 1
        h, w = (b.shape[-2], b.shape[-1])
        M, N = self.canvas_size

        self._img_h = h
        self._img_w = w

        # Per-image fade decision
        if self.fade_radius is not None and self.p_fade < 1.0:
            self._do_fade = torch.bernoulli(
                torch.full((n,), self.p_fade), generator=self.rng,
            ).bool()
        elif self.fade_radius is not None:
            self._do_fade = torch.ones(n, dtype=torch.bool)
        else:
            self._do_fade = torch.zeros(n, dtype=torch.bool)

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
            "do_fade": self._do_fade,
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
                if self._do_fade[i]:
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
            if self.p_fade != 1.0:
                parts.append(f"p_fade={self.p_fade}")
        if self.std is not None:
            parts.append(f"std={self.std}")

        if self.seed is not None:
            parts.append(f"seed={self.seed}")

        return f"{self.__class__.__name__}({', '.join(parts)})"


class RandomBackgroundBlend(BatchAugment):
    """Blend RGBA images onto a generated background using the alpha channel.

    Takes 4-channel RGBA input (from :class:`~slipstream.decoders.DecodeMultiResizeCropEmbed`
    via :class:`~slipstream.transforms.ToTorchImage`) and composites the RGB
    foreground onto a generated background using the alpha channel as opacity.
    Returns 3-channel RGB output, optionally normalized.

    The alpha channel is consumed during blending — downstream transforms
    receive standard 3-channel tensors.

    Edge fading can be applied to the alpha mask before blending to create
    smooth transitions between the foreground and background:

    - ``"circular"``: Gaussian fade from center outward (same as
      :class:`RandomEmbed` ``fade_radius``).
    - ``"cosine"``: Cosine-tapered rectangular window (Tukey window).
      Produces smooth edges that follow the image borders with
      C1-continuous transitions.

    Args:
        background: One of ``"zeros"``, ``"mean"``, ``"power_law"``.
        mean: Fill value for ``"mean"`` background, or normalisation mean
            when ``std`` is also provided.
        std: Per-channel std for normalisation.  When provided, output is
            normalised as ``(blended - mean) / std``.
        alpha_range: Power-law exponent.  Float or ``(min, max)``.
        color_noise: Independent noise per channel (default ``True``).
        noise_scale: Generate power-law noise at a fraction of the canvas
            resolution, then bilinear-upsample.  1.0 (default) generates at
            full resolution; 0.25 generates at 25% and upsamples (~9x faster
            but visibly blurrier for low alpha).
        fade_mode: ``None`` (hard rect), ``"circular"``, or ``"cosine"``.
        fade_radius: For ``"circular"`` mode: ``(inner, outer)`` as fractions
            of ``min(crop_h, crop_w)``.  Pixels within ``inner`` are fully
            opaque; beyond ``outer`` fully transparent.
        inset: For ``"cosine"`` mode: fade width.  Float in (0, 1) is a
            fraction of the crop dimension; int > 1 is pixel count.
        p_fade: Per-image probability of applying fade (default 1.0).
        p_background: Per-image probability of blending onto a generated
            background (default 1.0).  Images where the coin flip fails
            get a zero (black) background instead, which is useful for
            stochastic augmentation schedules.
        seed: Random seed for reproducible backgrounds and fade decisions.
        device: Device for the RNG.

    Example::

        blend = RandomBackgroundBlend(
            background='power_law', alpha_range=1.5,
            fade_mode='cosine', inset=0.05, p_fade=0.5,
            mean=IMAGENET_MEAN, std=IMAGENET_STD, seed=44,
        )
        # Input: [B, 4, H, W] RGBA float from ToTorchImage
        # Output: [B, 3, H, W] RGB float, normalized
        out = blend(rgba_tensor)
    """

    _VALID_BACKGROUNDS = ("zeros", "mean", "power_law")
    _VALID_FADE_MODES = (None, "circular", "cosine")

    def __init__(
        self,
        # Background ─────────────────────────────────────────
        background: str = "zeros",
        mean=None,
        std=None,
        alpha_range: float | tuple[float, float] = 2.0,
        color_noise: bool = True,
        noise_scale: float = 1.0,
        # Fade ────────────────────────────────────────────────
        fade_mode: str | None = None,
        fade_radius: tuple[float, float] | None = None,
        inset: float | int | None = None,
        p_fade: float = 1.0,
        p_background: float = 1.0,
        # Common ──────────────────────────────────────────────
        seed: int | None = None,
        device=None,
    ):
        if background not in self._VALID_BACKGROUNDS:
            raise ValueError(
                f"background must be one of {self._VALID_BACKGROUNDS}, got '{background}'"
            )
        if background == "mean" and mean is None:
            raise ValueError("'mean' is required for background='mean'")
        if std is not None and mean is None:
            raise ValueError(
                "'mean' is required when 'std' is provided"
            )
        if fade_mode not in self._VALID_FADE_MODES:
            raise ValueError(
                f"fade_mode must be one of {self._VALID_FADE_MODES}, got '{fade_mode}'"
            )
        if fade_mode == "circular":
            if fade_radius is None:
                raise ValueError("fade_radius is required for fade_mode='circular'")
            fade_radius = tuple(fade_radius)
            if len(fade_radius) != 2 or fade_radius[0] >= fade_radius[1]:
                raise ValueError(
                    f"fade_radius must be (inner, outer) with inner < outer, got {fade_radius}"
                )
        if fade_mode == "cosine" and inset is None:
            raise ValueError("inset is required for fade_mode='cosine'")

        self.background = background
        self.mean = mean
        self.std = std
        self.alpha_range = (
            (alpha_range, alpha_range) if isinstance(alpha_range, (int, float))
            else tuple(alpha_range)
        )
        self.color_noise = color_noise
        if not 0 < noise_scale <= 1.0:
            raise ValueError(f"noise_scale must be in (0, 1], got {noise_scale}")
        self.noise_scale = noise_scale
        self.fade_mode = fade_mode
        self.fade_radius = fade_radius
        self.inset = inset
        self.p_fade = p_fade
        self.p_background = p_background

        self.seed = seed
        self.rng = None
        if self.seed is not None:
            self.rng = torch.Generator("cpu" if device is None else device)
            self.rng.manual_seed(self.seed)

        self._do_fade = None
        self._do_background = None

    # ------------------------------------------------------------------ #
    #  BatchAugment interface                                              #
    # ------------------------------------------------------------------ #

    def before_call(self, b, **kwargs):
        n = b.shape[0] if b.ndim == 4 else 1
        if self.fade_mode is not None and self.p_fade < 1.0:
            self._do_fade = torch.bernoulli(
                torch.full((n,), self.p_fade), generator=self.rng,
            ).bool()
        elif self.fade_mode is not None:
            self._do_fade = torch.ones(n, dtype=torch.bool)
        else:
            self._do_fade = torch.zeros(n, dtype=torch.bool)

        if self.p_background < 1.0:
            self._do_background = torch.bernoulli(
                torch.full((n,), self.p_background), generator=self.rng,
            ).bool()
        else:
            self._do_background = torch.ones(n, dtype=torch.bool)

    def last_params(self):
        return {"do_fade": self._do_fade, "do_background": self._do_background}

    def apply_last(self, b):
        expanded = False
        if b.ndim == 3:
            b = b.unsqueeze(0)
            expanded = True

        B, C, H, W = b.shape
        if C not in (4,):
            raise ValueError(
                f"RandomBackgroundBlend expects 4-channel RGBA input, got {C} channels. "
                "Use DecodeMultiResizeCropEmbed to produce RGBA output."
            )

        rgb = b[:, :3]          # [B, 3, H, W]
        alpha = b[:, 3:4]       # [B, 1, H, W]

        # Apply fade to alpha
        if self.fade_mode is not None and self._do_fade.any():
            alpha = self._apply_fade(alpha, self._do_fade)

        # Generate background (zeros for images where do_background is False)
        canvas = torch.zeros(B, 3, H, W, device=b.device, dtype=b.dtype)
        if self._do_background.any():
            bg_idx = self._do_background.nonzero(as_tuple=True)[0]
            bg = self._generate_background(len(bg_idx), 3, H, W, b.device, b.dtype)
            canvas[bg_idx] = bg

        # Blend
        out = alpha * rgb + (1.0 - alpha) * canvas

        # Normalize if requested
        if self.std is not None:
            mean_vals = self.mean if not isinstance(self.mean, (int, float)) else [self.mean] * 3
            mean_t = torch.tensor(mean_vals, device=b.device, dtype=b.dtype).view(1, 3, 1, 1)
            std_t = torch.tensor(self.std, device=b.device, dtype=b.dtype).view(1, 3, 1, 1)
            out = (out - mean_t) / std_t

        if expanded:
            out = out.squeeze(0)
        return out

    # ------------------------------------------------------------------ #
    #  Fade mask generation                                                #
    # ------------------------------------------------------------------ #

    def _apply_fade(self, alpha, do_fade):
        """Apply fade to alpha channel for images where do_fade is True."""
        B, _, H, W = alpha.shape

        # Find image bounding boxes from alpha
        # All images with the same crop size share one mask
        alpha_out = alpha.clone()

        for i in range(B):
            if not do_fade[i]:
                continue

            # Find the bounding box of non-zero alpha for this image
            mask_2d = alpha[i, 0] > 0  # [H, W]
            if not mask_2d.any():
                continue

            rows = mask_2d.any(dim=1).nonzero(as_tuple=True)[0]
            cols = mask_2d.any(dim=0).nonzero(as_tuple=True)[0]
            y0, y1 = rows[0].item(), rows[-1].item() + 1
            x0, x1 = cols[0].item(), cols[-1].item() + 1
            h, w = y1 - y0, x1 - x0

            fade_mask = self._make_fade_mask(h, w, alpha.device, alpha.dtype)
            alpha_out[i, 0, y0:y1, x0:x1] = alpha[i, 0, y0:y1, x0:x1] * fade_mask

        return alpha_out

    def _make_fade_mask(self, h, w, device, dtype):
        """Build a fade mask of shape [h, w] based on fade_mode."""
        if self.fade_mode == "circular":
            return self._make_circular_fade(h, w, device, dtype)
        elif self.fade_mode == "cosine":
            return self._make_cosine_fade(h, w, device, dtype)
        return torch.ones(h, w, device=device, dtype=dtype)

    def _make_circular_fade(self, h, w, device, dtype):
        """Circular Gaussian fade mask (same as RandomEmbed)."""
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
        return mask

    def _make_cosine_fade(self, h, w, device, dtype):
        """Cosine-tapered rectangular fade mask (2D Tukey window)."""
        inset = self.inset
        if isinstance(inset, float) and inset <= 1.0:
            # Fraction of dimension
            inset_x = inset * w
            inset_y = inset * h
        else:
            inset_x = float(inset)
            inset_y = float(inset)

        # 1D distance from nearest edge
        tx = torch.arange(w, device=device, dtype=dtype)
        dist_x = torch.min(tx, (w - 1) - tx)  # [W]
        ty = torch.arange(h, device=device, dtype=dtype)
        dist_y = torch.min(ty, (h - 1) - ty)  # [H]

        # 2D: minimum distance to any edge
        dist_2d = torch.min(dist_y[:, None], dist_x[None, :])  # [H, W]

        # Use the minimum inset for the corner region
        inset_min = min(inset_x, inset_y)
        if inset_min <= 0:
            return torch.ones(h, w, device=device, dtype=dtype)

        # Normalize to [0, 1] and apply cosine annealing
        t = (dist_2d / inset_min).clamp(0, 1)
        mask = 0.5 * (1.0 - torch.cos(t * torch.pi))
        return mask

    # ------------------------------------------------------------------ #
    #  Background generation (shared logic with RandomEmbed)               #
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

        return canvas

    def _generate_power_law(self, B, C, M, N, device, dtype):
        import torch.nn.functional as F
        from slipstream.utils.noise import power_law_noise

        # Compute generation resolution from noise_scale
        canvas_size = max(M, N)
        if self.noise_scale < 1.0:
            gen_size = _even_ceil(max(2, int(canvas_size * self.noise_scale)))
        else:
            gen_size = _even_ceil(canvas_size)
        needs_upsample = gen_size < canvas_size
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

        if needs_upsample:
            # Ensure 4D for interpolate: [B, C, H, W]
            if bg.ndim == 3:
                bg = bg.unsqueeze(0)
            bg = F.interpolate(bg, size=(M, N), mode='bilinear', align_corners=False)
            return bg.to(dtype=dtype)

        return bg[:, :, :M, :N].contiguous().to(dtype=dtype)

    # ------------------------------------------------------------------ #
    #  Repr                                                                #
    # ------------------------------------------------------------------ #

    def __repr__(self):
        parts = [f"background='{self.background}'"]
        if self.background == "mean":
            parts.append(f"mean={self.mean}")
        elif self.background == "power_law":
            parts.append(f"alpha_range={self.alpha_range}")
            if not self.color_noise:
                parts.append("color_noise=False")
            if self.noise_scale != 1.0:
                parts.append(f"noise_scale={self.noise_scale}")
        if self.fade_mode is not None:
            parts.append(f"fade_mode='{self.fade_mode}'")
            if self.fade_mode == "circular":
                parts.append(f"fade_radius={self.fade_radius}")
            elif self.fade_mode == "cosine":
                parts.append(f"inset={self.inset}")
            if self.p_fade != 1.0:
                parts.append(f"p_fade={self.p_fade}")
        if self.p_background != 1.0:
            parts.append(f"p_background={self.p_background}")
        if self.std is not None:
            parts.append(f"mean={self.mean}, std={self.std}")
        if self.seed is not None:
            parts.append(f"seed={self.seed}")
        return f"RandomBackgroundBlend({', '.join(parts)})"
