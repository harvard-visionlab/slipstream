"""Mixup and CutMix transforms (per-sample / 'elem' mode only).

Each sample in the batch independently rolls its own λ and chooses mixup,
cutmix, or no-op. Fully vectorized — no Python loop over the batch.

Designed to be used via ``SlipstreamLoader(after_batch_transforms=[Mixup(...)])``,
since these ops consume both the image and label fields and emit a one-hot
mixed label.

References:
    mixup: Beyond Empirical Risk Minimization (https://arxiv.org/abs/1710.09412)
    CutMix: Regularization Strategy to Train Strong Classifiers (https://arxiv.org/abs/1905.04899)
    timm.data.mixup.Mixup
"""

from __future__ import annotations

import numpy as np
import torch


def one_hot(target: torch.Tensor, num_classes: int, on_value: float = 1.0,
            off_value: float = 0.0) -> torch.Tensor:
    """Integer class targets → one-hot, optionally with smoothing scalars."""
    t = target.long().view(-1, 1)
    out = torch.full(
        (t.size(0), num_classes), off_value,
        device=t.device, dtype=torch.float32,
    )
    return out.scatter_(1, t, on_value)


def mixup_target(target: torch.Tensor, num_classes: int, lam: torch.Tensor,
                 smoothing: float = 0.0) -> torch.Tensor:
    """Mix one-hot labels with the reversed batch by per-sample λ.

    Args:
        target: integer class tensor of shape [B].
        num_classes: total number of classes.
        lam: per-sample λ tensor of shape [B].
        smoothing: label smoothing factor in [0, 1).

    Returns:
        Float tensor [B, num_classes] of mixed (smoothed) one-hot labels.
    """
    off = smoothing / num_classes
    on = 1.0 - smoothing + off
    y1 = one_hot(target, num_classes, on, off)
    y2 = one_hot(target.flip(0), num_classes, on, off)
    lam = lam.to(y1.device).view(-1, 1).float()
    return y1 * lam + y2 * (1.0 - lam)


def _rand_bbox_per_sample(img_h: int, img_w: int, lam: np.ndarray,
                           rng: np.random.Generator):
    """Vectorized timm rand_bbox: square-ratio bbox derived from sqrt(1-λ).

    Returns four [N] int64 arrays: (yl, yh, xl, xh), top-left inclusive,
    bottom-right exclusive.
    """
    n = lam.shape[0]
    ratio = np.sqrt(np.clip(1.0 - lam, 0.0, 1.0))
    cut_h = (img_h * ratio).astype(np.int64)
    cut_w = (img_w * ratio).astype(np.int64)
    cy = rng.integers(0, img_h, size=n)
    cx = rng.integers(0, img_w, size=n)
    yl = np.clip(cy - cut_h // 2, 0, img_h)
    yh = np.clip(cy + cut_h // 2, 0, img_h)
    xl = np.clip(cx - cut_w // 2, 0, img_w)
    xh = np.clip(cx + cut_w // 2, 0, img_w)
    return yl, yh, xl, xh


def _rand_bbox_minmax_per_sample(img_h: int, img_w: int,
                                  minmax: tuple[float, float],
                                  n: int, rng: np.random.Generator):
    """Vectorized timm rand_bbox_minmax: per-sample uniform h/w within ratio bounds."""
    h_lo, h_hi = int(img_h * minmax[0]), int(img_h * minmax[1])
    w_lo, w_hi = int(img_w * minmax[0]), int(img_w * minmax[1])
    cut_h = rng.integers(h_lo, max(h_hi, h_lo + 1), size=n)
    cut_w = rng.integers(w_lo, max(w_hi, w_lo + 1), size=n)
    yl = (rng.random(n) * np.maximum(img_h - cut_h, 0)).astype(np.int64)
    xl = (rng.random(n) * np.maximum(img_w - cut_w, 0)).astype(np.int64)
    yh = np.clip(yl + cut_h, 0, img_h)
    xh = np.clip(xl + cut_w, 0, img_w)
    return yl, yh, xl, xh


class Mixup:
    """Per-sample Mixup / CutMix.

    For each image in the batch, independently:
      1. Roll Bernoulli(``prob``) — selects whether this sample mixes at all.
      2. If both ``mixup_alpha`` and ``cutmix_alpha`` are positive, roll
         Bernoulli(``switch_prob``) to choose cutmix vs mixup.
      3. Sample λ from the appropriate Beta distribution and apply the op.

    Pairing is deterministic: sample ``i`` mixes with sample ``B-i-1``
    (i.e., the reversed batch), matching timm.

    Operates on the entire batch dict — reads ``batch[image_key]`` and
    ``batch[label_key]``, writes back the mixed image tensor and the
    one-hot mixed label tensor (shape ``[B, num_classes]``, float).

    Args:
        mixup_alpha: Beta α for mixup λ. 0 disables mixup.
        cutmix_alpha: Beta α for cutmix λ. 0 disables cutmix.
        cutmix_minmax: Optional ``(min_ratio, max_ratio)`` for the cutmix
            bbox h/w. Overrides ``cutmix_alpha`` for bbox sampling when set.
        prob: Per-sample probability of any mixing.
        switch_prob: When both mixup and cutmix are active, per-sample
            probability of choosing cutmix over mixup.
        correct_lam: For cutmix samples, recompute λ from the actual
            (clipped) bbox area. Recommended.
        label_smoothing: Label smoothing factor in [0, 1).
        num_classes: Number of classes (for one-hot conversion).
        image_key: Key in the batch dict for the image tensor.
        label_key: Key in the batch dict for integer-class labels.
        seed: RNG seed. For DDP, use a per-rank seed (e.g. ``base + rank``).
    """

    def __init__(
        self,
        mixup_alpha: float = 0.8,
        cutmix_alpha: float = 1.0,
        cutmix_minmax: tuple[float, float] | None = None,
        prob: float = 1.0,
        switch_prob: float = 0.5,
        correct_lam: bool = True,
        label_smoothing: float = 0.0,
        num_classes: int = 1000,
        image_key: str = "image",
        label_key: str = "label",
        seed: int | None = None,
    ):
        if cutmix_minmax is not None:
            if len(cutmix_minmax) != 2:
                raise ValueError("cutmix_minmax must be a (min, max) pair")
            cutmix_alpha = 1.0  # match timm: minmax forces α=1

        if not (mixup_alpha > 0 or cutmix_alpha > 0 or cutmix_minmax is not None):
            raise ValueError(
                "At least one of mixup_alpha > 0, cutmix_alpha > 0, "
                "or cutmix_minmax must be set."
            )

        self.mixup_alpha = float(mixup_alpha)
        self.cutmix_alpha = float(cutmix_alpha)
        self.cutmix_minmax = tuple(cutmix_minmax) if cutmix_minmax is not None else None
        self.prob = float(prob)
        self.switch_prob = float(switch_prob)
        self.correct_lam = bool(correct_lam)
        self.label_smoothing = float(label_smoothing)
        self.num_classes = int(num_classes)
        self.image_key = image_key
        self.label_key = label_key
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Last-call params (useful for testing / visualization).
        self.last_lam: torch.Tensor | None = None
        self.last_use_mixup: np.ndarray | None = None
        self.last_use_cutmix: np.ndarray | None = None
        self.last_bbox: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None

    def _sample_params(self, batch_size: int):
        """Roll per-sample (lam, use_mixup, use_cutmix). All numpy."""
        lam = np.ones(batch_size, dtype=np.float32)
        use_mixup = np.zeros(batch_size, dtype=bool)
        use_cutmix = np.zeros(batch_size, dtype=bool)

        do_mix = self.rng.random(batch_size) < self.prob
        if not do_mix.any():
            return lam, use_mixup, use_cutmix

        if self.mixup_alpha > 0 and self.cutmix_alpha > 0:
            switch = self.rng.random(batch_size) < self.switch_prob
            use_cutmix = do_mix & switch
            use_mixup = do_mix & ~switch
        elif self.cutmix_alpha > 0 or self.cutmix_minmax is not None:
            use_cutmix = do_mix
        else:
            use_mixup = do_mix

        if use_mixup.any():
            lam[use_mixup] = self.rng.beta(
                self.mixup_alpha, self.mixup_alpha, size=int(use_mixup.sum())
            ).astype(np.float32)
        if use_cutmix.any():
            lam[use_cutmix] = self.rng.beta(
                self.cutmix_alpha, self.cutmix_alpha, size=int(use_cutmix.sum())
            ).astype(np.float32)

        return lam, use_mixup, use_cutmix

    def __call__(self, batch: dict) -> dict:
        if self.image_key not in batch:
            raise KeyError(f"image_key {self.image_key!r} not in batch (keys: {list(batch.keys())})")
        if self.label_key not in batch:
            raise KeyError(f"label_key {self.label_key!r} not in batch (keys: {list(batch.keys())})")

        x = batch[self.image_key]
        y = batch[self.label_key]

        if not isinstance(x, torch.Tensor) or x.ndim != 4:
            raise ValueError(
                f"Expected 4D image tensor [B, C, H, W] at batch[{self.image_key!r}], "
                f"got {type(x).__name__} with shape {getattr(x, 'shape', None)}"
            )
        if not x.dtype.is_floating_point:
            raise ValueError(
                f"Mixup requires a floating-point image tensor; got dtype {x.dtype}. "
                "Apply Normalize / ToFloat before Mixup."
            )

        B, C, H, W = x.shape
        if B % 2 != 0:
            raise ValueError(f"Mixup requires an even batch size, got B={B}.")

        device = x.device
        if not isinstance(y, torch.Tensor):
            y = torch.as_tensor(y)
        y = y.to(device)

        lam_np, use_mixup_np, use_cutmix_np = self._sample_params(B)

        x_flip = x.flip(0)

        # Cutmix bboxes (sampled vectorized; only applied to cutmix samples).
        bbox = None
        if use_cutmix_np.any():
            if self.cutmix_minmax is not None:
                yl_np, yh_np, xl_np, xh_np = _rand_bbox_minmax_per_sample(
                    H, W, self.cutmix_minmax, B, self.rng
                )
            else:
                yl_np, yh_np, xl_np, xh_np = _rand_bbox_per_sample(
                    H, W, lam_np, self.rng
                )
            bbox = (yl_np, yh_np, xl_np, xh_np)

            if self.correct_lam or self.cutmix_minmax is not None:
                bbox_area = ((yh_np - yl_np) * (xh_np - xl_np)).astype(np.float32)
                corrected = 1.0 - bbox_area / float(H * W)
                lam_np = np.where(use_cutmix_np, corrected, lam_np).astype(np.float32)

        lam = torch.from_numpy(lam_np).to(device=device, dtype=x.dtype)

        # Mixup output for ALL samples — for no-op samples (λ=1), this equals x.
        # For cutmix samples we'll override below.
        lam_view = lam.view(B, 1, 1, 1)
        x_out = x * lam_view + x_flip * (1.0 - lam_view)

        if use_cutmix_np.any():
            yl_np, yh_np, xl_np, xh_np = bbox
            yl = torch.from_numpy(yl_np).to(device).view(B, 1, 1)
            yh = torch.from_numpy(yh_np).to(device).view(B, 1, 1)
            xl = torch.from_numpy(xl_np).to(device).view(B, 1, 1)
            xh = torch.from_numpy(xh_np).to(device).view(B, 1, 1)
            rows = torch.arange(H, device=device).view(1, H, 1)
            cols = torch.arange(W, device=device).view(1, 1, W)
            in_rect = (rows >= yl) & (rows < yh) & (cols >= xl) & (cols < xh)
            use_cutmix_t = torch.from_numpy(use_cutmix_np).to(device)
            cut_mask = (in_rect & use_cutmix_t.view(B, 1, 1)).unsqueeze(1)  # [B,1,H,W]

            # Per-sample selection: cutmix samples get x with bbox replaced;
            # other samples keep their mixup output.
            x_cut = torch.where(cut_mask, x_flip, x)
            x_out = torch.where(use_cutmix_t.view(B, 1, 1, 1), x_cut, x_out)

        mixed_y = mixup_target(y, self.num_classes, lam, self.label_smoothing)

        # Stash for testing / visualization.
        self.last_lam = lam.detach().cpu()
        self.last_use_mixup = use_mixup_np.copy()
        self.last_use_cutmix = use_cutmix_np.copy()
        self.last_bbox = bbox

        batch[self.image_key] = x_out
        batch[self.label_key] = mixed_y
        return batch

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"mixup_alpha={self.mixup_alpha}, cutmix_alpha={self.cutmix_alpha}, "
            f"cutmix_minmax={self.cutmix_minmax}, prob={self.prob}, "
            f"switch_prob={self.switch_prob}, correct_lam={self.correct_lam}, "
            f"label_smoothing={self.label_smoothing}, num_classes={self.num_classes}, "
            f"image_key={self.image_key!r}, label_key={self.label_key!r}, seed={self.seed})"
        )
