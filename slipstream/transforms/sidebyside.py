"""Side-by-side search-pair transform (after_batch_transform).

Turns a batch of single ``H×W`` images into ``A | B`` side-by-side displays for
a minimal visual-search task: each sample's own image is the **target** on one
side and a within-batch **partner** image is the **distractor** on the other
side. The transform only builds the displays + labels; templates, steering, and
scoring live in the consumer (lrm-space-dev visual_search task).

Mirrors the conventions of :class:`slipstream.transforms.Mixup` /
:class:`CutMixClutter`: callable on a batch dict, vectorized + GPU-resident,
seeded RNG, ``last_*`` inspection attributes.

Pipeline placement: insert as an ``after_batch_transform`` operating on **[0,1]
RGB** images (NOT normalized — the consumer normalizes per-checkpoint
downstream). The transform is normalization-agnostic; it just composites pixels
and relabels.
"""

from __future__ import annotations

import numpy as np
import torch


def _derangement(n: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform random permutation of ``range(n)`` with no fixed points.

    Rejection sampling (expected ~e tries) — uniform among derangements, so no
    fixed class→class pairing bias. Requires ``n >= 2``.
    """
    if n < 2:
        raise ValueError(f"derangement requires n >= 2, got n={n}")
    idx = np.arange(n)
    while True:
        perm = rng.permutation(n)
        if not np.any(perm == idx):
            return perm


def _chunk_partner(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Different-class partner via label-chunked half-batch shift.

    Chunk so same-class samples are adjacent, but order groups by each label's
    **first appearance** in the (already-shuffled) batch — random per batch, not
    by label value, which avoids a fixed class→class mapping. Then pair position
    ``k`` in chunked order with ``(k + n//2) % n`` and map back to batch order.

    Returns a ``(n,)`` long tensor of partner indices (no self-pairs for n >= 2).
    """
    labels = labels.flatten()
    n = labels.numel()
    device = labels.device
    arange = torch.arange(n, device=device)
    # first[c] = index of the first sample with label c (sentinel n if absent).
    first = labels.new_full((num_classes,), n)
    first.scatter_reduce_(0, labels, arange, reduce="amin", include_self=True)
    key = first[labels]                              # first-appearance rank
    order = torch.argsort(key, stable=True)          # STABLE → group order det.

    half = n // 2
    k = torch.arange(n, device=device)
    partner_in_order = order[(k + half) % n]
    partner = torch.empty(n, dtype=torch.long, device=device)
    partner[order] = partner_in_order
    return partner


class SideBySideSearchPair:
    """Composite single images into side-by-side ``target | distractor`` displays.

    For each sample ``i`` in the batch:
      1. Pick a within-batch **partner** ``j`` (a derangement by default, or a
         different-class partner when ``require_different_class=True``).
      2. Roll which side the target goes on (``Bernoulli(p_left)`` → left).
      3. Build a ``(3, H, 2W)`` composite: sample ``i``'s own image (the target)
         on its side, the partner's image (the distractor) on the other side.

    Reads ``batch[image_key]`` ``(B, 3, H, W)`` float [0,1] RGB and
    ``batch[label_key]`` ``(B,)`` int, and writes:
      - ``batch[image_key]`` ← ``(B, 3, H, 2W)`` composite (reassigned; width
        doubles, so this is a fresh tensor not an in-place overwrite).
      - ``batch[label_key]`` ← ``(B,)`` the **target** label (sample ``i``'s
        original label, unchanged — plain integer classification of the target).
      - ``batch[target_side_key]`` ← ``(B,)`` int, ``0=left`` / ``1=right``.
      - ``batch[distractor_label_key]`` ← ``(B,)`` the partner's class label.
      - ``batch[valid_key]`` ← ``(B,)`` bool of usable trials (all True in the
        default derangement mode; ``label != partner_label`` when
        ``require_different_class=True``).

    Args:
        num_classes: Number of classes (used only by the
            ``require_different_class`` chunked-partner pairing).
        image_key: Batch key for the image tensor.
        label_key: Batch key for integer-class labels (the target label, written
            back unchanged).
        target_side_key: Batch key to write the target side (0=left, 1=right).
        distractor_label_key: Batch key to write the distractor's class label.
        valid_key: Batch key to write the per-sample validity mask.
        require_different_class: If False (default), partner is a seeded random
            derangement (instance search; ~all pairs different-class, same-class
            pairs are harmless harder examples) and ``valid`` is all True. If
            True, partner is drawn a half-batch apart in label-chunked order
            (category-prototype search) and ``valid[i] = label[i] !=
            label[partner[i]]``.
        p_left: Probability the target is placed on the left side.
        seed: RNG seed. For DDP, use a per-rank seed (e.g. ``base + rank``).
    """

    def __init__(
        self,
        *,
        num_classes: int,
        image_key: str = "image",
        label_key: str = "label",
        target_side_key: str = "target_side",
        distractor_label_key: str = "distractor_label",
        valid_key: str = "valid",
        require_different_class: bool = False,
        p_left: float = 0.5,
        seed: int | None = None,
    ):
        if not (0.0 <= p_left <= 1.0):
            raise ValueError(f"p_left must be in [0, 1], got {p_left}")

        out_keys = [target_side_key, distractor_label_key, valid_key]
        if len(set(out_keys)) != len(out_keys):
            raise ValueError(
                f"target_side_key / distractor_label_key / valid_key must be "
                f"distinct, got {out_keys}"
            )
        if image_key in out_keys:
            raise ValueError(
                f"image_key {image_key!r} collides with an output key {out_keys}"
            )

        self.num_classes = int(num_classes)
        self.image_key = image_key
        self.label_key = label_key
        self.target_side_key = target_side_key
        self.distractor_label_key = distractor_label_key
        self.valid_key = valid_key
        self.require_different_class = bool(require_different_class)
        self.p_left = float(p_left)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Last-call params (useful for testing / visualization).
        self.last_target_side: np.ndarray | None = None
        self.last_distractor_label: np.ndarray | None = None
        self.last_valid: np.ndarray | None = None
        self.last_partner_index: np.ndarray | None = None

    def __call__(self, batch: dict) -> dict:
        if self.image_key not in batch:
            raise KeyError(
                f"image_key {self.image_key!r} not in batch (keys: {list(batch.keys())})"
            )
        if self.label_key not in batch:
            raise KeyError(
                f"label_key {self.label_key!r} not in batch (keys: {list(batch.keys())})"
            )

        x = batch[self.image_key]
        y = batch[self.label_key]

        if not isinstance(x, torch.Tensor) or x.ndim != 4:
            raise ValueError(
                f"Expected 4D image tensor [B, C, H, W] at batch[{self.image_key!r}], "
                f"got {type(x).__name__} with shape {getattr(x, 'shape', None)}"
            )
        if not x.dtype.is_floating_point:
            raise ValueError(
                f"SideBySideSearchPair requires a floating-point image tensor; got "
                f"dtype {x.dtype}. Apply ToFloat / ToFloatDiv before this transform."
            )

        B, C, H, W = x.shape
        if B < 2:
            raise ValueError(f"SideBySideSearchPair requires B >= 2, got B={B}.")

        device = x.device
        if not isinstance(y, torch.Tensor):
            y = torch.as_tensor(y)
        y = y.to(device).long().flatten()

        # --- Partner selection ---
        if self.require_different_class:
            partner = _chunk_partner(y, self.num_classes)
            valid = y != y[partner]
        else:
            partner_np = _derangement(B, self.rng)
            partner = torch.from_numpy(partner_np).to(device=device, dtype=torch.long)
            valid = torch.ones(B, dtype=torch.bool, device=device)

        # --- Target side: 0=left, 1=right ---
        target_side_np = (self.rng.random(B) >= self.p_left).astype(np.int64)
        target_side = torch.from_numpy(target_side_np).to(device=device)
        left_is_target = (target_side == 0).view(B, 1, 1, 1)

        # --- Composite: own image (target) on its side, partner (distractor) other ---
        partner_img = x[partner]                              # (B, C, H, W)
        left = torch.where(left_is_target, x, partner_img)
        right = torch.where(left_is_target, partner_img, x)
        composite = torch.cat([left, right], dim=3)           # (B, C, H, 2W)

        distractor_label = y[partner]

        # Stash for testing / visualization.
        self.last_partner_index = partner.detach().cpu().numpy()
        self.last_target_side = target_side.detach().cpu().numpy()
        self.last_distractor_label = distractor_label.detach().cpu().numpy()
        self.last_valid = valid.detach().cpu().numpy()

        batch[self.image_key] = composite
        batch[self.label_key] = y                             # target label, unchanged
        batch[self.target_side_key] = target_side
        batch[self.distractor_label_key] = distractor_label
        batch[self.valid_key] = valid
        return batch

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"num_classes={self.num_classes}, image_key={self.image_key!r}, "
            f"label_key={self.label_key!r}, target_side_key={self.target_side_key!r}, "
            f"distractor_label_key={self.distractor_label_key!r}, "
            f"valid_key={self.valid_key!r}, "
            f"require_different_class={self.require_different_class}, "
            f"p_left={self.p_left}, seed={self.seed})"
        )
