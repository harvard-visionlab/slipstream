"""CollectParams: put the random parameters of a loader's pipelines into the batch dict.

An ``after_batch_transform`` for objectives that need to know what was done to each view
(equivariant SSL needs jitter factors, flips, rotations; VICRegL needs crop boxes). It holds
references to the same transform objects as the loader and reads their ``last_params()`` after the
pipelines have run on the batch. Opt-in: loaders without it do no extra work.

    pipelines = {"image": [
        DecodeMultiRandomResizedCrop({"g0": dict(size=224, seed=1), "g1": dict(size=224, seed=2)}),
        MultiCropPipeline({
            "g0": [ToTorchImage("cuda"), RandomHorizontalFlip(seed=3), RandomColorJitterHSV(p=0.8, seed=4)],
            "g1": [ToTorchImage("cuda"), RandomHorizontalFlip(seed=5), RandomColorJitterHSV(p=0.8, seed=6)],
        }),
    ]}
    loader = SlipstreamLoader(ds, pipelines=pipelines, after_batch_transforms=[CollectParams(pipelines)])
    batch["_params"]["image"]["g0"]["DecodeMultiRandomResizedCrop"]["box"]   # [B, 4] (x, y, w, h)
    batch["_params"]["image"]["g0"]["RandomColorJitterHSV"]["h"]             # [B], 0 where not applied

Layout: ``batch[key][field][view][TransformName] = {param: tensor}``. Views are the names that a
``MultiCropPipeline``, ``NamedCopies`` or multi-crop decoder gives the field's outputs. Each view
lists every random step that produced it, in order, including the shared steps before the split. A
field without views maps straight to ``{TransformName: params}``. A second instance of the same class
in one view is ``Name_1``. A multi-pipeline field (a list of sub-pipelines) gives a list.

Every tensor is per sample, with N = samples in the batch (``[B, T]`` for windowed loaders):
- ``applied`` (bool) replaces the transforms' ``do`` / ``idx``.
- Per-sample values come back full length. Rows that were not applied hold the identity value
  (h 0; s, v, b, c, sf 1; sigma 0). Other idx-style values are NaN there.
- RandomGaussianBlur gives ``sigma`` (and ``kernel``, -1 where not applied).
- Crop decoders give ``box`` int32 [N, 4] = (x, y, w, h) in source-image pixels, plus
  ``heights`` / ``widths`` and any other decoder params (``x_pos``, ``embed_rects``, ...).
  These are CPU tensors unless ``device`` is set. yuv420 caches store even dimensions, so there
  ``heights`` / ``widths`` may exceed the original by 1 px (padding at the right / bottom edge).
- Inside ``RandomApply``, a skipped batch gives only ``applied`` (all False).
- Other params (``deg``, ``xs``, ``mat``, ...) come back as the transform stores them.

Deterministic steps (ToTorchImage, Normalize, RGBToLab, ...) are left out.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .base import BatchAugment, Compose, RandomApply

# Identity value of a per-sample parameter for samples the transform skipped.
NEUTRAL = {"h": 0.0, "s": 1.0, "v": 1.0, "b": 1.0, "c": 1.0, "sf": 1.0, "sigma": 0.0}


def _view_names(t) -> list[str] | None:
    """Names of the views a pipeline step splits a field into, or None if it does not split."""
    from slipstream.decoders.multicrop import MultiCropPipeline, NamedCopies
    if isinstance(t, MultiCropPipeline):
        return list(t.pipelines)
    if isinstance(t, NamedCopies):
        return list(t.names)
    names = getattr(t, "_crop_names", None) or getattr(getattr(t, "_inner", None), "_crop_names", None)
    return list(names) if names is not None and hasattr(t, "last_params") else None   # DecodeMulti* decoders


def _flatten(t, guards: tuple) -> list[tuple[Any, tuple]]:
    """Expand Compose / RandomApply into (transform, guards) leaves."""
    if isinstance(t, Compose):
        return [leaf for x in t.transforms for leaf in _flatten(x, guards)]
    if isinstance(t, RandomApply):
        return [leaf for x in t.transforms for leaf in _flatten(x, guards + (t,))]
    if isinstance(t, (list, tuple)):
        return [leaf for x in t for leaf in _flatten(x, guards)]
    return [(t, guards)]


def _has_params(t) -> bool:
    lp = getattr(t, "last_params", None)
    if lp is None:
        return False
    if isinstance(t, BatchAugment) and type(t).last_params is BatchAugment.last_params:
        return False                                  # deterministic transform
    return True


def _plan(pipeline) -> dict:
    """{view or None: [(transform, view_for_its_params or None, guards)]} for one field pipeline."""
    from slipstream.decoders.multicrop import MultiCropPipeline
    cur: dict = {None: []}
    for step in (pipeline if isinstance(pipeline, (list, tuple)) else [pipeline]):
        names = _view_names(step)
        if names is None:
            leaves = [(t, None, g) for t, g in _flatten(step, ()) if _has_params(t)]
            for v in cur:
                cur[v].extend(leaves)
            continue
        shared = cur.pop(None, [])
        for n in names:
            cur.setdefault(n, list(shared))
        if isinstance(step, MultiCropPipeline):
            for n, sub in step.pipelines.items():
                cur[n].extend((t, None, g) for t, g in _flatten(sub, ()) if _has_params(t))
        elif _has_params(step):                        # multi-crop decoder: per-view boxes
            for n in names:
                cur[n].append((step, n, ()))
    return cur


def _full(v, applied, idx, n, neutral):
    """Per-sample tensor of length n; values listed by idx are scattered, others get `neutral`."""
    if idx is not None and v.ndim >= 1 and v.shape[0] == idx.numel():
        out = torch.full((n, *v.shape[1:]), neutral, dtype=v.dtype if v.is_floating_point() else torch.float32,
                         device=v.device)
        out[idx] = v.to(out.dtype)
        return out
    if applied is not None and v.ndim >= 1 and v.shape[0] == n and v.is_floating_point() and neutral == neutral:
        a = applied.to(v.device).view(-1, *([1] * (v.ndim - 1)))
        return torch.where(a, v, torch.as_tensor(neutral, dtype=v.dtype, device=v.device))
    return v


def _normalize(t, raw: dict, view: str | None, n: int) -> dict:
    """Uniform per-sample params of one transform (see module docstring)."""
    if view is not None:                               # multi-crop decoder: this view's entries
        sel = {}
        for k, v in raw.items():
            if isinstance(v, dict) and view in v:
                sel.update(v[view]) if k == "crops" and isinstance(v[view], dict) else sel.__setitem__(k, v[view])
            else:
                sel[k] = v
        raw = sel
    if "crop_params" in raw:                           # crop decoders: (x, y, w, h) boxes in source pixels
        out = {}
        for k, v in raw.items():
            k = "box" if k == "crop_params" else k
            out[k] = torch.from_numpy(np.ascontiguousarray(v)) if isinstance(v, np.ndarray) else v
        out["box"] = out["box"].to(torch.int32)
        return out

    dev = next((v.device for v in raw.values() if isinstance(v, torch.Tensor)), torch.device("cpu"))
    idx = raw.get("idx")
    do = raw.get("do")
    applied = None
    if isinstance(idx, torch.Tensor):
        applied = torch.zeros(n, dtype=torch.bool, device=idx.device)
        applied[idx] = True
    elif isinstance(do, torch.Tensor) and do.ndim == 1 and do.shape[0] == n:
        applied = do > 0
    elif isinstance(do, (bool, np.bool_)):
        applied = torch.full((n,), bool(do), dtype=torch.bool, device=dev)

    out = {} if applied is None else {"applied": applied}
    idx = idx if isinstance(idx, torch.Tensor) else None
    if "selected_kernels" in raw and hasattr(t, "sigmas"):
        k = raw["selected_kernels"]
        out["sigma"] = _full(t.sigmas.to(k.device)[k], applied, idx, n, NEUTRAL["sigma"])
        out["kernel"] = _full(k, applied, idx, n, -1.0).long()
    for key, v in raw.items():
        if key in ("do", "idx", "selected_kernels") or key in out:
            continue
        if isinstance(v, torch.Tensor):
            out[key] = _full(v, applied, idx, n, NEUTRAL.get(key, float("nan")))
        elif v is not None:
            out[key] = v
    return out


def _move(x, device, fold):
    if isinstance(x, dict):
        return {k: _move(v, device, fold) for k, v in x.items()}
    if isinstance(x, list):
        return [_move(v, device, fold) for v in x]
    if isinstance(x, torch.Tensor):
        if device is not None:
            x = x.to(device, non_blocking=True)
        if fold is not None and x.ndim >= 1 and x.shape[0] == fold[0] * fold[1]:
            x = x.reshape(fold[0], fold[1], *x.shape[1:])
    return x


class CollectParams:
    """``after_batch_transform`` that adds every pipeline step's random parameters to the batch.

    Args:
        pipelines: The same dict passed as ``SlipstreamLoader(pipelines=...)`` (same objects).
        key: Batch key for the params. Default ``"_params"``.
        fields: Only these fields (default: every field with a random step).
        device: Move every param tensor here (e.g. ``"cuda"``). Default: leave them where they are
            (GPU transforms' params on the GPU, decoder crop boxes on the CPU).

    See the module docstring for the layout. Cost: a handful of small tensor ops per random step
    (scatter for idx-style params), nothing for loaders without it.
    """

    def __init__(self, pipelines: dict, key: str = "_params", fields=None, device=None) -> None:
        self.key = key
        self.device = device
        self._plans: dict = {}
        for field, pipe in pipelines.items():
            if fields is not None and field not in fields:
                continue
            multi = isinstance(pipe, (list, tuple)) and len(pipe) > 0 and all(isinstance(p, (list, tuple)) for p in pipe)
            plans = [_plan(p) for p in pipe] if multi else _plan(pipe)
            if any(any(v) for p in (plans if multi else [plans]) for v in p.values()):
                self._plans[field] = (multi, plans)

    @staticmethod
    def _collect(plan: dict, n: int, dev) -> dict:
        out = {}
        for view, entries in plan.items():
            d: dict = {}
            for t, pview, guards in entries:
                if any(not g.do for g in guards):      # inside a RandomApply that skipped this batch
                    d[_unique(d, t)] = {"applied": torch.zeros(n, dtype=torch.bool, device=dev)}
                    continue
                lp = t.last_params
                raw = lp() if callable(lp) else lp
                if not raw:
                    continue
                d[_unique(d, t)] = _normalize(t, raw, pview, n)
            if view is None:
                out.update(d)
            else:
                out[view] = d
        return out

    def __call__(self, batch: dict) -> dict:
        ind = batch.get("_indices")
        fold = tuple(ind.shape) if isinstance(ind, torch.Tensor) and ind.ndim == 2 else None
        n = int(ind.numel()) if isinstance(ind, torch.Tensor) else None
        params = {}
        for field, (multi, plans) in self._plans.items():
            # a field's dict output (named views) is merged into the batch, so look for it under the view names
            names = [v for p in (plans if multi else [plans]) for v in p if v is not None]
            ref = batch.get(field, next((batch[v] for v in names if v in batch), None))
            nf = n if n is not None else _first_dim(ref)
            dev = _device(ref)
            params[field] = [self._collect(p, nf, dev) for p in plans] if multi else self._collect(plans, nf, dev)
        batch[self.key] = _move(params, self.device, fold)
        return batch

    def __repr__(self) -> str:
        return f"CollectParams(fields={list(self._plans)}, key={self.key!r})"


def _first_dim(x) -> int:
    if isinstance(x, dict):
        x = next(iter(x.values()))
    if isinstance(x, (list, tuple)):
        return len(x) if not x or not isinstance(x[0], (torch.Tensor, np.ndarray)) or x[0].ndim < 4 else _first_dim(x[0])
    return int(x.shape[0])


def _unique(d: dict, t) -> str:
    name, i = type(t).__name__, 1
    while name in d:
        name, i = f"{type(t).__name__}_{i}", i + 1
    return name


def _device(x):
    if isinstance(x, dict):
        x = next(iter(x.values()), None)
    if isinstance(x, (list, tuple)):
        x = x[0] if x else None
    return x.device if isinstance(x, torch.Tensor) else torch.device("cpu")
