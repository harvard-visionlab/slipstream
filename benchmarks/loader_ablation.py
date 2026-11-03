"""SlipstreamLoader throughput ablation: the bare decode benchmark -> a typical training pipeline.

Each step adds one thing, so a regression shows up at the step that introduced it:

    A  DecodeRandomResizedCrop(224) only, uint8 HWC on CPU, shuffle=False, use_threading=False
    B  + threaded prefetch (the loader default)
    C  + shuffle=True
    D  + ToTorchImage(device) fp32
    E  + Normalize fp32
    F  ToTorchImage / Normalize in bf16
    G  + RandomHorizontalFlip          (~ model-rearing's catsup_minimal)
    G@160, G@192  G at 160 / 192 px    (progressive resolution; G itself is 224)
    M  DecodeMultiRandomResizedCrop: 2 x 224 global + 4 x 98 local, uint8 CPU (SSL multicrop)
    Mf M with crop_mode="ffcv"                                          (``run_epochs`` only)

``run`` (best of N partial runs, warm) is for quick checks. ``run_epochs`` (the release gate) times full
passes over the val set: epoch 1 starts from an evicted page cache (cold), later epochs are warm.

Quiet by design (one line per step, no progress bars) so it can run inside the release gate.

    uv run python -m benchmarks.loader_ablation --formats jpeg yuv420 [--json out.json]

Caches: ``<cache-root>/imagenet1k-s256_l512-{fmt}-val`` (default root: $SLIPSTREAM_CACHE_DIR).
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from pathlib import Path

STEPS = [  # key, label, shuffle, threaded, stage
    ("A", "RRC only, uint8 CPU, sequential, simple", False, False, 0),
    ("B", "+ threaded prefetch", False, True, 0),
    ("C", "+ shuffle", True, True, 0),
    ("D", "+ ToTorchImage fp32", True, True, 1),
    ("E", "+ Normalize fp32", True, True, 2),
    ("F", "ToTorchImage/Normalize bf16", True, True, 3),
    ("G", "+ RandomHorizontalFlip", True, True, 4),
]


EXTRA_SIZES = (160, 192)      # full pipeline (G) at progressive-resolution sizes

MULTICROP_STEPS = [  # key, label, crop_mode
    ("M", "multicrop 2x224 + 4x98, uint8 CPU", "standard"),
    ("Mf", "multicrop, crop_mode='ffcv'", "ffcv"),
]


def _pipeline(stage: int, device: str, size: int = 224):
    import torch

    from slipstream.decoders import DecodeRandomResizedCrop
    from slipstream.transforms import IMAGENET_MEAN, IMAGENET_STD, Normalize, ToTorchImage
    from slipstream.transforms.geometric import RandomHorizontalFlip

    dtype = torch.bfloat16 if stage >= 3 else torch.float32
    p = [DecodeRandomResizedCrop(size, seed=1)]
    if stage >= 1:
        p.append(ToTorchImage(device, dtype=dtype))
    if stage >= 2:
        p.append(Normalize(IMAGENET_MEAN, IMAGENET_STD, dtype=dtype, device=device))
    if stage >= 4:
        p.append(RandomHorizontalFlip(p=0.5, seed=2, device=device))
    return {"image": p}


def _multicrop_pipeline(crop_mode: str):
    from slipstream.decoders.multicrop import DecodeMultiRandomResizedCrop

    crops = {f"g{i}": dict(size=224, scale=(0.30, 1.0), seed=10 + i) for i in range(2)}
    crops |= {f"l{i}": dict(size=98, scale=(0.05, 0.30), seed=20 + i) for i in range(4)}
    return {"image": [DecodeMultiRandomResizedCrop(crops, crop_mode=crop_mode)]}


def _n_images(batch) -> int:
    img = batch["image"] if "image" in batch else batch["g0"]
    return img[0].shape[0] if isinstance(img, (list, tuple)) else img.shape[0]


def evict(cache_dir: Path) -> int:
    """Drop the cache's files from the page cache (posix_fadvise DONTNEED, no root). Returns bytes resident
    afterwards (via ``fincore`` when available, else -1). Pages mapped by a live process are not dropped, so
    call this before the dataset is opened."""
    import gc
    import shutil
    import subprocess

    gc.collect()
    files = [f for f in Path(cache_dir).rglob("*") if f.is_file()]
    if not hasattr(os, "posix_fadvise"):      # macOS: no eviction; "cold" is not cold there
        return -1
    for f in files:
        fd = os.open(f, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
    if not shutil.which("fincore"):
        return -1
    out = subprocess.run(["fincore", "-b", "-n", "-o", "RES", *map(str, files)], capture_output=True, text=True).stdout
    return sum(int(x) for x in out.split())


def run_epochs(cache_root: Path, fmt: str, *, epochs: int = 3, batch_size: int = 512, device: str | None = None,
               steps: str = "ABCDEFG", sizes=EXTRA_SIZES, multicrop: bool = True, log=print) -> dict[str, dict]:
    """Full passes per step: ``{"A": {"epochs": [cold, warm, warm], "cold": .., "warm": .., "resident": ..}}``.

    Each step evicts the cache from the page cache, opens the dataset fresh and iterates ``epochs`` full
    epochs with one loader (loader start-up included). ``warm`` is the best of epochs 2+.
    """
    import gc

    import torch

    from slipstream import SlipstreamDataset, SlipstreamLoader

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cache_dir = cache_root / f"imagenet1k-s256_l512-{fmt}-val"
    todo = [(k, lbl, sh, th, (lambda st=st: _pipeline(st, device))) for k, lbl, sh, th, st in STEPS if k in steps]
    if "G" in steps:
        g = next(x for x in STEPS if x[0] == "G")
        todo += [(f"G@{sz}", f"{g[1]} @ {sz}px", g[2], g[3], (lambda sz=sz: _pipeline(g[4], device, sz)))
                 for sz in sizes]
    if multicrop:
        todo += [(k, lbl, True, True, (lambda m=m: _multicrop_pipeline(m))) for k, lbl, m in MULTICROP_STEPS]
    out: dict[str, dict] = {}
    for key, label, shuffle, threaded, make in todo:
        resident = evict(cache_dir)
        ds = SlipstreamDataset(input_dir=str(cache_dir))
        loader = SlipstreamLoader(ds, batch_size=batch_size, shuffle=shuffle, seed=0, image_format=fmt,
                                  verbose=False, use_threading=threaded, exclude_fields=["path"], pipelines=make())
        rates = []
        for _ in range(epochs):
            t = time.perf_counter()
            n = sum(_n_images(b) for b in loader)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            rates.append(n / (time.perf_counter() - t))
        loader.shutdown()
        del loader, ds
        gc.collect()
        out[key] = {"epochs": rates, "cold": rates[0], "warm": max(rates[1:]), "resident": resident}
        log(f"  {fmt:6s} {key:5s} {label:40s} cold {rates[0]:8,.0f}  warm {max(rates[1:]):8,.0f} img/s"
            f"  ({' / '.join(f'{r:,.0f}' for r in rates)})")
    return out


def run(cache_root: Path, fmt: str, *, batches: int = 60, batch_size: int = 512, repeats: int = 3,
        warmup: int = 5, device: str | None = None, steps: str = "ABCDEFG", sizes=EXTRA_SIZES,
        log=print) -> dict[str, float]:
    """Best-of-``repeats`` img/s for each step (``{"A": 38302.1, ...}``)."""
    import torch

    from slipstream import SlipstreamDataset, SlipstreamLoader

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ds = SlipstreamDataset(input_dir=str(cache_root / f"imagenet1k-s256_l512-{fmt}-val"))
    out: dict[str, float] = {}
    todo = [(k, lbl, sh, th, st, 224) for k, lbl, sh, th, st in STEPS if k in steps]
    if "G" in steps:
        g = next(x for x in STEPS if x[0] == "G")
        todo += [(f"G@{sz}", f"{g[1]} @ {sz}px", g[2], g[3], g[4], sz) for sz in sizes]
    for key, label, shuffle, threaded, stage, size in todo:
        best = 0.0
        for _ in range(repeats):
            loader = SlipstreamLoader(ds, batch_size=batch_size, shuffle=shuffle, seed=0, image_format=fmt,
                                      verbose=False, use_threading=threaded, exclude_fields=["path"],
                                      pipelines=_pipeline(stage, device, size))
            it = iter(loader)
            for _ in range(warmup):
                next(it)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            t = time.perf_counter()
            n = sum(next(it)["image"].shape[0] for _ in range(batches))
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            best = max(best, n / (time.perf_counter() - t))
            del it, loader
        out[key] = best
        log(f"  {fmt:6s} {key}  {label:40s} {best:10,.0f} img/s")
    return out


def main() -> None:
    warnings.simplefilter("ignore")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-root", type=Path, default=Path(os.environ.get("SLIPSTREAM_CACHE_DIR", "~/.slipstream")).expanduser())
    ap.add_argument("--formats", nargs="+", default=["jpeg", "yuv420"])
    ap.add_argument("--batches", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--steps", default="ABCDEFG")
    ap.add_argument("--device", default=None)
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args()
    res = {f: run(a.cache_root, f, batches=a.batches, batch_size=a.batch_size, repeats=a.repeats,
                  device=a.device, steps=a.steps) for f in a.formats}
    if a.json:
        a.json.write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
