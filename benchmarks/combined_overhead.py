"""CombinedDataset overhead: the same records through a combined loader vs a single-dataset loader.

Both read imagenet1k val with step C of the ablation (RRC 224, uint8 CPU, shuffled, threaded prefetch),
batch 512, 3 epochs each, warm = best of epochs 2-3. The combined loader mixes two sources that are the same
cache (weights 0.5/0.5, epoch_length 50,000), so it decodes the same kind of images and only the routing
differs: per-source reads into the prefetch bank, label/source fields. The release gate FAILs if combined
is more than --tolerance slower than single (George 2026-11-27: a separate combined line in the gate).

    uv run python -m benchmarks.combined_overhead [--fmt jpeg] [--epochs 3]
"""

from __future__ import annotations

import argparse
import os
import time
import warnings
from pathlib import Path

from benchmarks.loader_ablation import _n_images, _pipeline


def _rate(loader, epochs: int) -> list[float]:
    rates = []
    for _ in range(epochs):
        t = time.perf_counter()
        n = sum(_n_images(b) for b in loader)
        rates.append(n / (time.perf_counter() - t))
    loader.shutdown()
    return rates


def run(cache_root: Path, fmt: str, *, epochs: int = 3, batch_size: int = 512, log=print) -> dict:
    """``{"single": warm img/s, "combined": warm img/s, "ratio": combined / single, ...}``"""
    from slipstream import CombinedDataset, SlipstreamDataset, SlipstreamLoader

    path = str(cache_root / f"imagenet1k-s256_l512-{fmt}-val")
    kw = dict(batch_size=batch_size, shuffle=True, seed=0, image_format=fmt, verbose=False,
              exclude_fields=["path"])
    single = _rate(SlipstreamLoader(SlipstreamDataset(input_dir=path), pipelines=_pipeline(0, "cpu"), **kw), epochs)
    comb = CombinedDataset([SlipstreamDataset(input_dir=path), SlipstreamDataset(input_dir=path)],
                           weights=[0.5, 0.5], epoch_length=50_000, labels="shared")
    combined = _rate(SlipstreamLoader(comb, pipelines=_pipeline(0, "cpu"), **kw), epochs)
    s, c = max(single[1:]), max(combined[1:])
    log(f"  {fmt:6s} combined vs single (step C, warm): {c:8,.0f} vs {s:8,.0f} img/s ({c / s:.3f}x)")
    return {"single": s, "combined": c, "ratio": c / s, "single_epochs": single, "combined_epochs": combined}


def main() -> None:
    warnings.simplefilter("ignore")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-root", type=Path,
                    default=Path(os.environ.get("SLIPSTREAM_CACHE_DIR", "~/.slipstream")).expanduser())
    ap.add_argument("--fmt", default="jpeg", choices=["jpeg", "yuv420"])
    ap.add_argument("--epochs", type=int, default=3)
    a = ap.parse_args()
    run(a.cache_root, a.fmt, epochs=a.epochs)


if __name__ == "__main__":
    main()
