"""Throughput of SlipstreamLoader(window=(T, stride)) on a frame store.

Prints one line per configuration (no progress bars). Run from a warm cache::

    uv run python benchmarks/bench_window_loader.py --cache /path/to/frames5hz-456x256 \
        --T 40 --batch-size 16 --epochs 1 --warmup-batches 5 [--decode]

Without --decode the batch is the raw JPEG bytes dict (I/O only). With --decode a
DecodeCenterCrop(256) to uint8 CHW tensors is applied, i.e. [B, T, 3, 256, 256]
(use --size to change; --rrc for a seeded RandomResizedCrop instead).
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from slipstream.cache import OptimizedCache
from slipstream.dataset import CachedDataset
from slipstream.decoders import DecodeCenterCrop, DecodeRandomResizedCrop
from slipstream.loader import SlipstreamLoader


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--T", type=int, default=40)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-batches", type=int, default=0, help="stop after this many batches (0 = whole epoch)")
    ap.add_argument("--warmup-batches", type=int, default=5)
    ap.add_argument("--decode", action="store_true")
    ap.add_argument("--rrc", action="store_true")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--anchors", default=None, help=".npy of anchor record indices (default: all valid anchors)")
    ap.add_argument("--warm", action="store_true", help="warmup_cache(indices=anchors) before timing")
    a = ap.parse_args()

    ds = CachedDataset(a.cache)
    anchors = np.load(a.anchors) if a.anchors else None
    pipelines = None
    if a.decode:
        dec = (DecodeRandomResizedCrop(size=a.size, seed=0, to_tensor=True, permute=True) if a.rrc
               else DecodeCenterCrop(size=a.size, to_tensor=True, permute=True))
        pipelines = {"image": [dec]}
    loader = SlipstreamLoader(ds, batch_size=a.batch_size, shuffle=True, seed=0, drop_last=True,
                              indices=anchors, window=(a.T, a.stride), pipelines=pipelines, verbose=False)
    if a.warm:
        st = loader.warmup_cache(verbose=False)
        print(f"warmup: {st['total_bytes'] / 1e9:.1f} GB in {st['elapsed_sec']:.1f} s")

    n_win = n_frames = 0
    t0 = None
    for _ in range(a.epochs):
        for i, batch in enumerate(loader):
            if i == a.warmup_batches:
                t0 = time.perf_counter(); n_win = n_frames = 0
            img = batch["image"]
            B = img.shape[0] if hasattr(img, "shape") else img["data"].shape[0]
            n_win += B; n_frames += B * a.T
            if a.max_batches and i + 1 >= a.max_batches:
                break
    dt = time.perf_counter() - (t0 or time.perf_counter())
    shape = tuple(img.shape) if hasattr(img, "shape") else ("raw", *img["data"].shape[:2])
    print(f"T={a.T} stride={a.stride} B={a.batch_size} decode={a.decode} rrc={a.rrc} shape={shape}: "
          f"{n_win / dt:,.0f} windows/s, {n_frames / dt:,.0f} frames/s over {n_win:,} windows ({dt:.1f} s)")
    loader.shutdown()


if __name__ == "__main__":
    main()
