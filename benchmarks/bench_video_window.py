"""Throughput of DecodeVideoWindow over a `bytes` video store (h265 MP4 per record).

Prints one line per configuration (no progress bars). Example (machina, CPU pool)::

    uv run python benchmarks/bench_video_window.py --cache <store dir> --T 120 --rate 15 \
        --batch-size 8 --workers 32 --max-batches 50 [--indices anchors.npy] [--warm]

CUDA::

    uv run python benchmarks/bench_video_window.py --cache <store> --T 120 --rate 15 --device cuda:0 --workers 2

Add --crop 224 to append RandomResizedCropBatch(224)+RandomHorizontalFlip on the decoded frames,
--resize 256 for a decoder-side short-side resize.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from slipstream.dataset import SlipstreamDataset
from slipstream.decoders import DecodeVideoWindow
from slipstream.loader import SlipstreamLoader
from slipstream.transforms import RandomHorizontalFlip, RandomResizedCropBatch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--field", default="video")
    ap.add_argument("--T", type=int, default=120)
    ap.add_argument("--rate", type=float, default=15.0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--ffmpeg-threads", type=int, default=1)
    ap.add_argument("--device", default="cpu", help="cpu, cuda:0, or a comma list cuda:0,cuda:1 (one stage, threads pinned per device)")
    ap.add_argument("--output-device", default=None)
    ap.add_argument("--batches-ahead", type=int, default=None, help="default: ceil(workers / batch_size), min 3")
    ap.add_argument("--resize", type=int, default=None)
    ap.add_argument("--crop", type=int, default=None)
    ap.add_argument("--indices", default=None, help=".npy of record indices (clips with duration >= T/rate)")
    ap.add_argument("--max-batches", type=int, default=50)
    ap.add_argument("--warmup-batches", type=int, default=3)
    ap.add_argument("--warm", action="store_true", help="warmup_cache(indices) first")
    a = ap.parse_args()

    ds = SlipstreamDataset(local_dir=a.cache)   # a prebuilt .slipstream cache dir
    indices = np.load(a.indices) if a.indices else None
    inner = [RandomResizedCropBatch(a.crop, seed=1), RandomHorizontalFlip(p=0.5, seed=2)] if a.crop else None
    devices = a.device.split(",") if "," in a.device else a.device
    stage = DecodeVideoWindow(T=a.T, rate_hz=a.rate, seed=0, device=devices, output_device=a.output_device,
                              num_workers=a.workers, num_ffmpeg_threads=a.ffmpeg_threads, resize=a.resize, transforms=inner)
    ahead = a.batches_ahead or max(3, -(-stage.num_workers // a.batch_size))
    loader = SlipstreamLoader(ds, batch_size=a.batch_size, shuffle=True, seed=0, drop_last=True, indices=indices,
                              batches_ahead=ahead, image_field=a.field, pipelines={a.field: [stage]}, verbose=False)
    if a.warm:
        st = loader.warmup_cache(verbose=False)
        print(f"warmup: {st['total_bytes'] / 1e9:.1f} GB in {st['elapsed_sec']:.1f} s")

    n_win = 0
    t0 = None
    shape = None
    for i, batch in enumerate(loader):
        if i == a.warmup_batches:
            t0 = time.perf_counter(); n_win = 0
        v = batch[a.field]
        if v.device.type == "cuda":
            import torch
            torch.cuda.synchronize(v.device)
        n_win += v.shape[0]; shape = tuple(v.shape)
        if i + 1 >= a.max_batches + a.warmup_batches:
            break
    dt = time.perf_counter() - (t0 or time.perf_counter())
    print(f"T={a.T} rate={a.rate:g} B={a.batch_size} device={a.device} workers={stage.num_workers} ahead={ahead} "
          f"resize={a.resize} crop={a.crop} shape={shape}: {n_win / dt:,.1f} windows/s, "
          f"{n_win * a.T / dt:,.0f} frames/s over {n_win} windows ({dt:.1f} s)")
    loader.shutdown()


if __name__ == "__main__":
    main()
