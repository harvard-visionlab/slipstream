"""Training-step time with the loader's host-to-device copy under a busy GPU (pinned / async HtoD work).

The release gate runs on an idle GPU, so it cannot show whether a CPU->GPU batch copy stalls training. This
bench mimics a trainer: per step it takes a batch and queues `--compute-ms` of GPU work (bf16 matmuls) that
reads the batch, and it waits for the previous step's work (one step in flight, like a lagged loss log).
Prints one line: median ms/step with the spread, plus the GPU-busy reference (compute alone).

Video (DecodeVideoWindow, CPU decode, output on cuda), needs torchcodec::

    python -m benchmarks.htod_step video --cache <store> --B 8 --T 16 --resize 168 224
Image (decoder + ToTorchImage on cuda)::

    python -m benchmarks.htod_step image --fmt jpeg --batch-size 256
"""

from __future__ import annotations

import argparse
import os
import statistics
import time
import warnings
from pathlib import Path


def _matmul_work(ms: float):
    """A function that queues about `ms` of GPU work reading `x` (calibrated once)."""
    import torch
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    for _ in range(3):
        a @ a
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(20):
        a @ a
    torch.cuda.synchronize()
    per = (time.perf_counter() - t) / 20 * 1e3
    k = max(1, round(ms / per))

    def work(x):
        s = x.flatten()[:1].float()                # reads the batch: the step depends on the copy
        b = a
        for _ in range(k):
            b = a @ a
        return b[0, 0] + s.sum()
    return work, k * per


def _video_loader(a):
    from slipstream import SlipstreamDataset, SlipstreamLoader
    from slipstream.decoders import DecodeVideoWindow
    ds = SlipstreamDataset(local_dir=a.cache)
    resize = tuple(a.resize) if a.resize and len(a.resize) == 2 else (a.resize[0] if a.resize else None)
    stage = DecodeVideoWindow(T=a.T, rate_hz=a.rate, seed=0, device="cpu", output_device="cuda",
                              num_workers=a.workers, resize=resize)
    return SlipstreamLoader(ds, batch_size=a.B, shuffle=True, seed=0, drop_last=True, image_field=a.field,
                            pipelines={a.field: [stage]}, verbose=False), a.field


def _image_loader(a):
    from slipstream import SlipstreamDataset, SlipstreamLoader
    from slipstream.decoders import DecodeRandomResizedCrop
    from slipstream.transforms import ToTorchImage
    root = Path(os.environ.get("SLIPSTREAM_CACHE_DIR", "~/.slipstream")).expanduser()
    ds = SlipstreamDataset(input_dir=str(root / f"imagenet1k-s256_l512-{a.fmt}-val"))
    pipe = [DecodeRandomResizedCrop(size=a.size, seed=1), ToTorchImage("cuda")]
    return SlipstreamLoader(ds, batch_size=a.batch_size, shuffle=True, seed=0, drop_last=True, image_format=a.fmt,
                            exclude_fields=["path"], pipelines={"image": pipe}, verbose=False), "image"


def main() -> None:
    warnings.simplefilter("ignore")
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["video", "image"])
    ap.add_argument("--compute-ms", type=float, default=100.0)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--cache")
    ap.add_argument("--field", default="video")
    ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--T", type=int, default=16)
    ap.add_argument("--rate", type=float, default=15.0)
    ap.add_argument("--resize", type=int, nargs="+", default=[168, 224])
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--fmt", default="jpeg")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--size", type=int, default=224)
    a = ap.parse_args()
    torch.set_num_threads(1) if a.kind == "video" else None

    work, est = _matmul_work(a.compute_ms)
    loader, key = (_video_loader if a.kind == "video" else _image_loader)(a)
    times, prev = [], None
    it = iter(loader)
    try:
        for i in range(a.warmup + a.steps):
            t = time.perf_counter()
            b = next(it)
            done = torch.cuda.Event()
            out = work(b[key])
            done.record()
            if prev is not None:
                prev.synchronize()                    # one step in flight
            prev = done
            if i >= a.warmup:
                times.append((time.perf_counter() - t) * 1e3)
    except StopIteration:
        pass
    finally:
        torch.cuda.synchronize()
        loader.shutdown()
    del out
    q = statistics.quantiles(times, n=10)
    shape = tuple(b[key].shape)
    print(f"{a.kind} {shape} compute {est:.0f} ms: step median {statistics.median(times):.1f} ms "
          f"(p10 {q[0]:.1f}, p90 {q[-1]:.1f}, n={len(times)})", flush=True)


if __name__ == "__main__":
    main()
