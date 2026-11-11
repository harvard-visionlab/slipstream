"""DecodeVideoWindow augmentation cost: aug_mode="frame" (a draw per frame) vs "clip" (a draw per window).

Times ``DecodeVideoWindow.apply_transforms`` -- the only code the two modes differ in -- on real frames
(imagenet1k val, centre-cropped to 256 and grouped as B windows of T frames), with the SimCLR-TT+ chain
(RandomResizedCropBatch 224 scale (0.2, 1), flip, ToFloatDiv, colour jitter p=0.8, grayscale p=0.2).
Decoding is excluded: it is identical in both modes and would hide the difference. Needs no torchcodec.

    uv run python -m benchmarks.video_aug_modes [--device cuda] [--B 8 --T 64]
"""

from __future__ import annotations

import argparse
import os
import statistics
import time
from pathlib import Path

CHAIN = "RRC224(0.2,1) + flip + ToFloatDiv + HSV jitter p=0.8 + grayscale p=0.2"


def _frames(cache_root: Path, n: int):
    from slipstream import SlipstreamDataset, SlipstreamLoader
    from slipstream.decoders import DecodeCenterCrop

    ds = SlipstreamDataset(input_dir=str(cache_root / "imagenet1k-s256_l512-jpeg-val"))
    loader = SlipstreamLoader(ds, batch_size=n, shuffle=False, verbose=False, exclude_fields=["path"],
                              image_format="jpeg", pipelines={"image": [DecodeCenterCrop(256)]})
    try:
        img = next(iter(loader))["image"]
    finally:
        loader.shutdown()
    import torch
    img = torch.as_tensor(img)
    return img if img.shape[1] == 3 else img.permute(0, 3, 1, 2).contiguous()     # [n, 3, 256, 256] uint8


def _stage(mode: str, T: int, device: str = "cpu"):
    from slipstream.decoders import DecodeVideoWindow
    from slipstream.transforms import (
        RandomColorJitterHSV,
        RandomGrayscale,
        RandomHorizontalFlip,
        RandomResizedCropBatch,
        ToFloatDiv,
    )
    return DecodeVideoWindow(T=T, rate_hz=10.0, seed=0, aug_mode=mode, transforms=[
        RandomResizedCropBatch(224, scale=(0.2, 1.0), seed=1, device=device),
        RandomHorizontalFlip(p=0.5, seed=2, device=device), ToFloatDiv(255),
        RandomColorJitterHSV(p=0.8, hue=0.2, saturation=0.8, value=0.8, contrast=0.8, seed=3, device=device),
        RandomGrayscale(p=0.2, seed=4, device=device)])


def run(cache_root: Path, device: str, *, B: int = 8, T: int = 64, iters: int = 20, warmup: int = 3) -> dict:
    """``{"clip": frames/s, "frame": frames/s, "ratio": frame/clip}``; modes interleaved, median of `iters`."""
    import torch

    frames = _frames(cache_root, B * T).to(device).reshape(B, T, 3, 256, 256)
    stages = {m: _stage(m, T, device) for m in ("clip", "frame")}
    sync = torch.cuda.synchronize if device.startswith("cuda") else (lambda: None)
    times: dict[str, list[float]] = {m: [] for m in stages}
    for i in range(warmup + iters):
        for m, st in stages.items():
            sync()
            t = time.perf_counter()
            st.apply_transforms(frames)
            sync()
            if i >= warmup:
                times[m].append(time.perf_counter() - t)
    fps = {m: B * T / statistics.median(v) for m, v in times.items()}
    return {"clip": fps["clip"], "frame": fps["frame"], "ratio": fps["frame"] / fps["clip"],
            "B": B, "T": T, "chain": CHAIN}


def main() -> None:
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-root", type=Path,
                    default=Path(os.environ.get("SLIPSTREAM_CACHE_DIR", "~/.slipstream")).expanduser())
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--iters", type=int, default=20)
    a = ap.parse_args()
    r = run(a.cache_root, a.device, B=a.B, T=a.T, iters=a.iters)
    print(f"{a.device} B={a.B} T={a.T}: clip {r['clip']:,.0f}  frame {r['frame']:,.0f} frames/s  "
          f"(frame/clip {r['ratio']:.3f})")


if __name__ == "__main__":
    main()
